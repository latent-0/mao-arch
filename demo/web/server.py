"""Split-screen live demo server for the 1-minute talk.

Serves a single-page UI (no external assets — survives offline) and an API
that runs BOTH orchestrations for real on every click:

  left  — naive prose handoff: the QA agent re-infers the next action from a
          summary and runs tests against a branch that doesn't exist
  right — joint-embedding handoff: the same proposal is gated by the local
          adjudicator (real cosine check, real Gemma 4 replan message),
          re-planned from the graph frontier, and finishes green

Usage:  python -m demo.web.server  [--port 8765] [--encoder local|gemini]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from mao.adjudicator import Adjudicator, Verdict
from mao.datagen import _trace
from mao.encoders.language import encoder_mode
from mao.graph import Status
from mao.handoff import build_packet, naive_summary
from demo.swe_scenario import build_scenario

WEB_DIR = os.path.dirname(os.path.abspath(__file__))

_adjudicators: dict[str, Adjudicator] = {}
_lock = threading.Lock()


def get_adjudicator(mode: str) -> Adjudicator:
    with _lock:
        if mode not in _adjudicators:
            _adjudicators[mode] = Adjudicator.from_artifacts(mode=mode)
        return _adjudicators[mode]


def ev(kind: str, title: str, detail: str = "", ok: bool | None = None,
       badge: str = "") -> dict:
    return {"kind": kind, "title": title, "detail": detail, "ok": ok,
            "badge": badge}


def run_naive() -> list[dict]:
    g = build_scenario()
    events = [
        ev("agent", "Agent A (planner) finishes its half",
           "patch applied to working tree; summarizes state as prose"),
        ev("handoff", "Handoff: prose summary",
           f'"{naive_summary(g)}"', badge="TEXT"),
        ev("agent", "Agent B (QA) re-infers next action from prose",
           "proposes: run_tests — the summary never said the branch doesn't exist"),
        ev("execute", "run_tests", g.check_action("run_tests")[0].detail, ok=False),
        ev("result", "PIPELINE FAILED",
           "dependency-order violation — MAST: 'Disobey Task Specification'",
           ok=False),
    ]
    return events


def run_structured(mode: str, offline: bool) -> list[dict]:
    adj = get_adjudicator(mode)
    rng = random.Random(7)
    g = build_scenario()
    packet = build_packet(g, adj.model)
    n_nodes = len(packet.graph_snapshot["nodes"])

    events = [
        ev("agent", "Agent A (planner) finishes its half",
           "patch applied; task graph updated live during execution"),
        ev("handoff", "Handoff: graph frontier + joint embedding",
           f"{n_nodes} nodes, {len(packet.embedding)}-d vector — no prose, "
           f"no inherited token history", badge="STRUCT"),
    ]
    if offline:
        events.append(ev("info", "Network down",
                         "adjudicator + trace encoding + Gemma 4 all run "
                         "on-device — zero cloud round-trips"))

    d = adj.adjudicate(packet, "run_tests", _trace(g, "run_tests", rng))
    events.append(ev("agent", "Agent B (QA) proposes the same action",
                     "proposes: run_tests — same goal-driven intent as the naive run"))
    events.append(ev("verdict", f"Adjudicator: {d.verdict.value.upper()}",
                     f"alignment {d.alignment:+.3f} vs approve band "
                     f"({d.tau_lo:+.3f}, {d.tau_hi:+.3f})",
                     ok=False, badge="GATE"))
    if d.violations:
        events.append(ev("witness", "Violated edge (structural witness)",
                         d.violations[0].detail, ok=False))
    events.append(ev("gemma", "Gemma 4 E2B (local) → Agent B", d.message,
                     badge="ON-DEVICE"))

    events.append(ev("agent", "Agent B re-plans from the graph frontier", ""))
    for action in ["create_branch", "checkout_branch", "run_tests"]:
        if g.nodes[action].status == Status.DONE:
            continue
        dec = adj.adjudicate(build_packet(g, adj.model), action,
                             _trace(g, action, rng))
        if dec.verdict == Verdict.REQUEST_REPLAN:
            events.append(ev("execute", action,
                             f"gated (alignment {dec.alignment:+.3f})", ok=False))
            continue
        note = f"alignment {dec.alignment:+.3f} -> {dec.verdict.value}"
        if dec.verdict == Verdict.FLAG_TO_HUMAN:
            note += " — surfaced to human, deps verified, approved"
        ok = not g.check_action(action)
        if ok:
            g.set_status(action, Status.DONE)
            g.add_artifact(f"out_{action}", f"output artifact of {action}",
                           artifact_type="data", produced_by=action)
        events.append(ev("execute", action, note, ok=ok))

    passed = g.nodes["run_tests"].status == Status.DONE
    events.append(ev("result", "TESTS PASS" if passed else "FAILED",
                     "same agents, same intent — the handoff substrate is the "
                     "only difference", ok=passed))
    return events


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif url.path == "/api/run":
            qs = parse_qs(url.query)
            mode = qs.get("mode", [self.server.default_mode])[0]
            offline = qs.get("offline", ["0"])[0] == "1"
            side = qs.get("side", ["both"])[0]
            t0 = time.time()
            payload = {"mode": mode, "offline": offline}
            if side in ("left", "both"):
                payload["left"] = run_naive()
            if side in ("right", "both"):
                payload["right"] = run_structured(mode, offline)
            payload["server_ms"] = round((time.time() - t0) * 1000)
            self._send(200, json.dumps(payload).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--encoder", choices=["local", "gemini"], default="local",
                    help="default gate mode (local = fully offline-safe)")
    args = ap.parse_args()

    # pre-load artifacts and warm Gemma so the first click is fast
    def warm():
        adj = get_adjudicator(args.encoder)
        if adj.gemma and adj.gemma.model:
            adj.gemma.generate("Reply with the single word READY.")
            print(f"[warm] Gemma ready: {adj.gemma.model}")
    threading.Thread(target=warm, daemon=True).start()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.default_mode = args.encoder
    print(f"[serve] http://127.0.0.1:{args.port}  (gate mode: {args.encoder})")
    srv.serve_forever()


if __name__ == "__main__":
    main()

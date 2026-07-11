"""Live split-screen demo server. Nothing is simulated:

  GET /api/stream?side=naive        -> SSE: real Gemini agents + real git repo
  GET /api/stream?side=structured   -> SSE: same, gated by the real adjudicator
                 &offline=1         -> Agent B + gate run fully on-device (Gemma 4)

Each event is one JSON object per SSE `data:` line; the pipelines execute
while the connection is open, so the browser renders agent decisions, real
command output, gate verdicts, and task-graph updates as they happen.

Usage:  python -m demo.web.server  [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from mao.adjudicator import Adjudicator
from demo.web.live import run_naive, run_structured

WEB_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
        elif url.path == "/api/stream":
            self._stream(parse_qs(url.query))
        else:
            self._send(404, b"not found", "text/plain")

    def _stream(self, qs: dict):
        side = qs.get("side", ["naive"])[0]
        offline = qs.get("offline", ["0"])[0] == "1"
        scenario = qs.get("scenario", ["swe"])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: dict):
            try:
                payload = f"data: {json.dumps(event)}\n\n".encode()
                self.wfile.write(payload)
                self.wfile.flush()
            except (ConnectionAbortedError, BrokenPipeError, OSError):
                raise ClientGone()

        try:
            if side == "structured":
                run_structured(emit, offline, scenario)
            else:
                run_naive(emit, scenario)
        except ClientGone:
            pass


class ClientGone(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    def warm():  # load artifacts + keep Gemma hot so the first click is fast
        for mode in ("gemini" if os.environ.get("GEMINI_API_KEY") else "local",
                     "local"):
            try:
                adj = Adjudicator.from_artifacts(mode=mode)
                if adj.gemma and adj.gemma.model:
                    adj.gemma.generate("Reply with the single word READY.")
                    print(f"[warm] gate mode '{mode}' + Gemma ready")
                break
            except Exception as e:
                print(f"[warm] {mode}: {e}")
    threading.Thread(target=warm, daemon=True).start()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[serve] http://127.0.0.1:{args.port}  — live pipelines, no simulation")
    srv.serve_forever()


if __name__ == "__main__":
    main()

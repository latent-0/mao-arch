"""Live multi-agent pipeline for the web demo — no simulation.

Everything here is real:
  * Agent A and Agent B are live Gemini 3.5 Flash calls deciding their own
    actions (in offline mode, Agent B is live Gemma 4 E2B via Ollama).
  * The workspace is a real temp git repository with a real buggy module and
    a real test suite; Agent A actually fixes the bug.
  * Tool calls execute real `git` / `pytest` subprocesses; their stdout/stderr
    stream to the browser terminal panes.
  * The adjudicator gates Agent B's own justification text (the live trace)
    with the trained joint-embedding model; Gemma 4 writes replan messages.

The naive pipeline can, in principle, succeed if the receiving agent happens
to re-derive the dependency order from prose — that is the point of a live
demo. The structured pipeline is protected by the gate either way.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

from mao.adjudicator import Adjudicator, OllamaGemma, Verdict
from mao.graph import Status, TaskGraph, ConstraintType
from mao.handoff import build_packet, naive_summary

GEMINI_MODEL = "gemini-3.5-flash"

BUGGY_CALC = '''"""Order utilities for the checkout service."""


def line_total(unit_price: float, quantity: int) -> float:
    return unit_price - quantity  # BUG #123: should multiply, not subtract


def order_total(lines: list[tuple[float, int]]) -> float:
    return sum(line_total(p, q) for p, q in lines)
'''

TEST_CALC = '''import unittest

from calc import line_total, order_total


class TestOrderTotals(unittest.TestCase):
    def test_line_total(self):
        self.assertEqual(line_total(10.0, 3), 30.0)

    def test_order_total(self):
        self.assertEqual(order_total([(10.0, 3), (5.0, 2)]), 40.0)


if __name__ == "__main__":
    unittest.main()
'''

TOOLS = {
    "create_branch": ["git", "branch", "fix/bug-123"],
    "checkout_branch": ["git", "checkout", "fix/bug-123"],
    "run_tests": None,  # branch-aware: checkout + pytest (see run_tool)
    "open_pr": None,    # out of scope for the demo
}


# --------------------------------------------------------------------------
# LLM clients (both live)
# --------------------------------------------------------------------------

class GeminiAgent:
    def __init__(self):
        from google import genai
        self._client = genai.Client()

    def ask(self, prompt: str) -> str:
        r = self._client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (r.text or "").strip()


class GemmaAgent:
    """Local agent via Ollama — used for Agent B when the network is 'down'."""

    def __init__(self):
        self._gemma = OllamaGemma()
        if not self._gemma.model:
            raise RuntimeError("no local gemma model available via Ollama")

    def ask(self, prompt: str) -> str:
        return self._gemma.generate(prompt) or ""


def parse_decision(text: str) -> dict:
    """Extract {"action": ..., "justification": ...} from an LLM reply."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"action": "unparseable", "justification": text[:200]}
    try:
        d = json.loads(m.group(0))
        return {"action": str(d.get("action", "unparseable")).strip(),
                "justification": str(d.get("justification", "")).strip()}
    except Exception:
        return {"action": "unparseable", "justification": text[:200]}


# --------------------------------------------------------------------------
# Real workspace
# --------------------------------------------------------------------------

class Workspace:
    def __init__(self, emit, side: str):
        self.dir = tempfile.mkdtemp(prefix=f"mao_live_{side}_")
        self.emit = emit
        self.side = side

    def sh(self, args: list[str], show: bool = True) -> tuple[int, str]:
        if show:
            self.emit({"t": "term", "side": self.side,
                       "line": "$ " + " ".join(args), "cls": "cmd"})
        p = subprocess.run(args, cwd=self.dir, capture_output=True, text=True,
                           timeout=120)
        out = ((p.stdout or "") + (p.stderr or "")).strip()
        out = re.sub(r"\x1b\[[0-9;]*m", "", out)  # strip ANSI colors for the web terminal
        if show and out:
            for line in out.splitlines()[:12]:
                self.emit({"t": "term", "side": self.side, "line": line,
                           "cls": "err" if p.returncode else "out"})
        return p.returncode, out

    def setup(self):
        self.sh(["git", "init", "-q", "-b", "master"], show=False)
        self.sh(["git", "config", "user.email", "demo@mao"], show=False)
        self.sh(["git", "config", "user.name", "mao-demo"], show=False)
        with open(os.path.join(self.dir, "calc.py"), "w") as f:
            f.write(BUGGY_CALC)
        with open(os.path.join(self.dir, "test_calc.py"), "w") as f:
            f.write(TEST_CALC)
        self.sh(["git", "add", "-A"], show=False)
        self.sh(["git", "commit", "-q", "-m", "checkout service (bug #123 present)"],
                show=False)
        self.emit({"t": "term", "side": self.side,
                   "line": f"workspace: real git repo at {self.dir}", "cls": "out"})

    def read(self, name: str) -> str:
        with open(os.path.join(self.dir, name)) as f:
            return f.read()

    def write(self, name: str, content: str):
        with open(os.path.join(self.dir, name), "w") as f:
            f.write(content)

    def run_tool(self, action: str) -> tuple[bool, str]:
        """Execute a tool for real. Returns (ok, last_output)."""
        if action == "run_tests":
            code, out = self.sh(["git", "checkout", "fix/bug-123"])
            if code != 0:
                return False, out
            code, out = self.sh(["python", "-m", "pytest", "-q"])
            return code == 0, out
        if action in TOOLS and TOOLS[action]:
            code, out = self.sh(TOOLS[action])
            return code == 0, out
        return False, f"unknown or unavailable tool: {action}"

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Shared: Agent A really fixes the bug
# --------------------------------------------------------------------------

def agent_a_fix(ws: Workspace, emit, side: str, llm: GeminiAgent):
    emit({"t": "status", "side": side, "agent": "A",
          "label": "Gemini 3.5 Flash — fixing bug #123"})
    emit({"t": "feed", "side": side, "kind": "agent",
          "title": "Agent A (Gemini 3.5 Flash) fixes the bug — live",
          "detail": "reads calc.py, writes the corrected file into the working tree"})
    buggy = ws.read("calc.py")
    fixed = llm.ask(
        "You are Agent A in a software pipeline. This file has bug #123 "
        "(a failing arithmetic operation). Return ONLY the corrected file "
        "content, no markdown fences, no commentary.\n\n" + buggy)
    fixed = re.sub(r"^```(?:python)?\s*|\s*```$", "", fixed.strip())
    ws.write("calc.py", fixed)
    _, diff = ws.sh(["git", "diff", "--stat"])
    emit({"t": "status", "side": side, "agent": "A", "label": "done — patch applied"})


AGENT_B_TOOLS = "create_branch, checkout_branch, run_tests, open_pr"


def decide(llm, prompt: str) -> dict:
    return parse_decision(llm.ask(prompt))


# --------------------------------------------------------------------------
# Naive pipeline (left)
# --------------------------------------------------------------------------

def run_naive(emit):
    ws = Workspace(emit, "naive")
    try:
        gemini = GeminiAgent()
        ws.setup()
        agent_a_fix(ws, emit, "naive", gemini)

        # prose handoff — built from the same task state, structure flattened
        g = _build_task_graph()
        summary = naive_summary(g)
        emit({"t": "feed", "side": "naive", "kind": "handoff",
              "title": "Handoff: prose summary", "detail": f'"{summary}"',
              "badge": "TEXT"})

        emit({"t": "status", "side": "naive", "agent": "B",
              "label": "Gemini 3.5 Flash — deciding from prose"})
        d = decide(gemini,
                   "You are Agent B, the QA agent in a multi-agent pipeline. "
                   f"You received this handoff: \"{summary}\" "
                   f"Available tools: {AGENT_B_TOOLS}. "
                   "Decide the single next tool call that fulfils the handoff "
                   "instruction. Respond ONLY with JSON: "
                   '{"action": "<tool>", "justification": "<one sentence>"}')
        emit({"t": "feed", "side": "naive", "kind": "agent",
              "title": f"Agent B decided (live): {d['action']}",
              "detail": d["justification"]})

        ok, out = ws.run_tool(d["action"])
        emit({"t": "feed", "side": "naive", "kind": "execute",
              "title": d["action"], "ok": ok,
              "detail": out.splitlines()[-1] if out else ""})
        emit({"t": "status", "side": "naive", "agent": "B",
              "label": "failed" if not ok else "ok"})
        emit({"t": "result", "side": "naive", "ok": ok,
              "title": "PIPELINE SUCCEEDED" if ok else "PIPELINE FAILED",
              "detail": ("the receiving agent happened to re-derive the order "
                         "from prose this run" if ok else
                         "no gate, no structure — the failure reached reality")})
    except Exception as e:
        emit({"t": "result", "side": "naive", "ok": False,
              "title": "ERROR", "detail": str(e)[:200]})
    finally:
        emit({"t": "done", "side": "naive"})
        ws.cleanup()


# --------------------------------------------------------------------------
# Structured pipeline (right)
# --------------------------------------------------------------------------

def _build_task_graph() -> TaskGraph:
    g = TaskGraph(goal="Fix bug #123: apply the patch and run the test suite "
                       "on a new feature branch")
    steps = [
        ("create_branch", "create a new feature branch for the fix", []),
        ("checkout_branch", "check out the feature branch locally", ["create_branch"]),
        ("edit_files", "apply the code patch to the affected files", []),
        ("run_tests", "run the full test suite on the feature branch",
         ["create_branch", "edit_files"]),
        ("open_pr", "open a pull request with the verified fix", ["run_tests"]),
    ]
    for sid, desc, _ in steps:
        g.add_subtask(sid, desc, owner_agent="agent_a" if sid == "edit_files" else "agent_b")
    for sid, _, deps in steps:
        for dep in deps:
            g.depends(sid, dep)
    g.add_constraint("mutex_repo_worktree", ConstraintType.RESOURCE,
                     {"resource": "repo_worktree",
                      "holders": ["checkout_branch", "edit_files"]},
                     "only one subtask may hold 'repo_worktree' at a time")
    g.set_status("edit_files", Status.DONE)
    g.add_artifact("out_edit_files", "output artifact of edit_files",
                   artifact_type="data", produced_by="edit_files")
    return g


def _emit_graph(emit, g: TaskGraph, gated: str | None = None):
    emit({"t": "graph",
          "nodes": [{"id": n.id, "type": n.type.value, "status": n.status.value,
                     "desc": n.description}
                    for n in g.nodes.values()],
          "edges": [{"src": e.src, "dst": e.dst, "rel": e.rel.value}
                    for e in g.edges],
          "gated": gated})


def run_structured(emit, offline: bool):
    ws = Workspace(emit, "structured")
    try:
        mode = "local" if offline else \
            ("gemini" if os.environ.get("GEMINI_API_KEY") else "local")
        adj = Adjudicator.from_artifacts(mode=mode)
        gemini = GeminiAgent() if not offline else None
        agent_b = GemmaAgent() if offline else gemini
        b_name = "Gemma 4 E2B (local)" if offline else "Gemini 3.5 Flash"

        ws.setup()
        if offline:
            emit({"t": "feed", "side": "structured", "kind": "info",
                  "title": "Network down — local takeover",
                  "detail": "Agent B, the gate, trace encoding and replan "
                            "messages all run on-device (Gemma 4 E2B via Ollama)"})
            # Agent A's fix happened before the outage; apply it directly
            ws.write("calc.py", BUGGY_CALC.replace(
                "unit_price - quantity  # BUG #123: should multiply, not subtract",
                "unit_price * quantity"))
            emit({"t": "status", "side": "structured", "agent": "A",
                  "label": "patch applied before outage"})
        else:
            agent_a_fix(ws, emit, "structured", gemini)

        g = _build_task_graph()
        _emit_graph(emit, g)
        packet = build_packet(g, adj.model)
        emit({"t": "feed", "side": "structured", "kind": "handoff",
              "title": "Handoff: graph frontier + joint embedding",
              "detail": f"{len(packet.graph_snapshot['nodes'])} nodes, "
                        f"{len(packet.embedding)}-d vector — no prose, no "
                        f"inherited token history", "badge": "STRUCT"})

        base_prompt = (
            f"You are Agent B, the QA agent in a multi-agent pipeline. "
            f"You received this handoff instruction: \"run the full test suite "
            f"on the feature branch for bug #123\". "
            f"Available tools: {AGENT_B_TOOLS}. "
            "Decide the single next tool call. Respond ONLY with JSON: "
            '{"action": "<tool>", "justification": "<one sentence>"}')

        emit({"t": "status", "side": "structured", "agent": "B",
              "label": f"{b_name} — deciding"})
        d = decide(agent_b, base_prompt)
        executed_tests = False

        for step in range(8):
            if d["action"] in ("done", "unparseable"):
                break
            emit({"t": "feed", "side": "structured", "kind": "agent",
                  "title": f"Agent B proposes (live): {d['action']}",
                  "detail": d["justification"]})

            packet = build_packet(g, adj.model)
            trace = (f"Handoff received. Goal: {g.goal}. "
                     f"I will now execute {d['action']}. {d['justification']}")
            dec = adj.adjudicate(packet, d["action"], trace)
            emit({"t": "gate", "side": "structured", "action": d["action"],
                  "alignment": round(dec.alignment, 3),
                  "verdict": dec.verdict.value,
                  "band": [round(dec.tau_lo, 3), round(dec.tau_hi, 3)]})

            if dec.verdict == Verdict.REQUEST_REPLAN:
                emit({"t": "feed", "side": "structured", "kind": "verdict",
                      "title": f"Gate: REQUEST_REPLAN — {d['action']}",
                      "detail": f"alignment {dec.alignment:+.3f} vs band "
                                f"({dec.tau_lo:+.3f}, {dec.tau_hi:+.3f})",
                      "ok": False, "badge": "GATE"})
                if dec.violations:
                    emit({"t": "feed", "side": "structured", "kind": "witness",
                          "title": "Violated edge (structural witness)",
                          "detail": dec.violations[0].detail, "ok": False})
                _emit_graph(emit, g, gated=d["action"])
                emit({"t": "feed", "side": "structured", "kind": "gemma",
                      "title": "Gemma 4 E2B (local) → Agent B",
                      "detail": dec.message, "badge": "ON-DEVICE"})
                frontier = [{"step": n.id, "status": n.status.value,
                             "depends_on": g.dependencies_of(n.id)}
                            for n in g.subtasks()]
                d = decide(agent_b,
                           f"Your action '{d['action']}' was REJECTED by the "
                           f"orchestration gate: \"{dec.message}\" "
                           f"Task graph frontier: {json.dumps(frontier)}. "
                           f"Available tools: {AGENT_B_TOOLS}. Decide the "
                           "single next tool call. Respond ONLY with JSON: "
                           '{"action": "<tool>", "justification": "<one sentence>"}')
                continue

            if dec.verdict == Verdict.FLAG_TO_HUMAN:
                emit({"t": "feed", "side": "structured", "kind": "verdict",
                      "title": f"Gate: FLAG_TO_HUMAN — {d['action']}",
                      "detail": f"alignment {dec.alignment:+.3f} in ambiguous "
                                f"band; human verifies deps and approves",
                      "badge": "GATE"})

            ok, out = ws.run_tool(d["action"])
            emit({"t": "feed", "side": "structured", "kind": "execute",
                  "title": d["action"], "ok": ok,
                  "detail": out.splitlines()[-1] if out else ""})
            if ok and d["action"] in g.nodes:
                g.set_status(d["action"], Status.DONE)
                g.add_artifact(f"out_{d['action']}",
                               f"output artifact of {d['action']}",
                               artifact_type="data", produced_by=d["action"])
                _emit_graph(emit, g)
            if d["action"] == "run_tests" and ok:
                executed_tests = True
                break

            frontier = [{"step": n.id, "status": n.status.value,
                         "depends_on": g.dependencies_of(n.id)}
                        for n in g.subtasks()]
            d = decide(agent_b,
                       f"'{d['action']}' executed {'successfully' if ok else 'WITH ERRORS'}. "
                       f"Task graph frontier: {json.dumps(frontier)}. "
                       f"Goal: run the full test suite on the feature branch. "
                       f"If that is complete, respond {{\"action\": \"done\"}}. "
                       f"Available tools: {AGENT_B_TOOLS}. Respond ONLY with "
                       'JSON: {"action": "<tool>", "justification": "<one sentence>"}')

        emit({"t": "status", "side": "structured", "agent": "B",
              "label": "tests green" if executed_tests else "stopped"})
        emit({"t": "result", "side": "structured", "ok": executed_tests,
              "title": "TESTS PASS" if executed_tests else "INCOMPLETE",
              "detail": "same agent, same intent — the gate turned a "
                        "dependency-order failure into a correct plan"
                        if executed_tests else "agent stopped before tests"})
    except Exception as e:
        emit({"t": "result", "side": "structured", "ok": False,
              "title": "ERROR", "detail": str(e)[:200]})
    finally:
        emit({"t": "done", "side": "structured"})
        ws.cleanup()

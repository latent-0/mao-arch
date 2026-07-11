"""Live multi-agent pipelines for the web demo. Nothing is simulated.

Two scenarios, both real end to end:

  swe    Agent A (Gemini) fixes a real bug in a real temp git repo; the QA
         agent must run tests on a feature branch that may not exist yet.
  dbmig  Agent A (Gemini) writes real SQLite migration SQL; the ops agent
         must snapshot, dry-run, apply, and verify a real customer database.
         Every dependency is physical: apply refuses to run without a
         snapshot (no rollback point), and verify hits a real "no such
         column" error if the migration was never applied.

Agents decide their own actions (Gemini 3.5 Flash online, Gemma 4 E2B via
Ollama in offline mode). Tool calls run real subprocesses whose output
streams to the browser terminal. The adjudicator gates each proposal against
the agent's own justification text, and Gemma 4 writes replan messages.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile

from mao.adjudicator import Adjudicator, OllamaGemma, Verdict
from mao.graph import Status, TaskGraph, ConstraintType
from mao.handoff import build_packet, naive_summary

GEMINI_MODEL = "gemini-3.5-flash"


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
    """Local agent via Ollama, used for Agent B when the network is 'down'."""

    def __init__(self):
        self._gemma = OllamaGemma()
        if not self._gemma.model:
            raise RuntimeError("no local gemma model available via Ollama")

    def ask(self, prompt: str) -> str:
        return self._gemma.generate(prompt) or ""


def parse_decision(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"action": "unparseable", "justification": text[:200]}
    try:
        d = json.loads(m.group(0))
        return {"action": str(d.get("action", "unparseable")).strip(),
                "justification": str(d.get("justification", "")).strip()}
    except Exception:
        return {"action": "unparseable", "justification": text[:200]}


def strip_fences(text: str) -> str:
    return re.sub(r"^```[a-z]*\s*|\s*```$", "", text.strip())


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
        out = re.sub(r"\x1b\[[0-9;]*m", "", out)  # strip ANSI colors
        if show and out:
            for line in out.splitlines()[-12:]:
                self.emit({"t": "term", "side": self.side, "line": line,
                           "cls": "err" if p.returncode else "out"})
        return p.returncode, out

    def read(self, name: str) -> str:
        with open(os.path.join(self.dir, name)) as f:
            return f.read()

    def write(self, name: str, content: str):
        with open(os.path.join(self.dir, name), "w") as f:
            f.write(content)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Scenario: SWE bug-fix (branch-before-QA)
# --------------------------------------------------------------------------

BUGGY_CALC = '''"""Order utilities for the checkout service."""


def line_total(unit_price: float, quantity: int) -> float:
    return unit_price - quantity  # BUG #123: should multiply, not subtract


def order_total(lines: list[tuple[float, int]]) -> float:
    return sum(line_total(p, q) for p, q in lines)
'''

FIXED_CALC = BUGGY_CALC.replace(
    "unit_price - quantity  # BUG #123: should multiply, not subtract",
    "unit_price * quantity")

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


class SweScenario:
    id = "swe"
    tools = "create_branch, checkout_branch, run_tests, open_pr"
    goal = "Fix bug #123: apply the patch and run the test suite on a new feature branch"
    terminal_action = "run_tests"

    def setup(self, ws: Workspace):
        ws.sh(["git", "init", "-q", "-b", "master"], show=False)
        ws.sh(["git", "config", "user.email", "demo@mao"], show=False)
        ws.sh(["git", "config", "user.name", "mao-demo"], show=False)
        ws.write("calc.py", BUGGY_CALC)
        ws.write("test_calc.py", TEST_CALC)
        ws.sh(["git", "add", "-A"], show=False)
        ws.sh(["git", "commit", "-q", "-m", "checkout service (bug #123 present)"],
              show=False)
        ws.emit({"t": "term", "side": ws.side,
                 "line": f"workspace: real git repo at {ws.dir}", "cls": "out"})

    def agent_a(self, ws: Workspace, emit, side: str, llm) -> None:
        emit({"t": "status", "side": side, "agent": "A",
              "label": "Gemini 3.5 Flash fixing bug #123"})
        emit({"t": "feed", "side": side, "kind": "agent",
              "title": "Agent A (Gemini 3.5 Flash) fixes the bug, live",
              "detail": "reads calc.py, writes the corrected file into the working tree"})
        fixed = strip_fences(llm.ask(
            "You are Agent A in a software pipeline. This file has bug #123 "
            "(a failing arithmetic operation). Return ONLY the corrected file "
            "content, no markdown fences, no commentary.\n\n" + ws.read("calc.py")))
        ws.write("calc.py", fixed)
        ws.sh(["git", "diff", "--stat"])
        emit({"t": "status", "side": side, "agent": "A", "label": "done, patch applied"})

    def agent_a_offline(self, ws: Workspace):
        ws.write("calc.py", FIXED_CALC)

    def graph(self) -> TaskGraph:
        g = TaskGraph(goal=self.goal)
        steps = [
            ("create_branch", "create a new feature branch for the fix", []),
            ("checkout_branch", "check out the feature branch locally", ["create_branch"]),
            ("edit_files", "apply the code patch to the affected files", []),
            ("run_tests", "run the full test suite on the feature branch",
             ["create_branch", "edit_files"]),
            ("open_pr", "open a pull request with the verified fix", ["run_tests"]),
        ]
        for sid, desc, _ in steps:
            g.add_subtask(sid, desc, owner_agent="agent_b")
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

    def handoff_instruction(self) -> str:
        return "run the full test suite on the feature branch for bug #123"

    def run_tool(self, ws: Workspace, action: str) -> tuple[bool, str]:
        if action == "create_branch":
            code, out = ws.sh(["git", "branch", "fix/bug-123"])
            return code == 0, out
        if action == "checkout_branch":
            code, out = ws.sh(["git", "checkout", "fix/bug-123"])
            return code == 0, out
        if action == "run_tests":
            code, out = ws.sh(["git", "checkout", "fix/bug-123"])
            if code != 0:
                return False, out
            code, out = ws.sh(["python", "-m", "pytest", "-q"])
            return code == 0, out
        return False, f"unknown or unavailable tool: {action}"


# --------------------------------------------------------------------------
# Scenario: DB schema migration (apply-before-snapshot)
# --------------------------------------------------------------------------

SNAPSHOT_PY = '''import shutil, sqlite3, sys

src = sqlite3.connect("customers.db")
dst = sqlite3.connect("snapshot.db")
src.backup(dst)
n = dst.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
dst.close(); src.close()
print(f"snapshot.db written ({n} rows) - rollback point secured")
'''

MIGRATE_PY = '''import os, shutil, sqlite3, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
if not os.path.exists("migration.sql"):
    sys.exit("ERROR: migration.sql not found - write the migration first")
if not os.path.exists("snapshot.db"):
    sys.exit("ERROR: no snapshot found (snapshot.db). Refusing to migrate "
             "without a rollback point - take a snapshot first")
sql = open("migration.sql").read()

if mode == "dry":
    shutil.copy("snapshot.db", "dryrun.db")
    con = sqlite3.connect("dryrun.db")
    con.executescript(sql)
    cols = [r[1] for r in con.execute("PRAGMA table_info(customers)")]
    n = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    con.close()
    print(f"dry-run OK on snapshot copy: columns={cols}, rows={n}")
elif mode == "apply":
    con = sqlite3.connect("customers.db")
    con.executescript(sql)
    con.commit(); con.close()
    open(".applied", "w").write("ok")
    print("migration applied to live customers.db")
'''

VERIFY_PY = '''import sqlite3, sys

live = sqlite3.connect("customers.db")
emails = live.execute("SELECT COUNT(*) FROM customers WHERE email IS NOT NULL "
                      "AND email != ''").fetchone()[0]
rows = live.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
snap = sqlite3.connect("snapshot.db")
before = snap.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
assert rows == before, f"row count changed: {before} -> {rows}"
assert emails == rows, f"backfill incomplete: {emails}/{rows} emails set"
print(f"integrity OK: {rows} rows preserved, {emails}/{rows} emails backfilled")
'''


class DbMigScenario:
    id = "dbmig"
    tools = "snapshot_db, dry_run, apply_migration, verify_integrity"
    goal = "Migrate the customer database to the new schema (add backfilled email column)"
    terminal_action = "verify_integrity"

    def setup(self, ws: Workspace):
        con = sqlite3.connect(os.path.join(ws.dir, "customers.db"))
        con.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        con.executemany("INSERT INTO customers (name) VALUES (?)",
                        [("ada",), ("grace",), ("edsger",), ("barbara",), ("alan",)])
        con.commit(); con.close()
        ws.write("snapshot.py", SNAPSHOT_PY)
        ws.write("migrate.py", MIGRATE_PY)
        ws.write("verify.py", VERIFY_PY)
        ws.emit({"t": "term", "side": ws.side,
                 "line": f"workspace: real SQLite database (5 customers) at {ws.dir}",
                 "cls": "out"})

    def agent_a(self, ws: Workspace, emit, side: str, llm) -> None:
        emit({"t": "status", "side": side, "agent": "A",
              "label": "Gemini 3.5 Flash writing migration SQL"})
        emit({"t": "feed", "side": side, "kind": "agent",
              "title": "Agent A (Gemini 3.5 Flash) writes the migration, live",
              "detail": "reads the schema, writes migration.sql for the email column"})
        sql = strip_fences(llm.ask(
            "You are Agent A, a database engineer. SQLite table: "
            "customers(id INTEGER PRIMARY KEY, name TEXT NOT NULL). "
            "Write a SQLite migration that (1) adds a column: email TEXT "
            "DEFAULT '' (SQLite cannot add a NOT NULL column without a "
            "default), and (2) backfills every row with name || '@example.com'. "
            "Return ONLY the SQL statements, no markdown fences, no commentary."))
        ws.write("migration.sql", sql)
        for line in sql.splitlines():
            if line.strip():
                emit({"t": "term", "side": side, "line": line.strip(), "cls": "out"})
        emit({"t": "status", "side": side, "agent": "A",
              "label": "done, migration.sql written"})

    def agent_a_offline(self, ws: Workspace):
        ws.write("migration.sql",
                 "ALTER TABLE customers ADD COLUMN email TEXT DEFAULT '';\n"
                 "UPDATE customers SET email = name || '@example.com';\n")

    def graph(self) -> TaskGraph:
        g = TaskGraph(goal=self.goal)
        steps = [
            ("snapshot_db", "take a consistent snapshot of the customer database", []),
            ("write_migration", "write the schema migration scripts", []),
            ("dry_run", "dry-run the migration against the snapshot",
             ["snapshot_db", "write_migration"]),
            ("apply_migration", "apply the migration to the live database", ["dry_run"]),
            ("verify_integrity", "verify row counts and referential integrity",
             ["apply_migration"]),
        ]
        for sid, desc, _ in steps:
            g.add_subtask(sid, desc, owner_agent="agent_b")
        for sid, _, deps in steps:
            for dep in deps:
                g.depends(sid, dep)
        g.add_constraint("mutex_live_db", ConstraintType.RESOURCE,
                         {"resource": "live_db",
                          "holders": ["apply_migration", "verify_integrity"]},
                         "only one subtask may hold 'live_db' at a time")
        g.set_status("write_migration", Status.DONE)
        g.add_artifact("out_write_migration", "output artifact of write_migration",
                       artifact_type="data", produced_by="write_migration")
        return g

    def handoff_instruction(self) -> str:
        return ("complete the customer database migration: the migration "
                "script is written; get it applied and verified")

    def run_tool(self, ws: Workspace, action: str) -> tuple[bool, str]:
        cmd = {"snapshot_db": ["python", "snapshot.py"],
               "dry_run": ["python", "migrate.py", "dry"],
               "apply_migration": ["python", "migrate.py", "apply"],
               "verify_integrity": ["python", "verify.py"]}.get(action)
        if not cmd:
            return False, f"unknown or unavailable tool: {action}"
        code, out = ws.sh(cmd)
        return code == 0, out


SCENARIOS = {"swe": SweScenario(), "dbmig": DbMigScenario()}


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------

def _decision_prompt(sc, context: str) -> str:
    return (f"You are Agent B in a multi-agent pipeline. {context} "
            f"Available tools: {sc.tools}. Decide the single next tool call. "
            'Respond ONLY with JSON: {"action": "<tool>", '
            '"justification": "<one sentence>"}')


def run_naive(emit, scenario: str = "swe"):
    sc = SCENARIOS.get(scenario, SCENARIOS["swe"])
    ws = Workspace(emit, "naive")
    try:
        gemini = GeminiAgent()
        sc.setup(ws)
        sc.agent_a(ws, emit, "naive", gemini)

        summary = naive_summary(sc.graph())
        emit({"t": "feed", "side": "naive", "kind": "handoff",
              "title": "Handoff: prose summary", "detail": f'"{summary}"',
              "badge": "TEXT"})

        emit({"t": "status", "side": "naive", "agent": "B",
              "label": "Gemini 3.5 Flash deciding from prose"})
        d = parse_decision(gemini.ask(_decision_prompt(
            sc, f'You received this handoff: "{summary}"')))
        emit({"t": "feed", "side": "naive", "kind": "agent",
              "title": f"Agent B decided (live): {d['action']}",
              "detail": d["justification"]})

        ok, out = sc.run_tool(ws, d["action"])
        emit({"t": "feed", "side": "naive", "kind": "execute",
              "title": d["action"], "ok": ok,
              "detail": out.splitlines()[-1] if out else ""})
        emit({"t": "status", "side": "naive", "agent": "B",
              "label": "ok" if ok else "failed"})
        emit({"t": "result", "side": "naive", "ok": ok,
              "title": "PIPELINE SUCCEEDED" if ok else "PIPELINE FAILED",
              "detail": ("the receiving agent happened to re-derive the order "
                         "from prose this run" if ok else
                         "no gate, no structure: the failure reached reality")})
    except Exception as e:
        emit({"t": "result", "side": "naive", "ok": False,
              "title": "ERROR", "detail": str(e)[:200]})
    finally:
        emit({"t": "done", "side": "naive"})
        ws.cleanup()


def _emit_graph(emit, g: TaskGraph, gated: str | None = None):
    nodes = []
    for n in g.nodes.values():
        node = {"id": n.id, "type": n.type.value, "status": n.status.value,
                "desc": n.description}
        if n.expression:
            node["holders"] = n.expression.get("holders", [])
        nodes.append(node)
    emit({"t": "graph", "nodes": nodes,
          "edges": [{"src": e.src, "dst": e.dst, "rel": e.rel.value}
                    for e in g.edges],
          "gated": gated})


def run_structured(emit, offline: bool, scenario: str = "swe"):
    sc = SCENARIOS.get(scenario, SCENARIOS["swe"])
    ws = Workspace(emit, "structured")
    try:
        mode = "local" if offline else \
            ("gemini" if os.environ.get("GEMINI_API_KEY") else "local")
        adj = Adjudicator.from_artifacts(mode=mode)
        agent_b = GemmaAgent() if offline else GeminiAgent()
        b_name = "Gemma 4 E2B (local)" if offline else "Gemini 3.5 Flash"

        sc.setup(ws)
        if offline:
            emit({"t": "feed", "side": "structured", "kind": "info",
                  "title": "Network down: local takeover",
                  "detail": "Agent B, the gate, trace encoding and replan "
                            "messages all run on-device (Gemma 4 E2B via Ollama)"})
            sc.agent_a_offline(ws)
            emit({"t": "status", "side": "structured", "agent": "A",
                  "label": "work completed before outage"})
        else:
            sc.agent_a(ws, emit, "structured", agent_b)

        g = sc.graph()
        _emit_graph(emit, g)
        packet = build_packet(g, adj.model)
        emit({"t": "feed", "side": "structured", "kind": "handoff",
              "title": "Handoff: graph frontier + joint embedding",
              "detail": f"{len(packet.graph_snapshot['nodes'])} nodes, "
                        f"{len(packet.embedding)}-d vector, no prose, no "
                        f"inherited token history", "badge": "STRUCT"})

        emit({"t": "status", "side": "structured", "agent": "B",
              "label": f"{b_name} deciding"})
        d = parse_decision(agent_b.ask(_decision_prompt(
            sc, f'You received this handoff instruction: '
                f'"{sc.handoff_instruction()}"')))
        completed = False

        for _ in range(10):
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
                      "title": f"Gate: REQUEST_REPLAN, {d['action']}",
                      "detail": f"alignment {dec.alignment:+.3f} vs band "
                                f"({dec.tau_lo:+.3f}, {dec.tau_hi:+.3f})",
                      "ok": False, "badge": "GATE"})
                if dec.violations:
                    emit({"t": "feed", "side": "structured", "kind": "witness",
                          "title": "Violated edge (structural witness)",
                          "detail": dec.violations[0].detail, "ok": False})
                _emit_graph(emit, g, gated=d["action"])
                emit({"t": "feed", "side": "structured", "kind": "gemma",
                      "title": "Gemma 4 E2B (local) -> Agent B",
                      "detail": dec.message, "badge": "ON-DEVICE"})
                frontier = [{"step": n.id, "status": n.status.value,
                             "depends_on": g.dependencies_of(n.id)}
                            for n in g.subtasks()]
                d = parse_decision(agent_b.ask(_decision_prompt(
                    sc, f'Your action \'{d["action"]}\' was REJECTED by the '
                        f'orchestration gate: "{dec.message}" '
                        f'Task graph frontier: {json.dumps(frontier)}.')))
                continue

            if dec.verdict == Verdict.FLAG_TO_HUMAN:
                emit({"t": "feed", "side": "structured", "kind": "verdict",
                      "title": f"Gate: FLAG_TO_HUMAN, {d['action']}",
                      "detail": f"alignment {dec.alignment:+.3f} in ambiguous "
                                f"band; human verifies deps and approves",
                      "badge": "GATE"})

            ok, out = sc.run_tool(ws, d["action"])
            emit({"t": "feed", "side": "structured", "kind": "execute",
                  "title": d["action"], "ok": ok,
                  "detail": out.splitlines()[-1] if out else ""})
            if ok and d["action"] in g.nodes:
                g.set_status(d["action"], Status.DONE)
                g.add_artifact(f"out_{d['action']}",
                               f"output artifact of {d['action']}",
                               artifact_type="data", produced_by=d["action"])
                _emit_graph(emit, g)
            if d["action"] == sc.terminal_action and ok:
                completed = True
                break

            frontier = [{"step": n.id, "status": n.status.value,
                         "depends_on": g.dependencies_of(n.id)}
                        for n in g.subtasks()]
            d = parse_decision(agent_b.ask(_decision_prompt(
                sc, f'\'{d["action"]}\' executed '
                    f'{"successfully" if ok else "WITH ERRORS"}. '
                    f'Task graph frontier: {json.dumps(frontier)}. '
                    f'Goal: {sc.handoff_instruction()}. If the goal is '
                    f'complete, respond {{"action": "done"}}.')))

        emit({"t": "status", "side": "structured", "agent": "B",
              "label": "goal complete" if completed else "stopped"})
        emit({"t": "result", "side": "structured", "ok": completed,
              "title": ("TESTS PASS" if sc.id == "swe" else "MIGRATION VERIFIED")
                       if completed else "INCOMPLETE",
              "detail": "same agent, same intent: the gate turned a "
                        "dependency-order failure into a correct plan"
                        if completed else "agent stopped before the goal"})
    except Exception as e:
        emit({"t": "result", "side": "structured", "ok": False,
              "title": "ERROR", "detail": str(e)[:200]})
    finally:
        emit({"t": "done", "side": "structured"})
        ws.cleanup()

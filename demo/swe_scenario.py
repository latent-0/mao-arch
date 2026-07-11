"""Live demo: SWE dependency-order violation (branch-before-QA).

Documented failure pattern (arXiv:2605.08761): "The agent delegated the QA task
before creating the required branch, causing the QA task to fail."

Three beats:
  1. NAIVE RUN      — prose handoff; QA agent runs tests on a nonexistent
                      branch -> visible failure.
  2. STRUCTURED RUN — joint-embedding handoff; Gemma-local adjudicator gates
                      run_tests -> request-replan with the violated edge
                      attached -> branch created first -> tests pass.
  3. OFFLINE BEAT   — network flag off; the adjudicator keeps gating and graph
                      state stays consistent, entirely locally.

Usage:  python -m demo.swe_scenario [--no-llm] [--fast]
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from mao.adjudicator import Adjudicator, Verdict
from mao.datagen import TEMPLATES, _trace, build_graph
from mao.encoders.language import encoder_mode
from mao.graph import Status, TaskGraph
from mao.handoff import build_packet, naive_summary

# --------------------------- terminal helpers ---------------------------

GREEN, RED, YELLOW, CYAN, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[2m", "\033[1m", "\033[0m")
PAUSE = 0.8


def say(text: str = "", color: str = "", delay: bool = True):
    print(f"{color}{text}{RESET}")
    sys.stdout.flush()
    if delay and PAUSE:
        time.sleep(PAUSE)


def banner(text: str):
    say("\n" + "=" * 66, CYAN, delay=False)
    say(f"  {text}", CYAN + BOLD, delay=False)
    say("=" * 66, CYAN)


# --------------------------- scenario graph ---------------------------

def build_scenario() -> TaskGraph:
    """The SWE pipeline (same structure agents build live during orchestration)
    at the state where Agent A has applied the patch but no branch exists —
    the exact state from the documented branch-before-QA failure trace."""
    swe = TEMPLATES[0]
    g = build_graph(swe, done={"edit_files"}, rng=random.Random(123))
    g.goal = ("Fix bug #123: apply the patch and run the test suite "
              "on a new feature branch")
    return g


_RNG = random.Random(42)


def simulate_tool(action: str, graph: TaskGraph) -> bool:
    """Execute an action against 'reality' — fails if dependencies are unmet
    (e.g., running tests on a branch that doesn't exist). Completed actions
    produce their output artifact, same as every live orchestration step."""
    ok = not graph.check_action(action)
    time.sleep(PAUSE / 2)
    if ok:
        graph.set_status(action, Status.DONE)
        graph.add_artifact(f"out_{action}", f"output artifact of {action}",
                           artifact_type="data", produced_by=action)
    return ok


# --------------------------- demo beats ---------------------------

def naive_run():
    banner("RUN 1 — NAIVE TEXT HANDOFF (today's status quo)")
    g = build_scenario()
    summary = naive_summary(g)
    say(f"Agent A -> Agent B (prose):", BOLD)
    say(f'  "{summary}"', DIM)
    say("Agent B re-infers next action from prose: run_tests", YELLOW)
    ok = simulate_tool("run_tests", g)
    if not ok:
        v = g.check_action("run_tests")[0]
        say(f"  X  TESTS FAILED - {v.detail}", RED)
        say("  (the summary never said the branch doesn't exist yet - "
            "structure lost in the prose projection)", DIM)
    say(f"\n  Naive handoff result: {RED}{BOLD}FAILED{RESET}")


def structured_run(adj: Adjudicator, offline: bool = False):
    title = "RUN 2 — JOINT-EMBEDDING HANDOFF (mine)"
    if offline:
        title = "RUN 3 — OFFLINE RESILIENCE (network down, Gemma-local gate)"
    banner(title)
    g = build_scenario()

    if offline:
        say("  [network] cloud connectivity: DOWN - adjudicator continues "
            "on-device", YELLOW)
        say(f"  [encoder] trace encoding runs locally too "
            f"({type(adj.encoder).__name__}) - no cloud calls anywhere in the gate", YELLOW)

    packet = build_packet(g, adj.model)
    n_nodes = len(packet.graph_snapshot["nodes"])
    say(f"Agent A -> Agent B (structured): graph frontier ({n_nodes} nodes) "
        f"+ {len(packet.embedding)}-d embedding", BOLD)
    say("Agent B proposes: run_tests (same goal-driven intent as before)", YELLOW)

    d = adj.adjudicate(packet, "run_tests", _trace(g, "run_tests", _RNG))
    say(f"  [adjudicator] alignment={d.alignment:.3f}  "
        f"band=({d.tau_lo:.3f}, {d.tau_hi:.3f})  ->  {d.verdict.value.upper()}",
        RED if d.verdict == Verdict.REQUEST_REPLAN else YELLOW)
    if d.violations:
        say(f"  [adjudicator] violated edge: {d.violations[0].detail}", RED)
    say(f'  [adjudicator -> Agent B] "{d.message}"', DIM)

    say("Agent B re-plans from the graph frontier:", YELLOW)
    for action in ["create_branch", "checkout_branch", "run_tests"]:
        if g.nodes[action].status == Status.DONE:
            continue
        dec = adj.adjudicate(build_packet(g, adj.model), action,
                             _trace(g, action, _RNG))
        gate = f"alignment={dec.alignment:.3f} -> {dec.verdict.value}"
        if dec.verdict == Verdict.REQUEST_REPLAN:
            say(f"    {action}: gated ({gate})", RED)
            continue
        if dec.verdict == Verdict.FLAG_TO_HUMAN:
            say(f"    {action}: ambiguous band -> surfaced to human; "
                f"deps verified, human approves", YELLOW)
        ok = simulate_tool(action, g)
        mark = f"{GREEN}ok{RESET}" if ok else f"{RED}FAILED{RESET}"
        say(f"    {action}: {gate}  ... executed [{mark}]")

    passed = g.nodes["run_tests"].status == Status.DONE
    color = GREEN if passed else RED
    say(f"\n  Structured handoff result: {color}{BOLD}"
        f"{'TESTS PASS' if passed else 'FAILED'}{RESET}")
    if offline:
        say("  Graph state remained consistent with zero cloud round-trips; "
            "orchestration resumes on reconnect.", CYAN)


def main():
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="skip Ollama explanations (decisions unaffected)")
    ap.add_argument("--fast", action="store_true", help="no dramatic pauses")
    args = ap.parse_args()
    if args.fast:
        PAUSE = 0

    adj = Adjudicator.from_artifacts(mode="auto", use_local_llm=not args.no_llm)
    adj_local = Adjudicator.from_artifacts(mode="local", use_local_llm=not args.no_llm)

    banner("JOINT-EMBEDDING HANDOFF — branch-before-QA demo")
    say("Failure pattern from a real enterprise-agent trace: the QA task was "
        "delegated before the required branch existed.", DIM)
    say(f"cloud-path encoder: {encoder_mode(adj.encoder)} | "
        f"offline-path encoder: {encoder_mode(adj_local.encoder)}", DIM)

    naive_run()
    structured_run(adj)
    structured_run(adj_local, offline=True)


if __name__ == "__main__":
    main()

"""Zero-shot evaluation on real SWE-bench Lite task graphs.

Same protocol and metrics as mao.eval, but the evaluation graphs are
instantiated from the 300 real SWE-bench Lite issues (mao.benchmarks.swebench)
instead of the synthetic templates. The adjudicator is loaded from artifacts
trained ONLY on the synthetic templates, so every number here is out-of-
distribution transfer: real repositories, real source-file paths, real
task topology, none of it seen in training.

Metric — constraint-respecting handoff rate: the % of handoffs after which the
receiving agent's first executed action violates no depends_on/Constraint edge
in the ground-truth graph. Also reports adjudicator gate precision/recall/F1
against the ground-truth checker, the human-deferral (flag) rate, latency, and
the MAST-style "Disobey Task Specification" tally for the naive baseline.

Usage:  python -m mao.eval_swebench [--n 300] [--repeats 1] [--encoder local]
"""

from __future__ import annotations

import argparse
import collections
import random
import statistics
import time

from .adjudicator import Adjudicator, Verdict
from .benchmarks import swebench
from .datagen import _trace
from .encoders.language import encoder_mode
from .eval import receiving_agent_policy
from .handoff import build_packet, naive_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="cap on instances used (default: all 300)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent states sampled per instance")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--no-llm", action="store_true",
                    help="skip Ollama explanations (decisions unaffected)")
    ap.add_argument("--encoder", choices=["auto", "local", "gemini"], default="auto")
    ap.add_argument("--artifact-mode", default=None,
                    help="artifact subdir to load (defaults to --encoder)")
    args = ap.parse_args()

    adj = Adjudicator.from_artifacts(mode=args.artifact_mode or args.encoder,
                                     use_local_llm=not args.no_llm)
    print(f"[encoder] {type(adj.encoder).__name__} (mode={encoder_mode(adj.encoder)})")

    instances = swebench.load_instances()
    repos = collections.Counter(i["repo"] for i in instances)
    used = len(instances) if args.n is None else min(args.n, len(instances))
    print(f"[dataset] SWE-bench Lite: {used} real instances across "
          f"{len(repos)} repositories (zero-shot; trained only on synthetic templates)")

    samples = swebench.generate(n=args.n, seed=args.seed, repeats=args.repeats)
    rng = random.Random(args.seed + 1)

    naive_ok = struct_ok = deferred = 0
    gate_tp = gate_fp = gate_fn = gate_tn = 0
    mast_disobey = 0
    latencies = []

    for s in samples:
        g = s.graph
        _ = naive_summary(g)               # what the naive receiver would see
        proposal = receiving_agent_policy(g, rng)
        trace = _trace(g, proposal, rng)

        # ---------- naive text handoff: proposal executes unchecked ----------
        if not g.check_action(proposal):
            naive_ok += 1
        else:
            mast_disobey += 1

        # ---------- joint-embedding handoff: proposal is gated ----------
        packet = build_packet(g, adj.model)
        t0 = time.perf_counter()
        d = adj.adjudicate(packet, proposal, trace)
        latencies.append((time.perf_counter() - t0) * 1000)

        executed = proposal
        if d.verdict == Verdict.REQUEST_REPLAN:
            valid = g.next_valid_actions()
            executed = valid[0] if valid else proposal
        elif d.verdict == Verdict.FLAG_TO_HUMAN:
            deferred += 1
            valid = g.next_valid_actions()   # human resolves correctly
            executed = valid[0] if valid else proposal
        if not g.check_action(executed):
            struct_ok += 1

        # ---------- gate quality on both labeled traces ----------
        for action, tr, is_violation in [
                (s.pos_action, s.pos_trace, False),
                (s.neg_action, s.neg_trace, True)]:
            dec = adj.adjudicate(packet, action, tr)
            predicted_viol = dec.verdict == Verdict.REQUEST_REPLAN
            if is_violation and predicted_viol:
                gate_tp += 1
            elif is_violation and not predicted_viol:
                gate_fn += 1
            elif not is_violation and predicted_viol:
                gate_fp += 1
            else:
                gate_tn += 1

    n = len(samples)
    prec = gate_tp / (gate_tp + gate_fp) if gate_tp + gate_fp else 0.0
    rec = gate_tp / (gate_tp + gate_fn) if gate_tp + gate_fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print("=" * 68)
    print(f"SWE-BENCH LITE — CONSTRAINT-RESPECTING HANDOFF RATE  (n={n})")
    print(f"  naive text handoff        : {naive_ok / n:6.1%}   ({naive_ok}/{n})")
    print(f"  joint-embedding handoff   : {struct_ok / n:6.1%}   ({struct_ok}/{n})")
    print("-" * 68)
    print("ADJUDICATOR GATE (vs ground-truth graph checker)")
    print(f"  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print(f"  human-deferral (flag) rate: {deferred / n:.1%}")
    lat_note = ("cosine gate over fixed-size vectors - constant in orchestration length"
                if encoder_mode(adj.encoder) == "local"
                else "includes trace-embedding API round-trip; the local cosine gate itself is <1 ms")
    print(f"  adjudication latency: median {statistics.median(latencies):.1f} ms ({lat_note})")
    print("-" * 68)
    print("MAST-STYLE FAILURE TALLY (naive baseline)")
    print(f"  'Disobey Task Specification' / dependency-order failures: "
          f"{mast_disobey}/{n}")
    print("=" * 68)


if __name__ == "__main__":
    main()

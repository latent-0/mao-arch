# Joint-Embedding Handoff

Structural-semantic alignment for multi-agent orchestration. Built for the Google DeepMind Bangalore Hackathon, Problem Statement 2 (iAPI and Managed Agents) and the Gemma local-first track.

Multi-agent handoff today is a prose summary. That summary is a lossy projection of a structured object (the task-execution graph) onto a sentence, and the receiving agent has to re-infer structure from text with no fidelity guarantee. I replace prose handoff with a joint structural-semantic embedding: a relation-aware GNN over the live task graph, contrastively aligned with agent reasoning traces. Every proposed action is then checked by a local, on-device cosine-similarity adjudicator backed by Gemma 4. Full experimental setup and results are in [docs/experiments.md](docs/experiments.md).

## Quickstart

```
pip install -r requirements.txt
python -m mao.train            # about 70s on CPU: datagen, contrastive training, tau calibration
python -m mao.eval             # constraint-respecting handoff rate, naive vs joint-embedding
python -m mao.eval_swebench    # zero-shot on 300 real SWE-bench Lite issues (no training on them)
python -m demo.swe_scenario    # branch-before-QA terminal demo (add --fast to skip pauses)
python -m demo.web.server      # live split-screen web demo at http://127.0.0.1:8765
```

No API key or local model is required for the core pipeline: the language encoder falls back to a deterministic local featurizer, and the adjudicator's decision is plain vector math. With `GEMINI_API_KEY` set (in `.env`), trace embedding uses the Gemini embeddings API. With Gemma 4 E2B pulled in Ollama (`ollama pull gemma4:e2b`), the adjudicator's replan messages are written by local Gemma 4.

The web demo is fully live, with no simulated parts, and ships two scenarios selectable in the UI. In the SWE scenario, Agent A is a real Gemini 3.5 Flash call that fixes a real bug in a temporary git repository and the QA agent must run tests on a branch that may not exist yet. In the DB migration scenario, Agent A writes real SQLite migration SQL and the ops agent must snapshot, dry-run, apply, and verify a real customer database; every dependency there is physical (applying without a snapshot is refused for lack of a rollback point, verifying before applying hits a real "no such column" error). Agent B is a real Gemini call that decides its own next action from the handoff it receives. Tool calls execute real subprocesses, and their output streams into terminal panes in the page. Every proposal is gated by the trained adjudicator against the agent's own justification text, and Gemma 4 E2B writes the replan messages on device. The left pipeline hits a real git checkout error on the missing branch; the right pipeline gets gated, re-plans, and passes real tests. A second tab renders the task graph live (node statuses plus the gate log) as agents act. The "kill network" toggle swaps Agent B itself to local Gemma 4, so decisions, gate, and explanations all run on device.

## Results (held-out scenarios, n=300)

Constraint-respecting handoff rate for the naive text handoff baseline: 48.7% (154/300 MAST-style "Disobey Task Specification" failures).

| Encoder mode | Handoff rate | Gate P / R / F1 | Deferral rate | Adjudication latency |
|---|---|---|---|---|
| `gemini` (cloud embeddings) | 99.3% | 1.000 / 0.887 / 0.940 | 6.7% | 527 ms (includes embedding API round-trip) |
| `local` (fully offline) | 99.7% | 1.000 / 0.923 / 0.960 | 6.0% | 1.7 ms |

The cosine gate itself takes under 1 ms in both modes and its cost does not grow with orchestration length. In local mode the entire adjudication path, trace encoding included, runs on device with zero cloud round-trips. Gate precision is 1.0 by construction: a request-replan is only issued when the learned gate fires and a violated edge can be named from the graph snapshot (a structural witness). If the gate fires without a witness, the action is deferred to a human instead.

### Cross-workflow generalization (leave-one-template-out, n=200 per split)

Train on 4 workflow families, evaluate zero-shot on the 5th, which was never seen in training:

| Held-out workflow | local: handoff / recall | gemini: handoff / recall |
|---|---|---|
| SWE bug-fix | 100.0% / 0.960 | 98.0% / 0.855 |
| ETL / metrics report | 98.0% / 0.835 | 98.5% / 0.975 |
| Service deployment | 92.0% / 0.465 | 99.0% / 0.865 |
| Literature survey | 96.5% / 0.745 | 99.0% / 0.780 |
| DB schema migration | 61.5% / 0.205 | 81.0% / 0.445 |
| mean | 89.6% / 0.64 | 95.1% / 0.78 |

Precision stays at 1.000 in every split of both sweeps, so the witness-routing rule holds out of distribution. Three findings:

1. Under distribution shift the system escalates instead of failing silently. The human-deferral rate rises from about 6% in-distribution to 25-62% out-of-distribution, which is the designed behavior of the ambiguous band.
2. The lexical hashing encoder has no semantic transfer to unseen step vocabulary. Swapping in semantic (Gemini) trace embeddings recovers most of the gap: mean recall goes from 0.64 to 0.78, and the worst split improves from 61.5% to 81.0% handoff rate (recall 0.205 to 0.445). This was a hypothesis I tested, not an assumption.
3. The remaining gap on the hardest split has a specific cause: the GNN's node features are still hashed description text even in gemini mode, so unseen step names degrade the graph embedding itself. Semantic node features are the next step, and EmbeddingGemma would keep that fully local.

Both substrates receive the same receiving-agent proposal policy, so the difference under test is the handoff, not agent intelligence. Full protocols, hyperparameters, and the engineering findings behind these numbers are in [docs/experiments.md](docs/experiments.md).

### Real-benchmark evaluation: SWE-bench Lite (zero-shot, n=300)

The same adjudicator — trained *only* on the 5 synthetic templates — evaluated on the 300 real GitHub issues of [SWE-bench Lite](https://www.swebench.com/) across 12 open-source projects (django, sympy, scikit-learn, matplotlib, …), which it never saw in training. SWE-bench's own evaluation protocol defines the dependency order the gate protects: the `FAIL_TO_PASS` tests cannot pass until the code edit is applied. Each real instance is turned into a task graph from its real repo and gold-edited file (adapter: [mao/benchmarks/swebench.py](mao/benchmarks/swebench.py)); the real repo/file paths are out-of-distribution vocabulary.

| Substrate | Handoff rate | Gate P / R / F1 | Deferral |
|---|---|---|---|
| naive text handoff | 49.0% (147/300) | — | — |
| joint-embedding (local, offline) | **96.3%** (289/300) | 1.000 / 0.500 / 0.667 | 19.3% |

Trained on 5 synthetic templates, the gate nearly doubles the constraint-respecting handoff rate on 300 real, unseen issues (49.0% → 96.3%), and **precision stays 1.000 across every seed** — the witness-routing invariant is not a synthetic-data artifact. Recall drops to ~0.50 because the offline hashing encoder cannot transfer to real repo/file vocabulary: exactly the lexical ceiling the leave-one-template-out study above identifies, now confirmed on a real benchmark (semantic trace embeddings recover most of it in-family; semantic node features are the identified fix). Deferral rises from 6% in-distribution to ~19% here — the system escalates under distribution shift rather than silently approving. What is real vs. derived in this adaptation is stated plainly in [docs/experiments.md §6](docs/experiments.md).

## Layout

```
mao/graph.py               task execution graph (heterogeneous nodes/edges) + ground-truth checker
mao/encoders/structural.py relation-aware GNN (r-GAT) over the graph frontier, pure PyTorch
mao/encoders/language.py   trace encoder: Gemini embeddings backend + offline fallback
mao/joint.py               projection MLPs + cosine triplet loss (shared R^128)
mao/datagen.py             synthetic DAG task states + valid/violating traces (negatives mined free)
mao/train.py               training + tau* calibration (F1 sweep) -> artifacts/  (--holdout for LOTO)
mao/handoff.py             handoff packet {graph frontier, embedding} + naive prose baseline
mao/adjudicator.py         local gate: approve / flag-to-human / request-replan (+ Gemma 4 via Ollama)
mao/eval.py                constraint-respecting handoff rate + gate P/R/F1 + MAST tally
mao/eval_swebench.py       zero-shot eval on real SWE-bench Lite task graphs
mao/benchmarks/swebench.py SWE-bench Lite -> TaskGraph adapter (300 real issues, 12 repos)
demo/swe_scenario.py       branch-before-QA demo: naive fail -> gated replan -> offline beat
demo/web/                  live split-screen web demo (real agents, real repo, real tests)
```

## The demo scenario

The scenario reproduces a documented enterprise-agent failure: QA was delegated before the required branch existed. In the naive run, the prose summary omits that the branch does not exist, so the QA agent's tests fail. In the structured run, the `depends_on` edge is part of the handoff packet, the adjudicator rejects `run_tests` with the specific violated edge attached in a replan message written by Gemma 4 E2B, the agent re-plans, and the tests pass. Killing the network does not stop the gate, because it never needed the cloud.

## Acknowledgments

I would like to thank the Google DeepMind Hackathon and Cerebral Valley for hosting the event, and Google DeepMind for providing the cloud credits and models this project was built on: Gemini (embeddings and agent stack) and Gemma 4 (the on-device adjudicator, run as `gemma4:e2b` via Ollama).

Gemma is provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). This project builds on Google DeepMind's Gemma 4 open models. See the [Gemma 4 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/).

# Joint-Embedding Handoff

**Structural-semantic alignment for multi-agent orchestration** — Google DeepMind Bangalore Hackathon, Problem Statement 2 (iAPI & Managed Agents) + Gemma local-first track.

Multi-agent handoff today is a prose summary: a lossy projection of a structured object (the task-execution graph) onto a sentence. The receiving agent re-infers structure from text, with no fidelity guarantee. I replace prose handoff with a **joint structural-semantic embedding** — a relation-aware GNN over the live task graph, contrastively aligned with agent reasoning traces — and gate every proposed action through a **local, on-device cosine-similarity adjudicator** running against Gemma 4 (Gemma track). Full experimental setup and results: [docs/experiments.md](docs/experiments.md).

## Quickstart

```
pip install -r requirements.txt
python -m mao.train            # ~70s on CPU: datagen -> contrastive training -> tau calibration
python -m mao.eval             # constraint-respecting handoff rate, naive vs joint-embedding
python -m demo.swe_scenario    # the branch-before-QA terminal demo (add --fast to skip pauses)
python -m demo.web.server      # split-screen web demo at http://127.0.0.1:8765
```

The web demo runs both pipelines live on every click — real graph encoding, real cosine gate, real Gemma 4 replan messages — and animates them side by side: naive prose handoff failing on the left, the joint-embedding handoff getting gated, re-planned, and passing on the right. A "kill network" toggle demonstrates the fully on-device path. Self-contained page, no external assets — it survives offline.

No API key or local model is required to run it: the language encoder falls back to a deterministic local featurizer, and the adjudicator's *decision* is pure vector math. With `GEMINI_API_KEY` set (in `.env`), trace embedding uses the Gemini embeddings API; with **Gemma 4 E2B** pulled in Ollama (`ollama pull gemma4:e2b`), the adjudicator's replan messages are written by local Gemma 4.

## Results (held-out scenarios, n=300)

Constraint-respecting handoff rate — naive text handoff baseline: **48.7%** (154/300 MAST-style "Disobey Task Specification" failures).

| Encoder mode | Handoff rate | Gate P / R / F1 | Deferral rate | Adjudication latency |
|---|---|---|---|---|
| `gemini` (cloud embeddings) | **99.3%** | 1.000 / 0.887 / 0.940 | 6.7% | 527 ms (incl. embedding API round-trip) |
| `local` (fully offline) | **99.7%** | 1.000 / 0.923 / 0.960 | 6.0% | **1.7 ms** |

The cosine gate itself is <1 ms in both modes — constant in orchestration length; in local mode the entire adjudication path (trace encoding included) runs on-device with zero cloud round-trips. Gate precision is 1.0 by construction: a request-replan is only issued when the learned gate fires **and** a violated edge can be named from the graph snapshot (a structural witness); witness-less suspicion defers to a human instead of overruling the agent.

### Cross-workflow generalization (leave-one-template-out, n=200 per split)

Train on 4 workflow families, evaluate zero-shot on the 5th — never seen in training:

| Held-out workflow | local: handoff / recall | gemini: handoff / recall |
|---|---|---|
| SWE bug-fix | 100.0% / 0.960 | 98.0% / 0.855 |
| ETL / metrics report | 98.0% / 0.835 | 98.5% / 0.975 |
| Service deployment | 92.0% / 0.465 | 99.0% / 0.865 |
| Literature survey | 96.5% / 0.745 | 99.0% / 0.780 |
| DB schema migration | 61.5% / 0.205 | 81.0% / 0.445 |
| **mean** | **89.6% / 0.64** | **95.1% / 0.78** |

Precision stays 1.000 in every split of both sweeps (witness-routing holds out of distribution). Findings:

1. Under distribution shift the system **fails loud, not silent** — the human-deferral rate rises from ~6% in-distribution to 25–62% out-of-distribution: the designed behavior of the ambiguous band.
2. The lexical hashing encoder has no semantic transfer to unseen step vocabulary; swapping in **semantic (Gemini) trace embeddings recovers most of the gap** (mean recall 0.64 → 0.78; the worst split improves 61.5% → 81.0% handoff, recall 0.205 → 0.445) — a hypothesis I tested, not an assertion.
3. The residual gap on the hardest split has a precise cause: the GNN's **node features** are still hashed description text even in gemini mode, so unseen step names degrade the graph embedding itself. Semantic node features are the identified next step (and EmbeddingGemma would keep that fully local).

Both substrates receive the *same* receiving-agent proposal policy — the difference under test is the handoff, not agent intelligence. Full protocols, hyperparameters, and the engineering findings behind these numbers: [docs/experiments.md](docs/experiments.md).

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
demo/swe_scenario.py       branch-before-QA demo: naive fail -> gated replan -> offline beat
demo/web/                  split-screen live web demo (stdlib server + self-contained page)
```

## The demo in one breath

Documented enterprise-agent failure: *QA delegated before the required branch exists.* Naive run: the prose summary omits that the branch doesn't exist — tests fail. Structured run: the `depends_on` edge is in the handoff packet; the adjudicator rejects `run_tests` (alignment far below the approve band), routes back **with the specific violated edge attached** in a replan message written live by Gemma 4 E2B, the agent replans, tests pass. Then the network "dies" and the gate keeps working — it never needed the cloud.

## Acknowledgments

I would like to thank the **Google DeepMind Hackathon** and **Cerebral Valley** for hosting the event, and **Google DeepMind** for providing the cloud credits and models this project was built on — Gemini (embeddings and agent stack) and **Gemma 4** (the on-device adjudicator, run as `gemma4:e2b` via Ollama).

Gemma is provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). This project builds on Google DeepMind's Gemma 4 open models — see the [Gemma 4 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/).

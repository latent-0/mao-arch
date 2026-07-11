# Experiments

Complete documentation of the training data, model configuration, evaluation protocol, and every experiment I ran, in the order I ran them — including the failures and what they changed. All numbers are reproducible with the commands shown (fixed seeds throughout).

## 1. Training data — fully synthetic, no external benchmark

There is no public benchmark that isolates information loss during a single agent-to-agent handoff (MAST and GEMMAS are diagnostic frameworks, not scored datasets), so I generate all data synthetically and state that plainly.

### 1.1 Workflow templates

Five workflow families, each a real DAG of 5 steps with explicit dependency edges (branches and joins, not just chains) plus resource-mutex constraints ([mao/datagen.py](../mao/datagen.py)):

| # | Family | Structure |
|---|---|---|
| 0 | SWE bug-fix | two roots (`create_branch`, `edit_files`) joining at `run_tests`; worktree mutex |
| 1 | ETL / metrics report | linear chain; reporting-DB mutex |
| 2 | Service deployment | two roots joining at `deploy_service`; prod-cluster mutex |
| 3 | Literature survey | two roots joining at `screen_sources`; no mutex |
| 4 | DB schema migration | two roots joining at `dry_run`; live-DB mutex |

### 1.2 Sample construction

Each sample is a triplet (graph state, positive trace, negative trace):

1. **State**: sample a random **dependency-closed** done-set (a step can be done only if all its dependencies are done; inclusion probability 0.5 per eligible step, never the full set). Done steps get a produced-artifact node — the same convention the live demo follows when a step executes.
2. **Positive**: a pending step whose dependencies are all satisfied; a reasoning trace for it is generated from templated phrasing pools (opener + goal + completed-steps + intent + justification).
3. **Negative**: a pending step with unmet dependencies, or (25% of the time, when available) a resource-mutex conflict against an ACTIVE holder. Its trace is drawn from **the same phrasing pools** — deliberately no lexical tell; valid and violating traces differ only in which action they propose relative to the graph state.

The graph's ground-truth checker (`TaskGraph.check_action`) certifies every label; both assertions are enforced at generation time. This is the "negatives are mined for free" property: permute the order and the label is automatic — zero human annotation.

### 1.3 Splits

| Split | n | Seed | Note |
|---|---|---|---|
| Train | 2,000 | 0 | |
| Validation | 400 | 10,000 | used for triplet accuracy + τ calibration |
| Eval | 300 | 99 (+100 for the receiver policy) | disjoint seeds; same 5 templates |

**Honest scoping:** the main eval tests held-out *states and phrasings* within known workflow families. Cross-family generalization is tested separately (§5).

## 2. Model and training configuration

- **Structural encoder**: 2-layer relation-aware GAT (r-GAT formulation, arXiv:2109.05922) in pure PyTorch; per-relation weight matrices and attention vectors over 12 relation ids (6 edge types × 2 directions) + a self-loop transform; attention-weighted pooling over live (non-done) subtasks. Node features: 64-d hashed description text + node-type one-hot (4) + status one-hot (4). Hidden 128, output 128.
- **Language encoder** (frozen): either `gemini-embedding-001` at 512-d (cloud mode) or a deterministic unigram+bigram feature-hashing encoder at 512-d (local mode). Both L2-normalized. **Encoders are never mixed across a trained space.**
- **Joint space**: two MLP projectors (512→256→128 and 128→256→128), outputs L2-normalized.
- **Loss**: cosine triplet, `max(0, cos(g, t_neg) − cos(g, t_pos) + 0.3)`.
- **Optimization**: Adam, lr 1e-3, batch 32, 8 epochs, seed 0. ~70 s (local) / ~5 min (gemini, API-bound) on CPU.
- **Critical detail**: training embeds the graph **frontier** (non-done subtasks + 1-hop neighborhood + constraints), exactly what handoff packets carry at inference. See §6.2 for the bug that motivated this.

### Threshold calibration

After training I sweep every candidate threshold over validation alignment scores and pick **τ\*** maximizing F1 for violation detection (score < τ ⇒ violating). An ambiguous band around τ\* (half-width = 12.5% of the mean positive–negative score gap) defines the flag-to-human zone: approve above the band, request-replan below it, defer inside it.

## 3. Evaluation protocol

**Metric — constraint-respecting handoff rate**: the % of handoffs after which the receiving agent's first *executed* action violates no `depends_on`/constraint edge in the ground-truth graph.

**Receiver policy (identical under both substrates)**: a deliberately charitable model of prose re-inference — 40% it recovers the correct next step, 40% it follows the summary's goal-emphasizing imperative (the terminal step: the documented dependency-order delegation failure), 20% another remaining step. The naive substrate executes the proposal unchecked; the joint-embedding substrate gates it (replan → re-plan from the graph frontier; flag → human resolves correctly, counted in the deferral rate).

**Gate metrics** are computed separately against the ground-truth checker on both the labeled valid and violating trace of every scenario (600 adjudications per 300-scenario run).

**Witness-routing (adjudicator design, not metric bookkeeping)**: a request-replan is issued only when the learned gate fires *and* the graph snapshot names a violated edge. A gate firing without a structural witness downgrades to flag-to-human. Consequence: replan precision is 1.0 **by construction**; the learned component's quality shows in **recall** and the **deferral rate**, which I report unpolished.

## 4. In-distribution results (n=300, seed 99)

```
python -m mao.eval --n 300 --encoder local   # or gemini
```

Naive baseline: **48.7%** (154/300 dependency-order failures, MAST "Disobey Task Specification").

| Mode | Handoff rate | Precision | Recall | F1 | Deferral | Latency (median) |
|---|---|---|---|---|---|---|
| gemini | 99.3% | 1.000 | 0.887 | 0.940 | 6.7% | 527 ms (incl. embedding API call) |
| local | 99.7% | 1.000 | 0.923 | 0.960 | 6.0% | 1.7 ms |

## 5. Cross-workflow generalization (leave-one-template-out)

Train on 4 families (2,000/400 samples regenerated from the reduced pool), evaluate 200 scenarios drawn **only** from the excluded family:

```
python -m mao.train --encoder local --holdout 2
python -m mao.eval  --n 200 --encoder local --artifact-mode local_holdout2 --templates 2
```

| Held-out | local: handoff / P / R / defer | gemini: handoff / P / R / defer |
|---|---|---|
| 0 SWE bug-fix | 100.0% / 1.000 / 0.960 / 46.5% | 98.0% / 1.000 / 0.855 / 42.5% |
| 1 ETL | 98.0% / 1.000 / 0.835 / 62.0% | 98.5% / 1.000 / 0.975 / 56.0% |
| 2 Deployment | 92.0% / 1.000 / 0.465 / 31.5% | 99.0% / 1.000 / 0.865 / 25.0% |
| 3 Survey | 96.5% / 1.000 / 0.745 / 44.5% | 99.0% / 1.000 / 0.780 / 52.5% |
| 4 DB migration | 61.5% / 1.000 / 0.205 / 41.5% | 81.0% / 1.000 / 0.445 / 41.0% |
| **mean** | **89.6% / — / 0.64 / 45.2%** | **95.1% / — / 0.78 / 43.4%** |

Interpretation, in the order the evidence arrived:

1. **Fails loud, not silent.** Deferral rises from ~6% in-distribution to 25–62% out-of-distribution — the ambiguous band absorbing uncertainty instead of silently approving. This is the strongest evidence for the "clear boundaries for deferring to a human" requirement.
2. **Lexical → semantic hypothesis, tested.** The local hashing encoder cannot transfer to unseen step vocabulary, so its cross-family recall varies 0.96 → 0.21. Rerunning the identical sweep with Gemini embeddings recovered most of the gap (mean recall 0.64 → 0.78; worst split 61.5% → 81.0% handoff).
3. **Residual gap, diagnosed.** Even in gemini mode the GNN's *node features* are hashed text, so unseen step names still degrade the graph embedding. Semantic node features (EmbeddingGemma to keep it local) are the identified next step.
4. **The architectural invariant held everywhere:** precision 1.000 in all 10 splits.

## 6. Engineering findings (failures that changed the system)

These three failures occurred during development, were diagnosed from behavior, and each produced a design change. I consider them results.

### 6.1 Prefix-chain memorization

The first model reached 1.0 validation triplet accuracy and then failed on the demo's branching graph — it had memorized "done steps form a prefix of a chain" (the only pattern in v1 training data) instead of the rule "all of this action's dependencies are done." **Fix:** templates became real DAGs with branches/joins, and done-sets became randomly sampled dependency-closed subsets. The lesson: validation accuracy within a narrow distribution says nothing about having learned the right invariant.

### 6.2 Train/inference representation mismatches

Two subtler shifts surfaced through demo behavior (alignments decaying as steps executed): (a) training graphs attach an artifact node to every done step, but executed demo steps didn't — mid-execution states looked structurally alien; (b) training embedded full graphs while handoff packets embed the frontier. **Fix:** executed steps now produce artifacts (matching the datagen convention), and training embeds the frontier. Both are now conventions of the protocol, not incidental choices.

### 6.3 Witness-routing

The gate occasionally rejected valid actions near the threshold. Rather than tune thresholds to the demo, I changed the protocol: rejection requires a *named structural witness* from the graph snapshot; witness-less suspicion defers to a human. This moved all false replans into deferrals (precision → 1.0 architecturally) and gave the human-deferral boundary a precise definition: *quantitatively ambiguous alignment, or a rejection the adjudicator cannot justify structurally.*

## 7. Known limitations

- Synthetic data end-to-end: 5 workflow families, templated trace phrasing. LLM-generated (Gemini-paraphrased) traces are the natural extension; the pipeline supports it.
- The naive baseline receiver is a parameterized policy (charitable at 40% correct), not a measured LLM receiver; the demo failure pattern, however, mirrors a verbatim documented enterprise-agent trace.
- The Agent A/B loop in the demo is simulated; wiring the Managed Agents API (`antigravity-preview-05-2026`) is the intended cloud-side completion.
- A trace that misdescribes the intended action can mislead the gate; the graph snapshot bounds the damage (execution still hits the ground-truth checker in this implementation).

## 8. Compute and models

| Component | Model / hardware |
|---|---|
| Trace embeddings (cloud mode) | `gemini-embedding-001`, 512-d, via Gemini API |
| Adjudicator explanations | **Gemma 4 E2B** (`gemma4:e2b`), fully local via Ollama |
| Training / inference | CPU (Windows laptop), PyTorch 2.12 — no GPU required |

Built at the Google DeepMind Bangalore Hackathon with cloud credits and model access provided by Google DeepMind; event hosted with Cerebral Valley. Gemma is provided under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

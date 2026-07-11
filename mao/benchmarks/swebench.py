"""SWE-bench Lite -> TaskGraph adapter.

SWE-bench Lite (Jimenez et al., 2024) is 300 real GitHub issues from 12
open-source Python projects (django, sympy, scikit-learn, matplotlib, ...).
Each instance ships a *gold* patch and a *test* patch; the official evaluation
harness resolves an instance by (1) applying the code patch, (2) applying the
test patch, then (3) running the FAIL_TO_PASS tests and expecting them to pass.
That protocol defines a **real dependency order**: the failing tests cannot pass
until the code edit is applied. This is exactly the constraint the handoff gate
is built to protect, so SWE-bench Lite is a natural real-world test bed.

What is real here (loaded from `data/swebench_lite_instances.json`, the full
Lite instance set with the reference-solution file for each):
  * the instance set (all 300), the 12 repositories, the PR numbers,
  * the source file the accepted fix edits (the gold-patch file), and
  * the execution precedence (edit-before-test) taken from the harness protocol.

What is *derived* (and stated plainly, matching the synthetic-trace disclosure
in docs/experiments.md §7): the decomposition into the canonical SWE-agent
workflow steps (reproduce -> localize -> edit -> test -> submit), and the
reasoning traces, which are templated over the real repo/file/test tokens.
Per-test names and the issue prose live only in the HuggingFace dataset and are
not used; the test step is represented at suite granularity.

The point is a distribution shift, not a strawman: the repo names and source
paths (e.g. `astropy/modeling/separable.py`) are vocabulary the model never saw
during training on the 5 synthetic templates, which is precisely the lexical
generalization the leave-one-template-out study (§5) probes.
"""

from __future__ import annotations

import json
import os
import random

from ..datagen import Sample, _trace, _sample_done_set, build_graph
from ..graph import Status

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "swebench_lite_instances.json")


def load_instances(path: str = DATA_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def instance_template(inst: dict) -> dict:
    """Build a template dict (goal, steps, resources) for one real instance.

    `steps` are (id, description, [dependency ids]) in topological order — the
    same shape as mao.datagen.TEMPLATES, so the synthetic data machinery
    (done-set sampling, graph building, trace phrasing) is reused verbatim.
    Descriptions carry the real repo and file tokens, so the graph the model
    sees is lexically out-of-distribution.
    """
    repo = inst["repo"]
    files = inst.get("gold_files") or [f"{repo.split('/')[-1]}/core.py"]
    iid = inst["instance_id"]

    goal = (f"Resolve {iid}: fix {repo} so the failing tests pass "
            f"(edits to {', '.join(files)})")

    steps: list[tuple[str, str, list[str]]] = [
        ("reproduce", f"reproduce the reported failure in {repo}", []),
        ("localize", f"localize the fault to {files[0]}", ["reproduce"]),
    ]
    edit_ids: list[str] = []
    for i, fp in enumerate(files):
        eid = f"edit_{i}"
        edit_ids.append(eid)
        steps.append((eid, f"edit {fp} to implement the fix", ["localize"]))
    steps.append(("apply_test_patch",
                  f"apply the test patch adding the fail-to-pass tests for {repo}", []))
    # the real SWE-bench precedence: failing tests run only after the edit and
    # the test patch are both in place
    steps.append(("run_fail_to_pass",
                  f"run the fail-to-pass tests for {repo}",
                  edit_ids + ["apply_test_patch"]))
    steps.append(("run_pass_to_pass",
                  f"run the pass-to-pass regression tests for {repo}",
                  list(edit_ids)))
    steps.append(("submit",
                  f"open a pull request with the verified fix for {iid}",
                  ["run_fail_to_pass", "run_pass_to_pass"]))

    # the working tree is a shared exclusive resource across localize + edits,
    # mirroring the repo_worktree mutex in the synthetic SWE template
    resources = [("repo_worktree", ["localize"] + edit_ids)]
    return {"goal": goal, "steps": steps, "resources": resources}


def make_instance_sample(inst: dict, rng: random.Random,
                         max_tries: int = 60) -> Sample | None:
    """One contrastive sample (state, valid action, violating action) drawn from
    a real instance. Mirrors mao.datagen.make_sample but over a fixed instance
    template. Returns None if no state with both a valid and a violating action
    could be sampled (rare; handled by the caller)."""
    tmpl = instance_template(inst)
    steps = tmpl["steps"]
    resources = tmpl.get("resources", [])

    for _ in range(max_tries):
        done = _sample_done_set(steps, rng)
        g = build_graph(tmpl, done, rng)

        valid = g.next_valid_actions()
        violating = [sid for sid, _d, _deps in steps
                     if g.nodes[sid].status == Status.PENDING and g.check_action(sid)]

        # occasionally use a resource-mutex conflict as the negative (same 25%
        # convention as the synthetic generator)
        if resources and rng.random() < 0.25:
            res, holders = rng.choice(resources)
            pending_holders = [h for h in holders
                               if g.nodes[h].status == Status.PENDING]
            if len(pending_holders) >= 2:
                active, blocked = pending_holders[:2]
                g.set_status(active, Status.ACTIVE)
                valid = g.next_valid_actions()
                violating = [blocked]

        if valid and violating:
            pos_action = rng.choice(valid)
            neg_action = rng.choice(violating)
            assert not g.check_action(pos_action), "positive must be violation-free"
            assert g.check_action(neg_action), "negative must violate"
            return Sample(graph=g, pos_action=pos_action,
                          pos_trace=_trace(g, pos_action, rng),
                          neg_action=neg_action,
                          neg_trace=_trace(g, neg_action, rng))
    return None


def generate(n: int | None = None, seed: int = 99, repeats: int = 1,
             path: str = DATA_PATH) -> list[Sample]:
    """Contrastive samples over real SWE-bench Lite instances.

    n       : cap on the number of instances used (None = all 300).
    repeats : independent states sampled per instance (each a fresh done-set).
    The returned list is the eval set; its length is reported as the eval n.
    """
    instances = load_instances(path)
    if n is not None:
        instances = instances[:n]
    rng = random.Random(seed)
    samples: list[Sample] = []
    for _ in range(repeats):
        for inst in instances:
            s = make_instance_sample(inst, rng)
            if s is not None:
                samples.append(s)
    return samples

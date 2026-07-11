"""Synthetic contrastive training data for the joint space.

Each sample is (task graph state, positive trace, negative trace):
  * positive — reasoning trace for an action that respects every depends_on /
    Constraint edge given current statuses
  * negative — a *plausible-sounding* trace for an action whose dependencies
    are unmet or which conflicts with an active subtask. Negatives are mined
    by permuting step order / skipping precedence — no labeled data needed.

When a GEMINI_API_KEY is present, traces can additionally be paraphrased by
Gemini for richer surface variation (datagen stays fully functional offline).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .graph import TaskGraph, Status, ConstraintType

# ---------------------------------------------------------------------------
# Pipeline templates: steps are (id, description, [dependency ids]) — real DAGs
# with branches and joins, not just chains, so the model must learn the actual
# rule ("all of this action's dependencies are done") rather than memorizing
# prefix-done chain patterns. Optional resource groups share an exclusive lock.
# ---------------------------------------------------------------------------

TEMPLATES: list[dict] = [
    {
        "goal": "Fix bug #{n}: apply the patch and run the test suite on a new feature branch",
        "steps": [
            ("create_branch", "create a new feature branch for the fix", []),
            ("checkout_branch", "check out the feature branch locally", ["create_branch"]),
            ("edit_files", "apply the code patch to the affected files", []),
            ("run_tests", "run the full test suite on the feature branch",
             ["create_branch", "edit_files"]),
            ("open_pr", "open a pull request with the verified fix", ["run_tests"]),
        ],
        "resources": [("repo_worktree", ["checkout_branch", "edit_files"])],
    },
    {
        "goal": "Build the weekly metrics report from raw event data",
        "steps": [
            ("fetch_events", "fetch raw event data from the warehouse", []),
            ("validate_schema", "validate the event schema and drop malformed rows",
             ["fetch_events"]),
            ("transform", "aggregate events into weekly metrics tables",
             ["validate_schema"]),
            ("load_tables", "load the metrics tables into the reporting database",
             ["transform"]),
            ("render_report", "render the weekly report from the metrics tables",
             ["load_tables"]),
        ],
        "resources": [("reporting_db", ["load_tables", "render_report"])],
    },
    {
        "goal": "Deploy service v{n} to the production cluster",
        "steps": [
            ("build_image", "build the container image from the release tag", []),
            ("push_image", "push the image to the container registry", ["build_image"]),
            ("provision_config", "provision config and secrets for the new version", []),
            ("deploy_service", "roll out the new version to the production cluster",
             ["push_image", "provision_config"]),
            ("smoke_test", "run smoke tests against the deployed service",
             ["deploy_service"]),
        ],
        "resources": [("prod_cluster", ["deploy_service", "smoke_test"])],
    },
    {
        "goal": "Produce a literature survey on topic #{n}",
        "steps": [
            ("collect_sources", "collect candidate papers and sources", []),
            ("define_criteria", "define inclusion and quality criteria", []),
            ("screen_sources", "screen sources for relevance and quality",
             ["collect_sources", "define_criteria"]),
            ("extract_findings", "extract key findings from screened sources",
             ["screen_sources"]),
            ("write_survey", "synthesize findings and write the survey",
             ["extract_findings"]),
        ],
        "resources": [],
    },
    {
        "goal": "Migrate customer database to the new schema",
        "steps": [
            ("snapshot_db", "take a consistent snapshot of the customer database", []),
            ("write_migration", "write the schema migration scripts", []),
            ("dry_run", "dry-run the migration against the snapshot",
             ["snapshot_db", "write_migration"]),
            ("apply_migration", "apply the migration to the live database", ["dry_run"]),
            ("verify_integrity", "verify row counts and referential integrity",
             ["apply_migration"]),
        ],
        "resources": [("live_db", ["apply_migration", "verify_integrity"])],
    },
]

# Trace phrasing variation --------------------------------------------------

_OPENERS = [
    "Handoff received.", "Picking up the task.", "Continuing the plan.",
    "Taking over from the previous agent.", "Resuming orchestration.",
]
_DONE_PHRASES = [
    "Completed so far: {done}.", "The steps {done} are already finished.",
    "Prior agent finished {done}.", "{done} are done.",
]
_NONE_DONE = ["Nothing has been executed yet.", "No steps are complete yet.",
              "This is the first action of the plan."]
_INTENT = [
    "I will now {desc} ({step}).",
    "Next action: {step} — {desc}.",
    "Proceeding to {step}: {desc}.",
    "The right next step is to {desc}, so I am executing {step}.",
]
# One shared pool for valid AND violating traces — no lexical giveaway.
# The model must separate them from (graph state, action) alignment alone.
_JUSTIFY = [
    "This is the right next step for the plan.",
    "This moves us directly toward the goal.",
    "Proceeding with this step now.",
    "This is what the plan calls for at this point.",
    "Executing this step to keep the pipeline moving.",
]


@dataclass
class Sample:
    graph: TaskGraph
    pos_action: str
    pos_trace: str
    neg_action: str
    neg_trace: str


def _trace(g: TaskGraph, action: str, rng: random.Random) -> str:
    done = [n.id for n in g.subtasks() if n.status == Status.DONE]
    desc = g.nodes[action].description
    parts = [rng.choice(_OPENERS), f"Goal: {g.goal}."]
    if done:
        parts.append(rng.choice(_DONE_PHRASES).format(done=", ".join(done)))
    else:
        parts.append(rng.choice(_NONE_DONE))
    parts.append(rng.choice(_INTENT).format(step=action, desc=desc))
    parts.append(rng.choice(_JUSTIFY))
    return " ".join(parts)


def _sample_done_set(steps: list, rng: random.Random) -> set[str]:
    """A random dependency-closed done-set: a step can only be done if all of
    its dependencies are done. Never the full set (something must remain)."""
    done: set[str] = set()
    for sid, _desc, deps in steps:  # template order is topological
        if all(d in done for d in deps) and rng.random() < 0.5:
            done.add(sid)
    if len(done) == len(steps):
        done.discard(steps[-1][0])
    return done


def build_graph(template: dict, done: set[str], rng: random.Random) -> TaskGraph:
    g = TaskGraph(goal=template["goal"].format(n=rng.randint(100, 999)))
    steps = template["steps"]
    for sid, desc, _deps in steps:
        status = Status.DONE if sid in done else Status.PENDING
        g.add_subtask(sid, desc, status=status,
                      owner_agent=rng.choice(["agent_a", "agent_b"]))
    for sid, _desc, deps in steps:
        for d in deps:
            g.depends(sid, d)
    for res, holders in template.get("resources", []):
        g.add_constraint(f"mutex_{res}", ConstraintType.RESOURCE,
                         {"resource": res, "holders": holders},
                         f"only one subtask may hold '{res}' at a time")
    # artifacts produced by done steps make state concrete
    for sid in done:
        g.add_artifact(f"out_{sid}", f"output artifact of {sid}",
                       artifact_type="data", produced_by=sid)
    return g


def make_sample(rng: random.Random, pool: list[dict] | None = None) -> Sample:
    pool = pool if pool is not None else TEMPLATES
    while True:
        template = rng.choice(pool)
        steps = template["steps"]
        done = _sample_done_set(steps, rng)
        g = build_graph(template, done, rng)

        valid = g.next_valid_actions()
        violating = [sid for sid, _d, _deps in steps
                     if g.nodes[sid].status == Status.PENDING
                     and g.check_action(sid)]

        # occasionally use a resource-conflict negative instead of order violation
        resources = template.get("resources", [])
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
            break  # resample when this state has no contrastive pair

    assert not g.check_action(pos_action), "positive must be violation-free"
    assert g.check_action(neg_action), "negative must violate"

    return Sample(
        graph=g,
        pos_action=pos_action,
        pos_trace=_trace(g, pos_action, rng),
        neg_action=neg_action,
        neg_trace=_trace(g, neg_action, rng),
    )


def generate(n: int, seed: int = 0,
             template_ids: list[int] | None = None) -> list[Sample]:
    """template_ids restricts generation to a subset of TEMPLATES (by index) —
    used for leave-one-template-out generalization experiments."""
    rng = random.Random(seed)
    pool = [TEMPLATES[i] for i in template_ids] if template_ids is not None else None
    return [make_sample(rng, pool) for _ in range(n)]

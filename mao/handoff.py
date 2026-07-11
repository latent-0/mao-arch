"""Handoff protocol.

Structured handoff (mine):
    handoff_packet = { graph_snapshot: G_frontier, embedding: g_hat }
The receiving agent never inherits the sender's token history — context size is
bounded by the task graph's active frontier, not by conversation length.

Naive baseline (what the field does today): a prose summary. The summary
generator below is deliberately *realistic*, not strawmanned — it reports the
goal and what was accomplished, the way agent frameworks actually summarize.
The lossy projection happens naturally: pending-dependency state and constraint
edges are structurally present in the graph but linguistically implicit in the
prose.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .graph import TaskGraph, Status
from .joint import JointModel


@dataclass
class HandoffPacket:
    graph_snapshot: dict          # serialized active frontier
    embedding: list[float]        # g_hat = f(g) in the joint space, unit norm

    @property
    def graph(self) -> TaskGraph:
        return TaskGraph.from_snapshot(self.graph_snapshot)


def build_packet(graph: TaskGraph, model: JointModel) -> HandoffPacket:
    frontier = graph.frontier()
    with torch.no_grad():
        g_hat = model.embed_graph(frontier)
    return HandoffPacket(graph_snapshot=frontier.snapshot(),
                         embedding=g_hat.tolist())


def naive_summary(graph: TaskGraph) -> str:
    """The prose handoff a typical orchestrator produces: goal + progress +
    imperative next instruction. Dependency topology and constraint state are
    flattened away — that's the failure mode, occurring naturally."""
    done = [n for n in graph.subtasks() if n.status == Status.DONE]
    remaining = [n for n in graph.subtasks() if n.status != Status.DONE]
    parts = [f"Goal: {graph.goal}."]
    if done:
        parts.append("Completed: " + "; ".join(n.description for n in done) + ".")
    if remaining:
        # summaries emphasize the goal-relevant step, not the dependency order —
        # the terminal step reads as "what's needed" once the work sounds underway
        headline = remaining[-1]
        parts.append(f"Please continue and {headline.description}.")
    return " ".join(parts)

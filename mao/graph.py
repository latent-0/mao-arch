"""Task Execution Graph — the structural ground truth for multi-agent orchestration.

Heterogeneous graph G = (V, E, R):
  Node types: Subtask, ToolCall, Artifact, Constraint
  Edge types: depends_on, produces, consumes, owned_by, conflicts_with, violates

The graph is a live object: every agent action reads from and writes to it.
This module also contains the *ground-truth* violation checker used for
supervision (datagen), evaluation, and for attaching the specific violated
constraint to a request-replan decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum


class NodeType(str, Enum):
    SUBTASK = "subtask"
    TOOL_CALL = "tool_call"
    ARTIFACT = "artifact"
    CONSTRAINT = "constraint"


class Status(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"


class EdgeType(str, Enum):
    DEPENDS_ON = "depends_on"          # Subtask -> Subtask (precedence)
    PRODUCES = "produces"              # ToolCall/Subtask -> Artifact
    CONSUMES = "consumes"              # Subtask -> Artifact
    OWNED_BY = "owned_by"              # Subtask -> Agent (agent stored as node-less attr)
    CONFLICTS_WITH = "conflicts_with"  # Subtask -> Subtask (mutual exclusion)
    VIOLATES = "violates"              # proposed action -> Constraint (inference time)


class ConstraintType(str, Enum):
    PRECEDENCE = "precedence"  # expression: {"before": id, "after": id}
    RESOURCE = "resource"      # expression: {"resource": name, "holders": [ids]} (mutex)
    EXCLUSION = "exclusion"    # expression: {"a": id, "b": id} (never both active/done)


@dataclass
class Node:
    id: str
    type: NodeType
    description: str = ""
    status: Status = Status.PENDING
    owner_agent: str | None = None
    # Constraint-specific
    constraint_type: ConstraintType | None = None
    expression: dict = field(default_factory=dict)
    # Artifact-specific
    artifact_type: str | None = None


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    rel: EdgeType


@dataclass
class Violation:
    """A ground-truth constraint violation for a proposed action."""
    action_id: str
    kind: str                  # "unmet_dependency" | "resource_conflict" | "exclusion"
    detail: str                # human-readable, names the specific edge/constraint
    edge: tuple[str, str, str] | None = None  # (src, dst, rel) of the violated edge


class TaskGraph:
    def __init__(self, goal: str = ""):
        self.goal = goal
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    # ---------------- construction ----------------

    def add_subtask(self, id: str, description: str, *, status: Status = Status.PENDING,
                    owner_agent: str | None = None) -> Node:
        node = Node(id=id, type=NodeType.SUBTASK, description=description,
                    status=status, owner_agent=owner_agent)
        self.nodes[id] = node
        return node

    def add_artifact(self, id: str, description: str, artifact_type: str = "data",
                     produced_by: str | None = None) -> Node:
        node = Node(id=id, type=NodeType.ARTIFACT, description=description,
                    artifact_type=artifact_type)
        self.nodes[id] = node
        if produced_by:
            self.add_edge(produced_by, id, EdgeType.PRODUCES)
        return node

    def add_constraint(self, id: str, ctype: ConstraintType, expression: dict,
                       description: str = "") -> Node:
        node = Node(id=id, type=NodeType.CONSTRAINT, description=description,
                    constraint_type=ctype, expression=expression)
        self.nodes[id] = node
        return node

    def add_edge(self, src: str, dst: str, rel: EdgeType) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError(f"edge endpoints must exist: {src} -> {dst}")
        edge = Edge(src, dst, rel)
        if edge not in self.edges:
            self.edges.append(edge)

    def depends(self, later: str, earlier: str) -> None:
        """`later` depends_on `earlier` (earlier must be DONE first)."""
        self.add_edge(later, earlier, EdgeType.DEPENDS_ON)

    # ---------------- state updates ----------------

    def set_status(self, id: str, status: Status) -> None:
        self.nodes[id].status = status

    # ---------------- queries ----------------

    def subtasks(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.type == NodeType.SUBTASK]

    def constraints(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.type == NodeType.CONSTRAINT]

    def dependencies_of(self, id: str) -> list[str]:
        return [e.dst for e in self.edges if e.src == id and e.rel == EdgeType.DEPENDS_ON]

    def unmet_dependencies(self, id: str) -> list[str]:
        return [d for d in self.dependencies_of(id)
                if self.nodes[d].status != Status.DONE]

    def conflicts_of(self, id: str) -> list[str]:
        out = [e.dst for e in self.edges if e.src == id and e.rel == EdgeType.CONFLICTS_WITH]
        out += [e.src for e in self.edges if e.dst == id and e.rel == EdgeType.CONFLICTS_WITH]
        return out

    def next_valid_actions(self) -> list[str]:
        """Subtasks that are pending and have every dependency satisfied."""
        return [n.id for n in self.subtasks()
                if n.status == Status.PENDING and not self.unmet_dependencies(n.id)
                and not self.check_action(n.id)]

    # ---------------- ground-truth violation checker ----------------

    def check_action(self, action_id: str) -> list[Violation]:
        """Would executing `action_id` right now violate any structural constraint?"""
        violations: list[Violation] = []
        node = self.nodes.get(action_id)
        if node is None:
            return [Violation(action_id, "unknown_action", f"no such subtask: {action_id}")]

        for dep in self.unmet_dependencies(action_id):
            violations.append(Violation(
                action_id, "unmet_dependency",
                f"'{action_id}' depends_on '{dep}' but '{dep}' is "
                f"{self.nodes[dep].status.value}, not done",
                edge=(action_id, dep, EdgeType.DEPENDS_ON.value)))

        for other in self.conflicts_of(action_id):
            if self.nodes[other].status == Status.ACTIVE:
                violations.append(Violation(
                    action_id, "resource_conflict",
                    f"'{action_id}' conflicts_with '{other}' which is currently active",
                    edge=(action_id, other, EdgeType.CONFLICTS_WITH.value)))

        for c in self.constraints():
            expr = c.expression
            if c.constraint_type == ConstraintType.PRECEDENCE:
                if expr.get("after") == action_id:
                    before = expr.get("before")
                    if before in self.nodes and self.nodes[before].status != Status.DONE:
                        violations.append(Violation(
                            action_id, "unmet_dependency",
                            f"constraint '{c.id}': '{before}' must be done before "
                            f"'{action_id}' (currently {self.nodes[before].status.value})",
                            edge=(action_id, c.id, EdgeType.VIOLATES.value)))
            elif c.constraint_type == ConstraintType.RESOURCE:
                holders = expr.get("holders", [])
                if action_id in holders:
                    active = [h for h in holders if h != action_id
                              and h in self.nodes and self.nodes[h].status == Status.ACTIVE]
                    if active:
                        violations.append(Violation(
                            action_id, "resource_conflict",
                            f"constraint '{c.id}': resource '{expr.get('resource')}' is "
                            f"held by active subtask '{active[0]}'",
                            edge=(action_id, c.id, EdgeType.VIOLATES.value)))
            elif c.constraint_type == ConstraintType.EXCLUSION:
                a, b = expr.get("a"), expr.get("b")
                other = b if action_id == a else a if action_id == b else None
                if other and other in self.nodes and \
                        self.nodes[other].status in (Status.ACTIVE, Status.DONE):
                    violations.append(Violation(
                        action_id, "exclusion",
                        f"constraint '{c.id}': '{action_id}' and '{other}' are mutually "
                        f"exclusive and '{other}' is {self.nodes[other].status.value}",
                        edge=(action_id, c.id, EdgeType.VIOLATES.value)))
        return violations

    # ---------------- frontier & serialization ----------------

    def frontier(self) -> "TaskGraph":
        """The active frontier: non-done subtasks, their 1-hop structural
        neighborhood, and all constraints. This bounds handoff context size by
        the live task surface, not by conversation length."""
        keep: set[str] = set()
        for n in self.subtasks():
            if n.status != Status.DONE:
                keep.add(n.id)
        # 1-hop neighborhood of kept nodes (incl. done deps so status is visible)
        for e in self.edges:
            if e.src in keep:
                keep.add(e.dst)
            elif e.dst in keep:
                keep.add(e.src)
        for c in self.constraints():
            keep.add(c.id)

        sub = TaskGraph(goal=self.goal)
        for nid in keep:
            sub.nodes[nid] = self.nodes[nid]
        sub.edges = [e for e in self.edges if e.src in keep and e.dst in keep]
        return sub

    def snapshot(self) -> dict:
        return {
            "goal": self.goal,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
        }

    @classmethod
    def from_snapshot(cls, snap: dict) -> "TaskGraph":
        g = cls(goal=snap.get("goal", ""))
        for nd in snap["nodes"]:
            node = Node(
                id=nd["id"], type=NodeType(nd["type"]), description=nd["description"],
                status=Status(nd["status"]), owner_agent=nd.get("owner_agent"),
                constraint_type=ConstraintType(nd["constraint_type"]) if nd.get("constraint_type") else None,
                expression=nd.get("expression") or {},
                artifact_type=nd.get("artifact_type"))
            g.nodes[node.id] = node
        for ed in snap["edges"]:
            g.edges.append(Edge(ed["src"], ed["dst"], EdgeType(ed["rel"])))
        return g

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), indent=2)

    def __repr__(self) -> str:
        return (f"TaskGraph(goal={self.goal!r}, nodes={len(self.nodes)}, "
                f"edges={len(self.edges)})")

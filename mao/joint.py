"""Joint structural-semantic latent space.

Graph embedding g (from the r-GAT structural encoder) and trace embedding t
(from the frozen language encoder) are projected through small MLPs into a
shared R^k, trained with a cosine triplet objective:

    L = max(0, cos(f(g), f(t_neg)) - cos(f(g), f(t_pos)) + margin)

anchor  = current task-graph state
positive = trace of an action respecting every depends_on/Constraint edge
negative = trace of an action that would create a `violates` edge
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.structural import StructuralEncoder
from .graph import TaskGraph


class Projector(nn.Module):
    def __init__(self, in_dim: int, k: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class JointModel(nn.Module):
    """Structural encoder + graph projector + text projector into shared R^k."""

    def __init__(self, text_dim: int = 512, graph_dim: int = 128, k: int = 128):
        super().__init__()
        self.structural = StructuralEncoder(out_dim=graph_dim)
        self.proj_g = Projector(graph_dim, k)
        self.proj_t = Projector(text_dim, k)
        self.k = k

    def embed_graph(self, g: TaskGraph) -> torch.Tensor:
        return self.proj_g(self.structural(g))

    def embed_trace(self, t_vec: torch.Tensor) -> torch.Tensor:
        return self.proj_t(t_vec)

    def alignment(self, g: TaskGraph, t_vec: torch.Tensor) -> float:
        """cos(f(g), f(t)) — the quantity the adjudicator gates on."""
        with torch.no_grad():
            zg = self.embed_graph(g)
            zt = self.embed_trace(t_vec)
            return float((zg * zt).sum())


def triplet_loss(zg: torch.Tensor, zt_pos: torch.Tensor, zt_neg: torch.Tensor,
                 margin: float = 0.3) -> torch.Tensor:
    sim_pos = (zg * zt_pos).sum(-1)
    sim_neg = (zg * zt_neg).sum(-1)
    return F.relu(sim_neg - sim_pos + margin).mean()

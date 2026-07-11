"""Structural encoder: relation-aware graph attention (r-GAT) over the task-graph frontier.

    h_i^(l+1) = sigma( sum_{r in R} sum_{j in N_r(i)} alpha_ij^r * W_r * h_j^(l) )
    alpha_ij^r = softmax_j( LeakyReLU( a_r^T [W_r h_i || W_r h_j] ) )

Relation-specific weights W_r matter because `depends_on` and `conflicts_with`
carry opposite operational semantics (precedence guarantee vs. hard exclusion).

Pure PyTorch (edge-list scatter ops), no torch-geometric dependency — task
graphs are small, so dense per-relation loops are fine and portable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph import TaskGraph, NodeType, Status, EdgeType
from .language import HashingTextEncoder

NUM_RELATIONS = len(EdgeType) + 1  # +1 for added reverse/self handling via distinct rel ids

NODE_TYPES = list(NodeType)
STATUSES = list(Status)

_node_text_encoder = HashingTextEncoder(dim=64, seed=13)


def featurize_graph(g: TaskGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Returns (node_features [N,F], edge_index [2,E], edge_type [E], node_ids).

    Node features = hashed text embedding of description (64)
                    || node-type one-hot (4) || status one-hot (4).
    Edges are made bidirectional; the reverse direction gets its own relation id
    (rel + len(EdgeType)) so direction stays semantically meaningful.
    """
    node_ids = list(g.nodes.keys())
    idx = {nid: i for i, nid in enumerate(node_ids)}

    feats = []
    for nid in node_ids:
        n = g.nodes[nid]
        text = _node_text_encoder.encode(n.description or n.id)
        type_oh = [1.0 if n.type == t else 0.0 for t in NODE_TYPES]
        status_oh = [1.0 if n.status == s else 0.0 for s in STATUSES]
        feats.append(torch.cat([torch.tensor(text, dtype=torch.float32),
                                torch.tensor(type_oh + status_oh)]))
    x = torch.stack(feats) if feats else torch.zeros((0, 64 + len(NODE_TYPES) + len(STATUSES)))

    rel_ids = {r: i for i, r in enumerate(EdgeType)}
    srcs, dsts, rels = [], [], []
    for e in g.edges:
        s, d, r = idx[e.src], idx[e.dst], rel_ids[e.rel]
        srcs.append(s); dsts.append(d); rels.append(r)
        # reverse edge with offset relation id
        srcs.append(d); dsts.append(s); rels.append(r + len(EdgeType))
    edge_index = torch.tensor([srcs, dsts], dtype=torch.long) if srcs else \
        torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.tensor(rels, dtype=torch.long) if rels else \
        torch.zeros((0,), dtype=torch.long)
    return x, edge_index, edge_type, node_ids


class RGATLayer(nn.Module):
    """One layer of relation-aware graph attention over an edge list."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 2 * len(EdgeType)):
        super().__init__()
        self.num_relations = num_relations
        self.W = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.a = nn.Parameter(torch.empty(num_relations, 2 * out_dim))
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        out = self.W_self(h)  # self contribution (identity relation)
        if edge_index.shape[1] == 0:
            return F.elu(out)

        src, dst = edge_index[0], edge_index[1]
        # Per-edge relation-specific transforms of source and dest nodes.
        Wh_src = torch.einsum("ef,rfo->ero", h[src], self.W)  # [E, R, out] -> gather per edge rel
        Wh_dst = torch.einsum("ef,rfo->ero", h[dst], self.W)
        e_range = torch.arange(edge_type.shape[0])
        wh_s = Wh_src[e_range, edge_type]  # [E, out]
        wh_d = Wh_dst[e_range, edge_type]  # [E, out]

        att_vec = self.a[edge_type]  # [E, 2*out]
        logits = F.leaky_relu((att_vec * torch.cat([wh_d, wh_s], dim=-1)).sum(-1))  # [E]

        # softmax over incoming edges per (dst, relation)
        group = dst * self.num_relations + edge_type  # [E] group key
        logits = logits - logits.max()  # stability
        exp = logits.exp()
        denom = torch.zeros(n * self.num_relations, dtype=exp.dtype).scatter_add_(0, group, exp)
        alpha = exp / (denom[group] + 1e-12)  # [E]

        msg = alpha.unsqueeze(-1) * wh_s  # [E, out]
        agg = torch.zeros_like(out).scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return F.elu(out + agg)


class AttentionPooling(nn.Module):
    """Attention-weighted pooling over a masked node subset (the active frontier)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(),
                                   nn.Linear(dim // 2, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(h).squeeze(-1)  # [N]
        scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=0)
        w = torch.nan_to_num(w)  # all-masked edge case
        return (w.unsqueeze(-1) * h).sum(0)


class StructuralEncoder(nn.Module):
    """TaskGraph -> g in R^d via 2 r-GAT layers + frontier attention pooling."""

    def __init__(self, feat_dim: int = 64 + len(NODE_TYPES) + len(STATUSES),
                 hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.l1 = RGATLayer(feat_dim, hidden)
        self.l2 = RGATLayer(hidden, out_dim)
        self.pool = AttentionPooling(out_dim)
        self.out_dim = out_dim

    def forward(self, g: TaskGraph) -> torch.Tensor:
        x, edge_index, edge_type, node_ids = featurize_graph(g)
        h = self.l1(x, edge_index, edge_type)
        h = self.l2(h, edge_index, edge_type)
        # pool over live subtasks (the active frontier), not full history
        mask = torch.tensor([
            g.nodes[nid].type == NodeType.SUBTASK and g.nodes[nid].status != Status.DONE
            for nid in node_ids])
        if not mask.any():  # fully-done graph: pool over everything
            mask = torch.ones(len(node_ids), dtype=torch.bool)
        return self.pool(h, mask)

"""Train the joint space on synthetic contrastive samples, calibrate the
adjudicator threshold tau via F1 on held-out data, and save artifacts.

Usage:  python -m mao.train [--n-train 2000] [--n-val 400] [--epochs 8]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .adjudicator import artifact_subdir
from .datagen import generate
from .encoders.language import (encoder_mode, get_language_encoder,
                                get_node_encoder, node_encoder_mode)
from .joint import JointModel, triplet_loss

ARTIFACT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "artifacts")


def _embed_traces(encoder, samples):
    pos = torch.tensor(encoder.encode_batch([s.pos_trace for s in samples]),
                       dtype=torch.float32)
    neg = torch.tensor(encoder.encode_batch([s.neg_trace for s in samples]),
                       dtype=torch.float32)
    return pos, neg


def calibrate_tau(scores_pos: list[float], scores_neg: list[float]) -> dict:
    """Sweep a threshold over alignment scores; pick tau* maximizing F1 for
    detecting VIOLATIONS (score < tau => predicted violating)."""
    all_scores = sorted(set(scores_pos + scores_neg))
    best = {"tau": 0.0, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for tau in all_scores:
        tp = sum(1 for s in scores_neg if s < tau)          # violation caught
        fp = sum(1 for s in scores_pos if s < tau)          # valid flagged
        fn = sum(1 for s in scores_neg if s >= tau)         # violation missed
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best["f1"]:
            best = {"tau": float(tau), "f1": f1, "precision": prec, "recall": rec}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder", choices=["auto", "local", "gemini", "embeddinggemma"],
                    default="auto",
                    help="trace-encoder backend; artifacts are saved per mode")
    ap.add_argument("--node-encoder", choices=["hash", "embeddinggemma", "gemini"],
                    default="hash",
                    help="graph node-feature encoder: 'hash' (lexical, default) or a "
                         "semantic embedder (EmbeddingGemma local / Gemini). Semantic "
                         "node artifacts save to <mode>_snode-<node>.")
    ap.add_argument("--node-dim", type=int, default=None,
                    help="node-feature dim (default 64 hash / 256 semantic)")
    ap.add_argument("--holdout", type=int, default=None,
                    help="leave-one-template-out: exclude this TEMPLATES index "
                         "from training; artifacts saved to <mode>_holdout<i>")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()

    from .datagen import TEMPLATES
    template_ids = None
    if args.holdout is not None:
        template_ids = [i for i in range(len(TEMPLATES)) if i != args.holdout]
        print(f"[holdout] training WITHOUT template {args.holdout} "
              f"({TEMPLATES[args.holdout]['goal'][:50]}...)")

    print(f"[datagen] generating {args.n_train} train / {args.n_val} val samples...")
    train = generate(args.n_train, seed=args.seed, template_ids=template_ids)
    val = generate(args.n_val, seed=args.seed + 10_000, template_ids=template_ids)

    encoder = get_language_encoder(mode=args.encoder)
    mode = encoder_mode(encoder)
    node_encoder = get_node_encoder(mode=args.node_encoder, dim=args.node_dim)
    node_mode = node_encoder_mode(node_encoder)
    subdir = artifact_subdir(mode, node_mode, args.holdout)
    artifact_dir = os.path.join(ARTIFACT_ROOT, subdir)
    print(f"[encoder] trace backend: {type(encoder).__name__} (mode={mode})")
    print(f"[encoder] node backend : {type(node_encoder).__name__} "
          f"(mode={node_mode}, dim={node_encoder.dim})")
    tr_pos, tr_neg = _embed_traces(encoder, train)
    va_pos, va_neg = _embed_traces(encoder, val)

    model = JointModel(text_dim=encoder.dim, node_encoder=node_encoder)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train))
        total, nb = 0.0, 0
        for start in range(0, len(train), args.batch):
            idx = perm[start:start + args.batch]
            # embed the FRONTIER, matching what handoff packets carry at inference
            zg = torch.stack([model.embed_graph(train[i].graph.frontier())
                              for i in idx.tolist()])
            zp = model.embed_trace(tr_pos[idx])
            zn = model.embed_trace(tr_neg[idx])
            loss = triplet_loss(zg, zp, zn, margin=args.margin)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.detach().item(); nb += 1

        # validation triplet accuracy: is the positive closer than the negative?
        model.eval()
        with torch.no_grad():
            zg = torch.stack([model.embed_graph(s.graph.frontier()) for s in val])
            zp = model.embed_trace(va_pos)
            zn = model.embed_trace(va_neg)
            acc = float(((zg * zp).sum(-1) > (zg * zn).sum(-1)).float().mean())
        print(f"[epoch {epoch}/{args.epochs}] loss={total / nb:.4f}  val_triplet_acc={acc:.3f}")

    # ---- calibrate tau on validation alignment scores ----
    with torch.no_grad():
        zg = torch.stack([model.embed_graph(s.graph.frontier()) for s in val])
        sp = ((zg * model.embed_trace(va_pos)).sum(-1)).tolist()
        sn = ((zg * model.embed_trace(va_neg)).sum(-1)).tolist()
    cal = calibrate_tau(sp, sn)
    # ambiguous band around tau* -> flag-to-human zone
    band = 0.5 * (sum(sp) / len(sp) - sum(sn) / len(sn)) * 0.25
    cal["tau_lo"] = cal["tau"] - abs(band)
    cal["tau_hi"] = cal["tau"] + abs(band)
    print(f"[calibration] tau*={cal['tau']:.3f}  F1={cal['f1']:.3f}  "
          f"P={cal['precision']:.3f}  R={cal['recall']:.3f}  "
          f"flag band=({cal['tau_lo']:.3f}, {cal['tau_hi']:.3f})")

    # ---- negative bank: embeddings of known-violating traces (for nearest-
    # violation explanation at adjudication time) ----
    with torch.no_grad():
        neg_bank = model.embed_trace(tr_neg[:512])

    os.makedirs(artifact_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(artifact_dir, "joint_model.pt"))
    torch.save(neg_bank, os.path.join(artifact_dir, "negative_bank.pt"))
    with open(os.path.join(artifact_dir, "calibration.json"), "w") as f:
        json.dump({**cal, "text_dim": encoder.dim, "mode": mode,
                   "encoder": type(encoder).__name__,
                   "node_mode": node_mode, "node_dim": node_encoder.dim,
                   "node_encoder": type(node_encoder).__name__}, f, indent=2)
    print(f"[done] artifacts saved to {artifact_dir}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

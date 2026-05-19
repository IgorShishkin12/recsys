"""Full-catalog evaluation: HR@K, NDCG@K, MRR with filter-already-seen.

This is the gSASRec / Krichene-Rendle protocol. For each user we score the true
target item against ALL items in the catalog, mask out PAD and items already in
the user's history, take top-K, compute the metrics. No sampled negatives.

The filter-seen step is fully vectorised through a precomputed padded
[n_users, max_history] long tensor; per batch we gather the rows for the active
users and `scatter_` -inf onto the score matrix in one CUDA kernel. This shaves
~10x off eval time vs the per-user Python loop in TOPAPEC/esasrec and
V1adls1aV/esasrec.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

NEG_INF = -1e9


def build_padded_seen(
    user_seen: Dict[int, np.ndarray],
    n_users_total: int,
    cap: int = 3000,
) -> torch.Tensor:
    """[n_users_total, cap] long tensor; rows padded with 0 (PAD column already
    masked at scoring time). Histories longer than `cap` are truncated to the
    most recent `cap` items — for ML-20M cap=3000 covers >99.99% of users.
    """
    out = np.zeros((n_users_total, cap), dtype=np.int64)
    for u, items in user_seen.items():
        if not 0 <= u < n_users_total:
            continue
        if len(items) > cap:
            items = items[-cap:]
        out[u, :len(items)] = items
    return torch.from_numpy(out)


@torch.no_grad()
def evaluate_full_catalog(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    padded_seen: torch.Tensor,                # [n_users_total, max_seen]
    n_items: int,
    ks: Sequence[int] = (5, 10, 20),
    device: str = "cuda",
) -> Dict[str, float]:
    """Compute HR@K, NDCG@K, MRR over a leave-one-out eval split."""
    model.eval()
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    n_users = 0

    seen_dev = padded_seen.to(device, non_blocking=True)

    for batch in eval_loader:
        users = batch["user"].to(device, non_blocking=True)
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        time_buckets = batch["time_delta"].to(device, non_blocking=True) \
            if "time_delta" in batch else None

        scores = model.score_all(inputs, time_buckets)          # [B, V]
        scores[:, 0] = NEG_INF                                  # mask PAD column

        # Vectorised filter-seen: scatter -inf at all seen item ids.
        seen_batch = seen_dev[users]                            # [B, max_seen]
        scores.scatter_(1, seen_batch, NEG_INF)                 # pad index 0 -> already -inf

        # Make sure the true target is NOT filtered (in LOO it's the held-out
        # item, by construction not in train/val seen, but cheap to enforce).
        b_idx = torch.arange(scores.size(0), device=device)
        scores[b_idx, targets] = scores[b_idx, targets].clamp(min=NEG_INF / 2)

        topk_idx = torch.topk(scores, k=max_k, dim=-1).indices  # [B, max_k]
        match = (topk_idx == targets.unsqueeze(-1))             # [B, max_k]
        ranks = torch.where(
            match.any(dim=-1),
            match.float().argmax(dim=-1) + 1,                   # 1-indexed
            torch.zeros_like(targets),
        )                                                       # [B]

        for k in ks:
            hit_at_k = (ranks > 0) & (ranks <= k)
            hits[k] += hit_at_k.sum().item()
            ndcg_at_k = torch.where(
                hit_at_k,
                1.0 / torch.log2(ranks.float() + 1.0),
                torch.zeros_like(ranks, dtype=torch.float),
            )
            ndcgs[k] += ndcg_at_k.sum().item()

        mrr_at = torch.where(
            ranks > 0,
            1.0 / ranks.float(),
            torch.zeros_like(ranks, dtype=torch.float),
        )
        mrr_sum += mrr_at.sum().item()
        n_users += users.shape[0]

    out: Dict[str, float] = {}
    for k in ks:
        out[f"HR@{k}"] = hits[k] / max(n_users, 1)
        out[f"NDCG@{k}"] = ndcgs[k] / max(n_users, 1)
    out[f"MRR@{max_k}"] = mrr_sum / max(n_users, 1)
    out["n_eval_users"] = float(n_users)
    return out


def format_metrics(m: Dict[str, float]) -> str:
    keys = [k for k in m if k != "n_eval_users"]
    parts = [f"{k}={m[k]:.4f}" for k in keys]
    parts.append(f"n={int(m['n_eval_users'])}")
    return " ".join(parts)

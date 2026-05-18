#!/usr/bin/env python3
"""Evaluate each model per-user, save a predictions database, then plot quality
(NDCG@10, HR@10) bucketed by raw training sequence length.

WHY this matters
----------------
All models use max_len=200 at inference: users whose training history exceeds 200
items have their context truncated to the last 200. This script shows exactly
where each model wins or loses relative to sequence length, which informs both
model selection and the question of whether raising max_len is worthwhile.

Usage (from repo root):
    .venv/bin/python -m scripts.eval_by_seqlen [--out profiles] [--device cuda]
    # re-plot from saved predictions (no re-inference):
    .venv/bin/python -m scripts.eval_by_seqlen --from-preds profiles/predictions.csv

Outputs (profiles/):
    predictions.csv         — model, user_id, train_seq_len, effective_len, rank
    quality_by_seqlen.png   — NDCG@10 + HR@10 per seqlen bucket, one line / model
    quality_by_seqlen_ndcg.png / _hr.png  — separated panels
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_processed, split_loo, SeqEvalDataset, build_user_seen_lookup
from src.eval import build_padded_seen
from src.models import build_model
from src.sideinfo import build_sideinfo, SideInfoEmbedding

# ─────────────────────────── Config ──────────────────────────────────────────

CHECKPOINT_MAP = {
    "sasrec":          "ensemble_bases/sasrec/best.pt",
    "sasrec_baseline": "ensemble_bases/sasrec_baseline/best.pt",
    "fmlp":            "runs/fmlp/best.pt",
    "nextitnet":       "runs/nextitnet/best.pt",
    "fnet_hybrid":     "runs/fnet_hybrid/best.pt",
    "causal_fftconv":  "runs/causal_fftconv/best.pt",
    # linear_attn: no checkpoint saved
}

MAX_LEN      = 200
EVAL_BS      = 512
NEG_INF      = -1e9

# Bucket boundaries (right-exclusive); last bucket is open-ended.
# Aligns with powers of 2 and explicitly straddles max_len=200.
BUCKET_EDGES = [0, 20, 40, 80, 130, 200, 320, 512, 1024, 2048, 10**6]

# ─────────────────────────── Model loading ───────────────────────────────────

def _build_side(model_cfg: dict, data, data_dir: Path) -> Optional[nn.Module]:
    if not model_cfg.get("use_side", False):
        return None
    genome_csv = data_dir / "genome-scores.csv"
    side_tables = build_sideinfo(
        movies_csv=data_dir / "movies.csv",
        movie_id_to_idx=data.movie_id_to_idx,
        n_items=data.n_items,
        genome_scores_csv=genome_csv if (model_cfg.get("side_use_genome") and genome_csv.exists()) else None,
    )
    return SideInfoEmbedding(
        d=model_cfg["d"],
        side=side_tables,
        use_genome=model_cfg.get("side_use_genome", False),
        dropout=model_cfg.get("side_dropout", 0.1),
    )


def load_model(ck_path: Path, data, data_dir: Path, device: torch.device) -> nn.Module:
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model_cfg = dict(ck["config"]["model"])
    n_items = data.n_items

    side_module = _build_side(model_cfg, data, data_dir)
    model = build_model(model_cfg, n_items=n_items, side_module=side_module)

    sd = ck.get("ema_state_dict") or ck["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


# ─────────────────────────── Per-user evaluation ─────────────────────────────

@torch.no_grad()
def eval_per_user(
    model: nn.Module,
    loader: DataLoader,
    padded_seen: torch.Tensor,
    device: torch.device,
) -> List[Tuple[int, int]]:
    """Return list of (user_id, rank) where rank is 1-indexed full-catalog rank.

    Filtered-seen protocol (same as evaluate_full_catalog): seen items get
    score = -inf before ranking, true target is protected.
    """
    results: List[Tuple[int, int]] = []
    seen_dev = padded_seen.to(device, non_blocking=True)

    for batch in loader:
        users   = batch["user"].to(device, non_blocking=True)
        inputs  = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        B       = users.size(0)

        scores = model.score_all(inputs)          # [B, n_items+1]
        scores[:, 0] = NEG_INF                    # mask PAD column

        seen_batch = seen_dev[users]              # [B, max_seen]
        scores.scatter_(1, seen_batch, NEG_INF)

        b_idx = torch.arange(B, device=device)
        # Protect true target from being filtered (safety, not strictly needed)
        scores[b_idx, targets] = scores[b_idx, targets].clamp(min=NEG_INF / 2)

        # 1-indexed rank of true target among ALL items (after seen filtering)
        target_scores = scores[b_idx, targets].unsqueeze(1)   # [B, 1]
        rank = (scores > target_scores).sum(dim=1).add_(1)    # [B]

        results.extend(zip(users.cpu().tolist(), rank.cpu().tolist()))
    return results


# ─────────────────────────── Bucketing + metrics ─────────────────────────────

def bucket_label(lo: int, hi: int) -> str:
    if hi >= 10**6:
        return f">{lo}"
    return f"{lo+1}-{hi}"


BUCKET_LABELS = [
    bucket_label(BUCKET_EDGES[i], BUCKET_EDGES[i + 1])
    for i in range(len(BUCKET_EDGES) - 1)
]


def assign_bucket(length: int) -> int:
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] < length <= BUCKET_EDGES[i + 1]:
            return i
    return len(BUCKET_EDGES) - 2  # last bucket


def ndcg_at_k(rank: int, k: int) -> float:
    return (1.0 / np.log2(rank + 1)) if 0 < rank <= k else 0.0


def hr_at_k(rank: int, k: int) -> int:
    return 1 if 0 < rank <= k else 0


def bucket_metrics(
    rows: List[dict],
    k: int = 10,
) -> dict:
    """Compute mean NDCG@k and HR@k per bucket, per model.

    Returns {model: {bucket_idx: (ndcg, hr, n_users, sem_ndcg, sem_hr)}}.
    """
    from collections import defaultdict
    # {model: {bucket: [ranks]}}
    store: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        b = assign_bucket(r["train_seq_len"])
        store[r["model"]][b].append(r["rank"])

    out = {}
    for model, buckets in store.items():
        out[model] = {}
        for b, ranks in buckets.items():
            ndcgs = [ndcg_at_k(rk, k) for rk in ranks]
            hrs   = [hr_at_k(rk, k) for rk in ranks]
            n     = len(ranks)
            out[model][b] = (
                float(np.mean(ndcgs)),
                float(np.mean(hrs)),
                n,
                float(np.std(ndcgs) / np.sqrt(n)) if n > 1 else 0.0,
                float(np.std(hrs)   / np.sqrt(n)) if n > 1 else 0.0,
            )
    return out


# ─────────────────────────── Plotting ────────────────────────────────────────

def make_plots(
    bm: dict,
    out_dir: Path,
    k: int = 10,
    max_len: int = MAX_LEN,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models  = list(bm.keys())
    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    n_buckets = len(BUCKET_LABELS)
    xs = np.arange(n_buckets)

    # ── find which bucket contains max_len ──────────────────────────────────
    trunc_bucket = assign_bucket(max_len)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax_idx, (metric_key, metric_label) in enumerate(
        [("ndcg", f"NDCG@{k}"), ("hr", f"HR@{k}")]
    ):
        ax = axes[ax_idx]

        for i, model in enumerate(models):
            pts = bm[model]
            plot_xs, plot_ys, plot_err = [], [], []
            for b in range(n_buckets):
                if b in pts and pts[b][2] >= 10:   # skip bins with <10 users
                    plot_xs.append(b)
                    if metric_key == "ndcg":
                        plot_ys.append(pts[b][0])
                        plot_err.append(pts[b][3])
                    else:
                        plot_ys.append(pts[b][1])
                        plot_err.append(pts[b][4])

            if not plot_xs:
                continue
            ax.errorbar(
                plot_xs, plot_ys, yerr=plot_err,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linewidth=1.8, markersize=6,
                capsize=3, elinewidth=0.8,
                label=model,
            )

        # Truncation boundary
        ax.axvline(trunc_bucket + 0.5, color="black", linewidth=1.2,
                   linestyle=":", alpha=0.7,
                   label=f"max_len={max_len} (truncation boundary)")
        ax.fill_betweenx(
            [0, 1], trunc_bucket + 0.5, n_buckets - 0.5,
            alpha=0.05, color="grey",
            label="truncated region",
        )

        ax.set_xticks(xs)
        ax.set_xticklabels(BUCKET_LABELS, rotation=35, ha="right", fontsize=9)
        ax.set_xlabel("Training sequence length (raw items, before truncation)", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(
            f"{metric_label} by training sequence length\n"
            f"(full-catalog LOO eval; users with len>{max_len} get last {max_len} items)",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlim(-0.5, n_buckets - 0.5)

    fig.tight_layout()
    out = out_dir / "quality_by_seqlen.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    # ── per-metric single-panel versions ────────────────────────────────────
    for metric_key, metric_label, fname in [
        ("ndcg", f"NDCG@{k}", "quality_by_seqlen_ndcg.png"),
        ("hr",   f"HR@{k}",   "quality_by_seqlen_hr.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, model in enumerate(models):
            pts = bm[model]
            plot_xs, plot_ys, plot_err = [], [], []
            for b in range(n_buckets):
                if b in pts and pts[b][2] >= 10:
                    plot_xs.append(b)
                    if metric_key == "ndcg":
                        plot_ys.append(pts[b][0])
                        plot_err.append(pts[b][3])
                    else:
                        plot_ys.append(pts[b][1])
                        plot_err.append(pts[b][4])
            if not plot_xs:
                continue
            ax.errorbar(
                plot_xs, plot_ys, yerr=plot_err,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linewidth=1.8, markersize=6,
                capsize=3, elinewidth=0.8,
                label=model,
            )

        # Count users per bucket for annotation
        total_per_bucket: dict = {}
        for model_pts in bm.values():
            for b, v in model_pts.items():
                total_per_bucket[b] = max(total_per_bucket.get(b, 0), v[2])

        for b in range(n_buckets):
            n = total_per_bucket.get(b, 0)
            if n >= 10:
                ax.text(b, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
                        f"n={n}", ha="center", va="bottom", fontsize=7,
                        color="grey", rotation=90)

        ax.axvline(trunc_bucket + 0.5, color="black", linewidth=1.2,
                   linestyle=":", alpha=0.7,
                   label=f"max_len={max_len}")
        ax.fill_betweenx([0, 1], trunc_bucket + 0.5, n_buckets - 0.5,
                         alpha=0.05, color="grey")
        ax.set_xticks(xs)
        ax.set_xticklabels(BUCKET_LABELS, rotation=35, ha="right", fontsize=9)
        ax.set_xlabel("Training sequence length", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(
            f"{metric_label} by training sequence length\n"
            f"(shaded = len>{max_len}, truncated to last {max_len} items)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlim(-0.5, n_buckets - 0.5)
        fig.tight_layout()
        out = out_dir / fname
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")


# ─────────────────────────── CSV helpers ─────────────────────────────────────

def save_predictions(rows: List[dict], path: Path) -> None:
    fields = ["model", "user_id", "train_seq_len", "effective_len", "rank"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path}  ({len(rows):,} rows)")


def load_predictions(path: Path) -> List[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "model":        r["model"],
                "user_id":      int(r["user_id"]),
                "train_seq_len": int(r["train_seq_len"]),
                "effective_len": int(r["effective_len"]),
                "rank":         int(r["rank"]),
            })
    return rows


# ─────────────────────────── Main ────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",       default="data/processed.pkl")
    p.add_argument("--data-dir",   default="data/ml-20m",
                   help="directory containing movies.csv and genome-scores.csv")
    p.add_argument("--out",        default="profiles")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--from-preds", metavar="CSV",
                   help="Skip inference; load existing predictions.csv and replot.")
    p.add_argument("--k",          type=int, default=10,
                   help="Cutoff for NDCG@k and HR@k (default 10)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    if args.from_preds:
        print(f"Loading predictions from {args.from_preds} …")
        all_rows = load_predictions(Path(args.from_preds))
        print(f"  {len(all_rows):,} rows, "
              f"{len({r['model'] for r in all_rows})} models")
    else:
        device = torch.device(
            args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        print(f"Device : {device}")
        if device.type == "cuda":
            print(f"GPU    : {torch.cuda.get_device_name(device)}")
        print()

        print("Loading data …", flush=True)
        data = load_processed(Path(args.data))
        train_seq, val_map, test_map = split_loo(data.user_seq)

        # Build sequence-length lookup: user_id → train seq len (raw)
        train_seq_len = {u: len(s) for u, s in train_seq.items()}

        user_seen  = build_user_seen_lookup(train_seq)
        padded_seen = build_padded_seen(user_seen, data.n_users)

        # Annotate truncation
        n_trunc = sum(1 for l in train_seq_len.values() if l > MAX_LEN)
        pct     = 100 * n_trunc / len(train_seq_len)
        print(f"  {data.n_users:,} users | {data.n_items:,} items")
        print(f"  Users with train len > {MAX_LEN}: {n_trunc:,} ({pct:.1f}%) — "
              f"truncated to last {MAX_LEN} items at inference")
        print()

        test_dataset = SeqEvalDataset(
            train_seq=train_seq,
            target_map=test_map,
            max_len=MAX_LEN,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=EVAL_BS, shuffle=False,
            num_workers=4, pin_memory=(device.type == "cuda"),
        )

        all_rows: List[dict] = []

        for model_name, ck_rel in CHECKPOINT_MAP.items():
            ck_path = Path(ck_rel)
            if not ck_path.exists():
                print(f"[skip] {model_name}: checkpoint not found at {ck_path}")
                continue

            print(f"── {model_name} ──", flush=True)
            try:
                model = load_model(ck_path, data, data_dir, device)
            except Exception as exc:
                print(f"  [skip] load failed: {exc}")
                continue

            user_ranks = eval_per_user(model, test_loader, padded_seen, device)
            del model
            torch.cuda.empty_cache()

            for uid, rank in user_ranks:
                tlen = train_seq_len.get(uid, 0)
                all_rows.append({
                    "model":         model_name,
                    "user_id":       uid,
                    "train_seq_len": tlen,
                    "effective_len": min(tlen, MAX_LEN),
                    "rank":          rank,
                })
            print(f"  {len(user_ranks):,} users evaluated", flush=True)

        save_predictions(all_rows, out_dir / "predictions.csv")

    print()
    print(f"Computing NDCG@{args.k} / HR@{args.k} by seqlen bucket …", flush=True)
    bm = bucket_metrics(all_rows, k=args.k)

    # Print summary table
    print()
    header = f"  {'Model':<18}  " + "  ".join(
        f"{lbl:>10}" for lbl in BUCKET_LABELS
    )
    print("  NDCG@10 per bucket:")
    print(header)
    for model, pts in sorted(bm.items()):
        vals = []
        for b in range(len(BUCKET_LABELS)):
            if b in pts and pts[b][2] >= 10:
                vals.append(f"{pts[b][0]:>10.4f}")
            else:
                vals.append(f"{'—':>10}")
        print(f"  {model:<18}  " + "  ".join(vals))

    print()
    print("Generating plots …")
    make_plots(bm, out_dir, k=args.k)
    print(f"\nDone. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

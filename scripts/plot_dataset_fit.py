#!/usr/bin/env python3
"""Convolve model scaling laws with the actual sequence-length distribution.

At inference each user gets left_pad(train_history[-max_len:], max_len).
The *effective* sequence length is min(len(train_history), max_len).
This script asks: given our data distribution, what is the expected
per-sample cost for each model, and what max_len covers N% of users?

Usage (from repo root):
    python -m scripts.plot_dataset_fit [--data data/processed.pkl] [--out profiles]

Outputs (profiles/):
    seqlen_dist.png          — histogram + CDF of effective train sequence lengths
    dataset_fit_time.png     — time/sample cost curves + distribution
    dataset_fit_mem.png      — memory/sample cost curves + distribution
    dataset_fit_report.csv   — E[time], E[mem], cost at p50/p90/p99 max_len, rank
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────── Sequence-length distribution ────────────────────

def load_seq_lengths(data_path: Path) -> np.ndarray:
    """Return array of effective train-split sequence lengths, one per user."""
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    # LOO split: last item → test, second-last → val, rest → train.
    # Effective train sequence length = max(0, len(user_seq) - 2).
    lengths = []
    for seq in data.user_seq.values():
        train_len = max(0, len(seq) - 2)
        if train_len > 0:
            lengths.append(train_len)
    return np.array(lengths, dtype=np.int64)


def print_dist_stats(lengths: np.ndarray) -> None:
    ps = [10, 25, 50, 75, 90, 95, 99]
    print(f"  n_users : {len(lengths):,}")
    print(f"  min/max : {lengths.min()} / {lengths.max()}")
    print(f"  mean    : {lengths.mean():.1f}  std={lengths.std():.1f}")
    for p in ps:
        print(f"  p{p:02d}    : {np.percentile(lengths, p):.0f}")


# ─────────────────────────── Load scaling CSV ────────────────────────────────

def load_seqlen_results(
    csv_path: Path,
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """Return {model: {L: (time_ms_per_sample, mem_mb_per_sample)}}."""
    results: Dict[str, Dict[int, Tuple[float, float]]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["model"]
            L    = int(row["seq_len"])
            B    = int(row["batch_size"])
            t    = float(row["time_ms"]) / B
            m    = float(row["mem_mb"]) / B
            results.setdefault(name, {})[L] = (t, m)
    return results


# ─────────────────────────── Interpolation ───────────────────────────────────

def interp_loglog(
    L_query: np.ndarray,
    L_known: np.ndarray,
    y_known: np.ndarray,
) -> np.ndarray:
    """Log-log linear interpolation (power-law); extrapolates at the edges."""
    log_L = np.log2(L_known.astype(float))
    log_y = np.log2(y_known.astype(float))
    log_q = np.log2(np.clip(L_query.astype(float), 1, None))
    log_out = np.interp(log_q, log_L, log_y,
                        left=log_y[0], right=log_y[-1])
    return 2.0 ** log_out


# ─────────────────────────── Expected cost ───────────────────────────────────

def expected_cost(
    lengths: np.ndarray,
    model_results: Dict[int, Tuple[float, float]],
    max_len: int = 200,
) -> Tuple[float, float]:
    """E[time_per_sample], E[mem_per_sample] under the data distribution.

    The effective length used at inference is min(user_len, max_len).
    Cost is interpolated from the scaling sweep.
    """
    L_known = np.array(sorted(model_results.keys()))
    t_known = np.array([model_results[L][0] for L in L_known])
    m_known = np.array([model_results[L][1] for L in L_known])

    effective = np.minimum(lengths, max_len).astype(float)

    t_per = interp_loglog(effective, L_known, t_known)
    m_per = interp_loglog(effective, L_known, m_known)

    return float(t_per.mean()), float(m_per.mean())


# ─────────────────────────── Plotting ────────────────────────────────────────

def make_dist_plot(lengths: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram (log x, linear y)
    bins = np.logspace(np.log10(max(1, lengths.min())),
                       np.log10(lengths.max()), 60)
    ax1.hist(lengths, bins=bins, color="steelblue", edgecolor="none", alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_xlabel("Effective train sequence length", fontsize=12)
    ax1.set_ylabel("Number of users", fontsize=12)
    ax1.set_title("Sequence length distribution (train split)", fontsize=12, fontweight="bold")
    for p, lw in [(50, 1.2), (90, 1.2), (99, 1.2)]:
        v = np.percentile(lengths, p)
        ax1.axvline(v, color="tomato", linewidth=lw, linestyle="--", alpha=0.8,
                    label=f"p{p}={v:.0f}")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # CDF
    sorted_l = np.sort(lengths)
    cdf = np.arange(1, len(sorted_l) + 1) / len(sorted_l)
    ax2.plot(sorted_l, cdf, color="steelblue", linewidth=1.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Effective train sequence length", fontsize=12)
    ax2.set_ylabel("Fraction of users", fontsize=12)
    ax2.set_title("CDF of sequence lengths", fontsize=12, fontweight="bold")
    for target in [0.5, 0.75, 0.90, 0.95, 0.99]:
        thresh = int(np.percentile(lengths, target * 100))
        ax2.axhline(target, color="grey", linewidth=0.8, linestyle=":")
        ax2.axvline(thresh, color="tomato", linewidth=0.8, linestyle="--",
                    label=f"{int(target*100)}%→L={thresh}")
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def make_fit_plot(
    lengths: np.ndarray,
    results: Dict[str, Dict[int, Tuple[float, float]]],
    metric_idx: int,
    ylabel: str,
    out_path: Path,
    max_len: int = 200,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models  = list(results.keys())
    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, ax_main = plt.subplots(figsize=(12, 6))

    # Right y-axis: sequence length density (log x scale)
    ax_dist = ax_main.twinx()
    bins = np.logspace(np.log10(max(1, lengths.min())),
                       np.log10(max(lengths.max(), max_len * 2)),
                       80)
    counts, edges = np.histogram(lengths, bins=bins, density=True)
    ax_dist.fill_between((edges[:-1] + edges[1:]) / 2, counts,
                         alpha=0.12, color="grey", label="data distribution")
    ax_dist.set_ylabel("Sequence length density", fontsize=10, color="grey")
    ax_dist.tick_params(axis="y", labelcolor="grey", labelsize=8)
    ax_dist.set_ylim(bottom=0)

    # Cost curves
    all_L = sorted({L for m in results.values() for L in m})
    for i, name in enumerate(models):
        pts = results[name]
        L_known = np.array(sorted(pts.keys()))
        y_known = np.array([pts[L][metric_idx] for L in L_known])

        ax_main.plot(L_known, y_known,
                     color=colors[i % len(colors)],
                     marker=markers[i % len(markers)],
                     linewidth=1.8, markersize=6, label=name, zorder=3)

        # Interpolated curve between points
        L_fine = np.logspace(np.log2(L_known.min()), np.log2(L_known.max()),
                             200, base=2)
        y_fine = interp_loglog(L_fine, L_known, y_known)
        ax_main.plot(L_fine, y_fine,
                     color=colors[i % len(colors)],
                     linewidth=0.7, alpha=0.4, zorder=2)

    # Vertical line at max_len=200 (current training setting)
    ax_main.axvline(max_len, color="black", linewidth=1.2, linestyle=":",
                    label=f"current max_len={max_len}")

    # Percentile markers on x-axis
    for p_pct, ls in [(90, "--"), (95, "-.")]:
        p_val = np.percentile(lengths, p_pct)
        ax_main.axvline(p_val, color="tomato", linewidth=0.9, linestyle=ls,
                        alpha=0.7, label=f"data p{p_pct}={p_val:.0f}")

    ax_main.set_xscale("log", base=2)
    ax_main.set_yscale("log")
    ax_main.set_xlabel("Sequence length", fontsize=12)
    ax_main.set_ylabel(ylabel, fontsize=12)
    ax_main.set_title(
        f"Model cost vs seq-len, overlaid with data distribution\n"
        f"(per sample; max_len={max_len} for current runs)",
        fontsize=12, fontweight="bold")

    xticks = sorted({*all_L, max_len, int(np.percentile(lengths, 90)),
                     int(np.percentile(lengths, 95))})
    ax_main.set_xticks(xticks)
    ax_main.set_xticklabels([str(x) for x in xticks], fontsize=8, rotation=45)
    ax_main.grid(True, which="both", alpha=0.25)
    ax_main.legend(loc="upper left", bbox_to_anchor=(1.08, 1.0),
                   borderaxespad=0, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────── Report ──────────────────────────────────────────

def make_report(
    lengths: np.ndarray,
    results: Dict[str, Dict[int, Tuple[float, float]]],
    out_path: Path,
    max_len: int = 200,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(results.keys())

    # Percentile max_lens from data
    p50  = int(np.percentile(lengths, 50))
    p90  = int(np.percentile(lengths, 90))
    p95  = int(np.percentile(lengths, 95))
    p99  = int(np.percentile(lengths, 99))

    rows = []
    for name in models:
        pts = results[name]
        L_known = np.array(sorted(pts.keys()))
        t_known = np.array([pts[L][0] for L in L_known])
        m_known = np.array([pts[L][1] for L in L_known])

        e_t, e_m = expected_cost(lengths, pts, max_len=max_len)

        def _at(L):
            t = float(interp_loglog(np.array([L]), L_known, t_known)[0])
            m = float(interp_loglog(np.array([L]), L_known, m_known)[0])
            return t, m

        t50,  m50  = _at(p50)
        t90,  m90  = _at(p90)
        t95,  m95  = _at(p95)
        t99,  m99  = _at(p99)
        t200, m200 = _at(max_len)

        rows.append({
            "model":       name,
            "E[time_ms]":  round(e_t, 4),
            "E[mem_mb]":   round(e_m, 4),
            f"t@p50(L={p50})":  round(t50, 4),
            f"t@p90(L={p90})":  round(t90, 4),
            f"t@p95(L={p95})":  round(t95, 4),
            f"t@p99(L={p99})":  round(t99, 4),
            f"t@L={max_len}":   round(t200, 4),
            f"m@p50(L={p50})":  round(m50, 4),
            f"m@p90(L={p90})":  round(m90, 4),
            f"m@p95(L={p95})":  round(m95, 4),
            f"m@L={max_len}":   round(m200, 4),
        })

    rows_t = sorted(rows, key=lambda r: r["E[time_ms]"])
    rows_m = sorted(rows, key=lambda r: r["E[mem_mb]"])
    for rank, r in enumerate(rows_t):
        r["rank_time"] = rank + 1
    for rank, r in enumerate(rows_m):
        r["rank_mem"] = rank + 1

    # Sort by E[time] for CSV
    rows = sorted(rows, key=lambda r: r["E[time_ms]"])

    # Bar chart of E[cost]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    names_sorted_t = [r["model"] for r in rows]
    names_sorted_m = sorted(rows, key=lambda r: r["E[mem_mb]"])

    colors_bar = plt.cm.tab10.colors

    # Time bar
    vals_t = [r["E[time_ms]"] for r in rows]
    bars = ax1.barh(names_sorted_t, vals_t,
                    color=[colors_bar[i % 10] for i in range(len(rows))],
                    edgecolor="none", alpha=0.85)
    ax1.set_xlabel("E[time per sample]  (ms)", fontsize=11)
    ax1.set_title(
        f"Expected inference time under data distribution\n"
        f"(min(user_seq_len, {max_len}), log-log interpolated from sweep)",
        fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    for bar, v in zip(bars, vals_t):
        ax1.text(v * 1.02, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", fontsize=8)
    ax1.grid(True, axis="x", alpha=0.3)

    # Memory bar
    rows_by_mem = sorted(rows, key=lambda r: r["E[mem_mb]"])
    names_sorted_m = [r["model"] for r in rows_by_mem]
    vals_m = [r["E[mem_mb]"] for r in rows_by_mem]
    model_to_color = {r["model"]: colors_bar[i % 10] for i, r in enumerate(rows)}
    bars2 = ax2.barh(names_sorted_m, vals_m,
                     color=[model_to_color[n] for n in names_sorted_m],
                     edgecolor="none", alpha=0.85)
    ax2.set_xlabel("E[memory per sample]  (MB)", fontsize=11)
    ax2.set_title(
        f"Expected peak memory per sample under data distribution\n"
        f"(min(user_seq_len, {max_len}), log-log interpolated from sweep)",
        fontsize=10, fontweight="bold")
    ax2.invert_yaxis()
    for bar, v in zip(bars2, vals_m):
        ax2.text(v * 1.02, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", fontsize=8)
    ax2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    bar_path = out_path.parent / "dataset_fit_expected.png"
    fig.savefig(bar_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {bar_path}")

    # Save CSV
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {out_path}")

    # Print summary table
    print()
    print(f"  {'Model':<18}  {'E[time]':>10}  {'rank':>4}  {'E[mem]':>10}  {'rank':>4}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*4}  {'-'*10}  {'-'*4}")
    for r in rows:
        print(f"  {r['model']:<18}  {r['E[time_ms]']:>10.4f}  {r['rank_time']:>4}  "
              f"  {r['E[mem_mb]']:>10.4f}  {r['rank_mem']:>4}")


# ─────────────────────────── Main ────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",    default="data/processed.pkl")
    p.add_argument("--sl-csv",  default="profiles/scaling_seqlen_results.csv")
    p.add_argument("--out",     default="profiles")
    p.add_argument("--max-len", type=int, default=200,
                   help="max_len used during training/inference (default 200)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sequence lengths …", flush=True)
    lengths = load_seq_lengths(Path(args.data))
    print_dist_stats(lengths)
    print()

    print("Loading scaling results …", flush=True)
    results = load_seqlen_results(Path(args.sl_csv))
    print(f"  {len(results)} models, L={sorted(next(iter(results.values())).keys())}")
    print()

    print("Generating plots …", flush=True)
    make_dist_plot(lengths, out_dir / "seqlen_dist.png")
    make_fit_plot(lengths, results, metric_idx=0,
                  ylabel="Inference time per sample (ms)",
                  out_path=out_dir / "dataset_fit_time.png",
                  max_len=args.max_len)
    make_fit_plot(lengths, results, metric_idx=1,
                  ylabel="Peak memory per sample (MB)",
                  out_path=out_dir / "dataset_fit_mem.png",
                  max_len=args.max_len)
    print()

    print("Computing expected costs under data distribution …", flush=True)
    make_report(lengths, results, out_dir / "dataset_fit_report.csv",
                max_len=args.max_len)

    print(f"\nDone. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

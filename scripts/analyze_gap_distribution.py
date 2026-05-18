#!/usr/bin/env python3
"""Analyse how the number of rating-session clusters varies with gap threshold.

Two complementary views:
  1. n_sessions(gap)  — total sessions as gap threshold sweeps 0 → 1 day.
                        Monotonically decreasing; elbows mark natural boundaries.
  2. gap density      — histogram of all inter-rating gaps (= −d(n_sessions)/d(gap)).
                        Peaks reveal: automated imports (0–1 s), episode lengths
                        (~20–45 min), film lengths (~90–120 min), sleep (~8 h).

Three resolution zones:
  • 0–60 s   at 0.1 s  (600 pts) — spot exact-second and sub-minute automation
  • 60 s–1 h at 1 s    (3540 pts)
  • 1 h–24 h at 60 s   (1380 pts)
                       ≈ 5520 total threshold points

Usage (from repo root):
    python -m scripts.analyze_gap_distribution [--data data] [--out profiles]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────── Data loading ────────────────────────────────────

def load_gaps(ratings_csv: Path, processed_pkl: Path) -> np.ndarray:
    """Return sorted array of all inter-rating gaps (seconds) across all users."""
    with open(processed_pkl, "rb") as f:
        proc = pickle.load(f)
    valid_users  = set(proc.user_id_to_idx.keys())
    valid_movies = set(proc.movie_id_to_idx.keys())

    print(f"Loading {ratings_csv} …", flush=True)
    df = pd.read_csv(ratings_csv, usecols=["userId", "movieId", "timestamp"])
    df = df[df["userId"].isin(valid_users) & df["movieId"].isin(valid_movies)]
    df = df.sort_values(["userId", "timestamp"], kind="stable")
    print(f"  {len(df):,} ratings, {df['userId'].nunique():,} users")

    # Compute per-user consecutive gaps, skip the first row of each user group
    df["gap"] = df.groupby("userId")["timestamp"].diff()
    gaps = df["gap"].dropna().to_numpy(dtype=np.int64)
    # Negative gaps can occur if timestamps aren't strictly sorted within user
    # (rare, treat as 0)
    gaps = np.maximum(gaps, 0)
    gaps.sort()
    print(f"  {len(gaps):,} inter-rating gaps  "
          f"(min={gaps.min()} s, max={gaps.max()} s, "
          f"median={int(np.median(gaps))} s)")
    return gaps


# ─────────────────────────── Threshold sweep ─────────────────────────────────

def sweep_thresholds(
    gaps: np.ndarray,
    n_users: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (thresholds, n_sessions) arrays.

    n_sessions(g) = n_users + number_of_gaps_GREATER_THAN_g
    As g → 0:   n_sessions → n_users + len(gaps)   (every gap is a break)
    As g → inf: n_sessions → n_users               (no gaps are breaks)
    """
    # Three-zone grid:
    #   0–60 s    at 0.1 s  resolution
    #   60–3600 s at 1 s    resolution
    #   3600–86400 s at 60 s resolution
    t1 = np.arange(0, 60.1, 0.1)
    t2 = np.arange(61, 3601, 1, dtype=float)
    t3 = np.arange(3660, 86401, 60, dtype=float)
    thresholds = np.concatenate([t1, t2, t3])

    # For each threshold g: n_breaks = len(gaps) - searchsorted(gaps, g, 'right')
    # = number of gaps > g  (since gaps is sorted)
    indices = np.searchsorted(gaps, thresholds, side="right")
    n_breaks = len(gaps) - indices
    n_sessions = n_users + n_breaks

    return thresholds, n_sessions


# ─────────────────────────── Plotting ────────────────────────────────────────

ANNOTATIONS = [
    (1,       "1 s"),
    (5,       "5 s"),
    (30,      "30 s"),
    (300,     "5 min"),
    (1200,    "20 min\n(short ep)"),
    (1800,    "30 min"),
    (2700,    "45 min\n(long ep)"),
    (5400,    "90 min\n(short film)"),
    (7200,    "2 h\n(film)"),
    (10800,   "3 h"),
    (28800,   "8 h\n(sleep)"),
    (86400,   "1 day"),
]


def _add_vlines(ax, color="grey", alpha=0.35):
    for sec, label in ANNOTATIONS:
        ax.axvline(sec, color=color, linewidth=0.7, linestyle=":", alpha=alpha)
        ax.text(sec, ax.get_ylim()[1] * 0.97, label,
                ha="center", va="top", fontsize=6.5, color="dimgrey",
                rotation=90, clip_on=True)


def make_plots(
    gaps: np.ndarray,
    thresholds: np.ndarray,
    n_sessions: np.ndarray,
    n_users: int,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total_possible = n_users + len(gaps)   # n_sessions when gap=0

    # ── Figure 1: n_sessions(gap) on log-x, full range ──────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    def _plot_nsessions(ax, xlim, title):
        mask = (thresholds >= xlim[0]) & (thresholds <= xlim[1])
        ax.plot(thresholds[mask], n_sessions[mask],
                color="steelblue", linewidth=1.3)
        ax.set_xlim(xlim)
        ax.set_xlabel("Gap threshold (seconds)", fontsize=10)
        ax.set_ylabel("Total sessions", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
        ax.grid(True, alpha=0.3)
        _add_vlines(ax)
        return ax

    # (a) full range, log-x
    ax = axes[0, 0]
    ax.semilogx(thresholds[1:], n_sessions[1:], color="steelblue", linewidth=1.3)
    ax.set_xlabel("Gap threshold (seconds, log scale)", fontsize=10)
    ax.set_ylabel("Total sessions", fontsize=10)
    ax.set_title("n_sessions vs gap — full range (log x)", fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
    ax.grid(True, which="both", alpha=0.3)
    for sec, label in ANNOTATIONS:
        ax.axvline(sec, color="grey", linewidth=0.7, linestyle=":", alpha=0.4)

    # (b) zoom: 0–120 s (automation + sub-minute)
    _plot_nsessions(axes[0, 1], (0, 120),
                    "Zoom: 0–120 s  (automated imports?)")
    axes[0, 1].set_xlabel("Gap threshold (seconds)", fontsize=10)

    # (c) zoom: 0–7200 s (episode / film lengths)
    _plot_nsessions(axes[1, 0], (0, 7200),
                    "Zoom: 0–2 h  (episode / film lengths)")

    # (d) zoom: 3600–86400 s (multi-hour / daily patterns)
    _plot_nsessions(axes[1, 1], (3600, 86400),
                    "Zoom: 1 h–1 day")

    fig.suptitle("Number of sessions vs gap threshold — ML-20M",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "gap_nsessions.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    # ── Figure 2: gap density histogram (= −d n_sessions / d gap) ───────────
    # Use the derivative of n_sessions as a proxy for gap density.
    # Positive derivative magnitude = many gaps at that size.
    dn = -np.diff(n_sessions)   # positive: gaps "disappear" as threshold rises
    dt = np.diff(thresholds)
    density = dn / dt            # rate: sessions lost per second of threshold

    mid = (thresholds[:-1] + thresholds[1:]) / 2

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    def _plot_density(ax, xlim, title, log_y=False):
        mask = (mid >= xlim[0]) & (mid <= xlim[1])
        x = mid[mask]
        y = density[mask]
        ax.fill_between(x, y, alpha=0.6, color="steelblue")
        ax.plot(x, y, color="steelblue", linewidth=0.6)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_xlabel("Gap size (seconds)", fontsize=10)
        ax.set_ylabel("Density (sessions / s)", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        _add_vlines(ax)

    _plot_density(axes[0, 0], (0, 120),
                  "Gap density: 0–120 s  (automated bulk uploads?)")
    _plot_density(axes[0, 1], (0, 7200),
                  "Gap density: 0–2 h  (episode / film viewing patterns)")
    _plot_density(axes[1, 0], (0, 86400),
                  "Gap density: full range 0–24 h", log_y=False)
    _plot_density(axes[1, 1], (0, 86400),
                  "Gap density: full range 0–24 h  (log y)", log_y=True)

    fig.suptitle("Inter-rating gap density — ML-20M\n"
                 "(peaks = natural session boundaries)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "gap_density.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    # ── Figure 3: ultra-fine 0–10 s ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # exact counts for gap = 0, 1, 2, ..., 10 s
    exact_counts = {g: int(np.sum(gaps == g)) for g in range(11)}
    ax = axes[0]
    ax.bar(list(exact_counts.keys()), list(exact_counts.values()),
           color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Exact gap (seconds)", fontsize=11)
    ax.set_ylabel("Number of gaps", fontsize=11)
    ax.set_title("Exact gap counts: 0–10 s\n(0 s = same timestamp = likely bulk import)",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(range(11))
    for x, y in exact_counts.items():
        if y > 0:
            ax.text(x, y * 1.01, f"{y:,}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    mask_fine = mid <= 10
    ax.fill_between(mid[mask_fine], density[mask_fine], alpha=0.6, color="steelblue")
    ax.plot(mid[mask_fine], density[mask_fine], color="steelblue", linewidth=0.8)
    ax.set_xlabel("Gap size (seconds, 0.1 s bins)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Gap density: 0–10 s  (0.1 s resolution)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    n_zero    = exact_counts[0]
    n_lt1     = int(np.sum(gaps < 1))
    n_lt10    = int(np.sum(gaps < 10))
    frac_zero = n_zero / len(gaps)
    fig.suptitle(
        f"Sub-10 s gaps: exact-0 s = {n_zero:,} ({frac_zero:.2%} of all gaps);  "
        f"<1 s = {n_lt1:,};  <10 s = {n_lt10:,}",
        fontsize=11,
    )
    fig.tight_layout()
    out = out_dir / "gap_subsecond.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    # ── Print key numbers ────────────────────────────────────────────────────
    print()
    print("Key n_sessions values at notable thresholds:")
    header = f"  {'gap':>10}  {'n_sessions':>12}  {'vs gap=0':>10}"
    print(header)
    n_at_0 = n_sessions[0]
    for sec, label in [(0, "0 s"), (1, "1 s"), (5, "5 s"), (30, "30 s"),
                       (300, "5 min"), (1200, "20 min"), (1800, "30 min"),
                       (2700, "45 min"), (5400, "90 min"), (7200, "2 h"),
                       (10800, "3 h"), (28800, "8 h"), (86400, "1 day")]:
        idx = np.searchsorted(thresholds, sec, side="left")
        idx = min(idx, len(n_sessions) - 1)
        ns = n_sessions[idx]
        print(f"  {label:>10}  {ns:>12,}  {100*(n_at_0 - ns)/n_at_0:>9.2f}% merged")


# ─────────────────────────── Main ────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",  default="data")
    p.add_argument("--out",   default="profiles")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gaps = load_gaps(
        ratings_csv=data_dir / "ml-20m" / "ratings.csv",
        processed_pkl=data_dir / "processed.pkl",
    )

    with open(data_dir / "processed.pkl", "rb") as f:
        proc = pickle.load(f)
    n_users = len(proc.user_id_to_idx)

    print("Sweeping thresholds …", flush=True)
    thresholds, n_sessions = sweep_thresholds(gaps, n_users)
    print(f"  {len(thresholds):,} threshold points")

    print("Generating plots …")
    make_plots(gaps, thresholds, n_sessions, n_users, out_dir)
    print(f"\nDone. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

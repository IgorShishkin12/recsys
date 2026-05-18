#!/usr/bin/env python3
"""Analyse 30-minute rating-session clusters in ML-20M.

A *session* is a maximal consecutive run of ratings from one user where no
gap between adjacent ratings exceeds SESSION_GAP_SEC (default 1800 s = 30 min).
Sessions ≥ 2 ratings are candidates for within-session shuffling as data
augmentation, because their internal order likely reflects *logging* order,
not *viewing* order.

Usage (from repo root):
    python -m scripts.analyze_sessions [--data data] [--out profiles] [--gap 1800]

Outputs (profiles/):
    session_size_hist.png      — histogram + CDF of session sizes
    session_stats.csv          — per-user session counts and sizes
    session_summary.txt        — printed summary table
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_GAP_SEC = 1800   # 30 minutes


# ─────────────────────────── Session detection ───────────────────────────────

def find_sessions(
    timestamps: np.ndarray,
    gap: int = SESSION_GAP_SEC,
) -> list[list[int]]:
    """Given a sorted array of timestamps for one user, return list of sessions.

    Each session is a list of *indices* into `timestamps`.
    """
    if len(timestamps) == 0:
        return []
    sessions: list[list[int]] = [[0]]
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] <= gap:
            sessions[-1].append(i)
        else:
            sessions.append([i])
    return sessions


# ─────────────────────────── Main analysis ───────────────────────────────────

def analyse(
    ratings_csv: Path,
    processed_pkl: Path,
    gap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        user_df  — one row per user: n_ratings, n_sessions, n_shuffle_candidates,
                   mean_session_size, max_session_size, frac_in_multi
        size_series — Series of all session sizes (one entry per session)
    """
    # Load valid user/item sets from pickle (same filter as training)
    with open(processed_pkl, "rb") as f:
        proc = pickle.load(f)
    valid_users = set(proc.user_id_to_idx.keys())   # original userId values
    valid_movies = set(proc.movie_id_to_idx.keys())  # original movieId values

    print(f"Loading {ratings_csv} …", flush=True)
    df = pd.read_csv(ratings_csv, usecols=["userId", "movieId", "timestamp"])
    print(f"  raw: {len(df):,} ratings")

    df = df[df["userId"].isin(valid_users) & df["movieId"].isin(valid_movies)]
    print(f"  after 5-core filter: {len(df):,} ratings, "
          f"{df['userId'].nunique():,} users")

    df = df.sort_values(["userId", "timestamp"], kind="stable")

    user_rows = []
    all_sizes: list[int] = []

    for uid, grp in df.groupby("userId", sort=False):
        ts = grp["timestamp"].to_numpy()
        sessions = find_sessions(ts, gap=gap)
        sizes = [len(s) for s in sessions]

        n_ratings = len(ts)
        n_sessions = len(sessions)
        n_multi = sum(1 for s in sizes if s >= 2)       # sessions with ≥2 ratings
        n_in_multi = sum(s for s in sizes if s >= 2)    # ratings inside multi-rating sessions

        user_rows.append({
            "userId":           uid,
            "n_ratings":        n_ratings,
            "n_sessions":       n_sessions,
            "n_multi_sessions": n_multi,
            "max_session_size": max(sizes),
            "mean_session_size": round(float(np.mean(sizes)), 3),
            "frac_in_multi":    round(n_in_multi / n_ratings, 4),
        })
        all_sizes.extend(sizes)

    user_df = pd.DataFrame(user_rows)
    size_series = pd.Series(all_sizes, name="session_size")
    return user_df, size_series


# ─────────────────────────── Reporting ───────────────────────────────────────

def print_summary(
    user_df: pd.DataFrame,
    size_series: pd.Series,
    gap: int,
    out_txt: Path,
) -> None:
    lines = []

    total_sessions  = len(size_series)
    total_ratings   = size_series.sum()
    multi_mask      = size_series >= 2
    n_multi         = multi_mask.sum()
    ratings_in_multi = size_series[multi_mask].sum()

    lines += [
        f"SESSION ANALYSIS  (gap ≤ {gap} s = {gap//60} min)",
        "=" * 56,
        f"Users analysed          : {len(user_df):>10,}",
        f"Total ratings           : {total_ratings:>10,}",
        f"Total sessions          : {total_sessions:>10,}",
        f"  single-rating         : {(~multi_mask).sum():>10,}  "
          f"({100*(~multi_mask).mean():.1f}%)",
        f"  multi-rating (≥2)     : {n_multi:>10,}  "
          f"({100*multi_mask.mean():.1f}%)",
        f"Ratings inside multi-sessions: {ratings_in_multi:>6,}  "
          f"({100*ratings_in_multi/total_ratings:.1f}% of all ratings)",
        "",
        "Session-size distribution:",
        f"  {'size':>6}  {'count':>10}  {'% sessions':>11}  {'% ratings':>10}  cumul%sessions",
    ]
    cumul = 0.0
    size_counts = Counter(size_series.tolist())
    for sz in sorted(size_counts.keys()):
        if sz > 20 and sz not in (25, 30, 50, 100) and sz != max(size_counts):
            continue
        cnt = size_counts[sz]
        pct_s = 100 * cnt / total_sessions
        pct_r = 100 * cnt * sz / total_ratings
        cumul += pct_s
        lines.append(
            f"  {sz:>6}  {cnt:>10,}  {pct_s:>10.2f}%  {pct_r:>9.2f}%  {cumul:>10.1f}%"
        )

    lines += [
        "",
        "Per-user frac_in_multi (fraction of ratings inside a multi-rating session):",
        f"  mean  : {user_df['frac_in_multi'].mean():.3f}",
        f"  median: {user_df['frac_in_multi'].median():.3f}",
        f"  p90   : {user_df['frac_in_multi'].quantile(0.90):.3f}",
        f"  p99   : {user_df['frac_in_multi'].quantile(0.99):.3f}",
        "",
        "Users where >50% of ratings are inside multi-sessions: "
          f"{(user_df['frac_in_multi'] > 0.5).sum():,} "
          f"({100*(user_df['frac_in_multi'] > 0.5).mean():.1f}%)",
        "Users where >90% of ratings are inside multi-sessions: "
          f"{(user_df['frac_in_multi'] > 0.9).sum():,} "
          f"({100*(user_df['frac_in_multi'] > 0.9).mean():.1f}%)",
    ]

    text = "\n".join(lines)
    print(text)
    out_txt.write_text(text + "\n", encoding="utf-8")
    print(f"\n  → {out_txt}")


# ─────────────────────────── Plotting ────────────────────────────────────────

def make_plots(size_series: pd.Series, gap: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = size_series.values
    max_sz = int(sizes.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── (1) linear-y bar chart up to size 15 ─────────────────────────────────
    ax = axes[0]
    cap = 15
    cnt = Counter(sizes.tolist())
    xs  = list(range(1, cap + 1))
    ys  = [cnt.get(x, 0) for x in xs]
    overflow = sum(v for k, v in cnt.items() if k > cap)
    if overflow:
        xs.append(cap + 1)
        ys.append(overflow)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in range(1, cap + 1)] + [f">{cap}"], fontsize=8)
    ax.bar(xs[:len(ys)], ys, color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Session size (# ratings)", fontsize=11)
    ax.set_ylabel("Number of sessions", fontsize=11)
    ax.set_title(f"Session size distribution\n(gap ≤ {gap//60} min)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # ── (2) log-log tail ──────────────────────────────────────────────────────
    ax = axes[1]
    sz_sorted = np.array(sorted(cnt.keys()))
    sz_counts = np.array([cnt[k] for k in sz_sorted])
    ax.scatter(sz_sorted, sz_counts, s=12, color="steelblue", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Session size (log)", fontsize=11)
    ax.set_ylabel("Number of sessions (log)", fontsize=11)
    ax.set_title("Session size tail (log-log)", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)

    # ── (3) CDF of ratings covered vs session-size threshold ─────────────────
    ax = axes[2]
    total = sizes.sum()
    thresholds = np.arange(1, min(max_sz + 1, 201))
    # frac of ratings in sessions of size >= t
    frac_covered = np.array([
        sizes[sizes >= t].sum() / total for t in thresholds
    ])
    ax.plot(thresholds, frac_covered, color="steelblue", linewidth=1.8)
    ax.axhline(0.5,  color="tomato", linestyle="--", linewidth=0.9, label="50% of ratings")
    ax.axhline(0.25, color="orange", linestyle="--", linewidth=0.9, label="25% of ratings")
    for thresh, label in [(2, "≥2"), (3, "≥3"), (5, "≥5"), (10, "≥10")]:
        frac = sizes[sizes >= thresh].sum() / total
        ax.annotate(f"{label}: {frac:.1%}",
                    xy=(thresh, frac),
                    xytext=(thresh + 1, frac + 0.02),
                    fontsize=8, color="grey")
    ax.set_xlabel("Min session size threshold", fontsize=11)
    ax.set_ylabel("Fraction of ratings in sessions ≥ threshold", fontsize=11)
    ax.set_title("Coverage: ratings in sessions of size ≥ N", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"ML-20M rating-session analysis  (session gap ≤ {gap//60} min)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = out_dir / "session_size_hist.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    # ── (4) per-user frac_in_multi histogram ─────────────────────────────────
    # (requires user_df — handled by caller passing it in)


def make_user_plot(user_df: pd.DataFrame, gap: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(user_df["frac_in_multi"], bins=50, color="steelblue",
            edgecolor="none", alpha=0.85)
    ax.set_xlabel("Fraction of user's ratings inside multi-rating sessions", fontsize=11)
    ax.set_ylabel("Number of users", fontsize=11)
    ax.set_title(f"Per-user fraction of ratings in sessions ≥2\n(gap ≤ {gap//60} min)",
                 fontsize=11, fontweight="bold")
    ax.axvline(user_df["frac_in_multi"].median(), color="tomato",
               linestyle="--", linewidth=1.2,
               label=f"median={user_df['frac_in_multi'].median():.2f}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.scatter(user_df["n_ratings"], user_df["frac_in_multi"],
               s=2, alpha=0.15, color="steelblue")
    ax.set_xscale("log")
    ax.set_xlabel("User's total number of ratings (log)", fontsize=11)
    ax.set_ylabel("Fraction of ratings in multi-rating sessions", fontsize=11)
    ax.set_title("Batch-logging tendency vs activity level",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out = out_dir / "session_user_dist.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────── Main ────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",    default="data",
                   help="root data dir containing ml-20m/ and processed.pkl")
    p.add_argument("--out",     default="profiles")
    p.add_argument("--gap",     type=int, default=SESSION_GAP_SEC,
                   help="max gap in seconds to stay in the same session (default 1800)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_df, size_series = analyse(
        ratings_csv=data_dir / "ml-20m" / "ratings.csv",
        processed_pkl=data_dir / "processed.pkl",
        gap=args.gap,
    )

    print()
    print_summary(user_df, size_series, gap=args.gap,
                  out_txt=out_dir / "session_summary.txt")

    # Save per-user CSV
    csv_path = out_dir / "session_stats.csv"
    user_df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    print()
    print("Generating plots …")
    make_plots(size_series, gap=args.gap, out_dir=out_dir)
    make_user_plot(user_df, gap=args.gap, out_dir=out_dir)

    print(f"\nDone. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

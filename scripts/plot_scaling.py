#!/usr/bin/env python3
"""Plot inference scaling laws (time & memory) for all models.

Two sweeps:
  batch-size sweep  — vary B at fixed seq_len=200 (model built once per model)
  seq-len sweep     — vary L at balanced batch size B = prev_pow2(TOKEN_BUDGET // L)
                      (model rebuilt each point so max_len=L; OOM triggers B//2 retry)

Both sweeps report raw totals and per-sample (÷B) values.
Seq-len plots include a power-law fit (polyfit in log-log space) per model.

Usage (from repo root):
    # full run (both sweeps):
    python -m scripts.plot_scaling [--out profiles] [--device cuda]

    # re-plot from saved CSVs without re-sweeping:
    python -m scripts.plot_scaling \\
        --from-csv profiles/scaling_results.csv \\
        --from-seqlen-csv profiles/scaling_seqlen_results.csv

    # seq-len sweep only:
    python -m scripts.plot_scaling --seqlen-only

Outputs (profiles/):
    scaling_time_bs.png / scaling_mem_bs.png
    scaling_time_per_sample_bs.png / scaling_mem_per_sample_bs.png
    scaling_results.csv

    scaling_time_sl.png / scaling_mem_sl.png   ← per-sample, with power-law fits
    scaling_seqlen_results.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import build_model
from scripts.profile_models import MODEL_SPECS   # reuse config dict

# ─────────────────────────── Sweep parameters ────────────────────────────────

BATCH_SIZES   = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                 2048, 4096, 8192, 16384, 32768]
N_ITEMS       = 27279
FIXED_LEN     = 200
N_WARMUP      = 10
N_MEASURE     = 50

# Seq-len sweep.
# TOKEN_BUDGET  ≈ 1/3 of an 80 GB A100 worth of token activations.
# B = prev_pow2(TOKEN_BUDGET // L), so B × L ≈ TOKEN_BUDGET (rounded down).
# OOM recovery: halve B and retry.
SEQ_LENS      = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
TOKEN_BUDGET  = 262144   # 2^18; at L=256 → B=1024, at L=4096 → B=64


def _prev_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def balanced_bs(L: int) -> int:
    return _prev_pow2(TOKEN_BUDGET // L)


# ─────────────────────────── Measurement ─────────────────────────────────────

def measure(
    model: nn.Module,
    B: int,
    device: torch.device,
    seq_len: int = FIXED_LEN,
) -> Tuple[float, float]:
    """Return (time_ms, peak_mem_mb) for one (model, B, seq_len) point."""
    input_ids = torch.randint(1, N_ITEMS, (B, seq_len), device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(N_WARMUP):
            model.score_all(input_ids)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(N_MEASURE):
            model.score_all(input_ids)
    end.record()
    torch.cuda.synchronize(device)

    time_ms = start.elapsed_time(end) / N_MEASURE
    mem_mb  = torch.cuda.max_memory_allocated(device) / 1e6
    return time_ms, mem_mb


def sweep_seqlen(
    name: str,
    spec: dict,
    device: torch.device,
) -> Dict[int, Optional[Tuple[float, float, int]]]:
    """Sweep seq_len with balanced (and OOM-adaptive) batch size.

    Returns {L: (time_ms, mem_mb, actual_B)}.
    On OOM the batch size is halved and retried; None stored if B reaches 0.
    """
    out: Dict[int, Optional[Tuple[float, float, int]]] = {}
    for L in SEQ_LENS:
        B = balanced_bs(L)
        model = None
        success = False
        while B >= 1:
            try:
                cfg = copy.deepcopy(spec["model_cfg"])  # cfg is a dict
                cfg["max_len"] = L                       # update seq-len budget
                model = build_model(cfg, n_items=N_ITEMS).to(device)
                t, m = measure(model, B, device, seq_len=L)
                out[L] = (t, m, B)
                tag = "" if B == balanced_bs(L) else f"  (reduced from {balanced_bs(L)})"
                print(f"  L={L:<5}  B={B:<5}  {t:7.2f} ms  {m:7.0f} MB{tag}", flush=True)
                success = True
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if model is not None:
                    del model
                    model = None
                print(f"  L={L:<5}  B={B:<5}  OOM → trying B={B//2}", flush=True)
                B //= 2
            except Exception as exc:
                print(f"  L={L:<5}  B={B:<5}  ERROR: {exc}", flush=True)
                break
            finally:
                if model is not None:
                    del model
                    model = None
                torch.cuda.empty_cache()

        if not success:
            out[L] = None
            print(f"  L={L:<5}  failed (all batch sizes OOM)", flush=True)
    return out


# ─────────────────────────── Plotting helpers ────────────────────────────────

def _ref_line(
    ax,
    xs: List[float],
    ys_data: List[List[Optional[float]]],
    fn: Callable[[float], float],
    label: str,
    color: str = "grey",
    linestyle: str = "--",
) -> None:
    """Reference line shaped like fn(x), anchored to median of valid data."""
    import numpy as np
    all_valid = [y for row in ys_data for y in row if y is not None]
    if not all_valid:
        return
    anchor_y = float(np.median(all_valid))
    anchor_x = xs[len(xs) // 2]
    ref_ys = [anchor_y * fn(x) / fn(anchor_x) for x in xs]
    ax.plot(xs, ref_ys, color=color, linestyle=linestyle, linewidth=1.0,
            alpha=0.55, label=label, zorder=0)


def _power_fit(
    xs: List[float],
    ys: List[float],
) -> Tuple[float, float]:
    """Fit y = C * x^p in log2-log2 space; return (p, C)."""
    import numpy as np
    lx = np.log2(xs)
    ly = np.log2(ys)
    p, lc = np.polyfit(lx, ly, 1)
    return float(p), float(2 ** lc)


def _finish_ax(ax, xs, ylabel, title, xlabel):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, fontsize=9)


# ─────────────────────────── Batch-size plots ────────────────────────────────

def make_plots(
    results: Dict[str, Dict[int, Optional[Tuple[float, float]]]],
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models  = list(results.keys())
    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    xs = BATCH_SIZES

    for metric, per_sample, ylabel, fname in [
        ("time", False, "Inference time (ms / step)",              "scaling_time_bs.png"),
        ("mem",  False, "Peak GPU memory (MB)",                    "scaling_mem_bs.png"),
        ("time", True,  "Inference time per sample (ms / sample)", "scaling_time_per_sample_bs.png"),
        ("mem",  True,  "Peak GPU memory per sample (MB / sample)","scaling_mem_per_sample_bs.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        all_ys: List[List[Optional[float]]] = []

        for i, name in enumerate(models):
            ys: List[Optional[float]] = []
            last_valid_x = last_valid_y = None

            for B in xs:
                pt = results[name].get(B)
                if pt is None:
                    ys.append(None)
                else:
                    v = pt[0] if metric == "time" else pt[1]
                    if per_sample:
                        v = v / B
                    ys.append(v)
                    if v is not None:
                        last_valid_x, last_valid_y = B, v

            valid_x = [x for x, y in zip(xs, ys) if y is not None]
            valid_y = [y for y in ys if y is not None]
            if valid_x:
                ax.plot(valid_x, valid_y,
                        color=colors[i % len(colors)],
                        marker=markers[i % len(markers)],
                        linewidth=1.8, markersize=6, label=name)

            if last_valid_x is not None and ys[-1] is None:
                ax.annotate("↑OOM",
                            xy=(last_valid_x, last_valid_y),
                            xytext=(0, 10), textcoords="offset points",
                            ha="center", fontsize=8,
                            color=colors[i % len(colors)])
            all_ys.append(ys)

        _ref_line(ax, xs, all_ys, fn=lambda x: x,    label="O(n)")
        _ref_line(ax, xs, all_ys, fn=lambda x: x**2, label="O(n²)")
        _finish_ax(ax, xs, ylabel,
                   f"Inference scaling — {ylabel}\n"
                   f"(seq_len={FIXED_LEN}, n_items={N_ITEMS}, {N_MEASURE} steps avg)",
                   xlabel="Batch size")
        fig.tight_layout()
        out = out_dir / fname
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")


# ─────────────────────────── Seq-len plots ───────────────────────────────────

def make_seqlen_plots(
    results_sl: Dict[str, Dict[int, Optional[Tuple[float, float, int]]]],
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models  = list(results_sl.keys())
    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    xs = SEQ_LENS

    for metric_idx, ylabel, fname in [
        (0, "Inference time per sample (ms / sample)",  "scaling_time_sl.png"),
        (1, "Peak GPU memory per sample (MB / sample)", "scaling_mem_sl.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6))
        all_ys: List[List[Optional[float]]] = []

        for i, name in enumerate(models):
            ys: List[Optional[float]] = []
            last_valid_x = last_valid_y = None

            for L in xs:
                pt = results_sl[name].get(L)
                if pt is None:
                    ys.append(None)
                else:
                    raw_val = pt[metric_idx]
                    B = pt[2]
                    v = raw_val / B
                    ys.append(v)
                    if v is not None:
                        last_valid_x, last_valid_y = L, v

            valid_x = [x for x, y in zip(xs, ys) if y is not None]
            valid_y = [y for y in ys if y is not None]

            # Power-law fit (needs ≥2 points)
            slope_label = ""
            if len(valid_x) >= 2:
                p, C = _power_fit(valid_x, valid_y)
                slope_label = f"  ∝L^{p:.2f}"
                # Trend line (dotted, same colour)
                trend_ys = [C * L ** p for L in xs]
                ax.plot(xs, trend_ys,
                        color=colors[i % len(colors)],
                        linestyle=":", linewidth=0.9, alpha=0.55, zorder=1)

            if valid_x:
                ax.plot(valid_x, valid_y,
                        color=colors[i % len(colors)],
                        marker=markers[i % len(markers)],
                        linewidth=1.8, markersize=6,
                        label=f"{name}{slope_label}")

            if last_valid_x is not None and ys[-1] is None:
                ax.annotate("↑OOM",
                            xy=(last_valid_x, last_valid_y),
                            xytext=(0, 10), textcoords="offset points",
                            ha="center", fontsize=8,
                            color=colors[i % len(colors)])
            all_ys.append(ys)

        # Reference lines for O(L), O(L log L), O(L²)
        _ref_line(ax, xs, all_ys, fn=lambda x: x,                        label="O(L)")
        _ref_line(ax, xs, all_ys, fn=lambda x: x * math.log2(max(x, 2)), label="O(L log L)")
        _ref_line(ax, xs, all_ys, fn=lambda x: x ** 2,                   label="O(L²)")

        # Footnote: actual B used per L
        bs_note = "  |  ".join(f"L={L}→B≤{balanced_bs(L)}" for L in xs)
        _finish_ax(ax, xs, ylabel,
                   f"Seq-len scaling (per sample) — {ylabel}\n"
                   f"B×L≈{TOKEN_BUDGET} const, OOM→B//2  ({N_MEASURE} steps avg)",
                   xlabel="Sequence length")
        ax.text(0.01, 0.01, bs_note, transform=ax.transAxes,
                fontsize=7, color="grey", va="bottom")

        fig.tight_layout()
        out = out_dir / fname
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")


# ─────────────────────────── CSV helpers ─────────────────────────────────────

def _load_bs_csv(csv_path: Path) -> Dict[str, Dict[int, Tuple[float, float]]]:
    results: Dict[str, Dict[int, Tuple[float, float]]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            results.setdefault(row["model"], {})[int(row["batch_size"])] = (
                float(row["time_ms"]), float(row["mem_mb"]))
    return results


def _load_seqlen_csv(csv_path: Path) -> Dict[str, Dict[int, Tuple[float, float, int]]]:
    results: Dict[str, Dict[int, Tuple[float, float, int]]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            results.setdefault(row["model"], {})[int(row["seq_len"])] = (
                float(row["time_ms"]), float(row["mem_mb"]), int(row["batch_size"]))
    return results


# ─────────────────────────── Main ────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out",             default="profiles")
    p.add_argument("--device",          default="cuda")
    p.add_argument("--models",          nargs="*", default=None)
    p.add_argument("--from-csv",        metavar="CSV",
                   help="Skip batch-size sweep; load CSV and replot.")
    p.add_argument("--from-seqlen-csv", metavar="CSV",
                   help="Skip seq-len sweep; load CSV and replot.")
    p.add_argument("--seqlen-only",     action="store_true",
                   help="Run only the seq-len sweep.")
    p.add_argument("--bs-only",         action="store_true",
                   help="Run only the batch-size sweep.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    if not (args.from_csv and args.from_seqlen_csv):
        print(f"Device : {device}", flush=True)
        if device.type == "cuda":
            print(f"GPU    : {torch.cuda.get_device_name(device)}", flush=True)
        print()

    selected = args.models or list(MODEL_SPECS.keys())

    # ── batch-size sweep ──────────────────────────────────────────────────────
    if not args.seqlen_only:
        if args.from_csv:
            print(f"Loading batch-size results from {args.from_csv} …", flush=True)
            bs_results = _load_bs_csv(Path(args.from_csv))
            print(f"  {sum(len(v) for v in bs_results.values())} points, "
                  f"{len(bs_results)} models")
        else:
            print(f"── Batch-size sweep  (seq_len={FIXED_LEN}) ──", flush=True)
            print(f"   Models : {selected}", flush=True)
            print(f"   Sizes  : {BATCH_SIZES}", flush=True)
            print(f"   Timing : warmup={N_WARMUP}  measure={N_MEASURE}", flush=True)
            print()

            bs_results: Dict[str, Dict[int, Optional[Tuple[float, float]]]] = {}
            csv_rows: List[Dict] = []

            for name in selected:
                spec = MODEL_SPECS.get(name)
                if spec is None:
                    print(f"[skip] unknown: {name}")
                    continue
                print(f"  {name}", flush=True)
                try:
                    model = build_model(spec["model_cfg"], n_items=N_ITEMS).to(device)
                except Exception as exc:
                    print(f"    [SKIP] build failed: {exc}")
                    continue

                bs_results[name] = {}
                for B in BATCH_SIZES:
                    try:
                        t, m = measure(model, B, device)
                        bs_results[name][B] = (t, m)
                        print(f"    B={B:<5}  {t:7.2f} ms  {m:7.0f} MB", flush=True)
                        csv_rows.append({"model": name, "batch_size": B,
                                         "time_ms": round(t, 4), "mem_mb": round(m, 1)})
                    except torch.cuda.OutOfMemoryError:
                        bs_results[name][B] = None
                        print(f"    B={B:<5}  OOM", flush=True)
                        torch.cuda.empty_cache()
                    except Exception as exc:
                        bs_results[name][B] = None
                        print(f"    B={B:<5}  ERROR: {exc}", flush=True)

                del model
                torch.cuda.empty_cache()
                print()

            csv_path = out_dir / "scaling_results.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["model", "batch_size", "time_ms", "mem_mb"])
                w.writeheader()
                w.writerows(csv_rows)
            print(f"  → {csv_path}")

        print("Generating batch-size plots …", flush=True)
        try:
            make_plots(bs_results, out_dir)
        except ImportError:
            print("[warn] matplotlib not available — skipping plots")
        print()

    # ── seq-len sweep ─────────────────────────────────────────────────────────
    if not args.bs_only:
        if args.from_seqlen_csv:
            print(f"Loading seq-len results from {args.from_seqlen_csv} …", flush=True)
            sl_results = _load_seqlen_csv(Path(args.from_seqlen_csv))
            print(f"  {sum(len(v) for v in sl_results.values())} points, "
                  f"{len(sl_results)} models")
        else:
            print(f"── Seq-len sweep  (B×L≈{TOKEN_BUDGET} const) ──", flush=True)
            bs_at_L = "  ".join(f"L={L}→B≤{balanced_bs(L)}" for L in SEQ_LENS)
            print(f"   {bs_at_L}", flush=True)
            print(f"   Timing : warmup={N_WARMUP}  measure={N_MEASURE}", flush=True)
            print()

            sl_results: Dict[str, Dict[int, Optional[Tuple[float, float, int]]]] = {}
            sl_csv_rows: List[Dict] = []

            for name in selected:
                spec = MODEL_SPECS.get(name)
                if spec is None:
                    continue
                print(f"  {name}", flush=True)
                pts = sweep_seqlen(name, spec, device)
                sl_results[name] = pts
                for L, val in pts.items():
                    if val is not None:
                        t, m, B = val
                        sl_csv_rows.append({"model": name, "seq_len": L, "batch_size": B,
                                            "time_ms": round(t, 4), "mem_mb": round(m, 1)})
                print()

            sl_csv_path = out_dir / "scaling_seqlen_results.csv"
            with open(sl_csv_path, "w", newline="") as f:
                w = csv.DictWriter(f,
                    fieldnames=["model", "seq_len", "batch_size", "time_ms", "mem_mb"])
                w.writeheader()
                w.writerows(sl_csv_rows)
            print(f"  → {sl_csv_path}")

        print("Generating seq-len plots …", flush=True)
        try:
            make_seqlen_plots(sl_results, out_dir)
        except ImportError:
            print("[warn] matplotlib not available — skipping plots")
        print()

    print(f"Done. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

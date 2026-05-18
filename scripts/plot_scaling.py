#!/usr/bin/env python3
"""Plot inference scaling laws (time & memory) vs batch size for all models.

Sweeps batch_size ∈ {1,2,4,8,16,32,64,128,256,512,1024} at fixed seq_len=200
using CUDA Events for precise per-step timing and torch.cuda peak memory tracking.

Usage (from repo root):
    python -m scripts.plot_scaling [--out profiles] [--device cuda]

Outputs:
    profiles/scaling_time_bs.png   — inference time (ms) vs batch size, log-log
    profiles/scaling_mem_bs.png    — peak GPU memory (MB) vs batch size, log-log
    profiles/scaling_results.csv   — raw numbers
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import build_model
from scripts.profile_models import MODEL_SPECS   # reuse config dict

# ─────────────────────────── Sweep parameters ────────────────────────────────

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
N_ITEMS     = 27279
FIXED_LEN   = 200
N_WARMUP    = 10
N_MEASURE   = 50

# ─────────────────────────── Measurement ─────────────────────────────────────

def measure(
    model: nn.Module,
    B: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (time_ms, peak_mem_mb) for one (model, batch_size) point."""
    input_ids = torch.randint(1, N_ITEMS, (B, FIXED_LEN), device=device)
    model.eval()

    # Warmup — also warms the cuFFT planner, cuBLAS workspace, etc.
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


# ─────────────────────────── Plotting ────────────────────────────────────────

def _ref_line(
    ax,
    xs: List[int],
    ys_data: List[List[Optional[float]]],
    order: int,
    label: str,
) -> None:
    """Draw an O(n^order) reference line anchored to the median of valid data."""
    import numpy as np

    all_valid = [y for row in ys_data for y in row if y is not None]
    if not all_valid:
        return
    anchor_y = float(np.median(all_valid))
    anchor_x = xs[len(xs) // 2]

    ref_ys = [anchor_y * (x / anchor_x) ** order for x in xs]
    ax.plot(xs, ref_ys, color="grey", linestyle="--", linewidth=1.0,
            alpha=0.55, label=label, zorder=0)


def make_plots(
    results: Dict[str, Dict[int, Tuple[Optional[float], Optional[float]]]],
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

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
                        linewidth=1.8, markersize=6,
                        label=name)

            # OOM annotation at last valid point
            if last_valid_x is not None and ys[-1] is None:
                ax.annotate("↑OOM",
                            xy=(last_valid_x, last_valid_y),
                            xytext=(0, 10), textcoords="offset points",
                            ha="center", fontsize=8,
                            color=colors[i % len(colors)])

            all_ys.append(ys)

        # Reference lines
        _ref_line(ax, xs, all_ys, order=1, label="O(n)")
        _ref_line(ax, xs, all_ys, order=2, label="O(n²)")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Batch size", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"Inference scaling — {ylabel}\n"
            f"(seq_len={FIXED_LEN}, n_items={N_ITEMS}, {N_MEASURE} steps avg)",
            fontsize=12, fontweight="bold",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0, fontsize=9)

        fig.tight_layout()
        out = out_dir / fname
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")


# ─────────────────────────── Main ────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",       default="profiles")
    p.add_argument("--device",    default="cuda")
    p.add_argument("--models",    nargs="*", default=None)
    p.add_argument("--from-csv",  metavar="CSV",
                   help="Skip sweep; load existing scaling_results.csv and regenerate plots only.")
    return p.parse_args()


def _load_csv(csv_path: Path) -> Dict[str, Dict[int, Optional[Tuple[float, float]]]]:
    """Reconstruct results dict from a previously saved CSV."""
    results: Dict[str, Dict[int, Optional[Tuple[float, float]]]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["model"]
            B    = int(row["batch_size"])
            results.setdefault(name, {})[B] = (float(row["time_ms"]), float(row["mem_mb"]))
    return results


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        csv_path = Path(args.from_csv)
        print(f"Loading results from {csv_path} …", flush=True)
        results = _load_csv(csv_path)
        print(f"  {sum(len(v) for v in results.values())} data points across {len(results)} models")
    else:
        device = torch.device(
            args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        selected = args.models or list(MODEL_SPECS.keys())

        print(f"Device : {device}", flush=True)
        if device.type == "cuda":
            print(f"GPU    : {torch.cuda.get_device_name(device)}", flush=True)
        print(f"Models : {selected}", flush=True)
        print(f"Sweep  : batch_size={BATCH_SIZES}  seq_len={FIXED_LEN}", flush=True)
        print(f"Timing : warmup={N_WARMUP}  measure={N_MEASURE}", flush=True)
        print()

        # results[model_name][batch_size] = (time_ms, mem_mb) or None on OOM
        results: Dict[str, Dict[int, Optional[Tuple[float, float]]]] = {}
        csv_rows: List[Dict] = []

        for name in selected:
            spec = MODEL_SPECS.get(name)
            if spec is None:
                print(f"[skip] unknown: {name}")
                continue

            print(f"── {name} ──", flush=True)
            try:
                model = build_model(spec["model_cfg"], n_items=N_ITEMS).to(device)
            except Exception as exc:
                print(f"  [SKIP] build failed: {exc}")
                continue

            results[name] = {}

            for B in BATCH_SIZES:
                try:
                    t, m = measure(model, B, device)
                    results[name][B] = (t, m)
                    print(f"  B={B:<5}  {t:7.2f} ms  {m:7.0f} MB", flush=True)
                    csv_rows.append({"model": name, "batch_size": B,
                                     "time_ms": round(t, 4), "mem_mb": round(m, 1)})
                except torch.cuda.OutOfMemoryError:
                    results[name][B] = None
                    print(f"  B={B:<5}  OOM", flush=True)
                    torch.cuda.empty_cache()
                except Exception as exc:
                    results[name][B] = None
                    print(f"  B={B:<5}  ERROR: {exc}", flush=True)

            del model
            torch.cuda.empty_cache()
            print()

        # Save CSV
        csv_path = out_dir / "scaling_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "batch_size", "time_ms", "mem_mb"])
            w.writeheader()
            w.writerows(csv_rows)
        print(f"  → {csv_path}")

    # Make plots
    print("Generating plots …", flush=True)
    try:
        make_plots(results, out_dir)
    except ImportError:
        print("[warn] matplotlib not available — skipping plots")

    print(f"\nDone. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

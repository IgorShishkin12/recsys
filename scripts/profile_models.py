#!/usr/bin/env python3
"""Profile all sequential recommendation models using torch.profiler.

Generates SVG flame graphs (CPU + CUDA) and summary PNG charts for every
model × {train step, inference}, so we can compare architectures on time
and memory.

Usage (from repo root):
    python -m scripts.profile_models [--out profiles] [--device cuda]
    python -m scripts.profile_models --models sasrec fmlp --out profiles

Outputs per model:
    profiles/<model>/train_cpu_flamegraph.svg
    profiles/<model>/train_cuda_flamegraph.svg
    profiles/<model>/train_cpu_stacks.txt
    profiles/<model>/train_cuda_stacks.txt
    profiles/<model>/train_trace.json       (chrome trace, large — gitignored)
    profiles/<model>/infer_cpu_flamegraph.svg
    profiles/<model>/infer_cuda_flamegraph.svg
    profiles/<model>/infer_cpu_stacks.txt
    profiles/<model>/infer_cuda_stacks.txt
    profiles/<model>/infer_trace.json

Summary outputs:
    profiles/summary_train.png
    profiles/summary_infer.png
    profiles/top_ops_train.png
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

# Make sure repo root is on the path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import build_model

# ─────────────────────────── Model specifications ────────────────────────────
# Params taken from configs/*.yaml; dropout=0 for profiling so numerics don't
# add noise to the timing (dropout is a no-op at eval; for train we want the
# clean kernel path).

N_ITEMS = 27279   # approximate ML-20M item count
MAX_LEN = 200
N_NEG   = 64      # negatives per train step (paper uses 256; 64 is sufficient
                  # to exercise the einsum without ballooning memory)
N_WARMUP = 5      # GPU warm-up steps (not profiled)
N_ACTIVE = 20     # steps included in the profiler trace
INFER_BS = 256    # fixed inference batch for all models

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "sasrec": {
        "model_cfg": {
            "name": "sasrec",
            "d": 256, "n_blocks": 3, "n_heads": 4, "max_len": MAX_LEN,
            "dropout": 0.0, "ffn_mult": 4,
            "use_rope": True, "use_ligr_gates": False, "use_side": False,
        },
        "train_bs": 256,
    },
    "sasrec_baseline": {
        "model_cfg": {
            "name": "sasrec",
            "d": 256, "n_blocks": 2, "n_heads": 1, "max_len": MAX_LEN,
            "dropout": 0.0, "ffn_mult": 4,
            "use_rope": False, "use_ligr_gates": False, "use_side": False,
        },
        "train_bs": 128,
    },
    "nextitnet": {
        "model_cfg": {
            "name": "nextitnet",
            "d": 128, "kernel_size": 3, "block_num": 2,
            "dilations": [1, 2, 4, 8], "dropout": 0.0,
            "max_len": MAX_LEN, "use_glu": True, "use_side": False,
        },
        "train_bs": 256,
    },
    "fmlp": {
        "model_cfg": {
            "name": "fmlp",
            "d": 128, "n_blocks": 4, "max_len": MAX_LEN,
            "dropout": 0.0, "ffn_mult": 4, "use_side": False,
        },
        "train_bs": 256,
    },
    "linear_attn": {
        "model_cfg": {
            "name": "linear_attn",
            "d": 128, "n_blocks": 3, "n_heads": 4, "max_len": MAX_LEN,
            "dropout": 0.0, "ffn_mult": 4,
            "use_rope": True, "use_ligr_gates": True, "use_side": False,
        },
        # Halved from config default: kv outer product is [B,H,L,hd,hd]
        # which is ~4 GB at bs=256 — fine on 80 GB A100, but conservative.
        "train_bs": 64,
    },
    "fnet_hybrid": {
        "model_cfg": {
            "name": "fnet_hybrid",
            "d": 128, "n_blocks": 4, "n_attn_top": 2, "n_heads": 4,
            "max_len": MAX_LEN, "dropout": 0.0, "ffn_mult": 4,
            "use_rope": True, "use_ligr_gates": True, "use_side": False,
        },
        "train_bs": 256,
    },
    "causal_fftconv": {
        "model_cfg": {
            "name": "causal_fftconv",
            "d": 256, "n_blocks": 3, "max_len": MAX_LEN,
            "dropout": 0.0, "ffn_mult": 4, "use_side": False,
        },
        "train_bs": 256,
    },
}


# ──────────────────────────── Flame-graph SVG ─────────────────────────────────

def _frame_label(raw: str) -> str:
    """Extract a short readable name from a torch-profiler stack frame.

    Frames may be:
    - PyTorch op names:  "aten::mm", "autograd::engine::evaluate_function"
    - Python locations:  "/path/to/file.py(42):func_name"
    - CUDA kernels:      "volta_gemm_..."
    We keep just the last colon-separated token, or the whole string if no colon.
    """
    raw = raw.strip()
    if "(" in raw and ")" in raw:
        # Python location — take the part after the closing paren
        after = raw.split(")")[-1].lstrip(":")
        return after if after else raw
    if "::" in raw:
        return raw   # aten-style, keep as-is (short enough)
    if ":" in raw:
        return raw.rsplit(":", 1)[-1]
    return raw


def _warm_color(name: str) -> str:
    """Deterministic warm hue (red-orange-yellow) from function name."""
    digest = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    hue = (digest % 60) / 360.0          # 0–60° = red → yellow
    sat = 0.55 + (digest % 90) / 300.0
    val = 0.78 + (digest % 50) / 250.0
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def render_flamegraph_svg(
    stacks_path: str,
    title: str,
    output_path: str,
    width: int = 1400,
) -> None:
    """Parse a folded-stacks file and emit a self-contained SVG flame graph."""
    p = Path(stacks_path)
    if not p.exists() or p.stat().st_size == 0:
        print(f"    [warn] stacks empty/missing: {stacks_path}")
        return

    # ── Parse ──────────────────────────────────────────────────────────────
    entries: List[Tuple[List[str], int]] = []
    total_count = 0
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Format: "frame1;frame2;...;frameN count_us"
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            try:
                count = int(float(parts[1]))
            except ValueError:
                continue
            frames = [_frame_label(f) for f in parts[0].split(";") if f.strip()]
            if frames:
                entries.append((frames, count))
                total_count += count

    if total_count == 0:
        print(f"    [warn] stacks file has no data: {stacks_path}")
        return

    # ── Build tree ─────────────────────────────────────────────────────────
    def _node():
        return {"ch": {}, "total": 0, "self_t": 0, "_name": ""}

    root = _node()
    root["total"] = total_count
    for frames, count in entries:
        cur = root
        for i, name in enumerate(frames):
            if name not in cur["ch"]:
                cur["ch"][name] = _node()
                cur["ch"][name]["_name"] = name
            cur = cur["ch"][name]
            cur["total"] += count
            if i == len(frames) - 1:
                cur["self_t"] += count

    # ── Layout (BFS) ───────────────────────────────────────────────────────
    FRAME_H  = 19
    FONT_SZ  = 11
    HDR_H    = 54   # pixels reserved for the title block
    FOOT_H   = 10

    rects: List[Dict] = []
    queue = [(root, 0.0, float(width), -1)]
    max_depth = 0

    while queue:
        node, x, w, depth = queue.pop(0)
        if depth >= 0:
            rects.append({"x": x, "w": w, "d": depth, "node": node})
            max_depth = max(max_depth, depth)
        if node["total"] == 0:
            continue
        offset = x
        for name, child in node["ch"].items():
            cw = w * child["total"] / node["total"]
            if cw < 0.8:
                continue
            queue.append((child, offset, cw, depth + 1))
            offset += cw

    svg_h = HDR_H + (max_depth + 2) * FRAME_H + FOOT_H

    # ── Render ─────────────────────────────────────────────────────────────
    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{svg_h}">',
        '<style>'
        'text{font-family:monospace;pointer-events:none}'
        'rect:hover{stroke:#333;stroke-width:1.5}'
        '</style>',
        # background
        f'<rect width="{width}" height="{svg_h}" fill="#f8f8f8"/>',
        # title
        f'<text x="{width//2}" y="22" text-anchor="middle" '
        f'font-size="15" font-weight="bold" fill="#222">{title}</text>',
        f'<text x="{width//2}" y="40" text-anchor="middle" '
        f'font-size="11" fill="#555">total={total_count/1000:.1f} ms  '
        f'· {len(entries)} unique stacks  '
        f'· {N_ACTIVE} profiler steps  (wider = more time)</text>',
        f'<text x="{width//2}" y="52" text-anchor="middle" '
        f'font-size="10" fill="#888">'
        f'n_items={N_ITEMS}  max_len={MAX_LEN}</text>',
    ]

    for r in rects:
        x, w, depth = r["x"], r["w"], r["d"]
        node = r["node"]
        name = node["_name"] or "(root)"
        y = HDR_H + depth * FRAME_H
        color = _warm_color(name)
        pct = node["total"] / total_count * 100
        tip = f"{name} | {node['total']/1000:.2f} ms ({pct:.1f}%)"
        max_chars = max(1, int(w / 7))
        label = name[:max_chars] if len(name) > max_chars else name

        lines += [
            "<g>",
            f'  <title>{tip}</title>',
            f'  <rect x="{x:.1f}" y="{y}" width="{max(w - 0.8, 0.2):.1f}" '
            f'height="{FRAME_H - 1}" fill="{color}" '
            f'stroke="white" stroke-width="0.4"/>',
            f'  <text x="{x + 3:.1f}" y="{y + FRAME_H - 5}" '
            f'font-size="{FONT_SZ}" fill="#111">{label}</text>',
            "</g>",
        ]

    lines.append("</svg>")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


# ────────────────────────────── Profiling helpers ─────────────────────────────

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_inputs(
    B: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random item-ID tensors representative of ML-20M inputs."""
    input_ids = torch.randint(1, N_ITEMS, (B, MAX_LEN), device=device)
    targets   = torch.randint(1, N_ITEMS, (B, MAX_LEN), device=device)
    neg_ids   = torch.randint(1, N_ITEMS, (B, MAX_LEN, N_NEG), device=device)
    return input_ids, targets, neg_ids


def _profile_mode(
    model: nn.Module,
    mode: str,          # "train" | "infer"
    B: int,
    device: torch.device,
    out_dir: Path,
    model_name: str,
) -> Dict[str, Any]:
    """Warm up, profile, export stacks + chrome trace, render SVGs."""
    input_ids, targets, neg_ids = _make_inputs(B, device)

    if mode == "train":
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            pos_logits, neg_logits, mask = model.score_pairs(
                input_ids, targets, neg_ids
            )
            # Simple proxy loss that exercises both output tensors fully.
            loss = pos_logits.mean() + neg_logits.mean()
            loss.backward()
            optimizer.step()

    else:
        model.eval()

        def step() -> None:
            with torch.no_grad():
                model.score_all(input_ids)

    # ── Warm-up ────────────────────────────────────────────────────────────
    for _ in range(N_WARMUP):
        step()
    _sync(device)

    # ── Profile ────────────────────────────────────────────────────────────
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(N_ACTIVE):
            with record_function(f"{model_name}.{mode}"):
                step()
        _sync(device)

    # ── Export stacks ──────────────────────────────────────────────────────
    cpu_stacks  = str(out_dir / f"{mode}_cpu_stacks.txt")
    cuda_stacks = str(out_dir / f"{mode}_cuda_stacks.txt")
    prof.export_stacks(cpu_stacks,  metric="self_cpu_time_total")
    if device.type == "cuda":
        prof.export_stacks(cuda_stacks, metric="self_cuda_time_total")

    # ── Chrome trace ───────────────────────────────────────────────────────
    prof.export_chrome_trace(str(out_dir / f"{mode}_trace.json"))

    # ── Flame graph SVGs ───────────────────────────────────────────────────
    render_flamegraph_svg(
        cpu_stacks,
        f"{model_name} · {mode} · CPU",
        str(out_dir / f"{mode}_cpu_flamegraph.svg"),
    )
    if device.type == "cuda":
        render_flamegraph_svg(
            cuda_stacks,
            f"{model_name} · {mode} · CUDA",
            str(out_dir / f"{mode}_cuda_flamegraph.svg"),
        )

    # ── Stats ──────────────────────────────────────────────────────────────
    avgs = prof.key_averages()
    cpu_ms  = sum(a.cpu_time_total  for a in avgs) / 1000 / N_ACTIVE
    cuda_ms = (sum(a.cuda_time_total for a in avgs) / 1000 / N_ACTIVE
               if device.type == "cuda" else 0.0)
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6
               if device.type == "cuda" else 0.0)

    top_ops = [
        (a.key, round(a.cuda_time_total / 1000 / N_ACTIVE, 3))
        for a in sorted(avgs, key=lambda x: x.cuda_time_total, reverse=True)[:12]
        if a.cuda_time_total > 0
    ]

    print(f"    cpu={cpu_ms:.1f} ms/step  cuda={cuda_ms:.1f} ms/step  "
          f"peak_mem={peak_mb:.0f} MB")
    return {
        "cpu_ms": cpu_ms,
        "cuda_ms": cuda_ms,
        "peak_mem_mb": peak_mb,
        "top_ops": top_ops,
    }


# ──────────────────────────── Summary charts ─────────────────────────────────

def _make_summary_charts(results: Dict[str, Dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib/numpy unavailable — skipping summary charts")
        return

    models = list(results.keys())
    n = len(models)
    x = np.arange(n)
    bar_w = 0.35

    for mode in ("train", "infer"):
        if not all(mode in results[m] for m in models):
            continue
        cpu_ms  = [results[m][mode]["cpu_ms"]    for m in models]
        cuda_ms = [results[m][mode]["cuda_ms"]   for m in models]
        mem_mb  = [results[m][mode]["peak_mem_mb"] for m in models]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Profiling summary — {mode}  "
            f"(n_items={N_ITEMS}, max_len={MAX_LEN}, "
            f"bs={'varies' if mode == 'train' else INFER_BS}, "
            f"{N_ACTIVE} profiled steps)",
            fontsize=12, fontweight="bold",
        )

        # Time chart
        ax = axes[0]
        b1 = ax.bar(x - bar_w / 2, cpu_ms,  bar_w, label="CPU ms/step",  color="#4e79a7")
        b2 = ax.bar(x + bar_w / 2, cuda_ms, bar_w, label="CUDA ms/step", color="#f28e2b")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("ms / step")
        ax.set_title("Time per step")
        ax.legend()
        for bar in (*b1, *b2):
            h = bar.get_height()
            if h > 0.1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8,
                )

        # Memory chart
        ax = axes[1]
        bars = ax.bar(x, mem_mb, color="#59a14f")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("MB")
        ax.set_title("Peak GPU memory")
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=9,
                )

        fig.tight_layout()
        out = out_dir / f"summary_{mode}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")

    # Top-ops heatmap (train, CUDA time)
    _make_ops_heatmap(results, models, out_dir)


def _make_ops_heatmap(
    results: Dict[str, Dict], models: List[str], out_dir: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    all_ops: set = set()
    for m in models:
        for op, _ in results[m].get("train", {}).get("top_ops", []):
            all_ops.add(op)
    if not all_ops:
        return

    ops = sorted(all_ops)
    data = np.zeros((len(ops), len(models)))
    for j, m in enumerate(models):
        op_map = dict(results[m].get("train", {}).get("top_ops", []))
        for i, op in enumerate(ops):
            data[i, j] = op_map.get(op, 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.8), max(6, len(ops) * 0.45)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels(ops, fontsize=8)
    ax.set_title(
        f"Top CUDA ops — train  (ms / step, {N_ACTIVE} steps)",
        fontweight="bold", fontsize=12,
    )
    fig.colorbar(im, ax=ax, label="ms / step")
    fig.tight_layout()
    out = out_dir / "top_ops_train.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ──────────────────────────────────── Main ────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",    default="profiles", help="output directory")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--models", nargs="*", default=None,
        help="subset of models to run (default: all)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    selected = args.models or list(MODEL_SPECS.keys())
    print(f"Device : {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Models : {selected}", flush=True)
    print(f"Config : n_items={N_ITEMS}  max_len={MAX_LEN}  n_neg={N_NEG}  "
          f"warmup={N_WARMUP}  active={N_ACTIVE}", flush=True)
    print()

    results: Dict[str, Dict] = {}

    for name in selected:
        spec = MODEL_SPECS.get(name)
        if spec is None:
            print(f"[skip] unknown model: {name}")
            continue

        print(f"{'='*60}")
        print(f"  {name}  (train_bs={spec['train_bs']})")
        print(f"{'='*60}", flush=True)

        try:
            model = build_model(spec["model_cfg"], n_items=N_ITEMS).to(device)
        except Exception as exc:
            print(f"  [SKIP] cannot build model: {exc}")
            continue

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params : {n_params:,}", flush=True)

        out_dir = out_root / name
        out_dir.mkdir(exist_ok=True)
        results[name] = {}

        for mode, bs in [("train", spec["train_bs"]), ("infer", INFER_BS)]:
            print(f"  ── {mode} (bs={bs}) ──", flush=True)
            try:
                results[name][mode] = _profile_mode(
                    model, mode, bs, device, out_dir, name,
                )
            except Exception as exc:
                print(f"  [FAIL] {mode}: {exc}")
                import traceback; traceback.print_exc()
                results[name][mode] = {
                    "cpu_ms": 0, "cuda_ms": 0, "peak_mem_mb": 0, "top_ops": [],
                }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()

    # ── Summary charts ─────────────────────────────────────────────────────
    print("=== Summary charts ===", flush=True)
    _make_summary_charts(results, out_root)

    # ── Text table ─────────────────────────────────────────────────────────
    print()
    print("=== Results ===")
    hdr = (f"{'model':<20} {'train_cpu':>10} {'train_cuda':>11} {'train_mem':>10}"
           f"  {'infer_cpu':>10} {'infer_cuda':>11} {'infer_mem':>10}")
    print(hdr)
    print("-" * len(hdr))
    for m, r in results.items():
        tr  = r.get("train", {})
        inf = r.get("infer", {})
        print(
            f"{m:<20} {tr.get('cpu_ms',0):>9.1f}ms "
            f"{tr.get('cuda_ms',0):>10.1f}ms "
            f"{tr.get('peak_mem_mb',0):>8.0f}MB  "
            f"{inf.get('cpu_ms',0):>9.1f}ms "
            f"{inf.get('cuda_ms',0):>10.1f}ms "
            f"{inf.get('peak_mem_mb',0):>8.0f}MB"
        )

    print(f"\nOutputs: {out_root.resolve()}")


if __name__ == "__main__":
    main()

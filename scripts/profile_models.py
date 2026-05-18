#!/usr/bin/env python3
"""Profile all sequential recommendation models using torch.profiler.

Generates SVG flame graphs (CPU + CUDA) and summary PNG charts for every
model × {train step, inference}, so we can compare architectures on time
and memory.

Usage (from repo root):
    python -m scripts.profile_models [--out profiles] [--device cuda]
    python -m scripts.profile_models --models sasrec fmlp --out profiles

Outputs per model:
    profiles/<model>/train_cpu_flamegraph.svg   ← time-nested CPU ops
    profiles/<model>/train_cuda_flamegraph.svg  ← CUDA kernel breakdown
    profiles/<model>/train_trace.json           ← chrome trace (gitignored)
    profiles/<model>/infer_{cpu,cuda}_flamegraph.svg
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
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import build_model


def _xml(s: str) -> str:
    """Escape characters that are invalid inside SVG/XML text and attributes."""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))

# ─────────────────────────── Model specifications ────────────────────────────

N_ITEMS  = 27279
MAX_LEN  = 200
N_NEG    = 64
N_WARMUP = 5
N_ACTIVE = 20
INFER_BS = 256

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
        # kv outer product is [B,H,L,hd,hd] — halved vs training batch size
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


# ──────────────────────────── SVG helpers ─────────────────────────────────────

def _warm_color(name: str) -> str:
    digest = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    hue = (digest % 60) / 360.0
    sat = 0.55 + (digest % 90) / 300.0
    val = 0.78 + (digest % 50) / 250.0
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def _cool_color(name: str) -> str:
    """Blue-purple palette for CUDA kernels."""
    digest = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    hue = 0.55 + (digest % 80) / 400.0   # 0.55–0.75 = cyan → purple
    sat = 0.5 + (digest % 100) / 400.0
    val = 0.65 + (digest % 60) / 300.0
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def _render_svg_flame(
    folded: Dict[str, int],
    title: str,
    output_path: str,
    color_fn=None,
    width: int = 1500,
) -> None:
    """Render folded-stacks dict → SVG flame graph."""
    if not folded:
        print(f"    [warn] no data for {output_path}")
        return
    if color_fn is None:
        color_fn = _warm_color

    total = sum(folded.values())
    if total == 0:
        return

    # ── Build tree ─────────────────────────────────────────────────────────
    def _node(name=""):
        return {"name": name, "ch": {}, "total": 0, "self_t": 0}

    root = _node("(root)")
    root["total"] = total

    for path_str, count in folded.items():
        frames = path_str.split(";")
        cur = root
        for i, f in enumerate(frames):
            if f not in cur["ch"]:
                cur["ch"][f] = _node(f)
            cur = cur["ch"][f]
            cur["total"] += count
            if i == len(frames) - 1:
                cur["self_t"] += count

    # ── BFS layout ─────────────────────────────────────────────────────────
    FRAME_H = 20
    HDR_H   = 60
    FOOT_H  = 12

    rects: List[Dict] = []
    queue  = [(root, 0.0, float(width), -1)]
    max_d  = 0

    while queue:
        node, x, w, d = queue.pop(0)
        if d >= 0:
            rects.append({"x": x, "w": w, "d": d, "node": node})
            max_d = max(max_d, d)
        if not node["total"]:
            continue
        offset = x
        for name, child in sorted(
            node["ch"].items(), key=lambda kv: kv[1]["total"], reverse=True
        ):
            cw = w * child["total"] / node["total"]
            if cw < 1.0:
                continue
            queue.append((child, offset, cw, d + 1))
            offset += cw

    svg_h = HDR_H + (max_d + 2) * FRAME_H + FOOT_H

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{svg_h}">',
        "<style>text{font-family:monospace;pointer-events:none}"
        "rect{cursor:default}rect:hover{stroke:#111;stroke-width:1.5}</style>",
        f'<rect width="{width}" height="{svg_h}" fill="#f9f9f9"/>',
        f'<text x="{width//2}" y="22" text-anchor="middle" '
        f'font-size="15" font-weight="bold" fill="#222">{_xml(title)}</text>',
        f'<text x="{width//2}" y="40" text-anchor="middle" '
        f'font-size="11" fill="#555">'
        f'total={total/1000:.1f} ms · {len(folded)} unique stacks · '
        f'{N_ACTIVE} profiled steps</text>',
        f'<text x="{width//2}" y="55" text-anchor="middle" '
        f'font-size="10" fill="#888">'
        f'n_items={N_ITEMS}  max_len={MAX_LEN}  (wider = more time)</text>',
    ]

    FONT_SZ = 11
    for r in rects:
        x, w, d = r["x"], r["w"], r["d"]
        node = r["node"]
        name = node["name"]
        y = HDR_H + d * FRAME_H
        color = color_fn(name)
        pct   = node["total"] / total * 100
        tip   = _xml(f"{name} | {node['total']/1000:.2f} ms ({pct:.1f}%)")
        max_chars = max(1, int(w / 7))
        label = _xml(name[:max_chars] if len(name) > max_chars else name)
        lines += [
            "<g>",
            f'  <title>{tip}</title>',
            f'  <rect x="{x:.1f}" y="{y}" width="{max(w-0.8, 0.2):.1f}" '
            f'height="{FRAME_H-1}" fill="{color}" '
            f'stroke="white" stroke-width="0.4"/>',
            f'  <text x="{x+3:.1f}" y="{y+FRAME_H-5}" '
            f'font-size="{FONT_SZ}" fill="#111">{label}</text>',
            "</g>",
        ]

    lines.append("</svg>")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"    → {Path(output_path).name}")


# ────────────────── Flame graph from chrome trace ─────────────────────────────

def _shorten_op(name: str) -> str:
    """Keep aten:: names short; trim long CUDA kernel names."""
    if name.startswith("aten::") or name.startswith("autograd::"):
        return name
    if len(name) > 55:
        return name[:52] + "…"
    return name


def _cpu_folded_from_trace(trace_path: str) -> Dict[str, int]:
    """Build folded stacks from CPU op events in a chrome trace."""
    with open(trace_path) as f:
        data = json.load(f)

    raw = data["traceEvents"] if isinstance(data, dict) else data
    cpu_cats = {"cpu_op", "user_annotation", "python_function"}
    events = [
        e for e in raw
        if e.get("ph") == "X"
        and e.get("cat") in cpu_cats
        and "ts" in e and "dur" in e and e["dur"] > 0
    ]
    if not events:
        return {}

    # Group by thread; build time-nesting tree per thread.
    by_tid: Dict[int, List] = defaultdict(list)
    for e in events:
        by_tid[e.get("tid", 0)].append(e)

    folded: Dict[str, int] = defaultdict(int)

    for tid, tevents in by_tid.items():
        tevents.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack: List[Tuple[float, str]] = []  # (end_ts, name)

        for e in tevents:
            ts  = e["ts"]
            dur = e["dur"]
            name = _shorten_op(e.get("name", "?"))

            # Pop finished frames
            while stack and stack[-1][0] <= ts:
                stack.pop()

            path = ";".join(s[1] for s in stack) + ((";" + name) if stack else name)
            folded[path] += int(dur)
            stack.append((ts + dur, name))

    return dict(folded)


def _cuda_folded_from_trace(trace_path: str) -> Dict[str, int]:
    """Build folded stacks: CPU caller (deepest cpu_op) → CUDA kernel name."""
    with open(trace_path) as f:
        data = json.load(f)

    raw = data["traceEvents"] if isinstance(data, dict) else data

    cpu_ops = sorted(
        [e for e in raw if e.get("ph") == "X" and e.get("cat") == "cpu_op"
         and "ts" in e and "dur" in e],
        key=lambda e: e["ts"],
    )
    kernels = [
        e for e in raw
        if e.get("ph") == "X" and e.get("cat") == "kernel"
        and "ts" in e and "dur" in e and e["dur"] > 0
    ]
    if not kernels:
        return {}

    # For each kernel find the deepest (last-starting) CPU op that contains it.
    def find_cpu_caller(k_ts: float) -> str:
        caller = "(no cpu parent)"
        for e in cpu_ops:
            if e["ts"] <= k_ts <= e["ts"] + e["dur"]:
                caller = _shorten_op(e.get("name", "?"))
        return caller

    folded: Dict[str, int] = defaultdict(int)
    for k in kernels:
        caller = find_cpu_caller(k["ts"])
        kname  = _shorten_op(k.get("name", "kernel"))
        folded[f"{caller};{kname}"] += int(k["dur"])

    return dict(folded)


# ────────────────────────────── Profiling helpers ─────────────────────────────

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_inputs(B: int, device: torch.device):
    input_ids = torch.randint(1, N_ITEMS, (B, MAX_LEN), device=device)
    targets   = torch.randint(1, N_ITEMS, (B, MAX_LEN), device=device)
    neg_ids   = torch.randint(1, N_ITEMS, (B, MAX_LEN, N_NEG), device=device)
    return input_ids, targets, neg_ids


def _profile_mode(
    model: nn.Module,
    mode: str,
    B: int,
    device: torch.device,
    out_dir: Path,
    model_name: str,
) -> Dict[str, Any]:
    input_ids, targets, neg_ids = _make_inputs(B, device)

    if mode == "train":
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            pos_logits, neg_logits, _ = model.score_pairs(input_ids, targets, neg_ids)
            (pos_logits.mean() + neg_logits.mean()).backward()
            optimizer.step()
    else:
        model.eval()

        def step() -> None:
            with torch.no_grad():
                model.score_all(input_ids)

    # Warm-up
    for _ in range(N_WARMUP):
        step()
    _sync(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    trace_path = str(out_dir / f"{mode}_trace.json")

    with profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        for _ in range(N_ACTIVE):
            with record_function(f"{model_name}.{mode}"):
                step()
        _sync(device)

    prof.export_chrome_trace(trace_path)

    # Build flame graphs from chrome trace
    cpu_folded  = _cpu_folded_from_trace(trace_path)
    cuda_folded = _cuda_folded_from_trace(trace_path) if device.type == "cuda" else {}

    _render_svg_flame(
        cpu_folded,
        f"{model_name} · {mode} · CPU ops",
        str(out_dir / f"{mode}_cpu_flamegraph.svg"),
        color_fn=_warm_color,
    )
    if cuda_folded:
        _render_svg_flame(
            cuda_folded,
            f"{model_name} · {mode} · CUDA kernels",
            str(out_dir / f"{mode}_cuda_flamegraph.svg"),
            color_fn=_cool_color,
        )

    # Stats
    avgs = prof.key_averages()
    # PyTorch 2.5 renamed cuda_time_total → device_time_total
    def _dev_time(a) -> float:
        try:
            return a.device_time_total
        except AttributeError:
            return getattr(a, "cuda_time_total", 0.0)

    cpu_ms  = sum(a.cpu_time_total for a in avgs) / 1000 / N_ACTIVE
    cuda_ms = sum(_dev_time(a) for a in avgs) / 1000 / N_ACTIVE
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6
               if device.type == "cuda" else 0.0)

    top_ops = [
        (a.key, round(_dev_time(a) / 1000 / N_ACTIVE, 3))
        for a in sorted(avgs, key=_dev_time, reverse=True)[:12]
        if _dev_time(a) > 0
    ]

    print(f"    cpu={cpu_ms:.1f} ms/step  cuda={cuda_ms:.1f} ms/step  "
          f"peak_mem={peak_mb:.0f} MB", flush=True)

    return {
        "cpu_ms":     cpu_ms,
        "cuda_ms":    cuda_ms,
        "peak_mem_mb": peak_mb,
        "top_ops":    top_ops,
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
        cpu_ms  = [results[m][mode]["cpu_ms"]      for m in models]
        cuda_ms = [results[m][mode]["cuda_ms"]     for m in models]
        mem_mb  = [results[m][mode]["peak_mem_mb"] for m in models]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        bs_note = "bs=varies" if mode == "train" else f"bs={INFER_BS}"
        fig.suptitle(
            f"Profiling — {mode}  "
            f"(n_items={N_ITEMS}, max_len={MAX_LEN}, {bs_note}, {N_ACTIVE} steps)",
            fontsize=12, fontweight="bold",
        )

        ax = axes[0]
        b1 = ax.bar(x - bar_w / 2, cpu_ms,  bar_w, label="CPU ms/step",  color="#4e79a7")
        b2 = ax.bar(x + bar_w / 2, cuda_ms, bar_w, label="CUDA ms/step", color="#f28e2b")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("ms / step"); ax.set_title("Time per step"); ax.legend()
        for bar in (*b1, *b2):
            h = bar.get_height()
            if h > 0.1:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8)

        ax = axes[1]
        bars = ax.bar(x, mem_mb, color="#59a14f")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("MB"); ax.set_title("Peak GPU memory")
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        out = out_dir / f"summary_{mode}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")

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

    fig, ax = plt.subplots(
        figsize=(max(10, len(models) * 1.8), max(6, len(ops) * 0.45))
    )
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels(ops, fontsize=8)
    ax.set_title(f"Top CUDA ops — train (ms/step, {N_ACTIVE} steps)",
                 fontweight="bold", fontsize=12)
    fig.colorbar(im, ax=ax, label="ms / step")
    fig.tight_layout()
    out = out_dir / "top_ops_train.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ──────────────────────────────────── Main ────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",    default="profiles")
    p.add_argument("--device", default="cuda")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--regen-svg", action="store_true",
                   help="Re-render SVGs from existing trace JSONs without re-profiling")
    return p.parse_args()


def _regen_svg(out_root: Path, selected: List[str]) -> None:
    """Rebuild flame graph SVGs from already-present chrome trace files."""
    for name in selected:
        out_dir = out_root / name
        for mode in ("train", "infer"):
            trace = out_dir / f"{mode}_trace.json"
            if not trace.exists():
                print(f"  [skip] {name}/{mode} — trace not found")
                continue
            print(f"  {name} · {mode}", flush=True)
            cpu_folded  = _cpu_folded_from_trace(str(trace))
            cuda_folded = _cuda_folded_from_trace(str(trace))
            _render_svg_flame(cpu_folded,  f"{name} · {mode} · CPU ops",
                              str(out_dir / f"{mode}_cpu_flamegraph.svg"),  _warm_color)
            if cuda_folded:
                _render_svg_flame(cuda_folded, f"{name} · {mode} · CUDA kernels",
                                  str(out_dir / f"{mode}_cuda_flamegraph.svg"), _cool_color)


def main() -> None:
    args     = _parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    selected = args.models or list(MODEL_SPECS.keys())

    if args.regen_svg:
        print("Re-generating SVG flame graphs from existing traces …", flush=True)
        _regen_svg(out_root, selected)
        print("Done.")
        return

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    out_root.mkdir(parents=True, exist_ok=True)
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
        print(f"  {name}  (train_bs={spec['train_bs']})", flush=True)
        print(f"{'='*60}")

        try:
            model = build_model(spec["model_cfg"], n_items=N_ITEMS).to(device)
        except Exception as exc:
            print(f"  [SKIP] build failed: {exc}")
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
                import traceback
                print(f"  [FAIL] {mode}: {exc}")
                traceback.print_exc()
                results[name][mode] = {
                    "cpu_ms": 0, "cuda_ms": 0, "peak_mem_mb": 0, "top_ops": [],
                }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()

    print("=== Summary charts ===", flush=True)
    _make_summary_charts(results, out_root)

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
            f"{m:<20} {tr.get('cpu_ms', 0):>9.1f}ms "
            f"{tr.get('cuda_ms', 0):>10.1f}ms "
            f"{tr.get('peak_mem_mb', 0):>8.0f}MB  "
            f"{inf.get('cpu_ms', 0):>9.1f}ms "
            f"{inf.get('cuda_ms', 0):>10.1f}ms "
            f"{inf.get('peak_mem_mb', 0):>8.0f}MB"
        )

    print(f"\nOutputs: {out_root.resolve()}")


if __name__ == "__main__":
    main()

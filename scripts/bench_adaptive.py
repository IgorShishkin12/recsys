#!/usr/bin/env python3
"""Benchmark: fixed-length batching vs adaptive token-budget batching.

Measures forward+backward throughput (tokens/s and steps/s) for both modes
using the real data distribution and model. Run on GPU for meaningful numbers.

Usage (from repo root):
    .venv/bin/python scripts/bench_adaptive.py
"""
from __future__ import annotations
import sys, time
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import (
    load_processed, split_loo, SeqTrainDataset,
    BucketBatchSampler, pad_collate, left_pad,
)
from src.models import build_model

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PKL     = Path("data/processed.pkl")
BATCH_SIZE   = 256
TOKEN_BUDGET = BATCH_SIZE * 256   # = 65536
MAX_LEN      = 256
WARMUP_STEPS = 10
BENCH_STEPS  = 80
# ─────────────────────────────────────────────────────────────────────────────

def fixed_collate(batch):
    """Original behaviour: always pad to MAX_LEN (no adaptive routing)."""
    users, inputs, targets = [], [], []
    for b in batch:
        seq_in  = b["input"].tolist()
        seq_tgt = b["target"].tolist()
        users.append(b["user"])
        inputs.append(torch.tensor(left_pad(seq_in,  MAX_LEN), dtype=torch.long))
        targets.append(torch.tensor(left_pad(seq_tgt, MAX_LEN), dtype=torch.long))
    return {"user": torch.stack(users),
            "input": torch.stack(inputs),
            "target": torch.stack(targets)}


def run_bench(loader, model, device, label):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    times, tokens, batches = [], [], []

    step = 0
    while step < WARMUP_STEPS + BENCH_STEPS:
        for batch in loader:
            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            B, L = inp.shape

            if step >= WARMUP_STEPS:
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()

            with torch.autocast(device.type, dtype=torch.bfloat16):
                h = model.encode(inp)             # [B, L, d]
                E = model.output_embedding
                pos_emb = E[tgt]
                loss = -(h * pos_emb).sum(-1).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step >= WARMUP_STEPS:
                torch.cuda.synchronize(device)
                dt = time.perf_counter() - t0
                times.append(dt)
                tokens.append(B * L)
                batches.append(1)

            step += 1
            if step >= WARMUP_STEPS + BENCH_STEPS:
                break

    total_t  = sum(times)
    total_tok = sum(tokens)
    n_steps  = len(times)
    avg_L    = total_tok / n_steps / BATCH_SIZE  # approx avg L per step (B may vary)
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  steps:        {n_steps}")
    print(f"  avg ms/step:  {1000*total_t/n_steps:.1f}")
    print(f"  tokens/s:     {total_tok/total_t:,.0f}")
    print(f"  avg L/batch:  {avg_L:.1f}  (effective, after pow2 pad)")
    print(f"  avg B/batch:  {total_tok/total_t * total_t / n_steps / avg_L:.1f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(device)}")

    print("\nLoading data …")
    proc = load_processed(DATA_PKL)
    train_seq, _, _ = split_loo(proc.user_seq)

    ds = SeqTrainDataset(train_seq, max_len=MAX_LEN)
    seq_lens = np.array([len(train_seq[u]) for u in ds.users])

    model_cfg = dict(name="sasrec", d=256, n_blocks=3, n_heads=4,
                     max_len=MAX_LEN, dropout=0.0, ffn_mult=4,
                     use_rope=True, use_ligr_gates=False, use_side=False)
    model = build_model(model_cfg, n_items=proc.n_items).to(device)

    nw = 4
    # ── Baseline: fixed L=256, random batching ─────────────────────────────
    from torch.utils.data import RandomSampler
    fixed_loader = DataLoader(
        ds, batch_size=BATCH_SIZE, sampler=RandomSampler(ds),
        drop_last=True, num_workers=nw,
        pin_memory=(device.type == "cuda"), collate_fn=fixed_collate,
    )
    run_bench(fixed_loader, model, device, f"Fixed  L={MAX_LEN}, B={BATCH_SIZE} (no routing)")

    # ── Adaptive: token-budget bucket sampler ──────────────────────────────
    bucket_sampler = BucketBatchSampler(seq_lens, token_budget=TOKEN_BUDGET, drop_last=True)
    adaptive_loader = DataLoader(
        ds, batch_sampler=bucket_sampler, num_workers=nw,
        pin_memory=(device.type == "cuda"), collate_fn=pad_collate,
    )
    run_bench(adaptive_loader, model, device, f"Adaptive token_budget={TOKEN_BUDGET}, pow2 pad")

    print()


if __name__ == "__main__":
    main()

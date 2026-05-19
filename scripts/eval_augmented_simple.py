#!/usr/bin/env python3
"""Eval sasrec_augmented with the SAME protocol as eval_by_seqlen.py
(no prepend_seq, no val-target masking in seen set) so per-bucket numbers
are comparable to profiles/predictions.csv."""
from __future__ import annotations
import csv, sys, torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_processed, split_loo, SeqEvalDataset, build_user_seen_lookup
from src.eval import build_padded_seen
from src.models import build_model

CK_PATH  = Path("runs/sasrec_augmented/best.pt")
DATA_PKL = Path("data/processed.pkl")
OUT_CSV  = Path("profiles/predictions_augmented_simple.csv")
NEG_INF  = -1e9

data = load_processed(DATA_PKL)
train_seq, val_map, test_map = split_loo(data.user_seq)
train_seq_len = {u: len(s) for u, s in train_seq.items()}

ck      = torch.load(CK_PATH, map_location="cpu", weights_only=False)
cfg     = dict(ck["config"]["model"])
max_len = cfg["max_len"]
print(f"max_len={max_len}  best_epoch={ck.get('epoch', '?')}")

model = build_model(cfg, n_items=data.n_items)
model.load_state_dict(ck.get("ema_state_dict") or ck["state_dict"], strict=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
print(f"Device: {device}")

# Simple protocol: mask only train_seq items (same as eval_by_seqlen.py)
user_seen   = build_user_seen_lookup(train_seq)
padded_seen = build_padded_seen(user_seen, data.n_users).to(device)

# Input: train_seq only (no prepend_seq), same as eval_by_seqlen.py
test_ds = SeqEvalDataset(train_seq, test_map, max_len=max_len)
loader  = DataLoader(test_ds, batch_size=512, shuffle=False,
                     num_workers=4, pin_memory=(device.type == "cuda"))

rows = []
with torch.no_grad():
    for batch in loader:
        users   = batch["user"].to(device)
        inputs  = batch["input"].to(device)
        targets = batch["target"].to(device)
        scores  = model.score_all(inputs)
        scores[:, 0] = NEG_INF
        scores.scatter_(1, padded_seen[users], NEG_INF)
        b_idx = torch.arange(users.size(0), device=device)
        scores[b_idx, targets] = scores[b_idx, targets].clamp(min=NEG_INF / 2)
        tgt_s = scores[b_idx, targets].unsqueeze(1)
        rank = (scores > tgt_s).sum(dim=1).add_(1)
        for u, r in zip(users.cpu().tolist(), rank.cpu().tolist()):
            rows.append({
                "model":         "sasrec_augmented",
                "user_id":       u,
                "train_seq_len": train_seq_len.get(u, 0),
                "effective_len": min(train_seq_len.get(u, 0), max_len),
                "rank":          r,
            })

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model","user_id","train_seq_len","effective_len","rank"])
    w.writeheader()
    w.writerows(rows)

ndcg10 = np.mean([1/np.log2(r["rank"]+1) if r["rank"] <= 10 else 0.0 for r in rows])
print(f"Saved {len(rows):,} rows -> {OUT_CSV}")
print(f"NDCG@10 (simple protocol) = {ndcg10:.4f}")

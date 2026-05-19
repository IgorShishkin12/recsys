#!/usr/bin/env python3
"""Test-set evaluation for sasrec_notrim/best.pt.

Protocol: train_seq as input history (no val item prepended), mask train_seq items seen.
Matches the standard eval used by eval_by_seqlen.py for fair comparison.
"""
from __future__ import annotations
import sys, torch, numpy as np
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_processed, split_loo, SeqEvalDataset, build_user_seen_lookup
from src.eval import build_padded_seen, evaluate_full_catalog, format_metrics
from src.models import build_model

CK_PATH  = Path("runs/sasrec_notrim/best.pt")
DATA_PKL = Path("data/processed.pkl")
NEG_INF  = -1e9

data = load_processed(DATA_PKL)
train_seq, val_map, test_map = split_loo(data.user_seq)

ck      = torch.load(CK_PATH, map_location="cpu", weights_only=False)
cfg     = dict(ck["config"]["model"])
max_len = cfg["max_len"]
print(f"Checkpoint: {CK_PATH}  epoch={ck.get('epoch','?')}  max_len={max_len}")
print(f"Val NDCG@10 at checkpoint: {ck['val_metrics'].get('NDCG@10', '?'):.4f}")

model = build_model(cfg, n_items=data.n_items)
model.load_state_dict(ck.get("ema_state_dict") or ck["state_dict"], strict=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
print(f"Device: {device}\n")

user_seen = {}
for u, s in train_seq.items():
    items = list(s)
    if u in val_map:
        items.append(val_map[u])
    user_seen[u] = np.unique(np.asarray(items, dtype=np.int64))
padded_seen = build_padded_seen(user_seen, data.n_users).to(device)

test_ds = SeqEvalDataset(train_seq, test_map, max_len=max_len,
                         prepend_seq={u: [v] for u, v in val_map.items()})
loader  = DataLoader(test_ds, batch_size=1024, shuffle=False,
                     num_workers=4, pin_memory=(device.type == "cuda"))

metrics = evaluate_full_catalog(
    model, loader, padded_seen, data.n_items,
    ks=(5, 10, 20), device=str(device),
)
print("Test metrics:")
print(format_metrics(metrics))

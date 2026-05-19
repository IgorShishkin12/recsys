# Experiment: sasrec_augmented max_len=512 — FAILED / ABANDONED

## What we tried
SASRec with session-shuffle + random-window + length-curriculum at max_len=512
(vs baseline max_len=200). Goal: improve NDCG@10 for long-sequence users.

## Memory problem
`neg_emb` tensor shape is `[B, L, K, d]`. At B=256, L=512, K=256, d=256, bf16:
→ 16 GB just for neg_emb; another 16 GB for the backward-pass gradient.
GPU had ~44 GB free but other processes occupied ~35 GB, leaving ~9 GB — not enough.

Forced workaround: B=128, n_neg=128 → neg_emb = 4 GB, fits.
But this means **4× fewer training pairs per epoch** vs baseline (B×K: 16k vs 64k).

## Result (runs 1–2 in sasrec_augmented_L512_log.jsonl)
- Run 1 (B=256, K=256): OOM on first step
- Run 2 (B=128, K=128): 21 epochs, best NDCG@10 = **0.153**

Baseline (max_len=200, B=256, K=256): best NDCG@10 ≈ **0.200**
→ ~25% regression, attributed to 4× fewer training pairs + augmentation noise.

## Root cause
The training-signal reduction (not the augmentation itself) caused the regression.
Augmentation effect is confounded with the B×K reduction.

## Decision
Restart with max_len=256 (power-of-2, covers dataset p75), restoring B=256, K=256.
neg_emb = [256, 256, 256, 256] × bf16 = 8 GB; backward needs another 8 GB = 16 GB total.
Requires `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to defrag the allocator.
See configs/sasrec_augmented.yaml and runs/sasrec_augmented/ for the clean re-run.

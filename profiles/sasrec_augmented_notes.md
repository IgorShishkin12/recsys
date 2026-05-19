# Experiment: sasrec_augmented (max_len=256, shuffle+random_window+curriculum) — WORSE

## Config
`configs/sasrec_augmented.yaml` — max_len=256, B=256, n_neg=256, curriculum_warmup_frac=0.3

Augmentations applied simultaneously:
- `shuffle_sessions: true`   — permute items within 30-min session clusters
- `random_window: true`      — random contiguous window instead of always last max_len+1
- `curriculum_warmup_frac: 0.3` — ramp from uniform to length-proportional sampling over 30 epochs

## Results (same-protocol per-bucket comparison, simple eval, no prepend_seq)

```
Bucket       sasrec (baseline)   sasrec_augmented   diff
1–20              0.1995              0.1677          −16%
21–40             0.1710              0.1520          −11%
41–80             0.1376              0.1211          −12%
81–130            0.1133              0.1018          −10%
131–200           0.0963              0.0837          −13%
201–320           0.0847              0.0739          −13%
321–512           0.0769              0.0689          −10%
513–1024          0.0682              0.0618           −9%
1025–2048         0.0647              0.0630           −3%
>2048             0.0585              0.0395          −32%
```

Training-eval overall: baseline 0.1976 → augmented **0.163** (−17%).

The augmented model is **worse in every bucket**, including the long-sequence users it was
designed to help. The 1025–2048 bucket is nearly flat (−3%); all others regress 10–16%.

## Root cause hypothesis

`random_window: true` is the primary suspect. Randomly subsampling a contiguous window
during training teaches the model to predict from arbitrary history positions and discards
the most recent items, which sacrifices the recency bias critical for recommendation.

Session shuffle and curriculum sampling are less likely to cause uniform regression across
all sequence lengths.

## Decision

Ablate `random_window` out. Next run: `sasrec_shuffle_only` — only within-session shuffle,
no random window, no curriculum. Config: `configs/sasrec_shuffle_only.yaml`.

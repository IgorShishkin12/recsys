#!/usr/bin/env python3
"""Analyze SASRec attention patterns.

Extracts per-head attention weights by replaying the forward pass manually
(SDPA doesn't expose weights). Computes:
  - Recency bias: mean attention weight vs. relative position offset
  - Mean attended distance per head
  - Session-crossing fraction: how much attention crosses 30-min session
    boundaries vs. stays within the same session

Run from repo root:
    source .venv/bin/activate
    python scripts/analyze_attention.py
"""
from __future__ import annotations
import sys, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_processed, split_loo, SeqEvalDataset, left_pad
from src.models import build_model

# ── Config ──────────────────────────────────────────────────────────────────
CK_PATH  = Path("runs/sasrec_notrim/best.pt")
DATA_PKL = Path("data/processed.pkl")
OUT_DIR  = Path("profiles/attention")
N_USERS  = 500   # sample size for analysis
SEED     = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def extract_attention_weights(
    model, input_ids: torch.Tensor
) -> list[torch.Tensor]:
    """Return per-block attention weight tensors [B, H, L, L].

    Replays the forward pass manually so we can capture the softmax
    outputs that SDPA would normally discard.
    """
    B, L = input_ids.shape
    pad_mask = input_ids == 0

    x = model.item_emb(input_ids)
    if model.use_pos_emb:
        pos = torch.arange(L, device=input_ids.device)
        x = x + model.pos_emb(pos).unsqueeze(0)

    all_weights: list[torch.Tensor] = []

    for blk in model.blocks:
        x_ln = blk.ln1(x)
        attn  = blk.attn
        qkv   = attn.qkv(x_ln).view(B, L, 3, attn.n_heads, attn.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)   # [B, H, L, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if attn.rope is not None:
            q = attn.rope(q)
            k = attn.rope(k)

        # Build the same mask as in CausalAttention.forward
        device = input_ids.device
        idx   = torch.arange(L, device=device)
        causal = idx.view(1, L) <= idx.view(L, 1)
        keep   = causal.view(1, 1, L, L) & ~pad_mask.view(B, 1, 1, L)
        eye    = torch.eye(L, dtype=torch.bool, device=device).view(1, 1, L, L)
        keep   = keep | eye

        scale  = attn.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L]
        scores = scores.masked_fill(~keep, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        all_weights.append(weights.cpu())

        # Continue with the real computation so later blocks are correct
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=keep, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        a   = attn.proj(out)
        if blk.g_attn is not None:
            a = a * blk.g_attn
        x = x + a

        f = blk.ffn(blk.ln2(x))
        if blk.g_ffn is not None:
            f = f * blk.g_ffn
        x = x + f

    return all_weights  # list of n_blocks tensors, each [B, H, L, L]


def build_session_mask(seq: list[int], cuts: list[int], max_len: int) -> np.ndarray:
    """Return [L, L] bool matrix where entry (i,j) is True iff positions i and j
    are in the SAME session.  Positions corresponding to PAD are session 0."""
    seq_trunc = seq[-max_len:]
    n = len(seq_trunc)
    pad = max_len - n
    # Assign a session id to each real token
    sid = np.zeros(n, dtype=np.int32)
    # cuts are 0-based indices into the full seq; adjust to truncated seq
    offset = len(seq) - n
    s = 0
    for c in cuts:
        c_adj = c - offset
        if 0 < c_adj < n:
            sid[c_adj:] = s + 1
            s += 1
    # Full [max_len] array: pad positions get sid=-1 (never same-session)
    full_sid = np.full(max_len, -1, dtype=np.int32)
    full_sid[pad:] = sid
    same = (full_sid[:, None] == full_sid[None, :]) & (full_sid[:, None] >= 0)
    return same  # [L, L]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data and model …")
    data = load_processed(DATA_PKL)
    train_seq, val_map, test_map = split_loo(data.user_seq)

    ck      = torch.load(CK_PATH, map_location="cpu", weights_only=False)
    cfg     = dict(ck["config"]["model"])
    max_len = cfg["max_len"]
    n_blocks = cfg["n_blocks"]
    n_heads  = cfg["n_heads"]
    print(f"  max_len={max_len}  blocks={n_blocks}  heads={n_heads}  "
          f"epoch={ck.get('epoch','?')}  val NDCG@10={ck['val_metrics'].get('NDCG@10',0):.4f}")

    model = build_model(cfg, n_items=data.n_items)
    model.load_state_dict(ck.get("ema_state_dict") or ck["state_dict"], strict=True)
    model.to(device).eval()

    # Sample users with at least 2 sessions and seq_len >= 10
    session_cuts = data.user_session_cuts or {}
    eligible = [u for u in test_map
                if u in train_seq
                and len(train_seq[u]) >= 10
                and len(session_cuts.get(u, [])) >= 1]
    sample = random.sample(eligible, min(N_USERS, len(eligible)))
    print(f"  Sampled {len(sample)} users for analysis")

    # Accumulators
    # recency[block][head] -> np.array of length max_len (index = distance from query)
    recency    = [[np.zeros(max_len) for _ in range(n_heads)] for _ in range(n_blocks)]
    recency_ct = [[np.zeros(max_len) for _ in range(n_heads)] for _ in range(n_blocks)]
    # session_crossing[block][head] -> (within_sum, cross_sum)
    sess_within = np.zeros((n_blocks, n_heads))
    sess_cross  = np.zeros((n_blocks, n_heads))
    mean_dist   = np.zeros((n_blocks, n_heads))
    mean_dist_ct= np.zeros((n_blocks, n_heads))

    # Process in small batches
    batch_size = 32
    for start in range(0, len(sample), batch_size):
        batch_users = sample[start:start + batch_size]
        padded = [left_pad(train_seq[u], max_len) for u in batch_users]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        block_weights = extract_attention_weights(model, input_ids)
        # block_weights: list of [B, H, L, L] (already on CPU)

        for b_idx, weights in enumerate(block_weights):
            # weights: [B, H, L, L]  attn[b,h,t,s] = how much pos t attends to s
            B, H, L, _ = weights.shape

            for bi, u in enumerate(batch_users):
                seq_len = min(len(train_seq[u]), max_len)
                pad     = max_len - seq_len
                cuts    = session_cuts.get(u, [])
                same_sess = build_session_mask(train_seq[u], cuts, max_len)

                for h in range(H):
                    w = weights[bi, h].numpy()  # [L, L]
                    # Only look at real (non-PAD) query positions
                    for t in range(pad, L):
                        row = w[t, pad:t+1]  # attention from t to all real positions <= t
                        if row.sum() < 1e-8:
                            continue
                        offsets = np.arange(len(row))[::-1]  # distance from t (0=self)
                        # recency curve: accumulate weight at each distance
                        for d, wt in zip(offsets, row):
                            recency[b_idx][h][d]    += wt
                            recency_ct[b_idx][h][d] += 1
                        # mean attended distance
                        mean_dist[b_idx, h]    += np.dot(offsets, row)
                        mean_dist_ct[b_idx, h] += 1
                        # session-crossing
                        for s in range(pad, t+1):
                            if same_sess[t, s]:
                                sess_within[b_idx, h] += w[t, s]
                            else:
                                sess_cross[b_idx, h]  += w[t, s]

        if (start // batch_size) % 5 == 0:
            print(f"  processed {min(start+batch_size, len(sample))}/{len(sample)} users …")

    # ── Normalize ────────────────────────────────────────────────────────────
    mean_dist /= np.maximum(mean_dist_ct, 1)

    sess_total  = sess_within + sess_cross
    cross_frac  = sess_cross / np.maximum(sess_total, 1e-8)

    max_dist_plot = 64   # show first 64 offsets for recency curves

    # ── Plot 1: Recency curves ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 4), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, n_heads))
    for b in range(n_blocks):
        ax = axes[b] if n_blocks > 1 else axes
        for h in range(n_heads):
            ct = recency_ct[b][h][:max_dist_plot]
            r  = np.where(ct > 0, recency[b][h][:max_dist_plot] / ct, 0)
            ax.plot(range(max_dist_plot), r, label=f"H{h}", color=colors[h], alpha=0.85)
        ax.set_title(f"Block {b+1}")
        ax.set_xlabel("Distance from query (0 = self)")
        if b == 0:
            ax.set_ylabel("Mean attention weight")
        ax.legend(fontsize=7)
        ax.set_xlim(0, max_dist_plot - 1)
    fig.suptitle("Recency bias per head (how far back each head attends)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "recency_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved recency_curves.png")

    # ── Plot 2: Mean attended distance ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(n_heads)
    width = 0.25
    for b in range(n_blocks):
        ax.bar(x + b * width, mean_dist[b], width, label=f"Block {b+1}")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_ylabel("Mean attended distance (tokens)")
    ax.set_title("Mean attention distance per head")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "mean_distance.png", dpi=150)
    plt.close(fig)
    print(f"Saved mean_distance.png")

    # ── Plot 3: Session-crossing fraction ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3))
    for b in range(n_blocks):
        ax.bar(x + b * width, cross_frac[b], width, label=f"Block {b+1}")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_ylabel("Fraction of attention crossing session boundary")
    ax.set_title("Session-crossing attention (higher = more cross-session)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "session_crossing.png", dpi=150)
    plt.close(fig)
    print(f"Saved session_crossing.png")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n── Mean attended distance (tokens) ──")
    print(f"{'':8s}", end="")
    for h in range(n_heads): print(f"  H{h:1d}  ", end="")
    print()
    for b in range(n_blocks):
        print(f"Block {b+1}:", end="")
        for h in range(n_heads): print(f"  {mean_dist[b,h]:4.1f}", end="")
        print()

    print("\n── Cross-session attention fraction ──")
    print(f"{'':8s}", end="")
    for h in range(n_heads): print(f"  H{h:1d}  ", end="")
    print()
    for b in range(n_blocks):
        print(f"Block {b+1}:", end="")
        for h in range(n_heads): print(f"  {cross_frac[b,h]:.2f}", end="")
        print()


if __name__ == "__main__":
    main()

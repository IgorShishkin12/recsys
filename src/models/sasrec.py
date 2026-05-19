"""Modernised SASRec: Pre-LN + RoPE + SwiGLU + LiGR (gated residual) + tied weights.

Sequence convention: PAD = 0 lives on the LEFT (we left-pad). The final hidden
state of the last position summarises the whole user history.

Forward returns hidden states [B, L, d]; loss is computed externally so we can
swap gBCE / sampled softmax / full softmax without touching the model.

Tied weights: output logits are computed by the trainer as `hidden @ E.T` where
E is the (input) item embedding — saves |I| × d params and gives +1..3% NDCG.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SASRecConfig:
    n_items: int                 # excluding PAD
    d: int = 128
    n_blocks: int = 3
    n_heads: int = 4
    max_len: int = 200
    dropout: float = 0.2
    ffn_mult: int = 4            # FFN hidden = ffn_mult * d * 2/3 (SwiGLU keeps params)
    use_rope: bool = True
    use_ligr_gates: bool = True
    use_side: bool = False
    side_use_genome: bool = True
    use_time_delta: bool = False
    n_time_buckets: int = 9      # 0=pad/unknown, 1–8 log-scale buckets


# ----------------------------- RoPE ---------------------------------------- #
class RotaryEmbedding(nn.Module):
    """RoPE applied per-head on query and key tensors.

    Standard formulation (Su et al., 2021): for each pair of dims (2k, 2k+1),
    rotate by angle m·θ_k where m is the position index and
    θ_k = base^(-2k/d_head). Using FP32 for numerical stability of cos/sin even
    when the model runs in bf16/fp16.
    """

    def __init__(self, head_dim: int, max_len: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)                       # [L, head_dim/2]
        cos = freqs.cos()                                      # [L, head_dim/2]
        sin = freqs.sin()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x: [B, H, L, head_dim]. Returns x rotated."""
        L = x.shape[-2]
        cos = self.cos[offset:offset + L].to(x.dtype)          # [L, hd/2]
        sin = self.sin[offset:offset + L].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)                            # [B, H, L, hd/2]
        # (x1, x2) → (x1·cos − x2·sin, x1·sin + x2·cos)
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        return torch.cat([rot1, rot2], dim=-1)


# ----------------------------- Attention ----------------------------------- #
class CausalAttention(nn.Module):
    """Multi-head causal self-attention with optional RoPE.

    Uses torch.nn.functional.scaled_dot_product_attention which under the hood
    selects FlashAttention / mem-efficient kernels on A100.
    """

    def __init__(self, d: int, n_heads: int, max_len: int, dropout: float, use_rope: bool):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d ({d}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_len) if use_rope else None

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d]; key_padding_mask: [B, L] True for PAD positions.

        Builds combined causal + key-padding mask, ensuring at least the diagonal
        is kept on every row to avoid all-masked softmax → NaN. torch.SDPA on 2.5
        forbids passing attn_mask together with is_causal=True, so we encode
        causality ourselves.
        """
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                            # each [B, L, H, hd]
        q = q.transpose(1, 2)                                  # [B, H, L, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        device = x.device
        idx = torch.arange(L, device=device)
        causal = idx.view(1, L) <= idx.view(L, 1)              # [L, L] True=keep
        keep = causal.view(1, 1, L, L) & ~key_padding_mask.view(B, 1, 1, L)
        # Diagonal must always be true to prevent all-masked rows on full-PAD prefix.
        eye = torch.eye(L, dtype=torch.bool, device=device).view(1, 1, L, L)
        keep = keep | eye

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=keep,                                    # True = keep
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )                                                      # [B, H, L, hd]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.resid_drop(self.proj(out))


# ----------------------------- SwiGLU FFN ---------------------------------- #
class SwiGLU(nn.Module):
    """Shazeer-2020 SwiGLU FFN. Hidden width = (8/3) * d * mult / 4 ≈ keeps param
    count comparable to a vanilla 4d-ReLU FFN."""

    def __init__(self, d: int, mult: int = 4, dropout: float = 0.2):
        super().__init__()
        # 2/3 factor compensates for the gated structure to match param count.
        hidden = int((mult * d) * 2 / 3)
        # Round up to multiple of 8 for tensor core friendliness.
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ----------------------------- LiGR Block ---------------------------------- #
class LiGRBlock(nn.Module):
    """Pre-LN block with gated residual (eSASRec-style).

        y = x + g_attn ⊙ Attention(LN(x))
        z = y + g_ffn  ⊙ FFN(LN(y))

    Gates are learned per-channel scalars, initialised to 0 so the block starts
    as identity (cf. ReZero / LayerScale). This is what lets us safely stack 3-4
    blocks at d=128-256 without warmup.
    """

    def __init__(self, cfg: SASRecConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.attn = CausalAttention(cfg.d, cfg.n_heads, cfg.max_len, cfg.dropout, cfg.use_rope)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.ffn = SwiGLU(cfg.d, mult=cfg.ffn_mult, dropout=cfg.dropout)
        if cfg.use_ligr_gates:
            self.g_attn = nn.Parameter(torch.zeros(cfg.d))
            self.g_ffn = nn.Parameter(torch.zeros(cfg.d))
        else:
            self.register_parameter("g_attn", None)
            self.register_parameter("g_ffn", None)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        a = self.attn(self.ln1(x), key_padding_mask)
        if self.g_attn is not None:
            a = a * self.g_attn
        x = x + a
        f = self.ffn(self.ln2(x))
        if self.g_ffn is not None:
            f = f * self.g_ffn
        x = x + f
        return x


# ----------------------------- Full Model ---------------------------------- #
class SASRec(nn.Module):
    def __init__(self, cfg: SASRecConfig, side_module: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.d, padding_idx=0)
        self.time_delta_emb = (
            nn.Embedding(cfg.n_time_buckets, cfg.d, padding_idx=0)
            if cfg.use_time_delta else None
        )
        # Even with RoPE inside attention, a small learned positional bias
        # helps disambiguate left-padded short sequences. We DON'T use it when
        # use_rope=True; the input embedding alone is enough.
        self.use_pos_emb = not cfg.use_rope
        if self.use_pos_emb:
            self.pos_emb = nn.Embedding(cfg.max_len, cfg.d)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([LiGRBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_ln = nn.LayerNorm(cfg.d)

        self.side = side_module
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)
        if self.use_pos_emb:
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    @property
    def output_embedding(self) -> torch.Tensor:
        """Tied weights — output projection IS the input embedding matrix."""
        return self.item_emb.weight                            # [n_items+1, d]

    def encode(
        self,
        input_ids: torch.Tensor,
        time_buckets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """input_ids: [B, L] long → hidden [B, L, d]."""
        B, L = input_ids.shape
        pad_mask = input_ids == 0                              # [B, L] True=PAD
        x = self.item_emb(input_ids)
        if self.time_delta_emb is not None and time_buckets is not None:
            x = x + self.time_delta_emb(time_buckets)
        if self.side is not None:
            x = x + self.side(input_ids)
        if self.use_pos_emb:
            pos = torch.arange(L, device=input_ids.device)
            x = x + self.pos_emb(pos).unsqueeze(0)
        x = self.emb_drop(x)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.final_ln(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        time_buckets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(input_ids, time_buckets)

    @torch.no_grad()
    def score_all(
        self,
        input_ids: torch.Tensor,
        time_buckets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """For evaluation: return logits over the full catalog at the LAST position.
        Returns [B, n_items+1]; caller should mask PAD column and seen items.
        """
        h = self.encode(input_ids, time_buckets)               # [B, L, d]
        last = h[:, -1, :]                                     # [B, d]
        return last @ self.output_embedding.T                  # [B, n_items+1]

    def score_pairs(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        time_buckets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute logits for shifted-sequence training.

        input_ids:  [B, L] inputs (left-padded)
        target_ids: [B, L] true next items at each position
        neg_ids:    [B, L, K] sampled negatives per position

        Returns (pos_logits [B,L], neg_logits [B,L,K], target_mask [B,L]).
        """
        h = self.encode(input_ids, time_buckets)               # [B, L, d]
        E = self.output_embedding                              # [V, d]
        # Positive logits: <h_t, E[target_t]>
        pos_emb = E[target_ids]                                # [B, L, d]
        pos_logits = (h * pos_emb).sum(dim=-1)                 # [B, L]
        # Negative logits: <h_t, E[neg_t,k]>
        # Use bmm instead of einsum to avoid materialising an [B,L,K,d] intermediate
        # (einsum would expand h → 8 GB temporary on top of neg_emb's 8 GB).
        neg_emb = E[neg_ids]                                   # [B, L, K, d]
        B, L, K, dh = neg_emb.shape
        # bmm([BL, K, d], [BL, d, 1]) → [BL, K, 1]: both operands stay contiguous,
        # no transpose copy needed → peak memory = neg_emb (8 GB) not 2× that.
        neg_logits = torch.bmm(
            neg_emb.view(B * L, K, dh),
            h.reshape(B * L, dh, 1),
        ).squeeze(2).view(B, L, K)                             # [B, L, K]
        # Mask: position is valid for loss if input_t != PAD AND target_t != PAD.
        # We left-pad inputs, so a "real" position has both. Use input mask.
        target_mask = (input_ids != 0)                          # [B, L]
        return pos_logits, neg_logits, target_mask

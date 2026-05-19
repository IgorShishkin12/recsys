"""ML-20M preprocessing + sequential Dataset.

Protocol (canonical, matches gSASRec / TOPAPEC/esasrec):
- All ratings → events (no rating>=4 threshold).
- Iterative 5-core filtering of users and items.
- Dense item ids in 1..N (0 reserved for padding).
- Leave-one-out split: last item -> test, second-last -> val, rest -> train.
- Sequence truncation to last `max_len` items, left-padded with 0.
"""
from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, Dataset, Sampler


PAD_ID = 0
SESSION_GAP_SEC = 1800  # 30-minute session boundary

# Log-scale time-delta buckets (seconds).  Bucket 0 = pad/unknown (no gradient).
# 1: same second (batch-logged), 2: <1min, 3: 1–30min, 4: 30min–2h,
# 5: 2h–1day, 6: 1day–1week, 7: 1week–1month, 8: >1month
_TD_BOUNDS = (0, 1, 60, 1800, 7200, 86400, 604800, 2592000)
N_TIME_BUCKETS = len(_TD_BOUNDS) + 1   # = 9


def time_delta_bucket(delta_sec: int) -> int:
    """Map a non-negative time gap (seconds) to bucket index 1–8."""
    for i, b in enumerate(_TD_BOUNDS):
        if delta_sec <= b:
            return i + 1
    return N_TIME_BUCKETS - 1


def _seq_to_time_deltas(ts: List[int], n: int) -> List[int]:
    """Given a timestamp list of length n, return n time-delta bucket IDs.

    Index 0 is always bucket 0 (first item — no predecessor in this window).
    """
    buckets = [0]
    for i in range(1, n):
        buckets.append(time_delta_bucket(max(0, ts[i] - ts[i - 1])))
    return buckets


@dataclass
class ProcessedData:
    """Container for preprocessed sequences and id mappings."""

    user_seq: Dict[int, List[int]]         # uid -> full chronological item list
    n_users: int
    n_items: int                            # excluding PAD
    item_pop: np.ndarray                    # shape [n_items+1], item_pop[i] = freq, [0]=0
    movie_id_to_idx: Dict[int, int]         # original movieId -> 1..n_items
    user_id_to_idx: Dict[int, int]          # original userId -> 0..n_users-1
    # Session-start indices per user (0-based into user_seq); None for old pickles.
    user_session_cuts: Optional[Dict[int, List[int]]] = None
    # Unix timestamps aligned with user_seq; None for old pickles.
    user_timestamps: Optional[Dict[int, List[int]]] = None

    @property
    def vocab_size(self) -> int:
        """Item vocab size including PAD."""
        return self.n_items + 1


def _five_core_filter(
    df: pd.DataFrame,
    min_user: int = 5,
    min_item: int = 5,
) -> pd.DataFrame:
    """Iterate 5-core filter until fixed point."""
    while True:
        before = len(df)
        item_counts = df.groupby("movieId")["userId"].count()
        keep_items = item_counts[item_counts >= min_item].index
        df = df[df["movieId"].isin(keep_items)]

        user_counts = df.groupby("userId")["movieId"].count()
        keep_users = user_counts[user_counts >= min_user].index
        df = df[df["userId"].isin(keep_users)]

        if len(df) == before:
            break
    return df


def preprocess_ml20m(
    ratings_csv: str | Path,
    out_path: str | Path,
    min_user: int = 5,
    min_item: int = 5,
) -> ProcessedData:
    """Run preprocessing on ratings.csv and pickle the result."""
    ratings_csv = Path(ratings_csv)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[data] reading {ratings_csv} ...")
    df = pd.read_csv(ratings_csv, usecols=["userId", "movieId", "timestamp"])
    print(f"[data] raw: {len(df):,} ratings, {df.userId.nunique():,} users, "
          f"{df.movieId.nunique():,} movies")

    df = _five_core_filter(df, min_user=min_user, min_item=min_item)
    print(f"[data] 5-core: {len(df):,} ratings, {df.userId.nunique():,} users, "
          f"{df.movieId.nunique():,} movies")

    # Sort each user's interactions by timestamp.
    df = df.sort_values(["userId", "timestamp"], kind="stable")

    # Dense item ids 1..N, 0 reserved for padding.
    unique_movies = df["movieId"].unique()
    movie_id_to_idx = {m: i + 1 for i, m in enumerate(unique_movies)}
    df["item_idx"] = df["movieId"].map(movie_id_to_idx)

    unique_users = df["userId"].unique()
    user_id_to_idx = {u: i for i, u in enumerate(unique_users)}
    df["user_idx"] = df["userId"].map(user_id_to_idx)

    user_seq: Dict[int, List[int]] = {}
    user_session_cuts: Dict[int, List[int]] = {}
    user_timestamps: Dict[int, List[int]] = {}
    for uidx, group in df.groupby("user_idx", sort=False):
        user_seq[int(uidx)] = group["item_idx"].tolist()
        ts = group["timestamp"].to_numpy()
        user_timestamps[int(uidx)] = ts.tolist()
        gaps = np.diff(ts)
        cuts = (np.where(gaps > SESSION_GAP_SEC)[0] + 1).tolist()
        user_session_cuts[int(uidx)] = cuts

    n_users = len(user_seq)
    n_items = len(movie_id_to_idx)

    item_pop = np.zeros(n_items + 1, dtype=np.int64)
    for seq in user_seq.values():
        for i in seq:
            item_pop[i] += 1
    n_cuts_total = sum(len(c) for c in user_session_cuts.values())
    print(f"[data] n_users={n_users:,} n_items={n_items:,} "
          f"avg_seq_len={np.mean([len(s) for s in user_seq.values()]):.1f} "
          f"session_cuts={n_cuts_total:,}")

    processed = ProcessedData(
        user_seq=user_seq,
        n_users=n_users,
        n_items=n_items,
        item_pop=item_pop,
        movie_id_to_idx=movie_id_to_idx,
        user_id_to_idx=user_id_to_idx,
        user_session_cuts=user_session_cuts,
        user_timestamps=user_timestamps,
    )
    with open(out_path, "wb") as f:
        pickle.dump(processed, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[data] saved to {out_path}")
    return processed


def load_processed(path: str | Path) -> ProcessedData:
    with open(path, "rb") as f:
        return pickle.load(f)


def split_loo(
    user_seq: Dict[int, List[int]],
) -> Tuple[Dict[int, List[int]], Dict[int, int], Dict[int, int]]:
    """Leave-one-out: last → test, second-last → val, rest → train.

    Sequences with <3 items are dropped from val/test (cannot split).
    """
    train: Dict[int, List[int]] = {}
    val: Dict[int, int] = {}
    test: Dict[int, int] = {}
    for u, seq in user_seq.items():
        if len(seq) < 3:
            train[u] = seq
            continue
        train[u] = seq[:-2]
        val[u] = seq[-2]
        test[u] = seq[-1]
    return train, val, test


def left_pad(seq: Sequence[int], max_len: int) -> List[int]:
    seq = list(seq)
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [PAD_ID] * (max_len - len(seq)) + seq


def _shuffle_within_sessions(seq: List[int], cuts: List[int]) -> List[int]:
    """Shuffle item order within each 30-min session cluster, preserving cluster boundaries."""
    cuts_in_range = [c for c in cuts if 0 < c < len(seq)]
    boundaries = [0] + cuts_in_range + [len(seq)]
    out: List[int] = []
    for a, b in zip(boundaries, boundaries[1:]):
        chunk = seq[a:b]
        random.shuffle(chunk)
        out.extend(chunk)
    return out


class SeqTrainDataset(Dataset):
    """Shifted-sequence next-item training.

    For a user with train sequence [i_1, ..., i_n] of length n:
      input  = left_pad([i_1, ..., i_{n-1}], max_len)
      target = left_pad([i_2, ..., i_n],     max_len)
    Loss must be masked where input==PAD (these positions have no real history).

    Optional augmentations:
      shuffle_sessions: permute items within 30-min session clusters each call.
      random_window: sample a random contiguous window instead of always the last.
    """

    def __init__(
        self,
        train_seq: Dict[int, List[int]],
        max_len: int = 200,
        min_train_len: int = 2,
        session_cuts: Optional[Dict[int, List[int]]] = None,
        shuffle_sessions: bool = False,
        random_window: bool = False,
        user_timestamps: Optional[Dict[int, List[int]]] = None,
    ):
        self.users: List[int] = [u for u, s in train_seq.items() if len(s) >= min_train_len]
        self.train_seq = train_seq
        self.max_len = max_len
        self.session_cuts = session_cuts
        self.shuffle_sessions = shuffle_sessions
        self.random_window = random_window
        self.user_timestamps = user_timestamps

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        u = self.users[idx]
        seq = list(self.train_seq[u])
        # user_timestamps covers the full sequence including val/test items; slice to train only.
        ts: Optional[List[int]] = list(self.user_timestamps[u][:len(seq)]) if self.user_timestamps else None

        if self.shuffle_sessions and self.session_cuts is not None:
            cuts = self.session_cuts.get(u, [])
            if cuts:
                seq = _shuffle_within_sessions(seq, cuts)
                ts = None  # timestamps no longer aligned after shuffle

        if self.random_window and len(seq) > self.max_len + 1:
            start = random.randint(0, len(seq) - self.max_len - 1)
            seq = seq[start : start + self.max_len + 1]
            if ts is not None:
                ts = ts[start : start + self.max_len + 1]
        else:
            seq = seq[-(self.max_len + 1):]
            if ts is not None:
                ts = ts[-(self.max_len + 1):]

        inp = seq[:-1]
        tgt = seq[1:]
        item = {
            "user":   torch.tensor(u,   dtype=torch.long),
            "input":  torch.tensor(inp, dtype=torch.long),
            "target": torch.tensor(tgt, dtype=torch.long),
        }
        if ts is not None:
            inp_ts = ts[:-1]
            item["time_delta"] = torch.tensor(
                _seq_to_time_deltas(inp_ts, len(inp_ts)), dtype=torch.long
            )
        return item


class LengthCurriculumSampler(Sampler):
    """Linearly ramps from uniform to length-proportional user sampling.

    Epoch 0: all users equally likely.
    Epoch >= warmup_epochs: users sampled proportional to sequence length.
    Between: linear interpolation.
    """

    def __init__(self, seq_lens: np.ndarray, warmup_epochs: int):
        self.seq_lens = seq_lens.astype(np.float64)
        self.warmup_epochs = max(warmup_epochs, 1)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        t = min(self.epoch / self.warmup_epochs, 1.0)
        uniform = np.ones(len(self.seq_lens), dtype=np.float64)
        weights = (1.0 - t) * uniform + t * self.seq_lens
        weights /= weights.sum()
        indices = np.random.choice(len(self.seq_lens), size=len(self.seq_lens),
                                   replace=True, p=weights)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return len(self.seq_lens)


def make_fixed_collate(max_len: int):
    """Returns a collate_fn that always pads to exactly max_len (fixed input shape)."""
    def _collate(batch):
        has_td = "time_delta" in batch[0]
        users, inputs, targets, tds = [], [], [], []
        for b in batch:
            pad = max_len - b["input"].size(0)
            users.append(b["user"])
            inputs.append(F.pad(b["input"],  (pad, 0)))
            targets.append(F.pad(b["target"], (pad, 0)))
            if has_td:
                tds.append(F.pad(b["time_delta"], (pad, 0)))
        out = {"user": torch.stack(users), "input": torch.stack(inputs),
               "target": torch.stack(targets)}
        if has_td:
            out["time_delta"] = torch.stack(tds)
        return out
    return _collate


def pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Left-pad a batch of variable-length sequences to the next power of 2 >= batch max.

    Power-of-2 lengths reuse compiled CUDA kernels across batches (same shape →
    no recompilation) and align to tensor-core-friendly sizes.
    """
    max_L = max(b["input"].size(0) for b in batch)
    pad_L = 1 << (max_L - 1).bit_length()   # next power of 2 >= max_L
    has_td = "time_delta" in batch[0]
    users, inputs, targets, tds = [], [], [], []
    for b in batch:
        pad = pad_L - b["input"].size(0)
        users.append(b["user"])
        inputs.append(F.pad(b["input"],  (pad, 0)))
        targets.append(F.pad(b["target"], (pad, 0)))
        if has_td:
            tds.append(F.pad(b["time_delta"], (pad, 0)))
    out = {"user": torch.stack(users), "input": torch.stack(inputs),
           "target": torch.stack(targets)}
    if has_td:
        out["time_delta"] = torch.stack(tds)
    return out


class BucketBatchSampler(BatchSampler):
    """Batches sequences by approximate length with a fixed token budget.

    Batch size adapts per bucket: B = token_budget // next_pow2(bucket_max_len).
    This keeps neg_emb memory [B, L, K, d] constant regardless of sequence length,
    so long-sequence users train safely without OOM.

    Adds uniform noise of [0, bucket_width) to lengths before sorting so batch
    order and composition vary across epochs.
    """

    def __init__(
        self,
        seq_lens: np.ndarray,
        token_budget: int,
        drop_last: bool = True,
        bucket_width: int = 16,
        min_batch: int = 16,
    ):
        self.seq_lens     = np.asarray(seq_lens, dtype=np.float32)
        self.token_budget = token_budget
        self.drop_last    = drop_last
        self.bucket_width = bucket_width
        self.min_batch    = min_batch

    def _batch_size(self, L: int) -> int:
        pad_L = 1 << max(1, (L - 1)).bit_length()
        return max(self.min_batch, self.token_budget // pad_L)

    def __iter__(self):
        noise = np.random.uniform(0, self.bucket_width, len(self.seq_lens))
        order = np.argsort(self.seq_lens + noise)
        batches: list = []
        i = 0
        while i < len(order):
            bs = self._batch_size(int(self.seq_lens[order[i]]))
            batch = order[i : i + bs].tolist()
            if not self.drop_last or len(batch) == bs:
                batches.append(batch)
            i += bs
        np.random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        sorted_lens = np.sort(self.seq_lens)
        i = count = 0
        while i < len(sorted_lens):
            bs = self._batch_size(int(sorted_lens[i]))
            count += 1
            i += bs
        return count


class SeqEvalDataset(Dataset):
    """Holds (user, input_history, target_item) for val or test.

    Input history is train_seq + (val target if eval_split == 'test') left-padded.
    """

    def __init__(
        self,
        train_seq: Dict[int, List[int]],
        target_map: Dict[int, int],
        max_len: int = 200,
        prepend_seq: Dict[int, List[int]] | None = None,
        user_timestamps: Optional[Dict[int, List[int]]] = None,
    ):
        self.users: List[int] = [u for u in target_map if u in train_seq]
        self.train_seq = train_seq
        self.target_map = target_map
        self.max_len = max_len
        self.prepend_seq = prepend_seq or {}
        self.user_timestamps = user_timestamps

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        u = self.users[idx]
        history = list(self.train_seq[u])
        ts: Optional[List[int]] = list(self.user_timestamps[u]) if self.user_timestamps else None

        if u in self.prepend_seq:
            extra = list(self.prepend_seq[u])
            history = history + extra
            if ts is not None:
                n_train = len(self.train_seq[u])
                extra_ts = list(self.user_timestamps[u][n_train : n_train + len(extra)])
                ts = ts + extra_ts

        history = history[-self.max_len:]
        if ts is not None:
            ts = ts[-self.max_len:]

        inp = left_pad(history, self.max_len)
        item = {
            "user": torch.tensor(u, dtype=torch.long),
            "input": torch.tensor(inp, dtype=torch.long),
            "target": torch.tensor(self.target_map[u], dtype=torch.long),
        }
        if ts is not None:
            pad = self.max_len - len(ts)
            buckets = [0] * pad + _seq_to_time_deltas(ts, len(ts))
            item["time_delta"] = torch.tensor(buckets, dtype=torch.long)
        return item


def build_user_seen_lookup(
    user_seq: Dict[int, List[int]],
) -> Dict[int, np.ndarray]:
    """For full-catalog evaluation with filter-seen: u -> sorted unique item ids.

    Used to mask already-watched items at scoring time.
    """
    return {u: np.unique(np.asarray(s, dtype=np.int64)) for u, s in user_seq.items()}

"""Main training loop.

Single entry point: `python -m src.train --config configs/sasrec.yaml --out runs/sasrec_modern`

Capabilities:
- model registry with seven variants (SASRec / NextItNet / FMLP-Rec /
  Linear-attn SASRec / FNet-hybrid / Mamba4Rec — all consume the same
  `score_pairs` / `score_all` interface)
- gBCE or sampled softmax loss with optional logQ correction
- popularity-weighted or uniform negative sampling, with in-batch negatives
- bf16 mixed precision (preferred on A100), AdamW with no-decay group, warmup-
  cosine LR schedule, gradient clipping, EMA, optional snapshot ensembling
- SSE-PT input augmentation (toggleable)
- side-info module (genres + decade + Tag Genome scores) shared with model
- vectorised full-catalog evaluation with filter-seen
- StackRec multi-stage option for SASRec-family models (`stackrec.stages`)

The script writes `result.json` and `log.jsonl` to `--out` so that
`python -m src.results` can aggregate every run into a comparison table.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .augment import sse_pt_augment
from .data import (
    BucketBatchSampler,
    LengthCurriculumSampler,
    ProcessedData,
    SeqEvalDataset,
    SeqTrainDataset,
    load_processed,
    pad_collate,
    preprocess_ml20m,
    split_loo,
)
from .eval import build_padded_seen, evaluate_full_catalog, format_metrics
from .losses import (
    GBCELoss,
    SampledSoftmaxLoss,
    make_logq_popularity,
    make_logq_uniform,
    popularity_weights,
    sample_negatives,
)
from .models import build_model
from .sideinfo import SideInfoEmbedding, build_sideinfo
from .stackrec import stack_blocks, stackrec_optim_config
from .utils import (
    EMA,
    JSONLogger,
    SnapshotEnsemble,
    WarmupCosineLR,
    count_params,
    enable_tf32,
    set_seed,
    split_decay_params,
)


# ============================== CLI ====================================== #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--data-dir", default="data", type=str,
                   help="root with ml-20m/ subdir + processed.pkl cache")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=None, type=int)
    p.add_argument("--no-test", action="store_true",
                   help="skip final test evaluation (debug only)")
    p.add_argument("--max-epochs", default=None, type=int,
                   help="override config.training.max_epochs")
    return p.parse_args()


# ============================== Config helpers ============================ #
def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def device_or_cpu(req: str) -> torch.device:
    if req == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(req)


# ============================== Data setup =============================== #
def get_processed(data_dir: Path) -> ProcessedData:
    cache = data_dir / "processed.pkl"
    if cache.exists():
        return load_processed(cache)
    csv = data_dir / "ml-20m" / "ratings.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"Expected {csv}. Run scripts/download_data.sh first.")
    return preprocess_ml20m(csv, cache)


def build_side_module(cfg: Dict[str, Any], proc: ProcessedData, data_dir: Path,
                      d: int) -> Optional[SideInfoEmbedding]:
    if not cfg.get("model", {}).get("use_side", False):
        return None
    movies_csv = data_dir / "ml-20m" / "movies.csv"
    genome_csv = data_dir / "ml-20m" / "genome-scores.csv"
    if not movies_csv.exists():
        raise FileNotFoundError(f"Missing {movies_csv} for side-info.")
    use_genome = cfg["model"].get("side_use_genome", True)
    side = build_sideinfo(
        movies_csv,
        proc.movie_id_to_idx,
        proc.n_items,
        genome_scores_csv=(genome_csv if use_genome else None),
    )
    return SideInfoEmbedding(
        d=d, side=side,
        use_genome=use_genome,
        dropout=cfg["model"].get("side_dropout", 0.1),
    )


def build_user_seen_split(
    train_seq: Dict[int, List[int]],
    val_target: Dict[int, int],
    n_users_total: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Padded seen tensors for val and test eval.

    For val: filter only train items.
    For test: filter train + val target.
    """
    seen_val = {u: np.asarray(np.unique(s), dtype=np.int64)
                for u, s in train_seq.items()}
    seen_test = {}
    for u, s in train_seq.items():
        items = list(s)
        if u in val_target:
            items.append(val_target[u])
        seen_test[u] = np.asarray(np.unique(items), dtype=np.int64)
    cap = max(
        max((len(v) for v in seen_val.values()), default=1),
        max((len(v) for v in seen_test.values()), default=1),
    )
    cap = min(cap, 3000)
    val_pad = build_padded_seen(seen_val, n_users_total, cap=cap)
    test_pad = build_padded_seen(seen_test, n_users_total, cap=cap)
    return val_pad, test_pad


# ============================== Loss factory ============================= #
def build_loss(
    name: str,
    n_items: int,
    n_neg: int,
    cfg: Dict[str, Any],
    log_q: Optional[torch.Tensor],
    device: torch.device,
) -> nn.Module:
    if name == "gbce":
        t = cfg.get("gbce_t", 0.75)
        return GBCELoss(n_items=n_items, n_neg=n_neg, t=t).to(device)
    if name == "sampled_softmax":
        return SampledSoftmaxLoss(log_q_pos=log_q).to(device)
    raise ValueError(f"unknown loss '{name}'")


# ============================== One epoch ================================ #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineLR,
    ema: Optional[EMA],
    device: torch.device,
    cfg: Dict[str, Any],
    pop_weights: Optional[torch.Tensor],
    log_q: Optional[torch.Tensor],
    autocast_dtype: torch.dtype,
    log: JSONLogger,
    epoch: int,
    global_step: int,
) -> Tuple[int, float, float]:
    """Returns (global_step, epoch_avg_loss, epoch_avg_eff_len)."""
    model.train()
    n_neg = cfg["loss"]["n_neg"]
    n_items = cfg["_n_items"]
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    log_every = cfg["training"].get("log_every_steps", 100)
    use_inbatch = cfg["loss"].get("in_batch_negatives", False)
    sse_p = cfg.get("aug", {}).get("sse_pt_p", 0.0)

    t0 = time.time()
    running = 0.0
    n_steps = 0
    epoch_loss_sum = 0.0
    epoch_eff_len_sum = 0.0
    epoch_n_steps = 0
    for batch in loader:
        users = batch["user"].to(device, non_blocking=True)
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if sse_p > 0:
            inputs = sse_pt_augment(inputs, n_items=n_items, p=sse_p)

        neg_ids = sample_negatives(
            target_mask=(inputs != 0),
            n_neg=n_neg,
            n_items=n_items,
            device=device,
            pop_weights=pop_weights,
        )

        if use_inbatch:
            # Concatenate batch positives as extra negatives. Per position, the
            # other (B*L)-1 positives in the batch act as cheap shared negatives.
            # We just append the unique non-PAD batch targets.
            bat_negs = targets[(inputs != 0)]                  # 1D non-pad targets
            uniq = torch.unique(bat_negs)
            if uniq.numel() > 0:
                B, L, K = neg_ids.shape
                extra = uniq[torch.randint(0, uniq.numel(), (B, L, min(64, uniq.numel())),
                                           device=device)]
                neg_ids = torch.cat([neg_ids, extra], dim=-1)

        autocast = torch.autocast(device_type=device.type, dtype=autocast_dtype) \
            if device.type == "cuda" else nullcontext()
        with autocast:
            pos_logits, neg_logits, mask = model.score_pairs(inputs, targets, neg_ids)
            if isinstance(loss_fn, SampledSoftmaxLoss):
                loss = loss_fn(pos_logits, neg_logits, mask, targets, neg_ids)
            else:
                loss = loss_fn(pos_logits, neg_logits, mask)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            log.log(tag="nan_detected", epoch=epoch, step=global_step,
                    loss=loss_val)
            raise FloatingPointError(
                f"loss is {loss_val!r} at step {global_step}; aborting. "
                f"Check input pipeline / autocast dtype / lr.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = 0.0
        if grad_clip and grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        running += loss_val
        epoch_loss_sum += loss_val
        epoch_eff_len_sum += float((inputs != 0).float().sum(dim=1).mean().item())
        n_steps += 1
        epoch_n_steps += 1
        global_step += 1
        if global_step % log_every == 0:
            avg_loss = round(running / n_steps, 4)
            cur_lr = round(scheduler.get_lr(), 6)
            log.log(
                tag="train_step",
                epoch=epoch,
                step=global_step,
                loss=avg_loss,
                lr=cur_lr,
                grad_norm=round(grad_norm, 4),
            )
            print(f"[step {global_step:>7d} / ep {epoch:>3d}] "
                  f"loss={avg_loss:.4f}  lr={cur_lr:.2e}  grad_norm={grad_norm:.3f}")
            running, n_steps = 0.0, 0

    epoch_t = round(time.time() - t0, 1)
    epoch_avg_loss = epoch_loss_sum / max(epoch_n_steps, 1)
    epoch_avg_eff_len = epoch_eff_len_sum / max(epoch_n_steps, 1)
    log.log(
        tag="train_epoch",
        epoch=epoch,
        steps=global_step,
        wallclock_s=epoch_t,
        avg_loss=round(epoch_avg_loss, 4),
        avg_eff_len=round(epoch_avg_eff_len, 1),
    )
    print(f"[epoch {epoch} done] wall={epoch_t}s  total_steps={global_step}  "
          f"avg_loss={epoch_avg_loss:.4f}  avg_eff_len={epoch_avg_eff_len:.1f}")
    return global_step, epoch_avg_loss, epoch_avg_eff_len


# ============================== Main =================================== #
def _print_startup(cfg: Dict[str, Any], device: torch.device, autocast_dtype: torch.dtype,
                   n_params: int, n_train_users: int, n_items: int, total_steps: int):
    print("=" * 72)
    print(f"[startup] model      : {cfg['model']['name']}  d={cfg['model'].get('d')}  "
          f"n_blocks={cfg['model'].get('n_blocks', cfg['model'].get('block_num', '?'))}")
    print(f"[startup] loss       : {cfg['loss']['name']}  n_neg={cfg['loss']['n_neg']}  "
          f"sampling={cfg['loss'].get('sampling', 'uniform')}  in_batch={cfg['loss'].get('in_batch_negatives', False)}")
    print(f"[startup] params     : {n_params:,}")
    print(f"[startup] dataset    : {n_train_users:,} users, {n_items:,} items")
    print(f"[startup] training   : batch={cfg['training']['batch_size']}  "
          f"lr={cfg['training']['lr']}  wd={cfg['training'].get('weight_decay', 0)}  "
          f"epochs<={cfg['training']['max_epochs']}  patience={cfg['training'].get('early_stop_patience', '∞')}")
    print(f"[startup] device     : {device}  autocast={autocast_dtype}")
    print(f"[startup] total_steps: {total_steps:,}  warmup={int(cfg['training'].get('warmup_frac', 0.05) * total_steps):,}")
    if device.type == "cuda":
        try:
            free, total = torch.cuda.mem_get_info(device)
            print(f"[startup] GPU       : {torch.cuda.get_device_name(device)}  "
                  f"free={free/1e9:.1f}GB / total={total/1e9:.1f}GB")
        except Exception:
            pass
    print("=" * 72)


def run_training(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    enable_tf32()
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    device = device_or_cpu(args.device)
    autocast_dtype = torch.bfloat16 if (
        device.type == "cuda" and torch.cuda.is_bf16_supported()
    ) else torch.float32

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = JSONLogger(out_dir)
    log.log(tag="config", cfg=cfg, seed=seed, device=str(device),
            autocast_dtype=str(autocast_dtype))

    # ---- Data ----
    data_dir = Path(args.data_dir).resolve()
    proc = get_processed(data_dir)
    train_seq, val_target, test_target = split_loo(proc.user_seq)
    n_users_total = max(proc.n_users, max(proc.user_seq.keys()) + 1)
    cfg["_n_items"] = proc.n_items

    max_len = cfg["model"]["max_len"]
    aug_cfg = cfg.get("aug", {})
    shuffle_sessions = aug_cfg.get("shuffle_sessions", False)
    random_window    = aug_cfg.get("random_window", False)
    session_cuts = proc.user_session_cuts if shuffle_sessions else None
    train_ds = SeqTrainDataset(
        train_seq, max_len=max_len, min_train_len=2,
        session_cuts=session_cuts,
        shuffle_sessions=shuffle_sessions,
        random_window=random_window,
    )
    val_ds = SeqEvalDataset(train_seq, val_target, max_len=max_len)
    test_ds = SeqEvalDataset(train_seq, test_target, max_len=max_len,
                             prepend_seq={u: [v] for u, v in val_target.items()})

    bs_train = cfg["training"]["batch_size"]
    bs_eval = cfg["training"].get("eval_batch_size", 512)
    nw = cfg["training"].get("num_workers", 4)

    max_epochs = args.max_epochs or cfg["training"]["max_epochs"]
    warmup_frac = cfg["training"].get("curriculum_warmup_frac", 0.0)
    warmup_epochs = int(warmup_frac * max_epochs)
    seq_lens = np.array([len(train_seq[u]) for u in train_ds.users])
    curriculum_sampler = LengthCurriculumSampler(seq_lens, warmup_epochs=warmup_epochs)

    # token_budget = B * L_pad kept constant → neg_emb memory flat across all lengths.
    # Default: batch_size * max_len (same memory as fixed-length baseline).
    token_budget = cfg["training"].get("token_budget", bs_train * max_len)

    _loader_kw = dict(num_workers=nw, pin_memory=(device.type == "cuda"),
                      persistent_workers=(nw > 0), collate_fn=pad_collate)
    if warmup_epochs > 0:
        train_loader = DataLoader(
            train_ds, batch_size=bs_train, sampler=curriculum_sampler,
            drop_last=True, **_loader_kw,
        )
    else:
        bucket_sampler = BucketBatchSampler(seq_lens, token_budget=token_budget, drop_last=True)
        train_loader = DataLoader(train_ds, batch_sampler=bucket_sampler, **_loader_kw)
    val_loader = DataLoader(
        val_ds, batch_size=bs_eval, shuffle=False, drop_last=False,
        num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=(nw > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs_eval, shuffle=False, drop_last=False,
        num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=(nw > 0),
    )
    val_seen, test_seen = build_user_seen_split(train_seq, val_target, n_users_total)

    # ---- Model + side ----
    side = build_side_module(cfg, proc, data_dir, d=cfg["model"]["d"])
    model = build_model(cfg["model"], n_items=proc.n_items, side_module=side).to(device)
    if cfg.get("__warm_start__"):
        ws = torch.load(cfg["__warm_start__"], map_location=device)
        sd = ws.get("state_dict", ws)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.log(tag="warm_start", source=cfg["__warm_start__"],
                missing=len(missing), unexpected=len(unexpected))
    n_params = count_params(model)
    log.log(tag="model", **n_params, model_name=cfg["model"]["name"])
    n_train_users = len([u for u, s in train_seq.items() if len(s) >= 2])

    # ---- Loss ----
    pop_weights = None
    log_q = None
    sampling = cfg["loss"].get("sampling", "uniform")
    if sampling == "popularity":
        pop_weights = popularity_weights(proc.item_pop, exponent=0.75).to(device)
        log_q = make_logq_popularity(proc.item_pop, n_neg=cfg["loss"]["n_neg"]).to(device)
    else:
        log_q = make_logq_uniform(proc.n_items, n_neg=cfg["loss"]["n_neg"], device=device)
    loss_fn = build_loss(
        cfg["loss"]["name"], proc.n_items, cfg["loss"]["n_neg"],
        cfg["loss"], log_q if cfg["loss"]["name"] == "sampled_softmax" else None,
        device,
    )

    # ---- Optim ----
    wd = cfg["training"].get("weight_decay", 1e-2)
    base_lr = cfg["training"]["lr"]
    optimizer = torch.optim.AdamW(
        split_decay_params(model, weight_decay=wd),
        lr=base_lr, betas=(0.9, 0.98), eps=1e-8,
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(cfg["training"].get("warmup_frac", 0.05) * total_steps)
    scheduler = WarmupCosineLR(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
        min_lr_ratio=cfg["training"].get("min_lr_ratio", 0.01),
    )
    ema_decay = cfg["training"].get("ema_decay", 0.999)
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    _print_startup(cfg, device, autocast_dtype, n_params["total"],
                   n_train_users, proc.n_items, total_steps)

    # ---- Snapshot ensembling (optional) ----
    use_snapshots = cfg["training"].get("snapshot_ensemble", False)
    snap = SnapshotEnsemble(out_dir / "snapshots", max_snapshots=3) if use_snapshots else None

    # ---- Training loop ----
    eval_every = cfg["training"].get("eval_every_epochs", 1)
    patience = cfg["training"].get("early_stop_patience", 10)
    best_metric = -1.0
    best_epoch = -1
    no_improve = 0
    global_step = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, max_epochs + 1):
        if warmup_epochs > 0:
            curriculum_sampler.set_epoch(epoch - 1)
        global_step, _, _ = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, ema,
            device, cfg, pop_weights, log_q, autocast_dtype, log,
            epoch=epoch, global_step=global_step,
        )

        if epoch % eval_every == 0:
            if ema is not None:
                ema.apply_to(model)
            val_metrics = evaluate_full_catalog(
                model, val_loader, val_seen, proc.n_items,
                ks=tuple(cfg["eval"].get("ks", [5, 10, 20])), device=str(device),
            )
            if ema is not None:
                ema.restore(model)
            log.log(tag="val", epoch=epoch, **val_metrics)
            print(f"[epoch {epoch}] val: {format_metrics(val_metrics)}")
            cur = val_metrics.get("NDCG@10", 0.0)
            if cur > best_metric + 1e-6:
                best_metric = cur
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if snap is not None and epoch >= max_epochs * 0.5:
                    snap.save(model, tag=f"e{epoch}")
                torch.save(
                    {
                        "state_dict": best_state,
                        "ema_state_dict": ema.state_dict() if ema else None,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "config": cfg,
                    },
                    out_dir / "best.pt",
                )
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.log(tag="early_stop", epoch=epoch, best_epoch=best_epoch)
                    break

    # ---- Final test ----
    if best_state is not None:
        model.load_state_dict(best_state)
    if ema is not None:
        ema.apply_to(model)
    if snap is not None and len(snap.paths) >= 2:
        snap.average_into(model)

    test_metrics: Dict[str, float] = {}
    if not args.no_test:
        test_metrics = evaluate_full_catalog(
            model, test_loader, test_seen, proc.n_items,
            ks=tuple(cfg["eval"].get("ks", [5, 10, 20])), device=str(device),
        )
        log.log(tag="test", **test_metrics)
        print(f"[FINAL test] {format_metrics(test_metrics)}")

    n_blocks_eff = cfg["model"].get("n_blocks", -1)
    if cfg["model"]["name"] == "nextitnet":
        n_blocks_eff = cfg["model"].get("block_num", 1) * len(cfg["model"].get("dilations", []))
    summary = {
        "exp_name": out_dir.name,
        "config_path": args.config,
        "model": cfg["model"]["name"],
        "n_blocks": n_blocks_eff,
        "d": cfg["model"].get("d", -1),
        "params": n_params,
        "loss": cfg["loss"]["name"],
        "n_neg": cfg["loss"]["n_neg"],
        "best_epoch": best_epoch,
        "val_NDCG@10": round(best_metric, 4),
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()
                         if isinstance(v, float)},
        "device": str(device),
        "autocast_dtype": str(autocast_dtype),
    }
    log.write_result(summary)
    print(f"[done] {out_dir}/result.json — best NDCG@10={best_metric:.4f} @ epoch {best_epoch}")
    return summary


def run_stackrec(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Multi-stage StackRec training: train n0 blocks, double, train, double, train.

    StackRec stages are configured via cfg['stackrec']:
        stages: [4, 8, 16]            # block counts at each stage
        epochs: [30, 15, 10]          # epochs per stage
        mode: adjacent | sequential
    """
    base_out = Path(args.out)
    block_sizes = cfg["stackrec"]["stages"]
    epochs_per_stage = cfg["stackrec"]["epochs"]
    mode = cfg["stackrec"].get("mode", "adjacent")
    if len(block_sizes) != len(epochs_per_stage):
        raise ValueError("stackrec.stages and stackrec.epochs must be same length")

    device = device_or_cpu(args.device)
    set_seed(cfg.get("seed", 42))

    last_summary: Dict[str, Any] = {}
    last_state: Optional[Dict[str, torch.Tensor]] = None
    for stage, (nb, ep) in enumerate(zip(block_sizes, epochs_per_stage)):
        stage_dir = base_out / f"stage{stage}_L{nb}"
        stage_args = argparse.Namespace(**vars(args))
        stage_args.out = str(stage_dir)
        stage_args.max_epochs = ep
        stage_cfg = {**cfg}
        stage_cfg["model"] = {**cfg["model"], "n_blocks": nb}
        stage_cfg["training"] = {**cfg["training"]}
        adj = stackrec_optim_config(stage=stage, base_lr=cfg["training"]["lr"])
        stage_cfg["training"]["lr"] = adj["lr"]
        stage_cfg["training"]["warmup_frac"] = max(
            adj["warmup_frac"], stage_cfg["training"].get("warmup_frac", 0.05))
        stage_cfg["training"]["max_epochs"] = ep
        stage_cfg.pop("stackrec", None)

        # If we have a trained model from previous stage, stack and warm-start.
        if last_state is not None and stage > 0:
            proc = get_processed(Path(args.data_dir).resolve())
            stage_cfg["_n_items"] = proc.n_items
            side = build_side_module(stage_cfg, proc, Path(args.data_dir).resolve(),
                                     d=stage_cfg["model"]["d"])
            prev_cfg_model = {**stage_cfg["model"], "n_blocks": block_sizes[stage - 1]}
            prev_model = build_model(prev_cfg_model, n_items=proc.n_items, side_module=side)
            prev_model.load_state_dict(last_state)
            stacked = stack_blocks(prev_model, mode=mode)
            torch.save(
                {"state_dict": stacked.state_dict(), "from_stage": stage - 1},
                base_out / f"stack_init_stage{stage}.pt",
            )
            stage_cfg["__warm_start__"] = str(base_out / f"stack_init_stage{stage}.pt")
            del prev_model, stacked

        last_summary = run_training(stage_cfg, stage_args)
        ckpt = torch.load(stage_dir / "best.pt", map_location="cpu")
        last_state = ckpt["state_dict"]

    return last_summary


def main():
    import traceback
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if "stackrec" in cfg:
            run_stackrec(cfg, args)
        else:
            run_training(cfg, args)
    except Exception as e:                                                  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[FAILED] {e}\n{tb}")
        try:
            (out_dir / "FAILURE.txt").write_text(
                f"exception: {type(e).__name__}: {e}\n\n{tb}",
                encoding="utf-8",
            )
            # Drop a result.json marker so the aggregator still picks it up.
            import json
            (out_dir / "result.json").write_text(
                json.dumps({
                    "exp_name": out_dir.name,
                    "config_path": args.config,
                    "status": "FAILED",
                    "error": f"{type(e).__name__}: {e}",
                }, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()

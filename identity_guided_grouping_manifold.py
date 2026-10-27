#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional at runtime
    tqdm = None

from src import data, plotting
from src.io_utils import (
    build_transfer_case_name,
    finalize_bundle_args,
    infer_default_data_root,
    parse_int_list,
    resolve_device,
    safe_mean,
    save_json,
    to_jsonable,
)
from src.models import IdentityGuidedGroupingEncoder
from src.repro import snapshot_clean_pack_entrypoint
from src.identity_guided_grouping import build_identity_guided_grouping_encoder, build_time_batch, prepare_subset_cache
from src.transfer_context import (
    BaseFishContext,
    CaseFishContext,
    build_case_contexts,
    load_base_fish_context,
    sample_neuron_subset_indices,
)


@dataclass
class ManifoldEmbedding:
    fish_id: int
    method: str
    latent_td: np.ndarray
    subset_indices: np.ndarray | None
    metrics: dict
    metadata: dict
    latent_subset1_td: np.ndarray | None = None
    metrics_subset1: dict | None = None


def build_train_pos_radius_sampler(args: argparse.Namespace) -> dict | None:
    choices_text = str(getattr(args, "train_pos_radius_choices", "")).strip()
    if choices_text:
        choices = [int(v) for v in parse_int_list(choices_text) if int(v) >= 1]
        if choices:
            return {"mode": "choices", "choices": np.asarray(sorted(set(choices)), dtype=np.int64)}
    max_radius = int(getattr(args, "train_pos_radius_max", 0) or 0)
    if max_radius >= 1:
        half_life = max(1e-6, float(getattr(args, "train_pos_radius_half_life", 25.0)))
        radii = np.arange(1, int(max_radius) + 1, dtype=np.int64)
        weights = np.exp(-np.log(2.0) * (radii.astype(np.float64) - 1.0) / half_life)
        weights = np.asarray(weights / np.clip(weights.sum(), 1e-12, None), dtype=np.float64)
        return {
            "mode": "exp_range",
            "radii": radii,
            "weights": weights,
            "half_life": float(half_life),
        }
    return None


def sample_train_pos_radius(epoch_rng: np.random.Generator, args: argparse.Namespace, sampler: dict | None) -> int:
    if sampler is None:
        return int(args.pos_radius)
    mode = str(sampler.get("mode", "")).strip().lower()
    if mode == "choices":
        return int(epoch_rng.choice(sampler["choices"]))
    if mode == "exp_range":
        return int(epoch_rng.choice(sampler["radii"], p=sampler["weights"]))
    return int(args.pos_radius)


def iter_with_tqdm(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def build_manifold_model(
    reference_ctx: CaseFishContext,
    *,
    latent_dim: int,
    args: argparse.Namespace,
) -> IdentityGuidedGroupingEncoder:
    return build_identity_guided_grouping_encoder(
        reference_ctx,
        d_model=int(latent_dim),
        args=args,
        normalize_out=bool(args.normalize_latent),
    )


def window_valid_time_indices(ctx: CaseFishContext, model: torch.nn.Module) -> np.ndarray:
    t_len = int(ctx.base.fish.neural_nt.shape[1])
    wl = int(getattr(model, "window_left", 0))
    wr = int(getattr(model, "window_right", 0))
    if bool(getattr(model, "_time_offset_window_clamp", False)):
        start = int(wl)
        end = int(t_len - wr)
        if end <= start:
            raise ValueError(
                f"No valid time indices for fish {ctx.base.fish.fish_id}: T={t_len}, wl={wl}, wr={wr}"
            )
        return np.arange(start, end, dtype=np.int64)
    return np.arange(t_len, dtype=np.int64)


def split_tail_indices(full_idx: np.ndarray, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(full_idx, dtype=np.int64).reshape(-1)
    if idx.size < 4:
        raise ValueError(f"Need at least 4 time indices for train/val split, got {idx.size}")
    n_val = max(1, int(round(float(idx.size) * float(val_frac))))
    n_val = min(int(idx.size) - 1, int(n_val))
    n_train = int(idx.size) - int(n_val)
    return idx[:n_train].copy(), idx[n_train:].copy()


def iter_anchor_batches(idx: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    arr = np.asarray(idx, dtype=np.int64).reshape(-1)
    if arr.size <= 0:
        return []
    perm = arr[np.asarray(rng.permutation(arr.size), dtype=np.int64)]
    batches: list[np.ndarray] = []
    bs = max(1, int(batch_size))
    for start in range(0, perm.size, bs):
        batches.append(np.asarray(perm[start : start + bs], dtype=np.int64))
    return batches


def prepare_subset_context(
    model: torch.nn.Module,
    *,
    ctx: CaseFishContext,
    neuron_idx: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor | None, dict | None]:
    return prepare_subset_cache(
        model,
        ctx=ctx,
        neuron_indices=neuron_idx,
        device=device,
    )


def forward_unique_times(
    model: torch.nn.Module,
    *,
    x_nt_ctx: torch.Tensor | dict[str, torch.Tensor],
    static_t: torch.Tensor | None,
    static_cache: dict | None,
    time_indices: np.ndarray,
    device: torch.device,
    chunk_size_t: int,
) -> tuple[np.ndarray, torch.Tensor]:
    unique_t = np.asarray(np.unique(np.asarray(time_indices, dtype=np.int64)), dtype=np.int64)
    if unique_t.size <= 0:
        return unique_t, model.null_static[:0]
    chunk = max(1, int(chunk_size_t))
    z_chunks: list[torch.Tensor] = []
    for start in range(0, unique_t.size, chunk):
        t_chunk = unique_t[start : start + chunk]
        x_btn = build_time_batch(model, x_nt_ctx=x_nt_ctx, time_indices=t_chunk, device=device)
        z_chunks.append(data.model_forward_maybe_cached(model, x_btn, static_t, static_cache))
    z_ud = torch.cat(z_chunks, dim=0)
    return unique_t, z_ud


def sample_time_offset_triplets(
    *,
    anchor_t: np.ndarray,
    train_t: np.ndarray,
    time_offset: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(anchor_t, dtype=np.int64).reshape(-1)
    pool = np.asarray(train_t, dtype=np.int64).reshape(-1)
    if a.size == 0 or pool.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    off = max(1, int(time_offset))
    pool_set = set(int(v) for v in pool.tolist())
    valid_ref = np.asarray([int(t) for t in pool.tolist() if int(t) + off in pool_set], dtype=np.int64)
    if valid_ref.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    n_ref = int(a.size)
    sampled = np.asarray(rng.choice(valid_ref, size=2 * n_ref, replace=True), dtype=np.int64)
    ref_t = sampled[:n_ref].copy()
    neg_t = sampled[n_ref:].copy()
    pos_t = ref_t + off
    return ref_t, pos_t, neg_t


def batch_prior_info_nce(
    z_a: torch.Tensor,
    z_p: torch.Tensor,
    z_n: torch.Tensor,
    *,
    metric: str,
    temperature: float,
) -> torch.Tensor:
    if z_a.ndim != 2 or z_p.ndim != 2 or z_n.ndim != 2:
        return z_a.sum() * 0.0
    if z_a.shape[0] == 0 or z_p.shape[0] == 0 or z_n.shape[0] == 0:
        return z_a.sum() * 0.0
    temp = max(1e-6, float(temperature))
    if str(metric) == "euclidean":
        s_pos = -torch.sum((z_a - z_p) ** 2, dim=1) / temp
        a_sq = torch.sum(z_a ** 2, dim=1, keepdim=True)
        n_sq = torch.sum(z_n ** 2, dim=1, keepdim=True).transpose(0, 1)
        s_neg = -(a_sq + n_sq - 2.0 * (z_a @ z_n.transpose(0, 1))) / temp
    else:
        a_n = torch.nn.functional.normalize(z_a, dim=1)
        p_n = torch.nn.functional.normalize(z_p, dim=1)
        n_n = torch.nn.functional.normalize(z_n, dim=1)
        s_pos = torch.sum(a_n * p_n, dim=1) / temp
        s_neg = (a_n @ n_n.transpose(0, 1)) / temp
    c = s_neg.max(dim=1, keepdim=True).values.detach()
    s_pos_shift = s_pos - c.squeeze(1)
    s_neg_shift = s_neg - c
    return (-s_pos_shift + torch.logsumexp(s_neg_shift, dim=1)).mean()


def compute_batch_contrastive_loss(
    model: torch.nn.Module,
    *,
    ctx: CaseFishContext,
    anchor_t: np.ndarray,
    pool_t: np.ndarray,
    neuron_idx: np.ndarray,
    device: torch.device,
    metric: str,
    temperature: float,
    pos_radius: int,
    forward_chunk_size_t: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict]:
    ref_t, pos_t, neg_t = sample_time_offset_triplets(
        anchor_t=np.asarray(anchor_t, dtype=np.int64),
        train_t=np.asarray(pool_t, dtype=np.int64),
        time_offset=int(pos_radius),
        rng=rng,
    )
    if ref_t.size <= 0:
        return model.null_static.sum() * 0.0, {"num_pairs": 0}
    x_nt_ctx, static_t, static_cache = prepare_subset_context(model, ctx=ctx, neuron_idx=neuron_idx, device=device)
    all_t = np.concatenate([ref_t, pos_t, neg_t], axis=0)
    unique_t, z_ud = forward_unique_times(
        model,
        x_nt_ctx=x_nt_ctx,
        static_t=static_t,
        static_cache=static_cache,
        time_indices=all_t,
        device=device,
        chunk_size_t=int(forward_chunk_size_t),
    )
    time_to_idx = {int(t): i for i, t in enumerate(np.asarray(unique_t, dtype=np.int64).tolist())}
    dev = z_ud.device
    a_idx = np.asarray([time_to_idx[int(t)] for t in np.asarray(ref_t, dtype=np.int64)], dtype=np.int64)
    p_idx = np.asarray([time_to_idx[int(t)] for t in np.asarray(pos_t, dtype=np.int64)], dtype=np.int64)
    n_idx = np.asarray([time_to_idx[int(t)] for t in np.asarray(neg_t, dtype=np.int64)], dtype=np.int64)
    z_a = z_ud[torch.from_numpy(a_idx).to(dev)]
    z_p = z_ud[torch.from_numpy(p_idx).to(dev)]
    z_n = z_ud[torch.from_numpy(n_idx).to(dev)]
    loss = batch_prior_info_nce(
        z_a,
        z_p,
        z_n,
        metric=str(metric),
        temperature=float(temperature),
    )
    return loss, {
        "num_pairs": int(ref_t.size),
        "num_unique_times": int(unique_t.size),
        "num_negatives_per_anchor": int(neg_t.size),
    }


def evaluate_contrastive_split(
    model: torch.nn.Module,
    *,
    source_ctxs: dict[int, CaseFishContext],
    index_map: dict[int, np.ndarray],
    subset_map: dict[int, np.ndarray],
    batch_size: int,
    device: torch.device,
    metric: str,
    temperature: float,
    pos_radius: int,
    forward_chunk_size_t: int,
    eval_seed: int,
) -> dict:
    model.eval()
    per_fish = {}
    losses = []
    with torch.no_grad():
        for offset, fish_id in enumerate(sorted(source_ctxs)):
            ctx = source_ctxs[int(fish_id)]
            idx = np.asarray(index_map[int(fish_id)], dtype=np.int64)
            subset_idx = np.asarray(subset_map[int(fish_id)], dtype=np.int64)
            rng = np.random.default_rng(int(eval_seed) + 1000 * int(fish_id) + int(offset))
            batch_losses = []
            batch_pairs = 0
            for anchor_t in iter_anchor_batches(idx, int(batch_size), rng):
                loss_t, info = compute_batch_contrastive_loss(
                    model,
                    ctx=ctx,
                    anchor_t=anchor_t,
                    pool_t=idx,
                    neuron_idx=subset_idx,
                    device=device,
                    metric=str(metric),
                    temperature=float(temperature),
                    pos_radius=int(pos_radius),
                    forward_chunk_size_t=int(forward_chunk_size_t),
                    rng=rng,
                )
                loss_val = float(loss_t.detach().cpu().item())
                if np.isfinite(loss_val):
                    batch_losses.append(loss_val)
                    batch_pairs += int(info["num_pairs"])
            fish_loss = safe_mean(batch_losses)
            per_fish[str(int(fish_id))] = {
                "val_loss": fish_loss,
                "num_batches": int(len(batch_losses)),
                "num_pairs": int(batch_pairs),
            }
            if np.isfinite(fish_loss):
                losses.append(float(fish_loss))
    return {
        "metric": safe_mean(losses),
        "per_fish": per_fish,
    }


def plot_manifold_training_history(
    train_losses: list[float],
    val_losses: list[float],
    out_png: Path,
    *,
    title: str,
    best_epoch: int,
) -> None:
    if not train_losses:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    epochs = np.arange(1, len(train_losses) + 1, dtype=np.int64)
    ax.plot(epochs, np.asarray(train_losses, dtype=np.float32), color="#C05621", lw=1.6, label="train InfoNCE")
    ax.plot(epochs, np.asarray(val_losses, dtype=np.float32), color="#1D4ED8", lw=1.6, label="val InfoNCE")
    if int(best_epoch) > 0:
        ax.axvline(int(best_epoch), color="#111827", lw=1.0, ls="--", alpha=0.65)
    ax.set_xlabel("epoch")
    ax.set_ylabel("InfoNCE loss")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def persist_manifold_progress(
    split_dir: Path,
    *,
    train_index_map: dict[int, np.ndarray],
    val_index_map: dict[int, np.ndarray],
    val_subset_map: dict[int, np.ndarray],
    train_losses: list[float],
    val_losses: list[float],
    best_epoch: int,
    best_metric: float,
    best_metric_name: str = "val_loss",
    history_rows: list[dict],
) -> dict:
    payload = {
        "split_mode": "tail",
        "best_epoch": int(best_epoch),
        "best_metric_name": str(best_metric_name),
        "best_metric": None if not np.isfinite(best_metric) else float(best_metric),
        "stopped_epoch": int(len(train_losses)),
        "train_index_sizes": {str(k): int(len(v)) for k, v in train_index_map.items()},
        "val_index_sizes": {str(k): int(len(v)) for k, v in val_index_map.items()},
        "val_subset_sizes": {str(k): int(len(v)) for k, v in val_subset_map.items()},
        "history": history_rows,
    }
    save_json(split_dir / "split_summary.json", payload)
    plot_manifold_training_history(
        train_losses,
        val_losses,
        split_dir / "training_curve.png",
        title="IdentityGuidedGrouping manifold transfer | InfoNCE",
        best_epoch=int(best_epoch),
    )
    return payload


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": None if val_loss is None or not np.isfinite(val_loss) else float(val_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "meta": to_jsonable(meta),
    }
    torch.save(payload, path)


def fit_case_model(
    *,
    split_dir: Path,
    source_ctxs: dict[int, CaseFishContext],
    train_index_map: dict[int, np.ndarray],
    val_index_map: dict[int, np.ndarray],
    case_contexts: dict[int, CaseFishContext] | None,
    fish_order: list[int] | None,
    device: torch.device,
    case_seed: int,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict, dict[str, Path]]:
    data.set_global_seed(int(case_seed), deterministic=bool(args.deterministic))
    reference_ctx = next(iter(source_ctxs.values()))
    model = build_manifold_model(reference_ctx, latent_dim=int(args.latent_dim), args=args).to(device)
    train_pos_radius_sampler = build_train_pos_radius_sampler(args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    val_subset_map = {
        int(fid): sample_neuron_subset_indices(
            source_ctxs[int(fid)],
            requested=int(args.eval_neurons),
            seed=int(case_seed) + 5000 + int(fid),
        )
        for fid in sorted(source_ctxs)
    }
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_selection_score = float("inf")
    best_epoch = 0
    patience_count = 0
    history_rows: list[dict] = []
    shared_train_window_starts = contiguous_window_starts(
        [np.asarray(train_index_map[int(fid)], dtype=np.int64) for fid in sorted(source_ctxs)],
        window_t=int(getattr(args, "relational_window_t", 24)),
    )
    epoch_iter = iter_with_tqdm(
        range(1, int(args.max_epochs) + 1),
        desc=f"manifold {split_dir.parent.name}",
        dynamic_ncols=True,
        leave=False,
    )
    for epoch in epoch_iter:
        model.train()
        epoch_rng = np.random.default_rng(int(case_seed) + 10000 + int(epoch))
        per_fish_batches = {
            int(fid): iter_anchor_batches(np.asarray(train_index_map[int(fid)], dtype=np.int64), int(args.batch_size_t), epoch_rng)
            for fid in sorted(source_ctxs)
        }
        max_steps = max((len(v) for v in per_fish_batches.values()), default=0)
        batch_loss_vals = []
        batch_pair_count = 0
        batch_relational_vals = []
        batch_pos_radius_vals = []
        for step in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            loss_terms = []
            subset_idx_map: dict[int, np.ndarray] = {}
            train_pos_radius = sample_train_pos_radius(epoch_rng, args, train_pos_radius_sampler)
            batch_pos_radius_vals.append(int(train_pos_radius))
            for fish_rank, fish_id in enumerate(sorted(source_ctxs)):
                batches = per_fish_batches[int(fish_id)]
                if step >= len(batches):
                    continue
                ctx = source_ctxs[int(fish_id)]
                anchor_t = np.asarray(batches[step], dtype=np.int64)
                subset_idx = sample_neuron_subset_indices(
                    ctx,
                    requested=int(args.sampled_neurons),
                    seed=int(case_seed) + 100000 * int(epoch) + 1000 * int(step) + int(fish_id) + int(fish_rank),
                )
                subset_idx_map[int(fish_id)] = np.asarray(subset_idx, dtype=np.int64)
                loss_t, info = compute_batch_contrastive_loss(
                    model,
                    ctx=ctx,
                    anchor_t=anchor_t,
                    pool_t=train_index_map[int(fish_id)],
                    neuron_idx=subset_idx,
                    device=device,
                    metric=str(args.contrastive_metric),
                    temperature=float(args.temperature),
                    pos_radius=int(train_pos_radius),
                    forward_chunk_size_t=int(args.forward_chunk_size_t),
                    rng=epoch_rng,
                )
                if int(info["num_pairs"]) > 0:
                    loss_terms.append(loss_t)
                    batch_pair_count += int(info["num_pairs"])
            if not loss_terms:
                continue
            loss = torch.stack(loss_terms).mean()
            relational_loss_val = None
            if (
                float(getattr(args, "relational_loss_weight", 0.0)) > 0.0
                and len(subset_idx_map) >= 2
                and shared_train_window_starts.size > 0
                and int(getattr(args, "relational_every_steps", 1)) > 0
                and step % int(getattr(args, "relational_every_steps", 1)) == 0
            ):
                rel_start = int(epoch_rng.choice(shared_train_window_starts))
                rel_t = np.arange(
                    rel_start,
                    rel_start + int(getattr(args, "relational_window_t", 24)),
                    dtype=np.int64,
                )
                rel_loss = compute_cross_fish_relational_loss(
                    model,
                    source_ctxs=source_ctxs,
                    subset_idx_map=subset_idx_map,
                    time_indices=rel_t,
                    batch_size_t=int(args.forward_chunk_size_t),
                    device=device,
                )
                relational_loss_val = float(rel_loss.detach().cpu().item())
                loss = loss + float(getattr(args, "relational_loss_weight", 0.0)) * rel_loss
            loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
            optimizer.step()
            batch_loss_vals.append(float(loss.detach().cpu().item()))
            if relational_loss_val is not None and np.isfinite(relational_loss_val):
                batch_relational_vals.append(float(relational_loss_val))
        train_loss = safe_mean(batch_loss_vals)
        relational_loss_epoch = safe_mean(batch_relational_vals)
        val_stats = evaluate_contrastive_split(
            model,
            source_ctxs=source_ctxs,
            index_map=val_index_map,
            subset_map=val_subset_map,
            batch_size=int(args.eval_batch_size_t),
            device=device,
            metric=str(args.contrastive_metric),
            temperature=float(args.temperature),
            pos_radius=int(args.pos_radius),
            forward_chunk_size_t=int(args.forward_chunk_size_t),
            eval_seed=int(case_seed) + 90000 + int(epoch),
        )
        val_loss = float(val_stats["metric"])
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        val_improved = bool(np.isfinite(val_loss)) and (val_loss < (best_val_loss - float(args.early_stop_min_delta)))
        if val_improved:
            best_val_loss = float(val_loss)
            patience_count = 0
        else:
            patience_count += 1
        selection_stats = None
        selection_score = float("nan")
        if int(getattr(args, "checkpoint_select_every", 10)) > 0 and (
            epoch % int(getattr(args, "checkpoint_select_every", 10)) == 0
            or epoch == int(args.max_epochs)
        ):
            selection_stats = evaluate_checkpoint_selection(
                model,
                source_ctxs=source_ctxs,
                val_index_map=val_index_map,
                device=device,
                args=args,
                eval_seed=int(case_seed) + 190000 + int(epoch),
            )
            selection_score = float(selection_stats.get("composite_score", float("nan")))
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": None if not np.isfinite(train_loss) else float(train_loss),
                "val_loss": None if not np.isfinite(val_loss) else float(val_loss),
                "relational_loss": None if not np.isfinite(relational_loss_epoch) else float(relational_loss_epoch),
                "pairs_seen": int(batch_pair_count),
                "train_pos_radius_mean": None if not batch_pos_radius_vals else float(safe_mean(batch_pos_radius_vals)),
                "train_pos_radius_values": [int(v) for v in sorted(set(batch_pos_radius_vals))],
                "val_detail": val_stats["per_fish"],
                "checkpoint_selection": selection_stats,
            }
        )
        checkpoint_improved = bool(np.isfinite(selection_score)) and (
            selection_score < (best_selection_score - float(getattr(args, "selection_min_delta", 1.0e-4)))
        )
        if checkpoint_improved:
            best_selection_score = float(selection_score)
            best_epoch = int(epoch)
            save_checkpoint(
                split_dir / "checkpoints" / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                meta={
                    "task": "manifold",
                    "best": True,
                    "best_metric_name": "checkpoint_selection_composite",
                    "selection_score": float(selection_score),
                    "selection": selection_stats,
                    "source_fish_ids": [int(fid) for fid in sorted(source_ctxs)],
                },
            )
        if int(args.checkpoint_every) > 0 and epoch % int(args.checkpoint_every) == 0:
            save_checkpoint(
                split_dir / "checkpoints" / f"epoch_{int(epoch):04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                meta={"task": "manifold", "source_fish_ids": [int(fid) for fid in sorted(source_ctxs)]},
            )
        if (
            int(getattr(args, "snapshot_every", 0)) > 0
            and epoch % int(getattr(args, "snapshot_every", 0)) == 0
            and case_contexts is not None
            and fish_order is not None
        ):
            save_epoch_latent_snapshot(
                model,
                case_contexts=case_contexts,
                fish_order=fish_order,
                args=args,
                case_seed=int(case_seed),
                device=device,
                snapshot_dir=split_dir.parent / f"epoch_{int(epoch):04d}",
                snapshot_epoch=int(epoch),
            )
        persist_manifold_progress(
            split_dir,
            train_index_map=train_index_map,
            val_index_map=val_index_map,
            val_subset_map=val_subset_map,
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=int(best_epoch),
            best_metric=float(best_selection_score),
            best_metric_name="checkpoint_selection_composite",
            history_rows=history_rows,
        )
        if tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            postfix = {
                "train": f"{float(train_loss):.4f}" if np.isfinite(train_loss) else "nan",
                "val": f"{float(val_loss):.4f}" if np.isfinite(val_loss) else "nan",
                "best": int(best_epoch),
            }
            if np.isfinite(relational_loss_epoch):
                postfix["rel"] = f"{float(relational_loss_epoch):.4f}"
            if np.isfinite(selection_score):
                postfix["sel"] = f"{float(selection_score):.4f}"
            epoch_iter.set_postfix(postfix)
        if (
            not bool(getattr(args, "disable_early_stop", False))
            and epoch >= int(args.min_epochs)
            and patience_count >= int(args.patience)
        ):
            break
    save_checkpoint(
        split_dir / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(len(train_losses)),
        train_loss=float(train_losses[-1]) if train_losses else float("nan"),
        val_loss=float(val_losses[-1]) if val_losses else None,
        meta={"task": "manifold", "best": False, "source_fish_ids": [int(fid) for fid in sorted(source_ctxs)]},
    )
    if best_epoch <= 0:
        best_epoch = len(train_losses)
    best_ckpt_path = split_dir / "checkpoints" / "best.pt"
    last_ckpt_path = split_dir / "checkpoints" / "last.pt"
    if not best_ckpt_path.exists():
        shutil.copy2(last_ckpt_path, best_ckpt_path)
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    split_summary = persist_manifold_progress(
        split_dir,
        train_index_map=train_index_map,
        val_index_map=val_index_map,
        val_subset_map=val_subset_map,
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=int(best_epoch),
        best_metric=float(best_selection_score),
        best_metric_name="checkpoint_selection_composite",
        history_rows=history_rows,
    )
    return model, split_summary, {"best": best_ckpt_path, "last": last_ckpt_path}


def load_model_from_checkpoint(
    *,
    reference_ctx: CaseFishContext,
    ckpt_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    model = build_manifold_model(reference_ctx, latent_dim=int(args.latent_dim), args=args).to(device)
    ckpt = torch.load(Path(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def embed_time_series_for_subset(
    model: torch.nn.Module,
    *,
    ctx: CaseFishContext,
    neuron_idx: np.ndarray,
    batch_size_t: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    x_nt_ctx, static_t, static_cache = prepare_subset_context(model, ctx=ctx, neuron_idx=neuron_idx, device=device)
    all_t = np.arange(int(ctx.base.fish.neural_nt.shape[1]), dtype=np.int64)
    chunks = []
    with torch.no_grad():
        for start in range(0, all_t.size, max(1, int(batch_size_t))):
            t_chunk = all_t[start : start + int(batch_size_t)]
            x_btn = build_time_batch(model, x_nt_ctx=x_nt_ctx, time_indices=t_chunk, device=device)
            z = data.model_forward_maybe_cached(model, x_btn, static_t, static_cache)
            chunks.append(np.asarray(z.detach().cpu(), dtype=np.float32))
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def embed_with_ensemble(
    model: torch.nn.Module,
    *,
    ctx: CaseFishContext,
    subsets: list[np.ndarray],
    batch_size_t: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    latents = []
    for subset_idx in subsets:
        latents.append(
            embed_time_series_for_subset(
                model,
                ctx=ctx,
                neuron_idx=np.asarray(subset_idx, dtype=np.int64),
                batch_size_t=int(batch_size_t),
                device=device,
            )
        )
    stack = np.stack(latents, axis=0).astype(np.float32, copy=False)
    return np.mean(stack, axis=0).astype(np.float32, copy=False), stack


def embed_selected_times_for_subset(
    model: torch.nn.Module,
    *,
    ctx: CaseFishContext,
    neuron_idx: np.ndarray,
    time_indices: np.ndarray,
    batch_size_t: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    x_nt_ctx, static_t, static_cache = prepare_subset_context(model, ctx=ctx, neuron_idx=neuron_idx, device=device)
    all_t = np.asarray(time_indices, dtype=np.int64).reshape(-1)
    chunks = []
    with torch.no_grad():
        for start in range(0, all_t.size, max(1, int(batch_size_t))):
            t_chunk = all_t[start : start + int(batch_size_t)]
            t_b = torch.as_tensor(t_chunk, dtype=torch.int64, device=device)
            x_btn = data.build_time_batch_input(model, x_nt_ctx, t_b)
            z = data.model_forward_maybe_cached(model, x_btn, static_t, static_cache)
            chunks.append(np.asarray(z.detach().cpu(), dtype=np.float32))
    if not chunks:
        return np.zeros((0, int(getattr(model, "d_model", 0))), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def embed_selected_times_for_subset_torch(
    model: torch.nn.Module,
    *,
    ctx: CaseFishContext,
    neuron_idx: np.ndarray,
    time_indices: np.ndarray,
    batch_size_t: int,
    device: torch.device,
) -> torch.Tensor:
    x_nt_ctx, static_t, static_cache = prepare_subset_context(model, ctx=ctx, neuron_idx=neuron_idx, device=device)
    all_t = np.asarray(time_indices, dtype=np.int64).reshape(-1)
    chunks: list[torch.Tensor] = []
    for start in range(0, all_t.size, max(1, int(batch_size_t))):
        t_chunk = all_t[start : start + int(batch_size_t)]
        t_b = torch.as_tensor(t_chunk, dtype=torch.int64, device=device)
        x_btn = data.build_time_batch_input(model, x_nt_ctx, t_b)
        chunks.append(data.model_forward_maybe_cached(model, x_btn, static_t, static_cache))
    if not chunks:
        return model.null_static[:0]
    return torch.cat(chunks, dim=0)


def embed_selected_times_with_ensemble(
    model: torch.nn.Module,
    *,
    ctx: CaseFishContext,
    subsets: list[np.ndarray],
    time_indices: np.ndarray,
    batch_size_t: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    latents = []
    for subset_idx in subsets:
        latents.append(
            embed_selected_times_for_subset(
                model,
                ctx=ctx,
                neuron_idx=np.asarray(subset_idx, dtype=np.int64),
                time_indices=np.asarray(time_indices, dtype=np.int64),
                batch_size_t=int(batch_size_t),
                device=device,
            )
        )
    stack = np.stack(latents, axis=0).astype(np.float32, copy=False)
    return np.mean(stack, axis=0).astype(np.float32, copy=False), stack


def intersect_time_indices(index_list: list[np.ndarray]) -> np.ndarray:
    if not index_list:
        return np.zeros((0,), dtype=np.int64)
    common = np.asarray(index_list[0], dtype=np.int64).reshape(-1)
    for arr in index_list[1:]:
        common = np.intersect1d(common, np.asarray(arr, dtype=np.int64).reshape(-1), assume_unique=False)
        if common.size <= 0:
            break
    return np.asarray(common, dtype=np.int64)


def contiguous_window_starts(index_list: list[np.ndarray], window_t: int) -> np.ndarray:
    common = intersect_time_indices(index_list)
    w = max(2, int(window_t))
    if common.size < w:
        return np.zeros((0,), dtype=np.int64)
    common_set = set(int(v) for v in common.tolist())
    starts = [int(t) for t in common.tolist() if all((int(t) + off) in common_set for off in range(w))]
    return np.asarray(starts, dtype=np.int64)


def normalized_distance_matrix_np(z_td: np.ndarray) -> np.ndarray:
    z = np.asarray(z_td, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] < 2:
        return np.zeros((int(z.shape[0]), int(z.shape[0])), dtype=np.float32)
    diff = z[:, None, :] - z[None, :, :]
    dist = np.linalg.norm(diff, axis=-1).astype(np.float32, copy=False)
    mask = ~np.eye(dist.shape[0], dtype=bool)
    vals = dist[mask]
    vals = vals[np.isfinite(vals)]
    scale = float(np.mean(vals)) if vals.size > 0 else 1.0
    if not np.isfinite(scale) or scale < 1.0e-6:
        scale = 1.0
    dist = dist / float(scale)
    dist[~np.isfinite(dist)] = 0.0
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32, copy=False)


def relational_distance_matrix_discrepancy(z_a_td: np.ndarray, z_b_td: np.ndarray) -> float:
    a = np.asarray(z_a_td, dtype=np.float32)
    b = np.asarray(z_b_td, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] < 2:
        return float("nan")
    da = normalized_distance_matrix_np(a)
    db = normalized_distance_matrix_np(b)
    mask = ~np.eye(da.shape[0], dtype=bool)
    diff = da[mask] - db[mask]
    if diff.size <= 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(diff, dtype=np.float32), dtype=np.float32)))


def compute_subset_robustness_penalty(latent_stack_ktd: np.ndarray) -> float:
    stack = np.asarray(latent_stack_ktd, dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] < 2:
        return 0.0
    vals = []
    for i in range(int(stack.shape[0])):
        for j in range(i + 1, int(stack.shape[0])):
            vals.append(relational_distance_matrix_discrepancy(stack[i], stack[j]))
    return safe_mean(vals)


def normalized_distance_matrix_torch(z_td: torch.Tensor) -> torch.Tensor:
    if z_td.ndim != 2 or z_td.shape[0] < 2:
        return z_td.new_zeros((int(z_td.shape[0]), int(z_td.shape[0])))
    dist = torch.cdist(z_td, z_td, p=2)
    mask = ~torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)
    vals = dist.masked_select(mask)
    scale = vals.mean() if vals.numel() > 0 else dist.new_tensor(1.0)
    scale = torch.clamp(scale, min=1.0e-6)
    dist = dist / scale
    dist = torch.where(mask, dist, torch.zeros_like(dist))
    return dist


def relational_distance_matrix_loss_torch(latents: list[torch.Tensor]) -> torch.Tensor:
    if not latents:
        return torch.zeros((), dtype=torch.float32)
    if len(latents) < 2:
        return latents[0].sum() * 0.0
    terms = []
    for i in range(len(latents)):
        for j in range(i + 1, len(latents)):
            d_i = normalized_distance_matrix_torch(latents[i])
            d_j = normalized_distance_matrix_torch(latents[j])
            mask = ~torch.eye(d_i.shape[0], dtype=torch.bool, device=d_i.device)
            diff = d_i.masked_select(mask) - d_j.masked_select(mask)
            if diff.numel() > 0:
                terms.append(torch.mean(diff.square()))
    if not terms:
        return latents[0].sum() * 0.0
    return torch.stack(terms).mean()


def compute_cross_fish_relational_loss(
    model: torch.nn.Module,
    *,
    source_ctxs: dict[int, CaseFishContext],
    subset_idx_map: dict[int, np.ndarray],
    time_indices: np.ndarray,
    batch_size_t: int,
    device: torch.device,
) -> torch.Tensor:
    fish_ids = [int(fid) for fid in sorted(source_ctxs) if int(fid) in subset_idx_map]
    if len(fish_ids) < 2:
        return model.null_static.sum() * 0.0
    latents = []
    for fish_id in fish_ids:
        latents.append(
            embed_selected_times_for_subset_torch(
                model,
                ctx=source_ctxs[int(fish_id)],
                neuron_idx=np.asarray(subset_idx_map[int(fish_id)], dtype=np.int64),
                time_indices=np.asarray(time_indices, dtype=np.int64),
                batch_size_t=int(batch_size_t),
                device=device,
            )
        )
    return relational_distance_matrix_loss_torch(latents)


def evaluate_checkpoint_selection(
    model: torch.nn.Module,
    *,
    source_ctxs: dict[int, CaseFishContext],
    val_index_map: dict[int, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
    eval_seed: int,
) -> dict:
    fish_ids = [int(fid) for fid in sorted(source_ctxs)]
    if len(fish_ids) < 2:
        return {"status": "too_few_source_fish"}
    window_t = max(8, int(getattr(args, "selection_window_t", 96)))
    rng = np.random.default_rng(int(eval_seed))
    val_arrays = [np.asarray(val_index_map[int(fid)], dtype=np.int64) for fid in fish_ids]
    starts = contiguous_window_starts(val_arrays, window_t=window_t)
    common_val = intersect_time_indices(val_arrays)
    if starts.size > 0:
        start_t = int(rng.choice(starts))
        eval_t = np.arange(start_t, start_t + int(window_t), dtype=np.int64)
        eval_mode = "shared_val_window"
    else:
        if common_val.size < 8:
            return {
                "status": "insufficient_shared_val_times",
                "shared_val_count": int(common_val.size),
            }
        keep = min(int(window_t), int(common_val.size))
        pick = np.linspace(0, int(common_val.size) - 1, keep, dtype=np.int64)
        eval_t = np.asarray(common_val[pick], dtype=np.int64)
        eval_mode = "shared_val_subsample"

    subset_count = max(1, int(getattr(args, "selection_ensemble_subsets", 3)))
    continuity_vals = []
    subset_penalties = []
    per_fish = {}
    mean_latents: dict[int, np.ndarray] = {}
    for offset, fish_id in enumerate(fish_ids):
        ctx = source_ctxs[int(fish_id)]
        subsets = []
        for ens_idx in range(subset_count):
            subsets.append(
                sample_neuron_subset_indices(
                    ctx,
                    requested=int(args.eval_neurons),
                    seed=int(eval_seed) + 5000 * int(offset) + 113 * int(ens_idx) + int(fish_id),
                )
            )
        latent_mean, latent_stack = embed_selected_times_with_ensemble(
            model,
            ctx=ctx,
            subsets=subsets,
            time_indices=eval_t,
            batch_size_t=int(args.eval_batch_size_t),
            device=device,
        )
        mean_latents[int(fish_id)] = np.asarray(latent_mean, dtype=np.float32)
        metric_values = compute_manifold_metrics(latent_mean)
        cont_terms = []
        time_corr = float(metric_values.get("time_dist_spearman_local100", float("nan")))
        if np.isfinite(time_corr):
            cont_terms.append(float(np.clip(time_corr, 0.0, 1.0)))
        time_gap = float(metric_values.get("normalized_knn_time_gap", float("nan")))
        if np.isfinite(time_gap):
            cont_terms.append(float(np.clip(1.0 - time_gap, 0.0, 1.0)))
        continuity_score = safe_mean(cont_terms)
        subset_penalty = compute_subset_robustness_penalty(latent_stack)
        if np.isfinite(continuity_score):
            continuity_vals.append(float(continuity_score))
        if np.isfinite(subset_penalty):
            subset_penalties.append(float(subset_penalty))
        per_fish[str(int(fish_id))] = {
            "continuity_score": None if not np.isfinite(continuity_score) else float(continuity_score),
            "subset_robustness_penalty": None if not np.isfinite(subset_penalty) else float(subset_penalty),
            **{
                key: None if not np.isfinite(float(value)) else float(value)
                for key, value in metric_values.items()
            },
        }

    pairwise_rel = {}
    source_rel_vals = []
    for i, fish_i in enumerate(fish_ids):
        row = {}
        for fish_j in fish_ids:
            if int(fish_i) == int(fish_j):
                row[str(int(fish_j))] = 0.0
                continue
            val = relational_distance_matrix_discrepancy(mean_latents[int(fish_i)], mean_latents[int(fish_j)])
            row[str(int(fish_j))] = None if not np.isfinite(val) else float(val)
            if int(fish_j) > int(fish_i) and np.isfinite(val):
                source_rel_vals.append(float(val))
        pairwise_rel[str(int(fish_i))] = row

    continuity_penalty = float("nan")
    if continuity_vals:
        continuity_penalty = float(1.0 - np.clip(safe_mean(continuity_vals), 0.0, 1.0))
    subset_penalty_mean = safe_mean(subset_penalties)
    source_rel_penalty = safe_mean(source_rel_vals)
    weighted_terms = []
    for weight, value in [
        (float(getattr(args, "selection_weight_continuity", 1.0)), continuity_penalty),
        (float(getattr(args, "selection_weight_subset", 1.0)), subset_penalty_mean),
        (float(getattr(args, "selection_weight_relational", 1.0)), source_rel_penalty),
    ]:
        if weight > 0.0 and np.isfinite(value):
            weighted_terms.append((float(weight), float(value)))
    if weighted_terms:
        composite_score = float(sum(w * v for w, v in weighted_terms) / sum(w for w, _ in weighted_terms))
    else:
        composite_score = float("nan")
    return {
        "status": "ok",
        "eval_mode": str(eval_mode),
        "eval_time_count": int(eval_t.size),
        "continuity_penalty": None if not np.isfinite(continuity_penalty) else float(continuity_penalty),
        "subset_robustness_penalty": None if not np.isfinite(subset_penalty_mean) else float(subset_penalty_mean),
        "source_relational_penalty": None if not np.isfinite(source_rel_penalty) else float(source_rel_penalty),
        "composite_score": None if not np.isfinite(composite_score) else float(composite_score),
        "per_fish": per_fish,
        "pairwise_source_relational": pairwise_rel,
    }


def compute_normalized_knn_time_gap(
    z_td: np.ndarray,
    *,
    k: int = 10,
) -> float:
    z = np.asarray(z_td, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] < 3:
        return float("nan")
    n_neighbors = min(int(z.shape[0]), int(k) + 1)
    if n_neighbors <= 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(z)
    neigh_idx = np.asarray(nn.kneighbors(return_distance=False), dtype=np.int64)[:, 1:]
    ref_idx = np.arange(int(z.shape[0]), dtype=np.int64)[:, None]
    abs_dt = np.abs(neigh_idx - ref_idx).astype(np.float32, copy=False)
    k_use = min(int(k), int(abs_dt.shape[1]))
    if k_use <= 0:
        return float("nan")
    return float(np.mean(abs_dt[:, :k_use]) / max(1.0, float(z.shape[0] - 1)))


def compute_local_tangent_linearity(
    z_td: np.ndarray,
    *,
    window: int = 25,
    stride: int = 5,
) -> float:
    z = np.asarray(z_td, dtype=np.float64)
    w = max(5, int(window))
    s = max(1, int(stride))
    if z.ndim != 2 or z.shape[0] < w:
        return float("nan")
    ratios = []
    for start in range(0, int(z.shape[0]) - w + 1, s):
        seg = z[start : start + w]
        seg0 = seg - seg.mean(axis=0, keepdims=True)
        cov = np.cov(seg0, rowvar=False)
        eig = np.asarray(np.linalg.eigvalsh(cov), dtype=np.float64)
        eig = eig[np.isfinite(eig) & (eig > 1e-12)]
        if eig.size <= 0:
            continue
        ratios.append(float(np.max(eig) / np.sum(eig)))
    return safe_mean(ratios)


def compute_time_distance_spearman(
    z_td: np.ndarray,
    *,
    max_dt: int = 100,
    num_pairs: int = 20000,
    seed: int = 0,
) -> float:
    z = np.asarray(z_td, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 4:
        return float("nan")
    off_max = min(max(1, int(max_dt)), int(z.shape[0]) - 1)
    if off_max <= 0:
        return float("nan")
    rng = np.random.default_rng(int(seed) + int(z.shape[0]) * 37 + int(z.shape[1]) * 101)
    i = np.asarray(rng.integers(0, int(z.shape[0]) - 1, size=max(256, int(num_pairs))), dtype=np.int64)
    dt = np.asarray(rng.integers(1, off_max + 1, size=i.shape[0]), dtype=np.int64)
    j = i + dt
    valid = j < int(z.shape[0])
    if not np.any(valid):
        return float("nan")
    i = i[valid]
    j = j[valid]
    dt = dt[valid]
    dz = z[j] - z[i]
    dist = np.linalg.norm(dz, axis=1)
    mask = np.isfinite(dist)
    if int(np.sum(mask)) < 8:
        return float("nan")
    corr = spearmanr(np.asarray(dt[mask], dtype=np.float64), np.asarray(dist[mask], dtype=np.float64)).correlation
    return float(corr) if corr is not None and np.isfinite(corr) else float("nan")


def compute_manifold_metrics(z_td: np.ndarray) -> dict[str, float]:
    return {
        "time_dist_spearman_local100": float(compute_time_distance_spearman(z_td, max_dt=100, num_pairs=20000)),
        "normalized_knn_time_gap": float(compute_normalized_knn_time_gap(z_td, k=10)),
        "local_tangent_linearity_w25": float(compute_local_tangent_linearity(z_td, window=25, stride=5)),
    }


def project_latent_2d(z_td: np.ndarray) -> np.ndarray:
    z = np.asarray(z_td, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] < 2:
        return np.zeros((int(z.shape[0]), 2), dtype=np.float32)
    n_comp = min(2, int(z.shape[1]), int(z.shape[0]))
    pca = PCA(n_components=int(n_comp), svd_solver="full")
    xy = np.asarray(pca.fit_transform(z), dtype=np.float32)
    if xy.shape[1] == 1:
        xy = np.concatenate([xy, np.zeros((xy.shape[0], 1), dtype=np.float32)], axis=1)
    return xy


def project_latent_3d(z_td: np.ndarray) -> np.ndarray:
    z = np.asarray(z_td, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] < 2:
        return np.zeros((int(z.shape[0]), 3), dtype=np.float32)
    n_comp = min(3, int(z.shape[1]), int(z.shape[0]))
    pca = PCA(n_components=int(n_comp), svd_solver="full")
    xyz = np.asarray(pca.fit_transform(z), dtype=np.float32)
    if xyz.shape[1] < 3:
        xyz = np.pad(xyz, ((0, 0), (0, 3 - xyz.shape[1])), mode="constant")
    xyz -= np.mean(xyz, axis=0, keepdims=True)
    scale = float(np.max(np.ptp(xyz, axis=0)))
    if scale > 1e-8:
        xyz /= scale
    return xyz


def _style_panel_3d_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def _set_equal_limits_3d(ax, xyz: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[0] <= 0:
        return
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.56 * float(max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2]), 1e-3))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_method_fish_panel(
    embeddings_by_method: dict[str, dict[int, ManifoldEmbedding]],
    fish_order: list[int],
    contexts: dict[int, BaseFishContext],
    out_png: Path,
    *,
    title: str,
    color_mode: str,
    heldout_fish_id: int,
) -> None:
    preferred = [
        "identity_guided_grouping",
        "identity_guided_grouping_best",
        "identity_guided_grouping_last",
    ]
    methods = [m for m in preferred if m in embeddings_by_method]
    methods.extend([m for m in embeddings_by_method.keys() if m not in methods])
    n_rows = len(methods)
    n_cols = len(fish_order)
    if n_rows <= 0 or n_cols <= 0:
        return
    fig = plt.figure(figsize=(4.6 * n_cols, 4.0 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, left=0.04, right=0.98, bottom=0.05, top=0.93, wspace=0.02, hspace=0.08)
    axes: list[list[object]] = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    stim_palette = np.asarray(
        [
            [0.20, 0.20, 0.20],
            [0.85, 0.37, 0.01],
            [0.23, 0.51, 0.96],
            [0.16, 0.63, 0.47],
            [0.70, 0.30, 0.82],
            [0.90, 0.56, 0.12],
            [0.82, 0.20, 0.28],
            [0.12, 0.64, 0.74],
            [0.45, 0.45, 0.45],
        ],
        dtype=np.float32,
    )
    time_colorbar = None
    for r, method in enumerate(methods):
        for c, fish_id in enumerate(fish_order):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            axes[r][c] = ax
            emb = embeddings_by_method[method][int(fish_id)]
            xyz = project_latent_3d(emb.latent_td)
            n_t = int(xyz.shape[0])
            scatter_idx = np.linspace(0, n_t - 1, min(2200, n_t), dtype=np.int64) if n_t > 0 else np.zeros((0,), dtype=np.int64)
            line_idx = np.linspace(0, n_t - 1, min(420, n_t), dtype=np.int64) if n_t > 0 else np.zeros((0,), dtype=np.int64)
            if str(color_mode) == "stimulus":
                ctx = contexts[int(fish_id)]
                stim_class = ctx.stimulus_class_t
                active = ctx.stimulus_active_mask_t
                if stim_class is None or active is None:
                    colors = np.tile(np.asarray([[0.4, 0.4, 0.4]], dtype=np.float32), (n_t, 1))
                else:
                    colors = np.tile(np.asarray([[0.82, 0.82, 0.82]], dtype=np.float32), (n_t, 1))
                    active_idx = np.flatnonzero(np.asarray(active, dtype=bool))
                    if active_idx.size > 0:
                        cls = np.asarray(stim_class[active_idx], dtype=np.int64)
                        cls = np.clip(cls + 1, 0, stim_palette.shape[0] - 1)
                        colors[active_idx] = stim_palette[cls]
                ax.scatter(
                    xyz[scatter_idx, 0],
                    xyz[scatter_idx, 1],
                    xyz[scatter_idx, 2],
                    s=3.2,
                    c=colors[scatter_idx],
                    alpha=0.80,
                    linewidths=0,
                    depthshade=False,
                )
            else:
                tt = np.arange(n_t, dtype=np.float32)
                sc = ax.scatter(
                    xyz[scatter_idx, 0],
                    xyz[scatter_idx, 1],
                    xyz[scatter_idx, 2],
                    s=3.2,
                    c=tt[scatter_idx],
                    cmap="viridis",
                    alpha=0.80,
                    linewidths=0,
                    depthshade=False,
                )
                if time_colorbar is None:
                    time_colorbar = sc
            if line_idx.size > 0:
                path = xyz[line_idx]
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#1E1E1E", lw=0.55, alpha=0.35)
            _set_equal_limits_3d(ax, xyz)
            _style_panel_3d_axis(ax)
            ax.view_init(elev=22.0, azim=36.0)
            metrics = emb.metrics
            heldout_tag = " | heldout" if int(fish_id) == int(heldout_fish_id) and method.startswith("identity_guided_grouping") else ""
            ax.set_title(
                f"{method.replace('_', ' ')} | fish {int(fish_id)}{heldout_tag}\n"
                f"time corr={float(metrics.get('time_dist_spearman_local100', float('nan'))):.3f} | gap={float(metrics.get('normalized_knn_time_gap', float('nan'))):.3f}",
                fontsize=9,
            )
    if time_colorbar is not None and str(color_mode) != "stimulus":
        cax = fig.add_axes([0.43, 0.025, 0.14, 0.015])
        cb = fig.colorbar(time_colorbar, cax=cax, orientation="horizontal")
        cb.set_label("time", fontsize=8.5, labelpad=1)
        cb.ax.tick_params(labelsize=8, length=0, pad=1)
        cb.outline.set_visible(False)
    fig.suptitle(title, fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_run_summary_latent_figure(
    case_records: list[dict],
    base_contexts: dict[int, BaseFishContext],
    out_png: Path,
) -> None:
    methods = ["identity_guided_grouping"]
    if not case_records:
        return
    n_rows = len(methods)
    n_cols = len(case_records)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.9 * n_rows), squeeze=False)
    stim_palette = np.asarray(
        [
            [0.82, 0.82, 0.82],
            [0.85, 0.37, 0.01],
            [0.23, 0.51, 0.96],
            [0.16, 0.63, 0.47],
            [0.70, 0.30, 0.82],
            [0.90, 0.56, 0.12],
            [0.82, 0.20, 0.28],
            [0.12, 0.64, 0.74],
            [0.45, 0.45, 0.45],
        ],
        dtype=np.float32,
    )
    for col, rec in enumerate(case_records):
        heldout_fish_id = int(rec["heldout_fish_id"])
        identity_guided_grouping_embeddings = rec["identity_guided_grouping_embeddings"]
        per_method = {"identity_guided_grouping": identity_guided_grouping_embeddings[int(heldout_fish_id)]}
        ctx = base_contexts[int(heldout_fish_id)]
        stim_class = ctx.stimulus_class_t
        active = ctx.stimulus_active_mask_t
        for row, method in enumerate(methods):
            ax = axes[row, col]
            emb = per_method[method]
            xy = project_latent_2d(emb.latent_td)
            colors = np.tile(np.asarray([[0.82, 0.82, 0.82]], dtype=np.float32), (xy.shape[0], 1))
            if stim_class is not None and active is not None:
                active_idx = np.flatnonzero(np.asarray(active, dtype=bool))
                if active_idx.size > 0:
                    cls = np.asarray(stim_class[active_idx], dtype=np.int64)
                    cls = np.clip(cls + 1, 0, stim_palette.shape[0] - 1)
                    colors[active_idx] = stim_palette[cls]
            ax.scatter(xy[:, 0], xy[:, 1], s=3, c=colors, alpha=0.78, linewidths=0)
            metrics = emb.metrics
            ax.set_title(
                f"{method.replace('_', ' ')} | heldout fish {heldout_fish_id}\n"
                f"time corr={float(metrics.get('time_dist_spearman_local100', float('nan'))):.3f} | gap={float(metrics.get('normalized_knn_time_gap', float('nan'))):.3f}",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(alpha=0.12)
    fig.suptitle("Held-out fish latent comparison | IdentityGuidedGrouping transfer", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_run_summary_metrics_figure(
    case_rows: list[dict],
    out_png: Path,
) -> None:
    if not case_rows:
        return
    methods = ["identity_guided_grouping"]
    metric_specs = [
        ("time_dist_spearman_local100", "Temporal Distance Correlation"),
        ("normalized_knn_time_gap", "Normalized Latent-Neighbor Time Gap"),
        ("local_tangent_linearity_w25", "Local Tangent Linearity"),
    ]
    n_cols = min(5, len(metric_specs))
    n_rows = int(math.ceil(len(metric_specs) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.9 * n_rows), squeeze=False)
    colors = {
        "identity_guided_grouping": "#D55E00",
    }
    flat_axes = list(axes.ravel())
    for ax, (metric_key, metric_label) in zip(flat_axes, metric_specs):
        x = np.arange(len(case_rows), dtype=np.float32)
        width = 0.22 if len(methods) >= 3 else 0.28
        for midx, method in enumerate(methods):
            vals = []
            for row in case_rows:
                heldout = str(int(row["heldout_fish_id"]))
                vals.append(float(row["identity_guided_grouping"]["fish_metrics"][heldout].get(metric_key, float("nan"))))
            offset = (midx - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, color=colors.get(method, "#444444"), label=method.replace("_", " "))
            mean_val = safe_mean(vals)
            if np.isfinite(mean_val):
                ax.axhline(mean_val, color=colors.get(method, "#444444"), lw=1.0, ls="--", alpha=0.35)
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels([f"heldout {int(row['heldout_fish_id'])}" for row in case_rows], rotation=0)
        ax.grid(alpha=0.18, axis="y")
    for ax in flat_axes[len(metric_specs) :]:
        ax.axis("off")
    flat_axes[0].legend(loc="best", frameon=False)
    fig.suptitle("Held-out manifold metrics by method", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def embedding_metric_payload(embedding: ManifoldEmbedding) -> dict[str, float]:
    metrics = embedding.metrics
    return {
        "time_dist_spearman_local100": float(metrics.get("time_dist_spearman_local100", float("nan"))),
        "normalized_knn_time_gap": float(metrics.get("normalized_knn_time_gap", float("nan"))),
        "local_tangent_linearity_w25": float(metrics.get("local_tangent_linearity_w25", float("nan"))),
    }


def compute_case_identity_guided_grouping_embeddings(
    model: torch.nn.Module,
    *,
    case_contexts: dict[int, CaseFishContext],
    fish_order: list[int],
    args: argparse.Namespace,
    case_seed: int,
    device: torch.device,
    ensemble_subsets: int | None = None,
) -> dict[int, ManifoldEmbedding]:
    out = {}
    ensemble_count = max(
        1,
        int(args.eval_ensemble_subsets if ensemble_subsets is None or int(ensemble_subsets) <= 0 else ensemble_subsets),
    )
    for offset, fish_id in enumerate(fish_order):
        ctx = case_contexts[int(fish_id)]
        subsets = []
        for ens_idx in range(int(ensemble_count)):
            subsets.append(
                sample_neuron_subset_indices(
                    ctx,
                    requested=int(args.eval_neurons),
                    seed=int(case_seed) + 20000 * int(offset) + 1009 * int(ens_idx) + int(fish_id),
                )
            )
        latent_mean, latent_stack = embed_with_ensemble(
            model,
            ctx=ctx,
            subsets=subsets,
            batch_size_t=int(args.eval_batch_size_t),
            device=device,
        )
        metrics = compute_manifold_metrics(latent_mean)
        latent_subset1 = np.asarray(latent_stack[0], dtype=np.float32) if latent_stack.shape[0] > 0 else None
        metrics_subset1 = (
            compute_manifold_metrics(latent_subset1)
            if latent_subset1 is not None
            else None
        )
        out[int(fish_id)] = ManifoldEmbedding(
            fish_id=int(fish_id),
            method="identity_guided_grouping",
            latent_td=latent_mean,
            subset_indices=np.asarray(subsets[0], dtype=np.int64) if subsets else None,
            metrics=metrics,
            metadata={
                "ensemble_subsets": int(len(subsets)),
                "latent_stack_shape": list(latent_stack.shape),
                "saved_latent_views": ["ensemble_mean", "subset1"],
            },
            latent_subset1_td=latent_subset1,
            metrics_subset1=metrics_subset1,
        )
    return out


def _metrics_payload(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    return dict(metrics)


def _embedding_view_title(embedding: ManifoldEmbedding, *, view: str, label: str | None = None) -> str:
    parts = [str(embedding.method), f"fish {int(embedding.fish_id)}"]
    if label:
        parts.append(str(label))
    parts.append(str(view))
    return " | ".join(parts)


def save_embedding_snapshot_artifacts(
    embedding: ManifoldEmbedding,
    *,
    out_dir: Path,
    label: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "latent_td.npy", np.asarray(embedding.latent_td, dtype=np.float32))
    if embedding.subset_indices is not None:
        np.save(out_dir / "subset_idx.npy", np.asarray(embedding.subset_indices, dtype=np.int64))
    save_json(
        out_dir / "metrics.json",
        {
            "fish_id": int(embedding.fish_id),
            "method": str(embedding.method),
            "metadata": embedding.metadata,
            "metrics": _metrics_payload(embedding.metrics),
        },
    )
    plotting.plot_latent_3d_time(
        np.asarray(embedding.latent_td, dtype=np.float32),
        out_dir / "latent_3d_time.png",
        title=_embedding_view_title(embedding, view="ensemble mean", label=label),
    )
    if embedding.latent_subset1_td is not None:
        np.save(out_dir / "latent_subset1_td.npy", np.asarray(embedding.latent_subset1_td, dtype=np.float32))
        save_json(
            out_dir / "metrics_subset1.json",
            {
                "fish_id": int(embedding.fish_id),
                "method": str(embedding.method),
                "metadata": {
                    **embedding.metadata,
                    "view": "subset1",
                },
                "metrics": _metrics_payload(embedding.metrics_subset1),
            },
        )
        plotting.plot_latent_3d_time(
            np.asarray(embedding.latent_subset1_td, dtype=np.float32),
            out_dir / "latent_3d_time_subset1.png",
            title=_embedding_view_title(embedding, view="subset1", label=label),
        )


def save_epoch_latent_snapshot(
    model: torch.nn.Module,
    *,
    case_contexts: dict[int, CaseFishContext],
    fish_order: list[int],
    args: argparse.Namespace,
    case_seed: int,
    device: torch.device,
    snapshot_dir: Path,
    snapshot_epoch: int,
) -> None:
    snapshot_embeddings = compute_case_identity_guided_grouping_embeddings(
        model,
        case_contexts=case_contexts,
        args=args,
        case_seed=int(case_seed),
        device=device,
        fish_order=fish_order,
        ensemble_subsets=int(getattr(args, "snapshot_ensemble_subsets", 0)),
    )
    label = f"epoch {int(snapshot_epoch):04d}"
    for fish_id in fish_order:
        save_embedding_snapshot_artifacts(
            snapshot_embeddings[int(fish_id)],
            out_dir=snapshot_dir / f"fish_{int(fish_id)}",
            label=label,
        )
    save_json(
        snapshot_dir / "snapshot_meta.json",
        {
            "epoch": int(snapshot_epoch),
            "fish_order": [int(fid) for fid in fish_order],
            "ensemble_subsets": int(
                max(
                    1,
                    int(
                        args.eval_ensemble_subsets
                        if int(getattr(args, "snapshot_ensemble_subsets", 0)) <= 0
                        else int(getattr(args, "snapshot_ensemble_subsets", 0))
                    ),
                )
            ),
        },
    )


def save_embedding_artifacts(
    embedding: ManifoldEmbedding,
    *,
    out_dir: Path,
    base_ctx: BaseFishContext,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "latent_td.npy", np.asarray(embedding.latent_td, dtype=np.float32))
    if embedding.subset_indices is not None:
        np.save(out_dir / "subset_idx.npy", np.asarray(embedding.subset_indices, dtype=np.int64))
    save_json(
        out_dir / "metrics.json",
        {
            "fish_id": int(embedding.fish_id),
            "method": str(embedding.method),
            "metadata": embedding.metadata,
            "metrics": _metrics_payload(embedding.metrics),
        },
    )
    plotting.plot_latent_3d_time(
        np.asarray(embedding.latent_td, dtype=np.float32),
        out_dir / "latent_3d_time.png",
        title=_embedding_view_title(embedding, view="ensemble mean"),
    )
    if base_ctx.behavior_target_t is not None:
        behavior_raw = np.asarray(base_ctx.behavior_target_t, dtype=np.float32)
        behavior_log = np.log(np.clip(behavior_raw, 1.0e-12, None)).astype(np.float32, copy=False)
        behavior_cmap = LinearSegmentedColormap.from_list(
            "black_to_magenta",
            ["#000000", "#200012", "#550033", "#8b0066", "#c0008f", "#ff3bd7"],
        )
        plotting.plot_latent_3d_time(
            np.asarray(embedding.latent_td, dtype=np.float32),
            out_dir / "latent_3d_behavior.png",
            title=_embedding_view_title(embedding, view="ensemble mean"),
            color_values=behavior_log,
            color_label=f"log({str(base_ctx.behavior_target_label)})",
            cmap=behavior_cmap,
        )
    if embedding.latent_subset1_td is not None:
        np.save(out_dir / "latent_subset1_td.npy", np.asarray(embedding.latent_subset1_td, dtype=np.float32))
        save_json(
            out_dir / "metrics_subset1.json",
            {
                "fish_id": int(embedding.fish_id),
                "method": str(embedding.method),
                "metadata": {
                    **embedding.metadata,
                    "view": "subset1",
                },
                "metrics": _metrics_payload(embedding.metrics_subset1),
            },
        )
        plotting.plot_latent_3d_time(
            np.asarray(embedding.latent_subset1_td, dtype=np.float32),
            out_dir / "latent_3d_time_subset1.png",
            title=_embedding_view_title(embedding, view="subset1"),
        )
        if base_ctx.behavior_target_t is not None:
            plotting.plot_latent_3d_time(
                np.asarray(embedding.latent_subset1_td, dtype=np.float32),
                out_dir / "latent_3d_behavior_subset1.png",
                title=_embedding_view_title(embedding, view="subset1"),
                color_values=behavior_log,
                color_label=f"log({str(base_ctx.behavior_target_label)})",
                cmap=behavior_cmap,
            )


def parse_args() -> argparse.Namespace:
    pack_root = Path(__file__).resolve().parent
    default_data_root = infer_default_data_root(Path(__file__))
    p = argparse.ArgumentParser(
        description=(
            "Transferable IdentityGuidedGrouping manifold runner. "
            "Trains a shared IdentityGuidedGrouping on 2 source fish with time-only contrastive InfoNCE, "
            "and transfers to 1 held-out fish."
        )
    )
    p.add_argument("--exp-name", type=str, default="identity_guided_grouping_manifold")
    p.add_argument("--out-root", type=Path, default=pack_root / "results" / "manifold_transfer")
    p.add_argument("--data-root", type=Path, default=default_data_root)
    p.add_argument("--fish-root", type=Path, default=None)
    p.add_argument("--postprocessed-dir", type=Path, default=None)
    p.add_argument("--fish-ids", type=str, default="59,63,71")
    p.add_argument(
        "--only-heldout-fish-ids",
        type=str,
        default="",
        help="Optional comma-separated heldout fish ids to train. Empty runs all leave-one-out cases.",
    )
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")

    p.add_argument("--gene-source", choices=["none", "binary", "count"], default="count")
    p.add_argument("--use-gene", action="store_true", default=True)
    p.add_argument("--no-gene", dest="use_gene", action="store_false")
    p.add_argument("--use-pos", action="store_true", default=True)
    p.add_argument("--no-pos", dest="use_pos", action="store_false")
    p.add_argument("--use-atlas", action="store_true", default=False)
    p.add_argument("--no-atlas", dest="use_atlas", action="store_false")
    p.add_argument("--use-region", action="store_true", default=False)
    p.add_argument("--no-region", dest="use_region", action="store_false")
    p.add_argument(
        "--region-feature-mode",
        choices=["none", "brain_reg_onehot", "brainregions_multihot"],
        default="brain_reg_onehot",
    )
    p.add_argument("--use-activity", action="store_true", default=False)
    p.add_argument("--no-activity", dest="use_activity", action="store_false")
    p.add_argument("--activity-feature-mode", choices=["none", "robust_stats"], default="robust_stats")
    p.add_argument("--activity-cache-dir", type=Path, default=None)
    p.add_argument("--activity-force-recompute", action="store_true")
    p.add_argument("--activity-require-cache", action="store_true", default=True)
    p.add_argument("--activity-allow-inline-compute", dest="activity_require_cache", action="store_false")
    p.add_argument(
        "--atlas-precomputed-dir",
        type=Path,
        default=None,
    )
    p.add_argument("--normalize-neural", action="store_true", default=True)
    p.add_argument("--static-normalize-mode", choices=["none", "self", "source"], default="source")
    p.add_argument("--debug-load", action="store_true")
    p.add_argument("--stimulus-active-threshold", type=float, default=0.5)

    p.add_argument("--latent-dim", type=int, default=3)
    p.add_argument("--normalize-latent", action="store_true", default=True)
    p.add_argument("--no-normalize-latent", dest="normalize_latent", action="store_false")
    p.add_argument("--contrastive-metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--pos-radius", type=int, default=1)
    p.add_argument("--train-pos-radius-choices", type=str, default="")
    p.add_argument("--train-pos-radius-max", type=int, default=0)
    p.add_argument("--train-pos-radius-half-life", type=float, default=25.0)

    p.add_argument("--sampled-neurons", type=int, default=3000)
    p.add_argument("--eval-neurons", type=int, default=3000)
    p.add_argument("--eval-ensemble-subsets", type=int, default=11)

    p.add_argument("--window-left", type=int, default=15)
    p.add_argument("--window-right", type=int, default=15)
    p.add_argument("--token-mlp-hidden", type=int, default=16)
    p.add_argument("--slot-internal-dim", type=int, default=32)
    p.add_argument("--slot-static-hidden-dim", type=int, default=16)
    p.add_argument("--slot-num-slots", type=int, default=32)
    p.add_argument("--slot-mixer-hidden", type=int, default=32)
    p.add_argument("--slot-assign-temp", type=float, default=0.35)
    p.add_argument("--slot-assign-topk", type=int, default=4)

    p.add_argument("--batch-size-t", type=int, default=128)
    p.add_argument("--eval-batch-size-t", type=int, default=128)
    p.add_argument("--forward-chunk-size-t", type=int, default=128)
    p.add_argument("--snapshot-every", type=int, default=0)
    p.add_argument("--snapshot-ensemble-subsets", type=int, default=0)
    p.add_argument("--checkpoint-select-every", type=int, default=25)
    p.add_argument("--selection-window-t", type=int, default=96)
    p.add_argument("--selection-ensemble-subsets", type=int, default=3)
    p.add_argument("--selection-weight-continuity", type=float, default=1.0)
    p.add_argument("--selection-weight-subset", type=float, default=1.0)
    p.add_argument("--selection-weight-relational", type=float, default=1.0)
    p.add_argument("--selection-min-delta", type=float, default=1e-4)
    p.add_argument("--relational-loss-weight", type=float, default=0.05)
    p.add_argument("--relational-window-t", type=int, default=24)
    p.add_argument("--relational-every-steps", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--min-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--disable-early-stop", action="store_true")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--checkpoint-every", type=int, default=10)

    args = p.parse_args()
    return finalize_bundle_args(args, atlas_precomputed_fallback=data.DEFAULT_PRECOMPUTED_ATLAS_HCR_DIR)


def main() -> None:
    args = parse_args()
    device = resolve_device(str(args.device))
    if (
        not bool(args.use_gene)
        and not bool(args.use_pos)
        and not bool(args.use_atlas)
        and not bool(getattr(args, "use_region", False))
        and not bool(getattr(args, "use_activity", False))
    ):
        raise ValueError("At least one identity input must be enabled: gene, pos, atlas, region, or activity.")
    if not args.fish_root.exists():
        raise FileNotFoundError(f"fish_root not found: {args.fish_root}")
    if not args.postprocessed_dir.exists():
        raise FileNotFoundError(f"postprocessed_dir not found: {args.postprocessed_dir}")
    if bool(args.use_atlas):
        if not Path(args.atlas_precomputed_dir).exists():
            raise FileNotFoundError(f"atlas_precomputed_dir not found: {args.atlas_precomputed_dir}")

    fish_ids = parse_int_list(args.fish_ids)
    if len(fish_ids) != 3:
        raise ValueError(f"Expected exactly 3 fish ids for leave-one-fish-out, got {fish_ids}")
    only_heldout = {int(v) for v in parse_int_list(str(getattr(args, "only_heldout_fish_ids", "")))}
    if only_heldout:
        missing = sorted(int(v) for v in only_heldout if int(v) not in {int(fid) for fid in fish_ids})
        if missing:
            raise ValueError(f"--only-heldout-fish-ids contains ids not present in --fish-ids: {missing}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / f"{args.exp_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "run_args.json", vars(args))
    snapshot_clean_pack_entrypoint(
        run_dir=run_dir,
        argv=[str(x) for x in sys.argv],
        entry_script=Path(__file__).resolve(),
        extra_files=[Path(__file__).resolve().parent / "identity_guided_grouping_decoding.py"],
    )
    data.set_global_seed(int(args.seed), deterministic=bool(args.deterministic))
    print(f"[identity_guided_grouping-manifold] run_dir={run_dir}", flush=True)
    print(f"[identity_guided_grouping-manifold] device={device}", flush=True)

    base_contexts: dict[int, BaseFishContext] = {}
    print("[identity_guided_grouping-manifold] loading fish contexts", flush=True)
    for fish_id in fish_ids:
        base_contexts[int(fish_id)] = load_base_fish_context(int(fish_id), args)

    case_rows = []
    case_records = []
    for heldout_idx, heldout_fish_id in enumerate(fish_ids):
        if only_heldout and int(heldout_fish_id) not in only_heldout:
            print(f"[identity_guided_grouping-manifold] skipping heldout={heldout_fish_id}", flush=True)
            continue
        source_fish_ids = [int(fid) for fid in fish_ids if int(fid) != int(heldout_fish_id)]
        case_name = build_transfer_case_name(int(heldout_fish_id), source_fish_ids)
        case_dir = run_dir / "cases" / case_name
        split_dir = case_dir / "train_val"
        source_ctxs, target_ctx = build_case_contexts(
            base_contexts,
            source_fish_ids=source_fish_ids,
            target_fish_id=int(heldout_fish_id),
            static_normalize_mode=str(args.static_normalize_mode),
        )
        case_contexts = {**source_ctxs, int(heldout_fish_id): target_ctx}
        preview_model = build_manifold_model(next(iter(source_ctxs.values())), latent_dim=int(args.latent_dim), args=args)
        train_index_map = {}
        val_index_map = {}
        for fish_id in source_fish_ids:
            full_idx = window_valid_time_indices(source_ctxs[int(fish_id)], preview_model)
            tr_idx, va_idx = split_tail_indices(full_idx, float(args.val_frac))
            train_index_map[int(fish_id)] = tr_idx
            val_index_map[int(fish_id)] = va_idx
        del preview_model
        case_seed = int(args.seed) + 10000 * int(heldout_idx) + int(heldout_fish_id)
        fish_order = [int(source_fish_ids[0]), int(source_fish_ids[1]), int(heldout_fish_id)]
        print(
            f"[identity_guided_grouping-manifold] training case={case_name} heldout={heldout_fish_id} sources={source_fish_ids}",
            flush=True,
        )
        model, split_summary, ckpt_paths = fit_case_model(
            split_dir=split_dir,
            source_ctxs=source_ctxs,
            train_index_map=train_index_map,
            val_index_map=val_index_map,
            case_contexts=case_contexts,
            fish_order=fish_order,
            device=device,
            case_seed=int(case_seed),
            args=args,
        )
        identity_guided_grouping_embeddings_best = compute_case_identity_guided_grouping_embeddings(
            model,
            case_contexts=case_contexts,
            fish_order=fish_order,
            args=args,
            case_seed=int(case_seed),
            device=device,
        )
        last_model, last_ckpt = load_model_from_checkpoint(
            reference_ctx=next(iter(source_ctxs.values())),
            ckpt_path=ckpt_paths["last"],
            args=args,
            device=device,
        )
        identity_guided_grouping_embeddings_last = compute_case_identity_guided_grouping_embeddings(
            last_model,
            case_contexts=case_contexts,
            fish_order=fish_order,
            args=args,
            case_seed=int(case_seed),
            device=device,
        )
        for fish_id in fish_order:
            save_embedding_artifacts(
                identity_guided_grouping_embeddings_best[int(fish_id)],
                out_dir=case_dir / "identity_guided_grouping" / f"fish_{int(fish_id)}",
                base_ctx=base_contexts[int(fish_id)],
            )
            save_embedding_artifacts(
                identity_guided_grouping_embeddings_best[int(fish_id)],
                out_dir=case_dir / "identity_guided_grouping_best" / f"fish_{int(fish_id)}",
                base_ctx=base_contexts[int(fish_id)],
            )
            save_embedding_artifacts(
                identity_guided_grouping_embeddings_last[int(fish_id)],
                out_dir=case_dir / "identity_guided_grouping_last" / f"fish_{int(fish_id)}",
                base_ctx=base_contexts[int(fish_id)],
            )
        save_json(
            case_dir / "checkpoint_summary.json",
            {
                "best": {
                    "path": ckpt_paths["best"],
                },
                "last": {
                    "path": ckpt_paths["last"],
                    "epoch": int(last_ckpt.get("epoch", -1)),
                    "train_loss": last_ckpt.get("train_loss"),
                    "val_loss": last_ckpt.get("val_loss"),
                },
            },
        )
        print(f"[identity_guided_grouping-manifold] finished IdentityGuidedGrouping embeddings for case={case_name}", flush=True)
        case_records.append(
            {
                "case_name": str(case_name),
                "heldout_fish_id": int(heldout_fish_id),
                "source_fish_ids": [int(fid) for fid in source_fish_ids],
                "fish_order": [int(fid) for fid in fish_order],
                "case_dir": case_dir,
                "split_summary": split_summary,
                "identity_guided_grouping_embeddings": identity_guided_grouping_embeddings_best,
                "identity_guided_grouping_embeddings_best": identity_guided_grouping_embeddings_best,
                "identity_guided_grouping_embeddings_last": identity_guided_grouping_embeddings_last,
                "checkpoint_paths": {k: str(v) for k, v in ckpt_paths.items()},
            }
        )

    for rec in case_records:
        case_name = str(rec["case_name"])
        heldout_fish_id = int(rec["heldout_fish_id"])
        source_fish_ids = [int(fid) for fid in rec["source_fish_ids"]]
        fish_order = [int(fid) for fid in rec["fish_order"]]
        case_dir = Path(rec["case_dir"])
        split_summary = rec["split_summary"]
        identity_guided_grouping_embeddings = rec["identity_guided_grouping_embeddings"]
        identity_guided_grouping_embeddings_best = rec["identity_guided_grouping_embeddings_best"]
        identity_guided_grouping_embeddings_last = rec["identity_guided_grouping_embeddings_last"]
        embeddings_by_method = {"identity_guided_grouping": identity_guided_grouping_embeddings}
        plot_method_fish_panel(
            embeddings_by_method,
            fish_order=fish_order,
            contexts=base_contexts,
            out_png=case_dir / "latent_panel_time.png",
            title=f"Case {case_name} | time-colored latent comparison",
            color_mode="time",
            heldout_fish_id=int(heldout_fish_id),
        )
        plot_method_fish_panel(
            embeddings_by_method,
            fish_order=fish_order,
            contexts=base_contexts,
            out_png=case_dir / "latent_panel_stimulus.png",
            title=f"Case {case_name} | stimulus-colored latent comparison",
            color_mode="stimulus",
            heldout_fish_id=int(heldout_fish_id),
        )
        plot_method_fish_panel(
            {
                "identity_guided_grouping_best": identity_guided_grouping_embeddings_best,
                "identity_guided_grouping_last": identity_guided_grouping_embeddings_last,
            },
            fish_order=fish_order,
            contexts=base_contexts,
            out_png=case_dir / "identity_guided_grouping_checkpoint_compare_time.png",
            title=f"Case {case_name} | IdentityGuidedGrouping best vs last checkpoint",
            color_mode="time",
            heldout_fish_id=int(heldout_fish_id),
        )
        case_summary = {
            "case_name": str(case_name),
            "heldout_fish_id": int(heldout_fish_id),
            "source_fish_ids": [int(fid) for fid in source_fish_ids],
            "train_split": split_summary,
            "checkpoint_paths": rec["checkpoint_paths"],
            "identity_guided_grouping": {
                "fish_metrics": {
                    str(int(fid)): embedding_metric_payload(identity_guided_grouping_embeddings[int(fid)])
                    for fid in fish_order
                },
            },
            "identity_guided_grouping_last": {
                "fish_metrics": {
                    str(int(fid)): embedding_metric_payload(identity_guided_grouping_embeddings_last[int(fid)])
                    for fid in fish_order
                },
            },
        }
        save_json(case_dir / "case_summary.json", case_summary)
        case_rows.append(case_summary)

    plot_run_summary_latent_figure(
        case_records,
        base_contexts,
        run_dir / "comparison_summary_latent.png",
    )
    plot_run_summary_metrics_figure(
        case_rows,
        run_dir / "comparison_summary_metrics.png",
    )

    overall_summary = {
        "exp_name": str(args.exp_name),
        "timestamp": ts,
        "device": str(device),
        "fish_ids": [int(fid) for fid in fish_ids],
        "cases": case_rows,
    }
    save_json(run_dir / "summary.json", overall_summary)
    print(json.dumps(to_jsonable(overall_summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

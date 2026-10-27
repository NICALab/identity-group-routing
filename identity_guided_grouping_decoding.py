#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F

from src import data, eval as eval_utils, plotting
from src.io_utils import (
    build_transfer_case_name,
    finalize_bundle_args,
    infer_default_data_root,
    parse_int_list,
    resolve_device,
    safe_mean,
    safe_pearsonr,
    safe_std,
    save_json,
    save_torch_checkpoint,
)
from src.repro import snapshot_clean_pack_entrypoint
from src.identity_guided_grouping import build_identity_guided_grouping_encoder, build_time_batch, prepare_subset_cache
from src.transfer_context import (
    BaseFishContext,
    BehaviorNormalizer,
    CaseFishContext,
    apply_static_norm_stats,
    behavior_normalizer_to_dict,
    build_batches,
    build_behavior_normalizer,
    build_case_contexts,
    build_fixed_subset_map,
    build_subset_ensemble,
    build_tail_split,
    denormalize_behavior_array,
    get_targets_for_indices,
    infer_num_classes,
    load_base_fish_context,
    sample_neuron_subset_indices,
)
from src.models import IdentityGuidedGroupingEncoder


CLEAN_MODEL_MODALITIES = ("gene", "pos", "atlas", "region", "activity")


def get_enabled_modalities(args: argparse.Namespace) -> list[str]:
    enabled: list[str] = []
    if bool(args.use_gene):
        enabled.append("gene")
    if bool(args.use_pos):
        enabled.append("pos")
    if bool(args.use_atlas):
        enabled.append("atlas")
    if bool(getattr(args, "use_region", False)):
        enabled.append("region")
    if bool(getattr(args, "use_activity", False)):
        enabled.append("activity")
    return enabled


def canonicalize_model_kind(args: argparse.Namespace) -> str:
    args.model_kind = "identity_guided_grouping"
    return "identity_guided_grouping"

def finalize_decoding_args(args: argparse.Namespace) -> argparse.Namespace:
    args = finalize_bundle_args(args, atlas_precomputed_fallback=data.DEFAULT_PRECOMPUTED_ATLAS_HCR_DIR)
    canonicalize_model_kind(args)
    args.clean_modalities = list(CLEAN_MODEL_MODALITIES)
    args.enabled_modalities = get_enabled_modalities(args)
    return args


def build_static_features_for_subset(
    ctx: CaseFishContext,
    *,
    neuron_indices: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray | None:
    idx = np.asarray(neuron_indices, dtype=np.int64)
    if idx.size <= 0:
        return None if ctx.static_nf is None else np.empty((0, ctx.static_nf.shape[1]), dtype=np.float32)
    genes_nf = None
    if bool(args.use_gene) and ctx.base.fish.genes_nf is not None:
        genes_nf = np.asarray(ctx.base.fish.genes_nf[idx], dtype=np.float32)
    coords_nf = (
        None if not bool(args.use_pos) or ctx.base.fish.coords_nf is None else np.asarray(ctx.base.fish.coords_nf[idx], dtype=np.float32)
    )
    atlas_nf = (
        None
        if not bool(args.use_atlas) or ctx.base.fish.regions_nf is None
        else np.asarray(ctx.base.fish.regions_nf[idx], dtype=np.float32)
    )
    region_nf = (
        None
        if not bool(getattr(args, "use_region", False)) or ctx.base.fish.region_membership_nf is None
        else np.asarray(ctx.base.fish.region_membership_nf[idx], dtype=np.float32)
    )
    activity_nf = (
        None
        if not bool(getattr(args, "use_activity", False)) or ctx.base.fish.activity_nf is None
        else np.asarray(ctx.base.fish.activity_nf[idx], dtype=np.float32)
    )
    static_raw_nf = data.build_static_features(
        genes_nf,
        coords_nf,
        use_gene=bool(args.use_gene),
        use_pos=bool(args.use_pos),
        atlas_nf=atlas_nf,
        use_atlas=bool(args.use_atlas),
        region_nf=region_nf,
        use_region=bool(getattr(args, "use_region", False)),
        activity_nf=activity_nf,
        use_activity=bool(getattr(args, "use_activity", False)),
    )
    return apply_static_norm_stats(
        static_raw_nf,
        mean_f=ctx.static_norm_mean_f,
        std_f=ctx.static_norm_std_f,
    )


def build_model_inputs_for_subset(
    model: torch.nn.Module,
    *,
    ctx: CaseFishContext,
    time_indices: np.ndarray,
    neuron_indices: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor | None, dict | None]:
    n_idx = np.asarray(neuron_indices, dtype=np.int64)
    static_nf = None
    if ctx.static_nf is not None:
        static_nf = build_static_features_for_subset(
            ctx,
            neuron_indices=n_idx,
            args=args,
        )
    x_nt_ctx, static_t, static_cache = prepare_subset_cache(
        model,
        ctx=ctx,
        neuron_indices=n_idx,
        device=device,
        static_nf_override=static_nf,
    )
    x_btn = build_time_batch(model, x_nt_ctx=x_nt_ctx, time_indices=time_indices, device=device)
    return x_btn, static_t, static_cache


def build_model(
    reference_ctx: CaseFishContext,
    *,
    args: argparse.Namespace,
    out_dim: int,
) -> IdentityGuidedGroupingEncoder:
    return build_identity_guided_grouping_encoder(
        reference_ctx,
        d_model=int(out_dim),
        args=args,
        normalize_out=False,
    )



def build_stimulus_class_weights(
    source_ctxs: dict[int, CaseFishContext],
    train_index_map: dict[int, np.ndarray],
    *,
    num_classes: int,
    mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    if str(mode).strip().lower() == "none":
        return None
    labels = [
        get_targets_for_indices(source_ctxs[int(fid)], "stimulus", idx)
        for fid, idx in train_index_map.items()
        if np.asarray(idx).size > 0
    ]
    if not labels:
        return None
    y = np.concatenate(labels, axis=0).astype(np.int64, copy=False)
    counts = np.bincount(y, minlength=int(num_classes)).astype(np.float32)
    counts = np.where(counts > 0.0, counts, 1.0)
    weights = counts.sum() / (float(num_classes) * counts)
    weights = weights / max(float(weights.mean()), 1e-6)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def compute_behavior_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    huber_delta: float,
) -> torch.Tensor:
    mode_key = str(mode).strip().lower()
    if mode_key == "mse":
        return F.mse_loss(pred, target)
    if mode_key != "huber":
        raise ValueError(f"Unsupported behavior_loss_mode: {mode}")
    return F.huber_loss(pred, target, delta=float(huber_delta))


def train_one_epoch(
    model: torch.nn.Module,
    *,
    source_ctxs: dict[int, CaseFishContext],
    train_index_map: dict[int, np.ndarray],
    task: str,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    epoch_seed: int,
    grad_clip_norm: float,
    class_weights_t: torch.Tensor | None,
    behavior_normalizer: BehaviorNormalizer | None,
    args: argparse.Namespace,
) -> float:
    model.train()
    batches = build_batches(
        train_index_map,
        batch_size=int(batch_size),
        shuffle=True,
        seed=int(epoch_seed),
    )
    total_loss = 0.0
    total_count = 0
    for batch_num, (fish_id, batch_idx) in enumerate(batches, start=1):
        ctx = source_ctxs[int(fish_id)]
        y_np = get_targets_for_indices(ctx, task, batch_idx, behavior_normalizer=behavior_normalizer)
        neuron_idx = sample_neuron_subset_indices(
            ctx,
            requested=int(args.sampled_neurons),
            seed=int(epoch_seed) + 100000 * int(fish_id) + int(batch_num),
        )
        x_t, static_t, static_cache = build_model_inputs_for_subset(
            model,
            ctx=ctx,
            time_indices=batch_idx,
            neuron_indices=neuron_idx,
            device=device,
            args=args,
        )
        optimizer.zero_grad(set_to_none=True)
        out = model(x_t, static_t, static_cache=static_cache)
        if str(task) == "behavior":
            y_t = torch.as_tensor(y_np, dtype=torch.float32, device=device)
            loss = compute_behavior_loss(
                out.squeeze(-1),
                y_t,
                mode=str(getattr(args, "behavior_loss_mode", "huber")),
                huber_delta=float(args.behavior_huber_delta),
            )
        else:
            y_t = torch.as_tensor(y_np, dtype=torch.int64, device=device)
            loss = F.cross_entropy(out, y_t, weight=class_weights_t)
        loss.backward()
        if float(grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()
        total_loss += float(loss.item()) * int(batch_idx.size)
        total_count += int(batch_idx.size)
    return float(total_loss / max(1, total_count))


@torch.no_grad()
def predict_time_indices(
    model: torch.nn.Module,
    *,
    ctx: CaseFishContext,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    neuron_indices: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    idx = np.asarray(indices, dtype=np.int64)
    outputs: list[np.ndarray] = []
    if neuron_indices is None:
        neuron_idx = np.arange(int(ctx.base.fish.neural_nt.shape[0]), dtype=np.int64)
    else:
        neuron_idx = np.asarray(neuron_indices, dtype=np.int64)
    x_nt_ctx, static_t, static_cache = prepare_subset_cache(
        model,
        ctx=ctx,
        neuron_indices=neuron_idx,
        device=device,
    )
    for start in range(0, int(idx.size), int(batch_size)):
        batch_idx = idx[start:start + int(batch_size)]
        x_t = build_time_batch(model, x_nt_ctx=x_nt_ctx, time_indices=batch_idx, device=device)
        out = model(x_t, static_t, static_cache=static_cache)
        outputs.append(out.detach().cpu().numpy())
    if not outputs:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def evaluate_split(
    model: torch.nn.Module,
    *,
    source_ctxs: dict[int, CaseFishContext],
    index_map: dict[int, np.ndarray],
    task: str,
    batch_size: int,
    device: torch.device,
    subset_index_map: dict[int, np.ndarray] | None = None,
    behavior_normalizer: BehaviorNormalizer | None = None,
) -> dict:
    per_fish: dict[str, dict] = {}
    metric_values: list[float] = []
    loss_values: list[float] = []
    for fish_id, idx in index_map.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        if idx_arr.size <= 0:
            continue
        ctx = source_ctxs[int(fish_id)]
        pred = predict_time_indices(
            model,
            ctx=ctx,
            indices=idx_arr,
            batch_size=int(batch_size),
            device=device,
            neuron_indices=None if subset_index_map is None else subset_index_map.get(int(fish_id)),
        )
        if str(task) == "behavior":
            y_true = np.asarray(ctx.base.behavior_target_t[idx_arr], dtype=np.float32)
            pred_vec_norm = np.asarray(pred, dtype=np.float32).reshape(-1)
            pred_vec = denormalize_behavior_array(
                pred_vec_norm,
                behavior_normalizer,
                fish_id=int(ctx.base.fish.fish_id),
            )
            mse = float(np.mean((pred_vec - y_true) ** 2))
            corr = safe_pearsonr(y_true, pred_vec)
            metric_values.append(corr)
            loss_values.append(mse)
            per_fish[str(int(fish_id))] = {
                "corr": corr,
                "mse": mse,
                "n_frames": int(idx_arr.size),
            }
        else:
            y_true = get_targets_for_indices(ctx, task, idx_arr)
            logits = np.asarray(pred, dtype=np.float32)
            pred_cls = np.asarray(np.argmax(logits, axis=1), dtype=np.int64)
            metrics = eval_utils.compute_classification_metrics(y_true, pred_cls)
            metric_values.append(float(metrics["bal_acc"]))
            per_fish[str(int(fish_id))] = {
                **metrics,
                "n_frames": int(idx_arr.size),
            }
    return {
        "metric": safe_mean(metric_values),
        "loss": safe_mean(loss_values) if loss_values else float("nan"),
        "per_fish": per_fish,
    }


def build_split_summary_payload(
    *,
    task: str,
    args: argparse.Namespace,
    train_index_map: dict[int, np.ndarray],
    val_index_map: dict[int, np.ndarray],
    val_subset_map: dict[int, np.ndarray],
    best_epoch: int,
    best_metric: float,
    stopped_epoch: int,
    history_rows: list[dict],
    behavior_normalizer: BehaviorNormalizer | None,
    split_mode: str = "tail",
) -> dict:
    return {
        "split_mode": str(split_mode),
        "val_frac": float(args.val_frac),
        "best_epoch": int(best_epoch),
        "best_metric": None if not np.isfinite(best_metric) else float(best_metric),
        "stopped_epoch": int(stopped_epoch),
        "train_index_sizes": {str(fid): int(np.asarray(idx).size) for fid, idx in train_index_map.items()},
        "val_index_sizes": {str(fid): int(np.asarray(idx).size) for fid, idx in val_index_map.items()},
        "val_subset_sizes": {str(fid): int(np.asarray(idx).size) for fid, idx in val_subset_map.items()},
        "behavior_target_normalization": behavior_normalizer_to_dict(behavior_normalizer),
        "history": history_rows,
    }


def persist_split_progress(
    *,
    split_dir: Path,
    task: str,
    args: argparse.Namespace,
    train_index_map: dict[int, np.ndarray],
    val_index_map: dict[int, np.ndarray],
    val_subset_map: dict[int, np.ndarray],
    train_losses: list[float],
    val_scores: list[float],
    best_epoch: int,
    best_metric: float,
    history_rows: list[dict],
    behavior_normalizer: BehaviorNormalizer | None,
    save_svg: bool,
    split_mode: str = "tail",
) -> dict:
    split_summary = build_split_summary_payload(
        task=str(task),
        args=args,
        train_index_map=train_index_map,
        val_index_map=val_index_map,
        val_subset_map=val_subset_map,
        best_epoch=int(best_epoch),
        best_metric=float(best_metric),
        stopped_epoch=int(len(train_losses)),
        history_rows=history_rows,
        behavior_normalizer=behavior_normalizer,
        split_mode=str(split_mode),
    )
    save_json(split_dir / "split_summary.json", split_summary)
    plotting.plot_training_history(
        train_losses,
        val_scores,
        split_dir / "training_curve.png",
        title=f"{task} train/val split",
        score_label="val corr" if str(task) == "behavior" else "val bal_acc",
        best_epoch=int(best_epoch) if int(best_epoch) > 0 else None,
        save_svg=bool(save_svg),
    )
    return split_summary


def fit_single_split(
    *,
    split_dir: Path,
    source_ctxs: dict[int, CaseFishContext],
    train_index_map: dict[int, np.ndarray],
    val_index_map: dict[int, np.ndarray],
    task: str,
    args: argparse.Namespace,
    out_dim: int,
    device: torch.device,
    case_seed: int,
    num_classes: int | None,
    behavior_normalizer: BehaviorNormalizer | None,
    split_mode: str = "tail",
) -> tuple[torch.nn.Module, dict]:
    data.set_global_seed(int(case_seed) + 31, deterministic=bool(args.deterministic))
    reference_ctx = next(iter(source_ctxs.values()))
    model = build_model(reference_ctx, args=args, out_dim=int(out_dim)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    class_weights_t = None
    if str(task) == "stimulus":
        class_weights_t = build_stimulus_class_weights(
            source_ctxs,
            train_index_map,
            num_classes=int(num_classes),
            mode=str(args.stimulus_class_weight_mode),
            device=device,
        )
    val_subset_map = build_fixed_subset_map(
        source_ctxs,
        requested=int(args.eval_neurons),
        seed=int(case_seed) + 50000,
    )
    train_losses: list[float] = []
    val_scores: list[float] = []
    selection_scores: list[float] = []
    best_metric = -float("inf")
    best_selection_metric = -float("inf")
    best_epoch = 0
    patience_count = 0
    history_rows: list[dict] = []
    ema_selection_metric: float | None = None
    selection_mode = str(getattr(args, "val_selection_mode", "raw")).strip().lower()
    selection_min_epoch = max(1, int(getattr(args, "selection_min_epoch", 5)))
    ema_alpha = float(getattr(args, "val_selection_ema_alpha", 0.5))
    for epoch in range(1, int(args.max_epochs) + 1):
        train_loss = train_one_epoch(
            model,
            source_ctxs=source_ctxs,
            train_index_map=train_index_map,
            task=task,
            batch_size=int(args.batch_size),
            device=device,
            optimizer=optimizer,
            epoch_seed=int(case_seed) + 1000 + int(epoch),
            grad_clip_norm=float(args.grad_clip_norm),
            class_weights_t=class_weights_t,
            behavior_normalizer=behavior_normalizer,
            args=args,
        )
        val_stats = evaluate_split(
            model,
            source_ctxs=source_ctxs,
            index_map=val_index_map,
            task=task,
            batch_size=int(args.eval_batch_size),
            device=device,
            subset_index_map=val_subset_map,
            behavior_normalizer=behavior_normalizer,
        )
        val_metric = float(val_stats["metric"])
        train_losses.append(float(train_loss))
        val_scores.append(val_metric)
        selection_metric = val_metric
        if selection_mode == "ema":
            if ema_selection_metric is None or not np.isfinite(ema_selection_metric):
                ema_selection_metric = float(val_metric)
            else:
                ema_selection_metric = float(ema_alpha) * float(val_metric) + (1.0 - float(ema_alpha)) * float(
                    ema_selection_metric
                )
            selection_metric = float(ema_selection_metric)
        selection_scores.append(float(selection_metric))
        eligible_for_selection = int(epoch) >= int(selection_min_epoch)
        improved = eligible_for_selection and bool(np.isfinite(selection_metric)) and (
            selection_metric > (best_selection_metric + float(args.early_stop_min_delta))
        )
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_metric": None if not np.isfinite(val_metric) else float(val_metric),
                "selection_metric": None if not np.isfinite(selection_metric) else float(selection_metric),
                "val_detail": val_stats["per_fish"],
            }
        )
        if improved:
            best_metric = float(val_metric)
            best_selection_metric = float(selection_metric)
            best_epoch = int(epoch)
            patience_count = 0
            save_torch_checkpoint(
                split_dir / "checkpoints" / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_metric=val_metric,
                meta={
                    "task": str(task),
                    "split": str(split_mode),
                    "best": True,
                    "selection_metric": None if not np.isfinite(selection_metric) else float(selection_metric),
                    "source_fish_ids": [int(fid) for fid in sorted(source_ctxs)],
                },
            )
        else:
            if eligible_for_selection:
                patience_count += 1
        if int(args.checkpoint_every) > 0 and epoch % int(args.checkpoint_every) == 0:
            save_torch_checkpoint(
                split_dir / "checkpoints" / f"epoch_{int(epoch):04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_metric=val_metric,
                meta={
                    "task": str(task),
                    "split": str(split_mode),
                    "selection_metric": None if not np.isfinite(selection_metric) else float(selection_metric),
                    "source_fish_ids": [int(fid) for fid in sorted(source_ctxs)],
                },
            )
        persist_split_progress(
            split_dir=split_dir,
            task=str(task),
            args=args,
            train_index_map=train_index_map,
            val_index_map=val_index_map,
            val_subset_map=val_subset_map,
            train_losses=train_losses,
            val_scores=val_scores,
            best_epoch=int(best_epoch),
            best_metric=float(best_metric),
            history_rows=history_rows,
            behavior_normalizer=behavior_normalizer,
            save_svg=False,
            split_mode=str(split_mode),
        )
        if (
            not bool(getattr(args, "disable_early_stop", False))
            and epoch >= int(selection_min_epoch)
            and patience_count >= int(args.patience)
        ):
            break
    save_torch_checkpoint(
        split_dir / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(len(train_losses)),
        train_loss=float(train_losses[-1]) if train_losses else float("nan"),
        val_metric=float(val_scores[-1]) if val_scores else None,
        meta={
            "task": str(task),
            "split": str(split_mode),
            "best": False,
            "source_fish_ids": [int(fid) for fid in sorted(source_ctxs)],
        },
    )
    if best_epoch <= 0:
        best_epoch = len(train_losses)
    best_ckpt_path = split_dir / "checkpoints" / "best.pt"
    last_ckpt_path = split_dir / "checkpoints" / "last.pt"
    if not best_ckpt_path.exists():
        shutil.copy2(last_ckpt_path, best_ckpt_path)
    checkpoint_selection = str(getattr(args, "checkpoint_selection", "best")).strip().lower()
    if checkpoint_selection == "last":
        selected_ckpt_name = "last.pt"
        selected_ckpt_path = last_ckpt_path
    else:
        selected_ckpt_name = "best.pt"
        selected_ckpt_path = best_ckpt_path
    selected_ckpt = torch.load(selected_ckpt_path, map_location=device)
    model.load_state_dict(selected_ckpt["model_state_dict"])
    split_summary = persist_split_progress(
        split_dir=split_dir,
        task=str(task),
        args=args,
        train_index_map=train_index_map,
        val_index_map=val_index_map,
        val_subset_map=val_subset_map,
        train_losses=train_losses,
        val_scores=val_scores,
        best_epoch=int(best_epoch),
        best_metric=float(best_metric),
        history_rows=history_rows,
        behavior_normalizer=behavior_normalizer,
        save_svg=True,
        split_mode=str(split_mode),
    )
    split_summary["checkpoint_selection"] = str(checkpoint_selection)
    split_summary["selected_checkpoint_name"] = str(selected_ckpt_name)
    split_summary["selected_epoch"] = int(selected_ckpt.get("epoch", len(train_losses)))
    split_summary["val_selection_mode"] = str(selection_mode)
    split_summary["selection_min_epoch"] = int(selection_min_epoch)
    split_summary["best_selection_metric"] = (
        None if not np.isfinite(best_selection_metric) else float(best_selection_metric)
    )
    selected_val_metric = selected_ckpt.get("val_metric")
    split_summary["selected_val_metric"] = (
        None if selected_val_metric is None or not np.isfinite(selected_val_metric) else float(selected_val_metric)
    )
    split_summary["early_stop_enabled"] = not bool(getattr(args, "disable_early_stop", False))
    split_summary["trained_epochs"] = int(len(train_losses))
    split_summary["reached_max_epochs"] = int(len(train_losses)) >= int(args.max_epochs)
    save_json(split_dir / "split_summary.json", split_summary)
    return model, split_summary


def vote_from_logits(logits_sbc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_arr = np.asarray(logits_sbc, dtype=np.float32)
    if logits_arr.ndim != 3:
        raise ValueError(f"Expected logits_sbc to have shape (S,T,C), got {tuple(logits_arr.shape)}")
    pred_sub = np.asarray(np.argmax(logits_arr, axis=2), dtype=np.int64)
    num_subsets, t_len, num_classes = logits_arr.shape
    vote_counts = np.zeros((t_len, num_classes), dtype=np.int64)
    row_idx = np.repeat(np.arange(t_len, dtype=np.int64), int(num_subsets))
    cls_idx = pred_sub.transpose(1, 0).reshape(-1)
    np.add.at(vote_counts, (row_idx, cls_idx), 1)
    mean_logits = logits_arr.mean(axis=0)
    top_votes = vote_counts.max(axis=1, keepdims=True)
    tie_mask = vote_counts == top_votes
    tie_break_logits = np.where(tie_mask, mean_logits, -np.inf)
    pred_vote = np.asarray(np.argmax(tie_break_logits, axis=1), dtype=np.int64)
    return pred_vote, vote_counts, mean_logits


def evaluate_heldout(
    model: torch.nn.Module,
    *,
    target_ctx: CaseFishContext,
    task: str,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
    num_classes: int | None,
    source_fish_ids: list[int],
    neuron_subsets: list[np.ndarray] | None,
    behavior_normalizer: BehaviorNormalizer | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fish_id = int(target_ctx.base.fish.fish_id)
    all_idx = np.arange(int(target_ctx.base.fish.neural_nt.shape[1]), dtype=np.int64)
    subset_list = (
        [np.arange(int(target_ctx.base.fish.neural_nt.shape[0]), dtype=np.int64)]
        if neuron_subsets is None or len(neuron_subsets) <= 0
        else [np.asarray(subset_idx, dtype=np.int64) for subset_idx in neuron_subsets]
    )
    pred_subsets = [
        predict_time_indices(
            model,
            ctx=target_ctx,
            indices=all_idx,
            batch_size=int(batch_size),
            device=device,
            neuron_indices=subset_idx,
        )
        for subset_idx in subset_list
    ]
    subset_stack = np.stack(subset_list, axis=0)
    if str(task) == "behavior":
        y_true = np.asarray(target_ctx.base.behavior_target_t, dtype=np.float32)
        pred_stack_norm = np.asarray(pred_subsets, dtype=np.float32).reshape(len(subset_list), -1)
        y_pred_norm = pred_stack_norm.mean(axis=0)
        pred_stack = denormalize_behavior_array(
            pred_stack_norm,
            behavior_normalizer,
            fish_id=int(target_ctx.base.fish.fish_id),
        )
        y_pred = denormalize_behavior_array(
            y_pred_norm,
            behavior_normalizer,
            fish_id=int(target_ctx.base.fish.fish_id),
        )
        corr = safe_pearsonr(y_true, y_pred)
        mse = float(np.mean((y_true - y_pred) ** 2))
        payload = {
            "task": "behavior",
            "target_fish": int(fish_id),
            "source_fish_ids": [int(fid) for fid in source_fish_ids],
            "behavior_target_label": str(target_ctx.base.behavior_target_label),
            "test_corr": corr,
            "test_mse": mse,
            "n_frames": int(y_true.size),
            "n_eval_neurons": int(subset_stack.shape[1]),
            "n_eval_subsets": int(subset_stack.shape[0]),
            "eval_aggregation": "mean",
            "behavior_target_normalization": behavior_normalizer_to_dict(behavior_normalizer),
        }
        np.savez(
            out_dir / "behavior_predictions.npz",
            y_true=y_true,
            y_pred=y_pred,
            y_pred_norm=y_pred_norm,
            y_pred_subsets=pred_stack,
            y_pred_subsets_norm=pred_stack_norm,
            frame_idx=all_idx,
            neuron_idx_subsets=subset_stack,
        )
        plotting.plot_behavior_predictions(
            y_true,
            y_pred,
            out_dir / "behavior_predictions.png",
            title=(
                f"Behavior transfer | src={source_fish_ids} -> heldout={fish_id} "
                f"| corr={corr:.3f} | ens={int(subset_stack.shape[0])}"
            ),
        )
        save_json(out_dir / "behavior_metrics.json", payload)
        return payload

    if target_ctx.base.stimulus_class_t is None or target_ctx.base.stimulus_active_mask_t is None:
        raise ValueError(f"Held-out fish {fish_id} is missing stimulus targets.")
    logits_stack = np.asarray(pred_subsets, dtype=np.float32)
    pred_cls_subsets = np.asarray(np.argmax(logits_stack, axis=2), dtype=np.int64)
    pred_cls_all, vote_counts_all, mean_logits_all = vote_from_logits(logits_stack)
    active_mask = np.asarray(target_ctx.base.stimulus_active_mask_t, dtype=bool)
    active_idx = np.flatnonzero(active_mask)
    y_true = np.asarray(target_ctx.base.stimulus_class_t[active_idx], dtype=np.int64)
    y_pred = np.asarray(pred_cls_all[active_idx], dtype=np.int64)
    metrics = eval_utils.compute_classification_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(int(num_classes)))
    payload = {
        "task": "stimulus",
        "target_fish": int(fish_id),
        "source_fish_ids": [int(fid) for fid in source_fish_ids],
        "stimulus_target_label": target_ctx.base.stimulus_target_label,
        "n_active_frames": int(active_idx.size),
        "n_total_frames": int(active_mask.size),
        "n_eval_neurons": int(subset_stack.shape[1]),
        "n_eval_subsets": int(subset_stack.shape[0]),
        "eval_aggregation": "vote",
        "acc": float(metrics["acc"]),
        "bal_acc": float(metrics["bal_acc"]),
        "macro_f1": float(metrics["macro_f1"]),
        "confusion_matrix": cm.tolist(),
    }
    np.savez(
        out_dir / "stimulus_predictions.npz",
        logits_subsets=logits_stack,
        logits_mean_full=mean_logits_all,
        pred_subsets=pred_cls_subsets,
        pred_full=pred_cls_all,
        vote_counts_full=vote_counts_all,
        active_mask=active_mask.astype(np.uint8),
        active_idx=active_idx,
        y_true_active=y_true,
        y_pred_active=y_pred,
        neuron_idx_subsets=subset_stack,
    )
    plotting.plot_confusion_matrix(
        cm,
        out_dir / "stimulus_confusion_matrix.png",
        title=(
            f"Stimulus transfer | src={source_fish_ids} -> heldout={fish_id} "
            f"| bal_acc={metrics['bal_acc']:.3f} | ens={int(subset_stack.shape[0])}"
        ),
        normalize=True,
    )
    save_json(out_dir / "stimulus_metrics.json", payload)
    return payload

def run_task(
    *,
    task: str,
    base_contexts: dict[int, BaseFishContext],
    fish_ids: list[int],
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> dict:
    task_dir = run_dir / str(task)
    task_dir.mkdir(parents=True, exist_ok=True)
    num_classes = infer_num_classes(base_contexts, int(args.stimulus_num_classes)) if str(task) == "stimulus" else None
    case_summaries: list[dict] = []
    for heldout_fish_id in fish_ids:
        source_fish_ids = [int(fid) for fid in fish_ids if int(fid) != int(heldout_fish_id)]
        case_name = build_transfer_case_name(int(heldout_fish_id), source_fish_ids)
        case_dir = task_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        source_ctxs, target_ctx = build_case_contexts(
            base_contexts,
            source_fish_ids=source_fish_ids,
            target_fish_id=int(heldout_fish_id),
            static_normalize_mode=str(args.static_normalize_mode),
        )
        case_seed = int(args.seed) + 1000 + int(heldout_fish_id)
        split = build_tail_split(
            source_ctxs,
            task=str(task),
            val_frac=float(args.val_frac),
        )
        behavior_normalizer = (
            build_behavior_normalizer(
                source_ctxs,
                split["train"],
                mode=str(args.behavior_target_normalization),
            )
            if str(task) == "behavior"
            else None
        )
        best_model, split_summary = fit_single_split(
            split_dir=case_dir / "train_val",
            source_ctxs=source_ctxs,
            train_index_map=split["train"],
            val_index_map=split["val"],
            task=str(task),
            args=args,
            out_dim=1 if str(task) == "behavior" else int(num_classes),
            device=device,
            case_seed=int(case_seed),
            num_classes=num_classes,
            behavior_normalizer=behavior_normalizer,
        )
        heldout_subset_list = build_subset_ensemble(
            target_ctx,
            requested=int(args.eval_neurons),
            num_subsets=int(args.eval_ensemble_subsets),
            seed=int(case_seed) + 900000 + int(heldout_fish_id),
        )
        heldout_subset_stack = np.stack(heldout_subset_list, axis=0)
        np.save(case_dir / "heldout_eval_neuron_idx.npy", heldout_subset_stack[0])
        np.savez(case_dir / "heldout_eval_neuron_subsets.npz", neuron_idx_subsets=heldout_subset_stack)
        transfer_metrics = evaluate_heldout(
            best_model,
            target_ctx=target_ctx,
            task=str(task),
            batch_size=int(args.eval_batch_size),
            device=device,
            out_dir=case_dir / "heldout_eval",
            num_classes=num_classes,
            source_fish_ids=source_fish_ids,
            neuron_subsets=heldout_subset_list,
            behavior_normalizer=behavior_normalizer,
        )
        case_summary = {
            "task": str(task),
            "case_name": case_name,
            "heldout_fish_id": int(heldout_fish_id),
            "source_fish_ids": [int(fid) for fid in source_fish_ids],
            "train_sampled_neurons_requested": int(args.sampled_neurons),
            "eval_sampled_neurons": int(heldout_subset_stack.shape[1]),
            "eval_ensemble_subsets": int(heldout_subset_stack.shape[0]),
            "behavior_target_normalization": behavior_normalizer_to_dict(behavior_normalizer),
            "train_split": split_summary,
            "heldout_eval": transfer_metrics,
        }
        save_json(case_dir / "case_summary.json", case_summary)
        case_summaries.append(case_summary)
    if str(task) == "behavior":
        metric_key = "test_corr"
        metric_values = [float(case["heldout_eval"]["test_corr"]) for case in case_summaries]
    else:
        metric_key = "bal_acc"
        metric_values = [float(case["heldout_eval"]["bal_acc"]) for case in case_summaries]
    summary = {
        "task": str(task),
        "metric_key": metric_key,
        "num_cases": int(len(case_summaries)),
        "metric_mean": safe_mean(metric_values),
        "metric_std": safe_std(metric_values),
        "cases": case_summaries,
    }
    save_json(task_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    pack_root = Path(__file__).resolve().parent
    default_data_root = infer_default_data_root(Path(__file__))
    p = argparse.ArgumentParser(
        description=(
            "Minimal supervised IdentityGuidedGrouping transfer-decoding runner. "
            "Train on two fish, use the last validation tail from each source fish for early stopping, "
            "and evaluate the best checkpoint on the held-out fish."
        )
    )
    p.add_argument("--exp-name", type=str, default="identity_guided_grouping")
    p.add_argument("--out-root", type=Path, default=pack_root / "results" / "identity_guided_grouping")
    p.add_argument("--task", choices=["behavior", "stimulus", "both"], default="both")
    p.add_argument("--data-root", type=Path, default=default_data_root)
    p.add_argument("--fish-root", type=Path, default=None)
    p.add_argument("--postprocessed-dir", type=Path, default=None)
    p.add_argument("--stimulus-postprocessed-dir", type=Path, default=None)
    p.add_argument("--fish-ids", type=str, default="59,63,71")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")

    p.set_defaults(
        gene_source="count",
        use_gene=True,
        use_pos=True,
        use_atlas=True,
        use_region=True,
        use_activity=True,
        region_feature_mode="brainregions_multihot",
        activity_feature_mode="robust_stats",
        static_normalize_mode="source",
    )
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
    p.add_argument("--no-normalize-neural", dest="normalize_neural", action="store_false")
    p.add_argument("--debug-load", action="store_true")

    p.add_argument(
        "--behavior-target-normalization",
        choices=["none", "source_train_zscore", "source_session_zscore"],
        default="source_session_zscore",
    )
    p.add_argument("--behavior-loss-mode", choices=["huber", "mse"], default="huber")
    p.add_argument("--behavior-huber-delta", type=float, default=1.0)
    p.add_argument("--stimulus-active-threshold", type=float, default=0.5)
    p.add_argument("--stimulus-num-classes", type=int, default=0)
    p.add_argument("--stimulus-class-weight-mode", choices=["balanced", "none"], default="balanced")

    p.add_argument("--sampled-neurons", type=int, default=3000)
    p.add_argument("--eval-neurons", type=int, default=3000)

    p.add_argument("--window-left", type=int, default=5)
    p.add_argument("--window-right", type=int, default=5)
    p.add_argument("--token-mlp-hidden", type=int, default=128)
    p.add_argument("--slot-internal-dim", type=int, default=384)
    p.add_argument("--slot-static-hidden-dim", type=int, default=192)
    p.add_argument("--slot-num-slots", type=int, default=256)
    p.add_argument("--slot-mixer-hidden", type=int, default=64)
    p.add_argument("--slot-assign-temp", type=float, default=0.35)
    p.add_argument("--slot-assign-topk", type=int, default=5)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--eval-ensemble-subsets", type=int, default=21)
    p.add_argument("--max-epochs", type=int, default=180)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip-norm", type=float, default=0.5)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--checkpoint-selection", choices=["best", "last"], default="best")
    p.add_argument("--disable-early-stop", action="store_true")
    p.add_argument("--val-selection-mode", choices=["raw", "ema"], default="ema")
    p.add_argument("--val-selection-ema-alpha", type=float, default=0.5)
    p.add_argument("--selection-min-epoch", type=int, default=2)
    return p.parse_args()
def main() -> None:
    args = finalize_decoding_args(parse_args())
    fish_ids = parse_int_list(args.fish_ids)
    if len(fish_ids) != 3:
        raise ValueError(f"This runner expects exactly 3 fish ids for the 2-train/1-test benchmark, got {fish_ids}")
    if (
        not bool(args.use_gene)
        and not bool(args.use_pos)
        and not bool(args.use_atlas)
        and not bool(getattr(args, "use_region", False))
        and not bool(getattr(args, "use_activity", False))
    ):
        raise ValueError("At least one identity modality must be enabled.")
    if not args.fish_root.exists():
        raise FileNotFoundError(f"fish_root not found: {args.fish_root}")
    if not args.postprocessed_dir.exists():
        raise FileNotFoundError(f"postprocessed_dir not found: {args.postprocessed_dir}")
    if bool(args.use_atlas):
        if not Path(args.atlas_precomputed_dir).exists():
            raise FileNotFoundError(f"atlas_precomputed_dir not found: {args.atlas_precomputed_dir}")

    device = resolve_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / f"{args.exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "run_args.json", vars(args))
    snapshot_clean_pack_entrypoint(
        run_dir=run_dir,
        argv=[str(x) for x in sys.argv],
        entry_script=Path(__file__).resolve(),
    )

    data.set_global_seed(int(args.seed), deterministic=bool(args.deterministic))
    base_contexts: dict[int, BaseFishContext] = {}
    for fish_id in fish_ids:
        base_contexts[int(fish_id)] = load_base_fish_context(int(fish_id), args)

    task_list = ["behavior", "stimulus"] if str(args.task) == "both" else [str(args.task)]
    task_summaries: dict[str, dict] = {}
    for task in task_list:
        task_summaries[str(task)] = run_task(
            task=str(task),
            base_contexts=base_contexts,
            fish_ids=fish_ids,
            args=args,
            run_dir=run_dir,
            device=device,
        )

    root_summary = {
        "exp_name": str(args.exp_name),
        "timestamp": timestamp,
        "device": str(device),
        "model_name": "IdentityGuidedGrouping",
        "fish_ids": [int(fid) for fid in fish_ids],
        "clean_modalities": list(CLEAN_MODEL_MODALITIES),
        "enabled_modalities": list(args.enabled_modalities),
        "checkpoint_selection": str(getattr(args, "checkpoint_selection", "best")),
        "early_stop_enabled": not bool(getattr(args, "disable_early_stop", False)),
        "val_selection_mode": str(getattr(args, "val_selection_mode", "raw")),
        "val_selection_ema_alpha": float(getattr(args, "val_selection_ema_alpha", 0.3)),
        "selection_min_epoch": int(getattr(args, "selection_min_epoch", 1)),
        "tasks": task_summaries,
    }
    save_json(run_dir / "summary.json", root_summary)
    print(f"[done] run_dir={run_dir}")
    for task, summary in task_summaries.items():
        print(
            f"[{task}] {summary['metric_key']}_mean={summary['metric_mean']:.4f} "
            f"+/- {summary['metric_std']:.4f}"
        )


if __name__ == "__main__":
    main()

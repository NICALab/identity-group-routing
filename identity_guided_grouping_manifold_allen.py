#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import identity_guided_grouping_decoding_allen as allen_decoding
import identity_guided_grouping_manifold as manifold
from src import data
from src.io_utils import resolve_device, save_json, to_jsonable
from src.repro import snapshot_clean_pack_entrypoint
from src.transfer_context import CaseFishContext, build_case_contexts, sample_neuron_subset_indices


def parse_optional_int_csv(text: str) -> list[int]:
    return [int(tok.strip()) for tok in str(text).split(",") if tok.strip()]


def set_all_seeds(seed: int, deterministic: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    data.set_global_seed(int(seed), deterministic=bool(deterministic))


def load_allen_base_contexts(
    args: argparse.Namespace,
    *,
    session_paths: dict[int, Path] | None = None,
) -> dict[int, allen_decoding.AllenBaseContext]:
    session_info = (
        sorted((int(session_id), Path(path)) for session_id, path in session_paths.items())
        if session_paths is not None
        else allen_decoding.list_usable_session_paths(args.allen_root, args.dataset_id)
    )
    bundles = {
        int(session_id): allen_decoding.load_session_bundle(npz_path, args)
        for session_id, npz_path in session_info
    }
    task_spec = allen_decoding.AllenTaskSpec(name="manifold", task_kind="behavior")
    base_contexts, _ = allen_decoding.build_base_contexts_for_task(
        bundles,
        args=args,
        task_spec=task_spec,
    )
    return base_contexts


def build_train_heldout_split(
    session_ids: list[int],
    *,
    train_count: int,
    heldout_count: int,
    split_seed: int,
) -> dict[str, list[int]]:
    all_ids = sorted(int(x) for x in session_ids)
    if int(train_count) <= 0 or int(train_count) >= len(all_ids):
        raise ValueError(f"train_session_count must be in [1, {len(all_ids) - 1}], got {train_count}")
    rng = np.random.default_rng(int(split_seed))
    shuffled = np.asarray(all_ids, dtype=np.int64)[np.asarray(rng.permutation(len(all_ids)), dtype=np.int64)]
    train_ids = sorted(int(x) for x in shuffled[: int(train_count)].tolist())
    remaining = [int(x) for x in shuffled[int(train_count) :].tolist()]
    if int(heldout_count) > 0:
        remaining = remaining[: int(heldout_count)]
    heldout_ids = sorted(int(x) for x in remaining)
    unused_ids = sorted(int(x) for x in set(all_ids) - set(train_ids) - set(heldout_ids))
    if not heldout_ids:
        raise ValueError("No held-out sessions selected.")
    return {
        "train_session_ids": train_ids,
        "heldout_session_ids": heldout_ids,
        "unused_session_ids": unused_ids,
    }


def build_all_case_contexts(
    base_contexts: dict[int, allen_decoding.AllenBaseContext],
    *,
    train_session_ids: list[int],
    heldout_session_ids: list[int],
    static_normalize_mode: str,
) -> tuple[dict[int, CaseFishContext], dict[int, CaseFishContext]]:
    if not heldout_session_ids:
        raise ValueError("heldout_session_ids must be non-empty.")
    source_ctxs, first_target = build_case_contexts(
        base_contexts,
        source_fish_ids=[int(x) for x in train_session_ids],
        target_fish_id=int(heldout_session_ids[0]),
        static_normalize_mode=str(static_normalize_mode),
    )
    heldout_ctxs = {int(heldout_session_ids[0]): first_target}
    for heldout_id in heldout_session_ids[1:]:
        _, target_ctx = build_case_contexts(
            base_contexts,
            source_fish_ids=[int(x) for x in train_session_ids],
            target_fish_id=int(heldout_id),
            static_normalize_mode=str(static_normalize_mode),
        )
        heldout_ctxs[int(heldout_id)] = target_ctx
    return source_ctxs, heldout_ctxs


def manifold_valid_time_indices(ctx: CaseFishContext, model: torch.nn.Module) -> np.ndarray:
    idx = manifold.window_valid_time_indices(ctx, model)
    invalid_mask = getattr(ctx.base, "invalid_time_mask_t", None)
    if invalid_mask is None:
        return np.asarray(idx, dtype=np.int64)
    mask = np.asarray(invalid_mask, dtype=bool)
    idx = np.asarray(idx, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < mask.size)]
    return np.asarray(idx[~mask[idx]], dtype=np.int64)


def split_tail_indices_checked(idx: np.ndarray, val_frac: float, session_id: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(idx, dtype=np.int64).reshape(-1)
    if arr.size < 8:
        raise ValueError(f"Session {session_id} has too few valid manifold frames: {arr.size}")
    return manifold.split_tail_indices(arr, float(val_frac))


def save_time_index(out_dir: Path, time_idx: np.ndarray | None) -> None:
    if time_idx is not None:
        np.save(out_dir / "time_idx.npy", np.asarray(time_idx, dtype=np.int64))


def save_embedding(
    embedding: manifold.ManifoldEmbedding,
    *,
    out_dir: Path,
    base_ctx: allen_decoding.AllenBaseContext,
    time_idx: np.ndarray | None = None,
) -> None:
    manifold.save_embedding_artifacts(embedding, out_dir=out_dir, base_ctx=base_ctx)
    save_time_index(out_dir, time_idx)


def heldout_metric_payload(embedding: manifold.ManifoldEmbedding) -> dict[str, float]:
    payload = manifold.embedding_metric_payload(embedding)
    return {str(k): float(v) for k, v in payload.items()}


def summarize_joint_method(
    *,
    method: str,
    label: str,
    source_embeddings: dict[int, manifold.ManifoldEmbedding],
    heldout_embeddings: dict[int, manifold.ManifoldEmbedding],
) -> list[dict[str, Any]]:
    rows = []
    source_ids = [int(x) for x in sorted(source_embeddings)]
    for heldout_id, held_emb in sorted(heldout_embeddings.items()):
        row = {
            "method": str(method),
            "label": str(label),
            "heldout_session_id": int(heldout_id),
            "aggregation": "single_model_to_20_sources",
            "n_source_encoders": 1,
            "n_source_sessions": int(len(source_ids)),
            **heldout_metric_payload(held_emb),
        }
        rows.append(row)
    return rows


def run_identity_guided_grouping(
    *,
    run_dir: Path,
    base_contexts: dict[int, allen_decoding.AllenBaseContext],
    source_ctxs: dict[int, CaseFishContext],
    heldout_ctxs: dict[int, CaseFishContext],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method_dir = run_dir / "identity_guided_grouping"
    source_ids = [int(x) for x in sorted(source_ctxs)]
    heldout_ids = [int(x) for x in sorted(heldout_ctxs)]
    case_contexts = {**source_ctxs, **heldout_ctxs}
    fish_order = source_ids + heldout_ids
    preview_model = manifold.build_manifold_model(next(iter(source_ctxs.values())), latent_dim=int(args.latent_dim), args=args)
    train_index_map = {}
    val_index_map = {}
    for session_id in source_ids:
        valid_idx = manifold_valid_time_indices(source_ctxs[int(session_id)], preview_model)
        tr_idx, va_idx = split_tail_indices_checked(valid_idx, float(args.val_frac), int(session_id))
        train_index_map[int(session_id)] = tr_idx
        val_index_map[int(session_id)] = va_idx
    del preview_model

    model, split_summary, ckpt_paths = manifold.fit_case_model(
        split_dir=method_dir / "train_val",
        source_ctxs=source_ctxs,
        train_index_map=train_index_map,
        val_index_map=val_index_map,
        case_contexts=case_contexts,
        fish_order=fish_order,
        device=device,
        case_seed=int(args.seed) + 1000,
        args=args,
    )
    embeddings = manifold.compute_case_identity_guided_grouping_embeddings(
        model,
        case_contexts=case_contexts,
        fish_order=fish_order,
        args=args,
        case_seed=int(args.seed) + 1000,
        device=device,
    )
    for session_id in fish_order:
        save_embedding(
            embeddings[int(session_id)],
            out_dir=method_dir / f"session_{int(session_id)}",
            base_ctx=base_contexts[int(session_id)],
        )
    source_embeddings = {int(sid): embeddings[int(sid)] for sid in source_ids}
    heldout_embeddings = {int(sid): embeddings[int(sid)] for sid in heldout_ids}
    rows = summarize_joint_method(
        method="identity_guided_grouping",
        label="IdentityGuidedGrouping",
        source_embeddings=source_embeddings,
        heldout_embeddings=heldout_embeddings,
    )
    summary = {
        "method": "identity_guided_grouping",
        "checkpoint_paths": {k: str(v) for k, v in ckpt_paths.items()},
        "train_split": split_summary,
        "artifact_dir": str(method_dir),
    }
    save_json(method_dir / "summary.json", summary)
    return rows, summary



def write_metric_rows(paths_dir: Path, rows: list[dict[str, Any]]) -> None:
    paths_dir.mkdir(parents=True, exist_ok=True)
    json_path = paths_dir / "allen_manifold_metric_rows.json"
    csv_path = paths_dir / "allen_manifold_metric_rows.csv"
    json_path.write_text(json.dumps(to_jsonable(rows), indent=2), encoding="utf-8")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "behavior_key": "running_speed_cm_s", "bin_size_s": 0.5, "normalize_neural": True,
        "use_pos": True, "use_region": True, "use_activity": True,
        "region_prefix_keep": "", "activity_feature_mode": "robust_stats",
        "activity_cache_dir": None, "activity_force_recompute": False, "activity_require_cache": False,
        "static_normalize_mode": "source", "debug_load": False,
        "stimulus_active_threshold": 0.5,
        "latent_dim": 3, "normalize_latent": True, "contrastive_metric": "cosine", "temperature": 1.0,
        "pos_radius": 1, "train_pos_radius_choices": "",
        "train_pos_radius_max": 0, "train_pos_radius_half_life": 25.0,
        "sampled_neurons": 250, "eval_neurons": 250,
        "eval_ensemble_subsets": 21, "window_left": 5, "window_right": 5, "token_mlp_hidden": 128,
        "slot_internal_dim": 256, "slot_static_hidden_dim": 192, "slot_num_slots": 192,
        "slot_mixer_hidden": 64, "slot_assign_temp": 0.35, "slot_assign_topk": 5,
        "batch_size_t": 128, "eval_batch_size_t": 128, "forward_chunk_size_t": 128,
        "snapshot_every": 0, "snapshot_ensemble_subsets": 0, "checkpoint_select_every": 25,
        "selection_window_t": 96, "selection_ensemble_subsets": 3, "selection_weight_continuity": 1.0,
        "selection_weight_subset": 1.0, "selection_weight_relational": 1.0, "selection_min_delta": 1e-4,
        "relational_loss_weight": 0.05, "relational_window_t": 24, "relational_every_steps": 4,
        "max_epochs": 200, "min_epochs": 50, "patience": 15, "disable_early_stop": False,
        "early_stop_min_delta": 1e-4, "val_frac": 0.2, "lr": 1e-3, "weight_decay": 0.0,
        "grad_clip_norm": 0.5, "checkpoint_every": 10,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def parse_args() -> argparse.Namespace:
    pack_root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Allen IdentityGuidedGrouping manifold transfer runner.")
    p.add_argument("--exp-name", type=str, default="allen_identity_guided_grouping_manifold")
    p.add_argument("--out-root", type=Path, default=pack_root / "results" / "allen_manifold")
    p.add_argument("--allen-root", type=Path, default=allen_decoding.DEFAULT_ALLEN_ROOT)
    p.add_argument("--dataset-id", choices=["21", "22"], required=True)
    p.add_argument("--train-session-count", type=int, default=20)
    p.add_argument("--heldout-session-count", type=int, default=6)
    p.add_argument("--source-session-ids", type=str, default="")
    p.add_argument("--split-seed", type=int, default=41)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sampled-neurons", type=int, default=250)
    p.add_argument("--eval-neurons", type=int, default=250)
    p.add_argument("--eval-ensemble-subsets", type=int, default=21)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    args, _ = p.parse_known_args()
    args = _apply_defaults(args)
    args.allen_root = Path(args.allen_root)
    args.dataset_id = allen_decoding.canonicalize_dataset_id(args.dataset_id)
    if not args.allen_root.exists() and not bool(args.dry_run):
        raise FileNotFoundError(f"allen_root not found: {args.allen_root}")
    return args


def main() -> None:
    args = parse_args()
    if bool(args.dry_run) and not Path(args.allen_root).exists():
        payload = {"dataset_id": str(args.dataset_id), "allen_root": str(args.allen_root), "status": "dry_run_no_data"}
        print(json.dumps(to_jsonable(payload), indent=2), flush=True)
        return
    session_info = allen_decoding.list_usable_session_paths(args.allen_root, args.dataset_id)
    session_path_map = {int(session_id): Path(path) for session_id, path in session_info}
    split_spec = build_train_heldout_split(
        sorted(session_path_map),
        train_count=int(args.train_session_count),
        heldout_count=int(args.heldout_session_count),
        split_seed=int(args.split_seed),
    )
    if bool(args.dry_run):
        print(json.dumps(to_jsonable({"usable_session_ids": sorted(session_path_map), "session_split": split_spec}), indent=2), flush=True)
        return
    active_train_session_ids = [int(x) for x in split_spec["train_session_ids"]]
    requested_source_ids = parse_optional_int_csv(str(args.source_session_ids))
    if requested_source_ids:
        active_train_session_ids = sorted(int(x) for x in requested_source_ids)
    selected_session_ids = sorted(int(x) for x in [*active_train_session_ids, *split_spec["heldout_session_ids"]])
    set_all_seeds(int(args.seed), bool(args.deterministic))
    base_contexts = load_allen_base_contexts(
        args,
        session_paths={int(session_id): session_path_map[int(session_id)] for session_id in selected_session_ids},
    )
    device = resolve_device(str(args.device))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / f"{args.exp_name}_dataset_{args.dataset_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "run_args.json", vars(args))
    save_json(run_dir / "session_split.json", {**split_spec, "active_train_session_ids": active_train_session_ids})
    snapshot_clean_pack_entrypoint(run_dir=run_dir, argv=[str(x) for x in sys.argv], entry_script=Path(__file__).resolve())
    source_ctxs, heldout_ctxs = build_all_case_contexts(
        base_contexts,
        train_session_ids=active_train_session_ids,
        heldout_session_ids=split_spec["heldout_session_ids"],
        static_normalize_mode=str(args.static_normalize_mode),
    )
    rows, method_summary = run_identity_guided_grouping(
        run_dir=run_dir,
        base_contexts=base_contexts,
        source_ctxs=source_ctxs,
        heldout_ctxs=heldout_ctxs,
        args=args,
        device=device,
    )
    write_metric_rows(run_dir, rows)
    summary = {
        "exp_name": str(args.exp_name),
        "timestamp": timestamp,
        "device": str(device),
        "dataset_id": str(args.dataset_id),
        "loaded_session_ids": selected_session_ids,
        "session_split": split_spec,
        "method": "IdentityGuidedGrouping",
        "method_summary": method_summary,
        "metric_rows_path": str(run_dir / "allen_manifold_metric_rows.csv"),
    }
    save_json(run_dir / "summary.json", summary)
    print(f"[done] run_dir={run_dir}", flush=True)
    print(json.dumps(to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()

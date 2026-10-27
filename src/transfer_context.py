from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data
from .io_utils import DEFAULT_POSTPROCESSED_DIR


@dataclass
class BaseFishContext:
    fish: data.FishData
    static_raw_nf: np.ndarray | None
    behavior_target_t: np.ndarray
    behavior_target_label: str
    stimulus_st: np.ndarray | None
    stimulus_target_label: str | None
    stimulus_class_t: np.ndarray | None
    stimulus_active_mask_t: np.ndarray | None


@dataclass
class CaseFishContext:
    base: BaseFishContext
    static_nf: np.ndarray | None
    static_norm_mean_f: np.ndarray | None = None
    static_norm_std_f: np.ndarray | None = None


@dataclass
class BehaviorNormalizer:
    mode: str
    mean: float
    std: float
    per_fish: dict[int, dict[str, float]] | None = None


def behavior_normalizer_to_dict(normalizer: BehaviorNormalizer | None) -> dict | None:
    if normalizer is None:
        return None
    payload = {"mode": str(normalizer.mode), "mean": float(normalizer.mean), "std": float(normalizer.std)}
    if normalizer.per_fish is not None:
        payload["per_fish"] = {
            str(int(fish_id)): {
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
            }
            for fish_id, stats in normalizer.per_fish.items()
        }
    return payload


def build_behavior_target(behavior_t: np.ndarray, args) -> tuple[np.ndarray, str]:
    return data.build_decode_behavior_target(
        behavior_t=behavior_t,
    )


def load_base_fish_context(fish_id: int, args) -> BaseFishContext:
    fish = data.load_fish_data(
        fish_root=args.fish_root,
        fish_id=int(fish_id),
        postprocessed_dir=args.postprocessed_dir,
        gene_source=str(args.gene_source) if bool(args.use_gene) else "none",
        normalize_neural=bool(args.normalize_neural),
        debug=bool(args.debug_load),
        region_feature_mode=(
            str(getattr(args, "region_feature_mode", "none"))
            if bool(getattr(args, "use_region", False))
            else "none"
        ),
        atlas_feature_mode="precomputed_hcr_npy" if bool(args.use_atlas) else "none",
        atlas_precomputed_dir=Path(getattr(args, "atlas_precomputed_dir", data.DEFAULT_PRECOMPUTED_ATLAS_HCR_DIR)),
        activity_cache_dir=args.activity_cache_dir,
        activity_force_recompute=bool(getattr(args, "activity_force_recompute", False)),
        activity_require_cache=bool(getattr(args, "activity_require_cache", False)),
        activity_feature_mode=(
            str(getattr(args, "activity_feature_mode", "none"))
            if bool(getattr(args, "use_activity", False))
            else "none"
        ),
    )
    if int(fish.neural_nt.shape[0]) <= 0:
        raise ValueError(f"Fish {fish_id} has zero neurons after loading/filtering.")

    static_raw_nf = data.build_static_features(
        fish.genes_nf,
        fish.coords_nf,
        use_gene=bool(args.use_gene),
        use_pos=bool(args.use_pos),
        atlas_nf=fish.regions_nf,
        use_atlas=bool(args.use_atlas),
        region_nf=fish.region_membership_nf,
        use_region=bool(getattr(args, "use_region", False)),
        activity_nf=fish.activity_nf,
        use_activity=bool(getattr(args, "use_activity", False)),
    )
    behavior_target_t, behavior_target_label = build_behavior_target(fish.behavior_t, args)
    stimulus_st, stimulus_target_label = load_stimulus_matrix_from_args(
        args,
        int(fish.neural_nt.shape[1]),
    )
    stimulus_class_t, stimulus_active_mask_t = data.build_stimulus_class_target(
        stimulus_st,
        active_threshold=float(args.stimulus_active_threshold),
    )
    return BaseFishContext(
        fish=fish,
        static_raw_nf=static_raw_nf,
        behavior_target_t=np.asarray(behavior_target_t, dtype=np.float32),
        behavior_target_label=str(behavior_target_label),
        stimulus_st=stimulus_st,
        stimulus_target_label=stimulus_target_label,
        stimulus_class_t=stimulus_class_t,
        stimulus_active_mask_t=stimulus_active_mask_t,
    )


def _stimulus_search_dirs(args) -> list[Path]:
    raw_dirs = []
    explicit = getattr(args, "stimulus_postprocessed_dir", None)
    if explicit is not None:
        if isinstance(explicit, (list, tuple)):
            raw_dirs.extend(explicit)
        else:
            raw_dirs.append(explicit)
    raw_dirs.extend([getattr(args, "postprocessed_dir", None), DEFAULT_POSTPROCESSED_DIR])

    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_dirs:
        if raw is None:
            continue
        path = Path(raw)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def load_stimulus_matrix_from_args(args, t_len: int) -> tuple[np.ndarray | None, str | None]:
    for path in _stimulus_search_dirs(args):
        stimulus_st, stimulus_target_label = data.load_stimulus_matrix(
            path,
            int(t_len),
        )
        if stimulus_st is not None:
            return stimulus_st, f"{stimulus_target_label}@{path}"
    return None, None


def compute_static_norm_stats(train_sf: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if train_sf is None:
        return None, None
    train_arr = np.asarray(train_sf, dtype=np.float32)
    mean_f = train_arr.mean(axis=0, keepdims=True)
    std_f = train_arr.std(axis=0, keepdims=True)
    std_f = np.where(std_f < 1e-6, 1.0, std_f).astype(np.float32, copy=False)
    return mean_f.astype(np.float32, copy=False), std_f


def apply_static_norm_stats(
    static_raw_nf: np.ndarray | None,
    *,
    mean_f: np.ndarray | None,
    std_f: np.ndarray | None,
) -> np.ndarray | None:
    if static_raw_nf is None:
        return None
    raw = np.asarray(static_raw_nf, dtype=np.float32)
    if mean_f is None or std_f is None:
        return raw.copy()
    return ((raw - mean_f) / std_f).astype(np.float32, copy=False)


def build_case_contexts(
    base_contexts: dict[int, BaseFishContext],
    *,
    source_fish_ids: list[int],
    target_fish_id: int,
    static_normalize_mode: str,
) -> tuple[dict[int, CaseFishContext], CaseFishContext]:
    source_ref_blocks = [
        np.asarray(base_contexts[int(fid)].static_raw_nf, dtype=np.float32)
        for fid in source_fish_ids
        if base_contexts[int(fid)].static_raw_nf is not None
    ]
    source_reference_nf = np.concatenate(source_ref_blocks, axis=0) if source_ref_blocks else None
    mode_key = str(static_normalize_mode).strip().lower()
    shared_mean_f, shared_std_f = (
        compute_static_norm_stats(source_reference_nf) if mode_key == "source" else (None, None)
    )
    case_contexts: dict[int, CaseFishContext] = {}
    for fish_id, base_ctx in base_contexts.items():
        if mode_key == "source":
            mean_f, std_f = shared_mean_f, shared_std_f
        elif mode_key == "self":
            mean_f, std_f = compute_static_norm_stats(base_ctx.static_raw_nf)
        elif mode_key == "none":
            mean_f, std_f = None, None
        else:
            raise ValueError(f"Unsupported static_normalize_mode: {static_normalize_mode}")
        case_contexts[int(fish_id)] = CaseFishContext(
            base=base_ctx,
            static_nf=apply_static_norm_stats(base_ctx.static_raw_nf, mean_f=mean_f, std_f=std_f),
            static_norm_mean_f=None if mean_f is None else np.asarray(mean_f, dtype=np.float32).copy(),
            static_norm_std_f=None if std_f is None else np.asarray(std_f, dtype=np.float32).copy(),
        )
    source_ctxs = {int(fid): case_contexts[int(fid)] for fid in source_fish_ids}
    target_ctx = case_contexts[int(target_fish_id)]
    return source_ctxs, target_ctx


def resolve_neuron_subset_size(n_total: int, requested: int) -> int:
    req = int(requested)
    if req <= 0:
        return int(n_total)
    return int(min(int(n_total), req))


def sample_neuron_subset_indices(
    ctx: CaseFishContext,
    *,
    requested: int,
    seed: int,
) -> np.ndarray:
    n_total = int(ctx.base.fish.neural_nt.shape[0])
    n_sample = resolve_neuron_subset_size(n_total, int(requested))
    if n_sample >= n_total:
        return np.arange(n_total, dtype=np.int64)
    subset_idx = data.sample_neuron_indices(
        np.random.default_rng(int(seed)),
        n_total=n_total,
        n_sample=int(n_sample),
    )
    return np.sort(np.asarray(subset_idx, dtype=np.int64))


def _behavior_stats_for_fish(
    normalizer: BehaviorNormalizer,
    fish_id: int | None,
) -> tuple[float, float]:
    if normalizer.per_fish is not None and fish_id is not None and int(fish_id) in normalizer.per_fish:
        stats = normalizer.per_fish[int(fish_id)]
        return float(stats["mean"]), float(stats["std"])
    return float(normalizer.mean), float(normalizer.std)


def normalize_behavior_array(
    y: np.ndarray,
    normalizer: BehaviorNormalizer | None,
    *,
    fish_id: int | None = None,
) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if normalizer is None:
        return arr.copy()
    mean, std = _behavior_stats_for_fish(normalizer, fish_id)
    return ((arr - mean) / std).astype(np.float32, copy=False)


def denormalize_behavior_array(
    y: np.ndarray,
    normalizer: BehaviorNormalizer | None,
    *,
    fish_id: int | None = None,
) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if normalizer is None:
        return arr.copy()
    mean, std = _behavior_stats_for_fish(normalizer, fish_id)
    return (arr * std + mean).astype(np.float32, copy=False)


def build_fixed_subset_map(
    ctx_map: dict[int, CaseFishContext],
    *,
    requested: int,
    seed: int,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for offset, fish_id in enumerate(sorted(ctx_map)):
        out[int(fish_id)] = sample_neuron_subset_indices(
            ctx_map[int(fish_id)],
            requested=int(requested),
            seed=int(seed) + 100 * int(fish_id) + int(offset),
        )
    return out


def build_behavior_normalizer(
    source_ctxs: dict[int, CaseFishContext],
    train_index_map: dict[int, np.ndarray],
    *,
    mode: str,
) -> BehaviorNormalizer | None:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return None
    if mode_key not in {"source_train_zscore", "source_session_zscore"}:
        raise ValueError(f"Unsupported behavior_target_normalization: {mode}")
    target_blocks = []
    per_fish: dict[int, dict[str, float]] = {}
    for fish_id, idx in train_index_map.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        if idx_arr.size <= 0:
            continue
        y_fish = np.asarray(source_ctxs[int(fish_id)].base.behavior_target_t[idx_arr], dtype=np.float32)
        target_blocks.append(y_fish)
        per_fish[int(fish_id)] = {
            "mean": float(y_fish.mean()),
            "std": max(float(y_fish.std()), 1e-6),
        }
    if not target_blocks:
        raise ValueError("Cannot build behavior normalizer without source-train targets.")
    y = np.concatenate(target_blocks, axis=0).astype(np.float32, copy=False)
    std = max(float(y.std()), 1e-6)
    return BehaviorNormalizer(
        mode=mode_key,
        mean=float(y.mean()),
        std=std,
        per_fish=per_fish if mode_key == "source_session_zscore" else None,
    )


def build_subset_ensemble(
    ctx: CaseFishContext,
    *,
    requested: int,
    num_subsets: int,
    seed: int,
) -> list[np.ndarray]:
    subsets: list[np.ndarray] = []
    for ens_idx in range(max(1, int(num_subsets))):
        subsets.append(
            sample_neuron_subset_indices(
                ctx,
                requested=int(requested),
                seed=int(seed) + 1009 * int(ens_idx),
            )
        )
    return subsets


def get_task_indices(ctx: CaseFishContext, task: str) -> np.ndarray:
    if str(task) == "behavior":
        return np.arange(int(ctx.base.fish.neural_nt.shape[1]), dtype=np.int64)
    if ctx.base.stimulus_class_t is None or ctx.base.stimulus_active_mask_t is None:
        raise ValueError(f"Fish {ctx.base.fish.fish_id} is missing stimulus targets.")
    active_idx = np.flatnonzero(np.asarray(ctx.base.stimulus_active_mask_t, dtype=bool))
    if active_idx.size <= 0:
        raise ValueError(f"Fish {ctx.base.fish.fish_id} has no active stimulus frames.")
    return np.asarray(active_idx, dtype=np.int64)


def get_targets_for_indices(
    ctx: CaseFishContext,
    task: str,
    indices: np.ndarray,
    *,
    behavior_normalizer: BehaviorNormalizer | None = None,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if str(task) == "behavior":
        return normalize_behavior_array(
            ctx.base.behavior_target_t[idx],
            behavior_normalizer,
            fish_id=int(ctx.base.fish.fish_id),
        )
    if ctx.base.stimulus_class_t is None:
        raise ValueError(f"Fish {ctx.base.fish.fish_id} is missing stimulus targets.")
    return np.asarray(ctx.base.stimulus_class_t[idx], dtype=np.int64)


def build_tail_split(
    source_ctxs: dict[int, CaseFishContext],
    *,
    task: str,
    val_frac: float,
) -> dict[str, dict[int, np.ndarray]]:
    frac = float(val_frac)
    if not (0.0 < frac < 0.5):
        raise ValueError(f"val_frac must be in (0, 0.5), got {val_frac}")
    train_map: dict[int, np.ndarray] = {}
    val_map: dict[int, np.ndarray] = {}
    for fid, ctx in source_ctxs.items():
        t_len = int(ctx.base.fish.neural_nt.shape[1])
        split_t = int(round((1.0 - frac) * t_len))
        split_t = max(1, min(split_t, t_len - 1))
        valid_idx = np.asarray(get_task_indices(ctx, task), dtype=np.int64)
        if valid_idx.size < 8:
            raise ValueError(f"Fish {fid} has too few valid samples for a train/val split: {valid_idx.size}")
        train_idx = np.asarray(valid_idx[valid_idx < split_t], dtype=np.int64)
        val_idx = np.asarray(valid_idx[valid_idx >= split_t], dtype=np.int64)
        if train_idx.size <= 0 or val_idx.size <= 0:
            n_val = int(round(valid_idx.size * frac))
            n_val = max(1, min(n_val, valid_idx.size - 4))
            train_idx = np.asarray(valid_idx[:-n_val], dtype=np.int64)
            val_idx = np.asarray(valid_idx[-n_val:], dtype=np.int64)
        if train_idx.size <= 0 or val_idx.size <= 0:
            raise ValueError(f"Tail split failed for fish {fid}: train={train_idx.size}, val={val_idx.size}")
        train_map[int(fid)] = train_idx
        val_map[int(fid)] = val_idx
    return {"train": train_map, "val": val_map}


def build_batches(
    index_map: dict[int, np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[tuple[int, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    batches: list[tuple[int, np.ndarray]] = []
    for fish_id in sorted(index_map):
        idx = np.asarray(index_map[int(fish_id)], dtype=np.int64)
        if idx.size <= 0:
            continue
        idx_work = rng.permutation(idx) if shuffle else idx.copy()
        for start in range(0, int(idx_work.size), int(batch_size)):
            batches.append((int(fish_id), np.asarray(idx_work[start:start + int(batch_size)], dtype=np.int64)))
    if shuffle and batches:
        order = rng.permutation(len(batches))
        batches = [batches[int(i)] for i in order]
    return batches


def infer_num_classes(base_contexts: dict[int, BaseFishContext], override: int) -> int:
    if int(override) > 0:
        return int(override)
    num_classes = 0
    for ctx in base_contexts.values():
        if ctx.stimulus_st is not None:
            num_classes = max(num_classes, int(ctx.stimulus_st.shape[0]))
        if ctx.stimulus_class_t is not None and ctx.stimulus_class_t.size > 0:
            num_classes = max(num_classes, int(np.max(ctx.stimulus_class_t)) + 1)
    if num_classes <= 1:
        raise ValueError("Could not infer a valid number of stimulus classes.")
    return int(num_classes)

#!/usr/bin/env python3
"""
Minimal shared data/runtime utilities for the clean submission pack.

This module intentionally keeps only the pieces used by:
- `identity_guided_grouping_decoding.py`
- `identity_guided_grouping_manifold.py`

It keeps only the data loading and batching pieces used by the paper-facing
reproduction flow.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

def as_float32_clean(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def zscore_neuron_time(neural_nt: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = neural_nt.mean(axis=1, keepdims=True)
    s = neural_nt.std(axis=1, keepdims=True)
    s = np.where(s < eps, 1.0, s)
    return ((neural_nt - m) / s).astype(np.float32)


def set_global_seed(seed: int, deterministic: bool) -> None:
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True


@dataclass
class FishData:
    fish_id: int
    neural_nt: np.ndarray
    behavior_t: np.ndarray
    genes_nf: np.ndarray | None
    coords_nf: np.ndarray | None
    regions_nf: np.ndarray | None
    region_membership_nf: np.ndarray | None
    activity_nf: np.ndarray | None
    global_neuron_idx_n: np.ndarray
    region_labels: list[str] | None = None
    region_membership_labels: list[str] | None = None


DEFAULT_PRECOMPUTED_ATLAS_HCR_DIR = Path("data/precomputed_atlas_hcr")
_PRECOMPUTED_ATLAS_INDEX_CACHE: dict[str, dict[str, object]] = {}
PRECOMPUTED_ATLAS_CACHE_METHOD = "precomputed_multimodal_atlas_hcr_v1"
ACTIVITY_FEATURE_CACHE_METHOD = "robust_stats_v1"
ACTIVITY_FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "q25",
    "q50",
    "q75",
    "frac_pos",
    "energy",
    "diff_std",
]


def _load_precomputed_atlas_cache_index(cache_dir: Path) -> dict[str, object]:
    cache_dir = Path(cache_dir)
    cache_key = str(cache_dir.resolve())
    cached = _PRECOMPUTED_ATLAS_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    candidates = sorted(cache_dir.glob("warp_mapzebrain_hcr_*_genes_df_All.npy"))
    if len(candidates) <= 0:
        raise FileNotFoundError(
            f"No full-row precomputed atlas npy found under {cache_dir}. "
            "Expected a file like warp_mapzebrain_hcr_*_genes_df_All.npy"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple full-row precomputed atlas npy files found under {cache_dir}: "
            f"{[p.name for p in candidates]}"
        )
    features_npy = candidates[0]
    suffix = "_genes_df_All.npy"
    stem = features_npy.name[:-len(suffix)] if features_npy.name.endswith(suffix) else features_npy.stem
    markers_csv = cache_dir / f"{stem}_selected_markers.csv"
    meta_json = cache_dir / f"{stem}_metadata.json"
    if not markers_csv.exists():
        raise FileNotFoundError(f"Precomputed atlas selected_markers.csv not found: {markers_csv}")

    marker_names: list[str] = []
    with markers_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            marker_names.append(str(row["marker_name"]))
    if len(marker_names) <= 0:
        raise ValueError(f"No marker names found in {markers_csv}")

    meta: dict[str, object] = {}
    if meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))

    atlas_shape = tuple(int(v) for v in np.load(features_npy, mmap_mode="r").shape)
    out = {
        "features_npy": features_npy,
        "marker_names": marker_names,
        "atlas_shape": atlas_shape,
        "metadata": meta,
    }
    _PRECOMPUTED_ATLAS_INDEX_CACHE[cache_key] = out
    return out


def load_precomputed_multimodal_atlas_marker_matrix(
    *,
    fish_id: int,
    neuron_idx: np.ndarray,
    cache_dir: Path = DEFAULT_PRECOMPUTED_ATLAS_HCR_DIR,
) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    idx = np.asarray(neuron_idx, dtype=np.int64).reshape(-1)
    cache = _load_precomputed_atlas_cache_index(Path(cache_dir))
    marker_names = [str(v) for v in cache["marker_names"]]
    atlas_shape = tuple(int(v) for v in cache["atlas_shape"])
    if idx.size > 0:
        idx_min = int(idx.min())
        idx_max = int(idx.max())
        if idx_min < 0 or idx_max >= int(atlas_shape[0]):
            raise IndexError(
                f"neuron_idx out of bounds for precomputed atlas matrix: "
                f"min={idx_min} max={idx_max} rows={atlas_shape[0]}"
            )
    atlas_all = np.load(cache["features_npy"], mmap_mode="r")
    atlas_nf = np.asarray(atlas_all[idx], dtype=np.float32)
    finite_mask = np.all(np.isfinite(atlas_nf), axis=1)
    meta = {
        **dict(cache.get("metadata", {})),
        "atlas_compute_method": PRECOMPUTED_ATLAS_CACHE_METHOD,
        "precomputed_atlas_dir": str(Path(cache_dir)),
        "num_rows_requested": int(idx.shape[0]),
        "num_rows_finite_after_subset": int(finite_mask.sum()),
        "num_rows_nan_after_subset": int((~finite_mask).sum()),
        "fish_id": int(fish_id),
    }
    return atlas_nf, marker_names, finite_mask, meta


def canonicalize_region_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def infer_region_names_path(postprocessed_dir: Path) -> Path:
    postprocessed_dir = Path(postprocessed_dir)
    dataset_dir = postprocessed_dir.parent
    root_dir = dataset_dir.parent
    dataset_base = re.sub(r"\s*\(\d+\)\s*$", "", str(dataset_dir.name)).strip()
    direct_candidates = [
        postprocessed_dir / "region_names.npy",
        dataset_dir / "region_names.npy",
        dataset_dir / "Fish1" / "region_names.npy",
        root_dir / "Fish1" / "region_names.npy",
        root_dir / dataset_base / "Fish1" / "region_names.npy",
    ]
    for path in direct_candidates:
        if path.exists():
            return path
    for root in [dataset_dir, root_dir]:
        if root.exists():
            try:
                return next(root.rglob("region_names.npy"))
            except StopIteration:
                pass
    raise FileNotFoundError(f"Could not find region_names.npy near {postprocessed_dir}")


def load_full_region_membership(
    postprocessed_dir: Path,
    global_neuron_idx_n: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    reg_path = Path(postprocessed_dir) / "BrainRegions_All.npy"
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing fine region matrix: {reg_path}")
    region_names_path = infer_region_names_path(postprocessed_dir)
    region_names = [str(x) for x in np.load(region_names_path, allow_pickle=True).tolist()]
    region_matrix_all = np.asarray(np.load(reg_path, mmap_mode="r"), dtype=bool)
    region_matrix_nr = region_matrix_all[np.asarray(global_neuron_idx_n, dtype=np.int64)]
    if region_matrix_nr.ndim != 2:
        raise ValueError(f"Expected BrainRegions_All subset to be 2D, got {region_matrix_nr.shape}")
    if region_matrix_nr.shape[1] != len(region_names):
        raise ValueError(
            f"Region matrix/name mismatch: matrix has {region_matrix_nr.shape[1]} columns, "
            f"names has {len(region_names)} entries"
        )
    return region_matrix_nr, region_names


def validate_neuron_indices_match_fish(
    postprocessed_dir: Path,
    *,
    neuron_idx: np.ndarray,
    fish_id: int,
) -> None:
    fish_id_path = Path(postprocessed_dir) / "fish_id.npy"
    if not fish_id_path.exists():
        return
    fish_ids_all = np.asarray(np.load(fish_id_path, mmap_mode="r"))
    fish_ids_sub = np.asarray(fish_ids_all[np.asarray(neuron_idx, dtype=np.int64)]).reshape(-1)
    if fish_ids_sub.size <= 0:
        return
    if not np.all(np.isfinite(fish_ids_sub)):
        raise ValueError(f"fish_id.npy contains non-finite entries for fish {int(fish_id)} subset.")
    fish_ids_sub = np.asarray(np.rint(fish_ids_sub), dtype=np.int64)
    uniq = np.unique(fish_ids_sub)
    if uniq.size != 1 or int(uniq[0]) != int(fish_id):
        raise ValueError(
            f"Global-neuron subset mismatch for fish {int(fish_id)}: fish_id.npy subset contains {uniq.tolist()}"
        )


def load_region_features(
    postprocessed_dir: Path,
    neuron_idx: np.ndarray,
    region_feature_mode: str = "brain_reg_onehot",
) -> tuple[np.ndarray | None, np.ndarray | None, list[str] | None]:
    mode = str(region_feature_mode).strip().lower()
    if mode in {"none", "", "off"}:
        return None, None, None
    if mode == "brain_reg_onehot":
        reg_path = Path(postprocessed_dir) / "Brain_reg.npy"
        if not reg_path.exists():
            raise FileNotFoundError(f"Missing region id file: {reg_path}")
        region_all = np.asarray(np.load(reg_path, mmap_mode="r"), dtype=np.int64)
        region_ids_n = np.asarray(region_all[np.asarray(neuron_idx, dtype=np.int64)], dtype=np.int64)
        if region_ids_n.size == 0:
            return np.zeros((0, 0), dtype=np.float32), region_ids_n, []
        if np.any(region_ids_n < 0):
            raise ValueError(f"Brain_reg.npy contains negative region ids; min={int(region_ids_n.min())}")
        num_regions = int(region_all.max()) + 1
        regions_nf = np.eye(num_regions, dtype=np.float32)[region_ids_n]
        region_labels = [f"brain_reg_{i}" for i in range(num_regions)]
        return regions_nf, region_ids_n, region_labels
    if mode == "brainregions_multihot":
        regions_bool_nf, region_names = load_full_region_membership(
            postprocessed_dir=postprocessed_dir,
            global_neuron_idx_n=np.asarray(neuron_idx, dtype=np.int64),
        )
        return np.asarray(regions_bool_nf, dtype=np.float32), None, [str(x) for x in region_names]
    raise ValueError(
        f"Unsupported region_feature_mode: {region_feature_mode}. "
        "Expected one of {'none','brain_reg_onehot','brainregions_multihot'}."
    )


def build_activity_features(
    neural_nt: np.ndarray,
    *,
    mode: str = "robust_stats",
    chunk_size: int = 4096,
) -> np.ndarray | None:
    mode_key = str(mode).strip().lower()
    if mode_key in {"none", "", "off"}:
        return None
    if mode_key != "robust_stats":
        raise ValueError(f"Unsupported activity_feature_mode: {mode}")
    x = np.asarray(neural_nt, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected neural_nt shape (N, T), got {x.shape}")
    n_rows, t_len = int(x.shape[0]), int(x.shape[1])
    out = np.empty((n_rows, 10), dtype=np.float32)
    step = max(1, int(chunk_size))
    for start in range(0, n_rows, step):
        end = min(start + step, n_rows)
        block = as_float32_clean(x[start:end])
        q25, q50, q75 = np.quantile(block, [0.25, 0.5, 0.75], axis=1)
        diff_std = (
            np.std(block[:, 1:] - block[:, :-1], axis=1, dtype=np.float32)
            if t_len > 1
            else np.zeros((block.shape[0],), dtype=np.float32)
        )
        out[start:end] = np.stack(
            [
                block.mean(axis=1, dtype=np.float32),
                block.std(axis=1, dtype=np.float32),
                block.min(axis=1),
                block.max(axis=1),
                np.asarray(q25, dtype=np.float32),
                np.asarray(q50, dtype=np.float32),
                np.asarray(q75, dtype=np.float32),
                (block > 0.0).mean(axis=1, dtype=np.float32),
                np.mean(block * block, axis=1, dtype=np.float32),
                np.asarray(diff_std, dtype=np.float32),
            ],
            axis=1,
        )
    return out.astype(np.float32, copy=False)


def load_or_compute_activity_feature_matrix(
    neural_nt: np.ndarray,
    *,
    fish_id: int,
    cache_dir: Path,
    mode: str = "robust_stats",
    force_recompute: bool = False,
    require_cache: bool = False,
    chunk_size: int = 4096,
) -> tuple[np.ndarray | None, dict]:
    mode_key = str(mode).strip().lower()
    if mode_key in {"none", "", "off"}:
        return None, {"activity_feature_mode": "none", "activity_compute_method": "none", "activity_feature_names": []}
    if mode_key != "robust_stats":
        raise ValueError(f"Unsupported activity_feature_mode: {mode}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_stem = f"activity_{mode_key}_fish{int(fish_id)}_full"
    cache_npy = cache_dir / f"{cache_stem}.npy"
    cache_json = cache_dir / f"{cache_stem}.json"

    def _load_cache():
        activity_nf = np.asarray(np.load(cache_npy, mmap_mode="r"), dtype=np.float32)
        meta = json.loads(cache_json.read_text(encoding="utf-8"))
        if str(meta.get("activity_compute_method", "")) != ACTIVITY_FEATURE_CACHE_METHOD:
            return None
        if str(meta.get("activity_feature_mode", "")).strip().lower() != mode_key:
            return None
        if int(activity_nf.shape[0]) != int(np.asarray(neural_nt).shape[0]):
            return None
        return activity_nf, meta

    if cache_npy.exists() and cache_json.exists() and not force_recompute:
        loaded = _load_cache()
        if loaded is not None:
            return loaded

    if require_cache:
        raise FileNotFoundError(
            "Required activity cache is missing or stale. "
            f"Expected cache under {cache_dir} for fish {int(fish_id)} mode={mode_key}."
        )

    activity_nf = build_activity_features(np.asarray(neural_nt, dtype=np.float32), mode=mode_key, chunk_size=int(chunk_size))
    meta = {
        "fish_id": int(fish_id),
        "activity_feature_mode": mode_key,
        "activity_compute_method": ACTIVITY_FEATURE_CACHE_METHOD,
        "activity_feature_names": list(ACTIVITY_FEATURE_NAMES),
        "num_neurons": int(np.asarray(neural_nt).shape[0]),
        "num_timepoints": int(np.asarray(neural_nt).shape[1]),
        "chunk_size": int(chunk_size),
    }
    np.save(cache_npy, np.asarray(activity_nf, dtype=np.float32))
    cache_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return np.asarray(activity_nf, dtype=np.float32), meta


def load_fish_data(
    fish_root: Path,
    fish_id: int,
    postprocessed_dir: Path,
    gene_source: str,
    normalize_neural: bool,
    debug: bool = False,
    region_feature_mode: str = "brain_reg_onehot",
    atlas_feature_mode: str = "none",
    atlas_precomputed_dir: Path = DEFAULT_PRECOMPUTED_ATLAS_HCR_DIR,
    activity_cache_dir: Path | None = None,
    activity_force_recompute: bool = False,
    activity_require_cache: bool = False,
    activity_feature_mode: str = "none",
) -> FishData:
    fish_dir = Path(fish_root) / f"fish_{fish_id}"

    n_path = fish_dir / "dff_traceAll_sampled.npy"
    if not n_path.exists():
        raise FileNotFoundError(f"Missing neural file: {n_path}")
    neural_nt_raw = np.asarray(np.load(n_path, mmap_mode="r"), dtype=np.float32)

    b_path = Path(postprocessed_dir) / f"swim_power_good{fish_id}.npy"
    if not b_path.exists():
        raise FileNotFoundError(f"Missing behavior file: {b_path}")
    behavior_t = as_float32_clean(np.asarray(np.load(b_path, mmap_mode="r")))
    if behavior_t.shape[0] != neural_nt_raw.shape[1]:
        raise ValueError(
            f"Behavior length mismatch: neural T={neural_nt_raw.shape[1]}, behavior T={behavior_t.shape[0]}"
        )

    genes_nf = None
    if gene_source != "none":
        g_name = "BinaryGenes_All_sampled.npy" if gene_source == "binary" else "genes_df_All_sampled.npy"
        g_path = fish_dir / g_name
        if not g_path.exists():
            raise FileNotFoundError(f"Missing gene file: {g_path}")
        genes_nf = np.asarray(np.load(g_path, mmap_mode="r"), dtype=np.float32)

    idx_path = fish_dir / "neuron_indices.npy"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing neuron indices: {idx_path}")
    neuron_idx = np.asarray(np.load(idx_path, mmap_mode="r"), dtype=np.int64)
    validate_neuron_indices_match_fish(postprocessed_dir=Path(postprocessed_dir), neuron_idx=neuron_idx, fish_id=int(fish_id))

    c_all_path = Path(postprocessed_dir) / "Coords_All.npy"
    if not c_all_path.exists():
        raise FileNotFoundError(f"Missing coords file: {c_all_path}")
    coords_all = np.asarray(np.load(c_all_path, mmap_mode="r"), dtype=np.float32)
    coords_nf = coords_all[neuron_idx]

    region_membership_nf, _, region_membership_labels = load_region_features(
        postprocessed_dir=Path(postprocessed_dir),
        neuron_idx=neuron_idx,
        region_feature_mode=region_feature_mode,
    )

    atlas_mode_key = str(atlas_feature_mode).strip().lower()
    atlas_nf = None
    atlas_labels = None
    if atlas_mode_key not in {"none", "precomputed_hcr_npy"}:
        raise ValueError(
            f"Unsupported atlas_feature_mode in submission_pack_clean: {atlas_feature_mode}. "
            "Only {'none','precomputed_hcr_npy'} are kept in the clean pack."
        )

    activity_nf = None
    if str(activity_feature_mode).strip().lower() not in {"none", "", "off"}:
        activity_cache_root = Path(activity_cache_dir) if activity_cache_dir is not None else (fish_dir / "activity_cache")
        activity_nf, activity_meta = load_or_compute_activity_feature_matrix(
            neural_nt_raw,
            fish_id=int(fish_id),
            cache_dir=activity_cache_root,
            mode=str(activity_feature_mode),
            force_recompute=bool(activity_force_recompute),
            require_cache=bool(activity_require_cache),
        )
        if debug:
            print(
                f"[LoadDebug] fish={fish_id} activity_cache={None if activity_nf is None else activity_nf.shape} "
                f"mode={activity_meta.get('activity_feature_mode', 'none')}",
                flush=True,
            )

    keep_mask = np.ones(neural_nt_raw.shape[0], dtype=bool)
    keep_mask &= np.all(np.isfinite(neural_nt_raw), axis=1)
    keep_mask &= np.all(np.isfinite(coords_nf), axis=1)
    if genes_nf is not None:
        keep_mask &= np.all(np.isfinite(genes_nf), axis=1)
    if region_membership_nf is not None:
        keep_mask &= np.all(np.isfinite(region_membership_nf), axis=1)

    if atlas_mode_key == "precomputed_hcr_npy":
        atlas_nf, atlas_labels, atlas_finite_mask, atlas_meta = load_precomputed_multimodal_atlas_marker_matrix(
            fish_id=int(fish_id),
            neuron_idx=neuron_idx,
            cache_dir=Path(atlas_precomputed_dir),
        )
        keep_mask &= np.asarray(atlas_finite_mask, dtype=bool)
        if debug:
            print(
                f"[LoadDebug] fish={fish_id} precomputed_atlas={atlas_nf.shape} "
                f"markers={len(atlas_labels)} finite={int(atlas_meta['num_rows_finite_after_subset'])}/{int(atlas_meta['num_rows_requested'])}"
            )

    if int((~keep_mask).sum()) > 0:
        neural_nt_raw = neural_nt_raw[keep_mask]
        neuron_idx = neuron_idx[keep_mask]
        coords_nf = coords_nf[keep_mask]
        if genes_nf is not None:
            genes_nf = genes_nf[keep_mask]
        if region_membership_nf is not None:
            region_membership_nf = np.asarray(region_membership_nf[keep_mask], dtype=np.float32)
        if activity_nf is not None:
            activity_nf = np.asarray(activity_nf[keep_mask], dtype=np.float32)
        if atlas_nf is not None:
            atlas_nf = np.asarray(atlas_nf[keep_mask], dtype=np.float32)

    neural_nt = zscore_neuron_time(neural_nt_raw) if normalize_neural else as_float32_clean(neural_nt_raw)
    coords_nf = as_float32_clean(coords_nf)
    if genes_nf is not None:
        genes_nf = as_float32_clean(genes_nf)
    if region_membership_nf is not None:
        region_membership_nf = as_float32_clean(region_membership_nf)
    if atlas_nf is not None:
        atlas_nf = as_float32_clean(atlas_nf)
    if activity_nf is not None:
        activity_nf = as_float32_clean(activity_nf)

    if debug:
        print(
            f"[LoadDebug] fish={fish_id} neural={neural_nt.shape} behavior={behavior_t.shape} "
            f"genes={None if genes_nf is None else genes_nf.shape} "
            f"coords={coords_nf.shape} atlas={None if atlas_nf is None else atlas_nf.shape} "
            f"region={None if region_membership_nf is None else region_membership_nf.shape} "
            f"activity={None if activity_nf is None else activity_nf.shape}"
        )

    return FishData(
        fish_id=int(fish_id),
        neural_nt=neural_nt,
        behavior_t=behavior_t,
        genes_nf=genes_nf,
        coords_nf=coords_nf,
        regions_nf=atlas_nf,
        region_membership_nf=region_membership_nf,
        activity_nf=activity_nf,
        global_neuron_idx_n=np.asarray(neuron_idx, dtype=np.int64),
        region_labels=atlas_labels,
        region_membership_labels=region_membership_labels,
    )


def normalize_static_features(train_sf: np.ndarray, test_sf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = train_sf.mean(axis=0, keepdims=True)
    s = train_sf.std(axis=0, keepdims=True)
    s = np.where(s < 1e-6, 1.0, s)
    return ((train_sf - m) / s).astype(np.float32), ((test_sf - m) / s).astype(np.float32)


def build_static_features(
    genes_nf: np.ndarray | None,
    coords_nf: np.ndarray | None,
    use_gene: bool,
    use_pos: bool,
    atlas_nf: np.ndarray | None = None,
    use_atlas: bool = False,
    region_nf: np.ndarray | None = None,
    use_region: bool = False,
    activity_nf: np.ndarray | None = None,
    use_activity: bool = False,
) -> np.ndarray | None:
    feats = []
    if use_gene:
        if genes_nf is None:
            raise ValueError("use_gene=True but genes are not loaded")
        feats.append(np.asarray(genes_nf, dtype=np.float32))
    if use_pos:
        if coords_nf is None:
            raise ValueError("use_pos=True but coords are not loaded")
        feats.append(np.asarray(coords_nf, dtype=np.float32))
    if use_atlas:
        if atlas_nf is None:
            raise ValueError("use_atlas=True but atlas features are not loaded")
        feats.append(np.asarray(atlas_nf, dtype=np.float32))
    if use_region:
        if region_nf is None:
            raise ValueError("use_region=True but region features are not loaded")
        feats.append(np.asarray(region_nf, dtype=np.float32))
    if use_activity:
        if activity_nf is None:
            raise ValueError("use_activity=True but activity features are not loaded")
        feats.append(np.asarray(activity_nf, dtype=np.float32))
    if not feats:
        return None
    return np.concatenate(feats, axis=1).astype(np.float32)


def shift_time_matrix(x_ft: np.ndarray, shift_frames: int) -> np.ndarray:
    x = np.asarray(x_ft, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D time matrix, got shape={x.shape}")
    s = int(shift_frames)
    if s == 0:
        return x.copy()
    out = np.zeros_like(x)
    if s > 0:
        out[:, s:] = x[:, :-s]
    else:
        out[:, :s] = x[:, -s:]
    return out


def load_stimulus_matrix(
    postprocessed_dir: Path,
    t_len: int,
) -> tuple[np.ndarray | None, str | None]:
    src_name = "delay"
    src_path = Path(postprocessed_dir) / "regressors_SimpleSqr_AllDelay.npy"
    if src_path is None:
        return None, None
    if not src_path.exists():
        return None, None

    stim = np.asarray(np.load(src_path, mmap_mode="r"), dtype=np.float32)
    if stim.ndim != 2:
        return None, None
    if stim.shape[1] > t_len:
        stim = stim[:, :t_len]
    elif stim.shape[1] < t_len:
        pad = np.zeros((stim.shape[0], t_len - stim.shape[1]), dtype=np.float32)
        stim = np.concatenate([stim, pad], axis=1)
    return as_float32_clean(stim), src_name


def sample_neuron_indices(
    rng: np.random.Generator,
    n_total: int,
    n_sample: int,
) -> np.ndarray:
    k = min(int(n_sample), int(n_total))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= int(n_total):
        return np.arange(int(n_total), dtype=np.int64)
    return np.asarray(rng.choice(int(n_total), size=k, replace=False), dtype=np.int64)


def build_time_batch_input(
    model: nn.Module,
    x_nt_ctx: torch.Tensor | dict[str, torch.Tensor],
    t_b: torch.Tensor,
) -> torch.Tensor | dict[str, torch.Tensor]:
    raw_ctx = x_nt_ctx["raw"] if isinstance(x_nt_ctx, dict) else x_nt_ctx
    wl = int(getattr(model, "window_left", 0))
    wr = int(getattr(model, "window_right", 0))
    if wl == 0 and wr == 0:
        if not isinstance(x_nt_ctx, dict):
            return raw_ctx.index_select(1, t_b).transpose(0, 1).contiguous()
        return {key: val.index_select(1, t_b).transpose(0, 1).contiguous() for key, val in x_nt_ctx.items()}

    if raw_ctx.ndim == 3:
        if not isinstance(x_nt_ctx, dict):
            return raw_ctx.index_select(1, t_b).permute(1, 0, 2).contiguous()
        return {key: val.index_select(1, t_b).permute(1, 0, 2).contiguous() for key, val in x_nt_ctx.items()}

    offsets = torch.arange(-wl, wr + 1, device=t_b.device, dtype=t_b.dtype)
    t_max = raw_ctx.shape[1] - 1
    if bool(getattr(model, "_time_offset_window_clamp", False)):
        center = torch.clamp(t_b, min=wl, max=max(wl, t_max - wr))
        idx = center[:, None] + offsets[None, :]
    else:
        idx = torch.clamp(t_b[:, None] + offsets[None, :], min=0, max=t_max)
    flat = idx.reshape(-1)
    if not isinstance(x_nt_ctx, dict):
        x_n_bw = raw_ctx.index_select(1, flat).reshape(raw_ctx.shape[0], idx.shape[0], idx.shape[1])
        return x_n_bw.permute(1, 0, 2).contiguous()
    out: dict[str, torch.Tensor] = {}
    for key, val in x_nt_ctx.items():
        x_n_bw = val.index_select(1, flat).reshape(val.shape[0], idx.shape[0], idx.shape[1])
        out[key] = x_n_bw.permute(1, 0, 2).contiguous()
    return out


def _replicate_pad_time_2d(x_nt: torch.Tensor, wl: int, wr: int) -> torch.Tensor:
    if wl <= 0 and wr <= 0:
        return x_nt
    pads = []
    if wl > 0:
        pads.append(x_nt[:, :1].expand(-1, wl))
    pads.append(x_nt)
    if wr > 0:
        pads.append(x_nt[:, -1:].expand(-1, wr))
    return torch.cat(pads, dim=1)


def precompute_windowed_time_context(
    model: nn.Module,
    x_nt_ctx: torch.Tensor | dict[str, torch.Tensor],
) -> torch.Tensor | dict[str, torch.Tensor]:
    wl = int(getattr(model, "window_left", 0))
    wr = int(getattr(model, "window_right", 0))
    if wl == 0 and wr == 0:
        return x_nt_ctx
    if bool(getattr(model, "_time_offset_window_clamp", False)):
        return x_nt_ctx

    def _windowize(val: torch.Tensor) -> torch.Tensor:
        if val.ndim == 3:
            return val
        return _replicate_pad_time_2d(val, wl, wr).unfold(dimension=1, size=wl + wr + 1, step=1)

    if not isinstance(x_nt_ctx, dict):
        return _windowize(x_nt_ctx)
    return {key: _windowize(val) for key, val in x_nt_ctx.items()}


def precompute_model_static_cache(
    model: nn.Module,
    static_nf: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    if static_nf is None or not hasattr(model, "precompute_static_cache"):
        return None
    return model.precompute_static_cache(static_nf, dtype=dtype, device=device)


def model_forward_maybe_cached(
    model: nn.Module,
    x_btn: torch.Tensor | dict[str, torch.Tensor],
    static_nf: torch.Tensor | None,
    static_cache=None,
) -> torch.Tensor:
    if static_cache is not None and hasattr(model, "precompute_static_cache"):
        return model(x_btn, static_nf, static_cache=static_cache)
    return model(x_btn, static_nf)


def build_decode_behavior_target(
    behavior_t: np.ndarray,
) -> tuple[np.ndarray, str]:
    y_raw = np.asarray(behavior_t, dtype=np.float32).reshape(-1)
    return y_raw, "raw"


def build_stimulus_class_target(
    stimulus_st: np.ndarray | None,
    active_threshold: float = 0.5,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if stimulus_st is None:
        return None, None
    stim = np.asarray(stimulus_st, dtype=np.float32)
    if stim.ndim != 2 or stim.shape[0] < 2 or stim.shape[1] < 1:
        return None, None
    stim_binary = np.asarray(stim > float(active_threshold), dtype=np.int64)
    active_mask_t = np.sum(stim_binary, axis=0) > 0
    stim_class_t = np.argmax(stim_binary, axis=0).astype(np.int64)
    return stim_class_t, np.asarray(active_mask_t, dtype=bool)

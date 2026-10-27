#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import identity_guided_grouping_decoding as base_runner
from src import data, eval as eval_utils, plotting
from src.io_utils import resolve_device, safe_mean, safe_pearsonr, safe_std, save_json
from src.repro import snapshot_clean_pack_entrypoint
from src.transfer_context import (
    BaseFishContext,
    BehaviorNormalizer,
    CaseFishContext,
    behavior_normalizer_to_dict,
    build_behavior_normalizer,
    build_case_contexts,
    build_subset_ensemble,
)


DEFAULT_ALLEN_ROOT = Path("data/allen/preprocessed_data")
GROUPED_STIMULUS_FAMILY_MAP_21 = {
    "gabors": "gabors",
    "flashes": "flashes",
    "drifting_gratings": "gratings",
    "static_gratings": "gratings",
    "natural_movie_one": "natural_movies",
    "natural_movie_three": "natural_movies",
    "natural_scenes": "natural_scenes",
}
GROUPED_STIMULUS_FAMILY_MAP_22 = {
    "gabors": "gabors",
    "flashes": "flashes",
    "drifting_gratings_contrast": "gratings",
    "drifting_gratings_75_repeats": "gratings",
    "natural_movie_one_more_repeats": "natural_movies",
    "natural_movie_one_shuffled": "natural_movies",
    "dot_motion": "dot_motion",
}


@dataclass(frozen=True)
class AllenTaskSpec:
    name: str
    task_kind: str
    stimulus_family: str | None = None
    stimulus_label_key: str | None = None
    stimulus_condition_filter: dict[str, Any] | None = None
    stimulus_group_map: dict[str, str] | None = None
    val_split_mode: str = "tail"


@dataclass
class AllenSessionBundle:
    fish: data.FishData
    metadata: dict[str, Any]
    bin_edges_s: np.ndarray
    bin_centers_s: np.ndarray
    behavior_key: str
    behavior_valid_t: np.ndarray
    invalid_time_mask_t: np.ndarray
    stimulus_start_s: np.ndarray
    stimulus_stop_s: np.ndarray
    stimulus_family: np.ndarray
    stimulus_condition_json: np.ndarray


@dataclass
class AllenBaseContext(BaseFishContext):
    behavior_valid_mask_t: np.ndarray | None = None
    invalid_time_mask_t: np.ndarray | None = None
    stimulus_label_map: dict[int, str] | None = None
    session_metadata: dict[str, Any] | None = None


def _safe_slug(name: str) -> str:
    text = str(name).strip().lower()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def canonicalize_dataset_id(dataset_id: str | int) -> str:
    text = str(dataset_id).strip()
    if not text:
        raise ValueError("dataset_id must be non-empty.")
    digits = text.replace("dataset_", "")
    if digits not in {"21", "22", "000021", "000022"}:
        raise ValueError(f"Unsupported dataset_id={dataset_id}. Expected one of 21 or 22.")
    return f"{int(digits):06d}"


def dataset_dir_from_id(root: Path, dataset_id: str | int) -> Path:
    return Path(root) / f"dataset_{canonicalize_dataset_id(dataset_id)}"


def _load_metadata(bundle: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return json.loads(str(np.asarray(bundle["metadata_json"]).item()))


def list_usable_session_paths(root: Path, dataset_id: str | int) -> list[tuple[int, Path]]:
    dataset_dir = dataset_dir_from_id(root, dataset_id)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Allen dataset directory not found: {dataset_dir}")
    session_info: list[tuple[int, Path]] = []
    for npz_path in sorted(dataset_dir.glob("session_*.npz")):
        with np.load(npz_path, allow_pickle=False) as bundle:
            unit_id_n = np.asarray(bundle["unit_id_n"], dtype=np.int64)
            if unit_id_n.size <= 0:
                continue
            meta = _load_metadata(bundle)
            session_info.append((int(meta["session_id"]), npz_path))
    if not session_info:
        raise ValueError(f"No usable Allen sessions found under {dataset_dir}")
    return sorted(session_info, key=lambda item: int(item[0]))


def _make_bin_edges(stop_time_s: float, bin_size_s: float) -> np.ndarray:
    stop = float(stop_time_s)
    width = float(bin_size_s)
    if not math.isfinite(stop) or stop <= 0.0:
        raise ValueError(f"Invalid stop_time_s={stop_time_s}")
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError(f"bin_size_s must be positive, got {bin_size_s}")
    edges = np.arange(0.0, stop + width, width, dtype=np.float64)
    if float(edges[-1]) < stop:
        edges = np.append(edges, stop)
    return edges


def _bin_spikes(spike_times_s_flat: np.ndarray, spike_times_indptr: np.ndarray, bin_edges_s: np.ndarray) -> np.ndarray:
    n_units = int(spike_times_indptr.size) - 1
    out = np.zeros((n_units, int(bin_edges_s.size) - 1), dtype=np.float32)
    for unit_idx in range(n_units):
        start = int(spike_times_indptr[unit_idx])
        end = int(spike_times_indptr[unit_idx + 1])
        seg = np.asarray(spike_times_s_flat[start:end], dtype=np.float64)
        if seg.size <= 0:
            continue
        out[unit_idx], _ = np.histogram(seg, bins=bin_edges_s)
    return out


def _resample_behavior_stream(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    bin_centers_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    vals = np.asarray(values, dtype=np.float32)
    if ts.size <= 1:
        raise ValueError("Behavior stream is too short to resample.")
    if vals.ndim == 1:
        vals = vals[:, None]
    if vals.shape[0] != ts.size:
        raise ValueError(f"Behavior timestamp/value mismatch: ts={ts.shape} values={vals.shape}")
    order = np.argsort(ts)
    ts = ts[order]
    vals = vals[order]
    keep = np.concatenate(([True], np.diff(ts) > 0.0))
    ts = ts[keep]
    vals = vals[keep]
    out = np.empty((int(bin_centers_s.size), int(vals.shape[1])), dtype=np.float32)
    for dim in range(int(vals.shape[1])):
        out[:, dim] = np.interp(bin_centers_s, ts, vals[:, dim], left=np.nan, right=np.nan).astype(np.float32)
    valid_t = np.all(np.isfinite(out), axis=1)
    return out, np.asarray(valid_t, dtype=bool)


def _build_invalid_time_mask(bin_centers_s: np.ndarray, invalid_intervals_s: np.ndarray) -> np.ndarray:
    mask = np.zeros(int(bin_centers_s.size), dtype=bool)
    intervals = np.asarray(invalid_intervals_s, dtype=np.float64).reshape(-1, 2)
    for start, stop in intervals:
        mask |= (bin_centers_s >= float(start)) & (bin_centers_s < float(stop))
    return mask


def load_session_bundle(npz_path: Path, args: argparse.Namespace) -> AllenSessionBundle:
    with np.load(npz_path, allow_pickle=False) as bundle:
        metadata = _load_metadata(bundle)
        session_id = int(metadata["session_id"])
        stop_time_s = float(metadata["session_stop_time_s"])
        bin_edges_s = _make_bin_edges(stop_time_s=stop_time_s, bin_size_s=float(args.bin_size_s))
        bin_centers_s = 0.5 * (bin_edges_s[:-1] + bin_edges_s[1:])

        neural_nt_raw = _bin_spikes(
            np.asarray(bundle["spike_times_s_flat"], dtype=np.float64),
            np.asarray(bundle["spike_times_indptr"], dtype=np.int64),
            bin_edges_s,
        )

        behavior_keys = np.asarray(bundle["behavior_keys"]).astype(str).tolist()
        if str(args.behavior_key) not in behavior_keys:
            raise KeyError(
                f"Behavior key {args.behavior_key} not found for session {session_id}. "
                f"Available: {sorted(behavior_keys)}"
            )
        behavior_safe = _safe_slug(str(args.behavior_key))
        behavior_td, behavior_valid_t = _resample_behavior_stream(
            np.asarray(bundle[f"behavior__{behavior_safe}__timestamps_s"], dtype=np.float64),
            np.asarray(bundle[f"behavior__{behavior_safe}__values"], dtype=np.float32),
            bin_centers_s,
        )
        if int(behavior_td.shape[1]) != 1:
            raise ValueError(
                f"Allen runner currently expects scalar behavior for regression. "
                f"Session {session_id} key={args.behavior_key} has shape {behavior_td.shape}."
            )

        invalid_time_mask_t = _build_invalid_time_mask(
            bin_centers_s,
            np.asarray(bundle["invalid_intervals_s"], dtype=np.float64).reshape(-1, 2),
        )

        coords_nf = np.asarray(bundle["coords_nf"], dtype=np.float32)
        region_membership_nf = np.asarray(bundle["region_membership_nf"], dtype=np.float32)
        unit_id_n = np.asarray(bundle["unit_id_n"], dtype=np.int64)
        region_vocab = np.asarray(bundle["region_label_vocab"]).astype(str).tolist()

        activity_nf = None
        if bool(getattr(args, "use_activity", False)):
            if args.activity_cache_dir is None:
                cache_suffix = _safe_slug(f"bin{float(args.bin_size_s):.3f}s")
                activity_cache_dir = npz_path.parent / f"activity_cache_robust_stats_{cache_suffix}"
            else:
                activity_cache_dir = Path(args.activity_cache_dir)
            activity_nf, _ = data.load_or_compute_activity_feature_matrix(
                neural_nt_raw,
                fish_id=int(session_id),
                cache_dir=Path(activity_cache_dir),
                mode=str(args.activity_feature_mode),
                force_recompute=bool(getattr(args, "activity_force_recompute", False)),
                require_cache=bool(getattr(args, "activity_require_cache", False)),
            )

        keep_mask = np.ones(int(neural_nt_raw.shape[0]), dtype=bool)
        keep_mask &= np.all(np.isfinite(neural_nt_raw), axis=1)
        keep_mask &= np.all(np.isfinite(coords_nf), axis=1)
        keep_mask &= np.all(np.isfinite(region_membership_nf), axis=1)
        if activity_nf is not None:
            keep_mask &= np.all(np.isfinite(activity_nf), axis=1)
        if str(getattr(args, "region_prefix_keep", "")).strip():
            keep_mask &= _build_region_prefix_keep_mask(bundle["region_label_n"], str(args.region_prefix_keep))
        if int(keep_mask.sum()) <= 0:
            raise ValueError(
                f"Session {session_id} has no neurons after filtering. "
                f"region_prefix_keep={getattr(args, 'region_prefix_keep', '')!r}"
            )

        if int((~keep_mask).sum()) > 0:
            neural_nt_raw = np.asarray(neural_nt_raw[keep_mask], dtype=np.float32)
            coords_nf = np.asarray(coords_nf[keep_mask], dtype=np.float32)
            region_membership_nf = np.asarray(region_membership_nf[keep_mask], dtype=np.float32)
            unit_id_n = np.asarray(unit_id_n[keep_mask], dtype=np.int64)
            if activity_nf is not None:
                activity_nf = np.asarray(activity_nf[keep_mask], dtype=np.float32)

        neural_nt = (
            data.zscore_neuron_time(neural_nt_raw)
            if bool(args.normalize_neural)
            else data.as_float32_clean(neural_nt_raw)
        )

        if bool(getattr(args, "debug_load", False)):
            print(
                f"[AllenLoad] session={session_id} neural={neural_nt.shape} "
                f"behavior={behavior_td.shape} coords={coords_nf.shape} "
                f"region={region_membership_nf.shape} activity={None if activity_nf is None else activity_nf.shape}",
                flush=True,
            )

        fish = data.FishData(
            int(session_id),
            np.asarray(neural_nt, dtype=np.float32),
            np.asarray(behavior_td[:, 0], dtype=np.float32),
            None,
            data.as_float32_clean(coords_nf),
            None,
            data.as_float32_clean(region_membership_nf),
            None if activity_nf is None else data.as_float32_clean(activity_nf),
            np.asarray(unit_id_n, dtype=np.int64),
            None,
            [str(x) for x in region_vocab],
        )

        return AllenSessionBundle(
            fish=fish,
            metadata=dict(metadata),
            bin_edges_s=np.asarray(bin_edges_s, dtype=np.float64),
            bin_centers_s=np.asarray(bin_centers_s, dtype=np.float64),
            behavior_key=str(args.behavior_key),
            behavior_valid_t=np.asarray(behavior_valid_t, dtype=bool),
            invalid_time_mask_t=np.asarray(invalid_time_mask_t, dtype=bool),
            stimulus_start_s=np.asarray(bundle["stimulus_event_start_time_s"], dtype=np.float64),
            stimulus_stop_s=np.asarray(bundle["stimulus_event_stop_time_s"], dtype=np.float64),
            stimulus_family=np.asarray(bundle["stimulus_event_family"]).astype(str),
            stimulus_condition_json=np.asarray(bundle["stimulus_event_condition_json"]).astype(str),
        )


def _canonical_condition_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        num = float(value)
        if not math.isfinite(num):
            return None
        rounded = round(num)
        if abs(num - rounded) <= 1e-6:
            return int(rounded)
        return num
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "none":
            return None
        try:
            return _canonical_condition_value(float(text))
        except ValueError:
            return text
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_condition_value(v) for v in value)
    return str(value)


def _format_condition_value(value: Any) -> str:
    if isinstance(value, tuple):
        return "[" + ", ".join(_format_condition_value(v) for v in value) + "]"
    return str(value)


def _build_region_prefix_keep_mask(region_label_n: np.ndarray, prefixes_csv: str) -> np.ndarray:
    prefixes = [part.strip() for part in str(prefixes_csv).split(",") if part.strip()]
    if not prefixes:
        raise ValueError("region_prefix_keep must contain at least one non-empty prefix.")
    labels = np.asarray(region_label_n).astype(str)
    keep = np.zeros(int(labels.shape[0]), dtype=bool)
    for prefix in prefixes:
        keep |= np.char.startswith(labels, str(prefix))
    return np.asarray(keep, dtype=bool)


def _label_sort_key(value: Any) -> tuple[int, float, str]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value), str(value))
    return (1, 0.0, str(value))


def _stimulus_event_matches(
    *,
    family: str,
    cond: dict[str, Any],
    target_family: str,
    condition_filter: dict[str, Any] | None,
) -> bool:
    if str(family) != str(target_family):
        return False
    if not condition_filter:
        return True
    for key, expected in condition_filter.items():
        value = _canonical_condition_value(cond.get(str(key)))
        if value != _canonical_condition_value(expected):
            return False
    return True


def collect_stimulus_label_map(
    bundles: dict[int, AllenSessionBundle],
    *,
    stimulus_family: str,
    stimulus_label_key: str,
    stimulus_condition_filter: dict[str, Any] | None = None,
    stimulus_group_map: dict[str, str] | None = None,
) -> tuple[dict[Any, int], dict[int, str]]:
    if stimulus_group_map is not None:
        present_families: set[str] = set()
        for bundle in bundles.values():
            present_families.update(np.asarray(bundle.stimulus_family).astype(str).tolist())
        ordered_labels: list[str] = []
        seen_labels: set[str] = set()
        for family_name, label in stimulus_group_map.items():
            if str(family_name) not in present_families or str(label) in seen_labels:
                continue
            ordered_labels.append(str(label))
            seen_labels.add(str(label))
        if not ordered_labels:
            raise ValueError("No usable grouped stimulus labels found for the requested family map.")
        label_to_id = {label: idx for idx, label in enumerate(ordered_labels)}
        label_map = {idx: label for idx, label in enumerate(ordered_labels)}
        return label_to_id, label_map

    label_values: set[Any] = set()
    family = str(stimulus_family)
    key = str(stimulus_label_key)
    for bundle in bundles.values():
        families = np.asarray(bundle.stimulus_family).astype(str)
        cond_json = np.asarray(bundle.stimulus_condition_json).astype(str)
        for fam, cond_text in zip(families.tolist(), cond_json.tolist()):
            cond = json.loads(str(cond_text))
            if not _stimulus_event_matches(
                family=str(fam),
                cond=cond,
                target_family=family,
                condition_filter=stimulus_condition_filter,
            ):
                continue
            value = _canonical_condition_value(cond.get(key))
            if value is None:
                continue
            label_values.add(value)
    if not label_values:
        raise ValueError(
            f"No usable labels found for family={stimulus_family!r} key={stimulus_label_key!r}."
        )
    ordered_values = sorted(label_values, key=_label_sort_key)
    label_to_id = {value: idx for idx, value in enumerate(ordered_values)}
    label_map = {idx: _format_condition_value(value) for idx, value in enumerate(ordered_values)}
    return label_to_id, label_map


def build_selected_stimulus_targets(
    bundle: AllenSessionBundle,
    *,
    stimulus_family: str,
    stimulus_label_key: str,
    label_to_id: dict[Any, int],
    stimulus_condition_filter: dict[str, Any] | None = None,
    stimulus_group_map: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    t_len = int(bundle.bin_centers_s.size)
    class_t = np.zeros(t_len, dtype=np.int64)
    active_mask_t = np.zeros(t_len, dtype=bool)
    family = str(stimulus_family)
    key = str(stimulus_label_key)

    if stimulus_group_map is not None:
        for start_s, stop_s, fam in zip(
            bundle.stimulus_start_s.tolist(),
            bundle.stimulus_stop_s.tolist(),
            np.asarray(bundle.stimulus_family).astype(str).tolist(),
        ):
            label = stimulus_group_map.get(str(fam))
            if label is None:
                continue
            if label not in label_to_id:
                raise KeyError(f"Grouped stimulus label {label!r} missing from global label map.")
            event_mask = (bundle.bin_centers_s >= float(start_s)) & (bundle.bin_centers_s < float(stop_s))
            if bundle.invalid_time_mask_t.size == event_mask.size:
                event_mask &= ~np.asarray(bundle.invalid_time_mask_t, dtype=bool)
            if not np.any(event_mask):
                continue
            class_t[event_mask] = int(label_to_id[label])
            active_mask_t[event_mask] = True
        return np.asarray(class_t, dtype=np.int64), np.asarray(active_mask_t, dtype=bool)

    for start_s, stop_s, fam, cond_text in zip(
        bundle.stimulus_start_s.tolist(),
        bundle.stimulus_stop_s.tolist(),
        np.asarray(bundle.stimulus_family).astype(str).tolist(),
        np.asarray(bundle.stimulus_condition_json).astype(str).tolist(),
    ):
        cond = json.loads(str(cond_text))
        if not _stimulus_event_matches(
            family=str(fam),
            cond=cond,
            target_family=family,
            condition_filter=stimulus_condition_filter,
        ):
            continue
        value = _canonical_condition_value(cond.get(key))
        if value is None:
            continue
        if value not in label_to_id:
            raise KeyError(
                f"Stimulus label value {value!r} missing from global label map for family={family} key={key}."
            )
        event_mask = (bundle.bin_centers_s >= float(start_s)) & (bundle.bin_centers_s < float(stop_s))
        if bundle.invalid_time_mask_t.size == event_mask.size:
            event_mask &= ~np.asarray(bundle.invalid_time_mask_t, dtype=bool)
        if not np.any(event_mask):
            continue
        class_t[event_mask] = int(label_to_id[value])
        active_mask_t[event_mask] = True
    return np.asarray(class_t, dtype=np.int64), np.asarray(active_mask_t, dtype=bool)


def build_base_context_for_task(
    bundle: AllenSessionBundle,
    *,
    args: argparse.Namespace,
    task_spec: AllenTaskSpec,
    stimulus_label_to_id: dict[Any, int] | None,
    stimulus_label_map: dict[int, str] | None,
) -> AllenBaseContext:
    fish = bundle.fish
    static_raw_nf = data.build_static_features(
        None,
        fish.coords_nf,
        False,
        bool(args.use_pos),
        None,
        False,
        fish.region_membership_nf,
        bool(getattr(args, "use_region", False)),
        fish.activity_nf,
        bool(getattr(args, "use_activity", False)),
    )

    behavior_target_t, behavior_target_label = data.build_decode_behavior_target(
        behavior_t=fish.behavior_t,
    )

    stimulus_class_t = None
    stimulus_active_mask_t = None
    stimulus_target_label = None
    if str(task_spec.task_kind) == "stimulus":
        if stimulus_label_to_id is None or stimulus_label_map is None:
            raise ValueError(f"Stimulus task {task_spec.name} is missing a global label map.")
        stimulus_class_t, stimulus_active_mask_t = build_selected_stimulus_targets(
            bundle,
            stimulus_family=str(task_spec.stimulus_family),
            stimulus_label_key=str(task_spec.stimulus_label_key),
            label_to_id=stimulus_label_to_id,
            stimulus_condition_filter=task_spec.stimulus_condition_filter,
            stimulus_group_map=task_spec.stimulus_group_map,
        )
        if task_spec.stimulus_group_map is not None:
            stimulus_target_label = str(task_spec.name)
        else:
            stimulus_target_label = f"{task_spec.stimulus_family}:{task_spec.stimulus_label_key}"

    behavior_valid_mask_t = np.asarray(bundle.behavior_valid_t, dtype=bool)
    invalid_time_mask_t = np.asarray(bundle.invalid_time_mask_t, dtype=bool)
    if invalid_time_mask_t.size == behavior_valid_mask_t.size:
        behavior_valid_mask_t = behavior_valid_mask_t & (~invalid_time_mask_t)

    return AllenBaseContext(
        fish=fish,
        static_raw_nf=static_raw_nf,
        behavior_target_t=np.asarray(behavior_target_t, dtype=np.float32),
        behavior_target_label=str(behavior_target_label),
        stimulus_st=None,
        stimulus_target_label=stimulus_target_label,
        stimulus_class_t=stimulus_class_t,
        stimulus_active_mask_t=stimulus_active_mask_t,
        behavior_valid_mask_t=behavior_valid_mask_t,
        invalid_time_mask_t=invalid_time_mask_t,
        stimulus_label_map=None if stimulus_label_map is None else dict(stimulus_label_map),
        session_metadata={**dict(bundle.metadata), "behavior_key": str(bundle.behavior_key)},
    )


def build_base_contexts_for_task(
    bundles: dict[int, AllenSessionBundle],
    *,
    args: argparse.Namespace,
    task_spec: AllenTaskSpec,
) -> tuple[dict[int, AllenBaseContext], dict[int, str] | None]:
    stimulus_label_to_id = None
    stimulus_label_map = None
    if str(task_spec.task_kind) == "stimulus":
        stimulus_label_to_id, stimulus_label_map = collect_stimulus_label_map(
            bundles,
            stimulus_family=str(task_spec.stimulus_family),
            stimulus_label_key=str(task_spec.stimulus_label_key),
            stimulus_condition_filter=task_spec.stimulus_condition_filter,
            stimulus_group_map=task_spec.stimulus_group_map,
        )
    base_contexts: dict[int, AllenBaseContext] = {}
    for session_id, bundle in bundles.items():
        base_contexts[int(session_id)] = build_base_context_for_task(
            bundle,
            args=args,
            task_spec=task_spec,
            stimulus_label_to_id=stimulus_label_to_id,
            stimulus_label_map=stimulus_label_map,
        )
    return base_contexts, stimulus_label_map


def get_valid_task_indices(ctx: CaseFishContext, task_kind: str) -> np.ndarray:
    base = ctx.base
    if str(task_kind) == "behavior":
        mask = np.ones(int(base.fish.neural_nt.shape[1]), dtype=bool)
        valid_mask = getattr(base, "behavior_valid_mask_t", None)
        if valid_mask is not None:
            mask &= np.asarray(valid_mask, dtype=bool)
        invalid_mask = getattr(base, "invalid_time_mask_t", None)
        if invalid_mask is not None:
            mask &= ~np.asarray(invalid_mask, dtype=bool)
        idx = np.flatnonzero(mask)
        if idx.size <= 0:
            raise ValueError(f"Session {base.fish.fish_id} has no valid behavior frames.")
        return np.asarray(idx, dtype=np.int64)
    if base.stimulus_class_t is None or base.stimulus_active_mask_t is None:
        raise ValueError(f"Session {base.fish.fish_id} is missing stimulus targets.")
    mask = np.asarray(base.stimulus_active_mask_t, dtype=bool)
    invalid_mask = getattr(base, "invalid_time_mask_t", None)
    if invalid_mask is not None and np.asarray(invalid_mask).size == mask.size:
        mask &= ~np.asarray(invalid_mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size <= 0:
        raise ValueError(f"Session {base.fish.fish_id} has no active stimulus frames.")
    return np.asarray(idx, dtype=np.int64)


def build_tail_split_allen(
    source_ctxs: dict[int, CaseFishContext],
    *,
    task_kind: str,
    val_frac: float,
) -> dict[str, dict[int, np.ndarray]]:
    frac = float(val_frac)
    if not (0.0 < frac < 0.5):
        raise ValueError(f"val_frac must be in (0, 0.5), got {val_frac}")
    train_map: dict[int, np.ndarray] = {}
    val_map: dict[int, np.ndarray] = {}
    for session_id, ctx in source_ctxs.items():
        t_len = int(ctx.base.fish.neural_nt.shape[1])
        split_t = int(round((1.0 - frac) * t_len))
        split_t = max(1, min(split_t, t_len - 1))
        valid_idx = get_valid_task_indices(ctx, str(task_kind))
        if valid_idx.size < 8:
            raise ValueError(
                f"Session {session_id} has too few valid samples for a train/val split: {valid_idx.size}"
            )
        train_idx = np.asarray(valid_idx[valid_idx < split_t], dtype=np.int64)
        val_idx = np.asarray(valid_idx[valid_idx >= split_t], dtype=np.int64)
        if train_idx.size <= 0 or val_idx.size <= 0:
            n_val = int(round(valid_idx.size * frac))
            n_val = max(1, min(n_val, int(valid_idx.size) - 4))
            train_idx = np.asarray(valid_idx[:-n_val], dtype=np.int64)
            val_idx = np.asarray(valid_idx[-n_val:], dtype=np.int64)
        if train_idx.size <= 0 or val_idx.size <= 0:
            raise ValueError(f"Tail split failed for session {session_id}: train={train_idx.size}, val={val_idx.size}")
        train_map[int(session_id)] = train_idx
        val_map[int(session_id)] = val_idx
    return {"train": train_map, "val": val_map}


def _extract_stimulus_event_segments(ctx: CaseFishContext) -> dict[int, list[np.ndarray]]:
    base = ctx.base
    if base.stimulus_class_t is None or base.stimulus_active_mask_t is None:
        raise ValueError(f"Session {base.fish.fish_id} is missing stimulus targets.")
    active_mask = np.asarray(base.stimulus_active_mask_t, dtype=bool)
    invalid_mask = getattr(base, "invalid_time_mask_t", None)
    if invalid_mask is not None and np.asarray(invalid_mask).size == active_mask.size:
        active_mask &= ~np.asarray(invalid_mask, dtype=bool)
    active_idx = np.flatnonzero(active_mask)
    if active_idx.size <= 0:
        raise ValueError(f"Session {base.fish.fish_id} has no active stimulus frames.")
    class_t = np.asarray(base.stimulus_class_t, dtype=np.int64)
    split_points = np.flatnonzero(
        (np.diff(active_idx) != 1) | (class_t[active_idx[1:]] != class_t[active_idx[:-1]])
    ) + 1
    segments: dict[int, list[np.ndarray]] = {}
    for seg in np.split(active_idx, split_points):
        seg_idx = np.asarray(seg, dtype=np.int64)
        if seg_idx.size <= 0:
            continue
        label = int(class_t[int(seg_idx[0])])
        segments.setdefault(label, []).append(seg_idx)
    return segments


def _merge_index_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    merged = np.concatenate([np.asarray(chunk, dtype=np.int64) for chunk in chunks], axis=0)
    if merged.size <= 1:
        return np.asarray(merged, dtype=np.int64)
    return np.asarray(np.unique(merged), dtype=np.int64)


def build_event_stratified_split_allen(
    source_ctxs: dict[int, CaseFishContext],
    *,
    val_frac: float,
    split_seed: int,
) -> dict[str, dict[int, np.ndarray]]:
    frac = float(val_frac)
    if not (0.0 < frac < 0.5):
        raise ValueError(f"val_frac must be in (0, 0.5), got {val_frac}")

    train_chunks: dict[int, list[np.ndarray]] = {int(session_id): [] for session_id in source_ctxs}
    val_chunks: dict[int, list[np.ndarray]] = {int(session_id): [] for session_id in source_ctxs}
    segments_by_label: dict[int, list[tuple[int, np.ndarray]]] = {}
    for session_id, ctx in source_ctxs.items():
        for label, segments in _extract_stimulus_event_segments(ctx).items():
            entries = segments_by_label.setdefault(int(label), [])
            for seg_idx in segments:
                entries.append((int(session_id), np.asarray(seg_idx, dtype=np.int64)))

    if not segments_by_label:
        raise ValueError("No stimulus events available for event-stratified validation.")

    rng = np.random.default_rng(int(split_seed))
    for label, entries in sorted(segments_by_label.items()):
        if len(entries) <= 1:
            train_entries = entries
            val_entries: list[tuple[int, np.ndarray]] = []
        else:
            order = np.asarray(rng.permutation(len(entries)), dtype=np.int64)
            shuffled = [entries[int(i)] for i in order.tolist()]
            n_val = int(round(len(shuffled) * frac))
            n_val = max(1, min(n_val, len(shuffled) - 1))
            val_entries = shuffled[:n_val]
            train_entries = shuffled[n_val:]
        for session_id, seg_idx in train_entries:
            train_chunks[int(session_id)].append(np.asarray(seg_idx, dtype=np.int64))
        for session_id, seg_idx in val_entries:
            val_chunks[int(session_id)].append(np.asarray(seg_idx, dtype=np.int64))

    train_map: dict[int, np.ndarray] = {}
    val_map: dict[int, np.ndarray] = {}
    total_train = 0
    total_val = 0
    for session_id in sorted(source_ctxs):
        train_idx = _merge_index_chunks(train_chunks[int(session_id)])
        val_idx = _merge_index_chunks(val_chunks[int(session_id)])
        train_map[int(session_id)] = train_idx
        val_map[int(session_id)] = val_idx
        total_train += int(train_idx.size)
        total_val += int(val_idx.size)
    if total_train <= 0 or total_val <= 0:
        raise ValueError(
            f"Event-stratified split failed: total_train={total_train} total_val={total_val}"
        )
    return {"train": train_map, "val": val_map}


def evaluate_heldout_allen(
    model: torch.nn.Module,
    *,
    target_ctx: CaseFishContext,
    task_spec: AllenTaskSpec,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
    num_classes: int | None,
    train_session_ids: list[int],
    neuron_subsets: list[np.ndarray] | None,
    behavior_normalizer: BehaviorNormalizer | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = int(target_ctx.base.fish.fish_id)
    all_idx = np.arange(int(target_ctx.base.fish.neural_nt.shape[1]), dtype=np.int64)
    subset_list = (
        [np.arange(int(target_ctx.base.fish.neural_nt.shape[0]), dtype=np.int64)]
        if neuron_subsets is None or len(neuron_subsets) <= 0
        else [np.asarray(subset_idx, dtype=np.int64) for subset_idx in neuron_subsets]
    )
    pred_subsets = [
        base_runner.predict_time_indices(
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

    if str(task_spec.task_kind) == "behavior":
        valid_idx = get_valid_task_indices(target_ctx, "behavior")
        y_true_full = np.asarray(target_ctx.base.behavior_target_t, dtype=np.float32)
        pred_stack_norm = np.asarray(pred_subsets, dtype=np.float32).reshape(len(subset_list), -1)
        y_pred_norm = pred_stack_norm.mean(axis=0)
        pred_stack = base_runner.denormalize_behavior_array(
            pred_stack_norm,
            behavior_normalizer,
            fish_id=int(target_ctx.base.fish.fish_id),
        )
        y_pred = base_runner.denormalize_behavior_array(
            y_pred_norm,
            behavior_normalizer,
            fish_id=int(target_ctx.base.fish.fish_id),
        )
        y_true = np.asarray(y_true_full[valid_idx], dtype=np.float32)
        y_pred_valid = np.asarray(y_pred[valid_idx], dtype=np.float32)
        corr = safe_pearsonr(y_true, y_pred_valid)
        mse = float(np.mean((y_true - y_pred_valid) ** 2))
        payload = {
            "task": "behavior",
            "target_session_id": int(session_id),
            "train_session_ids": [int(fid) for fid in train_session_ids],
            "behavior_target_label": str(target_ctx.base.behavior_target_label),
            "behavior_key": getattr(target_ctx.base, "session_metadata", {}).get("behavior_key", None),
            "test_corr": corr,
            "test_mse": mse,
            "n_frames_total": int(y_true_full.size),
            "n_frames_valid": int(valid_idx.size),
            "n_eval_neurons": int(subset_stack.shape[1]),
            "n_eval_subsets": int(subset_stack.shape[0]),
            "eval_aggregation": "mean",
            "behavior_target_normalization": behavior_normalizer_to_dict(behavior_normalizer),
        }
        np.savez(
            out_dir / "behavior_predictions.npz",
            y_true_full=y_true_full,
            y_true_valid=y_true,
            y_pred_full=y_pred,
            y_pred_valid=y_pred_valid,
            y_pred_norm_full=y_pred_norm,
            y_pred_subsets=pred_stack,
            y_pred_subsets_norm=pred_stack_norm,
            frame_idx=all_idx,
            valid_idx=valid_idx,
            neuron_idx_subsets=subset_stack,
        )
        plotting.plot_behavior_predictions(
            y_true,
            y_pred_valid,
            out_dir / "behavior_predictions.png",
            title=(
                f"Allen behavior | train={len(train_session_ids)} sessions -> heldout={session_id} "
                f"| corr={corr:.3f} | ens={int(subset_stack.shape[0])}"
            ),
        )
        save_json(out_dir / "behavior_metrics.json", payload)
        return payload

    logits_stack = np.asarray(pred_subsets, dtype=np.float32)
    pred_cls_subsets = np.asarray(np.argmax(logits_stack, axis=2), dtype=np.int64)
    pred_cls_all, vote_counts_all, mean_logits_all = base_runner.vote_from_logits(logits_stack)
    active_idx = get_valid_task_indices(target_ctx, "stimulus")
    y_true = np.asarray(target_ctx.base.stimulus_class_t[active_idx], dtype=np.int64)
    y_pred = np.asarray(pred_cls_all[active_idx], dtype=np.int64)
    metrics = eval_utils.compute_classification_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(int(num_classes)))
    payload = {
        "task": "stimulus",
        "task_name": str(task_spec.name),
        "target_session_id": int(session_id),
        "train_session_ids": [int(fid) for fid in train_session_ids],
        "stimulus_target_label": target_ctx.base.stimulus_target_label,
        "stimulus_label_map": getattr(target_ctx.base, "stimulus_label_map", None),
        "n_active_frames": int(active_idx.size),
        "n_total_frames": int(target_ctx.base.fish.neural_nt.shape[1]),
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
        active_idx=active_idx,
        y_true_active=y_true,
        y_pred_active=y_pred,
        neuron_idx_subsets=subset_stack,
    )
    plotting.plot_confusion_matrix(
        cm,
        out_dir / "stimulus_confusion_matrix.png",
        title=(
            f"Allen stimulus | task={task_spec.name} | heldout={session_id} "
            f"| bal_acc={metrics['bal_acc']:.3f} | ens={int(subset_stack.shape[0])}"
        ),
        normalize=True,
    )
    save_json(out_dir / "stimulus_metrics.json", payload)
    return payload


def infer_num_classes_for_task(
    *,
    task_spec: AllenTaskSpec,
    stimulus_label_map: dict[int, str] | None,
    override: int,
) -> int | None:
    if str(task_spec.task_kind) != "stimulus":
        return None
    if int(override) > 0:
        return int(override)
    if stimulus_label_map is None or len(stimulus_label_map) <= 1:
        raise ValueError(f"Could not infer a valid number of classes for task {task_spec.name}.")
    return int(len(stimulus_label_map))


def build_fixed_session_split(
    session_ids: list[int],
    *,
    train_count: int,
    split_seed: int,
) -> dict[str, list[int]]:
    all_ids = sorted(int(x) for x in session_ids)
    if len(all_ids) < 2:
        raise ValueError("Need at least two usable sessions for train/test transfer.")
    if int(train_count) <= 0 or int(train_count) >= len(all_ids):
        raise ValueError(
            f"train_session_count must be in [1, {len(all_ids) - 1}], got {train_count}"
        )
    rng = np.random.default_rng(int(split_seed))
    shuffled = np.asarray(all_ids, dtype=np.int64)[np.asarray(rng.permutation(len(all_ids)), dtype=np.int64)]
    train_ids = sorted(int(x) for x in shuffled[: int(train_count)].tolist())
    heldout_ids = sorted(int(x) for x in shuffled[int(train_count) :].tolist())
    return {"train_session_ids": train_ids, "heldout_session_ids": heldout_ids}


def resolve_task_specs(dataset_id: str, task: str) -> list[AllenTaskSpec]:
    dataset_key = canonicalize_dataset_id(dataset_id)
    task_key = str(task).strip().lower()
    if task_key == "behavior":
        return [AllenTaskSpec(name="behavior", task_kind="behavior")]
    if task_key == "stimulus_family_grouped":
        if dataset_key == "000021":
            return [
                AllenTaskSpec(
                    name="stimulus_family_grouped",
                    task_kind="stimulus",
                    stimulus_group_map=dict(GROUPED_STIMULUS_FAMILY_MAP_21),
                    val_split_mode="event_stratified",
                )
            ]
        return [
            AllenTaskSpec(
                name="stimulus_family_grouped",
                task_kind="stimulus",
                stimulus_group_map=dict(GROUPED_STIMULUS_FAMILY_MAP_22),
                val_split_mode="event_stratified",
            )
        ]
    if task_key == "all":
        return [
            AllenTaskSpec(name="behavior", task_kind="behavior"),
            *resolve_task_specs(dataset_key, "stimulus_family_grouped"),
        ]
    raise ValueError(f"Unsupported task={task}")


def get_enabled_modalities(args: argparse.Namespace) -> list[str]:
    enabled = []
    if bool(args.use_pos):
        enabled.append("pos")
    if bool(getattr(args, "use_region", False)):
        enabled.append("region")
    if bool(getattr(args, "use_activity", False)):
        enabled.append("activity")
    return enabled


def finalize_allen_args(args: argparse.Namespace) -> argparse.Namespace:
    args.allen_root = Path(args.allen_root)
    args.dataset_id = canonicalize_dataset_id(args.dataset_id)
    base_runner.canonicalize_model_kind(args)
    if int(args.train_session_count) <= 0:
        args.train_session_count = 20
    args.clean_modalities = ["pos", "region", "activity"]
    args.enabled_modalities = get_enabled_modalities(args)
    if not args.allen_root.exists():
        raise FileNotFoundError(f"allen_root not found: {args.allen_root}")
    return args


def run_task_allen(
    *,
    task_spec: AllenTaskSpec,
    bundles: dict[int, AllenSessionBundle],
    train_session_ids: list[int],
    heldout_session_ids: list[int],
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> dict:
    task_dir = run_dir / str(task_spec.name)
    task_dir.mkdir(parents=True, exist_ok=True)
    base_contexts, stimulus_label_map = build_base_contexts_for_task(
        bundles,
        args=args,
        task_spec=task_spec,
    )
    num_classes = infer_num_classes_for_task(
        task_spec=task_spec,
        stimulus_label_map=stimulus_label_map,
        override=int(args.stimulus_num_classes),
    )
    if len(heldout_session_ids) <= 0:
        raise ValueError(f"Task {task_spec.name} has no held-out sessions to evaluate.")

    source_ctxs, _ = build_case_contexts(
        base_contexts,
        source_fish_ids=[int(fid) for fid in train_session_ids],
        target_fish_id=int(heldout_session_ids[0]),
        static_normalize_mode=str(args.static_normalize_mode),
    )
    task_seed = int(args.seed) + 1000 + sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(task_spec.name)))
    split_mode = str(getattr(task_spec, "val_split_mode", "tail")).strip().lower()
    if split_mode == "event_stratified" and str(task_spec.task_kind) == "stimulus":
        split = build_event_stratified_split_allen(
            source_ctxs,
            val_frac=float(args.val_frac),
            split_seed=int(args.split_seed),
        )
    else:
        split = build_tail_split_allen(
            source_ctxs,
            task_kind=str(task_spec.task_kind),
            val_frac=float(args.val_frac),
        )
    behavior_normalizer = (
        build_behavior_normalizer(
            source_ctxs,
            split["train"],
            mode=str(args.behavior_target_normalization),
        )
        if str(task_spec.task_kind) == "behavior"
        else None
    )
    best_model, split_summary = base_runner.fit_single_split(
        split_dir=task_dir / "train_val",
        source_ctxs=source_ctxs,
        train_index_map=split["train"],
        val_index_map=split["val"],
        task=str(task_spec.task_kind),
        args=args,
        out_dim=1 if str(task_spec.task_kind) == "behavior" else int(num_classes),
        device=device,
        case_seed=int(task_seed),
        num_classes=num_classes,
        behavior_normalizer=behavior_normalizer,
        split_mode=split_mode,
    )

    case_summaries: list[dict] = []
    for heldout_session_id in heldout_session_ids:
        case_dir = task_dir / f"heldout_session_{int(heldout_session_id)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        _, target_ctx = build_case_contexts(
            base_contexts,
            source_fish_ids=[int(fid) for fid in train_session_ids],
            target_fish_id=int(heldout_session_id),
            static_normalize_mode=str(args.static_normalize_mode),
        )
        heldout_subset_list = build_subset_ensemble(
            target_ctx,
            requested=int(args.eval_neurons),
            num_subsets=int(args.eval_ensemble_subsets),
            seed=int(task_seed) + 900000 + int(heldout_session_id),
        )
        heldout_subset_stack = np.stack(heldout_subset_list, axis=0)
        np.save(case_dir / "heldout_eval_neuron_idx.npy", heldout_subset_stack[0])
        np.savez(case_dir / "heldout_eval_neuron_subsets.npz", neuron_idx_subsets=heldout_subset_stack)
        transfer_metrics = evaluate_heldout_allen(
            best_model,
            target_ctx=target_ctx,
            task_spec=task_spec,
            batch_size=int(args.eval_batch_size),
            device=device,
            out_dir=case_dir / "heldout_eval",
            num_classes=num_classes,
            train_session_ids=[int(fid) for fid in train_session_ids],
            neuron_subsets=heldout_subset_list,
            behavior_normalizer=behavior_normalizer,
        )
        case_summary = {
            "task": str(task_spec.task_kind),
            "task_name": str(task_spec.name),
            "heldout_session_id": int(heldout_session_id),
            "train_session_ids": [int(fid) for fid in train_session_ids],
            "train_sampled_neurons_requested": int(args.sampled_neurons),
            "eval_sampled_neurons": int(heldout_subset_stack.shape[1]),
            "eval_ensemble_subsets": int(heldout_subset_stack.shape[0]),
            "behavior_target_normalization": behavior_normalizer_to_dict(behavior_normalizer),
            "stimulus_label_map": stimulus_label_map,
            "train_split": split_summary,
            "heldout_eval": transfer_metrics,
        }
        save_json(case_dir / "case_summary.json", case_summary)
        case_summaries.append(case_summary)
    if str(task_spec.task_kind) == "behavior":
        metric_key = "test_corr"
        metric_values = [float(case["heldout_eval"]["test_corr"]) for case in case_summaries]
    else:
        metric_key = "bal_acc"
        metric_values = [float(case["heldout_eval"]["bal_acc"]) for case in case_summaries]
    summary = {
        "task": str(task_spec.task_kind),
        "task_name": str(task_spec.name),
        "metric_key": metric_key,
        "num_cases": int(len(case_summaries)),
        "metric_mean": safe_mean(metric_values),
        "metric_std": safe_std(metric_values),
        "train_session_ids": [int(fid) for fid in train_session_ids],
        "heldout_session_ids": [int(fid) for fid in heldout_session_ids],
        "stimulus_label_map": stimulus_label_map,
        "cases": case_summaries,
    }
    save_json(task_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    pack_root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "Allen-specific supervised IdentityGuidedGrouping transfer-decoding runner. "
            "Use a fixed random train/test session split within dataset 21 or 22, "
            "fit on train sessions only, and evaluate zero-shot transfer on held-out sessions."
        )
    )
    p.add_argument("--exp-name", type=str, default="identity_guided_grouping_allen")
    p.add_argument("--out-root", type=Path, default=pack_root / "results" / "identity_guided_grouping_allen")
    p.add_argument("--allen-root", type=Path, default=DEFAULT_ALLEN_ROOT)
    p.add_argument("--dataset-id", choices=["21", "22"], required=True)
    p.add_argument(
        "--task",
        choices=[
            "behavior",
            "stimulus_family_grouped",
            "all",
        ],
        default="all",
    )
    p.add_argument("--train-session-count", type=int, default=20)
    p.add_argument("--split-seed", type=int, default=41)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")

    p.add_argument("--behavior-key", type=str, default="running_speed_cm_s")
    p.add_argument("--bin-size-s", type=float, default=0.5)
    p.add_argument("--normalize-neural", action="store_true", default=True)
    p.add_argument("--no-normalize-neural", dest="normalize_neural", action="store_false")
    p.set_defaults(
        use_gene=False,
        use_atlas=False,
        use_pos=True,
        use_region=True,
        use_activity=True,
        static_normalize_mode="source",
    )
    p.add_argument(
        "--region-prefix-keep",
        type=str,
        default="",
        help="Optional comma-separated region acronym prefixes to keep at load time, e.g. VIS",
    )
    p.add_argument("--activity-feature-mode", choices=["none", "robust_stats"], default="robust_stats")
    p.add_argument("--activity-cache-dir", type=Path, default=None)
    p.add_argument("--activity-force-recompute", action="store_true")
    p.add_argument("--activity-require-cache", action="store_true", default=False)
    p.add_argument("--activity-allow-inline-compute", dest="activity_require_cache", action="store_false")
    p.add_argument("--debug-load", action="store_true")

    p.add_argument(
        "--behavior-target-normalization",
        choices=["none", "source_train_zscore", "source_session_zscore"],
        default="source_session_zscore",
    )
    p.add_argument("--behavior-loss-mode", choices=["huber", "mse"], default="huber")
    p.add_argument("--behavior-huber-delta", type=float, default=1.0)
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
    args = finalize_allen_args(parse_args())
    if (
        not bool(args.use_pos)
        and not bool(getattr(args, "use_region", False))
        and not bool(getattr(args, "use_activity", False))
    ):
        raise ValueError("At least one identity modality must be enabled.")

    device = resolve_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = f"dataset_{args.dataset_id}"
    run_dir = args.out_root / f"{args.exp_name}_{dataset_tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "run_args.json", vars(args))
    snapshot_clean_pack_entrypoint(
        run_dir=run_dir,
        argv=[str(x) for x in sys.argv],
        entry_script=Path(__file__).resolve(),
    )

    data.set_global_seed(int(args.seed), deterministic=bool(args.deterministic))
    session_info = list_usable_session_paths(args.allen_root, args.dataset_id)
    session_ids = [int(session_id) for session_id, _ in session_info]
    split_spec = build_fixed_session_split(
        session_ids,
        train_count=int(args.train_session_count),
        split_seed=int(args.split_seed),
    )
    save_json(run_dir / "session_split.json", split_spec)

    bundles: dict[int, AllenSessionBundle] = {}
    for session_id, npz_path in session_info:
        bundles[int(session_id)] = load_session_bundle(npz_path, args)

    task_specs = resolve_task_specs(args.dataset_id, args.task)
    task_summaries: dict[str, dict] = {}
    for task_spec in task_specs:
        task_summaries[str(task_spec.name)] = run_task_allen(
            task_spec=task_spec,
            bundles=bundles,
            train_session_ids=split_spec["train_session_ids"],
            heldout_session_ids=split_spec["heldout_session_ids"],
            args=args,
            run_dir=run_dir,
            device=device,
        )

    root_summary = {
        "exp_name": str(args.exp_name),
        "timestamp": timestamp,
        "device": str(device),
        "model_name": "IdentityGuidedGrouping",
        "dataset_id": str(args.dataset_id),
        "dataset_dir": str(dataset_dir_from_id(args.allen_root, args.dataset_id)),
        "usable_session_ids": [int(x) for x in session_ids],
        "train_session_ids": [int(x) for x in split_spec["train_session_ids"]],
        "heldout_session_ids": [int(x) for x in split_spec["heldout_session_ids"]],
        "clean_modalities": list(args.clean_modalities),
        "enabled_modalities": list(args.enabled_modalities),
        "checkpoint_selection": str(getattr(args, "checkpoint_selection", "best")),
        "early_stop_enabled": not bool(getattr(args, "disable_early_stop", False)),
        "val_selection_mode": str(getattr(args, "val_selection_mode", "raw")),
        "val_selection_ema_alpha": float(getattr(args, "val_selection_ema_alpha", 0.5)),
        "selection_min_epoch": int(getattr(args, "selection_min_epoch", 1)),
        "task_summaries": task_summaries,
    }
    save_json(run_dir / "summary.json", root_summary)
    print(f"[done] run_dir={run_dir}")
    print(json.dumps(root_summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

DEFAULT_SHARED_DATA_ROOT = Path("data")
DEFAULT_POSTPROCESSED_DIR = Path("data/postprocessed")


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    return value


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected a non-empty comma-separated int list, got {text!r}")
    return vals


def resolve_device(device_arg: str) -> torch.device:
    key = str(device_arg).strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_arg))


def infer_default_data_root(script_path: Path) -> Path:
    if DEFAULT_SHARED_DATA_ROOT.exists():
        return DEFAULT_SHARED_DATA_ROOT
    script_dir = Path(script_path).resolve().parent
    for base in [script_dir / "data", script_dir.parent / "data"]:
        if (base / "data_sampled_fish_full").exists():
            return base
    return script_dir.parent / "data"


def resolve_bundle_paths(
    *,
    data_root: Path,
    fish_root: Path | None,
    postprocessed_dir: Path | None,
    activity_cache_dir: Path | None,
    atlas_precomputed_dir: Path | None,
    atlas_precomputed_fallback: Path,
) -> dict[str, Path]:
    root = Path(data_root)
    default_postprocessed = root / "postprocessed"
    default_activity_cache = root / "activity_cache_robust_stats"
    default_atlas_precomputed = root / "precomputed_atlas_hcr"
    return {
        "data_root": root,
        "fish_root": Path(fish_root) if fish_root is not None else (root / "data_sampled_fish_full"),
        "postprocessed_dir": (
            Path(postprocessed_dir)
            if postprocessed_dir is not None
            else (default_postprocessed if default_postprocessed.exists() else DEFAULT_POSTPROCESSED_DIR)
        ),
        "activity_cache_dir": (
            Path(activity_cache_dir) if activity_cache_dir is not None else default_activity_cache
        ),
        "atlas_precomputed_dir": (
            Path(atlas_precomputed_dir)
            if atlas_precomputed_dir is not None
            else (default_atlas_precomputed if default_atlas_precomputed.exists() else Path(atlas_precomputed_fallback))
        ),
    }


def finalize_bundle_args(args, *, atlas_precomputed_fallback: Path):
    resolved_paths = resolve_bundle_paths(
        data_root=Path(args.data_root),
        fish_root=args.fish_root,
        postprocessed_dir=args.postprocessed_dir,
        activity_cache_dir=args.activity_cache_dir,
        atlas_precomputed_dir=args.atlas_precomputed_dir,
        atlas_precomputed_fallback=atlas_precomputed_fallback,
    )
    args.data_root = resolved_paths["data_root"]
    args.fish_root = resolved_paths["fish_root"]
    args.postprocessed_dir = resolved_paths["postprocessed_dir"]
    args.activity_cache_dir = resolved_paths["activity_cache_dir"]
    args.atlas_precomputed_dir = resolved_paths["atlas_precomputed_dir"]
    return args


def safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def safe_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanstd(arr))


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.size != y_arr.size or x_arr.size < 2:
        return float("nan")
    x0 = x_arr - x_arr.mean()
    y0 = y_arr - y_arr.mean()
    denom = math.sqrt(float(np.dot(x0, x0)) * float(np.dot(y0, y0)))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x0, y0) / denom)


def build_transfer_case_name(target_fish_id: int, source_fish_ids: list[int]) -> str:
    source_tag = "_".join(str(int(fid)) for fid in source_fish_ids)
    return f"heldout_fish_{int(target_fish_id)}_train_{source_tag}"


def save_torch_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    train_loss: float,
    val_metric: float | None,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_metric": None if val_metric is None or not np.isfinite(val_metric) else float(val_metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "meta": to_jsonable(meta),
    }
    torch.save(payload, path)

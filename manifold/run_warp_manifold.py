#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_CONDA_PYTHON = Path("__no_default_conda_python__")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_python_bin() -> str:
    if DEFAULT_CONDA_PYTHON.exists():
        return str(DEFAULT_CONDA_PYTHON)
    return "python"


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_shared_data = root / "data"
    parser = argparse.ArgumentParser(description="Run WARP IdentityGuidedGrouping manifold transfer.")
    parser.add_argument("--python", type=str, default=default_python_bin())
    parser.add_argument("--out-root", type=Path, default=root / "results" / "warp_manifold")
    parser.add_argument("--data-root", type=Path, default=default_shared_data if default_shared_data.exists() else root.parent / "data")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu-id", type=str, default="")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()
    args.passthrough = passthrough
    return args


def main() -> None:
    args = parse_args()
    root = repo_root()
    command = [
        str(args.python),
        str(root / "identity_guided_grouping_manifold.py"),
        "--exp-name",
        "WARPMan_identity_guided_grouping",
        "--out-root",
        str(args.out_root),
        "--data-root",
        str(args.data_root),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--use-gene",
        "--use-pos",
        "--use-region",
        "--use-activity",
        "--activity-feature-mode",
        "robust_stats",
        "--window-left",
        "15",
        "--window-right",
        "15",
        "--token-mlp-hidden",
        "16",
        "--slot-internal-dim",
        "32",
        "--slot-num-slots",
        "32",
        "--slot-static-hidden-dim",
        "16",
        "--slot-mixer-hidden",
        "32",
        "--pos-radius",
        "1",
        *[str(x) for x in args.passthrough],
    ]
    if args.dry_run:
        print(shlex.join(command))
        return
    env = os.environ.copy()
    if str(args.gpu_id).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id).strip()
    raise SystemExit(subprocess.call(command, cwd=str(root), env=env))


if __name__ == "__main__":
    main()

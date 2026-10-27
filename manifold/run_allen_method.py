#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run Allen IdentityGuidedGrouping manifold transfer.")
    parser.add_argument("--dataset-id", choices=["21", "22"], required=True)
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--out-root", type=Path, default=root / "results" / "allen_manifold")
    parser.add_argument("--allen-root", type=Path, default=root / "data" / "allen" / "preprocessed_data")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu-id", type=str, default="")
    parser.add_argument("--train-session-count", type=int, default=20)
    parser.add_argument("--heldout-session-count", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=41)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--sampled-neurons", type=int, default=250)
    parser.add_argument("--eval-neurons", type=int, default=250)
    parser.add_argument("--eval-ensemble-subsets", type=int, default=21)
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()
    args.passthrough = passthrough
    return args


def build_command(args: argparse.Namespace) -> list[str]:
    root = repo_root()
    command = [
        str(args.python), str(root / "identity_guided_grouping_manifold_allen.py"),
        "--dataset-id", str(args.dataset_id),
        "--exp-name", "AllenMan_identity_guided_grouping",
        "--out-root", str(Path(args.out_root)),
        "--allen-root", str(Path(args.allen_root)),
        "--device", str(args.device),
        "--train-session-count", str(int(args.train_session_count)),
        "--heldout-session-count", str(int(args.heldout_session_count)),
        "--split-seed", str(int(args.split_seed)),
        "--seed", str(int(args.seed)),
        "--sampled-neurons", str(int(args.sampled_neurons)),
        "--eval-neurons", str(int(args.eval_neurons)),
        "--eval-ensemble-subsets", str(int(args.eval_ensemble_subsets)),
    ]
    command.extend([str(x) for x in args.passthrough])
    return command


def main() -> None:
    args = parse_args()
    command = build_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return
    env = os.environ.copy()
    if str(args.gpu_id).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id).strip()
    raise SystemExit(subprocess.call(command, cwd=str(repo_root()), env=env))


if __name__ == "__main__":
    main()

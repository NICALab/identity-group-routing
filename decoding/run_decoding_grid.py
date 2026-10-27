#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


RUN_DIR_PREFIX = "[done] run_dir="
DEFAULT_CONDA_PYTHON = Path("__no_default_conda_python__")
DEFAULT_WARP_STIMULUS_POSTPROCESSED_DIR = Path("data/postprocessed")


@dataclass(frozen=True)
class Job:
    dataset: str
    task: str
    seed: int
    command: list[str]


def parse_csv_tokens(text: str) -> list[str]:
    values = [tok.strip() for tok in str(text).split(",") if tok.strip()]
    if not values:
        raise ValueError(f"Expected a non-empty comma-separated list, got {text!r}")
    return values


def parse_csv_ints(text: str) -> list[int]:
    return [int(tok) for tok in parse_csv_tokens(text)]


def default_python_bin() -> str:
    if os.environ.get("PYTHON_BIN"):
        return str(os.environ["PYTHON_BIN"])
    if DEFAULT_CONDA_PYTHON.exists():
        return str(DEFAULT_CONDA_PYTHON)
    return "python"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def append_if_path(command: list[str], flag: str, path: Path | None) -> None:
    if path is not None:
        command.extend([flag, str(Path(path))])


def base_slot_args() -> list[str]:
    return [
        "--window-left",
        "5",
        "--window-right",
        "5",
        "--token-mlp-hidden",
        "128",
        "--slot-internal-dim",
        "384",
        "--slot-static-hidden-dim",
        "192",
        "--slot-num-slots",
        "256",
        "--slot-mixer-hidden",
        "64",
        "--slot-assign-temp",
        "0.35",
        "--slot-assign-topk",
        "5",
    ]


def base_train_args() -> list[str]:
    return [
        "--batch-size",
        "64",
        "--eval-batch-size",
        "128",
        "--eval-ensemble-subsets",
        "21",
        "--max-epochs",
        "180",
        "--patience",
        "5",
        "--early-stop-min-delta",
        "1e-4",
        "--val-frac",
        "0.2",
        "--lr",
        "2e-4",
        "--weight-decay",
        "0.0",
        "--checkpoint-every",
        "10",
        "--checkpoint-selection",
        "best",
        "--val-selection-mode",
        "ema",
        "--val-selection-ema-alpha",
        "0.5",
        "--selection-min-epoch",
        "2",
    ]


def build_warp_command(args: argparse.Namespace, *, task: str, seed: int) -> list[str]:
    root = repo_root()
    command = [
        str(args.python),
        str(root / "identity_guided_grouping_decoding.py"),
        "--out-root",
        str(Path(args.results_root) / "warp"),
        "--data-root",
        str(Path(args.warp_data_root)),
        "--fish-ids",
        str(args.fish_ids),
        "--device",
        "auto",
        "--seed",
        str(int(seed)),
        "--exp-name",
        f"WARP_main_{task}_seed{int(seed)}",
        "--task",
        str(task),
        "--sampled-neurons",
        str(int(args.neurons)),
        "--eval-neurons",
        str(int(args.neurons)),
        *base_slot_args(),
        *base_train_args(),
    ]
    append_if_path(command, "--fish-root", args.warp_fish_root)
    append_if_path(command, "--postprocessed-dir", args.warp_postprocessed_dir)
    append_if_path(command, "--stimulus-postprocessed-dir", args.warp_stimulus_postprocessed_dir)
    append_if_path(command, "--activity-cache-dir", args.warp_activity_cache_dir)
    append_if_path(command, "--atlas-precomputed-dir", args.warp_atlas_precomputed_dir)
    if str(task) == "behavior":
        command.extend(["--behavior-huber-delta", "1.0"])
    else:
        command.extend(["--stimulus-class-weight-mode", "balanced"])
    if str(args.extra_args).strip():
        command.extend(shlex.split(str(args.extra_args)))
    return command


def build_allen_command(
    args: argparse.Namespace,
    *,
    dataset_id: str,
    task: str,
    seed: int,
) -> list[str]:
    root = repo_root()
    task_arg = "behavior" if str(task) == "behavior" else str(args.allen_stimulus_task)
    command = [
        str(args.python),
        str(root / "identity_guided_grouping_decoding_allen.py"),
        "--out-root",
        str(Path(args.results_root) / f"allen_d{int(dataset_id)}"),
        "--allen-root",
        str(Path(args.allen_root)),
        "--dataset-id",
        str(dataset_id),
        "--task",
        task_arg,
        "--train-session-count",
        str(int(args.train_session_count)),
        "--device",
        "auto",
        "--seed",
        str(int(seed)),
        "--split-seed",
        str(int(seed)),
        "--exp-name",
        f"AllenD{int(dataset_id)}_main_{task}_seed{int(seed)}",
        "--sampled-neurons",
        str(int(args.allen_neurons)),
        "--eval-neurons",
        str(int(args.allen_neurons)),
        *base_slot_args(),
        *base_train_args(),
        "--behavior-huber-delta",
        "1.0",
        "--stimulus-class-weight-mode",
        "balanced",
    ]
    append_if_path(command, "--activity-cache-dir", args.allen_activity_cache_dir)
    if str(args.extra_args).strip():
        command.extend(shlex.split(str(args.extra_args)))
    return command


def build_jobs(args: argparse.Namespace) -> tuple[list[Job], list[str]]:
    seeds = parse_csv_ints(args.seed_list)
    gpu_ids = parse_csv_tokens(args.gpu_ids)
    datasets = ["warp", "allen"] if str(args.dataset) == "both" else [str(args.dataset)]
    jobs: list[Job] = []
    for dataset in datasets:
        for task in ("behavior", "stimulus"):
            for seed in seeds:
                if dataset == "warp":
                    jobs.append(
                        Job(
                            dataset="warp",
                            task=task,
                            seed=int(seed),
                            command=build_warp_command(args, task=task, seed=int(seed)),
                        )
                    )
                else:
                    for dataset_id in parse_csv_tokens(args.allen_datasets):
                        jobs.append(
                            Job(
                                dataset=f"allen_d{int(dataset_id)}",
                                task=task,
                                seed=int(seed),
                                command=build_allen_command(
                                    args,
                                    dataset_id=dataset_id,
                                    task=task,
                                    seed=int(seed),
                                ),
                            )
                        )
    return jobs, gpu_ids


def parse_run_dir(stdout_text: str) -> str | None:
    for line in stdout_text.splitlines():
        if line.startswith(RUN_DIR_PREFIX):
            return line[len(RUN_DIR_PREFIX) :].strip()
    return None


def read_summary_rows(job: Job, run_dir: Path) -> list[dict]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    task_map = summary.get("tasks") or summary.get("task_summaries") or {}
    rows: list[dict] = []
    for task_name, task_summary in task_map.items():
        metric_key = task_summary.get("metric_key")
        metric_mean = task_summary.get("metric_mean")
        if metric_key is None or metric_mean is None:
            continue
        rows.append(
            {
                "dataset": job.dataset,
                "requested_task": job.task,
                "task_name": str(task_name),
                "seed": int(job.seed),
                "metric_key": str(metric_key),
                "metric_mean": float(metric_mean),
                "metric_std_within_run": task_summary.get("metric_std"),
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
            }
        )
    return rows


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def worker_loop(
    *,
    worker_name: str,
    visible_device: str | None,
    jobs_q: queue.Queue[Job],
    run_root: Path,
    records: list[dict],
    metric_rows: list[dict],
    lock: threading.Lock,
) -> None:
    while True:
        try:
            job = jobs_q.get_nowait()
        except queue.Empty:
            return
        job_log = run_root / "job_logs" / f"{worker_name}_{job.dataset}_{job.task}_seed{job.seed}.log"
        job_log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if visible_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(visible_device)
        started_at = datetime.now().isoformat()
        proc = subprocess.run(
            job.command,
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        log_text = proc.stdout
        if proc.stderr:
            log_text += ("\n" if log_text and not log_text.endswith("\n") else "") + proc.stderr
        job_log.write_text(log_text, encoding="utf-8")
        run_dir_text = parse_run_dir(proc.stdout)
        record = {
            "dataset": job.dataset,
            "task": job.task,
            "seed": int(job.seed),
            "return_code": int(proc.returncode),
            "run_dir": run_dir_text,
            "log_path": str(job_log),
            "visible_device": visible_device,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "command": job.command,
        }
        rows = []
        if proc.returncode == 0 and run_dir_text is not None:
            rows = read_summary_rows(job, Path(run_dir_text))
        with lock:
            records.append(record)
            metric_rows.extend(rows)
            write_json(run_root / "job_results.json", records)
            write_json(run_root / "main_results.json", metric_rows)
            if metric_rows:
                write_csv(run_root / "main_results.csv", metric_rows)
        jobs_q.task_done()


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_shared_data = root / "data"
    default_warp_data = default_shared_data if default_shared_data.exists() else root.parent / "data"
    parser = argparse.ArgumentParser(description="Run main IdentityGuidedGrouping decoding experiments.")
    parser.add_argument("--python", type=str, default=default_python_bin())
    parser.add_argument("--dataset", choices=["warp", "allen", "both"], default="warp")
    parser.add_argument("--results-root", type=Path, default=root / "results" / "decoding")
    parser.add_argument("--runs-root", type=Path, default=root / "runs")
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--seed-list", type=str, default="41,42,43,44,45")
    parser.add_argument("--extra-args", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--warp-data-root", type=Path, default=default_warp_data)
    parser.add_argument("--warp-fish-root", type=Path, default=None)
    parser.add_argument("--warp-postprocessed-dir", type=Path, default=None)
    parser.add_argument(
        "--warp-stimulus-postprocessed-dir",
        type=Path,
        default=DEFAULT_WARP_STIMULUS_POSTPROCESSED_DIR if DEFAULT_WARP_STIMULUS_POSTPROCESSED_DIR.exists() else None,
    )
    parser.add_argument("--warp-activity-cache-dir", type=Path, default=None)
    parser.add_argument("--warp-atlas-precomputed-dir", type=Path, default=None)
    parser.add_argument("--fish-ids", type=str, default="59,63,71")
    parser.add_argument("--neurons", type=int, default=3000)

    parser.add_argument("--allen-root", type=Path, default=root / "data" / "allen" / "preprocessed_data")
    parser.add_argument("--allen-datasets", type=str, default="21,22")
    parser.add_argument("--allen-stimulus-task", type=str, default="stimulus_family_grouped")
    parser.add_argument("--allen-activity-cache-dir", type=Path, default=None)
    parser.add_argument("--train-session-count", type=int, default=20)
    parser.add_argument("--allen-neurons", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs, gpu_ids = build_jobs(args)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job.command))
        return

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.runs_root) / f"{args.dataset}_main_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(
        run_root / "manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "dataset": str(args.dataset),
            "seeds": parse_csv_ints(args.seed_list),
            "gpu_ids": gpu_ids,
            "num_jobs": len(jobs),
            "results_root": str(args.results_root),
        },
    )

    jobs_q: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        jobs_q.put(job)

    records: list[dict] = []
    metric_rows: list[dict] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []
    for worker_idx, gpu_id in enumerate(gpu_ids):
        visible_device = None if str(gpu_id).lower() in {"cpu", "none"} else str(gpu_id)
        thread = threading.Thread(
            target=worker_loop,
            kwargs={
                "worker_name": f"worker{worker_idx}",
                "visible_device": visible_device,
                "jobs_q": jobs_q,
                "run_root": run_root,
                "records": records,
                "metric_rows": metric_rows,
                "lock": lock,
            },
            daemon=False,
        )
        thread.start()
        threads.append(thread)
        time.sleep(0.5)

    for thread in threads:
        thread.join()

    write_json(run_root / "job_results.json", records)
    write_json(run_root / "main_results.json", metric_rows)
    if metric_rows:
        write_csv(run_root / "main_results.csv", metric_rows)
    failed = [row for row in records if int(row.get("return_code", 1)) != 0]
    print(f"[main_done] run_root={run_root}")
    print(f"[main_done] completed_jobs={len(records)} failed_jobs={len(failed)} metric_rows={len(metric_rows)}")
    if failed:
        print(f"[main_done] first_failed_log={failed[0]['log_path']}")


if __name__ == "__main__":
    main()

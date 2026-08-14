#!/usr/bin/env python3
"""Run one approved PtyChi smoke benchmark and emit auditable result files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TIMING_NAMES = (
    "io_load_time_s",
    "setup_time_s",
    "task_setup_time_s",
    "reconstruction_run_time_s",
    "save_time_s",
    "total_time_s",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--app-root", type=Path, default=Path("/home/zhong.zheng/pty-chi"))
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    root = args.bundle_root.expanduser().resolve()
    app = args.app_root.expanduser().resolve()
    plan_path = root / "experiment_plan.json"
    dataset_manifest_path = root / "datasets" / "smoke" / "manifest.json"
    plan = json.loads(plan_path.read_text())
    dataset = json.loads(dataset_manifest_path.read_text())
    experiment = plan["experiment"]
    run_id = experiment["run_id"]
    run_dir = root / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    power_path = run_dir / "power.csv"
    result_path = run_dir / "result.json"

    command = [
        args.python_bin,
        str(app / "scripts" / "run_ptychi.py"),
        "--dataset", dataset["dataset_stem"],
        "--algorithm", experiment["algorithm"],
        "--epochs", str(experiment["iterations"]),
        "--batch-size", str(experiment["requested_batch_size"]),
        "--device", experiment["device"],
        "--no-save-output",
        "--power-csv", str(power_path),
        "--monitor-script", "/home/zhong.zheng/PtychoPINN/scripts/monitor_gpu_power.py",
        "--vendor", "nvidia",
        "--devices", "0",
        "--interval", "0.2",
        "--power-label", run_id,
    ]
    command_json = json.dumps(command, separators=(",", ":"))
    started_at = utc_now()
    child_environment = os.environ.copy()
    application_source = str(app / "src")
    existing_pythonpath = child_environment.get("PYTHONPATH", "")
    child_environment["PYTHONPATH"] = (
        application_source
        if not existing_pythonpath
        else application_source + os.pathsep + existing_pythonpath
    )
    process = subprocess.run(
        command,
        cwd=app,
        env=child_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    finished_at = utc_now()
    combined = process.stdout + ("\n" + process.stderr if process.stderr else "")
    log_path.write_text(combined)
    timings = {}
    for name in TIMING_NAMES:
        matches = re.findall(rf"^{name}:\s*([0-9.eE+-]+)", combined, re.MULTILINE)
        timings[name] = float(matches[-1]) if matches else None

    try:
        git_commit = subprocess.run(
            ["git", "-C", str(app), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        git_commit = None
    environment = {
        "hostname": subprocess.run(
            ["hostname"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "python_bin": args.python_bin,
        "python_version": sys.version,
        "app_root": str(app),
        "app_git_commit": git_commit,
        "cobalt_jobid": os.environ.get("COBALT_JOBID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(root / "results" / "environment.json", environment)
    result = {
        "schema_version": "0.1",
        "run_id": run_id,
        "status": "completed" if process.returncode == 0 else "failed",
        "return_code": process.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "command": command,
        "command_sha256": hashlib.sha256(command_json.encode()).hexdigest(),
        "plan_sha256": sha256(plan_path),
        "dataset_manifest_sha256": sha256(dataset_manifest_path),
        "timings": timings,
        "log_path": str(log_path),
        "power_trace_path": str(power_path),
        "power_trace_exists": power_path.is_file(),
    }
    write_json(result_path, result)
    completion = {
        "schema_version": "0.1",
        "plan_id": plan["plan_id"],
        "status": "completed" if process.returncode == 0 else "failed",
        "planned_runs": 1,
        "completed_runs": int(process.returncode == 0),
        "failed_runs": int(process.returncode != 0),
        "result": str(result_path),
        "finished_at": finished_at,
    }
    write_json(root / "results" / "completion_manifest.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())

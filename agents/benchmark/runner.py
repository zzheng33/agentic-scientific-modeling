"""Deterministic benchmark command generation and controlled local execution."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TIMING_PATTERNS = {
    name: re.compile(rf"^{name}:\s*([0-9.eE+-]+)", re.MULTILINE)
    for name in (
        "io_load_time_s",
        "setup_time_s",
        "task_setup_time_s",
        "reconstruction_run_time_s",
        "save_time_s",
        "total_time_s",
    )
}
COMMAND_FIELDS = [
    "run_id",
    "algorithm_group_id",
    "point_id",
    "hardware_id",
    "repetition",
    "scan_point_count",
    "detector_height",
    "detector_width",
    "num_epochs",
    "batch_size",
    "dataset_id",
    "dataset_stem",
    "log_path",
    "power_trace_path",
    "command_sha256",
    "argv_json",
    "command",
]
MEASUREMENT_FIELDS = [
    "run_id",
    "status",
    "return_code",
    "started_at",
    "finished_at",
    "wall_duration_s",
    "command_sha256",
    "algorithm_group_id",
    "point_id",
    "hardware_id",
    "repetition",
    "scan_point_count",
    "detector_height",
    "detector_width",
    "num_epochs",
    "batch_size",
    "dataset_id",
    "io_load_time_s",
    "setup_time_s",
    "task_setup_time_s",
    "reconstruction_run_time_s",
    "save_time_s",
    "total_time_s",
    "power_trace_path",
    "log_path",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().resolve().open("rb") as stream:
        return tomllib.load(stream)


def _sha256_if_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_config(config_path: str | Path | None, application_root: str | Path) -> dict[str, Any]:
    document = _read_config(config_path)
    configured = document.get("benchmark", {})
    application = Path(application_root).expanduser().resolve()
    runner_script = Path(
        configured.get("runner_script", application / "scripts" / "run_ptychi.py")
    ).expanduser().resolve()
    python_executable = str(configured.get("python_executable", sys.executable))
    monitor_script = Path(
        configured.get(
            "monitor_script",
            "/home/zzhong/PtychoPINN/scripts/monitor_gpu_power.py",
        )
    ).expanduser().resolve()
    return {
        "runner_script": str(runner_script),
        "runner_script_sha256": _sha256_if_file(runner_script),
        "python_executable": python_executable,
        "monitor_script": str(monitor_script),
        "monitor_script_sha256": _sha256_if_file(monitor_script),
        "device": str(configured.get("device", "cuda")),
        "vendor": str(configured.get("vendor", "nvidia")),
        "devices": str(configured.get("devices", "0")),
        "power_interval_s": float(configured.get("power_interval_s", 0.2)),
        "continue_on_error": bool(configured.get("continue_on_error", True)),
    }


def _dataset_by_point(dataset_manifest: dict[str, Any], run_dir: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    root = run_dir / "datasets" / "pty-chi"
    for item in dataset_manifest.get("datasets", []):
        spec = item["dataset"]
        dataset_id = str(spec["dataset_id"])
        stem = root / dataset_id / str(item.get("dataset_stem", "dataset"))
        for point_id in spec.get("point_ids", []):
            if point_id in mapping:
                raise ValueError(f"Point is assigned to multiple datasets: {point_id}")
            mapping[str(point_id)] = {
                "dataset_id": dataset_id,
                "dataset_stem": str(stem),
            }
    return mapping


def _parse_detector_shape(value: str) -> tuple[int, int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise ValueError(f"Invalid detector_shape in matrix: {value}")
    return int(parsed[0]), int(parsed[1])


def _preflight(configuration: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for key in ("runner_script", "monitor_script"):
        path = Path(configuration[key])
        if not path.is_file():
            issues.append({"severity": "ERROR", "code": f"MISSING_{key.upper()}", "message": str(path)})
    python = Path(configuration["python_executable"]).expanduser()
    if not python.is_file():
        issues.append({"severity": "ERROR", "code": "MISSING_PYTHON", "message": str(python)})
    else:
        result = subprocess.run(
            [str(python), "-c", "import torch, ptychi"],
            cwd=Path(configuration["runner_script"]).parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "PTYCHI_ENVIRONMENT_UNAVAILABLE",
                    "message": (
                        f"{python} cannot import torch and ptychi; configure "
                        "benchmark.python_executable before execution"
                    ),
                }
            )
    return issues


def build_benchmark_commands(
    matrix_path: str | Path,
    dataset_manifest: dict[str, Any],
    run_dir: str | Path,
    configuration: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    run_root = Path(run_dir).expanduser().resolve()
    mapping = _dataset_by_point(dataset_manifest, run_root)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with Path(matrix_path).open(newline="", encoding="utf-8") as stream:
        for planned in csv.DictReader(stream):
            run_id = str(planned["run_id"])
            if run_id in seen:
                raise ValueError(f"Duplicate run_id in matrix: {run_id}")
            seen.add(run_id)
            point_id = str(planned["point_id"])
            if point_id not in mapping:
                raise ValueError(f"No generated dataset for point: {point_id}")
            height, width = _parse_detector_shape(planned["detector_shape"])
            output_dir = run_root / "benchmark" / "runs" / run_id
            log_path = output_dir / "run.log"
            power_path = output_dir / "power.csv"
            dataset = mapping[point_id]
            argv = [
                configuration["python_executable"],
                configuration["runner_script"],
                "--dataset",
                dataset["dataset_stem"],
                "--algorithm",
                planned["algorithm_group_id"],
                "--epochs",
                planned["num_epochs"],
                "--batch-size",
                planned["batch_size"],
                "--device",
                configuration["device"],
                "--no-save-output",
                "--power-csv",
                str(power_path),
                "--monitor-script",
                configuration["monitor_script"],
                "--vendor",
                configuration["vendor"],
                "--devices",
                configuration["devices"],
                "--interval",
                str(configuration["power_interval_s"]),
                "--power-label",
                run_id,
            ]
            argv_json = json.dumps(argv, separators=(",", ":"))
            rows.append(
                {
                    "run_id": run_id,
                    "algorithm_group_id": planned["algorithm_group_id"],
                    "point_id": point_id,
                    "hardware_id": planned["hardware_id"],
                    "repetition": planned["repetition"],
                    "scan_point_count": planned["scan_point_count"],
                    "detector_height": str(height),
                    "detector_width": str(width),
                    "num_epochs": planned["num_epochs"],
                    "batch_size": planned["batch_size"],
                    "dataset_id": dataset["dataset_id"],
                    "dataset_stem": dataset["dataset_stem"],
                    "log_path": str(log_path),
                    "power_trace_path": str(power_path),
                    "command_sha256": hashlib.sha256(argv_json.encode()).hexdigest(),
                    "argv_json": argv_json,
                    "command": shlex.join(argv),
                }
            )
    issues = _preflight(configuration)
    return rows, {
        "schema_version": "0.1",
        "status": "awaiting_human_review",
        "runner": configuration,
        "run_count": len(rows),
        "unique_run_ids": len(seen),
        "dataset_count": len({row["dataset_id"] for row in rows}),
        "algorithm_groups": sorted({row["algorithm_group_id"] for row in rows}),
        "hardware_ids": sorted({row["hardware_id"] for row in rows}),
        "validation": {
            "matrix_mapping_valid": True,
            "run_ids_unique": len(rows) == len(seen),
            "execution_ready": not any(issue["severity"] == "ERROR" for issue in issues),
            "issues": issues,
        },
    }


def render_commands_csv(rows: list[dict[str, str]]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COMMAND_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _timings(text: str) -> dict[str, str]:
    values = {name: "" for name in TIMING_PATTERNS}
    for name, pattern in TIMING_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            values[name] = f"{float(matches[-1]):.9g}"
    return values


def execute_commands(
    commands_path: str | Path,
    run_dir: str | Path,
    configuration: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    issues = _preflight(configuration)
    if any(issue["severity"] == "ERROR" for issue in issues):
        raise ValueError("Benchmark execution preflight failed: " + "; ".join(i["message"] for i in issues))
    run_root = Path(run_dir).expanduser().resolve()
    measurements: list[dict[str, Any]] = []
    with Path(commands_path).open(newline="", encoding="utf-8") as stream:
        commands = list(csv.DictReader(stream))
    for command in commands:
        run_id = command["run_id"]
        command_sha256 = command.get("command_sha256") or hashlib.sha256(
            command["argv_json"].encode()
        ).hexdigest()
        log_path = Path(command["log_path"])
        result_path = run_root / "benchmark" / "runs" / run_id / "result.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("command_sha256") != command_sha256:
                raise ValueError(f"Existing result command mismatch for {run_id}")
            measurements.append(existing)
            continue
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = _utc_now()
        start = datetime.now(timezone.utc)
        process = subprocess.run(
            json.loads(command["argv_json"]),
            cwd=Path(configuration["runner_script"]).parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        finished = _utc_now()
        wall = (datetime.now(timezone.utc) - start).total_seconds()
        combined = process.stdout + ("\n" + process.stderr if process.stderr else "")
        log_path.write_text(combined, encoding="utf-8")
        row: dict[str, Any] = {
            **{key: command[key] for key in MEASUREMENT_FIELDS if key in command},
            "status": "completed" if process.returncode == 0 else "failed",
            "return_code": process.returncode,
            "started_at": started,
            "finished_at": finished,
            "wall_duration_s": f"{wall:.9g}",
            "command_sha256": command_sha256,
            **_timings(combined),
        }
        result_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        measurements.append(row)
        if process.returncode != 0 and not configuration["continue_on_error"]:
            break
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MEASUREMENT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(measurements)
    completed = sum(row["status"] == "completed" for row in measurements)
    return output.getvalue(), {
        "schema_version": "0.1",
        "status": "execution_complete" if completed == len(commands) else "execution_incomplete",
        "planned_runs": len(commands),
        "attempted_runs": len(measurements),
        "completed_runs": completed,
        "failed_runs": sum(row["status"] == "failed" for row in measurements),
        "generated_at": _utc_now(),
    }

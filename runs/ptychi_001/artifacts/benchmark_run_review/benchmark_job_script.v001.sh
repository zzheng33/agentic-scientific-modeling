#!/usr/bin/env bash
set -euo pipefail
: "${BUNDLE_ROOT:?BUNDLE_ROOT is required}"
: "${APPLICATION_ROOT:?APPLICATION_ROOT is required}"
: "${PYTHON_BIN:?PYTHON_BIN is required}"
: "${POWER_MONITOR_SCRIPT:?POWER_MONITOR_SCRIPT is required}"
: "${ACCELERATOR_DEVICE:?ACCELERATOR_DEVICE is required}"
: "${POWER_VENDOR:?POWER_VENDOR is required}"
: "${POWER_DEVICES:?POWER_DEVICES is required}"

"$BUNDLE_ROOT/dataset_generation.sh"
mkdir -p "$BUNDLE_ROOT/results/runs" "$BUNDLE_ROOT/results/warmups"
export PYTHONPATH="$APPLICATION_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

HARNESS="$BUNDLE_ROOT/results/ptychi_matrix_harness.py"
cat > "$HARNESS" <<'PY'
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

application_root = Path(os.environ["APPLICATION_ROOT"])
runner_path = application_root / "scripts" / "run_ptychi.py"
spec = importlib.util.spec_from_file_location("reviewed_run_ptychi", runner_path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

import ptychi.api as api
import ptychi.device
from ptychi.api.task import PtychographyTask
from ptychi.utils import set_default_complex_dtype

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
config = json.loads(Path(args.config).read_text(encoding="utf-8"))
metrics_path = Path(config["metrics_path"])
metrics_path.parent.mkdir(parents=True, exist_ok=True)

def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()

def emit(metrics):
    text = json.dumps(metrics, sort_keys=True, indent=2) + "\n"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to replace existing metrics file: {metrics_path}")
    metrics_path.write_text(text, encoding="utf-8")

device = config["device"]
if device == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA accelerator requested but unavailable")
    allocation_probe = torch.empty((1024,), device="cuda")
    allocation_probe.add_(1)
    torch.cuda.synchronize()
    del allocation_probe
elif device == "xpu":
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU accelerator requested but unavailable")
    ptychi.device.set_torch_accelerator_module(torch.xpu)
elif device != "cpu":
    raise ValueError(f"Unsupported accelerator device: {device}")

torch.set_default_device(device)
torch.set_default_dtype(torch.float32)
set_default_complex_dtype(torch.complex64)

algorithm_group = config["algorithm_group_id"]
algorithm = "mpie" if algorithm_group == "mpie_runner_configuration" else algorithm_group
monitor = None
metrics = {}
try:
    sync(device)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    power_path = Path(config["power_path"]) if config.get("power_path") else None
    if power_path is not None:
        monitor_script = Path(os.environ["POWER_MONITOR_SCRIPT"])
        if not monitor_script.is_file():
            raise FileNotFoundError(f"Power monitor is unavailable: {monitor_script}")
        monitor = runner.start_power_monitor(
            power_path,
            monitor_script,
            os.environ["POWER_VENDOR"],
            float(config["power_interval_s"]),
            config["run_id"],
            os.environ.get("POWER_DEVICES") or None,
            device,
        )
    total_start = time.perf_counter()
    io_start = time.perf_counter()
    dp_file, para_file = runner.resolve_dataset(config["dataset_stem"], Path("."))
    data, probe, pixel_size_m, positions_px = runner.load_converted_data(
        dp_file, para_file, center_positions=True, scale_probe=True
    )
    io_load_time_s = time.perf_counter() - io_start

    setup_start = time.perf_counter()
    options = runner.make_options(
        algorithm,
        data,
        probe,
        pixel_size_m,
        positions_px,
        int(config["num_epochs"]),
        int(config["batch_size"]),
        0,
        0.1,
        0.1,
        optimize_probe=True,
    )
    object_shape = tuple(int(x) for x in config["object_shape"])
    options.object_options.initial_guess = torch.ones(object_shape, dtype=torch.complex64)
    options.reconstructor_options.batch_size = int(config["batch_size"])
    options.reconstructor_options.num_epochs = int(config["num_epochs"])
    options.reconstructor_options.allow_nondeterministic_algorithms = False
    options.reconstructor_options.random_seed = 20260818
    options.reconstructor_options.default_device = api.Devices.CPU if device == "cpu" else api.Devices.GPU
    options.reconstructor_options.forward_model_options.low_memory_mode = False
    options.reconstructor_options.forward_model_options.pad_for_shift = 4
    options.reconstructor_options.forward_model_options.diffraction_pattern_blur_sigma = None
    options.data_options.save_data_on_device = bool(config["data_residency"])
    options.data_options.free_space_propagation_distance_m = np.inf
    options.data_options.wavelength_m = 1.0e-9
    options.data_options.fft_shift = True
    if algorithm == "dm":
        options.reconstructor_options.chunk_length = int(config["dm_chunk_length"])
    if algorithm == "lsqml":
        options.reconstructor_options.rescale_probe_intensity_in_first_epoch = True
    if algorithm == "bh":
        options.reconstructor_options.method = "GD"
    setup_time_s = time.perf_counter() - setup_start

    task_setup_start = time.perf_counter()
    task = PtychographyTask(options)
    task_setup_time_s = time.perf_counter() - task_setup_start

    sync(device)
    run_start = time.perf_counter()
    task.run()
    sync(device)
    reconstruction_run_time_s = time.perf_counter() - run_start

    save_start = time.perf_counter()
    reconstruction = task.get_data_to_cpu("object", as_numpy=True)[0]
    probe_out = task.get_data_to_cpu("probe", as_numpy=True)
    sync(device)
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing reconstruction: {output_path}")
    np.savez(
        output_path,
        object=reconstruction,
        probe=probe_out,
        position_y_px=positions_px[:, 0],
        position_x_px=positions_px[:, 1],
        dp_file=str(dp_file),
        para_file=str(para_file),
        algorithm=algorithm_group,
        io_load_time_s=io_load_time_s,
        setup_time_s=setup_time_s,
        task_setup_time_s=task_setup_time_s,
        reconstruction_run_time_s=reconstruction_run_time_s,
    )
    save_time_s = time.perf_counter() - save_start
    sync(device)
    total_time_s = time.perf_counter() - total_start
    metrics = {
        "io_load_time_s": io_load_time_s,
        "setup_time_s": setup_time_s,
        "task_setup_time_s": task_setup_time_s,
        "reconstruction_run_time_s": reconstruction_run_time_s,
        "save_time_s": save_time_s,
        "total_time_s": total_time_s,
        "effective_batch_size": int(options.reconstructor_options.batch_size),
        "effective_dm_chunk_length": int(options.reconstructor_options.chunk_length) if algorithm == "dm" else None,
        "data_residency": bool(options.data_options.save_data_on_device),
        "lsqml_rescale_probe_intensity_in_first_epoch": True if algorithm == "lsqml" else None,
        "bh_method": "GD" if algorithm == "bh" else None,
        "peak_accelerator_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
        "peak_accelerator_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device == "cuda" else None,
        "output_path": str(output_path),
    }
    print("io_load_time_s: %.9f" % io_load_time_s, flush=True)
    print("setup_time_s: %.9f" % setup_time_s, flush=True)
    print("task_setup_time_s: %.9f" % task_setup_time_s, flush=True)
    print("reconstruction_run_time_s: %.9f" % reconstruction_run_time_s, flush=True)
    print("save_time_s: %.9f" % save_time_s, flush=True)
    print("total_time_s: %.9f" % total_time_s, flush=True)
finally:
    runner.stop_power_monitor(monitor)

emit(metrics)
PY

export APPLICATION_ROOT BUNDLE_ROOT PYTHON_BIN POWER_MONITOR_SCRIPT ACCELERATOR_DEVICE POWER_VENDOR POWER_DEVICES
"$PYTHON_BIN" - "$BUNDLE_ROOT/experiment_matrix.csv" "$BUNDLE_ROOT/datasets/dataset_manifest.json" "$BUNDLE_ROOT/results" "$HARNESS" <<'PY'
import ast
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

matrix_path = Path(sys.argv[1])
dataset_manifest_path = Path(sys.argv[2])
results_root = Path(sys.argv[3])
harness_path = Path(sys.argv[4])
python_bin = os.environ["PYTHON_BIN"]
device = os.environ["ACCELERATOR_DEVICE"]
power_interval_s = 0.2

header = "run_id,status,return_code,started_at,finished_at,wall_duration_s,command_sha256,algorithm_group_id,point_id,accelerator,repetition,scan_point_count,detector_height,detector_width,num_epochs,batch_size,dataset_id,io_load_time_s,setup_time_s,task_setup_time_s,reconstruction_run_time_s,save_time_s,total_time_s,power_trace_path,log_path".split(",")
measurements_path = results_root / "measurements.csv"
completion_path = results_root / "completion_manifest.json"


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_shape(text, length):
    value = ast.literal_eval(text)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"Invalid shape: {text}")
    return [int(x) for x in value]


def parse_bool(text):
    value = str(text).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Invalid boolean: {text}")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def relative(path):
    try:
        return str(path.relative_to(Path(os.environ["BUNDLE_ROOT"])))
    except ValueError:
        return str(path)


def write_json_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")


def run_config(row, manifest, run_dir, power_enabled):
    point = row["point_id"]
    dataset_id = manifest["point_to_dataset"][point]
    dataset = manifest["datasets"][dataset_id]
    return {
        "run_id": row["run_id"],
        "algorithm_group_id": row["algorithm_group_id"],
        "point_id": point,
        "dataset_id": dataset_id,
        "dataset_stem": dataset["stem"],
        "num_epochs": int(row["num_epochs"]),
        "batch_size": int(row["batch_size"]),
        "dm_chunk_length": int(row["dm_chunk_length"]),
        "object_shape": parse_shape(row["object_shape"], 3),
        "data_residency": parse_bool(row["data_residency"]),
        "device": device,
        "power_interval_s": power_interval_s,
        "power_path": str(run_dir / "power.csv") if power_enabled else None,
        "output_path": str(run_dir / "reconstruction.npz"),
        "metrics_path": str(run_dir / "application_metrics.json"),
    }


def execute(config, run_dir, log_path):
    config_path = run_dir / "config.json"
    write_json_exclusive(config_path, config)
    descriptor = {
        "interface": "PtychographyTask via reviewed harness",
        "harness_sha256": sha256_file(harness_path),
        "config": {k: v for k, v in config.items() if k not in {"metrics_path", "power_path", "output_path"}},
        "output_enabled": True,
        "power_enabled": bool(config.get("power_path")),
    }
    command_sha = canonical_hash(descriptor)
    started_at = utc_now()
    start = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        log.write("command_descriptor=" + json.dumps(descriptor, sort_keys=True) + "\n")
        log.flush()
        proc = subprocess.run(
            [python_bin, str(harness_path), "--config", str(config_path)],
            cwd=os.environ["APPLICATION_ROOT"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )
    wall = time.monotonic() - start
    finished_at = utc_now()
    return proc.returncode, started_at, finished_at, wall, command_sha

manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
with matrix_path.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
if len(rows) != 96:
    raise ValueError(f"Expected 96 approved measured rows, found {len(rows)}")
if completion_path.exists() or measurements_path.exists():
    raise FileExistsError("Completion or measurements artifact already exists; allow_rerun is false")
if sha256_file(matrix_path) != manifest["matrix_sha256"]:
    raise ValueError("Dataset manifest was not generated from this experiment matrix")

seen_ids = set()
measurement_rows = []
warmed = set()
for row in rows:
    run_id = row["run_id"]
    if run_id in seen_ids:
        raise ValueError(f"Duplicate run_id: {run_id}")
    seen_ids.add(run_id)
    combo = (row["algorithm_group_id"], row["point_id"], row["accelerator"])
    if combo not in warmed:
        warm_dir = results_root / "warmups" / (row["algorithm_group_id"] + "-" + row["point_id"] + "-" + row["accelerator"])
        warm_dir.mkdir(parents=True, exist_ok=False)
        warm_row = dict(row)
        warm_row["run_id"] = "warmup-" + row["algorithm_group_id"] + "-" + row["point_id"]
        warm_config = run_config(warm_row, manifest, warm_dir, False)
        warm_rc, warm_started, warm_finished, warm_wall, warm_sha = execute(warm_config, warm_dir, warm_dir / "run.log")
        write_json_exclusive(warm_dir / "result.json", {
            "run_id": warm_config["run_id"],
            "warmup": True,
            "status": "success" if warm_rc == 0 else "failed",
            "return_code": warm_rc,
            "started_at": warm_started,
            "finished_at": warm_finished,
            "wall_duration_s": warm_wall,
            "command_sha256": warm_sha,
        })
        warmed.add(combo)

    run_dir = results_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    power_path = run_dir / "power.csv"
    config = run_config(row, manifest, run_dir, device != "cpu")
    rc, started_at, finished_at, wall, command_sha = execute(config, run_dir, log_path)
    metrics_path = run_dir / "application_metrics.json"
    metrics = {}
    if rc == 0:
        if not metrics_path.is_file():
            rc = 70
        else:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if config["power_path"] and (not power_path.is_file() or power_path.stat().st_size == 0):
            rc = 71
    status = "success" if rc == 0 else "failed"
    detector = parse_shape(row["detector_shape"], 2)
    dataset_id = manifest["point_to_dataset"][row["point_id"]]
    result = {
        "run_id": run_id,
        "status": status,
        "return_code": rc,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_duration_s": wall,
        "command_sha256": command_sha,
        "algorithm_group_id": row["algorithm_group_id"],
        "point_id": row["point_id"],
        "accelerator": row["accelerator"],
        "repetition": int(row["repetition"]),
        "dataset_id": dataset_id,
        "application_metrics": metrics,
        "power_trace_path": relative(power_path) if power_path.is_file() else "",
        "log_path": relative(log_path),
    }
    write_json_exclusive(run_dir / "result.json", result)
    measurement_rows.append({
        "run_id": run_id,
        "status": status,
        "return_code": rc,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_duration_s": f"{wall:.9f}",
        "command_sha256": command_sha,
        "algorithm_group_id": row["algorithm_group_id"],
        "point_id": row["point_id"],
        "accelerator": row["accelerator"],
        "repetition": row["repetition"],
        "scan_point_count": row["n_scan_points"],
        "detector_height": detector[0],
        "detector_width": detector[1],
        "num_epochs": row["num_epochs"],
        "batch_size": row["batch_size"],
        "dataset_id": dataset_id,
        "io_load_time_s": metrics.get("io_load_time_s", ""),
        "setup_time_s": metrics.get("setup_time_s", ""),
        "task_setup_time_s": metrics.get("task_setup_time_s", ""),
        "reconstruction_run_time_s": metrics.get("reconstruction_run_time_s", ""),
        "save_time_s": metrics.get("save_time_s", ""),
        "total_time_s": metrics.get("total_time_s", ""),
        "power_trace_path": relative(power_path) if power_path.is_file() else "",
        "log_path": relative(log_path),
    })

with measurements_path.open("x", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(measurement_rows)

success_count = sum(r["status"] == "success" for r in measurement_rows)
failed_count = len(measurement_rows) - success_count
completion = {
    "schema_version": "1.0",
    "plan_id": "pty-chi-gh200-reduced-pilot-2026-08-18-v1",
    "matrix_sha256": sha256_file(matrix_path),
    "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
    "harness_sha256": sha256_file(harness_path),
    "completed_at": utc_now(),
    "expected_measured_runs": 96,
    "measured_runs": len(measurement_rows),
    "successful_runs": success_count,
    "failed_runs": failed_count,
    "warmup_combinations": len(warmed),
    "measurements_path": relative(measurements_path),
    "complete": len(measurement_rows) == 96,
}
write_json_exclusive(completion_path, completion)
if failed_count:
    print(f"Completed all rows with {failed_count} failed runs", file=sys.stderr)
    sys.exit(1)
print("Completed 96 measured runs successfully")
PY

#!/usr/bin/env bash
set -euo pipefail

: "${BUNDLE_ROOT:?BUNDLE_ROOT is required}"
: "${APPLICATION_ROOT:?APPLICATION_ROOT is required}"
: "${PYTHON_BIN:?PYTHON_BIN is required}"
: "${POWER_MONITOR_SCRIPT:?POWER_MONITOR_SCRIPT is required}"
: "${ACCELERATOR_DEVICE:?ACCELERATOR_DEVICE is required}"
: "${POWER_VENDOR:?POWER_VENDOR is required}"
: "${POWER_DEVICES:?POWER_DEVICES is required}"

DATASET_SCRIPT="$BUNDLE_ROOT/dataset_generation.sh"
if [[ ! -f "$DATASET_SCRIPT" ]]; then
  echo "Missing dataset generation script: $DATASET_SCRIPT" >&2
  exit 2
fi
bash "$DATASET_SCRIPT"

RESULTS_ROOT="$BUNDLE_ROOT/results"
mkdir -p "$RESULTS_ROOT/runs" "$RESULTS_ROOT/outputs"

cat > "$RESULTS_ROOT/approved_matrix.csv" <<'CSV'
run_id,algorithm_group_id,point_id,accelerator,repetition,n_scan_points,detector_shape,diffraction_storage_dtype,object_shape,probe_shape,num_epochs,batch_size,dm_chunk_length,batching_mode,optimized_parameter_groups,forward_propagation_configuration,forward_model_memory_options,optional_feature_schedule,compute_precision,data_residency
run-00001,pie,p001,GH200,1,64,"[64, 64]",float32,"[1, 128, 128]","[1, 1, 64, 64]",2,1,1,RANDOM,"{'object': {'optimizable': True, 'start': 0, 'stop': None, 'stride': 1}, 'probe': {'optimizable': True, 'start': 0, 'stop': None, 'stride': 1}, 'probe_positions': {'optimizable': False, 'start': 0, 'stop': None, 'stride': 1}, 'opr_mode_weights': {'optimizable': False, 'start': 0, 'stop': None, 'stride': 1}, 'slice_spacings': {'optimizable': False, 'start': 0, 'stop': None, 'stride': 1}}","{'free_space_propagation_distance_m': 'infinity', 'wavelength_m': 1e-09, 'slice_spacings_m': None}","{'low_memory_mode': False, 'pad_for_shift': 4, 'diffraction_pattern_blur_sigma': None}",{'enabled_features': []},"{'default_dtype': 'FLOAT32', 'use_double_precision_for_fft': False}",False
CSV

cat > "$RESULTS_ROOT/ptychi_measurement_harness.py" <<'PY'
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--application-root", required=True)
    p.add_argument("--dataset-stem", required=True)
    p.add_argument("--algorithm", required=True)
    p.add_argument("--epochs", required=True, type=int)
    p.add_argument("--batch-size", required=True, type=int)
    p.add_argument("--dm-chunk-length", required=True, type=int)
    p.add_argument("--device", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--data-residency", required=True, choices=("true", "false"))
    p.add_argument("--object-height", required=True, type=int)
    p.add_argument("--object-width", required=True, type=int)
    args = p.parse_args()

    app_root = Path(args.application_root).resolve()
    runner_path = app_root / "scripts" / "run_ptychi.py"
    spec = importlib.util.spec_from_file_location("reviewed_run_ptychi", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device_name = torch.cuda.get_device_name(0)
        torch.empty(1, device="cuda")
    elif args.device == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU was requested but is unavailable")
        import ptychi.device
        ptychi.device.set_torch_accelerator_module(torch.xpu)
        device_name = torch.xpu.get_device_name(0)
        torch.empty(1, device="xpu")
    else:
        raise RuntimeError("The approved accelerator run does not permit CPU fallback")

    torch.set_default_device(args.device)
    torch.set_default_dtype(torch.float32)
    runner.set_default_complex_dtype(torch.complex64)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif args.device == "xpu" and hasattr(torch.xpu, "reset_peak_memory_stats"):
        torch.xpu.reset_peak_memory_stats()
    sync(args.device)

    total_start = time.perf_counter()
    dp_file, para_file = runner.resolve_dataset(args.dataset_stem, Path("."))
    io_start = time.perf_counter()
    data, probe, pixel_size_m, positions_px = runner.load_converted_data(
        dp_file, para_file, center_positions=True, scale_probe=True
    )
    io_load_time_s = time.perf_counter() - io_start
    print(f"device_name: {device_name}", flush=True)
    print(f"data shape: {data.shape}, dtype: {data.dtype}", flush=True)
    print(f"probe shape: {tuple(probe.shape)}, dtype: {probe.dtype}", flush=True)
    print(f"io_load_time_s: {io_load_time_s:.6f}", flush=True)

    setup_start = time.perf_counter()
    options = runner.make_options(
        args.algorithm,
        data,
        probe,
        pixel_size_m,
        positions_px,
        args.epochs,
        args.batch_size,
        0,
        0.1,
        0.1,
        optimize_probe=True,
    )
    options.object_options.initial_guess = torch.ones(
        [1, args.object_height, args.object_width], dtype=torch.complex64
    )
    options.reconstructor_options.batch_size = min(args.batch_size, data.shape[0])
    options.reconstructor_options.num_epochs = args.epochs
    options.reconstructor_options.random_seed = args.seed
    options.reconstructor_options.allow_nondeterministic_algorithms = False
    options.reconstructor_options.default_device = runner.api.Devices.GPU
    options.reconstructor_options.use_double_precision_for_fft = False
    if hasattr(options.reconstructor_options, "chunk_length"):
        options.reconstructor_options.chunk_length = min(args.dm_chunk_length, data.shape[0])
    if hasattr(options.reconstructor_options, "forward_model_options"):
        fopts = options.reconstructor_options.forward_model_options
        if hasattr(fopts, "low_memory_mode"):
            fopts.low_memory_mode = False
        if hasattr(fopts, "pad_for_shift"):
            fopts.pad_for_shift = 4
        if hasattr(fopts, "diffraction_pattern_blur_sigma"):
            fopts.diffraction_pattern_blur_sigma = None
    options.data_options.wavelength_m = 1e-9
    options.data_options.free_space_propagation_distance_m = np.inf
    options.data_options.fft_shift = True
    options.data_options.save_data_on_device = args.data_residency == "true"
    setup_time_s = time.perf_counter() - setup_start
    print(f"object initial shape: {tuple(options.object_options.initial_guess.shape)}", flush=True)
    print(f"effective_batch_size: {options.reconstructor_options.batch_size}", flush=True)
    print(f"setup_time_s: {setup_time_s:.6f}", flush=True)

    task_setup_start = time.perf_counter()
    task = runner.PtychographyTask(options)
    task_setup_time_s = time.perf_counter() - task_setup_start
    print(f"task_setup_time_s: {task_setup_time_s:.6f}", flush=True)

    sync(args.device)
    run_start = time.perf_counter()
    task.run()
    sync(args.device)
    reconstruction_run_time_s = time.perf_counter() - run_start
    print(f"reconstruction_run_time_s: {reconstruction_run_time_s:.6f}", flush=True)

    save_start = time.perf_counter()
    recon = task.get_data_to_cpu("object", as_numpy=True)[0]
    probe_out = task.get_data_to_cpu("probe", as_numpy=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        object=recon,
        probe=probe_out,
        position_y_px=positions_px[:, 0],
        position_x_px=positions_px[:, 1],
        dp_file=str(dp_file),
        para_file=str(para_file),
        algorithm=args.algorithm,
        random_seed=args.seed,
        effective_batch_size=options.reconstructor_options.batch_size,
        data_residency=options.data_options.save_data_on_device,
        io_load_time_s=io_load_time_s,
        setup_time_s=setup_time_s,
        task_setup_time_s=task_setup_time_s,
        reconstruction_run_time_s=reconstruction_run_time_s,
    )
    sync(args.device)
    save_time_s = time.perf_counter() - save_start
    total_time_s = time.perf_counter() - total_start
    print(f"save_time_s: {save_time_s:.6f}", flush=True)
    print(f"total_time_s: {total_time_s:.6f}", flush=True)
    print(f"Saved reconstruction: {output}", flush=True)

    peak_allocated = None
    peak_reserved = None
    if args.device == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    elif args.device == "xpu":
        if hasattr(torch.xpu, "max_memory_allocated"):
            peak_allocated = int(torch.xpu.max_memory_allocated())
        if hasattr(torch.xpu, "max_memory_reserved"):
            peak_reserved = int(torch.xpu.max_memory_reserved())
    print("SYSTEMFLOW_RESULT_JSON:" + json.dumps({
        "device_name": device_name,
        "effective_batch_size": int(options.reconstructor_options.batch_size),
        "effective_dm_chunk_length": int(getattr(options.reconstructor_options, "chunk_length", args.dm_chunk_length)),
        "peak_accelerator_memory_allocated_bytes": peak_allocated,
        "peak_accelerator_memory_reserved_bytes": peak_reserved,
        "output_path": str(output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
PY

export SYSTEMFLOW_BUNDLE_ROOT="$BUNDLE_ROOT"
export SYSTEMFLOW_APPLICATION_ROOT="$APPLICATION_ROOT"
export SYSTEMFLOW_PYTHON_BIN="$PYTHON_BIN"
export SYSTEMFLOW_POWER_MONITOR_SCRIPT="$POWER_MONITOR_SCRIPT"
export SYSTEMFLOW_ACCELERATOR_DEVICE="$ACCELERATOR_DEVICE"
export SYSTEMFLOW_POWER_VENDOR="$POWER_VENDOR"
export SYSTEMFLOW_POWER_DEVICES="$POWER_DEVICES"

"$PYTHON_BIN" - "$RESULTS_ROOT/approved_matrix.csv" "$RESULTS_ROOT/ptychi_measurement_harness.py" <<'PY'
import ast
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

matrix_path = Path(sys.argv[1])
harness_path = Path(sys.argv[2])
bundle = Path(os.environ["SYSTEMFLOW_BUNDLE_ROOT"])
app = Path(os.environ["SYSTEMFLOW_APPLICATION_ROOT"])
python_bin = os.environ["SYSTEMFLOW_PYTHON_BIN"]
monitor_script = Path(os.environ["SYSTEMFLOW_POWER_MONITOR_SCRIPT"])
device = os.environ["SYSTEMFLOW_ACCELERATOR_DEVICE"]
vendor = os.environ["SYSTEMFLOW_POWER_VENDOR"]
power_devices = os.environ["SYSTEMFLOW_POWER_DEVICES"]
results = bundle / "results"
runs_root = results / "runs"
manifest_path = bundle / "datasets" / "dataset_manifest.json"
measurements_path = results / "measurements.csv"
completion_path = results / "completion_manifest.json"

header = "run_id,status,return_code,started_at,finished_at,wall_duration_s,command_sha256,algorithm_group_id,point_id,accelerator,repetition,scan_point_count,detector_height,detector_width,num_epochs,batch_size,dataset_id,io_load_time_s,setup_time_s,task_setup_time_s,reconstruction_run_time_s,save_time_s,total_time_s,power_trace_path,log_path".split(",")

with manifest_path.open("r", encoding="utf-8") as f:
    dataset_manifest = json.load(f)
dataset_by_id = {d["dataset_id"]: d for d in dataset_manifest["datasets"]}
point_map = dataset_manifest["point_to_dataset"]

with matrix_path.open("r", encoding="utf-8", newline="") as f:
    matrix = list(csv.DictReader(f))
if not matrix:
    raise SystemExit("Approved experiment matrix is empty")

if not monitor_script.is_file():
    raise SystemExit(f"Power monitor script does not exist: {monitor_script}")
if device not in ("cuda", "xpu"):
    raise SystemExit(f"Unsupported approved accelerator device: {device}")

def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def parse_timings(text):
    out = {}
    for key in ("io_load_time_s", "setup_time_s", "task_setup_time_s", "reconstruction_run_time_s", "save_time_s", "total_time_s"):
        matches = re.findall(rf"^{re.escape(key)}:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
        out[key] = f"{float(matches[-1]):.6f}" if matches else ""
    return out

def relative(path):
    return str(path.relative_to(bundle))

measurements = []
result_records = []
for row in matrix:
    run_id = row["run_id"]
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    power_path = run_dir / "power.csv"
    result_path = run_dir / "result.json"

    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        saved = existing.get("measurement")
        if existing.get("status") == "success" and isinstance(saved, dict) and set(header).issubset(saved):
            measurements.append({k: saved.get(k, "") for k in header})
            result_records.append(existing)
            continue
        raise SystemExit(f"Refusing to rerun non-successful existing run {run_id}; allow_rerun is false")

    detector = ast.literal_eval(row["detector_shape"])
    object_shape = ast.literal_eval(row["object_shape"])
    dataset_id = point_map[row["point_id"]]
    dataset_entry = dataset_by_id[dataset_id]
    stem = bundle / "datasets" / dataset_id
    dp_path = stem.with_name(stem.name + "_ptychodus_dp.hdf5")
    para_path = stem.with_name(stem.name + "_ptychodus_para.hdf5")
    if not dp_path.is_file() or not para_path.is_file():
        raise SystemExit(f"Dataset files are missing for {dataset_id}")
    if file_sha256(dp_path) != dataset_entry["files"]["diffraction"]["sha256"]:
        raise SystemExit(f"Diffraction hash mismatch for {dataset_id}")
    if file_sha256(para_path) != dataset_entry["files"]["parameters"]["sha256"]:
        raise SystemExit(f"Parameter hash mismatch for {dataset_id}")

    output_path = results / "outputs" / f"{run_id}.npz"
    data_residency = str(row["data_residency"]).strip().lower()
    cmd = [
        python_bin,
        str(harness_path),
        "--application-root", str(app),
        "--dataset-stem", str(stem),
        "--algorithm", row["algorithm_group_id"],
        "--epochs", row["num_epochs"],
        "--batch-size", row["batch_size"],
        "--dm-chunk-length", row["dm_chunk_length"],
        "--device", device,
        "--output", str(output_path),
        "--seed", "20260818",
        "--data-residency", data_residency,
        "--object-height", str(object_shape[1]),
        "--object-width", str(object_shape[2]),
    ]
    command_blob = json.dumps(cmd, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    command_hash = hashlib.sha256(command_blob).hexdigest()
    monitor_cmd = [
        python_bin,
        str(monitor_script),
        "--vendor", vendor,
        "--interval", "0.2",
        "--output", str(power_path),
        "--label", run_id,
    ]
    if power_devices:
        monitor_cmd.extend(["--devices", power_devices])

    started_at = utc_now()
    wall_start = time.monotonic()
    app_rc = 1
    monitor_rc = None
    monitor = None
    execution_error = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write("application_command: " + shlex.join(cmd) + "\n")
        log.write("power_monitor_command: " + shlex.join(monitor_cmd) + "\n")
        log.flush()
        try:
            monitor = subprocess.Popen(
                monitor_cmd,
                cwd=app,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            time.sleep(0.25)
            if monitor.poll() is not None:
                raise RuntimeError(f"Power monitor exited before application start with code {monitor.returncode}")
            env = os.environ.copy()
            source_path = str(app / "src")
            env["PYTHONPATH"] = source_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            process = subprocess.Popen(
                cmd,
                cwd=app,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            app_rc = process.wait()
        except Exception as exc:
            execution_error = repr(exc)
            log.write("execution_error: " + execution_error + "\n")
            log.flush()
        finally:
            if monitor is not None:
                if monitor.poll() is None:
                    monitor.terminate()
                    try:
                        monitor.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        monitor.kill()
                        monitor.wait()
                monitor_rc = monitor.returncode

    wall_duration = time.monotonic() - wall_start
    finished_at = utc_now()
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    timings = parse_timings(log_text)
    marker_matches = re.findall(r"^SYSTEMFLOW_RESULT_JSON:(\{.*\})$", log_text, re.MULTILINE)
    harness_metrics = {}
    if marker_matches:
        try:
            harness_metrics = json.loads(marker_matches[-1])
        except json.JSONDecodeError:
            harness_metrics = {}

    power_ok = power_path.is_file() and power_path.stat().st_size > 0
    output_ok = output_path.is_file() and output_path.stat().st_size > 0
    timing_ok = bool(timings["total_time_s"])
    success = app_rc == 0 and execution_error is None and power_ok and output_ok and timing_ok
    status = "success" if success else "failed"
    measurement = {
        "run_id": run_id,
        "status": status,
        "return_code": str(app_rc),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_duration_s": f"{wall_duration:.6f}",
        "command_sha256": command_hash,
        "algorithm_group_id": row["algorithm_group_id"],
        "point_id": row["point_id"],
        "accelerator": row["accelerator"],
        "repetition": row["repetition"],
        "scan_point_count": row["n_scan_points"],
        "detector_height": str(detector[0]),
        "detector_width": str(detector[1]),
        "num_epochs": row["num_epochs"],
        "batch_size": row["batch_size"],
        "dataset_id": dataset_id,
        "io_load_time_s": timings["io_load_time_s"],
        "setup_time_s": timings["setup_time_s"],
        "task_setup_time_s": timings["task_setup_time_s"],
        "reconstruction_run_time_s": timings["reconstruction_run_time_s"],
        "save_time_s": timings["save_time_s"],
        "total_time_s": timings["total_time_s"],
        "power_trace_path": relative(power_path) if power_ok else "",
        "log_path": relative(log_path),
    }
    result = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": status,
        "return_code": app_rc,
        "execution_error": execution_error,
        "monitor_return_code": monitor_rc,
        "power_trace_valid": power_ok,
        "output_valid": output_ok,
        "timing_valid": timing_ok,
        "command": cmd,
        "command_sha256": command_hash,
        "dataset_id": dataset_id,
        "dataset_hashes": {
            "diffraction_sha256": dataset_entry["files"]["diffraction"]["sha256"],
            "parameters_sha256": dataset_entry["files"]["parameters"]["sha256"],
        },
        "requested_inputs": row,
        "effective_runtime": harness_metrics,
        "measurement": measurement,
    }
    tmp = run_dir / "result.json.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, result_path)
    measurements.append(measurement)
    result_records.append(result)

measurements_tmp = results / "measurements.csv.tmp"
with measurements_tmp.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for measurement in measurements:
        writer.writerow({k: measurement.get(k, "") for k in header})
os.replace(measurements_tmp, measurements_path)

success_count = sum(r.get("status") == "success" for r in result_records)
failed_count = len(result_records) - success_count
completion = {
    "schema_version": "1.0",
    "plan_id": "pty-chi-gh200-reduced-pilot-2026-08-18-v1",
    "completed_at": utc_now(),
    "expected_run_count": len(matrix),
    "recorded_run_count": len(result_records),
    "successful_run_count": success_count,
    "failed_run_count": failed_count,
    "complete": len(result_records) == len(matrix),
    "all_successful": failed_count == 0,
    "measurements_path": relative(measurements_path),
    "measurements_sha256": file_sha256(measurements_path),
    "dataset_manifest_path": relative(manifest_path),
    "dataset_manifest_sha256": file_sha256(manifest_path),
    "run_results": [relative(runs_root / r["run_id"] / "result.json") for r in result_records],
}
completion_tmp = results / "completion_manifest.json.tmp"
with completion_tmp.open("w", encoding="utf-8") as f:
    json.dump(completion, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(completion_tmp, completion_path)

if failed_count:
    raise SystemExit(1)
PY

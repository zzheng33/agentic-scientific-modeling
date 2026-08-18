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

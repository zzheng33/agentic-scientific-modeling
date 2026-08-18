#!/usr/bin/env bash
set -euo pipefail

: "${BUNDLE_ROOT:?BUNDLE_ROOT is required}"
: "${PYTHON_BIN:?PYTHON_BIN is required}"

DATASET_ROOT="$BUNDLE_ROOT/datasets"
MANIFEST="$DATASET_ROOT/dataset_manifest.json"
mkdir -p "$DATASET_ROOT"

"$PYTHON_BIN" - "$DATASET_ROOT" "$MANIFEST" <<'PY'
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)

generator_version = "systemflow-ptychi-synthetic-v1"
seed = 20260818
dataset_id = "p001-n64-d64-o128-p64-seed20260818-v1"
stem = root / dataset_id
dp_path = stem.with_name(stem.name + "_ptychodus_dp.hdf5")
para_path = stem.with_name(stem.name + "_ptychodus_para.hdf5")

spec = {
    "dataset_id": dataset_id,
    "generator_version": generator_version,
    "seed": seed,
    "n_scan_points": 64,
    "detector_shape": [64, 64],
    "diffraction_storage_dtype": "float32",
    "object_shape": [1, 128, 128],
    "probe_shape": [1, 1, 64, 64],
    "wavelength_m": 1e-9,
    "free_space_propagation_distance_m": "infinity",
    "object_pixel_size_m": 1e-9,
    "fft_shift": True,
    "raster_shape": [8, 8],
    "raster_step_px": 8,
    "position_extent_px": [56, 56],
    "nominal_adjacent_overlap_fraction": 0.875,
    "center_positions": True,
    "scale_probe": True,
    "pad_for_shift": 4,
}

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def validate_files():
    with h5py.File(dp_path, "r") as f:
        if "dp" not in f:
            raise ValueError("diffraction file lacks /dp")
        d = f["dp"]
        if d.shape != (64, 64, 64) or d.dtype != np.dtype("float32"):
            raise ValueError(f"invalid diffraction shape/dtype: {d.shape} {d.dtype}")
        arr = d[...]
        if not np.isfinite(arr).all() or np.any(arr < 0) or not np.any(arr > 0):
            raise ValueError("diffraction values are not finite, nonnegative, and nonzero")
        if f.attrs.get("dataset_id", "") != dataset_id:
            raise ValueError("diffraction dataset_id mismatch")
    with h5py.File(para_path, "r") as f:
        required = {"probe", "object", "probe_position_y_m", "probe_position_x_m"}
        if not required.issubset(f.keys()):
            raise ValueError("parameter file lacks required datasets")
        if f["probe"].shape != (1, 64, 64) or f["probe"].dtype != np.dtype("complex64"):
            raise ValueError("invalid probe shape or dtype")
        if f["object"].shape != (1, 128, 128) or f["object"].dtype != np.dtype("complex64"):
            raise ValueError("invalid object shape or dtype")
        if f["probe_position_y_m"].shape != (64,) or f["probe_position_x_m"].shape != (64,):
            raise ValueError("invalid position shape")
        if float(f["object"].attrs["pixel_height_m"]) != spec["object_pixel_size_m"]:
            raise ValueError("object pixel size mismatch")
        if f.attrs.get("dataset_id", "") != dataset_id:
            raise ValueError("parameter dataset_id mismatch")
        probe = f["probe"][...]
        pos_y = f["probe_position_y_m"][...] / spec["object_pixel_size_m"]
        pos_x = f["probe_position_x_m"][...] / spec["object_pixel_size_m"]
        if not np.isfinite(probe).all() or float(np.sum(np.abs(probe) ** 2)) <= 0:
            raise ValueError("probe is invalid or has zero power")
        if not np.isfinite(pos_y).all() or not np.isfinite(pos_x).all():
            raise ValueError("positions are not finite")
        if len(np.unique(np.stack([pos_y, pos_x], axis=1), axis=0)) != 64:
            raise ValueError("scan positions are degenerate")
        if np.max(np.abs(pos_y)) + 32 + 4 > 64 or np.max(np.abs(pos_x)) + 32 + 4 > 64:
            raise ValueError("padded probe patches do not fit the object")

if manifest_path.exists() or dp_path.exists() or para_path.exists():
    if not (manifest_path.exists() and dp_path.exists() and para_path.exists()):
        raise SystemExit("Refusing to replace a partial existing dataset; all paired files and the manifest must exist")
    with manifest_path.open("r", encoding="utf-8") as f:
        existing = json.load(f)
    entries = existing.get("datasets", [])
    if existing.get("schema_version") != "1.0" or len(entries) != 1 or entries[0].get("specification") != spec:
        raise SystemExit("Existing dataset manifest does not match the requested specification")
    validate_files()
    expected = entries[0].get("files", {})
    if expected.get("diffraction", {}).get("sha256") != sha256_file(dp_path):
        raise SystemExit("Existing diffraction file hash does not match its manifest")
    if expected.get("parameters", {}).get("sha256") != sha256_file(para_path):
        raise SystemExit("Existing parameter file hash does not match its manifest")
    print(f"Validated existing dataset {dataset_id}")
    raise SystemExit(0)

rng = np.random.default_rng(seed)
y, x = np.mgrid[:128, :128].astype(np.float32)
yc = y - 63.5
xc = x - 63.5
amp = 0.88 + 0.08 * np.cos(xc / 11.0) * np.cos(yc / 13.0)
phase = 0.22 * np.sin(xc / 15.0) + 0.17 * np.cos(yc / 17.0)
phase += 0.025 * rng.standard_normal((128, 128), dtype=np.float32)
truth = (amp * np.exp(1j * phase)).astype(np.complex64)

py, px = np.mgrid[:64, :64].astype(np.float32)
py -= 31.5
px -= 31.5
envelope = np.exp(-0.5 * ((py / 13.0) ** 2 + (px / 13.0) ** 2))
probe_phase = 0.003 * px + 0.002 * py + 0.0005 * px * py
probe2d = (envelope * np.exp(1j * probe_phase)).astype(np.complex64)
probe2d /= np.sqrt(np.sum(np.abs(probe2d) ** 2, dtype=np.float64)).astype(np.float32)
probe = probe2d[None, :, :]

coords = np.arange(-28, 29, 8, dtype=np.float32)
positions_px = np.asarray([(yy, xx) for yy in coords for xx in coords], dtype=np.float32)
patterns = np.empty((64, 64, 64), dtype=np.float32)
for i, (yy, xx) in enumerate(positions_px):
    cy = 64 + int(yy)
    cx = 64 + int(xx)
    patch = truth[cy - 32:cy + 32, cx - 32:cx + 32]
    wave = patch * probe2d
    far = np.fft.fft2(wave, norm="ortho")
    intensity = np.abs(far) ** 2
    patterns[i] = (intensity + np.float32(1e-8)).astype(np.float32)

positions_m = positions_px.astype(np.float64) * spec["object_pixel_size_m"]

dp_tmp = root / (dp_path.name + f".tmp.{os.getpid()}")
para_tmp = root / (para_path.name + f".tmp.{os.getpid()}")
manifest_tmp = root / (manifest_path.name + f".tmp.{os.getpid()}")

with h5py.File(dp_tmp, "w") as f:
    f.create_dataset("dp", data=patterns, dtype="float32")
    f.attrs["dataset_id"] = dataset_id
    f.attrs["generator_version"] = generator_version
    f.attrs["seed"] = seed
    f.attrs["wavelength_m"] = spec["wavelength_m"]
    f.attrs["free_space_propagation_distance_m"] = np.inf
    f.attrs["fft_shift"] = True

with h5py.File(para_tmp, "w") as f:
    f.create_dataset("probe", data=probe, dtype="complex64")
    obj = f.create_dataset("object", data=truth[None, :, :], dtype="complex64")
    obj.attrs["pixel_height_m"] = spec["object_pixel_size_m"]
    obj.attrs["pixel_width_m"] = spec["object_pixel_size_m"]
    f.create_dataset("probe_position_y_m", data=positions_m[:, 0], dtype="float64")
    f.create_dataset("probe_position_x_m", data=positions_m[:, 1], dtype="float64")
    f.create_dataset("opr_weights", data=np.ones((64, 1), dtype=np.float32))
    f.attrs["dataset_id"] = dataset_id
    f.attrs["generator_version"] = generator_version
    f.attrs["seed"] = seed
    f.attrs["position_units"] = "meters"
    f.attrs["position_order"] = "y,x"
    f.attrs["raster_rows"] = 8
    f.attrs["raster_columns"] = 8
    f.attrs["raster_step_px"] = 8

os.replace(dp_tmp, dp_path)
os.replace(para_tmp, para_path)
validate_files()

entry = {
    "dataset_id": dataset_id,
    "point_ids": ["p001"],
    "specification": spec,
    "files": {
        "diffraction": {
            "path": str(dp_path.relative_to(root.parent)),
            "sha256": sha256_file(dp_path),
            "size_bytes": dp_path.stat().st_size,
        },
        "parameters": {
            "path": str(para_path.relative_to(root.parent)),
            "sha256": sha256_file(para_path),
            "size_bytes": para_path.stat().st_size,
        },
    },
    "validation": {
        "finite": True,
        "nonnegative_diffraction": True,
        "nonzero_probe_power": True,
        "unique_position_count": 64,
        "all_padded_patches_fit": True,
    },
}
manifest = {
    "schema_version": "1.0",
    "generator_version": generator_version,
    "seed": seed,
    "datasets": [entry],
    "point_to_dataset": {"p001": dataset_id},
}
with manifest_tmp.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(manifest_tmp, manifest_path)
print(f"Generated and validated dataset {dataset_id}")
PY

#!/usr/bin/env bash
set -euo pipefail
: "${BUNDLE_ROOT:?BUNDLE_ROOT is required}"
: "${APPLICATION_ROOT:?APPLICATION_ROOT is required}"
: "${PYTHON_BIN:?PYTHON_BIN is required}"

MATRIX="$BUNDLE_ROOT/experiment_matrix.csv"
DATASET_ROOT="$BUNDLE_ROOT/datasets"
MANIFEST="$DATASET_ROOT/dataset_manifest.json"

[[ -f "$MATRIX" ]] || { echo "Missing experiment matrix: $MATRIX" >&2; exit 2; }
mkdir -p "$DATASET_ROOT"
export PYTHONPATH="$APPLICATION_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - "$MATRIX" "$DATASET_ROOT" "$MANIFEST" <<'PY'
import ast
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np

matrix_path = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
seed = 20260818
generator_version = "pty-chi-pilot-generator-v1"
pixel_size_m = 1.0e-8
wavelength_m = 1.0e-9
pad_for_shift = 4


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_shape(text, length, name):
    value = ast.literal_eval(text)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"Invalid {name}: {text!r}")
    value = tuple(int(x) for x in value)
    if any(x <= 0 for x in value):
        raise ValueError(f"Nonpositive {name}: {value}")
    return value


def make_spec(row):
    detector = parse_shape(row["detector_shape"], 2, "detector_shape")
    obj = parse_shape(row["object_shape"], 3, "object_shape")
    probe = parse_shape(row["probe_shape"], 4, "probe_shape")
    spec = {
        "seed": seed,
        "generator_version": generator_version,
        "n_scan_points": int(row["n_scan_points"]),
        "detector_shape": list(detector),
        "diffraction_storage_dtype": row["diffraction_storage_dtype"],
        "object_shape": list(obj),
        "probe_shape": list(probe),
        "free_space_propagation_distance_m": "infinity",
        "wavelength_m": wavelength_m,
        "object_pixel_size_m": pixel_size_m,
        "fft_shift": True,
        "pad_for_shift": pad_for_shift,
    }
    if spec["diffraction_storage_dtype"] != "float32":
        raise ValueError("This approved pilot generator supports only float32 diffraction storage")
    n = spec["n_scan_points"]
    s, oh, ow = obj
    m_opr, m_probe, ph, pw = probe
    dh, dw = detector
    if s != 1 or m_opr != 1 or m_probe != 1:
        raise ValueError("Pilot requires S=M_opr=M_probe=1")
    if dh > ph or dw > pw:
        raise ValueError("Detector dimensions cannot exceed probe dimensions")
    if oh < ph + 2 * pad_for_shift or ow < pw + 2 * pad_for_shift:
        raise ValueError("Object is too small for probe and shift padding")
    if n <= 0:
        raise ValueError("n_scan_points must be positive")
    return spec


def dataset_id_for(spec):
    return "ds-" + hashlib.sha256(canonical_bytes(spec)).hexdigest()[:20]


def raster_positions(spec):
    n = spec["n_scan_points"]
    _, oh, ow = spec["object_shape"]
    _, _, ph, pw = spec["probe_shape"]
    ny = max(1, int(math.floor(math.sqrt(n))))
    nx = int(math.ceil(n / ny))
    while ny * nx < n:
        nx += 1
    y_extent = float(oh - ph - 2 * pad_for_shift)
    x_extent = float(ow - pw - 2 * pad_for_shift)
    ys = np.linspace(-0.5 * y_extent, 0.5 * y_extent, ny, dtype=np.float32) if ny > 1 else np.zeros(1, np.float32)
    xs = np.linspace(-0.5 * x_extent, 0.5 * x_extent, nx, dtype=np.float32) if nx > 1 else np.zeros(1, np.float32)
    pos = np.asarray([(y, x) for y in ys for x in xs], dtype=np.float32)[:n]
    pos -= pos.mean(axis=0, keepdims=True)
    step_y = float(abs(ys[1] - ys[0])) if ny > 1 else 0.0
    step_x = float(abs(xs[1] - xs[0])) if nx > 1 else 0.0
    overlap_y = 1.0 if step_y == 0 else max(0.0, 1.0 - step_y / ph)
    overlap_x = 1.0 if step_x == 0 else max(0.0, 1.0 - step_x / pw)
    return pos, {
        "raster_rows": ny,
        "raster_columns": nx,
        "step_y_px": step_y,
        "step_x_px": step_x,
        "nominal_overlap_y": overlap_y,
        "nominal_overlap_x": overlap_x,
        "coordinate_order": ["y", "x"],
        "centered": True,
    }


def synthesize(spec):
    n = spec["n_scan_points"]
    _, oh, ow = spec["object_shape"]
    _, _, ph, pw = spec["probe_shape"]
    dh, dw = spec["detector_shape"]
    identity_hash = hashlib.sha256(canonical_bytes(spec)).digest()
    local_seed = seed ^ int.from_bytes(identity_hash[:8], "little")
    rng = np.random.default_rng(local_seed)

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, oh, dtype=np.float32),
        np.linspace(-1.0, 1.0, ow, dtype=np.float32),
        indexing="ij",
    )
    amp = 0.82 + 0.12 * np.cos(3.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    phase = 0.35 * np.sin(2.0 * np.pi * xx) + 0.2 * np.cos(3.0 * np.pi * yy)
    phase += 0.02 * rng.standard_normal((oh, ow), dtype=np.float32)
    true_object = (amp * np.exp(1j * phase)).astype(np.complex64)[None, :, :]

    py, px = np.meshgrid(
        np.linspace(-1.0, 1.0, ph, dtype=np.float32),
        np.linspace(-1.0, 1.0, pw, dtype=np.float32),
        indexing="ij",
    )
    envelope = np.exp(-3.5 * (px * px + py * py)).astype(np.float32)
    probe_phase = (0.3 * px - 0.2 * py + 0.08 * px * py).astype(np.float32)
    probe = (envelope * np.exp(1j * probe_phase)).astype(np.complex64)[None, :, :]
    probe /= np.sqrt(np.sum(np.abs(probe) ** 2, dtype=np.float64)).astype(np.float32)

    positions, geometry = raster_positions(spec)
    patterns = np.empty((n, dh, dw), dtype=np.float32)
    cy0 = (ph - dh) // 2
    cx0 = (pw - dw) // 2
    for i, (y, x) in enumerate(positions):
        y0 = int(round((oh - ph) / 2.0 + float(y)))
        x0 = int(round((ow - pw) / 2.0 + float(x)))
        if y0 < pad_for_shift or x0 < pad_for_shift or y0 + ph + pad_for_shift > oh or x0 + pw + pad_for_shift > ow:
            raise ValueError(f"Generated patch {i} does not fit padded object")
        wave = true_object[0, y0:y0 + ph, x0:x0 + pw] * probe[0]
        far = np.fft.fftshift(np.fft.fft2(wave, norm="ortho"))
        intensity = np.abs(far) ** 2
        crop = intensity[cy0:cy0 + dh, cx0:cx0 + dw]
        patterns[i] = np.asarray(crop + np.float32(1.0e-7), dtype=np.float32)
    if not np.isfinite(patterns).all() or np.any(patterns < 0) or float(patterns.max()) <= 0:
        raise ValueError("Generated diffraction data failed finite/nonnegative/nonzero validation")
    return patterns, probe, true_object, positions, geometry


def inspect_files(dp_path, para_path, spec):
    n = spec["n_scan_points"]
    dh, dw = spec["detector_shape"]
    _, oh, ow = spec["object_shape"]
    _, _, ph, pw = spec["probe_shape"]
    with h5py.File(dp_path, "r") as f:
        if "dp" not in f or f["dp"].shape != (n, dh, dw) or f["dp"].dtype != np.dtype("float32"):
            raise ValueError(f"Existing diffraction file does not match specification: {dp_path}")
        data = f["dp"]
        if not np.isfinite(data[...]).all() or np.any(data[...] < 0) or float(np.max(data[...])) <= 0:
            raise ValueError(f"Invalid diffraction values in {dp_path}")
        if f.attrs.get("spec_sha256", "") != hashlib.sha256(canonical_bytes(spec)).hexdigest():
            raise ValueError(f"Specification hash mismatch in {dp_path}")
    with h5py.File(para_path, "r") as f:
        required = {"probe", "object", "probe_position_y_m", "probe_position_x_m", "synthetic_object"}
        if not required.issubset(f.keys()):
            raise ValueError(f"Missing parameter datasets in {para_path}")
        if f["probe"].shape != (1, ph, pw) or f["probe"].dtype != np.dtype("complex64"):
            raise ValueError(f"Probe mismatch in {para_path}")
        if f["synthetic_object"].shape != (1, oh, ow) or f["synthetic_object"].dtype != np.dtype("complex64"):
            raise ValueError(f"Synthetic object mismatch in {para_path}")
        if f["probe_position_y_m"].shape != (n,) or f["probe_position_x_m"].shape != (n,):
            raise ValueError(f"Position mismatch in {para_path}")
        if not np.isfinite(f["probe"][...]).all() or float(np.sum(np.abs(f["probe"][...]) ** 2)) <= 0:
            raise ValueError(f"Invalid probe in {para_path}")
        if float(f["object"].attrs["pixel_height_m"]) != pixel_size_m:
            raise ValueError(f"Pixel-size mismatch in {para_path}")
        if f.attrs.get("spec_sha256", "") != hashlib.sha256(canonical_bytes(spec)).hexdigest():
            raise ValueError(f"Specification hash mismatch in {para_path}")


def create_files(dp_path, para_path, spec):
    patterns, probe, true_object, positions, geometry = synthesize(spec)
    spec_hash = hashlib.sha256(canonical_bytes(spec)).hexdigest()
    dp_path.parent.mkdir(parents=True, exist_ok=True)
    if dp_path.exists() or para_path.exists():
        if not (dp_path.exists() and para_path.exists()):
            raise FileExistsError("Only one file of an existing dataset pair is present; refusing unsafe repair")
        inspect_files(dp_path, para_path, spec)
        return geometry
    with h5py.File(dp_path, "x") as f:
        f.create_dataset("dp", data=patterns, dtype=np.float32, chunks=(1, patterns.shape[1], patterns.shape[2]))
        f.attrs["spec_sha256"] = spec_hash
        f.attrs["generator_version"] = generator_version
        f.attrs["seed"] = seed
        f.attrs["wavelength_m"] = wavelength_m
        f.attrs["free_space_propagation_distance_m"] = "infinity"
        f.attrs["fft_shift_expected_by_application"] = True
    with h5py.File(para_path, "x") as f:
        f.create_dataset("probe", data=probe, dtype=np.complex64)
        obj_marker = f.create_dataset("object", data=np.asarray([1.0 + 0.0j], dtype=np.complex64))
        obj_marker.attrs["pixel_height_m"] = pixel_size_m
        obj_marker.attrs["pixel_width_m"] = pixel_size_m
        f.create_dataset("synthetic_object", data=true_object, dtype=np.complex64)
        f.create_dataset("probe_position_y_m", data=positions[:, 0] * pixel_size_m, dtype=np.float32)
        f.create_dataset("probe_position_x_m", data=positions[:, 1] * pixel_size_m, dtype=np.float32)
        f.create_dataset("opr_weights", data=np.ones((spec["n_scan_points"], 1), dtype=np.float32))
        f.attrs["spec_sha256"] = spec_hash
        f.attrs["generator_version"] = generator_version
        f.attrs["seed"] = seed
        f.attrs["position_units"] = "meters"
        f.attrs["position_centering"] = "application centers positions after loading"
        f.attrs["raster_rows"] = geometry["raster_rows"]
        f.attrs["raster_columns"] = geometry["raster_columns"]
        f.attrs["step_y_px"] = geometry["step_y_px"]
        f.attrs["step_x_px"] = geometry["step_x_px"]
        f.attrs["nominal_overlap_y"] = geometry["nominal_overlap_y"]
        f.attrs["nominal_overlap_x"] = geometry["nominal_overlap_x"]
    return geometry

with matrix_path.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise ValueError("Experiment matrix is empty")
required = {
    "run_id", "point_id", "n_scan_points", "detector_shape", "diffraction_storage_dtype",
    "object_shape", "probe_shape", "forward_propagation_configuration"
}
if not required.issubset(rows[0]):
    raise ValueError(f"Experiment matrix lacks required columns: {sorted(required - set(rows[0]))}")

specs = {}
point_to_dataset = {}
for row in rows:
    spec = make_spec(row)
    dataset_id = dataset_id_for(spec)
    prior = point_to_dataset.get(row["point_id"])
    if prior is not None and prior != dataset_id:
        raise ValueError(f"Point {row['point_id']} maps to multiple scientific datasets")
    point_to_dataset[row["point_id"]] = dataset_id
    specs[dataset_id] = spec

entries = {}
for dataset_id in sorted(specs):
    spec = specs[dataset_id]
    directory = dataset_root / dataset_id
    stem = directory / dataset_id
    dp_path = stem.with_name(stem.name + "_ptychodus_dp.hdf5")
    para_path = stem.with_name(stem.name + "_ptychodus_para.hdf5")
    geometry = create_files(dp_path, para_path, spec)
    inspect_files(dp_path, para_path, spec)
    entries[dataset_id] = {
        "dataset_id": dataset_id,
        "specification": spec,
        "spec_sha256": hashlib.sha256(canonical_bytes(spec)).hexdigest(),
        "stem": str(stem),
        "diffraction_path": str(dp_path),
        "parameter_path": str(para_path),
        "diffraction_sha256": sha256_file(dp_path),
        "parameter_sha256": sha256_file(para_path),
        "geometry": geometry,
    }

manifest = {
    "schema_version": "1.0",
    "generator": generator_version,
    "seed": seed,
    "matrix_sha256": sha256_file(matrix_path),
    "datasets": entries,
    "point_to_dataset": dict(sorted(point_to_dataset.items())),
}
encoded = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
if manifest_path.exists():
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing != manifest:
        raise RuntimeError("Existing dataset_manifest.json does not match validated requested datasets")
else:
    with manifest_path.open("x", encoding="utf-8") as f:
        f.write(encoded)
print(f"Validated {len(entries)} unique datasets for {len(rows)} matrix rows")
PY

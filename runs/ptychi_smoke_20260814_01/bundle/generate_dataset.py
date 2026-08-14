#!/usr/bin/env python3
"""Generate one tiny deterministic ptychography dataset for a pipeline smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np


PIXEL_SIZE_M = 1.0e-8
PHOTON_SCALE = 1000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scan-points", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.scan_points < 1 or args.resolution < 8:
        raise ValueError("scan-points must be positive and resolution must be at least 8")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "dataset"
    diffraction_path = output / "dataset_ptychodus_dp.hdf5"
    parameter_path = output / "dataset_ptychodus_para.hdf5"
    manifest_path = output / "manifest.json"

    rng = np.random.default_rng(args.seed)
    columns = math.ceil(math.sqrt(args.scan_points))
    rows = math.ceil(args.scan_points / columns)
    step = max(2, args.resolution // 4)
    y_px = np.repeat(np.arange(rows), columns)[: args.scan_points] * step
    x_px = np.tile(np.arange(columns), rows)[: args.scan_points] * step
    object_height = int(y_px.max()) + args.resolution
    object_width = int(x_px.max()) + args.resolution

    amplitude = 0.75 + 0.25 * rng.random((object_height, object_width))
    phase = rng.uniform(-np.pi, np.pi, (object_height, object_width))
    latent_object = (amplitude * np.exp(1j * phase)).astype(np.complex64)

    axis = np.arange(args.resolution, dtype=np.float32) - (args.resolution - 1) / 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    sigma = args.resolution / 5.0
    probe = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma)).astype(np.complex64)
    probe /= np.sqrt(np.sum(np.abs(probe) ** 2, dtype=np.float64)).astype(np.float32)

    patches = np.empty((args.scan_points, args.resolution, args.resolution), np.complex64)
    for index, (y0, x0) in enumerate(zip(y_px, x_px)):
        patches[index] = latent_object[
            int(y0) : int(y0) + args.resolution,
            int(x0) : int(x0) + args.resolution,
        ]
    far_field = np.fft.fft2(patches * probe, axes=(-2, -1), norm="ortho")
    expected = np.abs(far_field).astype(np.float64) ** 2 * PHOTON_SCALE
    diffraction = rng.poisson(expected).astype(np.float32)
    if not np.isfinite(diffraction).all() or np.any(diffraction < 0):
        raise ValueError("Generated diffraction data are invalid")

    with h5py.File(diffraction_path, "w", libver="latest") as stream:
        stream.create_dataset("dp", data=diffraction, track_times=False)
    with h5py.File(parameter_path, "w", libver="latest") as stream:
        stream.create_dataset("probe", data=probe[None, :, :], track_times=False)
        stream.create_dataset(
            "probe_position_y_m", data=y_px.astype(np.float64) * PIXEL_SIZE_M,
            track_times=False,
        )
        stream.create_dataset(
            "probe_position_x_m", data=x_px.astype(np.float64) * PIXEL_SIZE_M,
            track_times=False,
        )
        object_group = stream.create_group("object")
        object_group.attrs["pixel_height_m"] = PIXEL_SIZE_M

    manifest = {
        "schema_version": "0.1",
        "dataset_id": "smoke-r16-n32",
        "dataset_stem": str(stem),
        "seed": args.seed,
        "scan_point_count": args.scan_points,
        "detector_shape": [args.resolution, args.resolution],
        "diffraction_dtype": "float32",
        "probe_shape": [1, args.resolution, args.resolution],
        "probe_dtype": "complex64",
        "position_dtype": "float64",
        "pixel_size_m": PIXEL_SIZE_M,
        "files": {
            "diffraction": {
                "path": str(diffraction_path),
                "bytes": diffraction_path.stat().st_size,
                "sha256": sha256(diffraction_path),
            },
            "parameters": {
                "path": str(parameter_path),
                "bytes": parameter_path.stat().st_size,
                "sha256": sha256(parameter_path),
            },
        },
        "validation": {
            "finite": True,
            "nonnegative": True,
            "nonzero_total_power": bool(np.all(diffraction.sum(axis=(1, 2)) > 0)),
        },
    }
    if not manifest["validation"]["nonzero_total_power"]:
        raise ValueError("At least one diffraction frame has zero power")
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Deterministic streaming synthetic-input generation for the pty-chi workflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


GENERATOR_ID = "ptychi-hdf5-forward-v1"
PIXEL_SIZE_M = 1.0e-9
PHOTON_SCALE = 1.0e4
GENERATION_BATCH_SIZE = 16


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    scan_point_count: int
    detector_height: int
    detector_width: int
    point_ids: tuple[str, ...]
    seed: int

    @property
    def logical_diffraction_bytes(self) -> int:
        return self.scan_point_count * self.detector_height * self.detector_width * 4

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "scan_point_count": self.scan_point_count,
            "detector_shape": [self.detector_height, self.detector_width],
            "point_ids": list(self.point_ids),
            "seed": self.seed,
            "logical_diffraction_bytes": self.logical_diffraction_bytes,
        }


def _derived_seed(base_seed: int, scan_count: int, height: int, width: int) -> int:
    material = f"{GENERATOR_ID}:{base_seed}:{scan_count}:{height}:{width}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def extract_dataset_specs(plan: dict[str, Any]) -> list[DatasetSpec]:
    base_seed = int(plan.get("synthetic_input", {}).get("seed", 0))
    grouped: dict[tuple[int, int, int], list[str]] = {}
    for point in plan.get("matrix_design", {}).get("base_points", []):
        inputs = point.get("inputs", {})
        scan_count = int(inputs["scan_point_count"])
        detector_shape = inputs["detector_shape"]
        if (
            scan_count < 1
            or not isinstance(detector_shape, list)
            or len(detector_shape) != 2
        ):
            raise ValueError(f"Invalid synthetic dataset shape in {point.get('point_id')}")
        height, width = (int(detector_shape[0]), int(detector_shape[1]))
        if height < 1 or width < 1:
            raise ValueError(f"Detector dimensions must be positive in {point.get('point_id')}")
        grouped.setdefault((scan_count, height, width), []).append(str(point["point_id"]))

    specs = []
    for scan_count, height, width in sorted(grouped):
        specs.append(
            DatasetSpec(
                dataset_id=f"N{scan_count}_H{height}_W{width}",
                scan_point_count=scan_count,
                detector_height=height,
                detector_width=width,
                point_ids=tuple(grouped[(scan_count, height, width)]),
                seed=_derived_seed(base_seed, scan_count, height, width),
            )
        )
    if not specs:
        raise ValueError("Experiment plan contains no synthetic dataset requirements")
    return specs


def preview_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    specs = extract_dataset_specs(plan)
    return {
        "schema_version": "0.1",
        "generator_id": GENERATOR_ID,
        "plan_id": plan.get("plan", {}).get("plan_id"),
        "base_seed": int(plan.get("synthetic_input", {}).get("seed", 0)),
        "dataset_count": len(specs),
        "total_logical_diffraction_bytes": sum(
            spec.logical_diffraction_bytes for spec in specs
        ),
        "datasets": [spec.as_dict() for spec in specs],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(spec: DatasetSpec) -> str:
    payload = {
        **spec.as_dict(),
        "generator_id": GENERATOR_ID,
        "pixel_size_m": PIXEL_SIZE_M,
        "photon_scale": PHOTON_SCALE,
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "poisson_noise": True,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _positions(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    columns = math.ceil(math.sqrt(spec.scan_point_count))
    rows = math.ceil(spec.scan_point_count / columns)
    step = max(1, min(spec.detector_height, spec.detector_width) // 4)
    indices = np.arange(spec.scan_point_count, dtype=np.int64)
    y_px = (indices // columns) * step
    x_px = (indices % columns) * step
    return y_px, x_px, rows, columns, step


def _probe(height: int, width: int) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    amplitude = np.exp(-0.5 * ((yy / 0.32) ** 2 + (xx / 0.32) ** 2))
    phase = 0.15 * np.sin(np.pi * xx) * np.cos(np.pi * yy)
    probe = amplitude * np.exp(1j * phase)
    probe /= np.sqrt(np.sum(np.abs(probe) ** 2, dtype=np.float64))
    return probe.astype(np.complex64)


def _valid_existing(
    manifest_path: Path,
    diffraction_path: Path,
    parameter_path: Path,
    fingerprint: str,
) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if document.get("specification_sha256") != fingerprint:
        return None
    if not diffraction_path.is_file() or not parameter_path.is_file():
        return None
    if document.get("files", {}).get("diffraction", {}).get("sha256") != _sha256(
        diffraction_path
    ):
        return None
    if document.get("files", {}).get("parameters", {}).get("sha256") != _sha256(
        parameter_path
    ):
        return None
    return document


def _remove_incomplete_owned_output(
    manifest_path: Path,
    diffraction_path: Path,
    parameter_path: Path,
) -> None:
    """Remove only generator-owned files that cannot represent a committed dataset."""
    if manifest_path.exists():
        raise ValueError(
            "Dataset manifest exists but does not match the requested specification or "
            f"checksums: {manifest_path}. Refusing to overwrite it."
        )
    diffraction_path.unlink(missing_ok=True)
    parameter_path.unlink(missing_ok=True)


def _generate_one(spec: DatasetSpec, dataset_root: Path) -> dict[str, Any]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    stem = dataset_root / "dataset"
    diffraction_path = stem.with_name(stem.name + "_ptychodus_dp.hdf5")
    parameter_path = stem.with_name(stem.name + "_ptychodus_para.hdf5")
    manifest_path = dataset_root / "manifest.json"
    fingerprint = _fingerprint(spec)
    existing = _valid_existing(
        manifest_path, diffraction_path, parameter_path, fingerprint
    )
    if existing is not None:
        return existing
    if diffraction_path.exists() or parameter_path.exists():
        _remove_incomplete_owned_output(
            manifest_path,
            diffraction_path,
            parameter_path,
        )

    diffraction_partial = diffraction_path.with_suffix(diffraction_path.suffix + ".partial")
    parameter_partial = parameter_path.with_suffix(parameter_path.suffix + ".partial")
    diffraction_partial.unlink(missing_ok=True)
    parameter_partial.unlink(missing_ok=True)

    rng = np.random.default_rng(spec.seed)
    y_px, x_px, rows, columns, step = _positions(spec)
    object_height = (rows - 1) * step + spec.detector_height
    object_width = (columns - 1) * step + spec.detector_width
    object_amplitude = 0.75 + 0.25 * rng.random(
        (object_height, object_width), dtype=np.float32
    )
    object_phase = rng.uniform(-np.pi, np.pi, (object_height, object_width)).astype(
        np.float32
    )
    latent_object = (object_amplitude * np.exp(1j * object_phase)).astype(np.complex64)
    probe = _probe(spec.detector_height, spec.detector_width)

    observed_min = math.inf
    observed_max = -math.inf
    observed_sum = 0.0
    chunk_count = min(GENERATION_BATCH_SIZE, spec.scan_point_count)
    try:
        with h5py.File(diffraction_partial, "w", libver="latest") as stream:
            output = stream.create_dataset(
                "dp",
                shape=(
                    spec.scan_point_count,
                    spec.detector_height,
                    spec.detector_width,
                ),
                dtype=np.float32,
                chunks=(chunk_count, spec.detector_height, spec.detector_width),
                compression=None,
                track_times=False,
            )
            for start in range(0, spec.scan_point_count, GENERATION_BATCH_SIZE):
                stop = min(start + GENERATION_BATCH_SIZE, spec.scan_point_count)
                patches = np.empty(
                    (stop - start, spec.detector_height, spec.detector_width),
                    dtype=np.complex64,
                )
                for local, index in enumerate(range(start, stop)):
                    y0, x0 = int(y_px[index]), int(x_px[index])
                    patches[local] = latent_object[
                        y0 : y0 + spec.detector_height,
                        x0 : x0 + spec.detector_width,
                    ]
                far_field = np.fft.fft2(patches * probe, axes=(-2, -1), norm="ortho")
                expected = (
                    np.abs(far_field).astype(np.float64, copy=False) ** 2
                ) * PHOTON_SCALE
                intensities = rng.poisson(expected).astype(np.float32)
                if not np.isfinite(intensities).all() or np.any(intensities < 0):
                    raise ValueError(f"Generated invalid diffraction values for {spec.dataset_id}")
                output[start:stop] = intensities
                observed_min = min(observed_min, float(intensities.min()))
                observed_max = max(observed_max, float(intensities.max()))
                observed_sum += float(intensities.sum(dtype=np.float64))
            stream.flush()

        with h5py.File(parameter_partial, "w", libver="latest") as stream:
            stream.create_dataset("probe", data=probe[None, :, :], track_times=False)
            stream.create_dataset(
                "probe_position_y_m",
                data=y_px.astype(np.float64) * PIXEL_SIZE_M,
                track_times=False,
            )
            stream.create_dataset(
                "probe_position_x_m",
                data=x_px.astype(np.float64) * PIXEL_SIZE_M,
                track_times=False,
            )
            object_group = stream.create_group("object")
            object_group.attrs["pixel_height_m"] = PIXEL_SIZE_M
            stream.flush()

        os.replace(diffraction_partial, diffraction_path)
        os.replace(parameter_partial, parameter_path)
    finally:
        diffraction_partial.unlink(missing_ok=True)
        parameter_partial.unlink(missing_ok=True)

    with h5py.File(diffraction_path, "r") as stream:
        dataset = stream["dp"]
        if dataset.shape != (
            spec.scan_point_count,
            spec.detector_height,
            spec.detector_width,
        ) or dataset.dtype != np.dtype("float32"):
            raise ValueError(f"Diffraction HDF5 validation failed: {diffraction_path}")
        chunks = list(dataset.chunks or ())
    with h5py.File(parameter_path, "r") as stream:
        required = {"probe", "probe_position_y_m", "probe_position_x_m", "object"}
        if not required.issubset(stream.keys()):
            raise ValueError(f"Parameter HDF5 keys are incomplete: {parameter_path}")
        probe_dataset = stream["probe"]
        if probe_dataset.shape != (1, spec.detector_height, spec.detector_width):
            raise ValueError(f"Probe shape validation failed: {parameter_path}")
        if probe_dataset.dtype != np.dtype("complex64"):
            raise ValueError(f"Probe dtype validation failed: {parameter_path}")
        stored_probe = probe_dataset[...]
        if (
            not np.isfinite(stored_probe.real).all()
            or not np.isfinite(stored_probe.imag).all()
            or float(np.sum(np.abs(stored_probe) ** 2, dtype=np.float64)) <= 0.0
        ):
            raise ValueError(f"Probe value validation failed: {parameter_path}")
        for position_key in ("probe_position_y_m", "probe_position_x_m"):
            position_dataset = stream[position_key]
            if (
                position_dataset.shape != (spec.scan_point_count,)
                or position_dataset.dtype != np.dtype("float64")
            ):
                raise ValueError(
                    f"Position shape/dtype validation failed for {position_key}: "
                    f"{parameter_path}"
                )
            positions = position_dataset[...]
            if not np.isfinite(positions).all() or np.any(positions < 0):
                raise ValueError(
                    f"Position value validation failed for {position_key}: "
                    f"{parameter_path}"
                )
        if float(stream["object"].attrs["pixel_height_m"]) != PIXEL_SIZE_M:
            raise ValueError(f"Pixel-size metadata validation failed: {parameter_path}")

    manifest = {
        "schema_version": "0.1",
        "generator_id": GENERATOR_ID,
        "specification_sha256": fingerprint,
        "dataset": spec.as_dict(),
        "dataset_stem": stem.name,
        "generation": {
            "pixel_size_m": PIXEL_SIZE_M,
            "raster_step_px": step,
            "raster_rows": rows,
            "raster_columns": columns,
            "latent_object_shape": [object_height, object_width],
            "probe_shape": [1, spec.detector_height, spec.detector_width],
            "probe_dtype": "complex64",
            "diffraction_dtype": "float32",
            "position_dtype": "float64",
            "photon_scale": PHOTON_SCALE,
            "poisson_noise": True,
            "hdf5_chunks": chunks,
            "compression": None,
        },
        "validation": {
            "finite": True,
            "nonnegative": True,
            "minimum": observed_min,
            "maximum": observed_max,
            "mean": observed_sum
            / (spec.scan_point_count * spec.detector_height * spec.detector_width),
            "shape_valid": True,
            "dtype_valid": True,
            "position_lengths_valid": True,
            "probe_power_nonzero": True,
        },
        "files": {
            "diffraction": {
                "path": diffraction_path.name,
                "sha256": _sha256(diffraction_path),
                "size_bytes": diffraction_path.stat().st_size,
            },
            "parameters": {
                "path": parameter_path.name,
                "sha256": _sha256(parameter_path),
                "size_bytes": parameter_path.stat().st_size,
            },
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def generate_datasets(
    plan: dict[str, Any],
    run_dir: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    preview = preview_manifest(plan)
    required = int(preview["total_logical_diffraction_bytes"] * 1.05) + 256 * 2**20
    available = shutil.disk_usage(root).free
    if available < required:
        raise ValueError(
            f"Insufficient disk space: need approximately {required} bytes, have {available}"
        )

    dataset_root = root / "datasets" / "pty-chi"
    manifests = []
    for spec in extract_dataset_specs(plan):
        manifests.append(_generate_one(spec, dataset_root / spec.dataset_id))
    return {
        **preview,
        "dataset_root": str(dataset_root),
        "datasets": manifests,
        "validation": {
            "all_generated": True,
            "all_checksums_recorded": True,
            "all_shapes_and_dtypes_valid": True,
        },
    }

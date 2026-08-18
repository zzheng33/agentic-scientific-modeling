"""Measurement validation and compact per-algorithm resource-model fitting."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np


EXTRACTED_FIELDS = [
    "run_id", "usable", "issues", "algorithm_group_id", "accelerator", "point_id",
    "repetition", "scan_point_count", "detector_height", "detector_width",
    "num_epochs", "batch_size", "total_time_s", "io_load_time_s",
    "reconstruction_run_time_s", "avg_power_w", "peak_power_w", "energy_j", "peak_memory_mib",
    "throughput_images_s", "power_sample_count", "log_path", "power_trace_path",
]
FEATURES = (
    "intercept",
    "input_gpixels",
    "compute_tpixel_epochs",
    "epoch_batches_k",
    "batch_mpixels",
)
TARGETS = (
    "latency_s", "avg_power_w", "energy_j", "peak_memory_mib", "throughput_per_s"
)


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _power_metrics(
    path: str | Path,
) -> tuple[float | None, float | None, float | None, float | None, int]:
    power_path = Path(path)
    if not power_path.is_file():
        return None, None, None, None, 0
    times: list[float] = []
    powers: list[float] = []
    memories: list[float] = []
    with power_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            time_value = _float(row.get("Time(S)"))
            samples = [
                value for key, raw in row.items()
                if key.endswith("_Power(W)") and (value := _float(raw)) is not None
            ]
            if time_value is not None and samples:
                power = sum(samples)
                if times and time_value <= times[-1]:
                    if time_value == times[-1]:
                        powers[-1] = power
                    memories.extend(
                        value for key, raw in row.items()
                        if key.endswith("_MemoryUsed(MiB)")
                        and (value := _float(raw)) is not None
                    )
                    continue
                times.append(time_value)
                powers.append(power)
                memories.extend(
                    value for key, raw in row.items()
                    if key.endswith("_MemoryUsed(MiB)") and (value := _float(raw)) is not None
                )
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        return None, None, None, max(memories) if memories else None, len(times)
    time_array = np.asarray(times, dtype=np.float64)
    power_array = np.asarray(powers, dtype=np.float64)
    energy = float(np.trapezoid(power_array, time_array))
    duration = float(time_array[-1] - time_array[0])
    average = energy / duration if duration > 0 else None
    return average, float(np.max(power_array)), energy, max(memories) if memories else None, len(times)


def _result_peak_memory_mib(log_path: str | Path) -> float | None:
    """Use application-reported accelerator peak memory when telemetry omits it."""
    result_path = Path(log_path).parent / "result.json"
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    runtime = result.get("effective_runtime", {})
    candidates = [
        _float(runtime.get("peak_accelerator_memory_allocated_bytes")),
        _float(runtime.get("peak_accelerator_memory_reserved_bytes")),
    ]
    valid = [value for value in candidates if value is not None and value >= 0]
    return max(valid) / (1024.0 * 1024.0) if valid else None


def extract_measurements(
    measurements_path: str | Path,
    expected_run_count: int,
) -> tuple[str, dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    with Path(measurements_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        run_id = row["run_id"]
        if run_id in seen:
            duplicates.add(run_id)
        seen.add(run_id)
        issues: list[str] = []
        latency = _float(row.get("total_time_s"))
        if (
            str(row.get("status", "")).strip().lower()
            not in {"completed", "success"}
            or str(row.get("return_code")) != "0"
        ):
            issues.append("run_failed")
        if latency is None or latency <= 0:
            issues.append("invalid_total_time")
        average_power, peak_power, energy, peak_memory, sample_count = _power_metrics(
            row.get("power_trace_path", "")
        )
        if peak_memory is None:
            peak_memory = _result_peak_memory_mib(row.get("log_path", ""))
        if sample_count < 2:
            issues.append("insufficient_power_samples")
        if peak_memory is None:
            issues.append("missing_peak_memory")
        scan_count = int(row["scan_point_count"])
        throughput = scan_count / latency if latency and latency > 0 else None
        extracted.append(
            {
                "run_id": run_id,
                "usable": "true" if not issues else "false",
                "issues": ";".join(issues),
                **{key: row.get(key, "") for key in (
                    "algorithm_group_id", "accelerator", "point_id", "repetition",
                    "scan_point_count", "detector_height", "detector_width",
                    "num_epochs", "batch_size", "total_time_s", "io_load_time_s",
                    "reconstruction_run_time_s", "log_path", "power_trace_path",
                )},
                "avg_power_w": "" if average_power is None else f"{average_power:.9g}",
                "peak_power_w": "" if peak_power is None else f"{peak_power:.9g}",
                "energy_j": "" if energy is None else f"{energy:.9g}",
                "peak_memory_mib": "" if peak_memory is None else f"{peak_memory:.9g}",
                "throughput_images_s": "" if throughput is None else f"{throughput:.9g}",
                "power_sample_count": sample_count,
            }
        )

    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in extracted:
        by_group[(row["algorithm_group_id"], row["accelerator"], row["point_id"])].append(row)
    suspicious: set[str] = set()
    for group in by_group.values():
        values = [(row, _float(row["total_time_s"])) for row in group]
        valid = [(row, value) for row, value in values if value is not None]
        if len(valid) < 3:
            continue
        sample = np.asarray([value for _, value in valid], dtype=np.float64)
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        if mad > 0:
            for row, value in valid:
                if abs(value - median) / (1.4826 * mad) > 3.5:
                    suspicious.add(row["run_id"])
                    row["issues"] = ";".join(filter(None, [row["issues"], "latency_outlier"]))

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=EXTRACTED_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(extracted)
    included = [row["run_id"] for row in extracted if row["usable"] == "true"]
    validation = {
        "schema_version": "0.1",
        "status": "awaiting_human_review",
        "expected_runs": expected_run_count,
        "observed_runs": len(rows),
        "unique_runs": len(seen),
        "usable_runs": len(included),
        "failed_or_incomplete_runs": len(extracted) - len(included),
        "duplicate_run_ids": sorted(duplicates),
        "missing_run_count": max(0, expected_run_count - len(seen)),
        "suspicious_run_ids": sorted(suspicious),
        "run_decisions": [
            {
                "run_id": row["run_id"],
                "decision": "include" if row["usable"] == "true" else "exclude",
                "reason": row["issues"] or "complete_and_valid",
            }
            for row in extracted
        ],
        "validation": {
            "ready_for_modeling": len(included) > 0 and not duplicates,
            "power_required": True,
            "energy_method": "trapezoidal integration of summed GPU power over Time(S)",
        },
    }
    return output.getvalue(), validation


def _feature_values(row: dict[str, str]) -> dict[str, float]:
    n = float(row["scan_point_count"])
    h = float(row["detector_height"])
    w = float(row["detector_width"])
    epochs = float(row["num_epochs"])
    batch = min(n, float(row["batch_size"]))
    pixels = h * w
    return {
        "intercept": 1.0,
        "input_gpixels": n * pixels / 1e9,
        "compute_tpixel_epochs": n * pixels * epochs / 1e12,
        "epoch_batches_k": epochs * math.ceil(n / batch) / 1e3,
        "batch_mpixels": batch * pixels / 1e6,
    }


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1e-8) -> tuple[np.ndarray, float, float]:
    means = np.mean(x[:, 1:], axis=0)
    scales = np.std(x[:, 1:], axis=0)
    scales[scales == 0] = 1.0
    standardized = np.column_stack([np.ones(len(x)), (x[:, 1:] - means) / scales])
    penalty = np.eye(standardized.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(standardized.T @ standardized + penalty, standardized.T @ y)
    predicted = standardized @ coefficients
    residual = y - predicted
    ss_res = float(residual @ residual)
    centered = y - np.mean(y)
    ss_tot = float(centered @ centered)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(np.mean(residual**2)))
    packed = np.concatenate([[coefficients[0]], coefficients[1:], means, scales])
    return packed, r2, rmse


def fit_resource_model(
    extracted_path: str | Path,
    approved_validation: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    decisions = {
        item["run_id"]: item["decision"]
        for item in approved_validation.get("run_decisions", [])
    }
    with Path(extracted_path).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if decisions.get(row["run_id"]) == "include"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["accelerator"], row["algorithm_group_id"])].append(row)
    if not grouped:
        raise ValueError("No human-approved measurements are available for model fitting")

    coefficient_rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for (accelerator, algorithm), samples in sorted(grouped.items()):
        if len(samples) < len(FEATURES):
            raise ValueError(
                f"Need at least {len(FEATURES)} included runs for {accelerator}/{algorithm}; "
                f"found {len(samples)}"
            )
        x = np.asarray(
            [[_feature_values(row)[feature] for feature in FEATURES] for row in samples],
            dtype=np.float64,
        )
        targets = {
            "latency_s": np.asarray([float(row["total_time_s"]) for row in samples]),
            "avg_power_w": np.asarray([float(row["avg_power_w"]) for row in samples]),
            "energy_j": np.asarray([float(row["energy_j"]) for row in samples]),
            "peak_memory_mib": np.asarray([float(row["peak_memory_mib"]) for row in samples]),
            "throughput_per_s": np.asarray(
                [float(row["throughput_images_s"]) for row in samples]
            ),
        }
        target_models: dict[str, Any] = {}
        for target, y in targets.items():
            if not np.isfinite(y).all() or np.any(y <= 0):
                raise ValueError(f"Invalid {target} values for {accelerator}/{algorithm}")
            packed, r2, rmse = _ridge_fit(x, y)
            count = len(FEATURES) - 1
            model = {
                "intercept": float(packed[0]),
                "standardized_coefficients": {
                    feature: float(value)
                    for feature, value in zip(FEATURES[1:], packed[1 : 1 + count])
                },
                "feature_means": {
                    feature: float(value)
                    for feature, value in zip(FEATURES[1:], packed[1 + count : 1 + 2 * count])
                },
                "feature_scales": {
                    feature: float(value)
                    for feature, value in zip(FEATURES[1:], packed[1 + 2 * count :])
                },
                "r2": r2,
                "rmse": rmse,
                "n": len(samples),
            }
            target_models[target] = model
            for feature in FEATURES:
                coefficient_rows.append(
                    {
                        "accelerator": accelerator,
                        "algorithm": algorithm,
                        "target": target,
                        "feature": feature,
                        "coefficient": model["intercept"] if feature == "intercept" else model["standardized_coefficients"][feature],
                        "feature_mean": 0.0 if feature == "intercept" else model["feature_means"][feature],
                        "feature_scale": 1.0 if feature == "intercept" else model["feature_scales"][feature],
                        "r2": r2,
                        "rmse": rmse,
                        "n": len(samples),
                    }
                )
        groups.append({"accelerator": accelerator, "algorithm": algorithm, "targets": target_models})

    domain = {
        "scan_point_count": [min(int(row["scan_point_count"]) for row in rows), max(int(row["scan_point_count"]) for row in rows)],
        "detector_height": [min(int(row["detector_height"]) for row in rows), max(int(row["detector_height"]) for row in rows)],
        "detector_width": [min(int(row["detector_width"]) for row in rows), max(int(row["detector_width"]) for row in rows)],
        "num_epochs": [min(int(row["num_epochs"]) for row in rows), max(int(row["num_epochs"]) for row in rows)],
        "batch_size": [min(int(row["batch_size"]) for row in rows), max(int(row["batch_size"]) for row in rows)],
    }
    model_document = {
        "schema_version": "systemflow-application-resource-model-0.1",
        "status": "awaiting_human_review",
        "model_inputs": ["scan_point_count", "detector_shape", "num_epochs", "batch_size"],
        "grouping": ["accelerator", "algorithm"],
        "feature_definitions": {
            "input_gpixels": (
                "scan_point_count * detector_shape[0] * detector_shape[1] / 1e9"
            ),
            "compute_tpixel_epochs": (
                "scan_point_count * detector_shape[0] * detector_shape[1] * "
                "num_epochs / 1e12"
            ),
            "epoch_batches_k": (
                "num_epochs * ceil(scan_point_count / min(batch_size, "
                "scan_point_count)) / 1e3"
            ),
            "batch_mpixels": (
                "min(batch_size, scan_point_count) * detector_shape[0] * "
                "detector_shape[1] / 1e6"
            ),
        },
        "supported_domain": domain,
        "groups": groups,
        "prediction_policy": "warn outside supported_domain; clamp predictions to positive epsilon",
    }
    output = StringIO(newline="")
    fieldnames = ["accelerator", "algorithm", "target", "feature", "coefficient", "feature_mean", "feature_scale", "r2", "rmse", "n"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(coefficient_rows)
    return model_document, output.getvalue()


def render_model_json(model: dict[str, Any]) -> str:
    return json.dumps(model, indent=2, sort_keys=True) + "\n"

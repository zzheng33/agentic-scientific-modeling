from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from agents.modeling.runner import extract_measurements


class MeasurementExtractionTests(unittest.TestCase):
    def test_success_status_duplicate_power_time_and_result_memory_are_usable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "run-1"
            run.mkdir(parents=True)
            log = run / "run.log"
            log.write_text("completed\n")
            power = run / "power.csv"
            power.write_text(
                "Time(S),GPU0_Power(W)\n"
                "0.01,100\n"
                "0.01,101\n"
                "0.21,120\n"
            )
            (run / "result.json").write_text(
                json.dumps(
                    {
                        "effective_runtime": {
                            "peak_accelerator_memory_allocated_bytes": 16 * 1024 * 1024,
                            "peak_accelerator_memory_reserved_bytes": 24 * 1024 * 1024,
                        }
                    }
                )
            )
            measurements = root / "measurements.csv"
            fields = [
                "run_id", "status", "return_code", "algorithm_group_id",
                "accelerator", "point_id", "repetition", "scan_point_count",
                "detector_height", "detector_width", "num_epochs", "batch_size",
                "total_time_s", "io_load_time_s", "reconstruction_run_time_s",
                "log_path", "power_trace_path",
            ]
            with measurements.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "run_id": "run-1", "status": "success", "return_code": "0",
                        "algorithm_group_id": "pie", "accelerator": "GH200",
                        "point_id": "p001", "repetition": "1",
                        "scan_point_count": "64", "detector_height": "64",
                        "detector_width": "64", "num_epochs": "2", "batch_size": "1",
                        "total_time_s": "1.25", "io_load_time_s": "0.1",
                        "reconstruction_run_time_s": "1.0",
                        "log_path": str(log), "power_trace_path": str(power),
                    }
                )

            extracted, validation = extract_measurements(measurements, 1)

        self.assertTrue(validation["validation"]["ready_for_modeling"])
        self.assertEqual(validation["usable_runs"], 1)
        self.assertIn(",24,", extracted)


if __name__ == "__main__":
    unittest.main()

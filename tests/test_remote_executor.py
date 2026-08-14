from __future__ import annotations

import tarfile
import tempfile
import unittest
from pathlib import Path

from agents.planner.script_generator import MEASUREMENT_HEADER, validate_generated_script
from agents.remote_executor.runner import RemoteExecutorConfig, _build_bundle


class RemoteExecutionTests(unittest.TestCase):
    def test_generated_script_contract_and_forbidden_scheduler_command(self) -> None:
        dataset = """#!/usr/bin/env bash
set -euo pipefail
cd "$APPLICATION_ROOT"
"$PYTHON_BIN" -c 'print("generate")'
mkdir -p "$BUNDLE_ROOT/datasets"
echo '{}' > "$BUNDLE_ROOT/datasets/dataset_manifest.json"
"""
        benchmark = f"""#!/usr/bin/env bash
set -euo pipefail
cd "$APPLICATION_ROOT"
"$PYTHON_BIN" -c 'print("benchmark")'
bash "$BUNDLE_ROOT/dataset_generation.sh"
HEADER='{MEASUREMENT_HEADER}'
echo "$HEADER" > "$BUNDLE_ROOT/results/measurements.csv"
echo power.csv result.json completion_manifest.json
"""
        validate_generated_script(dataset, kind="dataset_generation")
        validate_generated_script(benchmark, kind="benchmark")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_generated_script(benchmark + "\nqsub ./job.sh\n", kind="benchmark")

    def test_bundle_contains_approved_scripts_and_application(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            application = root / "application"
            application.mkdir()
            (application / "main.py").write_text("print('ok')\n")
            dataset = root / "dataset.sh"
            dataset.write_text("#!/bin/bash\n")
            benchmark = root / "benchmark.sh"
            benchmark.write_text("#!/bin/bash\n")
            plan = root / "plan.yaml"
            plan.write_text("plan: {}\n")
            matrix = root / "matrix.csv"
            matrix.write_text("run_id\nrun-001\n")
            config = RemoteExecutorConfig(
                host="user@login.example.org",
                remote_runs_root="/home/user/runs",
                remote_application_path="/home/user/application",
            )

            archive, summary = _build_bundle(
                dataset, benchmark, plan, matrix, application,
                run_dir, "workflow-001", 1, config,
            )

            with tarfile.open(archive) as stream:
                names = set(stream.getnames())
            self.assertIn("benchmark-v001/dataset_generation.sh", names)
            self.assertIn("benchmark-v001/benchmark_job.sh", names)
            self.assertIn("benchmark-v001/application/main.py", names)
            self.assertNotIn("benchmark-v001/remote_worker.py", names)
            self.assertEqual(summary["run_count"], 1)


if __name__ == "__main__":
    unittest.main()

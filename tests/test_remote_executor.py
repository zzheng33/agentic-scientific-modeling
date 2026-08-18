from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agents.planner.script_generator import (
    MEASUREMENT_HEADER,
    ExecutionScriptAgent,
    validate_generated_script,
)
from agents.remote_executor.runner import (
    RemoteExecutorConfig,
    _build_bundle,
    _dataset_preparation_script,
)
from workflow.artifacts import ArtifactStore


class RemoteExecutionTests(unittest.TestCase):
    def test_next_artifact_version_skips_partial_stage_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            stage = Path(directory) / "artifacts" / "benchmark_run_review"
            stage.mkdir(parents=True)
            (stage / "dataset_generation_script.v001.sh").write_text("partial")
            (stage / "benchmark_job_script.v003.sh").write_text("partial")

            self.assertEqual(
                store.next_artifact_version("benchmark_run_review", minimum=2),
                4,
            )

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
cat "$BUNDLE_ROOT/datasets/dataset_manifest.json"
HEADER='{MEASUREMENT_HEADER}'
echo "$HEADER" > "$BUNDLE_ROOT/results/measurements.csv"
echo power.csv result.json completion_manifest.json
"""
        validate_generated_script(dataset, kind="dataset_generation")
        validate_generated_script(benchmark, kind="benchmark")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_generated_script(benchmark + "\nqsub ./job.sh\n", kind="benchmark")
        with self.assertRaisesRegex(ValueError, "pre-generated dataset"):
            validate_generated_script(
                benchmark + '\nbash "$BUNDLE_ROOT/dataset_generation.sh"\n',
                kind="benchmark",
            )

    def test_braced_variables_and_independent_dataset_generator_are_allowed(self) -> None:
        dataset = """#!/usr/bin/env bash
set -euo pipefail
"${PYTHON_BIN}" -c 'print("generate")'
mkdir -p "${BUNDLE_ROOT}/datasets"
echo '{}' > "${BUNDLE_ROOT}/datasets/dataset_manifest.json"
"""
        validate_generated_script(dataset, kind="dataset_generation")

        benchmark = f"""#!/usr/bin/env bash
set -euo pipefail
"${{PYTHON_BIN}}" -c 'print("benchmark")'
cat "${{BUNDLE_ROOT}}/datasets/dataset_manifest.json"
HEADER='{MEASUREMENT_HEADER}'
echo "$HEADER" > "${{BUNDLE_ROOT}}/results/measurements.csv"
echo power.csv result.json completion_manifest.json
"""
        with self.assertRaisesRegex(ValueError, "controlled application path"):
            validate_generated_script(benchmark, kind="benchmark")

    def test_script_agent_asks_for_correction_after_validation_failure(self) -> None:
        invalid_dataset = """#!/usr/bin/env bash
set -euo pipefail
"$PYTHON_BIN" -c 'print("generate")'
echo dataset_manifest.json
"""
        valid_dataset = """#!/usr/bin/env bash
set -euo pipefail
"$PYTHON_BIN" -c 'print("generate")'
mkdir -p "$BUNDLE_ROOT/datasets"
echo '{}' > "$BUNDLE_ROOT/datasets/dataset_manifest.json"
"""
        valid_benchmark = f"""#!/usr/bin/env bash
set -euo pipefail
cd "$APPLICATION_ROOT"
"$PYTHON_BIN" -c 'print("benchmark")'
cat "$BUNDLE_ROOT/datasets/dataset_manifest.json"
HEADER='{MEASUREMENT_HEADER}'
echo "$HEADER" > "$BUNDLE_ROOT/results/measurements.csv"
echo power.csv result.json completion_manifest.json
"""

        class FakeMessage:
            def __init__(self, content: str) -> None:
                self.content = content
                self.tool_calls = []

            def model_dump(self, **_kwargs):
                return {"role": "assistant", "content": self.content}

        class FakeCompletions:
            def __init__(self) -> None:
                self.outputs = [
                    json.dumps(
                        {
                            "dataset_generation_script": invalid_dataset,
                            "benchmark_job_script": valid_benchmark,
                        }
                    ),
                    json.dumps(
                        {
                            "dataset_generation_script": valid_dataset,
                            "benchmark_job_script": valid_benchmark,
                        }
                    ),
                ]
                self.requests = []

            def create(self, **kwargs):
                self.requests.append([dict(item) for item in kwargs["messages"]])
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=FakeMessage(self.outputs.pop(0)))]
                )

        with tempfile.TemporaryDirectory() as directory:
            completions = FakeCompletions()
            agent = object.__new__(ExecutionScriptAgent)
            agent.config = SimpleNamespace(max_tool_rounds=2, model="test-model")
            agent.client = SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            )
            agent.system_prompt = "test prompt"
            agent.operational_retriever = None

            result = agent.generate(
                directory,
                characterization={},
                plan={"hardware": {"targets": [{"accelerator": "GH200"}]}},
                matrix_csv="run_id,accelerator\nrun-1,GH200\n",
                platform_profile={"machines": []},
            )

        self.assertEqual(result["dataset_generation_script"], valid_dataset)
        self.assertEqual(len(completions.requests), 2)
        correction = completions.requests[1][-1]
        self.assertEqual(correction["role"], "user")
        self.assertIn("must use the controlled bundle path", correction["content"])

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
            self.assertIn("benchmark-v001-gh200/dataset_generation.sh", names)
            self.assertIn("benchmark-v001-gh200/benchmark_job.sh", names)
            self.assertIn("benchmark-v001-gh200/prepare_dataset.sh", names)
            self.assertIn("benchmark-v001-gh200/application/main.py", names)
            self.assertNotIn("benchmark-v001-gh200/remote_worker.py", names)
            self.assertEqual(summary["run_count"], 1)
            self.assertNotIn("ssh_password", summary["remote"])

    def test_remote_dataset_preparation_runs_generation_and_validates_manifest(self) -> None:
        config = RemoteExecutorConfig(
            host="user@login.example.org",
            remote_runs_root="/home/user/runs",
            remote_application_path="/home/user/application",
        )
        script = _dataset_preparation_script(
            config, "/home/user/runs/workflow/benchmark-v001-gh200"
        )

        self.assertIn('bash "$BUNDLE_ROOT/dataset_generation.sh"', script)
        self.assertIn('test -s "$BUNDLE_ROOT/datasets/dataset_manifest.json"', script)
        self.assertIn("json.load", script)


if __name__ == "__main__":
    unittest.main()

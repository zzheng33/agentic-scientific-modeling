from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.planner.runner import (
    PlannerConfig,
    apply_configured_machines,
    build_smoke_plan,
)
from agents.remote_executor.runner import RemoteExecutorConfig
from workflow.downstream_graph import (
    _combine_measurement_csv,
    _filter_matrix_for_machine,
    _planned_machine_aliases,
)


class MachineConfigurationTests(unittest.TestCase):
    def _write_config(self, root: Path, *, include_machine: bool = True) -> Path:
        application = root / "application"
        application.mkdir()
        machine = """
[[machine]]
accelerator = "GH200"
queue = "gpu_gh200"
modules = ["cuda/12.9.1", "conda/gh200"]
conda_env = "gh200-env"
remote_monitor_script = "/home/user/monitor.py"
device = "cuda"
power_vendor = "nvidia"

[[machine]]
accelerator = "A100"
queue = "gpu_a100_test"
modules = ["cuda/test", "conda/a100"]
conda_env = "a100-env"
remote_monitor_script = "/home/user/monitor.py"
device = "cuda"
power_vendor = "nvidia"
""" if include_machine else ""
        path = root / "config.toml"
        path.write_text(
            f"""
[openai]
api_key = "test-key"
base_url = "https://example.invalid/v1"
model = "test-model"

[application]
path = "{application}"

{machine}

[planner]
max_tool_rounds = 5
max_total_runs = 10
repetitions = 2

[remote_executor]
enabled = true
host = "user@login.example.org"
remote_runs_root = "/home/user/runs"
remote_application_path = "/home/user/application"
""",
            encoding="utf-8",
        )
        return path

    def test_planner_and_executor_read_the_same_machine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(Path(directory))

            planner = PlannerConfig.from_file(config_path)
            executors = RemoteExecutorConfig.all_from_file(config_path)

            self.assertEqual(planner.machine_accelerators, ("GH200", "A100"))
            self.assertEqual(
                tuple(item.accelerator for item in executors),
                planner.machine_accelerators,
            )
            self.assertEqual(executors[0].queue, "gpu_gh200")
            self.assertEqual(executors[1].queue, "gpu_a100_test")
            self.assertEqual(
                RemoteExecutorConfig.from_file(config_path, "GH200").queue,
                "gpu_gh200",
            )

    def test_machine_section_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(Path(directory), include_machine=False)

            with self.assertRaisesRegex(ValueError, r"at least one \[\[machine\]\]"):
                PlannerConfig.from_file(config_path)
            with self.assertRaisesRegex(ValueError, r"at least one \[\[machine\]\]"):
                RemoteExecutorConfig.all_from_file(config_path)

    def test_configured_machine_replaces_llm_hardware_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = PlannerConfig.from_file(self._write_config(Path(directory)))
            plan = {
                "hardware": {
                    "allowed_catalog": ["A100"],
                    "targets": [
                        {
                            "accelerator": "A100",
                        }
                    ],
                }
            }

            apply_configured_machines(plan, config)

            self.assertEqual(
                plan["hardware"]["targets"],
                [
                    {
                        "accelerator": "GH200",
                        "notes": "Accelerator fixed by config.toml [[machine]].",
                    },
                    {
                        "accelerator": "A100",
                        "notes": "Accelerator fixed by config.toml [[machine]].",
                    },
                ],
            )

    def test_matrix_is_split_by_accelerator_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory) / "matrix.csv"
            matrix.write_text(
                "run_id,hardware_id\nrun-gh,legacy-gh-id\nrun-a,A100\n",
                encoding="utf-8",
            )
            aliases = _planned_machine_aliases(
                {
                    "hardware": {
                        "targets": [
                            {"hardware_id": "legacy-gh-id", "accelerator": "GH200"},
                            {"accelerator": "A100"},
                        ]
                    }
                }
            )

            gh200 = _filter_matrix_for_machine(matrix, "GH200", aliases["GH200"])
            a100 = _filter_matrix_for_machine(matrix, "A100", aliases["A100"])

            self.assertIn("run_id,accelerator", gh200)
            self.assertIn("run-gh,GH200", gh200)
            self.assertNotIn("run-a,A100", gh200)
            self.assertIn("run-a,A100", a100)
            combined = _combine_measurement_csv([gh200, a100])
            self.assertIn("run-gh,GH200", combined)
            self.assertIn("run-a,A100", combined)

    def test_smoke_plan_keeps_one_group_point_and_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = PlannerConfig.from_file(self._write_config(Path(directory)))
            plan = {
                "plan": {"status": "approved", "summary": "full"},
                "algorithm_groups": [
                    {"algorithm_group_id": "pie", "display_name": "PIE"},
                    {"algorithm_group_id": "dm", "display_name": "DM"},
                ],
                "hardware": {"targets": [{"accelerator": "GH200"}]},
                "matrix_design": {
                    "base_points": [
                        {"point_id": "p001", "inputs": {"size": 1}},
                        {"point_id": "p002", "inputs": {"size": 2}},
                    ],
                    "estimated_total_runs": 12,
                    "matrix_artifact": {"path": "old.csv"},
                },
                "measurement": {"warmup_runs": 1, "measured_repetitions": 3},
                "approval": {"status": "approved", "reviewer": "old"},
                "execution": {"mode": "remote", "runner_status": "approved"},
                "validation": {"issues": []},
            }

            smoke = build_smoke_plan(
                plan,
                config,
                algorithm_group_id="pie",
                point_id="p001",
            )

            self.assertEqual(len(smoke["algorithm_groups"]), 1)
            self.assertEqual(len(smoke["matrix_design"]["base_points"]), 1)
            self.assertEqual(smoke["measurement"]["warmup_runs"], 0)
            self.assertEqual(smoke["measurement"]["measured_repetitions"], 1)
            self.assertEqual(smoke["matrix_design"]["estimated_total_runs"], 2)
            self.assertNotIn("matrix_artifact", smoke["matrix_design"])


if __name__ == "__main__":
    unittest.main()

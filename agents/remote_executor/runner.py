"""SSH/SCP/Cobalt execution backend for an approved benchmark manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any


_HOST = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$")
_COMMAND = re.compile(r"^[A-Za-z0-9._/-]+$")
_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")


@dataclass(frozen=True)
class RemoteExecutorConfig:
    host: str
    remote_runs_root: str
    remote_application_path: str
    hardware_id: str = "GH200"
    queue: str = "gpu_gh200"
    nodes: int = 1
    walltime_minutes: int = 30
    poll_interval_s: int = 15
    poll_timeout_s: int = 3600
    qsub_command: str = "qsub"
    qstat_command: str = "qstat"
    module_path: str = "/soft/modulefiles"
    modules: tuple[str, ...] = (
        "cuda/12.9.1",
        "conda/nvidia/suse15.6/2025.01-11",
    )
    conda_env: str = "ptychopinn_torch_arm"
    remote_monitor_script: str = "/home/zhong.zheng/PtychoPINN/scripts/monitor_gpu_power.py"
    device: str = "cuda"
    vendor: str = "nvidia"
    devices: str = "0"
    power_interval_s: float = 0.2
    continue_on_error: bool = True
    upload_application: bool = True

    @classmethod
    def from_file(cls, path: str | Path | None) -> "RemoteExecutorConfig":
        if path is None:
            raise ValueError("Remote execution requires a configuration file")
        with Path(path).expanduser().resolve(strict=True).open("rb") as stream:
            document = tomllib.load(stream)
        configured = document.get("remote_executor", {})
        if not bool(configured.get("enabled", False)):
            raise ValueError("Set remote_executor.enabled=true before remote execution")
        instance = cls(
            host=str(configured.get("host", "")).strip(),
            remote_runs_root=str(configured.get("remote_runs_root", "")).strip(),
            remote_application_path=str(
                configured.get("remote_application_path", "")
            ).strip(),
            hardware_id=str(configured.get("hardware_id", "GH200")).strip(),
            queue=str(configured.get("queue", "gpu_gh200")).strip(),
            nodes=int(configured.get("nodes", 1)),
            walltime_minutes=int(configured.get("walltime_minutes", 30)),
            poll_interval_s=int(configured.get("poll_interval_s", 15)),
            poll_timeout_s=int(configured.get("poll_timeout_s", 3600)),
            qsub_command=str(configured.get("qsub_command", "qsub")).strip(),
            qstat_command=str(configured.get("qstat_command", "qstat")).strip(),
            module_path=str(configured.get("module_path", "/soft/modulefiles")),
            modules=tuple(
                str(item) for item in configured.get(
                    "modules",
                    ["cuda/12.9.1", "conda/nvidia/suse15.6/2025.01-11"],
                )
            ),
            conda_env=str(configured.get("conda_env", "ptychopinn_torch_arm")),
            remote_monitor_script=str(
                configured.get(
                    "remote_monitor_script",
                    "/home/zhong.zheng/PtychoPINN/scripts/monitor_gpu_power.py",
                )
            ),
            device=str(configured.get("device", "cuda")),
            vendor=str(configured.get("vendor", "nvidia")),
            devices=str(configured.get("devices", "0")),
            power_interval_s=float(configured.get("power_interval_s", 0.2)),
            continue_on_error=bool(configured.get("continue_on_error", True)),
            upload_application=bool(configured.get("upload_application", True)),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        if not _HOST.fullmatch(self.host):
            raise ValueError("remote_executor.host must have the form user@host")
        for name, value in (
            ("remote_runs_root", self.remote_runs_root),
            ("remote_application_path", self.remote_application_path),
            ("remote_monitor_script", self.remote_monitor_script),
        ):
            path = PurePosixPath(value)
            if (
                not path.is_absolute()
                or ".." in path.parts
                or not _REMOTE_PATH.fullmatch(value)
            ):
                raise ValueError(f"remote_executor.{name} must be an absolute safe path")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.queue):
            raise ValueError("remote_executor.queue contains invalid characters")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.hardware_id):
            raise ValueError("remote_executor.hardware_id contains invalid characters")
        if not _COMMAND.fullmatch(self.qsub_command) or not _COMMAND.fullmatch(
            self.qstat_command
        ):
            raise ValueError("Remote scheduler commands contain invalid characters")
        if not self.modules or any(not _COMMAND.fullmatch(item) for item in self.modules):
            raise ValueError("remote_executor.modules must contain safe module names")
        if min(
            self.nodes,
            self.walltime_minutes,
            self.poll_interval_s,
            self.poll_timeout_s,
        ) < 1:
            raise ValueError("Remote scheduler and polling limits must be positive")

    def platform_profile(self) -> dict[str, Any]:
        """Fixed infrastructure values exposed to the script-generating LLM."""
        return {
            "hardware_id": self.hardware_id,
            "queue": self.queue,
            "nodes": self.nodes,
            "walltime_minutes": self.walltime_minutes,
            "module_path": self.module_path,
            "modules": list(self.modules),
            "conda_env": self.conda_env,
            "accelerator_device": self.device,
            "power_vendor": self.vendor,
            "power_devices": self.devices,
            "power_interval_s": self.power_interval_s,
            "power_monitor_script": self.remote_monitor_script,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _localize_measurement_paths(measurements_path: Path, results_root: Path) -> str:
    """Replace remote log/telemetry paths with their downloaded local equivalents."""
    with measurements_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError("Downloaded measurements.csv has no header")
    for row in rows:
        local_run = results_root / "runs" / row["run_id"]
        row["log_path"] = str(local_run / "run.log")
        row["power_trace_path"] = str(local_run / "power.csv")
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    content = output.getvalue()
    measurements_path.write_text(content, encoding="utf-8")
    return content


def _run(command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Command failed ({command[0]}): {detail}")
    return result


def _ssh_options(control_path: Path) -> list[str]:
    return [
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=600",
        "-o", f"ControlPath={control_path}",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=6",
    ]


def _ensure_control_master(
    config: RemoteExecutorConfig, options: list[str]
) -> None:
    """Open one visible interactive login, then reuse it for SSH and SCP."""
    check = subprocess.run(
        ["ssh", *options, "-O", "check", config.host],
        text=True,
        capture_output=True,
        check=False,
    )
    if check.returncode == 0:
        return
    connected = subprocess.run(
        ["ssh", *options, "-M", "-N", "-f", config.host],
        text=True,
        check=False,
    )
    if connected.returncode != 0:
        raise RuntimeError("Could not establish the interactive JLSE SSH connection")


def _ssh(config: RemoteExecutorConfig, options: list[str], script: str) -> list[str]:
    return ["ssh", *options, config.host, "bash", "-lc", shlex.quote(script)]


def _job_script(config: RemoteExecutorConfig, remote_bundle: str) -> str:
    values = {
        "bundle": shlex.quote(remote_bundle),
        "module_path": shlex.quote(config.module_path),
        "conda_env": shlex.quote(config.conda_env),
        "app": shlex.quote(
            remote_bundle + "/application"
            if config.upload_application
            else config.remote_application_path
        ),
    }
    device_probe = (
        "import torch; assert torch.cuda.is_available(); "
        "x=torch.empty(1, device='cuda'); print(torch.cuda.get_device_name(0), x.device)"
        if config.device == "cuda"
        else "import torch; assert torch.xpu.is_available(); "
        "x=torch.empty(1, device='xpu'); print(torch.xpu.get_device_name(0), x.device)"
    )
    module_loads = "\n".join(
        f"module load {shlex.quote(module)}" for module in config.modules
    )
    return f"""#!/bin/bash
set -euo pipefail
BUNDLE_ROOT={values['bundle']}
MODULE_PATH={values['module_path']}
export CONDA_ENV={values['conda_env']}

set +u
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
module use "$MODULE_PATH"
{module_loads}
hash -r
MODULE_PYTHON="$(command -v python)"
CONDA_BASE="$(dirname "$(dirname "$MODULE_PYTHON")")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
hash -r
set -u

PYTHON_BIN="${{PYTHON_BIN:-$(command -v python)}}"
APP_ROOT={values['app']}
export PYTHONPATH="$APP_ROOT/src${{PYTHONPATH:+:$PYTHONPATH}}"
export APPLICATION_ROOT="$APP_ROOT"
export POWER_MONITOR_SCRIPT={shlex.quote(config.remote_monitor_script)}
export ACCELERATOR_DEVICE={shlex.quote(config.device)}
export POWER_VENDOR={shlex.quote(config.vendor)}
export POWER_DEVICES={shlex.quote(config.devices)}
export POWER_INTERVAL_S={shlex.quote(str(config.power_interval_s))}
cd "$BUNDLE_ROOT"
mkdir -p results
module list 2>&1 || true
export PYTHON_BIN
"$PYTHON_BIN" -c {shlex.quote(device_probe)}
bash "$BUNDLE_ROOT/benchmark_job.sh"
"""


def _build_bundle(
    dataset_script_path: Path,
    benchmark_script_path: Path,
    plan_path: Path,
    matrix_path: Path,
    application_path: Path,
    run_dir: Path,
    workflow_id: str,
    version: int,
    config: RemoteExecutorConfig,
) -> tuple[Path, dict[str, Any]]:
    output_root = run_dir / "remote_executor"
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_name = f"benchmark-v{version:03d}"
    archive = output_root / f"{bundle_name}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="remote-bundle-", dir=output_root) as temporary:
        bundle = Path(temporary) / bundle_name
        bundle.mkdir()
        shutil.copy2(dataset_script_path, bundle / "dataset_generation.sh")
        shutil.copy2(benchmark_script_path, bundle / "benchmark_job.sh")
        shutil.copy2(plan_path, bundle / "experiment_plan.yaml")
        shutil.copy2(matrix_path, bundle / "experiment_matrix.csv")
        (bundle / "dataset_generation.sh").chmod(0o755)
        (bundle / "benchmark_job.sh").chmod(0o755)
        if config.upload_application:
            ignored = shutil.ignore_patterns(
                ".git", ".venv", "venv", "__pycache__", "*.pyc", ".env*",
                "*.pem", "*.key", "node_modules", "build", "dist", "models",
            )
            shutil.copytree(
                application_path,
                bundle / "application",
                symlinks=True,
                ignore=ignored,
            )
        remote_bundle = str(
            PurePosixPath(config.remote_runs_root) / workflow_id / bundle_name
        )
        job_path = bundle / "job.sh"
        job_path.write_text(_job_script(config, remote_bundle))
        job_path.chmod(0o755)
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(bundle, arcname=bundle_name)
    with matrix_path.open(newline="", encoding="utf-8") as stream:
        run_count = sum(1 for _ in csv.DictReader(stream))
    return archive, {
        "schema_version": "0.1",
        "workflow_id": workflow_id,
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "remote_bundle": str(
            PurePosixPath(config.remote_runs_root) / workflow_id / bundle_name
        ),
        "run_count": run_count,
        "dataset_script_sha256": _sha256(dataset_script_path),
        "benchmark_script_sha256": _sha256(benchmark_script_path),
        "application_uploaded": config.upload_application,
        "remote": asdict(config),
    }


def execute_remote_benchmark(
    dataset_script_path: str | Path,
    benchmark_script_path: str | Path,
    plan_path: str | Path,
    matrix_path: str | Path,
    application_path: str | Path,
    run_dir: str | Path,
    workflow_id: str,
    version: int,
    config: RemoteExecutorConfig,
) -> tuple[str, dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    dataset_script = Path(dataset_script_path).expanduser().resolve(strict=True)
    benchmark_script = Path(benchmark_script_path).expanduser().resolve(strict=True)
    plan = Path(plan_path).expanduser().resolve(strict=True)
    matrix = Path(matrix_path).expanduser().resolve(strict=True)
    application = Path(application_path).expanduser().resolve(strict=True)
    archive, summary = _build_bundle(
        dataset_script,
        benchmark_script,
        plan,
        matrix,
        application,
        root,
        workflow_id,
        version,
        config,
    )
    remote_bundle = summary["remote_bundle"]
    remote_parent = str(PurePosixPath(remote_bundle).parent)
    remote_archive = str(PurePosixPath(remote_parent) / archive.name)
    socket_name = hashlib.sha256(f"{config.host}:{workflow_id}".encode()).hexdigest()[:16]
    control_path = Path(tempfile.gettempdir()) / f"agentic-ssh-{socket_name}"
    options = _ssh_options(control_path)

    _ensure_control_master(config, options)
    _run(_ssh(config, options, f"mkdir -p {shlex.quote(remote_parent)}"))
    _run(["scp", *options, str(archive), f"{config.host}:{remote_archive}"])
    setup = (
        f"echo {shlex.quote(summary['archive_sha256'] + '  ' + remote_archive)} "
        f"| sha256sum -c - && "
        f"mkdir -p {shlex.quote(remote_bundle)} && "
        f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_parent)} && "
        f"mkdir -p {shlex.quote(remote_bundle + '/results')}"
    )
    _run(_ssh(config, options, setup))
    submit = (
        f"cd {shlex.quote(remote_bundle)} && "
        f"{shlex.quote(config.qsub_command)} -n {config.nodes} "
        f"-t {config.walltime_minutes} -q {shlex.quote(config.queue)} --mode script "
        f"-O {shlex.quote(remote_bundle + '/results/cobalt')} ./job.sh"
    )
    submitted = _run(_ssh(config, options, submit))
    matches = re.findall(r"\b\d+\b", submitted.stdout)
    if not matches:
        raise RuntimeError(f"Could not parse Cobalt job ID from: {submitted.stdout.strip()}")
    job_id = matches[-1]
    summary["job_id"] = job_id

    deadline = time.monotonic() + config.poll_timeout_s
    while time.monotonic() < deadline:
        status = subprocess.run(
            _ssh(config, options, f"{shlex.quote(config.qstat_command)} {shlex.quote(job_id)}"),
            text=True,
            capture_output=True,
            check=False,
        )
        if status.returncode != 0:
            break
        time.sleep(config.poll_interval_s)
    else:
        raise RuntimeError(
            f"Remote job {job_id} exceeded poll_timeout_s={config.poll_timeout_s}"
        )

    local_remote_root = root / "remote_results" / f"v{version:03d}"
    if local_remote_root.exists():
        raise ValueError(f"Remote result destination already exists: {local_remote_root}")
    local_remote_root.mkdir(parents=True)
    _run(
        [
            "scp", *options, "-r",
            f"{config.host}:{remote_bundle}/results",
            str(local_remote_root),
        ]
    )
    results = local_remote_root / "results"
    completion_path = results / "completion_manifest.json"
    measurements_path = results / "measurements.csv"
    if not completion_path.is_file() or not measurements_path.is_file():
        raise RuntimeError(
            f"Remote job {job_id} ended without completion_manifest.json and measurements.csv"
        )
    completion = json.loads(completion_path.read_text())
    if int(completion.get("planned_runs", -1)) != int(summary["run_count"]):
        raise RuntimeError("Remote completion manifest run count does not match the bundle")
    localized_measurements = _localize_measurement_paths(measurements_path, results)
    summary.update(completion)
    summary.update(
        {
            "execution_backend": "ssh_scp_cobalt",
            "local_results": str(results),
            "measurements_sha256": _sha256(measurements_path),
            "completion_manifest_sha256": _sha256(completion_path),
        }
    )
    return localized_measurements, summary

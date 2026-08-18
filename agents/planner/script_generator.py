"""LLM-assisted, human-reviewed execution script drafting."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from agents.characterization.rag_store import PersistentCorpusRetriever
from agents.characterization.tools import CodebaseTools

from .runner import PlannerConfig


MEASUREMENT_HEADER = (
    "run_id,status,return_code,started_at,finished_at,wall_duration_s,command_sha256,"
    "algorithm_group_id,point_id,accelerator,repetition,scan_point_count,"
    "detector_height,detector_width,num_epochs,batch_size,dataset_id,io_load_time_s,"
    "setup_time_s,task_setup_time_s,reconstruction_run_time_s,save_time_s,total_time_s,"
    "power_trace_path,log_path"
)
_BANNED = (
    r"\brm\b",
    r"\bsudo\b",
    r"\b(?:ssh|scp|sftp|rsync)\b",
    r"\bq(?:sub|stat|del)\b",
    r"\bmodule\s+(?:load|use|purge)\b",
    r"\bconda\s+(?:activate|install|create)\b",
    r"\b(?:pip|pip3)\s+install\b",
    r"\b(?:curl|wget)\b",
    r"\beval\b",
    r"\bsource\s+<(?:\(|<)",
    r"\b(?:mkfs|dd|chown|pkill)\b",
    r"\bfind\b[^\n]*\s-delete\b",
    r"/(?:home|Users|tmp)/",
)


def _references_shell_variable(script: str, name: str) -> bool:
    return re.search(rf"\$(?:{re.escape(name)}\b|\{{{re.escape(name)}\}})", script) is not None


def validate_generated_script(script: str, *, kind: str) -> None:
    if not script.startswith("#!/"):
        raise ValueError(f"Generated {kind} script must start with a shebang")
    if "set -euo pipefail" not in script:
        raise ValueError(f"Generated {kind} script must use set -euo pipefail")
    if not _references_shell_variable(script, "BUNDLE_ROOT"):
        raise ValueError(f"Generated {kind} script must use the controlled bundle path")
    if kind == "benchmark" and not _references_shell_variable(script, "APPLICATION_ROOT"):
        raise ValueError("Generated benchmark script must use the controlled application path")
    if not _references_shell_variable(script, "PYTHON_BIN"):
        raise ValueError(f"Generated {kind} script must use the controlled Python executable")
    if len(script) > 100_000:
        raise ValueError(f"Generated {kind} script exceeds 100000 characters")
    for pattern in _BANNED:
        if re.search(pattern, script, re.IGNORECASE):
            raise ValueError(f"Generated {kind} script contains forbidden pattern: {pattern}")
    if kind == "dataset_generation" and "dataset_manifest.json" not in script:
        raise ValueError("Dataset script must write dataset_manifest.json")
    if kind == "benchmark":
        if "dataset_generation.sh" in script:
            raise ValueError(
                "Benchmark script must consume the pre-generated dataset, not invoke "
                "dataset_generation.sh"
            )
        for required in (
            "dataset_manifest.json",
            "measurements.csv",
            "completion_manifest.json",
            "power.csv",
            "result.json",
        ):
            if required not in script:
                raise ValueError(f"Benchmark script must reference {required}")
        missing_fields = [
            field for field in MEASUREMENT_HEADER.split(",") if field not in script
        ]
        if missing_fields:
            raise ValueError(
                "Benchmark script is missing measurement fields: "
                + ", ".join(missing_fields)
            )
    syntax = subprocess.run(
        ["bash", "-n"], input=script, text=True, capture_output=True, check=False
    )
    if syntax.returncode != 0:
        raise ValueError(f"Generated {kind} script has invalid Bash syntax: {syntax.stderr.strip()}")


class ExecutionScriptAgent:
    def __init__(self, config: PlannerConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The OpenAI SDK is required for script generation") from exc
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        prompt = Path(__file__).resolve().parent / "prompts" / "execution_scripts_prompt.md"
        self.system_prompt = prompt.read_text(encoding="utf-8")
        self.operational_retriever = (
            PersistentCorpusRetriever(
                config.operational_rag_corpus_path,
                config.operational_rag_index_path,
                config.operational_rag_settings,
                source_label="operational_source",
            )
            if config.operational_rag_enabled
            and config.operational_rag_corpus_path is not None
            and config.operational_rag_index_path is not None
            else None
        )

    def generate(
        self,
        application_path: str | Path,
        characterization: dict[str, Any],
        plan: dict[str, Any],
        matrix_csv: str,
        platform_profile: dict[str, Any],
        *,
        revision_feedback: str = "",
    ) -> dict[str, Any]:
        codebase = CodebaseTools(application_path)
        hardware = " ".join(
            str(item.get("accelerator", ""))
            for item in plan.get("hardware", {}).get("targets", [])
        )
        rag_context = ""
        if self.operational_retriever is not None:
            rag_context = self.operational_retriever.render_context(
                f"JLSE execution modules environment dataset benchmark power {hardware}",
                top_k=self.config.operational_rag_top_k,
                max_chars=self.config.operational_rag_max_context_chars,
                parent_context_chars=self.config.operational_rag_parent_context_chars,
            )
        request = {
            "approved_characterization": characterization,
            "approved_experiment_plan": plan,
            "experiment_matrix_csv": matrix_csv,
            "fixed_platform_profiles": platform_profile.get("machines", []),
            "revision_feedback": revision_feedback or None,
            "required_measurement_header": MEASUREMENT_HEADER,
            "retrieved_jlse_operational_context": rag_context or None,
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": "Inspect the application and generate the two reviewed execution scripts.\n\n"
                + json.dumps(request, ensure_ascii=False, indent=2),
            },
        ]
        tools = [
            {
                "type": "function",
                "function": {key: value for key, value in schema.items() if key != "type"},
            }
            for schema in codebase.schemas
        ]
        last_validation_error: str | None = None
        for _round in range(self.config.max_tool_rounds + 1):
            response = self.client.chat.completions.create(
                model=self.config.model, messages=messages, tools=tools
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                try:
                    result = self._parse(message.content or "")
                    validate_generated_script(
                        result["dataset_generation_script"], kind="dataset_generation"
                    )
                    validate_generated_script(
                        result["benchmark_job_script"], kind="benchmark"
                    )
                except ValueError as exc:
                    last_validation_error = str(exc)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The generated scripts failed deterministic local validation: "
                                f"{last_validation_error}\n\n"
                                "Return a complete corrected JSON object containing both scripts. "
                                "Preserve the approved experiment contract and fix the reported "
                                "validation error. Do not return a patch or explanation."
                            ),
                        }
                    )
                    continue
                return result
            for call in message.tool_calls:
                try:
                    output = {
                        "ok": True,
                        "result": codebase.call(
                            call.function.name, json.loads(call.function.arguments)
                        ),
                    }
                except Exception as exc:
                    output = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                )
        detail = (
            f" Last validation error: {last_validation_error}"
            if last_validation_error
            else ""
        )
        raise RuntimeError(
            f"Execution script agent exceeded its tool-round limit.{detail}"
        )

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = "\n".join(candidate.splitlines()[1:-1])
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Script agent returned invalid JSON: {exc}") from exc
        if not isinstance(result, dict):
            raise ValueError("Script agent output must be a JSON object")
        for key in ("dataset_generation_script", "benchmark_job_script"):
            if not isinstance(result.get(key), str) or not result[key].strip():
                raise ValueError(f"Script agent output is missing {key}")
        result["assumptions"] = list(result.get("assumptions") or [])
        result["application_interfaces"] = list(result.get("application_interfaces") or [])
        return result

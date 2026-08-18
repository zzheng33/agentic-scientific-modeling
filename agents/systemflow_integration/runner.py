"""Agent-drafted application mapping and deterministic SystemFlow validation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from agents.characterization.runner import CharacterizationConfig
from agents.characterization.tools import CodebaseTools


SOURCES = {"message.fields", "message.properties", "component.parameters"}
DESTINATIONS = {"message.fields", "message.properties", "host.properties"}
TRANSFORMS = {"identity", "int", "float", "tuple_int", "list_int", "str"}
_APPLICATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy an immutable deployment asset without exposing a partial file."""
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != _sha256(source):
            raise ValueError(
                f"SystemFlow deployment asset already exists with different content: "
                f"{destination}"
            )
        return
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def publish_systemflow_application_model(
    systemflow_root: str | Path,
    application_id: str,
    model_path: str | Path,
    mapping_path: str | Path,
    integration_report_path: str | Path,
    *,
    scientific_use: bool,
) -> dict[str, Any]:
    """Publish approved integration assets into SystemFlow's loadable data tree."""
    if not _APPLICATION_ID.fullmatch(application_id):
        raise ValueError(f"Invalid SystemFlow application_id: {application_id!r}")

    root = Path(systemflow_root).expanduser().resolve(strict=True)
    runtime = root / "systemflow" / "application_models.py"
    if not runtime.is_file():
        raise ValueError(f"SystemFlow generic application runtime is missing: {runtime}")

    sources = {
        "model": Path(model_path).expanduser().resolve(strict=True),
        "mapping": Path(mapping_path).expanduser().resolve(strict=True),
        "integration_report": Path(integration_report_path).expanduser().resolve(strict=True),
    }
    if any(not path.is_file() for path in sources.values()):
        raise ValueError("Every SystemFlow deployment source must be a file")

    destination = root / "systemflow" / "application_model_data" / application_id
    destination.mkdir(parents=True, exist_ok=True)
    assets: dict[str, dict[str, str]] = {}
    for name, source in sources.items():
        deployed = destination / source.name
        source_hash = _sha256(source)
        if deployed.exists() and (
            not deployed.is_file() or _sha256(deployed) != source_hash
        ):
            # Artifact versions are local to a workflow, so another workflow can
            # legitimately produce different v001 content for the same app.
            deployed = destination / f"{source.stem}.{source_hash[:12]}{source.suffix}"
        _atomic_copy(source, deployed)
        assets[name] = {"path": deployed.name, "sha256": _sha256(deployed)}

    manifest = {
        "schema_version": "systemflow-application-deployment-0.1",
        "application_id": application_id,
        "runtime": "systemflow.application_models",
        "scientific_use": bool(scientific_use),
        "assets": assets,
    }
    manifest_path = destination / "manifest.json"
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=".manifest.", dir=destination)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


class SystemFlowIntegrationAgent:
    def __init__(self, config: CharacterizationConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The OpenAI SDK is required; install setup/requirements.txt") from exc
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        module_dir = Path(__file__).resolve().parent
        self.system_prompt = (module_dir / "prompts" / "system_prompt.md").read_text(
            encoding="utf-8"
        )
        self.output_schema = (module_dir / "mapping_schema.yaml").read_text(
            encoding="utf-8"
        )

    def draft_mapping(
        self,
        application_root: str | Path,
        approved_characterization: dict[str, Any],
        approved_model: dict[str, Any],
        experiment_plan: dict[str, Any],
        *,
        revision_feedback: str = "",
    ) -> dict[str, Any]:
        codebase = CodebaseTools(application_root)
        instructions = (
            self.system_prompt
            + "\n\n# Required output schema\n\n```yaml\n"
            + self.output_schema
            + "\n```\n"
        )
        request = {
            "approved_characterization": approved_characterization,
            "approved_resource_model_contract": {
                "schema_version": approved_model.get("schema_version"),
                "model_inputs": approved_model.get("model_inputs"),
                "grouping": approved_model.get("grouping"),
                "fitted_targets": sorted(
                    approved_model.get("groups", [{}])[0].get("targets", {})
                ),
                "supported_domain": approved_model.get("supported_domain"),
            },
            "experiment_plan_context": {
                "application_name": experiment_plan.get("source_characterization", {}).get(
                    "application_name"
                ),
                "algorithm_groups": experiment_plan.get("algorithm_groups", []),
                "hardware_targets": experiment_plan.get("hardware", {}).get("targets", []),
            },
            "systemflow_runtime_contract": {
                "generic_loader": "systemflow.application_models.WorkflowApplicationResourceModel",
                "generic_mutation": "systemflow.application_models.ScientificApplicationModel",
                "mapping_is_declarative": True,
                "domain_specific_graph_required": False,
            },
            "human_revision_feedback": revision_feedback.strip() or None,
        }
        mapping = self._run_tool_loop(
            codebase,
            instructions,
            "Draft the generic SystemFlow application mapping for this reviewed request:\n\n"
            + json.dumps(request, indent=2, ensure_ascii=False),
        )
        validate_mapping(mapping, approved_model)
        return mapping

    def _run_tool_loop(
        self,
        codebase: CodebaseTools,
        instructions: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_prompt},
        ]
        tools = [
            {
                "type": "function",
                "function": {key: value for key, value in schema.items() if key != "type"},
            }
            for schema in codebase.schemas
        ]
        for _round in range(self.config.max_tool_rounds + 1):
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools,
            )
            if not response.choices:
                raise RuntimeError("Model returned no completion choices")
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            calls = message.tool_calls or []
            if not calls:
                if not isinstance(message.content, str) or not message.content.strip():
                    raise ValueError("Model returned neither tool calls nor mapping output")
                candidate = message.content.strip()
                if candidate.startswith("```"):
                    lines = candidate.splitlines()
                    candidate = "\n".join(lines[1:-1])
                    if candidate.lstrip().startswith("json"):
                        candidate = candidate.lstrip()[4:].lstrip()
                document = json.loads(candidate)
                if not isinstance(document, dict):
                    raise ValueError("SystemFlow mapping output must be one JSON object")
                return document
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments)
                    result = codebase.call(call.function.name, arguments)
                    output = {"ok": True, "result": result}
                except Exception as exc:
                    output = {
                        "ok": False,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                )
        raise RuntimeError(f"Integration agent exceeded {self.config.max_tool_rounds} rounds")


def validate_mapping(mapping: dict[str, Any], model: dict[str, Any]) -> None:
    required = {
        "schema_version", "status", "application_id", "component_name",
        "model_input_mapping", "group_mapping", "output_mapping",
        "metadata_output_key", "assumptions", "requested_decisions",
    }
    missing = sorted(required - mapping.keys())
    if missing:
        raise ValueError(f"SystemFlow mapping is missing keys: {', '.join(missing)}")
    if mapping["schema_version"] != "systemflow-application-mapping-0.1":
        raise ValueError("Unsupported SystemFlow application mapping schema")
    if mapping["status"] != "awaiting_human_review":
        raise ValueError("Agent-generated SystemFlow mapping must await human review")
    expected_inputs = set(model.get("model_inputs", []))
    expected_groups = set(model.get("grouping", []))
    expected_targets = set(model.get("groups", [{}])[0].get("targets", {}))
    if set(mapping["model_input_mapping"]) != expected_inputs:
        raise ValueError("SystemFlow mapping must cover exactly the fitted model inputs")
    if set(mapping["group_mapping"]) != expected_groups:
        raise ValueError("SystemFlow mapping must cover exactly the fitted grouping keys")
    if set(mapping["output_mapping"]) != expected_targets:
        raise ValueError("SystemFlow mapping must cover exactly the fitted targets")
    for collection_name in ("model_input_mapping", "group_mapping"):
        for identifier, specification in mapping[collection_name].items():
            if specification.get("source") not in SOURCES:
                raise ValueError(f"Invalid source for {identifier}")
            if specification.get("transform", "identity") not in TRANSFORMS:
                raise ValueError(f"Invalid transform for {identifier}")
            if not str(specification.get("key", "")).strip():
                raise ValueError(f"Missing SystemFlow key for {identifier}")
    output_keys: set[tuple[str, str]] = set()
    for target, specification in mapping["output_mapping"].items():
        destination = specification.get("destination")
        key = str(specification.get("key", "")).strip()
        if destination not in DESTINATIONS or not key:
            raise ValueError(f"Invalid output mapping for {target}")
        identity = (destination, key)
        if identity in output_keys:
            raise ValueError(f"Duplicate mapped output destination/key: {identity}")
        output_keys.add(identity)


def prepare_systemflow_model(
    approved_model: dict[str, Any],
    experiment_plan: dict[str, Any],
) -> dict[str, Any]:
    document = copy.deepcopy(approved_model)
    document["status"] = "approved_model_mapped_for_systemflow"
    document["group_aliases"] = {}
    plan_inputs = {
        item.get("input_id") for item in experiment_plan.get("variables", [])
    }
    known_aliases = {"scan_point_count": "n_scan_points"}
    document["input_aliases"] = {
        input_id: (
            input_id
            if input_id in plan_inputs
            else known_aliases.get(input_id, input_id)
        )
        for input_id in document.get("model_inputs", [])
    }
    document["systemflow_contract"] = {
        "loader": "systemflow.application_models.WorkflowApplicationResourceModel",
        "mutation": "systemflow.application_models.ScientificApplicationModel",
        "mapping_schema": "systemflow-application-mapping-0.1",
    }
    return document


def render_systemflow_json(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _value_for_group(group: dict[str, Any], name: str) -> Any:
    if name not in group:
        raise ValueError(f"Fitted model group is missing selector {name}")
    return group[name]


def validate_systemflow_integration(
    model_path: str | Path,
    mapping: dict[str, Any],
    experiment_plan: dict[str, Any],
    systemflow_root: str | Path,
) -> dict[str, Any]:
    root = Path(systemflow_root).expanduser().resolve()
    if not (root / "systemflow" / "node.py").is_file():
        raise ValueError(f"SystemFlow runtime is missing: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        from systemflow.application_models import (
            ApplicationInputSource,
            ScientificApplicationModel,
            WorkflowApplicationResourceModel,
        )
        runtime_source = "systemflow.application_models"
    except ModuleNotFoundError as exc:
        if exc.name != "systemflow.application_models":
            raise
        from .compat_runtime import (
            ApplicationInputSource,
            ScientificApplicationModel,
            WorkflowApplicationResourceModel,
        )
        runtime_source = "agents.systemflow_integration.compat_runtime"
    from systemflow.node import Component, DefaultLink, ExecutionGraph

    resource_model = WorkflowApplicationResourceModel(model_path)
    validate_mapping(mapping, resource_model.document)
    base_points = experiment_plan.get("matrix_design", {}).get("base_points", [])
    if not base_points:
        raise ValueError("Experiment plan has no validation base points")
    predictions = []
    for group in resource_model.groups:
        for point in base_points:
            point_inputs = point["inputs"]
            input_aliases = resource_model.document.get("input_aliases", {})
            raw_inputs = {
                input_id: point_inputs[input_aliases.get(input_id, input_id)]
                for input_id in resource_model.model_inputs
            }
            raw_groups = {
                name: _value_for_group(group, name)
                for name in resource_model.grouping
            }
            source_parameters: dict[str, Any] = {}
            application_parameters: dict[str, Any] = {}
            for values, collection in (
                (raw_inputs, mapping["model_input_mapping"]),
                (raw_groups, mapping["group_mapping"]),
            ):
                for identifier, specification in collection.items():
                    if specification["source"].startswith("message."):
                        source_parameters[specification["key"]] = values[identifier]
                    else:
                        application_parameters[specification["key"]] = values[identifier]
            source = Component(
                "Application inputs",
                [ApplicationInputSource(mapping)],
                parameters=source_parameters,
            )
            application = Component(
                mapping["component_name"],
                [ScientificApplicationModel(resource_model, mapping)],
                parameters=application_parameters,
            )
            graph = ExecutionGraph(
                f"{mapping['application_id']} resource simulation",
                [source, application],
                [DefaultLink("inputs to application", source.name, application.name)],
            )
            result = graph()
            application_result = result.get_node(mapping["component_name"])
            values_by_target: dict[str, float] = {}
            for target, specification in mapping["output_mapping"].items():
                destination, key = specification["destination"], specification["key"]
                if destination == "message.fields":
                    value = result.root_node.output_msg.fields[key]
                elif destination == "message.properties":
                    value = result.root_node.output_msg.properties[key]
                else:
                    value = application_result.properties[key]
                numeric = float(value)
                if numeric <= 0:
                    raise ValueError(
                        f"Non-positive prediction for {target}/{point['point_id']}"
                    )
                values_by_target[target] = numeric
            predictions.append(
                {
                    "group": {name: group[name] for name in resource_model.grouping},
                    "point_id": point["point_id"],
                    "targets": values_by_target,
                }
            )
    return {
        "schema_version": "0.2",
        "status": "awaiting_human_review",
        "systemflow_root": str(root),
        "model_path": str(Path(model_path).resolve()),
        "application_id": mapping["application_id"],
        "component_name": mapping["component_name"],
        "loader": "WorkflowApplicationResourceModel",
        "mutation": "ScientificApplicationModel",
        "runtime_source": runtime_source,
        "execution_graph_count": len(predictions),
        "validation": {
            "generic_model_loaded": True,
            "mapping_contract_valid": True,
            "actual_execution_graphs_executed": True,
            "all_predictions_positive": True,
            "all_model_groups_covered": True,
            "domain_specific_adapter_required": False,
        },
        "predictions": predictions,
    }

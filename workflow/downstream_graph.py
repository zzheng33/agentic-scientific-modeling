"""Post-planning benchmark, modeling, mapping, and SystemFlow graph nodes."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agents.modeling.runner import (
    extract_measurements,
    fit_resource_model,
    render_model_json,
)
from agents.planner.runner import PlannerConfig
from agents.planner.script_generator import ExecutionScriptAgent
from agents.remote_executor.runner import RemoteExecutorConfig, execute_remote_benchmark
from agents.systemflow_integration.runner import (
    SystemFlowIntegrationAgent,
    prepare_systemflow_model,
    render_systemflow_json,
    validate_mapping,
    validate_systemflow_integration,
)
from agents.characterization.runner import CharacterizationConfig
from schemas.artifacts import ArtifactRef, EditedArtifactRef, Provenance
from schemas.review import ReviewSubmission
from workflow.artifacts import ArtifactStore, utc_now
from workflow.state import WorkflowState


BENCHMARK_STAGE = "benchmark_run_review"
VALIDATION_STAGE = "measurement_validation_review"
MODEL_STAGE = "resource_model_review"
MAPPING_STAGE = "systemflow_mapping_review"
INTEGRATION_STAGE = "systemflow_integration_review"


def _store(state: WorkflowState) -> ArtifactStore:
    return ArtifactStore(state["run_dir"])


def _ref(state: WorkflowState, key: str) -> ArtifactRef:
    if not state.get(key):
        raise ValueError(f"Workflow state is missing required artifact reference: {key}")
    return ArtifactRef.model_validate(state[key])


def benchmark_entry(_state: WorkflowState) -> dict[str, Any]:
    return {
        "current_stage": "benchmark_preparation",
        "workflow_status": "benchmark_preparing",
        "route": None,
    }


def _planned_machine_aliases(plan: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for target in plan.get("hardware", {}).get("targets", []):
        accelerator = str(target.get("accelerator", "")).strip()
        if not accelerator or accelerator in aliases:
            raise ValueError("Plan hardware targets require unique accelerator values")
        aliases[accelerator] = str(target.get("hardware_id") or accelerator).strip()
    if not aliases:
        raise ValueError("Plan does not contain hardware targets")
    return aliases


def _filter_matrix_for_machine(
    matrix_path: Path,
    accelerator: str,
    hardware_alias: str,
) -> str:
    normalized = _normalize_matrix_accelerators(
        matrix_path,
        {accelerator: hardware_alias},
        allow_other_aliases=True,
    )
    reader = csv.DictReader(io.StringIO(normalized))
    fieldnames = list(reader.fieldnames or [])
    rows = [row for row in reader if row.get("accelerator") == accelerator]
    if not rows:
        raise ValueError(f"Experiment matrix has no runs for {accelerator}")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _normalize_matrix_accelerators(
    matrix_path: Path,
    aliases: dict[str, str],
    *,
    allow_other_aliases: bool = False,
) -> str:
    with matrix_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        source_fields = list(reader.fieldnames or [])
        if "accelerator" in source_fields:
            source_key = "accelerator"
        elif "hardware_id" in source_fields:
            source_key = "hardware_id"
        else:
            raise ValueError("Experiment matrix is missing accelerator")
        fieldnames = [
            "accelerator" if field == source_key else field
            for field in source_fields
        ]
        alias_to_accelerator = {alias: name for name, alias in aliases.items()}
        rows = []
        for source in reader:
            raw = str(source.get(source_key, ""))
            resolved = raw if source_key == "accelerator" else alias_to_accelerator.get(raw)
            if resolved is None:
                if allow_other_aliases:
                    continue
                raise ValueError(f"Experiment matrix contains unknown hardware alias {raw}")
            row = {
                ("accelerator" if key == source_key else key): value
                for key, value in source.items()
            }
            row["accelerator"] = resolved
            rows.append(row)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _combine_measurement_csv(documents: list[str]) -> str:
    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    for document in documents:
        reader = csv.DictReader(io.StringIO(document))
        current = list(reader.fieldnames or [])
        if fieldnames is None:
            fieldnames = current
        elif current != fieldnames:
            raise ValueError("Machine measurement CSV headers do not match")
        rows.extend(reader)
    if not fieldnames:
        raise ValueError("Remote execution returned no measurement CSV header")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def prepare_benchmark(state: WorkflowState) -> dict[str, Any]:
    store = _store(state)
    plan_ref = _ref(state, "approved_experiment_plan_ref")
    matrix_ref = _ref(state, "experiment_matrix_ref")
    plan = store.read_artifact(plan_ref)
    characterization = store.read_artifact(_ref(state, "approved_characterization_ref"))
    matrix_path = store.verify_artifact(matrix_ref)
    remotes = RemoteExecutorConfig.all_from_file(state.get("config_path"))
    aliases = _planned_machine_aliases(plan)
    matrix_csv = _normalize_matrix_accelerators(matrix_path, aliases)
    configured_accelerators = {item.accelerator for item in remotes}
    if set(aliases) != configured_accelerators:
        raise ValueError(
            "Plan accelerators must exactly match configured [[machine]] profiles; "
            f"configured {sorted(configured_accelerators)}, plan {sorted(aliases)}"
        )
    generated = ExecutionScriptAgent(
        PlannerConfig.from_file(state.get("config_path"))
    ).generate(
        state["application_path"],
        characterization,
        plan,
        matrix_csv,
        {"machines": [item.platform_profile() for item in remotes]},
        revision_feedback=state.get("benchmark_feedback") or "",
    )
    version = store.next_artifact_version(
        BENCHMARK_STAGE,
        minimum=int(state.get("benchmark_revision", 0)) + 1,
    )
    dataset_script_ref = store.write_text_artifact(
        stage=BENCHMARK_STAGE,
        artifact_type="dataset_generation_script",
        version=version,
        extension="sh",
        content=generated["dataset_generation_script"].rstrip() + "\n",
    )
    benchmark_script_ref = store.write_text_artifact(
        stage=BENCHMARK_STAGE,
        artifact_type="benchmark_job_script",
        version=version,
        extension="sh",
        content=generated["benchmark_job_script"].rstrip() + "\n",
    )
    with matrix_path.open(newline="", encoding="utf-8") as stream:
        run_count = sum(1 for _ in csv.DictReader(stream))
    manifest = {
        "schema_version": "0.2",
        "status": "awaiting_human_review",
        "execution_backend": "remote_llm_scripts",
        "run_count": run_count,
        "approved_plan": plan_ref.model_dump(mode="json"),
        "experiment_matrix": matrix_ref.model_dump(mode="json"),
        "application_path": state["application_path"],
        "application_revision": state.get("source_revision"),
        "platform_profiles": [item.platform_profile() for item in remotes],
        "scripts": {
            "dataset_generation": dataset_script_ref.model_dump(mode="json"),
            "benchmark_job": benchmark_script_ref.model_dump(mode="json"),
        },
        "application_interfaces": generated["application_interfaces"],
        "assumptions": generated["assumptions"],
        "human_feedback": state.get("benchmark_feedback"),
        "validation": {
            "execution_ready": True,
            "scripts_require_human_review": True,
            "issues": [],
        },
    }
    manifest_ref = store.write_artifact(
        stage=BENCHMARK_STAGE,
        artifact_type="benchmark_run_manifest",
        version=version,
        payload=manifest,
    )
    return {
        "artifact_ref": manifest_ref.model_dump(mode="json"),
        "benchmark_manifest_ref": manifest_ref.model_dump(mode="json"),
        "dataset_generation_script_ref": dataset_script_ref.model_dump(mode="json"),
        "benchmark_job_script_ref": benchmark_script_ref.model_dump(mode="json"),
        "benchmark_revision": version,
        "benchmark_feedback": None,
        "workflow_status": "benchmark_scripts_ready",
        "current_stage": BENCHMARK_STAGE,
    }


def _prepare_review(
    state: WorkflowState,
    artifact_key: str,
    *,
    agent_version: str,
    tool_version: str,
) -> dict[str, Any]:
    artifact = _ref(state, artifact_key)
    store = _store(state)
    review_path = store.write_review_template(
        workflow_id=state["workflow_id"],
        artifact=artifact,
        provenance=Provenance(
            source_revision=state.get("source_revision"),
            generated_at=utc_now(),
            agent_version=agent_version,
            prompt_version=None,
            tool_version=tool_version,
        ),
    )
    return {
        "artifact_ref": artifact.model_dump(mode="json"),
        "pending_review": {
            "workflow_id": state["workflow_id"],
            "stage": artifact.stage,
            "artifact": artifact.model_dump(mode="json"),
            "review_path": store.relative_path(review_path),
        },
        "submitted_review": None,
        "workflow_status": "awaiting_review",
    }


def prepare_benchmark_review(state: WorkflowState) -> dict[str, Any]:
    return _prepare_review(
        state, "benchmark_manifest_ref",
        agent_version="execution-script-agent-0.2", tool_version="script-validator-0.2"
    )


def downstream_review_gate(state: WorkflowState) -> dict[str, Any]:
    review = ReviewSubmission.model_validate(interrupt(state["pending_review"]))
    return {
        "submitted_review": review.model_dump(mode="json"),
        "workflow_status": "review_received",
    }


def _reviewed_artifact(
    state: WorkflowState,
    current_key: str,
) -> tuple[ReviewSubmission, ArtifactRef | None]:
    review = ReviewSubmission.model_validate(state["submitted_review"])
    current = _ref(state, current_key)
    store = _store(state)
    approved: ArtifactRef | None = None
    if review.decision == "approve":
        approved = current
    elif review.decision == "edit":
        edited = EditedArtifactRef.model_validate(review.edited_artifact)
        edited_path = store.verify_edited_file(edited.path, edited.sha256)
        approved = store.ingest_edit(current, edited_path)
    store.write_review_record(
        review=state["submitted_review"], approved_artifact=approved
    )
    return review, approved


def apply_benchmark_review(state: WorkflowState) -> dict[str, Any]:
    review, approved = _reviewed_artifact(state, "benchmark_manifest_ref")
    common = {
        "pending_review": None,
        "review_history": [review.model_dump(mode="json")],
    }
    if approved is None:
        return {
            **common,
            "route": "revise_benchmark",
            "benchmark_feedback": review.feedback,
            "workflow_status": "benchmark_needs_revision",
        }
    return {
        **common,
        "benchmark_manifest_ref": approved.model_dump(mode="json"),
        "approved_benchmark_manifest_ref": approved.model_dump(mode="json"),
        "benchmark_revision": approved.version,
        "route": "benchmark_ready",
        "current_stage": "benchmark_ready",
        "workflow_status": "benchmark_ready",
    }


def route_after_benchmark_review(state: WorkflowState) -> str:
    return str(state["route"])


def remote_executor(state: WorkflowState) -> dict[str, Any]:
    """Upload an approved bundle, submit Cobalt, and ingest downloaded results."""
    store = _store(state)
    manifest = store.read_artifact(_ref(state, "approved_benchmark_manifest_ref"))
    scripts = manifest.get("scripts", {})
    dataset_script_ref = ArtifactRef.model_validate(scripts.get("dataset_generation"))
    benchmark_script_ref = ArtifactRef.model_validate(scripts.get("benchmark_job"))
    if (
        dataset_script_ref.stage != BENCHMARK_STAGE
        or dataset_script_ref.artifact_type != "dataset_generation_script"
        or benchmark_script_ref.stage != BENCHMARK_STAGE
        or benchmark_script_ref.artifact_type != "benchmark_job_script"
    ):
        raise ValueError("Approved benchmark manifest contains invalid script references")
    dataset_script_path = store.verify_artifact(dataset_script_ref)
    benchmark_script_path = store.verify_artifact(benchmark_script_ref)
    plan_ref = _ref(state, "approved_experiment_plan_ref")
    plan_path = store.verify_artifact(plan_ref)
    plan = store.read_artifact(plan_ref)
    matrix_path = store.verify_artifact(_ref(state, "experiment_matrix_ref"))
    benchmark_version = int(state["benchmark_revision"])
    execution_version = store.next_artifact_version(
        "benchmark_execution", minimum=benchmark_version
    )
    configurations = RemoteExecutorConfig.all_from_file(state.get("config_path"))
    aliases = _planned_machine_aliases(plan)
    measurement_documents: list[str] = []
    execution_summaries: list[dict[str, Any]] = []
    for configuration in configurations:
        if configuration.accelerator not in aliases:
            raise ValueError(
                f"Approved plan does not target configured accelerator {configuration.accelerator}"
            )
        machine_matrix = store.write_text_artifact(
            stage="benchmark_execution",
            artifact_type=f"experiment_matrix_{configuration.accelerator.lower()}",
            version=execution_version,
            extension="csv",
            content=_filter_matrix_for_machine(
                matrix_path,
                configuration.accelerator,
                aliases[configuration.accelerator],
            ),
        )
        machine_measurements, machine_summary = execute_remote_benchmark(
            dataset_script_path,
            benchmark_script_path,
            plan_path,
            store.verify_artifact(machine_matrix),
            state["application_path"],
            state["run_dir"],
            state["workflow_id"],
            benchmark_version,
            configuration,
        )
        machine_summary["matrix_artifact"] = machine_matrix.model_dump(mode="json")
        measurement_documents.append(machine_measurements)
        execution_summaries.append(machine_summary)
    measurements_csv = _combine_measurement_csv(measurement_documents)
    summary = {
        "schema_version": "0.2",
        "status": (
            "completed"
            if all(item.get("status") == "completed" for item in execution_summaries)
            else "completed_with_failures"
        ),
        "execution_backend": "ssh_scp_cobalt_multi_machine",
        "executions": execution_summaries,
    }
    measurements_ref = store.write_text_artifact(
        stage="benchmark_execution",
        artifact_type="raw_measurements",
        version=execution_version,
        extension="csv",
        content=measurements_csv,
    )
    summary.update(
        {
            "approved_benchmark_manifest": _ref(
                state, "approved_benchmark_manifest_ref"
            ).model_dump(mode="json"),
            "planned_runs": int(manifest["run_count"]),
            "benchmark_revision": benchmark_version,
            "execution_revision": execution_version,
            "measurements_artifact": measurements_ref.model_dump(mode="json"),
        }
    )
    summary_ref = store.write_artifact(
        stage="benchmark_execution",
        artifact_type="remote_execution_summary",
        version=execution_version,
        payload=summary,
    )
    return {
        "measurements_ref": measurements_ref.model_dump(mode="json"),
        "benchmark_execution_ref": summary_ref.model_dump(mode="json"),
        "remote_execution_ref": summary_ref.model_dump(mode="json"),
        "workflow_status": summary["status"],
        "current_stage": "remote_execution_complete",
        "validation_revision": int(state.get("validation_revision", 0)),
    }


def extract_and_validate_measurements(state: WorkflowState) -> dict[str, Any]:
    store = _store(state)
    measurements_path = store.verify_artifact(_ref(state, "measurements_ref"))
    benchmark = store.read_artifact(_ref(state, "approved_benchmark_manifest_ref"))
    extracted_csv, validation = extract_measurements(
        measurements_path, int(benchmark["run_count"])
    )
    version = int(state.get("validation_revision", 0)) + 1
    extracted_ref = store.write_text_artifact(
        stage=VALIDATION_STAGE,
        artifact_type="modeling_runs_extracted",
        version=version,
        extension="csv",
        content=extracted_csv,
    )
    validation["extracted_runs_artifact"] = extracted_ref.model_dump(mode="json")
    validation_ref = store.write_artifact(
        stage=VALIDATION_STAGE,
        artifact_type="measurement_validation",
        version=version,
        payload=validation,
    )
    return {
        "artifact_ref": validation_ref.model_dump(mode="json"),
        "extracted_runs_ref": extracted_ref.model_dump(mode="json"),
        "measurement_validation_ref": validation_ref.model_dump(mode="json"),
        "validation_revision": version,
        "current_stage": VALIDATION_STAGE,
        "workflow_status": "measurement_validation_ready",
    }


def prepare_validation_review(state: WorkflowState) -> dict[str, Any]:
    return _prepare_review(
        state, "measurement_validation_ref",
        agent_version="data-validation-0.1", tool_version="measurement-parser-0.1"
    )


def apply_validation_review(state: WorkflowState) -> dict[str, Any]:
    review, approved = _reviewed_artifact(state, "measurement_validation_ref")
    common = {"pending_review": None, "review_history": [review.model_dump(mode="json")]}
    if approved is None:
        return {
            **common,
            "route": "validation_stop",
            "validation_feedback": review.feedback,
            "workflow_status": "measurements_need_attention",
        }
    return {
        **common,
        "measurement_validation_ref": approved.model_dump(mode="json"),
        "approved_validation_ref": approved.model_dump(mode="json"),
        "validation_revision": approved.version,
        "route": "fit_model",
        "workflow_status": "measurements_approved",
    }


def route_after_validation_review(state: WorkflowState) -> str:
    return str(state["route"])


def fit_model(state: WorkflowState) -> dict[str, Any]:
    store = _store(state)
    validation = store.read_artifact(_ref(state, "approved_validation_ref"))
    extracted_path = store.verify_artifact(_ref(state, "extracted_runs_ref"))
    model, coefficients = fit_resource_model(extracted_path, validation)
    version = int(state.get("modeling_revision", 0)) + 1
    coefficients_ref = store.write_text_artifact(
        stage=MODEL_STAGE,
        artifact_type="resource_model_coefficients",
        version=version,
        extension="csv",
        content=coefficients,
    )
    json_ref = store.write_text_artifact(
        stage=MODEL_STAGE,
        artifact_type="resource_model_definition",
        version=version,
        extension="json",
        content=render_model_json(model),
    )
    model["coefficients_artifact"] = coefficients_ref.model_dump(mode="json")
    model["json_artifact"] = json_ref.model_dump(mode="json")
    model_ref = store.write_artifact(
        stage=MODEL_STAGE,
        artifact_type="resource_model",
        version=version,
        payload=model,
    )
    return {
        "artifact_ref": model_ref.model_dump(mode="json"),
        "resource_model_ref": model_ref.model_dump(mode="json"),
        "resource_model_json_ref": json_ref.model_dump(mode="json"),
        "resource_coefficients_ref": coefficients_ref.model_dump(mode="json"),
        "modeling_revision": version,
        "current_stage": MODEL_STAGE,
        "workflow_status": "resource_model_ready",
    }


def prepare_model_review(state: WorkflowState) -> dict[str, Any]:
    return _prepare_review(
        state, "resource_model_ref",
        agent_version="resource-modeling-0.1", tool_version="ridge-model-0.1"
    )


def apply_model_review(state: WorkflowState) -> dict[str, Any]:
    review, approved = _reviewed_artifact(state, "resource_model_ref")
    common = {"pending_review": None, "review_history": [review.model_dump(mode="json")]}
    if approved is None:
        return {
            **common,
            "route": "model_stop",
            "modeling_feedback": review.feedback,
            "workflow_status": "resource_model_needs_revision",
        }
    return {
        **common,
        "resource_model_ref": approved.model_dump(mode="json"),
        "approved_resource_model_ref": approved.model_dump(mode="json"),
        "modeling_revision": approved.version,
        "route": "draft_systemflow_mapping",
        "workflow_status": "resource_model_approved",
    }


def route_after_model_review(state: WorkflowState) -> str:
    return str(state["route"])


def draft_systemflow_mapping(state: WorkflowState) -> dict[str, Any]:
    store = _store(state)
    characterization = store.read_artifact(_ref(state, "approved_characterization_ref"))
    model = store.read_artifact(_ref(state, "approved_resource_model_ref"))
    plan = store.read_artifact(_ref(state, "approved_experiment_plan_ref"))
    mapping = SystemFlowIntegrationAgent(
        CharacterizationConfig.from_file(state.get("config_path"))
    ).draft_mapping(
        state["application_path"],
        characterization,
        model,
        plan,
        revision_feedback=state.get("mapping_feedback") or "",
    )
    return {
        "working_artifact": mapping,
        "current_stage": "systemflow_mapping",
        "workflow_status": "systemflow_mapping_drafted",
        "mapping_feedback": None,
    }


def write_systemflow_mapping(state: WorkflowState) -> dict[str, Any]:
    model = _store(state).read_artifact(_ref(state, "approved_resource_model_ref"))
    validate_mapping(state["working_artifact"], model)
    version = int(state.get("mapping_revision", 0)) + 1
    reference = _store(state).write_artifact(
        stage=MAPPING_STAGE,
        artifact_type="systemflow_application_mapping",
        version=version,
        payload=state["working_artifact"],
    )
    return {
        "working_artifact": None,
        "artifact_ref": reference.model_dump(mode="json"),
        "systemflow_mapping_ref": reference.model_dump(mode="json"),
        "mapping_revision": version,
        "current_stage": MAPPING_STAGE,
        "workflow_status": "systemflow_mapping_ready",
    }


def prepare_mapping_review(state: WorkflowState) -> dict[str, Any]:
    return _prepare_review(
        state, "systemflow_mapping_ref",
        agent_version="systemflow-mapping-agent-0.1", tool_version="mapping-validator-0.1"
    )


def apply_mapping_review(state: WorkflowState) -> dict[str, Any]:
    review = ReviewSubmission.model_validate(state["submitted_review"])
    current = _ref(state, "systemflow_mapping_ref")
    store = _store(state)
    model = store.read_artifact(_ref(state, "approved_resource_model_ref"))
    approved: ArtifactRef | None = None
    if review.decision == "approve":
        approved = current
    elif review.decision == "edit":
        edited = EditedArtifactRef.model_validate(review.edited_artifact)
        edited_path = store.verify_edited_file(edited.path, edited.sha256)
        validate_mapping(store.read_yaml_file(edited_path), model)
        approved = store.ingest_edit(current, edited_path)
    store.write_review_record(
        review=state["submitted_review"], approved_artifact=approved
    )
    common = {"pending_review": None, "review_history": [review.model_dump(mode="json")]}
    if approved is None:
        return {
            **common,
            "route": "revise_mapping",
            "mapping_feedback": review.feedback,
            "workflow_status": "systemflow_mapping_needs_revision",
        }
    return {
        **common,
        "systemflow_mapping_ref": approved.model_dump(mode="json"),
        "approved_systemflow_mapping_ref": approved.model_dump(mode="json"),
        "mapping_revision": approved.version,
        "route": "integrate_systemflow",
        "workflow_status": "systemflow_mapping_approved",
    }


def route_after_mapping_review(state: WorkflowState) -> str:
    return str(state["route"])


def integrate_systemflow(state: WorkflowState) -> dict[str, Any]:
    store = _store(state)
    model = store.read_artifact(_ref(state, "approved_resource_model_ref"))
    mapping = store.read_artifact(_ref(state, "approved_systemflow_mapping_ref"))
    plan = store.read_artifact(_ref(state, "approved_experiment_plan_ref"))
    mapped = prepare_systemflow_model(model, plan)
    version = store.next_artifact_version(
        "systemflow_mapping", minimum=int(state.get("integration_revision", 0)) + 1
    )
    mapped_ref = store.write_text_artifact(
        stage="systemflow_mapping",
        artifact_type="workflow_application_resource_model",
        version=version,
        extension="json",
        content=render_systemflow_json(mapped),
    )
    systemflow_root = Path(state["run_dir"]).parents[2] / "systemflow"
    if state.get("config_path"):
        import tomllib
        with Path(state["config_path"]).open("rb") as stream:
            configured = tomllib.load(stream).get("systemflow", {}).get("path")
        if configured:
            systemflow_root = Path(configured).expanduser().resolve()
    report = validate_systemflow_integration(
        store.verify_artifact(mapped_ref), mapping, plan, systemflow_root
    )
    report["mapped_model_artifact"] = mapped_ref.model_dump(mode="json")
    report["mapping_artifact"] = _ref(
        state, "approved_systemflow_mapping_ref"
    ).model_dump(mode="json")
    report_ref = store.write_artifact(
        stage=INTEGRATION_STAGE,
        artifact_type="systemflow_integration_report",
        version=version,
        payload=report,
    )
    return {
        "artifact_ref": report_ref.model_dump(mode="json"),
        "systemflow_model_ref": mapped_ref.model_dump(mode="json"),
        "integration_report_ref": report_ref.model_dump(mode="json"),
        "integration_revision": version,
        "current_stage": INTEGRATION_STAGE,
        "workflow_status": "systemflow_integration_ready",
    }


def prepare_integration_review(state: WorkflowState) -> dict[str, Any]:
    return _prepare_review(
        state, "integration_report_ref",
        agent_version="systemflow-integration-0.1", tool_version="systemflow-adapter-0.1"
    )


def apply_integration_review(state: WorkflowState) -> dict[str, Any]:
    review, approved = _reviewed_artifact(state, "integration_report_ref")
    common = {"pending_review": None, "review_history": [review.model_dump(mode="json")]}
    if approved is None:
        return {
            **common,
            "route": "integration_stop",
            "integration_feedback": review.feedback,
            "workflow_status": "systemflow_integration_needs_revision",
        }
    return {
        **common,
        "integration_report_ref": approved.model_dump(mode="json"),
        "approved_integration_ref": approved.model_dump(mode="json"),
        "route": "complete",
        "current_stage": "complete",
        "workflow_status": "complete",
    }


def route_after_integration_review(state: WorkflowState) -> str:
    return str(state["route"])


def add_downstream_nodes(builder: StateGraph) -> None:
    builder.add_node("benchmark_entry", benchmark_entry)
    builder.add_node("prepare_benchmark", prepare_benchmark)
    builder.add_node("prepare_benchmark_review", prepare_benchmark_review)
    builder.add_node("benchmark_review_gate", downstream_review_gate)
    builder.add_node("apply_benchmark_review", apply_benchmark_review)
    builder.add_node("remote_executor", remote_executor)
    builder.add_node("extract_measurements", extract_and_validate_measurements)
    builder.add_node("prepare_validation_review", prepare_validation_review)
    builder.add_node("validation_review_gate", downstream_review_gate)
    builder.add_node("apply_validation_review", apply_validation_review)
    builder.add_node("fit_resource_model", fit_model)
    builder.add_node("prepare_model_review", prepare_model_review)
    builder.add_node("model_review_gate", downstream_review_gate)
    builder.add_node("apply_model_review", apply_model_review)
    builder.add_node("draft_systemflow_mapping", draft_systemflow_mapping)
    builder.add_node("write_systemflow_mapping", write_systemflow_mapping)
    builder.add_node("prepare_mapping_review", prepare_mapping_review)
    builder.add_node("mapping_review_gate", downstream_review_gate)
    builder.add_node("apply_mapping_review", apply_mapping_review)
    builder.add_node("integrate_systemflow", integrate_systemflow)
    builder.add_node("prepare_integration_review", prepare_integration_review)
    builder.add_node("integration_review_gate", downstream_review_gate)
    builder.add_node("apply_integration_review", apply_integration_review)

    builder.add_edge("benchmark_entry", "prepare_benchmark")
    builder.add_edge("prepare_benchmark", "prepare_benchmark_review")
    builder.add_edge("prepare_benchmark_review", "benchmark_review_gate")
    builder.add_edge("benchmark_review_gate", "apply_benchmark_review")
    builder.add_conditional_edges(
        "apply_benchmark_review", route_after_benchmark_review,
        {
            "revise_benchmark": "prepare_benchmark",
            "benchmark_ready": END,
            "remote_execute_benchmark": "remote_executor",
        },
    )
    builder.add_edge("remote_executor", "extract_measurements")
    builder.add_edge("extract_measurements", "prepare_validation_review")
    builder.add_edge("prepare_validation_review", "validation_review_gate")
    builder.add_edge("validation_review_gate", "apply_validation_review")
    builder.add_conditional_edges(
        "apply_validation_review", route_after_validation_review,
        {"validation_stop": END, "fit_model": "fit_resource_model"},
    )
    builder.add_edge("fit_resource_model", "prepare_model_review")
    builder.add_edge("prepare_model_review", "model_review_gate")
    builder.add_edge("model_review_gate", "apply_model_review")
    builder.add_conditional_edges(
        "apply_model_review", route_after_model_review,
        {"model_stop": END, "draft_systemflow_mapping": "draft_systemflow_mapping"},
    )
    builder.add_edge("draft_systemflow_mapping", "write_systemflow_mapping")
    builder.add_edge("write_systemflow_mapping", "prepare_mapping_review")
    builder.add_edge("prepare_mapping_review", "mapping_review_gate")
    builder.add_edge("mapping_review_gate", "apply_mapping_review")
    builder.add_conditional_edges(
        "apply_mapping_review", route_after_mapping_review,
        {"revise_mapping": "draft_systemflow_mapping", "integrate_systemflow": "integrate_systemflow"},
    )
    builder.add_edge("integrate_systemflow", "prepare_integration_review")
    builder.add_edge("prepare_integration_review", "integration_review_gate")
    builder.add_edge("integration_review_gate", "apply_integration_review")
    builder.add_conditional_edges(
        "apply_integration_review", route_after_integration_review,
        {"integration_stop": END, "complete": END},
    )

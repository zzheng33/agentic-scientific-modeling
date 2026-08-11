"""Typed, small LangGraph state containing artifact references rather than payloads."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class WorkflowState(TypedDict, total=False):
    workflow_id: str
    workflow_type: str
    application_path: str
    config_path: str | None
    user_context: str
    run_dir: str
    source_revision: str | None
    current_stage: str
    workflow_status: str
    working_artifact: dict[str, Any] | None
    route: str | None
    artifact_ref: dict[str, Any]
    pending_review: dict[str, Any] | None
    submitted_review: dict[str, Any] | None
    approved_artifact_ref: dict[str, Any] | None
    candidate_inputs_ref: dict[str, Any] | None
    approved_inputs_ref: dict[str, Any] | None
    characterization_ref: dict[str, Any] | None
    approved_characterization_ref: dict[str, Any] | None
    experiment_plan_ref: dict[str, Any] | None
    experiment_matrix_ref: dict[str, Any] | None
    approved_experiment_plan_ref: dict[str, Any] | None
    synthetic_dataset_manifest_ref: dict[str, Any] | None
    benchmark_manifest_ref: dict[str, Any] | None
    benchmark_commands_ref: dict[str, Any] | None
    approved_benchmark_manifest_ref: dict[str, Any] | None
    benchmark_execution_ref: dict[str, Any] | None
    measurements_ref: dict[str, Any] | None
    extracted_runs_ref: dict[str, Any] | None
    measurement_validation_ref: dict[str, Any] | None
    approved_validation_ref: dict[str, Any] | None
    resource_model_ref: dict[str, Any] | None
    resource_model_json_ref: dict[str, Any] | None
    resource_coefficients_ref: dict[str, Any] | None
    approved_resource_model_ref: dict[str, Any] | None
    systemflow_mapping_ref: dict[str, Any] | None
    approved_systemflow_mapping_ref: dict[str, Any] | None
    systemflow_model_ref: dict[str, Any] | None
    integration_report_ref: dict[str, Any] | None
    approved_integration_ref: dict[str, Any] | None
    input_revision: int
    characterization_revision: int
    planning_revision: int
    benchmark_revision: int
    validation_revision: int
    modeling_revision: int
    integration_revision: int
    mapping_revision: int
    input_feedback: str | None
    characterization_feedback: str | None
    planning_feedback: str | None
    benchmark_feedback: str | None
    validation_feedback: str | None
    modeling_feedback: str | None
    integration_feedback: str | None
    mapping_feedback: str | None
    rejection_feedback: str | None
    revision_count: int
    review_history: Annotated[list[dict[str, Any]], operator.add]

"""Persistent experiment-planning nodes and human-review gate."""

from __future__ import annotations

import copy
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agents.planner.runner import PlannerConfig, PlanningAgent, render_matrix_csv
from schemas.artifacts import ArtifactRef, EditedArtifactRef, Provenance
from schemas.review import ReviewSubmission
from workflow.artifacts import ArtifactStore, utc_now
from workflow.state import WorkflowState


PLANNING_STAGE = "experiment_plan_review"


def _store(state: WorkflowState) -> ArtifactStore:
    return ArtifactStore(state["run_dir"])


def _agent(state: WorkflowState) -> PlanningAgent:
    return PlanningAgent(PlannerConfig.from_file(state.get("config_path")))


def _characterization_with_variants(state: WorkflowState) -> dict[str, Any]:
    store = _store(state)
    characterization = copy.deepcopy(
        store.read_artifact(
            ArtifactRef.model_validate(state["approved_characterization_ref"])
        )
    )
    approved_inputs = store.read_artifact(
        ArtifactRef.model_validate(state["approved_inputs_ref"])
    )
    variants = approved_inputs.get("application", {}).get("variants")
    if variants:
        characterization.setdefault("application", {})["variants"] = variants
    return characterization


def planning_entry(_state: WorkflowState) -> dict[str, Any]:
    return {
        "current_stage": "experiment_planning",
        "workflow_status": "planning",
        "route": None,
    }


def derive_experiment_plan(state: WorkflowState) -> dict[str, Any]:
    characterization = _characterization_with_variants(state)
    plan = _agent(state).plan(
        characterization,
        user_context=state.get("user_context", ""),
        revision_feedback=state.get("planning_feedback") or "",
        externally_approved=True,
    )
    return {
        "working_artifact": plan,
        "current_stage": PLANNING_STAGE,
        "workflow_status": "running",
        "submitted_review": None,
        "route": None,
    }


def _write_plan_version(
    state: WorkflowState,
    plan: dict[str, Any],
    version: int,
) -> tuple[ArtifactRef, ArtifactRef]:
    store = _store(state)
    matrix = store.write_text_artifact(
        stage=PLANNING_STAGE,
        artifact_type="dry_run_matrix",
        version=version,
        extension="csv",
        content=render_matrix_csv(plan),
    )
    plan.setdefault("matrix_design", {})["matrix_artifact"] = matrix.model_dump(mode="json")
    reference = store.write_artifact(
        stage=PLANNING_STAGE,
        artifact_type="experiment_plan",
        version=version,
        payload=plan,
    )
    return reference, matrix


def write_experiment_plan(state: WorkflowState) -> dict[str, Any]:
    version = int(state.get("planning_revision", 0)) + 1
    reference, matrix = _write_plan_version(state, state["working_artifact"], version)
    return {
        "working_artifact": None,
        "artifact_ref": reference.model_dump(mode="json"),
        "experiment_plan_ref": reference.model_dump(mode="json"),
        "experiment_matrix_ref": matrix.model_dump(mode="json"),
        "planning_revision": version,
        "planning_feedback": None,
    }


def prepare_planning_review(state: WorkflowState) -> dict[str, Any]:
    artifact = ArtifactRef.model_validate(state["experiment_plan_ref"])
    store = _store(state)
    review_path = store.write_review_template(
        workflow_id=state["workflow_id"],
        artifact=artifact,
        provenance=Provenance(
            source_revision=state.get("source_revision"),
            generated_at=utc_now(),
            agent_version="planning-graph-0.1",
            prompt_version="experiment-planning-0.2",
            tool_version="codebase-tools-0.1",
        ),
    )
    return {
        "pending_review": {
            "workflow_id": state["workflow_id"],
            "stage": artifact.stage,
            "artifact": artifact.model_dump(mode="json"),
            "review_path": store.relative_path(review_path),
        },
        "submitted_review": None,
        "workflow_status": "awaiting_review",
    }


def planning_review_gate(state: WorkflowState) -> dict[str, Any]:
    review = ReviewSubmission.model_validate(interrupt(state["pending_review"]))
    return {
        "submitted_review": review.model_dump(mode="json"),
        "workflow_status": "review_received",
    }


def apply_planning_review(state: WorkflowState) -> dict[str, Any]:
    review = ReviewSubmission.model_validate(state["submitted_review"])
    current = ArtifactRef.model_validate(state["experiment_plan_ref"])
    store = _store(state)
    approved: ArtifactRef | None = None
    matrix = (
        ArtifactRef.model_validate(state["experiment_matrix_ref"])
        if state.get("experiment_matrix_ref")
        else None
    )

    if review.decision == "approve":
        approved = current
    elif review.decision == "edit":
        edited = EditedArtifactRef.model_validate(review.edited_artifact)
        edited_path = store.verify_edited_file(edited.path, edited.sha256)
        edited_plan = store.read_yaml_file(edited_path)
        characterization = _characterization_with_variants(state)
        finalized = _agent(state)._validate_and_finalize(edited_plan, characterization)
        approved, matrix = _write_plan_version(state, finalized, current.version + 1)

    store.write_review_record(
        review=state["submitted_review"],
        approved_artifact=approved,
    )
    common: dict[str, Any] = {
        "pending_review": None,
        "review_history": [review.model_dump(mode="json")],
    }
    if approved is None:
        return {
            **common,
            "route": "revise_plan",
            "workflow_status": "needs_revision",
            "planning_feedback": review.feedback,
        }
    return {
        **common,
        "artifact_ref": approved.model_dump(mode="json"),
        "experiment_plan_ref": approved.model_dump(mode="json"),
        "experiment_matrix_ref": matrix.model_dump(mode="json") if matrix else None,
        "approved_experiment_plan_ref": approved.model_dump(mode="json"),
        "planning_revision": approved.version,
        "route": "complete",
        "current_stage": "complete",
        "workflow_status": "approved",
    }


def route_after_planning_review(state: WorkflowState) -> str:
    return state["route"]


def add_planning_nodes(builder: StateGraph) -> None:
    builder.add_node("planning_entry", planning_entry)
    builder.add_node("derive_experiment_plan", derive_experiment_plan)
    builder.add_node("write_experiment_plan", write_experiment_plan)
    builder.add_node("prepare_planning_review", prepare_planning_review)
    builder.add_node("planning_review_gate", planning_review_gate)
    builder.add_node("apply_planning_review", apply_planning_review)
    builder.add_edge("planning_entry", "derive_experiment_plan")
    builder.add_edge("derive_experiment_plan", "write_experiment_plan")
    builder.add_edge("write_experiment_plan", "prepare_planning_review")
    builder.add_edge("prepare_planning_review", "planning_review_gate")
    builder.add_edge("planning_review_gate", "apply_planning_review")
    builder.add_conditional_edges(
        "apply_planning_review",
        route_after_planning_review,
        {
            "revise_plan": "derive_experiment_plan",
            "start_benchmark": "benchmark_entry",
            "complete": END,
        },
    )

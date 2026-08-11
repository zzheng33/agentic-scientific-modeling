"""Persistent characterization graph with input and model review gates."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agents.characterization.runner import CharacterizationAgent, CharacterizationConfig
from schemas.artifacts import ArtifactRef, EditedArtifactRef, Provenance
from schemas.characterization import CandidateInputsDocument
from schemas.review import ReviewSubmission
from workflow.artifacts import ArtifactStore, utc_now
from workflow.downstream_graph import add_downstream_nodes
from workflow.planning_graph import add_planning_nodes
from workflow.state import WorkflowState


INPUT_STAGE = "candidate_inputs_review"
CHARACTERIZATION_STAGE = "characterization_review"


def _store(state: WorkflowState) -> ArtifactStore:
    return ArtifactStore(state["run_dir"])


def _agent(state: WorkflowState) -> CharacterizationAgent:
    return CharacterizationAgent(CharacterizationConfig.from_file(state.get("config_path")))


def discover_inputs(state: WorkflowState) -> dict[str, Any]:
    artifact = _agent(state).discover_inputs(
        state["application_path"],
        user_context=state.get("user_context", ""),
        revision_feedback=state.get("input_feedback") or "",
    )
    return {
        "working_artifact": artifact,
        "current_stage": INPUT_STAGE,
        "workflow_status": "running",
        "submitted_review": None,
        "route": None,
    }


def write_candidate_inputs(state: WorkflowState) -> dict[str, Any]:
    document = CandidateInputsDocument.model_validate(state["working_artifact"])
    version = int(state.get("input_revision", 0)) + 1
    artifact = _store(state).write_artifact(
        stage=INPUT_STAGE,
        artifact_type="candidate_inputs",
        version=version,
        payload=document.model_dump(mode="json"),
    )
    return {
        "working_artifact": None,
        "artifact_ref": artifact.model_dump(mode="json"),
        "candidate_inputs_ref": artifact.model_dump(mode="json"),
        "input_revision": version,
        "input_feedback": None,
    }


def _prepare_review(state: WorkflowState, *, prompt_version: str) -> dict[str, Any]:
    artifact = ArtifactRef.model_validate(state["artifact_ref"])
    store = _store(state)
    review_path = store.write_review_template(
        workflow_id=state["workflow_id"],
        artifact=artifact,
        provenance=Provenance(
            source_revision=state.get("source_revision"),
            generated_at=utc_now(),
            agent_version="characterization-graph-0.1",
            prompt_version=prompt_version,
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


def prepare_input_review(state: WorkflowState) -> dict[str, Any]:
    return _prepare_review(state, prompt_version="input-discovery-0.1")


def prepare_characterization_review(state: WorkflowState) -> dict[str, Any]:
    return _prepare_review(state, prompt_version="characterization-0.1")


def review_gate(state: WorkflowState) -> dict[str, Any]:
    submitted = interrupt(state["pending_review"])
    review = ReviewSubmission.model_validate(submitted)
    return {
        "submitted_review": review.model_dump(mode="json"),
        "workflow_status": "review_received",
    }


def apply_input_review(state: WorkflowState) -> dict[str, Any]:
    review = ReviewSubmission.model_validate(state["submitted_review"])
    current = ArtifactRef.model_validate(state["candidate_inputs_ref"])
    store = _store(state)
    approved: ArtifactRef | None = None

    if review.decision == "approve":
        approved = current
    elif review.decision == "edit":
        edited = EditedArtifactRef.model_validate(review.edited_artifact)
        edited_path = store.verify_edited_file(edited.path, edited.sha256)
        CandidateInputsDocument.model_validate(store.read_yaml_file(edited_path))
        approved = store.ingest_edit(current, edited_path)

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
            "route": "revise_inputs",
            "workflow_status": "needs_revision",
            "input_feedback": review.feedback,
        }
    return {
        **common,
        "artifact_ref": approved.model_dump(mode="json"),
        "candidate_inputs_ref": approved.model_dump(mode="json"),
        "approved_inputs_ref": approved.model_dump(mode="json"),
        "input_revision": approved.version,
        "route": "derive_characterization",
        "workflow_status": "inputs_approved",
    }


def route_after_input_review(state: WorkflowState) -> str:
    return state["route"]


def derive_characterization(state: WorkflowState) -> dict[str, Any]:
    approved_ref = ArtifactRef.model_validate(state["approved_inputs_ref"])
    approved_inputs = _store(state).read_artifact(approved_ref)
    artifact = _agent(state).derive_from_approved_inputs(
        state["application_path"],
        approved_inputs,
        user_context=state.get("user_context", ""),
        revision_feedback=state.get("characterization_feedback") or "",
    )
    return {
        "working_artifact": artifact,
        "current_stage": CHARACTERIZATION_STAGE,
        "workflow_status": "running",
        "submitted_review": None,
        "route": None,
    }


def _validate_against_approved_inputs(
    artifact: dict[str, Any],
    approved_inputs: dict[str, Any],
    *,
    allow_approved: bool = False,
) -> None:
    CharacterizationAgent._validate_artifact(artifact, allow_approved=allow_approved)
    expected = CandidateInputsDocument.model_validate(approved_inputs)
    if artifact["analysis"]["analysis_id"] != expected.analysis_id:
        raise ValueError("Characterization changed the approved analysis_id")
    actual = CandidateInputsDocument.model_validate(
        {
            "analysis_id": artifact["analysis"]["analysis_id"],
            "application": artifact["application"],
            "entrypoints": artifact["entrypoints"],
            "candidate_inputs": artifact["candidate_inputs"],
            "requested_decisions": expected.requested_decisions,
        }
    )
    if actual.candidate_inputs != expected.candidate_inputs:
        raise ValueError("Characterization changed human-approved candidate inputs")


def write_characterization(state: WorkflowState) -> dict[str, Any]:
    artifact = state["working_artifact"]
    approved_inputs = _store(state).read_artifact(
        ArtifactRef.model_validate(state["approved_inputs_ref"])
    )
    _validate_against_approved_inputs(artifact, approved_inputs)

    version = int(state.get("characterization_revision", 0)) + 1
    reference = _store(state).write_artifact(
        stage=CHARACTERIZATION_STAGE,
        artifact_type="application_characterization",
        version=version,
        payload=artifact,
    )
    return {
        "working_artifact": None,
        "artifact_ref": reference.model_dump(mode="json"),
        "characterization_ref": reference.model_dump(mode="json"),
        "characterization_revision": version,
        "characterization_feedback": None,
    }


def apply_characterization_review(state: WorkflowState) -> dict[str, Any]:
    review = ReviewSubmission.model_validate(state["submitted_review"])
    current = ArtifactRef.model_validate(state["characterization_ref"])
    store = _store(state)
    approved: ArtifactRef | None = None

    if review.decision == "approve":
        approved = current
    elif review.decision == "edit":
        edited = EditedArtifactRef.model_validate(review.edited_artifact)
        edited_path = store.verify_edited_file(edited.path, edited.sha256)
        edited_document = store.read_yaml_file(edited_path)
        approved_inputs = store.read_artifact(
            ArtifactRef.model_validate(state["approved_inputs_ref"])
        )
        _validate_against_approved_inputs(
            edited_document,
            approved_inputs,
            allow_approved=True,
        )
        approved = store.ingest_edit(current, edited_path)

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
            "route": "revise_characterization",
            "workflow_status": "needs_revision",
            "characterization_feedback": review.feedback,
        }
    return {
        **common,
        "artifact_ref": approved.model_dump(mode="json"),
        "characterization_ref": approved.model_dump(mode="json"),
        "approved_characterization_ref": approved.model_dump(mode="json"),
        "characterization_revision": approved.version,
        "route": "complete",
        "current_stage": "complete",
        "workflow_status": "approved",
    }


def route_after_characterization_review(state: WorkflowState) -> str:
    return state["route"]


def build_characterization_graph(checkpointer: SqliteSaver):
    builder = StateGraph(WorkflowState)
    add_planning_nodes(builder)
    add_downstream_nodes(builder)
    builder.add_node("discover_inputs", discover_inputs)
    builder.add_node("write_candidate_inputs", write_candidate_inputs)
    builder.add_node("prepare_input_review", prepare_input_review)
    builder.add_node("input_review_gate", review_gate)
    builder.add_node("apply_input_review", apply_input_review)
    builder.add_node("derive_characterization", derive_characterization)
    builder.add_node("write_characterization", write_characterization)
    builder.add_node("prepare_characterization_review", prepare_characterization_review)
    builder.add_node("characterization_review_gate", review_gate)
    builder.add_node("apply_characterization_review", apply_characterization_review)

    builder.add_edge(START, "discover_inputs")
    builder.add_edge("discover_inputs", "write_candidate_inputs")
    builder.add_edge("write_candidate_inputs", "prepare_input_review")
    builder.add_edge("prepare_input_review", "input_review_gate")
    builder.add_edge("input_review_gate", "apply_input_review")
    builder.add_conditional_edges(
        "apply_input_review",
        route_after_input_review,
        {
            "revise_inputs": "discover_inputs",
            "derive_characterization": "derive_characterization",
        },
    )
    builder.add_edge("derive_characterization", "write_characterization")
    builder.add_edge("write_characterization", "prepare_characterization_review")
    builder.add_edge("prepare_characterization_review", "characterization_review_gate")
    builder.add_edge("characterization_review_gate", "apply_characterization_review")
    builder.add_conditional_edges(
        "apply_characterization_review",
        route_after_characterization_review,
        {
            "revise_characterization": "derive_characterization",
            "start_planning": "planning_entry",
            "complete": END,
        },
    )
    return builder.compile(checkpointer=checkpointer)

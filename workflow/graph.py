"""Persistent LangGraph Milestone 0 with one YAML human-review gate."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from schemas.artifacts import ArtifactRef, EditedArtifactRef, Provenance
from schemas.review import ReviewSubmission
from workflow.artifacts import ArtifactStore, utc_now
from workflow.state import WorkflowState


MILESTONE_STAGE = "milestone0_review"
ARTIFACT_TYPE = "workflow_foundation_demo"


def _store(state: WorkflowState) -> ArtifactStore:
    return ArtifactStore(state["run_dir"])


def generate_draft(state: WorkflowState) -> dict[str, Any]:
    store = _store(state)
    payload = {
        "artifact": {
            "workflow_id": state["workflow_id"],
            "stage": MILESTONE_STAGE,
            "version": 1,
            "status": "draft",
            "application_path": state["application_path"],
            "message": "LangGraph persistence and YAML review foundation draft.",
        }
    }
    artifact = store.write_artifact(
        stage=MILESTONE_STAGE,
        artifact_type=ARTIFACT_TYPE,
        version=1,
        payload=payload,
    )
    return {
        "artifact_ref": artifact.model_dump(mode="json"),
        "current_stage": MILESTONE_STAGE,
        "workflow_status": "running",
        "revision_count": 0,
    }


def prepare_review(state: WorkflowState) -> dict[str, Any]:
    store = _store(state)
    artifact = ArtifactRef.model_validate(state["artifact_ref"])
    provenance = Provenance(
        source_revision=state.get("source_revision"),
        generated_at=utc_now(),
        agent_version="workflow-foundation-0.1",
        prompt_version=None,
        tool_version="artifact-store-0.1",
    )
    review_path = store.write_review_template(
        workflow_id=state["workflow_id"],
        artifact=artifact,
        provenance=provenance,
    )
    pending = {
        "workflow_id": state["workflow_id"],
        "stage": artifact.stage,
        "artifact": artifact.model_dump(mode="json"),
        "review_path": store.relative_path(review_path),
    }
    return {
        "pending_review": pending,
        "submitted_review": None,
        "workflow_status": "awaiting_review",
    }


def review_gate(state: WorkflowState) -> dict[str, Any]:
    submitted = interrupt(state["pending_review"])
    review = ReviewSubmission.model_validate(submitted)
    return {
        "submitted_review": review.model_dump(mode="json"),
        "workflow_status": "review_received",
    }


def apply_review(state: WorkflowState) -> dict[str, Any]:
    store = _store(state)
    review = ReviewSubmission.model_validate(state["submitted_review"])
    current = ArtifactRef.model_validate(state["artifact_ref"])
    decision = review.decision
    history = [review.model_dump(mode="json")]

    if decision == "approve":
        store.write_review_record(
            review=state["submitted_review"],
            approved_artifact=current,
        )
        return {
            "approved_artifact_ref": current.model_dump(mode="json"),
            "pending_review": None,
            "workflow_status": "approved",
            "current_stage": "complete",
            "review_history": history,
        }

    if decision == "edit":
        edited = EditedArtifactRef.model_validate(review.edited_artifact)
        edited_path = store.verify_edited_file(edited.path, edited.sha256)
        approved = store.ingest_edit(current, edited_path)
        store.write_review_record(
            review=state["submitted_review"],
            approved_artifact=approved,
        )
        return {
            "artifact_ref": approved.model_dump(mode="json"),
            "approved_artifact_ref": approved.model_dump(mode="json"),
            "pending_review": None,
            "workflow_status": "approved",
            "current_stage": "complete",
            "review_history": history,
        }

    store.write_review_record(
        review=state["submitted_review"],
        approved_artifact=None,
    )
    return {
        "pending_review": None,
        "workflow_status": "needs_revision",
        "rejection_feedback": review.feedback,
        "review_history": history,
    }


def route_after_review(state: WorkflowState) -> str:
    review = ReviewSubmission.model_validate(state["submitted_review"])
    return "regenerate" if review.decision == "reject" else "complete"


def regenerate_draft(state: WorkflowState) -> dict[str, Any]:
    store = _store(state)
    current = ArtifactRef.model_validate(state["artifact_ref"])
    next_version = current.version + 1
    payload = {
        "artifact": {
            "workflow_id": state["workflow_id"],
            "stage": MILESTONE_STAGE,
            "version": next_version,
            "status": "draft",
            "application_path": state["application_path"],
            "message": "Regenerated after human rejection.",
            "human_feedback": state.get("rejection_feedback"),
            "parent_sha256": current.sha256,
        }
    }
    artifact = store.write_artifact(
        stage=MILESTONE_STAGE,
        artifact_type=ARTIFACT_TYPE,
        version=next_version,
        payload=payload,
    )
    return {
        "artifact_ref": artifact.model_dump(mode="json"),
        "submitted_review": None,
        "rejection_feedback": None,
        "revision_count": int(state.get("revision_count", 0)) + 1,
        "workflow_status": "running",
    }


def build_foundation_graph(checkpointer: SqliteSaver):
    builder = StateGraph(WorkflowState)
    builder.add_node("generate_draft", generate_draft)
    builder.add_node("prepare_review", prepare_review)
    builder.add_node("review_gate", review_gate)
    builder.add_node("apply_review", apply_review)
    builder.add_node("regenerate", regenerate_draft)
    builder.add_edge(START, "generate_draft")
    builder.add_edge("generate_draft", "prepare_review")
    builder.add_edge("prepare_review", "review_gate")
    builder.add_edge("review_gate", "apply_review")
    builder.add_conditional_edges(
        "apply_review",
        route_after_review,
        {"regenerate": "regenerate", "complete": END},
    )
    builder.add_edge("regenerate", "prepare_review")
    return builder.compile(checkpointer=checkpointer)


@contextmanager
def open_graph(
    run_dir: str | Path,
    workflow_type: str = "foundation",
) -> Iterator[Any]:
    database = Path(run_dir) / "checkpoints.sqlite"
    connection = sqlite3.connect(database, check_same_thread=False)
    serializer = JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=[],
        allowed_msgpack_modules=[],
    )
    checkpointer = SqliteSaver(connection, serde=serializer)
    try:
        if workflow_type == "foundation":
            graph = build_foundation_graph(checkpointer)
        elif workflow_type == "characterization":
            from workflow.characterization_graph import build_characterization_graph

            graph = build_characterization_graph(checkpointer)
        else:
            raise ValueError(f"Unsupported workflow type: {workflow_type}")
        yield graph
    finally:
        connection.close()

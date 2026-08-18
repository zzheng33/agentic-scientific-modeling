"""Pydantic contract for YAML human-review submissions."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import ArtifactRef, EditedArtifactRef, Provenance


class ReviewSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    status: Literal["awaiting_review", "completed", "invalid"]
    decision: Literal["approve", "edit", "reject"] | None = None
    artifact: ArtifactRef
    edited_artifact: EditedArtifactRef | None = None
    feedback: str | None = None
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    accepted_at: datetime | None = None
    provenance: Provenance

    @model_validator(mode="after")
    def validate_completed_review(self) -> "ReviewSubmission":
        if self.status == "completed":
            if self.decision is None:
                raise ValueError("A completed review requires a decision")
            if self.decision == "edit" and self.edited_artifact is None:
                raise ValueError("An edit decision requires edited_artifact")
            if self.decision != "edit" and self.edited_artifact is not None:
                raise ValueError("edited_artifact is only valid for an edit decision")
            if self.decision == "reject" and not (self.feedback or "").strip():
                raise ValueError("A reject decision requires feedback")
        return self


class ReviewDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review: ReviewSubmission

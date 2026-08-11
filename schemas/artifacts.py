"""Typed references and provenance for immutable workflow artifacts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    version: int = Field(ge=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EditedArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_revision: str | None = None
    generated_at: datetime
    agent_version: str
    prompt_version: str | None = None
    tool_version: str

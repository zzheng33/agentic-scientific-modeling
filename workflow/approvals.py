"""Review-file loading, stale-review detection, hashing, and archival."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from schemas.artifacts import ArtifactRef
from schemas.review import ReviewDocument
from workflow.artifacts import ArtifactStore


def validate_review_file(
    review_path: str | Path,
    pending_review: dict[str, Any],
    store: ArtifactStore,
) -> dict[str, Any]:
    path = Path(review_path).expanduser().resolve(strict=True)
    expected_artifact = ArtifactRef.model_validate(pending_review["artifact"])
    store.archive_review_submission(path, expected_artifact.stage, expected_artifact.version)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        document = ReviewDocument.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid review file {path}: {exc}") from exc

    review = document.review
    if review.status != "completed":
        raise ValueError("Review status must be completed before resume")
    if review.workflow_id != pending_review["workflow_id"]:
        raise ValueError("Review workflow_id does not match the paused workflow")
    if review.stage != pending_review["stage"]:
        raise ValueError("Review stage does not match the paused stage")
    if review.artifact != expected_artifact:
        raise ValueError("Stale review: artifact path, version, or hash does not match")
    store.verify_artifact(review.artifact)

    if review.edited_artifact is not None:
        store.verify_edited_file(review.edited_artifact.path, review.edited_artifact.sha256)

    normalized = review.model_dump(mode="json")
    normalized["accepted_at"] = datetime.now(timezone.utc).isoformat()
    return normalized

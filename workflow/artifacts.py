"""Immutable YAML artifact storage with byte-level SHA-256 references."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from schemas.artifacts import ArtifactRef, Provenance
from schemas.review import ReviewDocument


_WORKFLOW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ARTIFACT_VERSION = re.compile(r"\.v(\d+)\.")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()

    @staticmethod
    def validate_workflow_id(workflow_id: str) -> str:
        if not _WORKFLOW_ID.fullmatch(workflow_id):
            raise ValueError(
                "workflow_id must use 1-64 letters, numbers, dots, underscores, or hyphens"
            )
        return workflow_id

    def initialize(
        self,
        workflow_id: str,
        application_path: str | Path,
        *,
        workflow_type: str = "foundation",
        config_path: str | Path | None = None,
        user_context: str = "",
    ) -> None:
        self.validate_workflow_id(workflow_id)
        if self.run_dir.exists():
            raise ValueError(f"Workflow already exists: {self.run_dir}")
        application = Path(application_path).expanduser().resolve(strict=True)
        if not application.is_dir():
            raise ValueError(f"Application path is not a directory: {application}")
        for relative in (
            "artifacts",
            "reviews/pending",
            "reviews/archive",
            "reviews/accepted",
            "edits",
            "logs",
        ):
            (self.run_dir / relative).mkdir(parents=True, exist_ok=False)
        self._write_new(
            self.run_dir / "workflow.yaml",
            self._yaml_bytes(
                {
                    "workflow_id": workflow_id,
                    "workflow_type": workflow_type,
                    "application_path": str(application),
                    "config_path": (
                        str(Path(config_path).expanduser().resolve())
                        if config_path is not None
                        else None
                    ),
                    "user_context": user_context,
                    "created_at": utc_now().isoformat(),
                    "status": "created",
                }
            ),
        )

    def load_metadata(self) -> dict[str, Any]:
        path = self._resolve("workflow.yaml")
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Workflow metadata does not exist: {path}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"Workflow metadata must be a YAML mapping: {path}")
        return document

    def write_artifact(
        self,
        *,
        stage: str,
        artifact_type: str,
        version: int,
        payload: dict[str, Any],
    ) -> ArtifactRef:
        relative = Path("artifacts") / stage / f"{artifact_type}.v{version:03d}.yaml"
        path = self._resolve(relative)
        content = self._yaml_bytes(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Immutable artifact already exists with different content: {path}")
        else:
            self._write_new(path, content)
        return ArtifactRef(
            artifact_type=artifact_type,
            stage=stage,
            version=version,
            path=relative.as_posix(),
            sha256=sha256_file(path),
        )

    def next_artifact_version(self, stage: str, *, minimum: int = 1) -> int:
        """Return a version above any complete or partial artifact in a stage."""
        if minimum < 1:
            raise ValueError("Artifact version minimum must be positive")
        directory = self._resolve(Path("artifacts") / stage)
        highest = 0
        if directory.is_dir():
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                match = _ARTIFACT_VERSION.search(path.name)
                if match:
                    highest = max(highest, int(match.group(1)))
        return max(minimum, highest + 1)

    def write_text_artifact(
        self,
        *,
        stage: str,
        artifact_type: str,
        version: int,
        extension: str,
        content: str,
    ) -> ArtifactRef:
        if not re.fullmatch(r"[A-Za-z0-9]+", extension):
            raise ValueError(f"Invalid artifact extension: {extension}")
        relative = Path("artifacts") / stage / (
            f"{artifact_type}.v{version:03d}.{extension.lower()}"
        )
        path = self._resolve(relative)
        encoded = content.encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != encoded:
                raise ValueError(f"Immutable artifact already exists with different content: {path}")
        else:
            self._write_new(path, encoded)
        return ArtifactRef(
            artifact_type=artifact_type,
            stage=stage,
            version=version,
            path=relative.as_posix(),
            sha256=sha256_file(path),
        )

    def write_review_template(
        self,
        *,
        workflow_id: str,
        artifact: ArtifactRef,
        provenance: Provenance,
    ) -> Path:
        relative = (
            Path("reviews")
            / "pending"
            / f"{artifact.stage}.v{artifact.version:03d}.review.yaml"
        )
        path = self._resolve(relative)
        if path.exists():
            existing = ReviewDocument.model_validate(
                yaml.safe_load(path.read_text(encoding="utf-8"))
            ).review
            if existing.workflow_id != workflow_id or existing.artifact != artifact:
                raise ValueError(f"Existing review template targets another artifact: {path}")
            return path
        template = ReviewDocument.model_validate(
            {
                "review": {
                    "workflow_id": workflow_id,
                    "stage": artifact.stage,
                    "status": "awaiting_review",
                    "decision": None,
                    "artifact": artifact.model_dump(mode="json"),
                    "edited_artifact": None,
                    "feedback": None,
                    "reviewer": None,
                    "reviewed_at": None,
                    "provenance": provenance.model_dump(mode="json"),
                }
            }
        )
        content = self._yaml_bytes(template.model_dump(mode="json"))
        self._write_new(path, content)
        return path

    def verify_artifact(self, artifact: ArtifactRef) -> Path:
        path = self._resolve(artifact.path)
        if not path.is_file():
            raise ValueError(f"Artifact file does not exist: {path}")
        actual = sha256_file(path)
        if actual != artifact.sha256:
            raise ValueError(f"Artifact hash mismatch for {artifact.path}")
        return path

    def read_artifact(self, artifact: ArtifactRef) -> dict[str, Any]:
        path = self.verify_artifact(artifact)
        return self.read_yaml_file(path)

    def read_yaml_file(self, path: str | Path) -> dict[str, Any]:
        resolved = self._resolve(path)
        document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"YAML file must contain one mapping: {path}")
        return document

    def verify_edited_file(self, path: str, expected_hash: str) -> Path:
        edited = self._resolve(path)
        if not edited.is_file():
            raise ValueError(f"Edited artifact does not exist: {edited}")
        if sha256_file(edited) != expected_hash:
            raise ValueError(f"Edited artifact hash mismatch: {path}")
        document = yaml.safe_load(edited.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Edited artifact must contain one YAML mapping")
        return edited

    def ingest_edit(self, current: ArtifactRef, edited_path: Path) -> ArtifactRef:
        document = yaml.safe_load(edited_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Edited artifact must contain one YAML mapping")
        return self.write_artifact(
            stage=current.stage,
            artifact_type=current.artifact_type,
            version=current.version + 1,
            payload=document,
        )

    def archive_review_submission(self, review_path: str | Path, stage: str, version: int) -> Path:
        source = Path(review_path).expanduser().resolve(strict=True)
        archive_dir = self._resolve(Path("reviews") / "archive")
        attempt = 1
        while True:
            destination = archive_dir / (
                f"{stage}.v{version:03d}.submission{attempt:03d}.yaml"
            )
            if not destination.exists():
                break
            attempt += 1
        shutil.copyfile(source, destination)
        return destination

    def write_review_record(
        self,
        *,
        review: dict[str, Any],
        approved_artifact: ArtifactRef | None,
    ) -> Path:
        stage = str(review["stage"])
        version = int(review["artifact"]["version"])
        decision = str(review["decision"])
        relative = (
            Path("reviews")
            / "accepted"
            / f"{stage}.v{version:03d}.{decision}.yaml"
        )
        path = self._resolve(relative)
        content = self._yaml_bytes(
            {
                "review": review,
                "approved_artifact": (
                    approved_artifact.model_dump(mode="json") if approved_artifact else None
                ),
            }
        )
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Review record already exists with different content: {path}")
        else:
            self._write_new(path, content)
        pending = self._resolve(
            Path("reviews") / "pending" / f"{stage}.v{version:03d}.review.yaml"
        )
        pending.unlink(missing_ok=True)
        return path

    def relative_path(self, path: str | Path) -> str:
        return self._resolve(path).relative_to(self.run_dir).as_posix()

    def _resolve(self, path: str | Path) -> Path:
        requested = Path(path)
        candidate = requested.resolve() if requested.is_absolute() else (self.run_dir / requested).resolve()
        if not candidate.is_relative_to(self.run_dir):
            raise ValueError(f"Path escapes workflow run directory: {path}")
        return candidate

    @staticmethod
    def _yaml_bytes(document: dict[str, Any]) -> bytes:
        return yaml.safe_dump(
            document,
            sort_keys=False,
            allow_unicode=True,
        ).encode("utf-8")

    @staticmethod
    def _write_new(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                raise ValueError(f"Refusing to overwrite existing file: {path}")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from schemas.review import ReviewSubmission


def _review(**overrides):
    document = {
        "workflow_id": "workflow-001",
        "stage": "candidate_inputs_review",
        "status": "completed",
        "decision": "approve",
        "artifact": {
            "artifact_type": "candidate_inputs",
            "stage": "candidate_inputs_review",
            "version": 1,
            "path": "artifacts/candidate_inputs_review/candidate_inputs.v001.yaml",
            "sha256": "a" * 64,
        },
        "edited_artifact": None,
        "feedback": None,
        "reviewer": None,
        "reviewed_at": None,
        "accepted_at": None,
        "provenance": {
            "source_revision": None,
            "generated_at": datetime.now(timezone.utc),
            "agent_version": "test",
            "prompt_version": None,
            "tool_version": "test",
        },
    }
    document.update(overrides)
    return document


class ReviewSubmissionTests(unittest.TestCase):
    def test_completed_approval_does_not_require_reviewer(self) -> None:
        review = ReviewSubmission.model_validate(_review())

        self.assertIsNone(review.reviewer)
        self.assertEqual(review.decision, "approve")

    def test_reject_still_requires_feedback(self) -> None:
        with self.assertRaisesRegex(ValueError, "reject decision requires feedback"):
            ReviewSubmission.model_validate(_review(decision="reject"))


if __name__ == "__main__":
    unittest.main()

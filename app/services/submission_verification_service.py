from __future__ import annotations

from typing import Any, Callable

from app.schemas.submission import AggregatedSubmission
from app.schemas.submission_verification import (
    SubmissionVerificationIssue,
    SubmissionVerificationResult,
)


class SubmissionVerificationService:
    """
    Verifies an aggregated student submission using an injectable verifier.

    The verifier is intentionally injected so that:
    - unit tests can use deterministic fake verifiers
    - the actual second model can be plugged in later
    - the service itself remains independent of any specific model provider
    """

    def __init__(
        self,
        verify_fn: Callable[
            [AggregatedSubmission, int],
            SubmissionVerificationResult,
        ],
    ) -> None:
        if not callable(verify_fn):
            raise TypeError("verify_fn must be callable")

        self.verify_fn = verify_fn

    def verify(
        self,
        submission: AggregatedSubmission,
        *,
        attempt: int,
    ) -> SubmissionVerificationResult:
        if not isinstance(submission, AggregatedSubmission):
            raise TypeError(
                "submission must be an AggregatedSubmission"
            )

        if attempt < 1:
            raise ValueError("attempt must be >= 1")

        result = self.verify_fn(submission, attempt)

        if not isinstance(result, SubmissionVerificationResult):
            raise TypeError(
                "verify_fn must return SubmissionVerificationResult"
            )

        result.attempt = attempt

        return result

    @staticmethod
    def build_feedback(
        result: SubmissionVerificationResult,
    ) -> str:
        """
        Converts verifier issues into deterministic retry feedback.

        This feedback is intended for the extraction/re-processing layer,
        not directly for teacher communication.
        """
        if result.feedback and result.feedback.strip():
            return result.feedback.strip()

        if not result.issues:
            return "Verification failed. Re-check the extracted submission."

        lines: list[str] = []

        for issue in result.issues:
            question = (
                f"Q{issue.question_no}"
                if issue.question_no is not None
                else "Submission"
            )

            lines.append(
                f"{question}: {issue.issue_type} "
                f"({issue.severity}) - {issue.message}"
            )

        return "\n".join(lines)

    @staticmethod
    def normalize_result(
        result: SubmissionVerificationResult,
        *,
        attempt: int,
    ) -> SubmissionVerificationResult:
        """
        Normalize the verifier result and prevent contradictory PASS output.

        A high-severity issue always forces passed=False.
        """
        result.attempt = attempt

        high_severity_issues = [
            issue
            for issue in result.issues
            if issue.severity == "high"
        ]

        # A high-severity issue can never coexist with PASS.
        result.passed = (
            False
            if high_severity_issues
            else result.passed
        )

        if not result.passed:
            result.feedback = (
                SubmissionVerificationService.build_feedback(result)
            )

        return result

    @staticmethod
    def create_issue(
        *,
        issue_type: str,
        message: str,
        severity: str = "medium",
        question_no: int | None = None,
    ) -> SubmissionVerificationIssue:
        return SubmissionVerificationIssue(
            question_no=question_no,
            issue_type=issue_type,
            severity=severity,
            message=message,
        )

    @staticmethod
    def submission_summary(
        submission: AggregatedSubmission,
    ) -> dict[str, Any]:
        """
        Returns a compact deterministic summary useful for logs/debugging.
        """
        return {
            "student_number": submission.student_number,
            "job_id": submission.job_id,
            "total_pages": submission.total_pages,
            "processed_pages": submission.processed_pages,
            "answer_count": len(submission.answers),
            "missing_questions": submission.missing_questions,
            "repeated_questions": submission.repeated_questions,
            "conflicting_questions": submission.conflicting_questions,
            "unresolved_items": submission.unresolved_items,
            "review_required": submission.review_required,
        }

from __future__ import annotations

from collections.abc import Callable, Sequence

from app.schemas.extraction import PageExtractionResult
from app.schemas.submission import AggregatedSubmission
from app.schemas.submission_verification import (
    SubmissionVerificationLoopResult,
    SubmissionVerificationResult,
)
from app.services.submission_aggregation_service import (
    SubmissionAggregationService,
)
from app.services.submission_verification_service import (
    SubmissionVerificationService,
)


class SubmissionVerificationWorkflow:
    """
    End-to-end combined submission verification loop.

    Flow:
        page extraction
            ↓
        submission aggregation
            ↓
        combined verification
            ↓
        PASS -> return
        FAIL -> feedback -> re-extract -> aggregate -> verify again

    Maximum number of attempts is configurable.
    """

    def __init__(
        self,
        *,
        aggregation_service: SubmissionAggregationService,
        verification_service: SubmissionVerificationService,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        self.aggregation_service = aggregation_service
        self.verification_service = verification_service
        self.max_attempts = max_attempts

    def run(
        self,
        *,
        extract_fn: Callable[
            [int, str | None],
            Sequence[PageExtractionResult],
        ],
        job_id: str | None = None,
        student_number: str | None = None,
        expected_question_numbers: list[int] | None = None,
        metadata: dict | None = None,
    ) -> SubmissionVerificationLoopResult:
        feedback: str | None = None
        warnings: list[str] = []
        last_submission: AggregatedSubmission | None = None
        last_verification: SubmissionVerificationResult | None = None

        for attempt in range(1, self.max_attempts + 1):
            page_results = list(
                extract_fn(
                    attempt,
                    feedback,
                )
            )

            submission = self.aggregation_service.aggregate(
                page_results,
                job_id=job_id,
                student_number=student_number,
                expected_question_numbers=expected_question_numbers,
                metadata=metadata,
            )

            verification = self.verification_service.verify(
                submission,
                attempt=attempt,
            )

            verification = self.verification_service.normalize_result(
                verification,
                attempt=attempt,
            )

            last_submission = submission
            last_verification = verification

            if verification.passed:
                return SubmissionVerificationLoopResult(
                    passed=True,
                    attempts=attempt,
                    submission=submission.model_dump(),
                    verification=verification,
                    review_required=False,
                    warnings=warnings,
                )

            feedback = self.verification_service.build_feedback(
                verification
            )

            warnings.append(
                f"Combined verification failed on attempt {attempt}."
            )

            if attempt < self.max_attempts:
                continue

            warnings.append(
                f"Maximum verification attempts ({self.max_attempts}) "
                "exhausted."
            )

        if last_submission is None or last_verification is None:
            raise RuntimeError(
                "Verification workflow completed without a result."
            )

        return SubmissionVerificationLoopResult(
            passed=False,
            attempts=self.max_attempts,
            submission=last_submission.model_dump(),
            verification=last_verification,
            review_required=True,
            warnings=warnings,
        )

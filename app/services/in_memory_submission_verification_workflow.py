from __future__ import annotations

from typing import Callable, Sequence

from PIL import Image

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


class InMemorySubmissionVerificationWorkflow:
    """
    Coordinates the complete in-memory extraction -> aggregation ->
    verification loop.

    Page images stay in RAM during one complete verification attempt.

    Flow:

        PDF page
          ↓
        RAM image
          ↓
        page extraction
          ↓
        all pages aggregated
          ↓
        original RAM pages + aggregation
          ↓
        verification

        FAIL:
          ↓
        verifier feedback
          ↓
        re-extract all pages
          ↓
        re-aggregate
          ↓
        re-verify

    Maximum attempts are configurable.
    """

    def __init__(
        self,
        *,
        page_count_fn: Callable[[str], int],
        render_page_fn: Callable[[str, int], Image.Image],
        extract_page_fn: Callable[
            [Image.Image, int, list[int], bool, str | None],
            PageExtractionResult,
        ],
        aggregation_service: SubmissionAggregationService,
        verifier_model,
        max_attempts: int = 3,
    ) -> None:

        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        self.page_count_fn = page_count_fn
        self.render_page_fn = render_page_fn
        self.extract_page_fn = extract_page_fn

        self.aggregation_service = aggregation_service
        self.verifier_model = verifier_model

        self.max_attempts = max_attempts

    def _extract_attempt(
        self,
        pdf_path: str,
        expected_question_numbers: list[int],
        *,
        enable_recovery: bool,
        feedback: str | None,
    ) -> tuple[
        list[PageExtractionResult],
        list[tuple[int, Image.Image]],
    ]:

        total_pages = self.page_count_fn(pdf_path)

        page_results: list[PageExtractionResult] = []
        page_images: list[tuple[int, Image.Image]] = []

        try:

            for page_no in range(1, total_pages + 1):

                image = self.render_page_fn(
                    pdf_path,
                    page_no,
                )

                page_images.append(
                    (page_no, image)
                )

                result = self.extract_page_fn(
                    image,
                    page_no,
                    expected_question_numbers,
                    enable_recovery,
                    feedback,
                )

                page_results.append(result)

        except Exception:

            for _, image in page_images:
                try:
                    image.close()
                except Exception:
                    pass

            raise

        return page_results, page_images

    def run(
        self,
        *,
        pdf_path: str,
        expected_question_numbers: list[int] | None = None,
        job_id: str | None = None,
        student_number: str | None = None,
        enable_recovery: bool = True,
        metadata: dict | None = None,
    ) -> SubmissionVerificationLoopResult:

        expected = sorted(
            set(expected_question_numbers or [])
        )

        feedback: str | None = None
        warnings: list[str] = []

        last_submission: AggregatedSubmission | None = None
        last_verification: SubmissionVerificationResult | None = None

        for attempt in range(1, self.max_attempts + 1):

            page_results, page_images = self._extract_attempt(
                pdf_path,
                expected,
                enable_recovery=enable_recovery,
                feedback=feedback,
            )

            try:

                submission = self.aggregation_service.aggregate(
                    page_results,
                    job_id=job_id,
                    student_number=student_number,
                    expected_question_numbers=expected,
                    metadata=metadata,
                )

                verification_service = (
                    SubmissionVerificationService(
                        lambda current_submission, current_attempt:
                            self.verifier_model.verify(
                                current_submission,
                                attempt=current_attempt,
                                page_images=page_images,
                                temperature=0,
                            )
                    )
                )

                verification = verification_service.verify(
                    submission,
                    attempt=attempt,
                )

                verification = (
                    verification_service.normalize_result(
                        verification,
                        attempt=attempt,
                    )
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

                feedback = (
                    verification_service.build_feedback(
                        verification
                    )
                )

                warnings.append(
                    "Combined verification failed on "
                    f"attempt {attempt}."
                )

            finally:

                for _, image in page_images:
                    try:
                        image.close()
                    except Exception:
                        pass

        if (
            last_submission is None
            or last_verification is None
        ):
            raise RuntimeError(
                "Verification workflow completed without a result."
            )

        warnings.append(
            f"Maximum verification attempts "
            f"({self.max_attempts}) exhausted."
        )

        return SubmissionVerificationLoopResult(
            passed=False,
            attempts=self.max_attempts,
            submission=last_submission.model_dump(),
            verification=last_verification,
            review_required=True,
            warnings=warnings,
        )

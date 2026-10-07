from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

from app.core.config import settings
from app.schemas.evaluation_result import EvaluationResult
from app.schemas.marking_scheme import MarkingScheme
from app.schemas.submission import AggregatedSubmission
from app.services.checked_copy_renderer import CheckedCopyRenderer
from app.services.grading_model import GradingModel
from app.services.submission_aggregation_service import SubmissionAggregationService


class InMemoryGradingWorkflow:
    """
    Pipeline:

        PDF
        ↓
        Whole-page extraction
        ↓
        Combine / aggregate
        ↓
        Locked marking scheme
        ↓
        Grading model
        ↓
        Question-wise marks
        ↓
        Checked-copy PDF

    Current temporary scope:
        - first 2 pages
        - one extraction call per page
        - recovery disabled
        - grading active
        - grading-review model disabled

    Later, max_pages can be changed to None for full-copy processing.
    """

    def __init__(
        self,
        *,
        page_count_fn: Callable[[str], int],
        render_page_fn: Callable[[str, int], Image.Image],
        extract_page_fn: Callable[..., object],
        aggregation_service: SubmissionAggregationService,
        grading_model: GradingModel,
        checked_copy_renderer: CheckedCopyRenderer | None = None,
        max_pages: int | None = 2,
    ) -> None:
        self.page_count_fn = page_count_fn
        self.render_page_fn = render_page_fn
        self.extract_page_fn = extract_page_fn
        self.aggregation_service = aggregation_service
        self.grading_model = grading_model
        self.checked_copy_renderer = (
            checked_copy_renderer or CheckedCopyRenderer()
        )
        self.max_pages = max_pages

    def run(
        self,
        *,
        pdf_path: str,
        marking_scheme: MarkingScheme,
        job_id: str,
        student_number: str,
        assessment_id: str | None = None,
        expected_question_numbers: list[int] | None = None,
        enable_recovery: bool = False,
        output_dir: str | Path | None = None,
        metadata: dict | None = None,
    ) -> tuple[EvaluationResult, AggregatedSubmission, str]:

        actual_page_count = self.page_count_fn(pdf_path)

        if actual_page_count <= 0:
            raise ValueError("PDF contains no pages.")

        if self.max_pages is None:
            pages_to_process = actual_page_count
        else:
            pages_to_process = min(
                actual_page_count,
                self.max_pages,
            )

        page_results = []
        page_images: list[tuple[int, Image.Image]] = []

        try:
            # ====================================================
            # 1. WHOLE-PAGE EXTRACTION
            # ====================================================

            for page_no in range(1, pages_to_process + 1):

                print("\n" + "=" * 100)
                print(f"GRADING WORKFLOW - EXTRACTION PAGE {page_no}")
                print("=" * 100)

                image = self.render_page_fn(
                    pdf_path,
                    page_no,
                )

                page_images.append(
                    (page_no, image)
                )

                page_result = self.extract_page_fn(
                    image=image,
                    page_no=page_no,
                    expected_question_numbers=(
                        expected_question_numbers or []
                    ),
                    enable_recovery=enable_recovery,
                    feedback=None,
                )

                page_results.append(page_result)

            # ====================================================
            # 2. COMBINE / AGGREGATE
            # ====================================================

            submission = self.aggregation_service.aggregate(
                page_results,
                job_id=job_id,
                student_number=student_number,
                expected_question_numbers=(
                    expected_question_numbers or []
                ),
                metadata={
                    **(metadata or {}),
                    "workflow": "in_memory_grading",
                    "pages_in_current_test": pages_to_process,
                },
            )

            print("\n" + "=" * 100)
            print("COMBINED SUBMISSION")
            print("=" * 100)
            print(
                f"Pages processed: "
                f"{submission.processed_pages}"
            )
            print(
                f"Questions found: "
                f"{len(submission.answers)}"
            )
            print(
                f"Review required: "
                f"{submission.review_required}"
            )

            if submission.review_required:
                raise ValueError(
                    "Combined extraction requires review before grading. "
                    f"Warnings: {submission.warnings}"
                )

            # ====================================================
            # 3. GRADING
            # ====================================================

            print("\n" + "=" * 100)
            print("GRADING")
            print("=" * 100)

            evaluation_result = self.grading_model.grade_submission(
                submission=submission,
                marking_scheme=marking_scheme,
                job_id=job_id,
                student_number=student_number,
                assessment_id=assessment_id,
            )

            print(
                f"Obtained marks: "
                f"{evaluation_result.obtained_marks}"
                f" / "
                f"{evaluation_result.total_marks}"
            )

            print(
                f"Percentage: "
                f"{evaluation_result.percentage:.2f}%"
            )

            # ====================================================
            # 4. CHECKED COPY
            # ====================================================

            resolved_output_dir = Path(
                output_dir
                or settings.output_dir
            )

            checked_copy_path = (
                self.checked_copy_renderer.render(
                    page_images=page_images,
                    evaluation_result=evaluation_result,
                    output_dir=resolved_output_dir,
                    file_name=f"{job_id}_checked.pdf",
                )
            )

            print("\n" + "=" * 100)
            print("CHECKED COPY CREATED")
            print("=" * 100)
            print(f"Output: {checked_copy_path}")

            return (
                evaluation_result,
                submission,
                str(checked_copy_path),
            )

        finally:
            for _, image in page_images:
                try:
                    image.close()
                except Exception:
                    pass


# ====================================================================
# FUTURE GRADING REVIEW MODEL
# ====================================================================
#
# DISABLED FOR NOW.
#
# Later:
#
#     Grading Model
#          ↓
#     Grading Review Model
#          ↓
#      ┌───────────┐
#      │           │
#     PASS        FAIL
#      │           │
#      ↓           ↓
# Teacher Review  Re-grade affected questions
#
# Example future implementation:
#
# class GradingReviewModel:
#
#     def review(
#         self,
#         submission: AggregatedSubmission,
#         marking_scheme: MarkingScheme,
#         evaluation_result: EvaluationResult,
#     ) -> dict:
#         ...
#
# This reviewer must remain independent from the grading model.
# ====================================================================

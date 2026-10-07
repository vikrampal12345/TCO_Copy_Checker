from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL import Image

from app.schemas.extraction import PageExtractionResult
from app.schemas.extraction_verification import (
    ExtractionVerificationIssue,
    ExtractionVerificationResult,
)


class ExtractionVerificationService:
    """
    Independent verification layer for handwritten-answer extraction.

    Responsibilities:
    - compare extracted content against the original page
    - identify extraction errors
    - return PASS / FAIL
    - generate structured feedback

    Important:
    This service does NOT directly modify the extraction result.

    The actual vision model is injected through `verify_fn`.
    This keeps the verification layer independent of any specific
    provider/model and makes testing easy.
    """

    def __init__(
        self,
        verify_fn: Callable[
            [Image.Image, PageExtractionResult, int],
            ExtractionVerificationResult,
        ],
    ) -> None:
        self.verify_fn = verify_fn

    def verify(
        self,
        image: Image.Image,
        extraction: PageExtractionResult,
        attempt: int = 1,
    ) -> ExtractionVerificationResult:
        """
        Verify one page extraction against the original page image.
        """

        if image is None:
            raise ValueError(
                "Page image is required for extraction verification."
            )

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image instance."
            )

        if not isinstance(
            extraction,
            PageExtractionResult,
        ):
            raise TypeError(
                "extraction must be a PageExtractionResult."
            )

        if attempt < 1:
            raise ValueError(
                "Verification attempt must be >= 1."
            )

        result = self.verify_fn(
            image,
            extraction,
            attempt,
        )

        if not isinstance(
            result,
            ExtractionVerificationResult,
        ):
            raise TypeError(
                "verify_fn must return "
                "ExtractionVerificationResult."
            )

        # Ensure the actual attempt number always represents
        # the current workflow attempt.
        result.attempt = attempt

        return result

    @staticmethod
    def build_feedback(
        result: ExtractionVerificationResult,
    ) -> str:
        """
        Convert structured verifier issues into feedback for
        the next extraction attempt.

        This is intentionally deterministic.
        """

        if result.passed:
            return ""

        feedback_parts: list[str] = []

        if result.feedback:
            feedback_parts.append(
                result.feedback.strip()
            )

        for issue in result.issues:
            question_prefix = ""

            if issue.question_no is not None:
                question_prefix = (
                    f"Q{issue.question_no}: "
                )

            feedback_parts.append(
                f"{question_prefix}"
                f"{issue.message}"
            )

        if not feedback_parts:
            feedback_parts.append(
                "Extraction did not pass verification. "
                "Re-check the original page carefully and "
                "correct question-number and answer associations."
            )

        return "\n".join(
            feedback_parts
        )

    @staticmethod
    def normalize_verifier_result(
        result: ExtractionVerificationResult,
    ) -> ExtractionVerificationResult:
        """
        Ensure verifier output is safe and consistent.
        """

        feedback = (
            result.feedback.strip()
            if result.feedback
            else None
        )

        warnings = [
            warning.strip()
            for warning in result.warnings
            if warning and warning.strip()
        ]

        return ExtractionVerificationResult(
            page_no=result.page_no,
            passed=result.passed,
            confidence=result.confidence,
            issues=result.issues,
            feedback=feedback,
            warnings=warnings,
            attempt=result.attempt,
        )
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL import Image

from app.schemas.extraction import PageExtractionResult
from app.schemas.extraction_verification import (
    ExtractionVerificationLoopResult,
    ExtractionVerificationResult,
)
from app.services.extraction_verification_service import (
    ExtractionVerificationService,
)


class ExtractionVerificationWorkflow:
    """
    Controls the complete extraction verification loop.

    Workflow:

        Extraction
             ↓
        Verification
             ↓
          PASS?
          /   \
        YES    NO
         ↓      ↓
      Continue Feedback
                ↓
          Re-extraction
                ↓
          Verification

    The workflow stops when:
    - verification passes, OR
    - maximum attempts are exhausted.

    If attempts are exhausted without passing,
    the result is marked as review_required.
    """

    def __init__(
        self,
        verification_service: ExtractionVerificationService,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be >= 1."
            )

        self.verification_service = (
            verification_service
        )

        self.max_attempts = max_attempts

    def run(
        self,
        image: Image.Image,
        page_no: int,
        extract_fn: Callable[
            [
                Image.Image,
                int,
                str | None,
            ],
            PageExtractionResult,
        ],
    ) -> ExtractionVerificationLoopResult:
        """
        Execute extraction + verification + feedback retry loop.

        Parameters
        ----------
        image:
            Original page image held in memory.

        page_no:
            Current PDF page number.

        extract_fn:
            Extraction function.

            Signature:

                extract_fn(
                    image,
                    page_no,
                    feedback
                )

            On first attempt feedback is None.

            On subsequent attempts feedback contains the
            verifier's previous feedback.
        """

        if image is None:
            raise ValueError(
                "Page image is required."
            )

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "image must be a PIL.Image.Image instance."
            )

        if page_no < 1:
            raise ValueError(
                "page_no must be >= 1."
            )

        if not callable(extract_fn):
            raise TypeError(
                "extract_fn must be callable."
            )

        current_extraction: PageExtractionResult | None = None

        current_verification: (
            ExtractionVerificationResult | None
        ) = None

        feedback: str | None = None

        warnings: list[str] = []

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):
            # -------------------------------------------------
            # EXTRACTION
            # -------------------------------------------------

            current_extraction = extract_fn(
                image,
                page_no,
                feedback,
            )

            if not isinstance(
                current_extraction,
                PageExtractionResult,
            ):
                raise TypeError(
                    "extract_fn must return "
                    "PageExtractionResult."
                )

            # -------------------------------------------------
            # VERIFICATION
            # -------------------------------------------------

            current_verification = (
                self.verification_service.verify(
                    image=image,
                    extraction=current_extraction,
                    attempt=attempt,
                )
            )

            current_verification = (
                self.verification_service
                .normalize_verifier_result(
                    current_verification
                )
            )

            # -------------------------------------------------
            # PASS
            # -------------------------------------------------

            if current_verification.passed:
                return ExtractionVerificationLoopResult(
                    page_no=page_no,
                    passed=True,
                    attempts=attempt,
                    extraction=current_extraction.model_dump(),
                    verification=current_verification,
                    review_required=False,
                    warnings=warnings
                    + current_verification.warnings,
                )

            # -------------------------------------------------
            # FAIL
            # -------------------------------------------------

            feedback = (
                self.verification_service.build_feedback(
                    current_verification
                )
            )

            if feedback:
                warnings.append(
                    f"Extraction verification failed "
                    f"on attempt {attempt}: "
                    f"{feedback}"
                )

            # If this was the last allowed attempt,
            # do not perform another extraction.
            if attempt == self.max_attempts:
                break

        # -----------------------------------------------------
        # MAX ATTEMPTS EXHAUSTED
        # -----------------------------------------------------

        assert current_extraction is not None
        assert current_verification is not None

        warnings.append(
            "Maximum extraction verification attempts "
            "were exhausted."
        )

        return ExtractionVerificationLoopResult(
            page_no=page_no,
            passed=False,
            attempts=self.max_attempts,
            extraction=current_extraction.model_dump(),
            verification=current_verification,
            review_required=True,
            warnings=warnings
            + current_verification.warnings,
        )
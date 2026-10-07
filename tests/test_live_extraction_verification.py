from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from app.schemas.extraction import PageExtractionResult
from app.services.answer_extraction_service import (
    AnswerExtractionService,
)
from app.services.extraction_verifier_model import (
    ExtractionVerifierModel,
)
from app.services.extraction_verification_service import (
    ExtractionVerificationService,
)


PDF_PATH = Path(
    "data/test_copies/Student_23.pdf"
)

PAGE_OUTPUT_DIR = Path(
    "data/test_verification/pages"
)

TILE_OUTPUT_DIR = Path(
    "data/test_verification/tiles"
)

# For this workflow test we allow a maximum of 2 passes:
#
# Attempt 1:
#   extraction -> verifier
#
# Attempt 2:
#   extraction/recovery -> verifier again
#
MAX_ATTEMPTS = 2


def render_page_to_disk_for_workflow_test(
    pdf_path: Path,
    page_no: int,
    output_path: Path,
) -> Path:
    """
    Render one PDF page for this integration test.

    IMPORTANT:
    This is ONLY a temporary bridge because the existing
    AnswerExtractionService expects an image_path.

    The final architecture can later move this stage fully
    to in-memory PIL images.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = pymupdf.open(
        pdf_path
    )

    try:
        page = document.load_page(
            page_no - 1
        )

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(
                200 / 72.0,
                200 / 72.0,
            ),
            alpha=False,
        )

        pixmap.save(
            str(output_path)
        )

    finally:
        document.close()

    return output_path


def extraction_with_feedback(
    extractor: AnswerExtractionService,
    image_path: Path,
    page_no: int,
    tile_dir: Path,
    feedback: str | None,
) -> PageExtractionResult:
    """
    Call the existing extraction service.

    The current extractor already performs its own tile-based
    recovery internally.

    The feedback is displayed so we can verify that the
    orchestration layer is passing verifier feedback into the
    retry stage.

    NOTE:
    The current AnswerExtractionService API does not yet expose
    a feedback parameter, so the feedback is currently logged
    and the existing recovery mechanism is reused.
    """

    print()
    print("=" * 80)
    print(
        f"EXTRACTION ATTEMPT FOR PAGE {page_no}"
    )
    print("=" * 80)

    if feedback:
        print("VERIFIER FEEDBACK:")
        print(feedback)
    else:
        print(
            "No verifier feedback. "
            "This is the first extraction attempt."
        )

    result = extractor.extract_page(
        image_path=image_path,
        page_no=page_no,
        tile_output_dir=tile_dir,
        expected_question_numbers=[],
        enable_recovery=True,
    )

    return result


def run_page_verification(
    page_no: int,
) -> None:
    print()
    print("#" * 90)
    print(
        f"TCO LIVE EXTRACTION VERIFICATION - PAGE {page_no}"
    )
    print("#" * 90)

    # ---------------------------------------------------------
    # Prepare page image
    # ---------------------------------------------------------

    page_image = (
        PAGE_OUTPUT_DIR
        / f"page_{page_no:03d}.png"
    )

    render_page_to_disk_for_workflow_test(
        pdf_path=PDF_PATH,
        page_no=page_no,
        output_path=page_image,
    )

    print(
        f"Page image: {page_image}"
    )

    # ---------------------------------------------------------
    # Services
    # ---------------------------------------------------------

    extractor = (
        AnswerExtractionService()
    )

    verifier_model = (
        ExtractionVerifierModel()
    )

    verifier_service = (
        ExtractionVerificationService(
            verify_fn=(
                lambda image, extraction, attempt:
                verifier_model.verify(
                    image_path=page_image,
                    extraction=extraction,
                    attempt=attempt,
                )
            )
        )
    )

    feedback: str | None = None

    last_extraction: PageExtractionResult | None = None

    last_verification = None

    # ---------------------------------------------------------
    # Retry loop
    # ---------------------------------------------------------

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        print()
        print("*" * 80)
        print(
            f"WORKFLOW ATTEMPT {attempt}/{MAX_ATTEMPTS}"
        )
        print("*" * 80)

        tile_dir = (
            TILE_OUTPUT_DIR
            / f"page_{page_no:03d}"
            / f"attempt_{attempt}"
        )

        last_extraction = (
            extraction_with_feedback(
                extractor=extractor,
                image_path=page_image,
                page_no=page_no,
                tile_dir=tile_dir,
                feedback=feedback,
            )
        )

        print()
        print(
            "EXTRACTION RESULT:"
        )
        print(
            json.dumps(
                last_extraction.model_dump(),
                indent=2,
                ensure_ascii=False,
            )
        )

        # -----------------------------------------------------
        # Verification
        # -----------------------------------------------------

        dummy_image = (
            __import__(
                "PIL.Image",
                fromlist=["Image"],
            ).Image.open(
                page_image
            )
        )

        try:
            last_verification = (
                verifier_service.verify(
                    image=dummy_image,
                    extraction=last_extraction,
                    attempt=attempt,
                )
            )

        finally:
            dummy_image.close()

        print()
        print(
            "VERIFICATION RESULT:"
        )

        print(
            json.dumps(
                last_verification.model_dump(),
                indent=2,
                ensure_ascii=False,
            )
        )

        # -----------------------------------------------------
        # PASS
        # -----------------------------------------------------

        if last_verification.passed:
            print()
            print(
                "=" * 80
            )
            print(
                f"PAGE {page_no} EXTRACTION VERIFIED"
            )
            print(
                f"Attempts used: {attempt}"
            )
            print(
                "=" * 80
            )
            return

        # -----------------------------------------------------
        # FAIL -> FEEDBACK
        # -----------------------------------------------------

        feedback = (
            verifier_service.build_feedback(
                last_verification
            )
        )

        print()
        print(
            "VERIFIER FEEDBACK FOR NEXT ATTEMPT:"
        )
        print(
            feedback
        )

    # ---------------------------------------------------------
    # Final review state
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"PAGE {page_no} DID NOT PASS VERIFICATION"
    )
    print(
        "Workflow should move to REVIEW_REQUIRED."
    )
    print("=" * 80)


def test_live_student_23_extraction_verification() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Student PDF not found: "
            f"{PDF_PATH.resolve()}"
        )

    document = pymupdf.open(
        PDF_PATH
    )

    try:
        total_pages = len(
            document
        )

    finally:
        document.close()

    print()
    print("=" * 90)
    print(
        "TCO COPY CHECKER"
    )
    print(
        "LIVE EXTRACTION + VERIFICATION WORKFLOW"
    )
    print("=" * 90)

    print(
        f"PDF: {PDF_PATH.resolve()}"
    )

    print(
        f"Total pages: {total_pages}"
    )

    print(
        f"Verifier model: "
        f"{ExtractionVerifierModel().model_name}"
    )

    # ---------------------------------------------------------
    # For the first live workflow test we process ONE page.
    #
    # This keeps the debugging cycle fast.
    # ---------------------------------------------------------

    run_page_verification(
        page_no=1
    )

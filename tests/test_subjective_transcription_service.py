import json
from pathlib import Path

import pytest

from app.services.subjective_transcription_service import (
    SubjectiveTranscriptionService,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

# Change only these two values if required.
PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "test_copies"
    / "Student_23.pdf"
)

PAGE_NO = 2

PAGE_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "pages"
    / "subjective_service_test"
    / f"page_{PAGE_NO:03d}.png"
)

FALLBACK_DIR = (
    PROJECT_ROOT
    / "data"
    / "pages"
    / "subjective_service_test"
    / "fallback_regions"
)


def render_page() -> Path:

    import pymupdf

    PAGE_IMAGE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = pymupdf.open(
        str(PDF_PATH)
    )

    try:

        if PAGE_NO < 1 or PAGE_NO > len(document):

            raise ValueError(
                f"PDF has {len(document)} pages, "
                f"but PAGE_NO={PAGE_NO}."
            )

        page = document.load_page(
            PAGE_NO - 1
        )

        dpi = 300

        zoom = dpi / 72.0

        matrix = pymupdf.Matrix(
            zoom,
            zoom,
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        pixmap.save(
            str(PAGE_IMAGE)
        )

    finally:

        document.close()

    return PAGE_IMAGE


def test_subjective_transcription_service():

    print("\n")
    print("=" * 70)
    print(
        "TCO COPY CHECKER"
    )
    print(
        "SUBJECTIVE TRANSCRIPTION SERVICE TEST"
    )
    print("=" * 70)

    if not PDF_PATH.exists():

        pytest.fail(
            f"Test PDF not found:\n"
            f"{PDF_PATH}"
        )

    # --------------------------------------------------------
    # Render page
    # --------------------------------------------------------

    page_image = render_page()

    print(
        f"\nPDF: {PDF_PATH}"
    )

    print(
        f"Page: {PAGE_NO}"
    )

    print(
        f"Rendered image: {page_image}"
    )

    # --------------------------------------------------------
    # Service
    # --------------------------------------------------------

    service = (
        SubjectiveTranscriptionService()
    )

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    result = service.extract_page(
        image_path=page_image,
        page_no=PAGE_NO,
        fallback_output_dir=FALLBACK_DIR,
        force_fallback=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "FINAL SUBJECTIVE RESULT"
    )
    print("=" * 70)

    print(
        json.dumps(
            result.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Human-readable output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "TRANSCRIBED ANSWERS"
    )
    print("=" * 70)

    for block in result.blocks:

        question = (
            f"Q{block.question_no}"
            if block.question_no is not None
            else "Q[UNCLEAR]"
        )

        print("\n")
        print(
            f"{question}"
            f" | status={block.status}"
            f" | confidence={block.confidence:.2f}"
        )

        print(
            block.answer_text
            or "[NO TEXT]"
        )

        if block.equations:

            print(
                "Equations:"
            )

            for equation in block.equations:

                print(
                    f"  - {equation.text}"
                )

        if block.diagrams:

            print(
                "Diagrams:"
            )

            for diagram in block.diagrams:

                print(
                    f"  - "
                    f"{diagram.description}"
                )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print("\n")
    print(
        "Extraction mode:",
        result.extraction_mode,
    )

    print(
        "Fallback used:",
        result.fallback_used,
    )

    print(
        "Regions processed:",
        result.regions_processed,
    )

    print(
        "Review required:",
        result.review_required,
    )

    if result.warnings:

        print("\nWarnings:")

        for warning in result.warnings:

            print(
                f"- {warning}"
            )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert (
        result.page_no
        == PAGE_NO
    )

    assert isinstance(
        result.blocks,
        list,
    )

    assert (
        result.extraction_mode
        in (
            "whole_page",
            "segmented_fallback",
        )
    )

    assert (
        result.regions_processed
        >= 1
    )

    for block in result.blocks:

        assert (
            isinstance(
                block.answer_text,
                str,
            )
        )

        assert (
            0
            <= block.confidence
            <= 1
        )

        assert block.status in (
            "EXTRACTED",
            "REVIEW_REQUIRED",
        )

    # --------------------------------------------------------
    # Passed
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "SUBJECTIVE TRANSCRIPTION SERVICE TEST PASSED"
    )
    print("=" * 70)
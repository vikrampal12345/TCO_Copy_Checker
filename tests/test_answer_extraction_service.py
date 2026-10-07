import json
from pathlib import Path

import pytest

from app.services.answer_extraction_service import (
    AnswerExtractionService,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PAGE_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "pages"
    / "Student_2"
    / "page_001_vision.jpg"
)

TILE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "pages"
    / "Student_2"
    / "service_tiles_recovery"
)


def test_answer_extraction_service():

    print("\n")
    print("=" * 70)
    print(
        "TCO COPY CHECKER"
    )
    print(
        "ANSWER EXTRACTION + RECOVERY TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not PAGE_IMAGE.exists():

        pytest.fail(
            f"Test page not found:\n"
            f"{PAGE_IMAGE}"
        )

    # --------------------------------------------------------
    # Expected questions
    #
    # Student_2 page currently appears to be a 20-question
    # MCQ page, so we explicitly test against Q1-Q20.
    # --------------------------------------------------------

    expected_questions = list(
        range(
            1,
            21,
        )
    )

    print(
        "\nExpected questions:",
        expected_questions,
    )

    # --------------------------------------------------------
    # Service
    # --------------------------------------------------------

    service = (
        AnswerExtractionService()
    )

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    result = service.extract_page(
        image_path=PAGE_IMAGE,
        page_no=1,
        tile_output_dir=TILE_OUTPUT_DIR,
        expected_question_numbers=(
            expected_questions
        ),
        enable_recovery=True,
    )

    # --------------------------------------------------------
    # Complete result
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "FINAL PAGE EXTRACTION RESULT"
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
    # Question summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "QUESTION SUMMARY"
    )
    print("=" * 70)

    for question in result.questions:

        print(
            f"Q{question.question_no}: "
            f"{question.answer} "
            f"| confidence="
            f"{question.confidence:.2f} "
            f"| status="
            f"{question.status} "
            f"| source="
            f"{question.source_tile}"
        )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    print("\n")
    print(
        "Missing questions:",
        result.missing_questions,
    )

    # --------------------------------------------------------
    # Recovery
    # --------------------------------------------------------

    print(
        "Recovery attempted:",
        result.recovery_attempted,
    )

    print(
        "Recovery tiles:",
        result.recovery_tiles,
    )

    # --------------------------------------------------------
    # Review
    # --------------------------------------------------------

    print(
        "Review required:",
        result.review_required,
    )

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    if result.warnings:

        print("\nWarnings:")

        for warning in result.warnings:

            print(
                f"- {warning}"
            )

    # ========================================================
    # ASSERTIONS
    # ========================================================

    assert (
        result.page_no
        == 1
    )

    assert (
        result.processed_tiles
        == 8
    )

    assert (
        result.recovery_tiles
        in (
            0,
            8,
        )
    )

    assert isinstance(
        result.questions,
        list,
    )

    assert isinstance(
        result.missing_questions,
        list,
    )

    # --------------------------------------------------------
    # Validate question objects
    # --------------------------------------------------------

    for question in result.questions:

        assert (
            question.question_no
            is not None
        )

        assert isinstance(
            question.answer,
            str,
        )

        assert (
            0
            <= question.confidence
            <= 1
        )

        assert question.status in (
            "EXTRACTED",
            "REVIEW_REQUIRED",
        )

    # --------------------------------------------------------
    # No duplicate final question numbers
    # --------------------------------------------------------

    question_numbers = [
        item.question_no
        for item in result.questions
    ]

    assert len(
        question_numbers
    ) == len(
        set(question_numbers)
    )

    # --------------------------------------------------------
    # Missing questions must be from expected list
    # --------------------------------------------------------

    for missing in (
        result.missing_questions
    ):

        assert missing in (
            expected_questions
        )

    # --------------------------------------------------------
    # Final result can be serialized
    # --------------------------------------------------------

    serialized = result.model_dump_json()

    assert (
        isinstance(
            serialized,
            str,
        )
    )

    assert serialized.strip()

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "ANSWER EXTRACTION + RECOVERY TEST PASSED"
    )
    print("=" * 70)
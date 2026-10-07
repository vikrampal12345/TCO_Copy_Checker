from app.schemas.extraction import ExtractedAnswer, PageExtractionResult
from app.services.submission_aggregation_service import (
    SubmissionAggregationService,
)


def make_page(
    page_no: int,
    questions: list[ExtractedAnswer],
    *,
    review_required: bool = False,
    unresolved_items=None,
    warnings=None,
):
    return PageExtractionResult(
        page_no=page_no,
        questions=questions,
        unresolved_items=unresolved_items or [],
        missing_questions=[],
        review_required=review_required,
        warnings=warnings or [],
        processed_tiles=8,
        recovery_attempted=False,
        recovery_tiles=0,
    )


def test_aggregates_questions_across_pages():
    service = SubmissionAggregationService()

    pages = [
        make_page(
            1,
            [
                ExtractedAnswer(
                    question_no=1,
                    answer="The answer to question one",
                    confidence=0.90,
                    source_tile="tile_01",
                    status="EXTRACTED",
                )
            ],
        ),
        make_page(
            2,
            [
                ExtractedAnswer(
                    question_no=2,
                    answer="The answer to question two",
                    confidence=0.95,
                    source_tile="tile_03",
                    status="EXTRACTED",
                )
            ],
        ),
    ]

    result = service.aggregate(
        pages,
        job_id="JOB001",
        student_number="STU047",
        expected_question_numbers=[1, 2],
    )

    assert result.job_id == "JOB001"
    assert result.student_number == "STU047"
    assert result.total_pages == 2
    assert result.processed_pages == 2
    assert [item.question_no for item in result.answers] == [1, 2]
    assert result.missing_questions == []
    assert result.review_required is False


def test_detects_missing_questions():
    service = SubmissionAggregationService()

    pages = [
        make_page(
            1,
            [
                ExtractedAnswer(
                    question_no=1,
                    answer="Answer one",
                    confidence=0.90,
                    status="EXTRACTED",
                )
            ],
        )
    ]

    result = service.aggregate(
        pages,
        expected_question_numbers=[1, 2, 3],
    )

    assert result.missing_questions == [2, 3]
    assert result.review_required is True


def test_detects_repeated_question():
    service = SubmissionAggregationService()

    pages = [
        make_page(
            1,
            [
                ExtractedAnswer(
                    question_no=3,
                    answer="First observation",
                    confidence=0.80,
                    status="EXTRACTED",
                )
            ],
        ),
        make_page(
            2,
            [
                ExtractedAnswer(
                    question_no=3,
                    answer="First observation",
                    confidence=0.95,
                    status="EXTRACTED",
                )
            ],
        ),
    ]

    result = service.aggregate(pages)

    assert result.repeated_questions == [3]
    assert result.conflicting_questions == []
    assert result.answers[0].answer == "First observation"
    assert result.answers[0].confidence == 0.95


def test_detects_conflicting_observations():
    service = SubmissionAggregationService()

    pages = [
        make_page(
            1,
            [
                ExtractedAnswer(
                    question_no=5,
                    answer="Photosynthesis occurs in plants",
                    confidence=0.80,
                    status="EXTRACTED",
                )
            ],
        ),
        make_page(
            2,
            [
                ExtractedAnswer(
                    question_no=5,
                    answer="Newton discovered gravity",
                    confidence=0.90,
                    status="EXTRACTED",
                )
            ],
        ),
    ]

    result = service.aggregate(pages)

    assert result.repeated_questions == [5]
    assert result.conflicting_questions == [5]
    assert result.review_required is True
    assert result.answers[0].review_required is True


def test_unreadable_question_requires_review():
    service = SubmissionAggregationService()

    pages = [
        make_page(
            1,
            [
                ExtractedAnswer(
                    question_no=7,
                    answer="[UNCLEAR]",
                    confidence=0.10,
                    status="REVIEW_REQUIRED",
                )
            ],
        )
    ]

    result = service.aggregate(pages)

    assert len(result.answers) == 1
    assert result.answers[0].answer is None
    assert result.answers[0].review_required is True
    assert result.review_required is True

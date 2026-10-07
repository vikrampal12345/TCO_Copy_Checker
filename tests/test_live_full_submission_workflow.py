from app.services.checked_copy_renderer import CheckedCopyRenderer
from app.services.grading_model import GradingModel
from app.services.in_memory_answer_extraction_service import (
    InMemoryAnswerExtractionService,
)
from app.services.in_memory_grading_workflow import (
    InMemoryGradingWorkflow,
)
from app.services.in_memory_page_processor import (
    InMemoryPageProcessor,
)
from app.services.submission_aggregation_service import (
    SubmissionAggregationService,
)
from app.schemas.marking_scheme import (
    MarkingCriterion,
    MarkingScheme,
    QuestionMarkingScheme,
)


PDF_PATH = r"data\test_copies\Student_23.pdf"
TEST_PAGE_LIMIT = 2


def build_test_marking_scheme() -> MarkingScheme:
    """
    Temporary teacher-approved/locked marking scheme for the live
    pipeline test.

    This is only for testing the workflow architecture.
    The production flow will receive the real teacher-approved
    marking scheme from the TCO Master Agent.
    """

    question_numbers = [
        *range(1, 19),
        29,
        30,
    ]

    questions = []

    for question_no in question_numbers:
        questions.append(
            QuestionMarkingScheme(
                question_no=question_no,
                max_marks=1.0,
                criteria=[
                    MarkingCriterion(
                        criterion_id=f"Q{question_no}-C1",
                        description=(
                            "Award up to 1 mark for a valid and "
                            "relevant student response."
                        ),
                        marks=1.0,
                        accepted_points=[],
                    )
                ],
                evaluation_guidance=(
                    "Evaluate the student's extracted answer "
                    "according to the criterion."
                ),
            )
        )

    return MarkingScheme(
        assessment_id="ASSESSMENT-LIVE-001",
        version=1,
        questions=questions,
        total_marks=float(len(question_numbers)),
        status="locked",
        generated_by="teacher",
    )


def test_live_full_submission_workflow():

    print("\n" + "=" * 100)
    print("TCO COPY CHECKER")
    print("2-PAGE EXTRACTION -> COMBINE -> GRADING -> CHECKED COPY")
    print("=" * 100)
    print(f"PDF: {PDF_PATH}")

    processor = InMemoryPageProcessor()
    extractor = InMemoryAnswerExtractionService()
    grader = GradingModel()
    renderer = CheckedCopyRenderer()

    actual_page_count = processor.get_page_count(
        PDF_PATH
    )

    print(
        f"PDF total pages: "
        f"{actual_page_count}"
    )

    print(
        f"Extraction model: "
        f"{extractor.vision_service.get_model_name()}"
    )

    print(
        f"Grading model: "
        f"{grader.model}"
    )

    print(
        f"Live test scope: "
        f"FIRST {TEST_PAGE_LIMIT} PAGES ONLY"
    )

    print("Extraction recovery: DISABLED")
    print("Extraction verification model: DISABLED")
    print("Grading review model: DISABLED")
    print()

    assert actual_page_count >= TEST_PAGE_LIMIT, (
        f"Expected at least {TEST_PAGE_LIMIT} pages, "
        f"found {actual_page_count}"
    )

    marking_scheme = (
        build_test_marking_scheme()
    )

    print(
        f"Marking scheme: "
        f"{len(marking_scheme.questions)} questions"
    )

    def page_count_fn(
        pdf_path: str,
    ) -> int:
        return min(
            processor.get_page_count(
                pdf_path
            ),
            TEST_PAGE_LIMIT,
        )

    def render_page_fn(
        pdf_path: str,
        page_no: int,
    ):
        return processor.render_page(
            pdf_path,
            page_no,
        )

    def extract_page_fn(
        image,
        page_no: int,
        expected_question_numbers: list[int],
        enable_recovery: bool,
        feedback: str | None,
    ):

        print("\n" + "=" * 100)
        print(
            f"WHOLE-PAGE EXTRACTION "
            f"PAGE {page_no}"
        )
        print("=" * 100)

        print(
            "Verifier disabled. "
            "Single extraction pass."
        )

        return extractor.extract_page_in_memory(
            image=image,
            page_no=page_no,
            expected_question_numbers=(
                expected_question_numbers
            ),
            enable_recovery=False,
            feedback=None,
        )

    workflow = InMemoryGradingWorkflow(
        page_count_fn=page_count_fn,
        render_page_fn=render_page_fn,
        extract_page_fn=extract_page_fn,
        aggregation_service=(
            SubmissionAggregationService()
        ),
        grading_model=grader,
        checked_copy_renderer=renderer,
        max_pages=TEST_PAGE_LIMIT,
    )

    result, submission, checked_copy_path = (
        workflow.run(
            pdf_path=PDF_PATH,
            marking_scheme=marking_scheme,
            job_id="LIVE-GRADING-001",
            student_number="STU023",
            assessment_id="ASSESSMENT-LIVE-001",
            expected_question_numbers=[],
            enable_recovery=False,
            metadata={
                "test": True,
                "scope": "first_2_pages_only",
                "verification": "disabled",
                "grading_review": "disabled",
            },
        )
    )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    print("\n" + "=" * 100)
    print("FINAL GRADING RESULT")
    print("=" * 100)

    print(
        f"Status: "
        f"{result.status}"
    )

    print(
        f"Student: "
        f"{result.student_id}"
    )

    print(
        f"Assessment: "
        f"{result.assessment_id}"
    )

    print(
        f"Total marks: "
        f"{result.total_marks}"
    )

    print(
        f"Obtained marks: "
        f"{result.obtained_marks}"
    )

    print(
        f"Percentage: "
        f"{result.percentage:.2f}%"
    )

    print(
        f"Pages: "
        f"{result.pages_processed} / "
        f"{result.total_pages}"
    )

    print("\nQUESTION-WISE MARKS")

    for page in result.pages:

        print(
            f"\nPage {page['page_number']}"
        )

        for question in page["questions"]:

            print(
                f"Q{question['question_no']}: "
                f"{question['marks_awarded']} / "
                f"{question['max_marks']} | "
                f"{question['reason']}"
            )

    print("\n" + "=" * 100)
    print("CHECKED COPY")
    print("=" * 100)
    print(
        f"Created: "
        f"{checked_copy_path}"
    )

    # =========================================================
    # SAFETY ASSERTIONS
    # =========================================================

    assert result.status == "graded"

    assert result.student_id == "STU023"

    assert result.assessment_id == (
        "ASSESSMENT-LIVE-001"
    )

    assert result.total_pages == TEST_PAGE_LIMIT

    assert result.pages_processed == (
        TEST_PAGE_LIMIT
    )

    assert result.total_marks == (
        float(
            len(marking_scheme.questions)
        )
    )

    assert result.obtained_marks is not None

    assert result.percentage is not None

    assert 0 <= result.obtained_marks <= result.total_marks

    assert 0 <= result.percentage <= 100

    assert isinstance(
        checked_copy_path,
        str,
    )

    assert checked_copy_path.endswith(
        "_checked.pdf"
    )

    assert submission.review_required is False

    print("\n" + "=" * 100)
    print(
        "2-PAGE EXTRACTION -> COMBINE -> "
        "GRADING -> CHECKED COPY COMPLETED"
    )
    print("=" * 100)


# =========================================================================
# FUTURE GRADING REVIEW MODEL
# =========================================================================
#
# CURRENTLY DISABLED.
#
# Later, after a better independent model is available:
#
#     grading_result = GradingModel(...)
#
#                    ↓
#
#     GradingReviewModel(...)
#
#                    ↓
#
#              ┌─────┴─────┐
#              ↓           ↓
#            PASS         FAIL
#              ↓           ↓
#       Teacher Review   Re-grade affected
#                        questions
#
# The review model should independently inspect:
#     1. Student extracted answers
#     2. Locked marking scheme
#     3. Awarded marks
#     4. Grading reasons
#
# It must NOT be the same grading call.
# =========================================================================

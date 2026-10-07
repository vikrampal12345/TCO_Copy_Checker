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
    Temporary teacher-approved/locked marking scheme for
    subjective grading validation.

    This test intentionally grades only the subjective questions
    extracted from page 2 of Student_23.pdf.

    Production will receive the real teacher-approved marking
    scheme from the TCO Master Agent.
    """

    question_data = {
        21: {
            "guidance": (
                "Evaluate the definition of supervised learning. "
                "A strong answer should explain that the model is trained "
                "using labelled data containing input data and corresponding "
                "correct output/target values, and that the model learns the "
                "relationship needed to predict outputs."
            ),
            "criteria": [
                (
                    "Q21-C1",
                    "Identifies supervised learning as training with labelled "
                    "input data and corresponding correct output/target data.",
                    1.0,
                    [
                        "labelled data",
                        "input data",
                        "correct output",
                        "target output",
                        "input-output pairs",
                    ],
                ),
                (
                    "Q21-C2",
                    "Explains that the model learns the relationship from the "
                    "training examples so it can predict the output.",
                    1.0,
                    [
                        "learns mapping",
                        "learns relationship",
                        "predict output",
                        "training examples",
                    ],
                ),
            ],
        },
        23: {
            "guidance": (
                "Evaluate the explanation of regression analysis as a "
                "machine-learning technique for predicting a desired or "
                "numerical/continuous value from data."
            ),
            "criteria": [
                (
                    "Q23-C1",
                    "Identifies regression as a machine-learning technique "
                    "used for prediction.",
                    1.0,
                    [
                        "machine learning technique",
                        "prediction",
                        "predict",
                    ],
                ),
                (
                    "Q23-C2",
                    "Explains that regression predicts a numerical or "
                    "continuous/desired value from data.",
                    1.0,
                    [
                        "numerical value",
                        "continuous value",
                        "desired value",
                        "value from dataset",
                    ],
                ),
            ],
        },
        24: {
            "guidance": (
                "Evaluate whether the student identifies and explains "
                "relevant challenges in machine-learning work, especially "
                "data collection/cleaning/management and difficulty "
                "obtaining or analyzing correct outputs."
            ),
            "criteria": [
                (
                    "Q24-C1",
                    "Identifies a valid challenge related to collecting, "
                    "cleaning, preprocessing, or managing datasets.",
                    1.0,
                    [
                        "data collection",
                        "data cleaning",
                        "data preprocessing",
                        "dataset management",
                    ],
                ),
                (
                    "Q24-C2",
                    "Identifies a valid challenge related to obtaining, "
                    "analyzing, or interpreting correct model outputs.",
                    1.0,
                    [
                        "correct output",
                        "analyze output",
                        "output analysis",
                        "prediction analysis",
                    ],
                ),
            ],
        },
        25: {
            "guidance": (
                "Evaluate whether the student correctly identifies major "
                "applications of data science and gives relevant examples."
            ),
            "criteria": [
                (
                    "Q25-C1",
                    "Identifies healthcare as a valid application of "
                    "data science.",
                    1.0,
                    [
                        "healthcare",
                        "health care",
                        "medical",
                    ],
                ),
                (
                    "Q25-C2",
                    "Identifies finance as a valid application of "
                    "data science.",
                    1.0,
                    [
                        "finance",
                        "financial",
                    ],
                ),
            ],
        },
        26: {
            "guidance": (
                "Evaluate the distinction between structured data and "
                "semi-structured data. Structured data is highly organized "
                "and follows a defined structure/schema. Semi-structured "
                "data has some organization but does not require the same "
                "rigid tabular structure."
            ),
            "criteria": [
                (
                    "Q26-C1",
                    "Correctly describes structured data as highly organized "
                    "and following a defined structure/schema.",
                    1.0,
                    [
                        "structured data",
                        "highly organized",
                        "defined structure",
                        "schema",
                    ],
                ),
                (
                    "Q26-C2",
                    "Correctly distinguishes semi-structured data as having "
                    "some organization without the same rigid structure.",
                    1.0,
                    [
                        "semi-structured data",
                        "some organization",
                        "less rigid structure",
                        "not fully structured",
                    ],
                ),
            ],
        },
    }

    questions: list[QuestionMarkingScheme] = []

    for question_no, data in question_data.items():
        criteria = [
            MarkingCriterion(
                criterion_id=criterion_id,
                description=description,
                marks=marks,
                accepted_points=accepted_points,
            )
            for (
                criterion_id,
                description,
                marks,
                accepted_points,
            ) in data["criteria"]
        ]

        questions.append(
            QuestionMarkingScheme(
                question_no=question_no,
                max_marks=2.0,
                question_type="subjective",
                criteria=criteria,
                evaluation_guidance=data["guidance"],
            )
        )

    return MarkingScheme(
        assessment_id="ASSESSMENT-SUBJECTIVE-LIVE-001",
        version=1,
        questions=questions,
        total_marks=sum(
            question.max_marks
            for question in questions
        ),
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

    assert result.total_marks == marking_scheme.total_marks

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





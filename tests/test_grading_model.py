from types import SimpleNamespace

import pytest

from app.schemas.marking_scheme import (
    MarkingCriterion,
    MarkingScheme,
    QuestionMarkingScheme,
)
from app.schemas.submission import (
    AggregatedAnswer,
    AggregatedSubmission,
    AnswerObservation,
)
from app.services.grading_model import (
    GradingModel,
    GradingModelError,
)


class FakeCompletions:

    def create(self, **kwargs):

        content = """
        {
          "questions": [
            {
              "question_no": 1,
              "marks_awarded": 1.5,
              "reason": "Correct core concept but explanation is incomplete.",
              "confidence": 0.90
            },
            {
              "question_no": 2,
              "marks_awarded": 2.0,
              "reason": "Answer satisfies all marking criteria.",
              "confidence": 0.95
            }
          ]
        }
        """

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content
                    )
                )
            ]
        )


class FakeClient:

    def __init__(self):
        self.chat = SimpleNamespace(
            completions=FakeCompletions()
        )


class FakeProvider:

    def __init__(self):
        self.client = FakeClient()

    def get_client(self):
        return self.client

    def get_text_model(self):
        return "fake-grading-model"


def build_scheme(
    status: str = "locked",
) -> MarkingScheme:

    return MarkingScheme(
        assessment_id="ASSESS-001",
        version=1,
        status=status,
        generated_by="teacher",
        total_marks=5.0,
        questions=[
            QuestionMarkingScheme(
                question_no=1,
                max_marks=2.0,
                criteria=[
                    MarkingCriterion(
                        criterion_id="Q1-C1",
                        description="Correct concept",
                        marks=2.0,
                    )
                ],
            ),
            QuestionMarkingScheme(
                question_no=2,
                max_marks=3.0,
                criteria=[
                    MarkingCriterion(
                        criterion_id="Q2-C1",
                        description="Complete answer",
                        marks=3.0,
                    )
                ],
            ),
        ],
    )


def build_submission() -> AggregatedSubmission:

    return AggregatedSubmission(
        job_id="JOB-001",
        student_number="STU023",
        total_pages=2,
        processed_pages=2,
        answers=[
            AggregatedAnswer(
                question_no=1,
                answer="The student answer for Q1.",
                confidence=0.95,
                observations=[
                    AnswerObservation(
                        page_no=1,
                        question_no=1,
                        answer="The student answer for Q1.",
                        confidence=0.95,
                        source_tile="page_001",
                    )
                ],
            ),
            AggregatedAnswer(
                question_no=2,
                answer="The student answer for Q2.",
                confidence=0.92,
                observations=[
                    AnswerObservation(
                        page_no=2,
                        question_no=2,
                        answer="The student answer for Q2.",
                        confidence=0.92,
                        source_tile="page_002",
                    )
                ],
            ),
        ],
        review_required=False,
    )


def test_grading_returns_question_marks_and_total():

    grader = GradingModel(
        provider=FakeProvider()
    )

    result = grader.grade_submission(
        submission=build_submission(),
        marking_scheme=build_scheme(),
    )

    assert result.status == "graded"
    assert result.total_marks == 5.0
    assert result.obtained_marks == 3.5
    assert result.percentage == 70.0

    assert len(result.pages) == 2

    questions = [
        question
        for page in result.pages
        for question in page["questions"]
    ]

    assert len(questions) == 2

    assert questions[0]["question_no"] == "1"
    assert questions[0]["marks_awarded"] == 1.5

    assert questions[1]["question_no"] == "2"
    assert questions[1]["marks_awarded"] == 2.0


def test_grading_requires_locked_scheme():

    grader = GradingModel(
        provider=FakeProvider()
    )

    with pytest.raises(
        GradingModelError,
        match="locked marking scheme",
    ):
        grader.grade_submission(
            submission=build_submission(),
            marking_scheme=build_scheme(
                status="approved"
            ),
        )


def test_grading_rejects_review_required_submission():

    grader = GradingModel(
        provider=FakeProvider()
    )

    submission = build_submission()
    submission.review_required = True

    with pytest.raises(
        GradingModelError,
        match="requires review before grading",
    ):
        grader.grade_submission(
            submission=submission,
            marking_scheme=build_scheme(),
        )

# ============================================================
# MCQ TESTS
# ============================================================


# ============================================================
# MCQ TESTS
# ============================================================


class FakeMCQCompletions:

    def create(self, **kwargs):

        # Intentionally return an incorrect model score.
        # Deterministic MCQ scoring must override it.
        content = """
        {
          "questions": [
            {
              "question_no": 1,
              "marks_awarded": 0,
              "reason": "Model deliberately returned zero.",
              "confidence": 0.50
            }
          ]
        }
        """

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content
                    )
                )
            ]
        )


class FakeMCQClient:

    def __init__(self):
        self.chat = SimpleNamespace(
            completions=FakeMCQCompletions()
        )


class FakeMCQProvider:

    def __init__(self):
        self.client = FakeMCQClient()

    def get_client(self):
        return self.client

    def get_text_model(self):
        return "fake-mcq-grading-model"


def build_mcq_scheme(
    correct_answer: str | None = "B",
) -> MarkingScheme:

    return MarkingScheme(
        assessment_id="MCQ-TEST-001",
        version=1,
        status="locked",
        generated_by="teacher",
        total_marks=1.0,
        questions=[
            QuestionMarkingScheme(
                question_no=1,
                max_marks=1.0,
                question_type="mcq",
                correct_answer=correct_answer,
            )
        ],
    )


def build_mcq_submission(
    answer: str,
) -> AggregatedSubmission:

    return AggregatedSubmission(
        job_id="MCQ-JOB-001",
        student_number="STU001",
        total_pages=1,
        processed_pages=1,
        answers=[
            AggregatedAnswer(
                question_no=1,
                answer=answer,
                confidence=0.95,
                observations=[
                    AnswerObservation(
                        page_no=1,
                        question_no=1,
                        answer=answer,
                        confidence=0.95,
                        source_tile="page_001",
                    )
                ],
            )
        ],
        review_required=False,
    )


def test_mcq_deterministic_scoring_correct_answer():

    grader = GradingModel(
        provider=FakeMCQProvider()
    )

    result = grader.grade_submission(
        submission=build_mcq_submission("B"),
        marking_scheme=build_mcq_scheme("B"),
    )

    assert result.obtained_marks == 1.0
    assert result.percentage == 100.0

    question = result.pages[0]["questions"][0]

    assert question["marks_awarded"] == 1.0
    assert question["confidence"] == 1.0
    assert "approved MCQ answer key" in question["reason"]


def test_mcq_deterministic_scoring_wrong_answer():

    grader = GradingModel(
        provider=FakeMCQProvider()
    )

    result = grader.grade_submission(
        submission=build_mcq_submission("C"),
        marking_scheme=build_mcq_scheme("B"),
    )

    assert result.obtained_marks == 0.0
    assert result.percentage == 0.0

    question = result.pages[0]["questions"][0]

    assert question["marks_awarded"] == 0.0
    assert question["confidence"] == 1.0
    assert "does not match" in question["reason"]


@pytest.mark.parametrize(
    "student_answer",
    [
        "B",
        "b",
        "(B)",
        "[B]",
        "Option B",
        "Option B.",
        "B.",
    ],
)
def test_mcq_answer_normalization(student_answer):

    normalized = GradingModel._normalize_mcq_answer(
        student_answer
    )

    assert normalized == "b"


def test_mcq_requires_correct_answer():

    grader = GradingModel(
        provider=FakeProvider()
    )

    with pytest.raises(
        GradingModelError,
        match="MCQ requires a correct_answer",
    ):
        grader.grade_submission(
            submission=build_mcq_submission("B"),
            marking_scheme=build_mcq_scheme(None),
        )

from app.schemas.marking_scheme import (
    MarkingCriterion,
    MarkingScheme,
    QuestionMarkingScheme,
)

from app.services.marking_scheme_service import (
    MarkingSchemeService,
    MarkingSchemeValidationError,
)


def build_valid_scheme() -> MarkingScheme:

    return MarkingScheme(
        assessment_id="SCI_TEST_01",
        version=1,
        total_marks=10,
        status="proposed",
        questions=[
            QuestionMarkingScheme(
                question_no=21,
                max_marks=5,
                criteria=[
                    MarkingCriterion(
                        criterion_id="q21_c1",
                        description=(
                            "Defines supervised learning"
                        ),
                        marks=2,
                    ),
                    MarkingCriterion(
                        criterion_id="q21_c2",
                        description=(
                            "Explains labelled training data"
                        ),
                        marks=2,
                    ),
                    MarkingCriterion(
                        criterion_id="q21_c3",
                        description=(
                            "Provides a relevant example"
                        ),
                        marks=1,
                    ),
                ],
            ),
            QuestionMarkingScheme(
                question_no=22,
                max_marks=5,
                criteria=[
                    MarkingCriterion(
                        criterion_id="q22_c1",
                        description="Correct concept",
                        marks=3,
                    ),
                    MarkingCriterion(
                        criterion_id="q22_c2",
                        description="Relevant example",
                        marks=2,
                    ),
                ],
            ),
        ],
    )


def test_valid_marking_scheme():

    scheme = build_valid_scheme()

    validated = MarkingSchemeService.validate(
        scheme
    )

    assert validated.status == "proposed"
    assert validated.total_marks == 10


def test_reject_marking_scheme():

    scheme = build_valid_scheme()

    rejected = MarkingSchemeService.reject(
        scheme,
        "Q21 definition should have only 1 mark.",
    )

    assert rejected.status == "rejected"

    assert (
        rejected.teacher_feedback
        == "Q21 definition should have only 1 mark."
    )

    assert rejected.version == 2


def test_only_approved_scheme_can_be_locked():

    scheme = build_valid_scheme()

    try:
        MarkingSchemeService.lock(scheme)
        assert False
    except MarkingSchemeValidationError:
        pass


def test_approved_scheme_can_be_locked():

    scheme = build_valid_scheme()

    approved = MarkingSchemeService.approve(
        scheme
    )

    locked = MarkingSchemeService.lock(
        approved
    )

    assert locked.status == "locked"


def test_duplicate_question_numbers_are_rejected():

    scheme = build_valid_scheme()

    scheme.questions.append(
        QuestionMarkingScheme(
            question_no=21,
            max_marks=1,
            criteria=[],
        )
    )

    try:
        MarkingSchemeService.validate(scheme)
        assert False
    except MarkingSchemeValidationError as exc:
        assert "Duplicate question number" in str(exc)


def test_criteria_cannot_exceed_question_marks():

    scheme = MarkingScheme(
        assessment_id="TEST_01",
        total_marks=5,
        status="proposed",
        questions=[
            QuestionMarkingScheme(
                question_no=1,
                max_marks=5,
                criteria=[
                    MarkingCriterion(
                        criterion_id="c1",
                        description="Criterion 1",
                        marks=4,
                    ),
                    MarkingCriterion(
                        criterion_id="c2",
                        description="Criterion 2",
                        marks=3,
                    ),
                ],
            )
        ],
    )

    try:
        MarkingSchemeService.validate(scheme)
        assert False
    except MarkingSchemeValidationError as exc:
        assert "exceed question max marks" in str(exc)


def test_total_marks_must_match_questions():

    scheme = MarkingScheme(
        assessment_id="TEST_02",
        total_marks=20,
        status="proposed",
        questions=[
            QuestionMarkingScheme(
                question_no=1,
                max_marks=5,
                criteria=[],
            )
        ],
    )

    try:
        MarkingSchemeService.validate(scheme)
        assert False
    except MarkingSchemeValidationError as exc:
        assert "total marks" in str(exc)
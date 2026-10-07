from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# MARKING CRITERION
# ============================================================


class MarkingCriterion(BaseModel):
    """
    One criterion used to award marks for a question.

    Example:
        Defines supervised learning -> 2 marks
        Explains labelled data     -> 2 marks
        Relevant example           -> 1 mark
    """

    criterion_id: str = Field(
        min_length=1,
        description="Unique identifier for this criterion.",
    )

    description: str = Field(
        min_length=1,
        description="What the student must demonstrate.",
    )

    marks: float = Field(
        ge=0,
        description="Maximum marks assigned to this criterion.",
    )

    accepted_points: list[str] = Field(
        default_factory=list,
        description=(
            "Concepts/points that may satisfy this criterion. "
            "Exact wording is not required."
        ),
    )


# ============================================================
# QUESTION MARKING SCHEME
# ============================================================


class QuestionMarkingScheme(BaseModel):
    """
    Marking scheme for one question.
    """

    question_no: int = Field(
        ge=1,
        description="Question number.",
    )

    max_marks: float = Field(
        ge=0,
        description="Maximum marks for the question.",
    )

    question_type: Literal[
        "mcq",
        "true_false",
        "fill_in",
        "subjective",
        "numerical",
        "formula",
        "diagram",
        "mixed",
    ] = "subjective"

    # --------------------------------------------------------
    # MCQ ANSWER KEY
    # --------------------------------------------------------

    correct_answer: str | None = Field(
        default=None,
        description=(
            "Correct answer for objective questions such as MCQ. "
            "For MCQ this may be an option marker such as A, B, C, D "
            "or the corresponding answer text."
        ),
    )

    criteria: list[MarkingCriterion] = Field(
        default_factory=list,
        description="Criteria used for partial-credit evaluation.",
    )

    evaluation_guidance: str | None = Field(
        default=None,
        description="Additional guidance for evaluating this question.",
    )


# ============================================================
# COMPLETE MARKING SCHEME
# ============================================================


class MarkingScheme(BaseModel):
    """
    Complete marking scheme for one assessment.
    """

    assessment_id: str = Field(
        min_length=1,
        description="Assessment identifier.",
    )

    version: int = Field(
        default=1,
        ge=1,
        description="Revision/version number of the scheme.",
    )

    questions: list[QuestionMarkingScheme] = Field(
        default_factory=list,
    )

    total_marks: float = Field(
        ge=0,
        description="Total marks of the assessment.",
    )

    status: Literal[
        "not_provided",
        "proposed",
        "rejected",
        "approved",
        "locked",
    ] = "not_provided"

    generated_by: Literal[
        "teacher",
        "agent",
    ] = "agent"

    teacher_feedback: str | None = None

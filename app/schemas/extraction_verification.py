from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExtractionVerificationIssue(BaseModel):
    """
    One issue discovered by the extraction verification model.

    The verifier identifies the problem but does not silently
    modify the extracted answer.
    """

    question_no: int | None = Field(
        default=None,
        ge=1,
    )

    issue_type: str = Field(
        min_length=1,
    )

    severity: Literal[
        "low",
        "medium",
        "high",
    ] = "medium"

    message: str = Field(
        min_length=1,
    )


class ExtractionVerificationResult(BaseModel):
    """
    Result returned by the independent extraction verifier.

    The verifier answers:
        "Is the extraction faithful to the original page?"

    It does not perform grading.
    It does not directly modify the extraction.
    """

    page_no: int = Field(
        ge=1,
    )

    passed: bool

    confidence: float = Field(
        ge=0,
        le=1,
    )

    issues: list[ExtractionVerificationIssue] = Field(
        default_factory=list,
    )

    feedback: str | None = None

    warnings: list[str] = Field(
        default_factory=list,
    )

    attempt: int = Field(
        default=1,
        ge=1,
    )


class ExtractionVerificationLoopResult(BaseModel):
    """
    Final result of the extraction + verification + retry loop.
    """

    page_no: int = Field(
        ge=1,
    )

    passed: bool

    attempts: int = Field(
        ge=1,
    )

    extraction: dict = Field(
        default_factory=dict,
    )

    verification: ExtractionVerificationResult

    review_required: bool = False

    warnings: list[str] = Field(
        default_factory=list,
    )
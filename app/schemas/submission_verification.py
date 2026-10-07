from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SubmissionVerificationIssue(BaseModel):
    question_no: int | None = Field(default=None, ge=1)
    issue_type: str = Field(min_length=1)
    severity: Literal["low", "medium", "high"] = "medium"
    message: str = Field(min_length=1)


class SubmissionVerificationResult(BaseModel):
    passed: bool
    confidence: float = Field(ge=0, le=1)

    issues: list[SubmissionVerificationIssue] = Field(
        default_factory=list
    )

    feedback: str | None = None

    warnings: list[str] = Field(default_factory=list)

    attempt: int = Field(default=1, ge=1)


class SubmissionVerificationLoopResult(BaseModel):
    passed: bool
    attempts: int = Field(ge=1)

    submission: dict = Field(default_factory=dict)

    verification: SubmissionVerificationResult

    review_required: bool = False

    warnings: list[str] = Field(default_factory=list)

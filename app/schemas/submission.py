from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnswerObservation(BaseModel):
    page_no: int = Field(ge=1)
    question_no: int = Field(ge=1)
    answer: str
    confidence: float = Field(ge=0, le=1)
    source_tile: str | None = None
    status: str = "EXTRACTED"
    warning: str | None = None


class AggregatedAnswer(BaseModel):
    question_no: int = Field(ge=1)
    answer: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    observations: list[AnswerObservation] = Field(default_factory=list)
    status: str = "EXTRACTED"
    review_required: bool = False
    warning: str | None = None


class AggregatedSubmission(BaseModel):
    job_id: str | None = None
    student_number: str | None = None

    total_pages: int = Field(default=0, ge=0)
    processed_pages: int = Field(default=0, ge=0)

    answers: list[AggregatedAnswer] = Field(default_factory=list)

    expected_questions: list[int] = Field(default_factory=list)
    missing_questions: list[int] = Field(default_factory=list)
    repeated_questions: list[int] = Field(default_factory=list)
    conflicting_questions: list[int] = Field(default_factory=list)

    unresolved_items: int = Field(default=0, ge=0)

    review_required: bool = False

    warnings: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

from typing import Any

from pydantic import BaseModel, Field


class QuestionEvaluation(BaseModel):
    question_no: str

    max_marks: float | None = None
    marks_awarded: float | None = None

    extracted_answer: str | None = None
    reason: str | None = None

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PageResult(BaseModel):
    job_id: str
    page_number: int

    status: str

    questions: list[QuestionEvaluation] = []

    page_max_marks: float | None = None
    page_marks_awarded: float | None = None

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    warnings: list[str] = []
    metadata: dict[str, Any] = {}
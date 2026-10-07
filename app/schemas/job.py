from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    EVALUATION_READY = "evaluation_ready"
    REVIEW_REQUIRED = "review_required"
    COMPLETED = "completed"
    FAILED = "failed"


class PageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    EXTRACTED = "extracted"
    EVALUATED = "evaluated"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    FAILED = "failed"


class PageState(BaseModel):
    job_id: str
    page_number: int

    status: PageStatus = PageStatus.PENDING

    image_path: str | None = None

    extracted_answers: list[dict[str, Any]] = Field(
        default_factory=list
    )

    evaluation: dict[str, Any] | None = None

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    feedback: str | None = None

    warnings: list[str] = Field(
        default_factory=list
    )


class EvaluationJob(BaseModel):
    job_id: str

    status: JobStatus = JobStatus.CREATED

    student_id: str | None = None
    assessment_id: str | None = None

    subject: str | None = None
    class_level: str | None = None

    source_pdf: str

    total_pages: int
    processed_pages: int = 0

    pages: list[PageState] = Field(
        default_factory=list
    )

    total_marks: float | None = None
    obtained_marks: float | None = None
    percentage: float | None = None

    warnings: list[str] = Field(
        default_factory=list
    )
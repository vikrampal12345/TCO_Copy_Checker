from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    job_id: str

    status: str

    student_id: str | None = None
    assessment_id: str | None = None

    pages_processed: int = 0
    total_pages: int = 0

    total_marks: float | None = None
    obtained_marks: float | None = None

    percentage: float | None = None

    pages: list[dict] = []

    warnings: list[str] = []
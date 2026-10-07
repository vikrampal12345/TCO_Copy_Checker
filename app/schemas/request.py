from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    student_id: str | None = None
    assessment_id: str | None = None

    subject: str | None = None
    class_level: str | None = None

    total_marks: float | None = Field(default=None, gt=0)

    reference_text: str | None = None
    rubric_text: str | None = None
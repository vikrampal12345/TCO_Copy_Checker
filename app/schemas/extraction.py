from typing import Optional

from pydantic import BaseModel, Field


class ExtractedAnswer(BaseModel):
    """
    One question-answer candidate extracted from a page.
    """

    question_no: Optional[int] = None

    answer: str = "[UNCLEAR]"

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    source_tile: Optional[str] = None

    status: str = "REVIEW_REQUIRED"

    warning: Optional[str] = None


class PageExtractionResult(BaseModel):
    """
    Final extraction result for one page.
    """

    page_no: int

    questions: list[ExtractedAnswer] = Field(
        default_factory=list
    )

    unresolved_items: list[ExtractedAnswer] = Field(
        default_factory=list
    )

    missing_questions: list[int] = Field(
        default_factory=list
    )

    review_required: bool = False

    warnings: list[str] = Field(
        default_factory=list
    )

    processed_tiles: int = 0

    recovery_attempted: bool = False

    recovery_tiles: int = 0
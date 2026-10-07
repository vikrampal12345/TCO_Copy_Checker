from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SubjectiveLine(BaseModel):
    """
    One transcribed handwritten line.
    """

    text: str

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class SubjectiveEquation(BaseModel):
    """
    One equation/formula detected in the answer.
    """

    text: str

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class SubjectiveDiagram(BaseModel):
    """
    Description of a visible diagram/figure.
    """

    description: str

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class SubjectiveAnswerBlock(BaseModel):
    """
    One question's subjective answer.

    question_no can be None because question mapping may
    sometimes be unresolved.
    """

    question_no: Optional[int] = None

    answer_text: str = ""

    lines: list[SubjectiveLine] = Field(
        default_factory=list
    )

    equations: list[SubjectiveEquation] = Field(
        default_factory=list
    )

    diagrams: list[SubjectiveDiagram] = Field(
        default_factory=list
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    status: str = "REVIEW_REQUIRED"

    source_region: Optional[str] = None

    warning: Optional[str] = None


class SubjectivePageResult(BaseModel):
    """
    Complete subjective transcription result for one page.
    """

    page_no: int

    blocks: list[SubjectiveAnswerBlock] = Field(
        default_factory=list
    )

    extraction_mode: str = "whole_page"

    review_required: bool = False

    warnings: list[str] = Field(
        default_factory=list
    )

    regions_processed: int = 1

    fallback_used: bool = False
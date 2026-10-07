from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import settings
from app.schemas.evaluation_result import EvaluationResult
from app.services.page_service import PageService
from app.services.pdf_service import PDFService


class AnswerSheetEvaluationAgent:
    """
    Specialist agent for handwritten answer-sheet evaluation.

    Current responsibility:
        - identify student from answer-sheet filename
        - validate the PDF
        - create an evaluation job
        - render PDF pages
        - prepare the job for handwriting extraction

    Teacher-facing conversation is NOT handled here.
    """

    name = "answer_sheet_evaluation_agent"

    capability = "answer_sheet.evaluate"

    def __init__(self) -> None:
        self.pdf_service = PDFService()
        self.page_service = PageService()

    # ========================================================
    # STUDENT IDENTITY
    # ========================================================

    @staticmethod
    def extract_student_number(
        pdf_path: str | Path,
    ) -> str:

        path = Path(pdf_path)

        student_number = path.stem.strip()

        if not student_number:
            raise ValueError(
                "Student number could not be determined "
                "from the answer-sheet filename."
            )

        return student_number

    # ========================================================
    # JOB CREATION
    # ========================================================

    def create_job(
        self,
        pdf_path: str,
        assessment_id: str | None = None,
        student_number: str | None = None,
    ) -> EvaluationResult:

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise ValueError(
                f"Answer-sheet file not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF answer sheets are supported."
            )

        # ----------------------------------------------------
        # Student number
        # ----------------------------------------------------

        resolved_student_number = (
            student_number.strip()
            if student_number
            else self.extract_student_number(pdf_path)
        )

        if not resolved_student_number:
            raise ValueError(
                "Student number is required."
            )

        # ----------------------------------------------------
        # Validate PDF
        # ----------------------------------------------------

        self.pdf_service.validate_page_count(
            pdf_path,
            settings.max_pdf_pages,
        )

        # ----------------------------------------------------
        # Job ID
        # ----------------------------------------------------
        #
        # Keep student filename-based identity separate
        # from the internal job identifier.
        #

        job_id = pdf_path.stem

        # ----------------------------------------------------
        # Render pages
        # ----------------------------------------------------

        pages_dir = (
            Path(settings.pages_dir)
            / job_id
        )

        page_files = self.page_service.render_pages(
            pdf_path=pdf_path,
            output_dir=pages_dir,
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return EvaluationResult(
            job_id=job_id,
            status="pages_ready",
            student_id=resolved_student_number,
            assessment_id=assessment_id,
            pages_processed=0,
            total_pages=len(page_files),
            pages=[],
            warnings=[],
        )
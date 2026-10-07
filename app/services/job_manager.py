from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.schemas.job import (
    EvaluationJob,
    JobStatus,
    PageState,
    PageStatus,
)
from app.services.page_service import PageService
from app.services.pdf_service import PDFService


class JobManager:
    """
    Creates and manages evaluation jobs.

    This layer is intentionally independent of:
    - LLM providers
    - OCR providers
    - TCO database
    - React frontend

    That makes later TCO integration easier.
    """

    def __init__(self) -> None:
        self.pdf_service = PDFService()
        self.page_service = PageService()

    def create_job(
        self,
        pdf_path: str | Path,
        student_id: str | None = None,
        assessment_id: str | None = None,
        subject: str | None = None,
        class_level: str | None = None,
    ) -> EvaluationJob:
        """
        Create an evaluation job from a student answer-sheet PDF.

        Steps:
        1. Validate that the PDF exists.
        2. Validate the PDF page count.
        3. Generate a unique job ID.
        4. Render PDF pages into the job-specific page directory.
        5. Create a PageState for every rendered page.
        6. Return the initial EvaluationJob.
        """

        pdf_path = Path(pdf_path)

        # Validate source PDF.
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if not pdf_path.is_file():
            raise ValueError(
                f"PDF path is not a file: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Only PDF files are supported: {pdf_path}"
            )

        # Validate and get page count.
        total_pages = self.pdf_service.validate_page_count(
            pdf_path,
            settings.max_pdf_pages,
        )

        # Generate unique job ID.
        job_id = uuid4().hex

        # Each evaluation job gets its own page directory.
        pages_dir = (
            Path(settings.pages_dir)
            / job_id
        )

        # Render PDF pages.
        page_files = self.page_service.render_pages(
            pdf_path=pdf_path,
            output_dir=pages_dir,
        )

        # Create page states.
        pages: list[PageState] = []

        for page_number, page_file in enumerate(
            page_files,
            start=1,
        ):
            pages.append(
                PageState(
                    job_id=job_id,
                    page_number=page_number,
                    status=PageStatus.PENDING,

                    # PageService returns Path objects.
                    # PageState expects image_path as a string.
                    image_path=str(page_file),
                )
            )

        # Return initial evaluation job.
        return EvaluationJob(
            job_id=job_id,
            status=JobStatus.CREATED,
            student_id=student_id,
            assessment_id=assessment_id,
            subject=subject,
            class_level=class_level,
            source_pdf=str(pdf_path),
            total_pages=total_pages,
            processed_pages=0,
            pages=pages,
        )
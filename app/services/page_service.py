
from pathlib import Path
from typing import List

import pymupdf


class PageService:
    """
    Converts PDF pages into image files.

    The service is intentionally independent from:
    - LangGraph
    - FastAPI
    - Master Agent
    - Evaluation Agent
    - Database
    - LLM provider

    This keeps the Copy Checker easy to integrate into TCO later.
    """

    SUPPORTED_IMAGE_FORMAT = "png"

    def __init__(self, pages_dir: str = "data/pages"):
        self.pages_dir = Path(pages_dir)
        self.pages_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # INTERNAL RENDERING METHOD
    # ========================================================

    def _render_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        dpi: int = 300,
    ) -> List[Path]:
        """
        Internal PDF rendering implementation.

        All public rendering methods use this method so that
        there is only one actual rendering implementation.
        """

        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        # ----------------------------------------------------
        # Validate input PDF
        # ----------------------------------------------------

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if not pdf_path.is_file():
            raise ValueError(
                f"PDF path is not a file: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected PDF file, got: {pdf_path.suffix}"
            )

        if dpi <= 0:
            raise ValueError(
                "DPI must be greater than zero."
            )

        # ----------------------------------------------------
        # Create output directory
        # ----------------------------------------------------

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        rendered_pages: List[Path] = []

        # ----------------------------------------------------
        # Open PDF
        # ----------------------------------------------------

        try:
            document = pymupdf.open(pdf_path)

        except Exception as exc:
            raise RuntimeError(
                f"Could not open PDF: {pdf_path}"
            ) from exc

        # ----------------------------------------------------
        # Render pages
        # ----------------------------------------------------

        try:
            zoom = dpi / 72.0

            matrix = pymupdf.Matrix(
                zoom,
                zoom,
            )

            for page_index in range(len(document)):

                page = document.load_page(
                    page_index
                )

                # 300 DPI gives better handwriting quality.
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    alpha=False,
                )

                page_number = page_index + 1

                output_path = (
                    output_dir
                    / f"page_{page_number:03d}.png"
                )

                pixmap.save(
                    str(output_path)
                )

                rendered_pages.append(
                    output_path
                )

        finally:
            document.close()

        return rendered_pages

    # ========================================================
    # EXISTING PUBLIC METHOD
    # ========================================================

    def render_pdf_to_pages(
        self,
        pdf_path: str | Path,
        job_id: str,
        dpi: int = 300,
    ) -> List[Path]:
        """
        Render every PDF page into PNG images.

        Output:
            data/pages/<job_id>/page_001.png
            data/pages/<job_id>/page_002.png
            ...
        """

        output_dir = (
            self.pages_dir
            / job_id
        )

        return self._render_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            dpi=dpi,
        )

    # ========================================================
    # COMPATIBILITY METHOD
    # ========================================================

    def render_pages(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        dpi: int = 300,
    ) -> List[Path]:
        """
        Render PDF pages directly into the supplied output
        directory.

        This method is used by:
            - JobManager
            - Answer Sheet Agent
            - existing page-rendering tests

        Output:
            <output_dir>/page_001.png
            <output_dir>/page_002.png
            ...
        """

        return self._render_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            dpi=dpi,
        )

    # ========================================================
    # PAGE COUNT
    # ========================================================

    def get_page_count(
        self,
        pdf_path: str | Path,
    ) -> int:
        """
        Return the number of pages in a PDF.
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if not pdf_path.is_file():
            raise ValueError(
                f"PDF path is not a file: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected PDF file, got: {pdf_path.suffix}"
            )

        try:
            document = pymupdf.open(pdf_path)

        except Exception as exc:
            raise RuntimeError(
                f"Could not open PDF: {pdf_path}"
            ) from exc

        try:
            return len(document)

        finally:
            document.close()
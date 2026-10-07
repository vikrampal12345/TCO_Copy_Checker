from pathlib import Path

import fitz


class PDFService:
    def get_page_count(self, pdf_path: str | Path) -> int:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        with fitz.open(pdf_path) as document:
            return len(document)

    def validate_page_count(
        self,
        pdf_path: str | Path,
        max_pages: int,
    ) -> int:
        page_count = self.get_page_count(pdf_path)

        if page_count == 0:
            raise ValueError("PDF contains no pages.")

        if page_count > max_pages:
            raise ValueError(
                f"PDF has {page_count} pages. "
                f"Maximum supported pages: {max_pages}."
            )

        return page_count
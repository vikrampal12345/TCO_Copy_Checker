from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pymupdf
from PIL import Image


class InMemoryPageProcessor:
    """
    Render PDF pages directly into RAM.

    Design goals:
    - Keep the original PDF on disk.
    - Render only one page at a time.
    - Return a PIL Image stored in memory.
    - Never write rendered pages to disk.
    - Optionally resize very large pages to control RAM usage.
    """

    def __init__(
        self,
        dpi: int = 200,
        max_dimension: int = 1800,
    ) -> None:
        if dpi <= 0:
            raise ValueError("dpi must be greater than 0.")

        if max_dimension <= 0:
            raise ValueError(
                "max_dimension must be greater than 0."
            )

        self.dpi = dpi
        self.max_dimension = max_dimension

    # ============================================================
    # PDF VALIDATION
    # ============================================================

    @staticmethod
    def _validate_pdf_path(
        pdf_path: str | Path,
    ) -> Path:
        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"PDF path is not a file: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, got: {path.suffix}"
            )

        return path

    # ============================================================
    # PAGE COUNT
    # ============================================================

    def get_page_count(
        self,
        pdf_path: str | Path,
    ) -> int:
        """
        Return the total number of pages without rendering them.
        """

        path = self._validate_pdf_path(pdf_path)

        document = pymupdf.open(path)

        try:
            return len(document)
        finally:
            document.close()

    # ============================================================
    # RESIZE
    # ============================================================

    def _resize_if_needed(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Resize the image only when either dimension exceeds
        max_dimension.

        Aspect ratio is preserved.
        """

        width, height = image.size

        largest_dimension = max(
            width,
            height,
        )

        if largest_dimension <= self.max_dimension:
            return image

        scale = (
            self.max_dimension
            / largest_dimension
        )

        new_width = max(
            1,
            int(width * scale),
        )

        new_height = max(
            1,
            int(height * scale),
        )

        return image.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.LANCZOS,
        )

    # ============================================================
    # RENDER ONE PAGE
    # ============================================================

    def render_page(
        self,
        pdf_path: str | Path,
        page_number: int,
    ) -> Image.Image:
        """
        Render exactly one PDF page into RAM.

        page_number is 1-based.
        """

        path = self._validate_pdf_path(pdf_path)

        if page_number < 1:
            raise ValueError(
                "page_number must be >= 1."
            )

        document = pymupdf.open(path)

        try:
            total_pages = len(document)

            if page_number > total_pages:
                raise IndexError(
                    f"Page {page_number} is out of range. "
                    f"PDF contains {total_pages} page(s)."
                )

            page = document[
                page_number - 1
            ]

            zoom = self.dpi / 72.0

            matrix = pymupdf.Matrix(
                zoom,
                zoom,
            )

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                (
                    pixmap.width,
                    pixmap.height,
                ),
                pixmap.samples,
            )

            resized = self._resize_if_needed(
                image
            )

            if resized is not image:
                image.close()
                image = resized

            return image

        finally:
            document.close()

    # ============================================================
    # ITERATE PAGES ONE AT A TIME
    # ============================================================

    def iter_pages(
        self,
        pdf_path: str | Path,
    ) -> Iterator[
        tuple[int, Image.Image]
    ]:
        """
        Yield one page image at a time.

        IMPORTANT:
        The caller owns the returned PIL image and should call
        image.close() as soon as processing for that page ends.
        """

        path = self._validate_pdf_path(pdf_path)

        document = pymupdf.open(path)

        try:
            total_pages = len(document)

            for page_index in range(
                total_pages
            ):
                page_number = page_index + 1

                page = document[
                    page_index
                ]

                zoom = self.dpi / 72.0

                matrix = pymupdf.Matrix(
                    zoom,
                    zoom,
                )

                pixmap = page.get_pixmap(
                    matrix=matrix,
                    alpha=False,
                )

                image = Image.frombytes(
                    "RGB",
                    (
                        pixmap.width,
                        pixmap.height,
                    ),
                    pixmap.samples,
                )

                resized = self._resize_if_needed(
                    image
                )

                if resized is not image:
                    image.close()
                    image = resized

                try:
                    yield (
                        page_number,
                        image,
                    )
                finally:
                    # Ensure each yielded page image is released
                    # before moving to the next page.
                    image.close()

        finally:
            document.close()


__all__ = [
    "InMemoryPageProcessor",
]
from __future__ import annotations

from PIL import Image

from app.schemas.extraction import PageExtractionResult


class InMemorySubmissionExtractionResult:
    """
    Holds one complete extraction attempt.

    Page images remain in RAM until the verification stage finishes.
    """

    def __init__(
        self,
        page_results: list[PageExtractionResult],
        page_images: list[tuple[int, Image.Image]],
    ) -> None:
        self.page_results = page_results
        self.page_images = page_images

    def close(self) -> None:
        for _, image in self.page_images:
            try:
                image.close()
            except Exception:
                pass

        self.page_images.clear()

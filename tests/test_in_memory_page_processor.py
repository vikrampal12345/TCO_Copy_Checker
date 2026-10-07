from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.services.in_memory_page_processor import (
    InMemoryPageProcessor,
)


PDF_PATH = Path(
    "data/test_copies/Student_23.pdf"
)


def test_get_page_count() -> None:
    service = InMemoryPageProcessor(
        dpi=150,
        max_dimension=1800,
    )

    total_pages = service.get_page_count(
        PDF_PATH
    )

    assert total_pages == 4


def test_render_single_page_to_ram() -> None:
    service = InMemoryPageProcessor(
        dpi=150,
        max_dimension=1800,
    )

    image = service.render_page(
        pdf_path=PDF_PATH,
        page_number=1,
    )

    try:
        assert isinstance(
            image,
            Image.Image,
        )

        assert image.mode == "RGB"

        width, height = image.size

        assert width > 0
        assert height > 0

        assert max(
            width,
            height,
        ) <= 1800

    finally:
        image.close()


def test_iter_pages_one_at_a_time() -> None:
    service = InMemoryPageProcessor(
        dpi=150,
        max_dimension=1800,
    )

    processed_pages = []

    for page_number, image in service.iter_pages(
        PDF_PATH
    ):
        assert isinstance(
            image,
            Image.Image,
        )

        assert image.mode == "RGB"

        width, height = image.size

        assert width > 0
        assert height > 0

        processed_pages.append(
            page_number
        )

    assert processed_pages == [
        1,
        2,
        3,
        4,
    ]
from PIL import Image

from app.schemas.extraction import PageExtractionResult
from app.schemas.in_memory_submission import (
    InMemorySubmissionExtractionResult,
)


def test_in_memory_submission_holds_pages_and_images():
    page_result = PageExtractionResult(
        page_no=1,
        questions=[],
        unresolved_items=[],
        missing_questions=[],
        review_required=False,
        warnings=[],
        processed_tiles=8,
        recovery_attempted=False,
        recovery_tiles=0,
    )

    image = Image.new("RGB", (100, 100), "white")

    result = InMemorySubmissionExtractionResult(
        page_results=[page_result],
        page_images=[(1, image)],
    )

    assert len(result.page_results) == 1
    assert len(result.page_images) == 1
    assert result.page_images[0][0] == 1
    assert result.page_images[0][1].size == (100, 100)

    result.close()

    assert result.page_images == []


def test_close_is_safe_when_called_multiple_times():
    image = Image.new("RGB", (50, 50), "white")

    result = InMemorySubmissionExtractionResult(
        page_results=[],
        page_images=[(1, image)],
    )

    result.close()
    result.close()

    assert result.page_images == []

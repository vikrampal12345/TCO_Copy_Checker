from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from app.services.in_memory_answer_extraction_service import (
    InMemoryAnswerExtractionService,
)


class FakeVisionService:
    def __init__(self) -> None:
        self.calls = 0

    def analyze_pil_image(
        self,
        image: Image.Image,
        prompt: str,
        **kwargs,
    ):
        self.calls += 1

        return {
            "raw": (
                '{"questions": ['
                '{"question_no": 1, '
                '"answer": "A", '
                '"confidence": 0.95}'
                ']}'
            ),
            "parsed": {
                "questions": [
                    {
                        "question_no": 1,
                        "answer": "A",
                        "confidence": 0.95,
                    }
                ]
            },
            "model": "fake-model",
            "provider": "fake-provider",
            "image_path": None,
            "image_source": "memory",
        }


def test_create_tiles_stays_in_memory() -> None:
    service = (
        InMemoryAnswerExtractionService(
            vision_service=FakeVisionService()
        )
    )

    page = Image.new(
        "RGB",
        (
            1200,
            1600,
        ),
        "white",
    )

    try:
        tiles = (
            service.create_tiles_in_memory(
                page
            )
        )

        assert len(tiles) == 8

        for tile_name, tile_image in tiles:
            assert tile_name.startswith(
                "tile_"
            )

            assert isinstance(
                tile_image,
                Image.Image,
            )

            assert tile_image.mode == "RGB"

    finally:
        for _, tile_image in tiles:
            tile_image.close()

        page.close()


def test_extract_page_in_memory() -> None:
    fake_vision = FakeVisionService()

    service = (
        InMemoryAnswerExtractionService(
            vision_service=fake_vision
        )
    )

    page = Image.new(
        "RGB",
        (
            1200,
            1600,
        ),
        "white",
    )

    try:
        result = (
            service.extract_page_in_memory(
                image=page,
                page_no=1,
                expected_question_numbers=[],
                enable_recovery=False,
            )
        )

        assert result.page_no == 1

        assert result.processed_tiles == 8

        assert len(result.questions) == 1

        assert (
            result.questions[0]
            .question_no
            == 1
        )

        assert (
            result.questions[0]
            .answer
            == "A"
        )

        assert fake_vision.calls == 8

    finally:
        page.close()
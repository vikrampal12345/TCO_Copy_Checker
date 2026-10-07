from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from app.services.in_memory_vision_service import (
    InMemoryVisionService,
)


class FakeCompletions:
    def create(
        self,
        **kwargs,
    ):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"questions": ['
                            '{"question_no": 1,'
                            '"answer": "A",'
                            '"confidence": 0.95}'
                            ']}'
                        )
                    )
                )
            ]
        )


class FakeChat:
    def __init__(self) -> None:
        self.completions = (
            FakeCompletions()
        )


class FakeClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


class FakeProvider:
    def __init__(self) -> None:
        self.client = FakeClient()

    def get_client(self):
        return self.client

    def get_vision_model(self) -> str:
        return "fake-vision-model"

    def get_provider_name(self) -> str:
        return "fake-provider"

    def health_check(self):
        return {
            "healthy": True
        }


def test_analyze_pil_image_without_disk_file() -> None:
    provider = FakeProvider()

    service = InMemoryVisionService(
        provider=provider
    )

    image = Image.new(
        "RGB",
        (
            1200,
            1600,
        ),
        "white",
    )

    try:
        result = service.analyze_pil_image(
            image=image,
            prompt=(
                "Return valid JSON containing "
                "the detected questions."
            ),
        )

        assert result["image_path"] is None

        assert result["image_source"] == (
            "memory"
        )

        assert result["model"] == (
            "fake-vision-model"
        )

        assert result["provider"] == (
            "fake-provider"
        )

        assert result["parsed"] == {
            "questions": [
                {
                    "question_no": 1,
                    "answer": "A",
                    "confidence": 0.95,
                }
            ]
        }

    finally:
        image.close()
from __future__ import annotations

import base64
from typing import Any

from PIL import Image

from app.services.vision_service import VisionService


class InMemoryVisionService(VisionService):
    """
    Vision inference service for PIL images already loaded in RAM.

    This class intentionally reuses the existing VisionService's:
    - model provider
    - OpenAI-compatible client
    - vision model
    - image normalization
    - JSON parsing
    - provider information

    It only adds an inference entry point that accepts a PIL Image
    instead of requiring an image file path.
    """

    def analyze_pil_image(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_dimension: int = 2048,
        jpeg_quality: int = 95,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Send a PIL image directly to the configured vision model.

        The image never needs to be written to disk.

        Parameters
        ----------
        image:
            PIL Image already present in memory.

        prompt:
            Vision extraction/verification prompt.

        max_dimension:
            Maximum image dimension before transport normalization.

        jpeg_quality:
            JPEG quality used for transport.

        temperature:
            Model sampling temperature.
        """

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "image must be a PIL.Image.Image instance."
            )

        if not prompt or not prompt.strip():
            raise ValueError(
                "Vision prompt must not be empty."
            )

        # --------------------------------------------------------
        # Normalize PIL image -> JPEG bytes
        # --------------------------------------------------------

        image_bytes = self.normalize_image(
            image=image,
            max_dimension=max_dimension,
            jpeg_quality=jpeg_quality,
        )

        # --------------------------------------------------------
        # JPEG bytes -> data URL
        # --------------------------------------------------------

        encoded = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        image_url = (
            "data:image/jpeg;base64,"
            + encoded
        )

        # --------------------------------------------------------
        # Vision model request
        # --------------------------------------------------------

        try:
            response = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise "
                                "document vision system. "
                                "Follow the user's output "
                                "format exactly."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                    },
                                },
                            ],
                        },
                    ],
                )
            )

        except Exception as exc:
            raise RuntimeError(
                "Vision model request failed for "
                "in-memory image.\n"
                f"Provider: "
                f"{self.get_provider_name()}\n"
                f"Model: {self.get_model_name()}\n"
                f"Error: {exc}"
            ) from exc

        # --------------------------------------------------------
        # Validate response
        # --------------------------------------------------------

        if not response.choices:
            raise RuntimeError(
                "Vision model returned no choices."
            )

        message = (
            response
            .choices[0]
            .message
        )

        raw = message.content

        if raw is None:
            raise RuntimeError(
                "Vision model returned no content."
            )

        # --------------------------------------------------------
        # Parse structured JSON
        # --------------------------------------------------------

        parsed = self.parse_json_response(
            raw
        )

        return {
            "raw": raw,
            "parsed": parsed,
            "model": self.model,
            "provider": (
                self.get_provider_name()
            ),
            "image_path": None,
            "image_source": "memory",
        }


__all__ = [
    "InMemoryVisionService",
]


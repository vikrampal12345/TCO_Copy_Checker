from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.model_provider import (
    get_model_provider,
)


class VisionService:
    """
    Model-independent vision capability.

    Responsibilities:
        - validate images
        - normalize images
        - convert images to API-compatible format
        - send vision requests
        - parse structured JSON responses

    NOT responsible for:
        - grading
        - marks
        - evaluation
        - ResultStore
        - TCO Master Agent
        - LangGraph orchestration

    The actual model is supplied by ModelProvider.
    """

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    def __init__(
        self,
        provider=None,
    ) -> None:

        self.provider = (
            provider
            or get_model_provider()
        )

        self.client = (
            self.provider.get_client()
        )

        self.model = (
            self.provider.get_vision_model()
        )

    # ========================================================
    # PROVIDER INFO
    # ========================================================

    def get_model_name(self) -> str:
        return self.model

    def get_provider_name(self) -> str:
        return (
            self.provider
            .get_provider_name()
        )

    # ========================================================
    # IMAGE VALIDATION
    # ========================================================

    def validate_image(
        self,
        image_path: str | Path,
    ) -> Path:
        """
        Validate that the supplied path is an existing,
        supported image file.
        """

        path = Path(
            image_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Image not found:\n{path}"
            )

        if not path.is_file():

            raise ValueError(
                f"Image path is not a file:\n{path}"
            )

        suffix = (
            path.suffix.lower()
        )

        if suffix not in (
            self.SUPPORTED_EXTENSIONS
        ):

            raise ValueError(
                "Unsupported image format: "
                f"{suffix}\n"
                "Supported formats: "
                f"{sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        return path

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    def load_image(
        self,
        image_path: str | Path,
    ) -> Image.Image:
        """
        Load and fully decode the image.
        """

        path = self.validate_image(
            image_path
        )

        try:

            image = Image.open(
                path
            )

            image.load()

        except Exception as exc:

            raise ValueError(
                f"Could not decode image:\n"
                f"{path}\n"
                f"Reason: {exc}"
            ) from exc

        return image.convert(
            "RGB"
        )

    # ========================================================
    # NORMALIZE IMAGE
    # ========================================================

    def normalize_image(
        self,
        image: Image.Image,
        max_dimension: int = 2048,
        jpeg_quality: int = 95,
    ) -> bytes:
        """
        Normalize the image for vision inference.

        Everything is converted to RGB JPEG.

        This gives us a stable transport format regardless
        of whether the original image was PNG/JPEG/WebP.
        """

        image = image.convert(
            "RGB"
        )

        width, height = image.size

        largest_dimension = max(
            width,
            height,
        )

        if largest_dimension > max_dimension:

            scale = (
                max_dimension
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

            image = image.resize(
                (
                    new_width,
                    new_height,
                ),
                Image.Resampling.LANCZOS,
            )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
        )

        image_bytes = (
            buffer.getvalue()
        )

        if not image_bytes:

            raise ValueError(
                "Image normalization produced "
                "empty bytes."
            )

        return image_bytes

    # ========================================================
    # DATA URL
    # ========================================================

    def image_to_data_url(
        self,
        image_path: str | Path,
        max_dimension: int = 2048,
        jpeg_quality: int = 95,
    ) -> str:
        """
        Convert image into a stable JPEG data URL.
        """

        image = self.load_image(
            image_path
        )

        image_bytes = (
            self.normalize_image(
                image=image,
                max_dimension=max_dimension,
                jpeg_quality=jpeg_quality,
            )
        )

        encoded = (
            base64
            .b64encode(image_bytes)
            .decode("utf-8")
        )

        return (
            "data:image/jpeg;base64,"
            + encoded
        )

    # ========================================================
    # JSON PARSER
    # ========================================================

    @staticmethod
    def parse_json_response(
        raw: str,
    ) -> dict:
        """
        Parse model JSON and normalize list responses.

        Supported:

        {
            "questions": [...]
        }

        OR:

        [...]
        """

        if raw is None:

            raise ValueError(
                "Vision model returned no content."
            )

        cleaned = (
            raw.strip()
        )

        if not cleaned:

            raise ValueError(
                "Vision model returned empty content."
            )

        # ----------------------------------------------------
        # Remove markdown code fences
        # ----------------------------------------------------

        if cleaned.startswith(
            "```"
        ):

            lines = (
                cleaned
                .splitlines()
            )

            if (
                lines
                and lines[0]
                .strip()
                .startswith("```")
            ):

                lines = lines[1:]

            if (
                lines
                and lines[-1]
                .strip()
                == "```"
            ):

                lines = lines[:-1]

            cleaned = (
                "\n".join(
                    lines
                )
                .strip()
            )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            data = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Vision model returned invalid JSON.\n\n"
                f"RAW OUTPUT:\n{raw}\n\n"
                f"JSON ERROR:\n{exc}"
            ) from exc

        # ----------------------------------------------------
        # Normalize list
        # ----------------------------------------------------

        if isinstance(
            data,
            list,
        ):

            data = {
                "questions": data
            }

        if not isinstance(
            data,
            dict,
        ):

            raise ValueError(
                "Vision response must be "
                "a JSON object or list."
            )

        return data

    # ========================================================
    # ANALYZE IMAGE
    # ========================================================

    def analyze_image(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        max_dimension: int = 2048,
        jpeg_quality: int = 95,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Send an image to the configured vision model.

        Returns:

        {
            "raw": "...",
            "parsed": {...},
            "model": "...",
            "provider": "...",
            "image_path": "..."
        }
        """

        path = self.validate_image(
            image_path
        )

        image_url = (
            self.image_to_data_url(
                path,
                max_dimension=max_dimension,
                jpeg_quality=jpeg_quality,
            )
        )

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
                "Vision model request failed.\n"
                f"Provider: "
                f"{self.get_provider_name()}\n"
                f"Model: {self.get_model_name()}\n"
                f"Image: {path}\n"
                f"Error: {exc}"
            ) from exc

        # ----------------------------------------------------
        # Validate response
        # ----------------------------------------------------

        if not response.choices:

            raise RuntimeError(
                "Vision model returned no choices."
            )

        message = (
            response
            .choices[0]
            .message
        )

        raw = (
            message.content
        )

        # ----------------------------------------------------
        # Parse
        # ----------------------------------------------------

        parsed = (
            self.parse_json_response(
                raw
            )
        )

        return {
            "raw": raw,
            "parsed": parsed,
            "model": self.model,
            "provider": (
                self.get_provider_name()
            ),
            "image_path": str(path),
        }

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self) -> dict[str, Any]:
        """
        Return provider/model health information.
        """

        return (
            self.provider.health_check()
        )
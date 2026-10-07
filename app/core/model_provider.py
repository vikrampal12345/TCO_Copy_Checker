from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from openai import OpenAI

from app.core.config import settings


@dataclass(frozen=True)
class ModelConfig:
    """
    Central model configuration.

    Application services should NEVER read .env directly.
    They should get model information through ModelProvider.
    """

    provider_name: str
    base_url: str
    api_key: str
    vision_model: str
    extraction_verifier_model: str
    text_model: str


class ModelProvider:
    """
    Central abstraction for model access.

    Current implementation:
        LM Studio + OpenAI-compatible API

    Future examples:
        - OpenAI
        - Gemini
        - another OpenAI-compatible server
        - self-hosted vision model
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:

        self.config = config

        if not config.base_url:
            raise ValueError(
                "MODEL_BASE_URL is missing."
            )

        if not config.vision_model:
            raise ValueError(
                "VISION_MODEL_NAME is missing."
            )

        if not config.extraction_verifier_model:
            raise ValueError(
                "EXTRACTION_VERIFIER_MODEL_NAME is missing."
            )

        if not config.text_model:
            raise ValueError(
                "TEXT_MODEL_NAME is missing."
            )

        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    # ========================================================
    # CLIENT
    # ========================================================

    def get_client(self) -> OpenAI:
        return self._client

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def get_provider_name(self) -> str:
        return self.config.provider_name

    def get_vision_model(self) -> str:
        return self.config.vision_model

    def get_extraction_verifier_model(self) -> str:
        return self.config.extraction_verifier_model

    def get_text_model(self) -> str:
        return self.config.text_model

    def get_base_url(self) -> str:
        return self.config.base_url

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "vision": True,
            "text": True,
            "structured_output": True,
            "image_input": True,
            "multi_image": True,
            "provider": self.config.provider_name,
        }

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(self) -> dict[str, Any]:

        try:

            models = self._client.models.list()

            model_ids = [
                getattr(
                    model,
                    "id",
                    None,
                )
                for model in models.data
            ]

            return {
                "status": "ok",
                "provider": self.get_provider_name(),
                "vision_model": self.get_vision_model(),
                "extraction_verifier_model": (
                    self.get_extraction_verifier_model()
                ),
                "text_model": self.get_text_model(),
                "available_models": model_ids,
            }

        except Exception as exc:

            return {
                "status": "error",
                "provider": self.get_provider_name(),
                "vision_model": self.get_vision_model(),
                "extraction_verifier_model": (
                    self.get_extraction_verifier_model()
                ),
                "text_model": self.get_text_model(),
                "error": str(exc),
            }


# ============================================================
# FACTORY
# ============================================================

def _build_model_config() -> ModelConfig:
    return ModelConfig(
        provider_name=settings.model_provider,
        base_url=settings.model_base_url,
        api_key=settings.model_api_key,
        vision_model=settings.vision_model_name,
        extraction_verifier_model=(
            settings.extraction_verifier_model_name
        ),
        text_model=settings.text_model_name,
    )


@lru_cache(maxsize=1)
def get_model_provider() -> ModelProvider:

    config = _build_model_config()

    return ModelProvider(
        config=config
    )

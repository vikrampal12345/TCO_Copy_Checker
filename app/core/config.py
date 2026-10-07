from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TCO Copy Checker"
    app_version: str = "0.1.0"

    upload_dir: str = "data/uploads"
    pages_dir: str = "data/pages"
    output_dir: str = "data/outputs"

    max_pdf_pages: int = 100

    # Model provider configuration
    model_provider: str = "lm_studio"

    vision_model_name: str = "qwen/qwen2.5-vl-3b"
    extraction_verifier_model_name: str = (
        "qwen.qwen2.5-vl-3b-instruct"
    )
    text_model_name: str = "qwen/qwen3-4b-2507"

    model_base_url: str = "http://127.0.0.1:1234/v1"
    model_api_key: str = "lm-studio"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()

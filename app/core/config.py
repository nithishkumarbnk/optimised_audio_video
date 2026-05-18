"""
Core configuration module for the AI Proctoring Backend.

All settings are loaded from environment variables (with documented defaults)
using pydantic-settings. A `.env` file is also supported for local development.
"""

from __future__ import annotations

import functools
from typing import List

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration loaded from environment variables.

    Required fields (no default):
        OPENROUTER_API_KEY: API key for the OpenRouter LLM gateway.
            Raises a ValidationError at startup if absent or empty.

    Optional fields (with defaults):
        LOG_LEVEL:         Root logger level.            Default: "INFO"
        MAX_AUDIO_SIZE_MB: Maximum audio upload size.   Default: 20
        YOLO_CONFIDENCE:   YOLO detection threshold.    Default: 0.4
        ENABLE_GPU:        Use CUDA for PyTorch inference. Default: False
        WORKER_POOL_SIZE:  Async worker pool size.      Default: 4
        CORS_ORIGINS:      Allowed CORS origins.        Default: ["*"]
    """
    # --- Required ---
    OPENROUTER_API_KEY: str

    # --- Optional with defaults ---
    LOG_LEVEL: str = "INFO"
    MAX_AUDIO_SIZE_MB: int = 20
    YOLO_CONFIDENCE: float = 0.4
    ENABLE_GPU: bool = False
    WORKER_POOL_SIZE: int = 4
    CORS_ORIGINS: List[str] = ["*"]

    # --- HuggingFace cache isolation ---
    HF_MODULES_CACHE: str | None = None

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }



@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the application Config singleton.

    The result is cached after the first call so that environment variables
    and the `.env` file are parsed exactly once per process lifetime.

    Returns:
        Config: The populated configuration object.

    Raises:
        pydantic.ValidationError: If OPENROUTER_API_KEY is missing or any
            field value cannot be coerced to its declared type.
    """
    return Config()

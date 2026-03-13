"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    """Configuration for the embedding model."""

    name: str
    max_model_length: int
    dtype: str
    trust_remote_code: bool


DEFAULT_MODEL_SETTINGS = ModelSettings(
    name="Qwen/Qwen3-Embedding-8B",
    max_model_length=8192,
    dtype="half",
    trust_remote_code=True,
)

"""Typed contracts for embedding requests and responses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingRequest:
    """Normalized embedding request payload."""

    model: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding payload produced by the model."""

    vector: list[float]
    prompt_token_count: int


@dataclass(frozen=True)
class ErrorPayload:
    """OpenAI-compatible error payload."""

    message: str
    error_type: str
    param: str | None = None
    code: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the error payload for API responses.

        Returns:
            A dictionary using the OpenAI error schema.
        """

        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }

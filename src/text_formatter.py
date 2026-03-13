"""Input formatting for the embedding model."""

from __future__ import annotations

from src.contracts import EmbeddingRequest


class EmbeddingTextFormatter:
    """Prepare model-ready text inputs."""

    def format(self, request: EmbeddingRequest) -> tuple[str, ...]:
        """Convert a request into the final text prompts.

        Args:
            request: Normalized embedding request.

        Returns:
            Text inputs formatted for the embedding model.
        """

        return request.inputs

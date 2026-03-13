"""Response factories for embedding APIs."""

from __future__ import annotations

from time import time
from typing import Sequence
from uuid import uuid4

from src.contracts import EmbeddingResult


type EmbeddingResponseDataPayload = dict[str, int | str | list[float]]
type EmbeddingUsagePayload = dict[str, int]
type EmbeddingResponsePayload = dict[
    str, str | int | EmbeddingUsagePayload | list[EmbeddingResponseDataPayload]
]


class OpenAIEmbeddingResponseFactory:
    """Build OpenAI-compatible embedding responses."""

    def create(
        self,
        model_name: str,
        embeddings: Sequence[EmbeddingResult],
    ) -> EmbeddingResponsePayload:
        """Create a successful embeddings response.

        Args:
            model_name: Model identifier returned to clients.
            embeddings: Embedding results in request order.

        Returns:
            A dictionary following the OpenAI embeddings response schema.
        """

        prompt_token_count = sum(
            embedding.prompt_token_count for embedding in embeddings
        )
        return {
            "id": f"embd-{uuid4().hex}",
            "object": "list",
            "created": int(time()),
            "model": model_name,
            "data": self._create_response_data(embeddings),
            "usage": {
                "prompt_tokens": prompt_token_count,
                "total_tokens": prompt_token_count,
            },
        }

    def _create_response_data(
        self,
        embeddings: Sequence[EmbeddingResult],
    ) -> list[EmbeddingResponseDataPayload]:
        """Create the response data entries for each embedding."""

        return [
            {
                "index": index,
                "object": "embedding",
                "embedding": embedding.vector,
            }
            for index, embedding in enumerate(embeddings)
        ]

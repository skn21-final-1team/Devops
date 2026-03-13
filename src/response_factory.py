"""Response factories for embedding APIs."""

from __future__ import annotations

from time import time
from typing import Sequence
from uuid import uuid4

from vllm.entrypoints.openai.protocol import (
    EmbeddingResponse,
    EmbeddingResponseData,
    UsageInfo,
)

from src.embedding_service import EmbeddingResult


class OpenAIEmbeddingResponseFactory:
    """Build OpenAI-compatible embedding responses with vLLM protocol models."""

    def create(
        self,
        model_name: str,
        embeddings: Sequence[EmbeddingResult],
    ) -> dict[str, object]:
        """Create a successful embeddings response.

        Args:
            model_name: Model identifier returned to clients.
            embeddings: Embedding results in request order.

        Returns:
            A dictionary following the OpenAI embeddings response schema.
        """

        response = EmbeddingResponse(
            id=f"embd-{uuid4().hex}",
            object="list",
            created=int(time()),
            model=model_name,
            data=[
                EmbeddingResponseData(
                    index=index,
                    object="embedding",
                    embedding=embedding.vector,
                )
                for index, embedding in enumerate(embeddings)
            ],
            usage=UsageInfo(
                prompt_tokens=sum(
                    embedding.prompt_token_count for embedding in embeddings
                ),
                total_tokens=sum(
                    embedding.prompt_token_count for embedding in embeddings
                ),
            ),
        )
        return self._serialize(response)

    def _serialize(self, response: EmbeddingResponse) -> dict[str, object]:
        """Serialize a protocol response across supported Pydantic versions.

        Args:
            response: vLLM protocol response model.

        Returns:
            A plain dictionary that matches the OpenAI embeddings schema.
        """

        if hasattr(response, "model_dump"):
            return response.model_dump()

        return response.dict()

"""Embedding generation through vLLM."""

from __future__ import annotations

from typing import Sequence

from vllm import LLM

from src.config import ModelSettings
from src.contracts import EmbeddingResult


class VllmEmbeddingService:
    """Generate embeddings using a vLLM pooling runner."""

    def __init__(self, model_settings: ModelSettings) -> None:
        """Initialize the embedding model.

        Args:
            model_settings: Model configuration to load into vLLM.
        """

        self._llm = LLM(
            model=model_settings.name,
            runner="pooling",
            dtype=model_settings.dtype,
            trust_remote_code=model_settings.trust_remote_code,
            max_model_len=model_settings.max_model_length,
        )

    def embed(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        """Generate embeddings for the provided texts.

        Args:
            texts: Model-ready text inputs.

        Returns:
            Embedding results in request order.
        """

        outputs = self._llm.embed(list(texts))
        return [
            EmbeddingResult(
                vector=list(output.outputs.embedding),
                prompt_token_count=self._count_prompt_tokens(output.prompt_token_ids),
            )
            for output in outputs
        ]

    def _count_prompt_tokens(self, prompt_token_ids: Sequence[int] | None) -> int:
        """Return the number of prompt tokens reported by vLLM.

        Args:
            prompt_token_ids: Token identifiers returned by vLLM for one prompt.

        Returns:
            The prompt token count or zero when token ids are unavailable.
        """

        if prompt_token_ids is None:
            return 0

        return len(prompt_token_ids)

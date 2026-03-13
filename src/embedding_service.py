"""Embedding generation through vLLM."""

from dataclasses import dataclass
from typing import Sequence

from vllm import LLM


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    prompt_token_count: int


class VllmEmbeddingService:
    def __init__(self) -> None:
        self._llm = LLM(
            model="Qwen/Qwen3-Embedding-8B",
            dtype="half",
            runner="pooling",
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=8192,
        )

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        outputs = self._llm.embed(texts)
        return [
            EmbeddingResult(
                vector=list(output.outputs.embedding),
                prompt_token_count=self._count_prompt_tokens(output.prompt_token_ids),
            )
            for output in outputs
        ]

    def _count_prompt_tokens(self, prompt_token_ids: Sequence[int] | None) -> int:
        if prompt_token_ids is None:
            return 0

        return len(prompt_token_ids)

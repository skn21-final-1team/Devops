"""Application bootstrap helpers."""

from __future__ import annotations

from src.application import EmbeddingApplication
from src.config import DEFAULT_MODEL_SETTINGS
from src.embedding_service import VllmEmbeddingService
from src.request_parser import EmbeddingRequestParser
from src.response_factory import OpenAIEmbeddingResponseFactory
from src.text_formatter import EmbeddingTextFormatter


def create_embedding_application() -> EmbeddingApplication:
    """Create the embedding application with production dependencies.

    Returns:
        A fully wired embedding application.
    """

    model_settings = DEFAULT_MODEL_SETTINGS
    return EmbeddingApplication(
        parser=EmbeddingRequestParser(model_settings),
        formatter=EmbeddingTextFormatter(),
        embedding_service=VllmEmbeddingService(model_settings),
        response_factory=OpenAIEmbeddingResponseFactory(),
    )

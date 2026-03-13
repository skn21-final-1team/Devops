"""Application service for embedding requests."""

from __future__ import annotations

from src.contracts import ErrorPayload
from src.embedding_service import VllmEmbeddingService
from src.request_parser import EmbeddingRequestParser
from src.response_factory import OpenAIEmbeddingResponseFactory
from src.text_formatter import EmbeddingTextFormatter


class EmbeddingApplication:
    """Coordinate parsing, formatting, embedding, and response mapping."""

    def __init__(
        self,
        parser: EmbeddingRequestParser,
        formatter: EmbeddingTextFormatter,
        embedding_service: VllmEmbeddingService,
        response_factory: OpenAIEmbeddingResponseFactory,
    ) -> None:
        """Compose the application dependencies.

        Args:
            parser: Request parser.
            formatter: Input formatter.
            embedding_service: Embedding generator.
            response_factory: Success response factory.
        """

        self._parser = parser
        self._formatter = formatter
        self._embedding_service = embedding_service
        self._response_factory = response_factory

    def handle_payload(self, payload: dict[str, object]) -> dict[str, object]:
        """Process an embedding payload.

        Args:
            payload: OpenAI-style embeddings request payload.

        Returns:
            A success or error response payload.
        """

        request = self._parser.parse_payload(payload)
        if isinstance(request, ErrorPayload):
            return request.to_dict()

        formatted_texts = self._formatter.format(request)
        embeddings = self._embedding_service.embed(formatted_texts)
        return self._response_factory.create(request.model, embeddings)

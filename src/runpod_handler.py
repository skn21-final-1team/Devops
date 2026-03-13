"""Runpod adapter for the embedding application."""

from __future__ import annotations

from src.application import EmbeddingApplication
from src.contracts import ErrorPayload


class RunpodEmbeddingHandler:
    """Adapt Runpod serverless events to the embedding application."""

    def __init__(self, application: EmbeddingApplication) -> None:
        """Store the application dependency.

        Args:
            application: Embedding application service.
        """

        self._application = application

    def handle(self, event: dict[str, object]) -> dict[str, object]:
        """Process a Runpod event.

        Args:
            event: Runpod serverless event.

        Returns:
            An OpenAI-compatible success or error payload.
        """

        raw_input = event.get("input")
        if isinstance(raw_input, dict):
            return self._application.handle_payload(raw_input)

        return ErrorPayload(
            message="Missing required object: input",
            error_type="invalid_request_error",
            param="input",
        ).to_dict()

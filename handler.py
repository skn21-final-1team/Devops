"""Runpod serverless entrypoint for OpenAI-compatible embeddings."""

from __future__ import annotations

import runpod

from src.bootstrap import create_embedding_application
from src.runpod_handler import RunpodEmbeddingHandler

application = create_embedding_application()
runpod_handler = RunpodEmbeddingHandler(application)


def handler(event: dict[str, object]) -> dict[str, object]:
    """Handle a Runpod serverless invocation.

    Args:
        event: Runpod serverless event payload.

    Returns:
        An OpenAI-compatible embeddings response or error payload.
    """

    return runpod_handler.handle(event)


runpod.serverless.start({"handler": handler})

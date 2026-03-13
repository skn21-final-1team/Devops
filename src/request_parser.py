"""Request parsing for embedding APIs."""

from __future__ import annotations

from src.config import ModelSettings
from src.contracts import EmbeddingRequest, ErrorPayload


class EmbeddingRequestParser:
    """Validate and normalize embedding requests."""

    def __init__(self, model_settings: ModelSettings) -> None:
        """Store supported model configuration.

        Args:
            model_settings: Supported model settings.
        """

        self._model_settings = model_settings

    def parse_payload(self, payload: dict[str, object]) -> EmbeddingRequest | ErrorPayload:
        """Validate a payload that follows OpenAI embeddings request fields.

        Args:
            payload: Request body payload.

        Returns:
            A normalized embedding request or a structured error.
        """

        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            return ErrorPayload(
                message="Missing required field: model",
                error_type="invalid_request_error",
                param="model",
            )

        if model != self._model_settings.name:
            return ErrorPayload(
                message=(
                    f"Unsupported model '{model}'. "
                    f"Supported model is '{self._model_settings.name}'."
                ),
                error_type="invalid_request_error",
                param="model",
            )

        normalized_inputs = self._parse_inputs(payload.get("input"))
        if isinstance(normalized_inputs, ErrorPayload):
            return normalized_inputs

        return EmbeddingRequest(
            model=model,
            inputs=normalized_inputs,
        )

    def _parse_inputs(self, raw_value: object) -> tuple[str, ...] | ErrorPayload:
        """Validate the `input` field.

        Args:
            raw_value: Raw input field value.

        Returns:
            A normalized tuple of strings or a validation error.
        """

        if isinstance(raw_value, str):
            normalized_value = raw_value.strip()
            if normalized_value:
                return (normalized_value,)
            return self._invalid_input_error()

        if not isinstance(raw_value, list) or not raw_value:
            return self._invalid_input_error()

        normalized_inputs: list[str] = []
        for item in raw_value:
            if not isinstance(item, str):
                return self._invalid_input_error()

            normalized_item = item.strip()
            if not normalized_item:
                return self._invalid_input_error()

            normalized_inputs.append(normalized_item)

        return tuple(normalized_inputs)

    def _invalid_input_error(self) -> ErrorPayload:
        """Create a shared input validation error payload.

        Returns:
            A structured invalid request error.
        """

        return ErrorPayload(
            message="Field 'input' must be a non-empty string or a non-empty list of strings.",
            error_type="invalid_request_error",
            param="input",
        )

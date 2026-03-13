from dataclasses import dataclass
from http import HTTPStatus


@dataclass(frozen=True)
class ErrorResponse:
    object: str = "error"
    message: str
    type: str
    code: int


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ErrorResponse:
    return ErrorResponse(message=message, type=err_type, code=status_code.value)

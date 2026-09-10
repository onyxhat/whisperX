from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("server")


class APIError(Exception):
    status: int = 500
    err_type: str = "api_error"

    def __init__(self, message: str, *, param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code


class AuthError(APIError):
    status, err_type = 401, "invalid_request_error"


class BadRequestError(APIError):
    status, err_type = 400, "invalid_request_error"


class PayloadTooLargeError(APIError):
    status, err_type = 413, "invalid_request_error"


class AudioDecodeError(APIError):
    status, err_type = 400, "invalid_request_error"


class QueueFullError(APIError):
    status, err_type = 429, "rate_limit_error"


class RequestTimeoutError(APIError):
    status, err_type = 504, "timeout_error"


class DiarizationError(APIError):
    status, err_type = 502, "api_error"


class NotReadyError(APIError):
    status, err_type = 503, "api_error"


class InferenceError(APIError):
    status, err_type = 500, "api_error"


def to_envelope(exc: APIError) -> dict:
    return {
        "error": {
            "message": exc.message,
            "type": exc.err_type,
            "param": exc.param,
            "code": exc.code,
        }
    }


def register_exception_handlers(app) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError):
        if exc.status >= 500:
            logger.error("APIError %s: %s", exc.status, exc.message)
        else:
            logger.warning("APIError %s: %s", exc.status, exc.message)
        return JSONResponse(status_code=exc.status, content=to_envelope(exc))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "internal error", "type": "api_error",
                               "param": None, "code": None}},
        )

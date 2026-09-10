import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import errors
from server.errors import (
    APIError, AuthError, BadRequestError, DiarizationError, InferenceError,
    NotReadyError, PayloadTooLargeError, QueueFullError, RequestTimeoutError,
    to_envelope,
)


@pytest.mark.parametrize("exc_cls,status,err_type", [
    (AuthError, 401, "invalid_request_error"),
    (BadRequestError, 400, "invalid_request_error"),
    (PayloadTooLargeError, 413, "invalid_request_error"),
    (QueueFullError, 429, "rate_limit_error"),
    (RequestTimeoutError, 504, "timeout_error"),
    (DiarizationError, 502, "api_error"),
    (NotReadyError, 503, "api_error"),
    (InferenceError, 500, "api_error"),
])
def test_subclass_status_and_type(exc_cls, status, err_type):
    e = exc_cls("boom")
    assert isinstance(e, APIError)
    assert e.status == status
    assert e.err_type == err_type


def test_envelope_shape_includes_param_and_code():
    e = BadRequestError("bad file", param="file", code="missing_file")
    assert to_envelope(e) == {
        "error": {
            "message": "bad file",
            "type": "invalid_request_error",
            "param": "file",
            "code": "missing_file",
        }
    }


def test_envelope_param_and_code_default_to_none():
    assert to_envelope(AuthError("nope"))["error"]["param"] is None
    assert to_envelope(AuthError("nope"))["error"]["code"] is None


def test_handlers_render_apierror_and_generic():
    app = FastAPI()
    errors.register_exception_handlers(app)

    @app.get("/known")
    def known():
        raise QueueFullError("queue full")

    @app.get("/boom")
    def boom():
        raise RuntimeError("unexpected")

    client = TestClient(app, raise_server_exceptions=False)

    r1 = client.get("/known")
    assert r1.status_code == 429
    assert r1.json()["error"]["type"] == "rate_limit_error"

    r2 = client.get("/boom")
    assert r2.status_code == 500
    body = r2.json()
    assert body["error"]["type"] == "api_error"
    assert "unexpected" not in body["error"]["message"]  # internal detail not leaked

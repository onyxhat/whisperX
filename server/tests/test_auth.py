import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from server.auth import require_auth
from server.errors import register_exception_handlers


def _app(api_key):
    app = FastAPI()
    register_exception_handlers(app)
    app.state.settings = type("S", (), {"api_key": api_key})()

    @app.get("/x", dependencies=[Depends(require_auth)])
    def x():
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


def test_no_key_configured_allows_all():
    assert _app(None).get("/x").status_code == 200


def test_key_configured_rejects_missing_header():
    r = _app("secret").get("/x")
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_key_configured_rejects_wrong_token():
    r = _app("secret").get("/x", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_key_configured_accepts_correct_token():
    r = _app("secret").get("/x", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200


def test_non_ascii_token_rejected_not_500():
    # Sent as raw bytes so the client does not ASCII-reject it; Starlette then
    # hands `require_auth` a latin-1-decoded str with a codepoint > 127.
    r = _app("secret").get("/x", headers={"Authorization": b"Bearer s\xe9cret"})
    assert r.status_code == 401
    body = r.json()
    assert list(body) == ["error"]
    assert body["error"]["type"] == "invalid_request_error"

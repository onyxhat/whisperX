from __future__ import annotations

import secrets

from fastapi import Request

from server.errors import AuthError


async def require_auth(request: Request) -> None:
    settings = request.app.state.settings
    expected = getattr(settings, "api_key", None)
    if not expected:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    # Compare as bytes: Starlette hands headers back latin-1-decoded, so a token
    # with a byte > 127 makes `secrets.compare_digest` on `str` raise. Encoding
    # both sides is identical for ASCII keys and never raises.
    if scheme.lower() != "bearer" or not token or not secrets.compare_digest(
        token.encode("utf-8", "surrogateescape"), expected.encode("utf-8", "surrogateescape")
    ):
        raise AuthError("missing or invalid API key", code="invalid_api_key")

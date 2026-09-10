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
    if scheme.lower() != "bearer" or not token or not secrets.compare_digest(token, expected):
        raise AuthError("missing or invalid API key", code="invalid_api_key")

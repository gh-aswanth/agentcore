"""The error envelope, described once so every route can point at it.

`app.main` wraps handled failures as `{"error": {"status": ..., "detail": ...}}`.
Without a model for that shape, Swagger shows every 401/404/409 as an untyped
blob and a generated client has nothing to deserialise into.
"""
from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    status: int
    detail: Any


class ErrorResponse(BaseModel):
    error: ErrorDetail


def _response(description: str, example: Any) -> dict:
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {
            "application/json": {
                "example": {"error": {"status": example[0], "detail": example[1]}}
            }
        },
    }


UNAUTHORIZED = _response(
    "Missing, malformed or expired bearer token.", (401, "Could not validate credentials")
)
NOT_FOUND = _response(
    "No such resource, or it belongs to another user. Ownership is part of the "
    "WHERE clause, so someone else's id is indistinguishable from a missing one — "
    "a 403 here would confirm the id exists.",
    (404, "Run not found"),
)
CONFLICT = _response("The run has already reached a terminal state.", (409, "Run already completed"))
TOO_MANY_REQUESTS = _response(
    "Too many runs already queued or running for this user.",
    (429, "You already have 10 active runs. Wait for one to finish."),
)
UNPROCESSABLE = _response(
    "Request body failed validation — for example an unknown tool name.",
    (422, [{"loc": ["body", "tools_enabled", 0], "msg": "Input should be 'calculator', ..."}]),
)

AUTHENTICATED = {401: UNAUTHORIZED}
OWNED = {401: UNAUTHORIZED, 404: NOT_FOUND}

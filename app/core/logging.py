"""structlog configuration: JSON out, context bound once and carried everywhere.

The point of binding rather than passing is that one grep reconstructs a whole
run. `run_id` is set once when the worker claims the task and every subsequent
record inside that task carries it — including records emitted by SQLAlchemy or
Celery, because the stdlib root logger is routed through the same processor
chain.

Context lives in contextvars, which propagate across `await` boundaries into the
tasks an async request spawns, and are naturally isolated between concurrent
requests.
"""
from __future__ import annotations

import logging
import sys
import time
import uuid

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.core.config import settings

# The keys the brief asks for, plus the request correlation id.
CONTEXT_KEYS = ("request_id", "user_id", "session_id", "run_id", "step_type")


def configure_logging() -> None:
    shared = [
        structlog.contextvars.merge_contextvars,   # <- the bound keys land here
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer = (
        structlog.dev.ConsoleRenderer()
        if settings.LOG_FORMAT == "console"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (uvicorn, celery, sqlalchemy) through the same chain,
    # so a run's records are one uniform stream rather than JSON interleaved
    # with someone else's format.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.LOG_LEVEL.upper())

    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "celery", "sqlalchemy.engine"):
        logging.getLogger(noisy).handlers = []
        logging.getLogger(noisy).propagate = True


def get_logger(name: str | None = None):
    return structlog.stdlib.get_logger(name)


def bind(**kwargs) -> None:
    """Bind context for the rest of this request or task. Values of None are
    dropped rather than logged as nulls."""
    bind_contextvars(**{k: v for k, v in kwargs.items() if v is not None})


def clear() -> None:
    clear_contextvars()


def current(key: str) -> str | None:
    """Read a bound value back.

    Used to carry `request_id` across the process boundary: the API knows it, the
    worker cannot discover it, so it travels as a task argument and is re-bound
    on the other side. Without that, a run's worker records are unlinkable to the
    request that submitted it.
    """
    return structlog.contextvars.get_contextvars().get(key)


class LoggingContextMiddleware:
    """Pure ASGI, deliberately — not ``@app.middleware("http")``.

    Starlette's BaseHTTPMiddleware runs the downstream app in a *separate task*,
    so a contextvar bound inside a dependency (``user_id``, in
    ``get_current_user``) is invisible by the time the middleware logs the
    request. A raw ASGI middleware awaits the app in the same task, so the
    binding is still there.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        from starlette.datastructures import MutableHeaders

        clear()
        headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}
        request_id = headers.get("x-request-id") or uuid.uuid4().hex[:12]
        bind(request_id=request_id)

        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            path = scope.get("path", "")
            if path != "/health":
                await get_logger("app.request").ainfo(
                    "http_request",
                    method=scope.get("method"),
                    path=path,
                    status_code=status_code,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            clear()

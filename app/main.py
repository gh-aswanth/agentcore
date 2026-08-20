from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from scalar_fastapi import get_scalar_api_reference

from app.core.dependencies import create_redis
from app.core.logging import LoggingContextMiddleware, configure_logging, get_logger
from app.routers import auth, memory, runs, sessions, stream


configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled Redis client for the whole process, not one per request.
    app.state.redis = create_redis()
    try:
        yield
    finally:
        await app.state.redis.aclose()


TAGS_METADATA = [
    {
        "name": "auth",
        "description": "Register and obtain a bearer token. Every other endpoint "
                       "requires one, and scopes its query to that user.",
    },
    {
        "name": "sessions",
        "description": "An agent session: a name, a system prompt, and the tool "
                       "allowlist its runs may call. Unknown tool names are "
                       "rejected with a 422 naming the valid ones.",
    },
    {
        "name": "runs",
        "description": "Submit a message and inspect the resulting run. `POST /run` "
                       "returns 202 immediately and a Celery worker executes the "
                       "ReAct loop; the trace is append-only.",
    },
    {
        "name": "stream",
        "description": "Server-Sent Events for one run. Backed by a Redis Stream, so "
                       "connecting late replays the whole trace and `Last-Event-ID` "
                       "resumes exactly. Frame types: `llm_call`, `tool_call`, "
                       "`tool_result`, `content_delta`, `tool_call_delta`, "
                       "`final_answer`, `needs_input`, `done`.",
    },
    {
        "name": "memory",
        "description": "Long-term memory: discrete facts the agent stored with "
                       "`remember_fact`, retrieved by pgvector cosine similarity and "
                       "scoped to the caller.",
    },
    {"name": "meta", "description": "Liveness."},
]

DESCRIPTION = """\
A ReAct agent backend: sessions with a tool allowlist, a background LLM↔tool loop
that persists every step, two-tier memory (Redis + pgvector), and a live SSE trace
with replay.

**Getting started** — `POST /auth/register` returns a bearer token; paste it into
*Authorize* and every request below is scoped to that user. Create a session with
the tools it may use, `POST` a message to it, then watch `GET /runs/{id}/stream`.

Ownership is enforced in the `WHERE` clause, so another user's resource returns
**404, not 403** — a 403 would confirm the id exists.
"""

app = FastAPI(
    title="AgentCore",
    version="1.0.0",
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None
)

app.add_middleware(LoggingContextMiddleware)

app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(runs.router)
app.include_router(memory.router)
app.include_router(stream.router)


@app.get(
    "/health",
    tags=["meta"],
    summary="Liveness probe",
    description="Used by the container healthcheck. Does not touch Postgres or Redis, "
                "so it reports that the process is up, not that its dependencies are.",
)
async def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
@app.get("/scalar", include_in_schema=False)
async def scalar_reference():
    """Scalar API reference, alongside FastAPI's own /docs and /redoc.

    The Scalar bundle loads from a CDN in the browser, so this page needs
    internet access on the client side; /docs keeps working without it.
    """
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=f"{app.title} API reference",
        # keep the bearer token across reloads — otherwise every page refresh
        # means registering again to try an endpoint
        persist_auth=True,
        default_open_all_tags=True,
        # the library enables this by default; a deliverable should not phone
        # home from someone else's machine
        telemetry=False,
    )


# Consistent error envelope across handled failures and validation failures.
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"status": exc.status_code, "detail": exc.detail}},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        # jsonable_encoder: Pydantic v2 error dicts carry a live exception object
        # in `ctx`, which JSONResponse cannot serialise on its own.
        content=jsonable_encoder({"error": {"status": 422, "detail": exc.errors()}}),
    )

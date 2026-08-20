"""C-13 — SSE endpoint. Authorise, then relay one run's events to one client."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.agent.events import BEGINNING, read_events, stream_exists
from app.core.dependencies import CurrentUser, DbSession, RedisDep
from app.core.logging import bind, get_logger
from app.models.agent import TERMINAL_STATUSES, AgentRun, RunStep
from app.routers.runs import get_owned_run
from app.schemas.errors import OWNED

log = get_logger(__name__)
router = APIRouter(tags=["stream"])


def _frame(event_id: str, data: dict) -> dict:
    """`id:` is the Redis entry ID unchanged — that identity is what makes
    Last-Event-ID resume exact. `event:` lets clients bind per-type handlers
    instead of parsing every payload."""
    return {
        "id": event_id,
        "event": data.get("step_type", "message"),
        "data": json.dumps(data),
    }


async def _replay_from_db(db, run: AgentRun):
    """C-13.4 — a run older than the stream TTL has no stream left; XREAD would
    block on a key that will never receive another entry. Replaying the durable
    trace turns that hang into a clean, complete response, which is precisely
    why the trace is persisted separately from the transport."""
    stmt = (
        select(RunStep)
        .where(RunStep.run_id == run.id)
        .order_by(RunStep.occurred_at, RunStep.id)
    )
    for step in (await db.execute(stmt)).scalars():
        yield _frame(step.id, {"step_type": step.step_type, "payload": step.payload})
    yield _frame("done", {"step_type": "done", "status": run.status.value})


@router.get(
    "/runs/{run_id}/stream",
    summary="Stream a run as Server-Sent Events",
    description=(
        "Backed by a Redis Stream read from cursor `0`, so connecting *after* the run "
        "started replays the whole trace and then continues live — there is no separate "
        "replay mode. The SSE `id:` field is the Redis entry id, so a browser "
        "`EventSource` reconnect resumes exactly via `Last-Event-ID`.\n\n"
        "Frame types: `llm_call`, `tool_call`, `tool_result`, `tool_timeout`, "
        "`content_delta`, `tool_call_delta`, `needs_input`, `final_answer`, `error`, "
        "and a final `done` carrying the terminal status.\n\n"
        "A completed run whose stream has expired (1h TTL) is replayed from Postgres "
        "instead of hanging."
    ),
    response_class=EventSourceResponse,
    responses={
        **OWNED,
        200: {
            "description": "An SSE stream, terminated by a `done` frame.",
            "content": {"text/event-stream": {"example":
                'id: 1755600000000-0\nevent: tool_call\ndata: {"step_type": "tool_call", ...}\n\n'}},
        },
    },
)
async def stream_run(
    run_id: str,
    request: Request,
    user: CurrentUser,
    db: DbSession,
    redis: RedisDep,
) -> EventSourceResponse:
    run = await get_owned_run(db, run_id, user.id)     # 404, not 403
    bind(run_id=run_id, session_id=run.session_id)

    if run.status in TERMINAL_STATUSES and not await stream_exists(redis, run_id):
        await log.ainfo("stream_replayed_from_db", status=run.status.value)
        return EventSourceResponse(_replay_from_db(db, run))

    # EventSource replays the last id: it saw in this header on reconnect, and
    # because that id *is* a Redis entry ID it drops straight in as the cursor.
    cursor = request.headers.get("Last-Event-ID") or BEGINNING

    await log.ainfo("stream_opened", cursor=cursor, status=run.status.value)

    async def event_generator():
        frames = 0
        try:
            async for event in read_events(redis, run_id, last_id=cursor):
                if await request.is_disconnected():
                    break
                if event is None:
                    continue          # heartbeat tick: the yield is what lets us check disconnect
                yield _frame(event.id, event.data)
                frames += 1
                if event.data.get("step_type") == "done":
                    break
            await log.ainfo("stream_closed", frames=frames, reason="done")
        except asyncio.CancelledError:
            # A vanished client is expected, not an error.
            await log.adebug("stream_closed", frames=frames, reason="client_disconnected")
            raise

    return EventSourceResponse(event_generator())

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import func, select

from app.agent.runner import CANCEL_KEY
from app.core.config import settings
from app.core.dependencies import CurrentUser, DbSession, RedisDep
from app.core.limits import claim_run_slot, release_run_slot
from app.core.logging import bind, current, get_logger
from app.models.agent import TERMINAL_STATUSES, AgentRun, AgentSession, RunStatus, RunStep
from app.routers.sessions import get_owned_session
from app.schemas.agent import RunRequest, RunResponse, RunStatusOut, StepOut
from app.schemas.errors import CONFLICT, OWNED, TOO_MANY_REQUESTS, UNPROCESSABLE
from app.tasks.agent_tasks import execute_agent_run, revoke_run

router = APIRouter(tags=["runs"])
log = get_logger(__name__)


async def _abandon(db, run: AgentRun) -> None:
    """Close out a run whose task was never queued. Best effort: this already
    runs on a failure path, and a secondary failure here must not replace the
    original error the caller is about to see."""
    try:
        run.status = RunStatus.FAILED
        run.completed_at = datetime.now(timezone.utc)
        db.add(
            RunStep(
                run_id=run.id,
                step_type="error",
                payload={"error": "The run was created but could not be queued for execution."},
            )
        )
        await db.commit()
    except Exception:                                   # pragma: no cover - defensive
        await log.aexception("run_abandon_failed", run_id=run.id)


def _owned_run_stmt(run_id: str, user_id: str):
    """Runs carry no user_id — isolation is reached by joining agent_sessions."""
    return (
        select(AgentRun)
        .join(AgentSession, AgentRun.session_id == AgentSession.id)
        .where(AgentRun.id == run_id, AgentSession.user_id == user_id)
    )


async def get_owned_run(db, run_id: str, user_id: str) -> AgentRun:
    run = (await db.execute(_owned_run_stmt(run_id, user_id))).scalar_one_or_none()
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return run


@router.post(
    "/sessions/{session_id}/run",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,     # queued, not complete
    summary="Submit a message to the agent",
    description="Returns immediately with a `run_id`; a Celery worker executes the "
                "ReAct loop. Watch it with `GET /runs/{run_id}/stream` or poll "
                "`GET /runs/{run_id}/status`.\n\n"
                "Limited to 10 concurrent QUEUED-or-RUNNING runs per user.",
    responses={**OWNED, 429: TOO_MANY_REQUESTS, 422: UNPROCESSABLE},
)
async def create_run(
    session_id: str, body: RunRequest, user: CurrentUser, db: DbSession, redis: RedisDep
):
    session = await get_owned_session(db, session_id, user.id)

    bind(session_id=session.id)

    granted, active = await claim_run_slot(redis, db, user.id)
    if not granted:
        await log.awarning("run_rejected_rate_limit", active_runs=active,
                           limit=settings.MAX_CONCURRENT_RUNS_PER_USER)
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"You already have {active} active runs "
            f"(limit {settings.MAX_CONCURRENT_RUNS_PER_USER}). "
            "Wait for one to finish or cancel one.",
            headers={"Retry-After": "5"},
        )

    run: AgentRun | None = None
    try:
        run = AgentRun(session_id=session.id, user_message=body.message, status=RunStatus.QUEUED)
        db.add(run)
        await db.commit()
        # request_id rides along so the worker's records join up with the HTTP
        # request that created them.
        execute_agent_run.delay(run.id, request_id=current("request_id"))
    except Exception:
        await log.aexception("run_enqueue_failed")
        # Two different failures land here and they need different cleanup.
        #
        # If the commit never landed there is no row, and nothing else will ever
        # release this slot — so release it and be done.
        #
        # If the commit landed but the enqueue did not, the row exists as QUEUED
        # with no task behind it. Left alone it is counted by count_active_runs
        # forever, holding one of the user's ten slots for good, and a later
        # DELETE would release the slot a second time. Marking it FAILED settles
        # both: it stops counting as active, and DELETE now answers 409.
        if run is not None and run.id is not None:
            await _abandon(db, run)
        await release_run_slot(redis, user.id, run_id=run.id if run is not None else None)
        raise

    bind(run_id=run.id)
    await log.ainfo("run_submitted", active_runs=active, message_chars=len(body.message))
    return RunResponse(run_id=run.id, status=run.status)


@router.get(
    "/runs/{run_id}/status",
    response_model=RunStatusOut,
    summary="Poll a run's status",
    description="`step_count` is a COUNT projection rather than a loaded relationship, "
                "so polling this every second does not drag the whole trace with it.",
    responses=OWNED,
)
async def run_status(run_id: str, user: CurrentUser, db: DbSession):
    # C-11.4 — step_count is a COUNT projection, not len(run.steps). The
    # supplied lazy='selectin' would otherwise load the entire trace on every
    # poll, and clients poll every second.
    stmt = (
        select(AgentRun, func.count(RunStep.id))
        .join(AgentSession, AgentRun.session_id == AgentSession.id)
        .outerjoin(RunStep, RunStep.run_id == AgentRun.id)
        .where(AgentRun.id == run_id, AgentSession.user_id == user.id)
        .group_by(AgentRun.id)
    )
    row = (await db.execute(stmt)).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")

    run, step_count = row
    return RunStatusOut(
        run_id=run.id,
        status=run.status,
        tokens_used=run.tokens_used,
        reasoning_tokens=run.reasoning_tokens,
        step_count=step_count,
        final_answer=run.final_answer,
        created_at=run.created_at,
        completed_at=run.completed_at,
    )


@router.get(
    "/runs/{run_id}/steps",
    response_model=list[StepOut],
    summary="Read the full trace",
    description="Every step in order. Append-only: nothing here is ever updated or "
                "deleted. Step types: `llm_call`, `tool_call`, `tool_result`, "
                "`tool_timeout`, `needs_input`, `final_answer`, `error`.",
    responses=OWNED,
)
async def run_steps(run_id: str, user: CurrentUser, db: DbSession):
    await get_owned_run(db, run_id, user.id)
    stmt = (
        select(RunStep)
        .where(RunStep.run_id == run_id)
        .order_by(RunStep.occurred_at, RunStep.id)
    )
    return list((await db.execute(stmt)).scalars())


@router.delete(
    "/runs/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a run",
    description="Revokes the Celery task with `SIGUSR1`, which raises inside the "
                "worker and stops the loop mid-iteration rather than waiting for it "
                "to finish. A Redis flag is also set as a backstop for the window "
                "before the task is picked up.",
    responses={**OWNED, 409: CONFLICT},
)
async def cancel_run(run_id: str, user: CurrentUser, db: DbSession, redis: RedisDep):
    run = await get_owned_run(db, run_id, user.id)
    if run.status in TERMINAL_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Run already {run.status.value}")

    bind(run_id=run_id, session_id=run.session_id)
    was_queued = run.status is RunStatus.QUEUED

    # Backstop: covers the window where the task is still on the broker and has
    # no worker to signal, and the moment after a revoke lands.
    await redis.set(CANCEL_KEY.format(run_id=run_id), "1", ex=3600)
    revoke_run(run.celery_task_id)

    run.status = RunStatus.CANCELLED
    await db.commit()

    # A run still QUEUED was never claimed by a worker, so nothing else will
    # release its slot. One that was RUNNING is released by the worker's finally.
    if was_queued:
        await release_run_slot(redis, user.id, run_id=run_id)

    await log.ainfo("run_cancelled", was_queued=was_queued, task_id=run.celery_task_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

"""C-10 — background execution."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

try:
    import uvloop
except ImportError:                                    # pragma: no cover
    uvloop = None

from celery.exceptions import SoftTimeLimitExceeded

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.agent import events
from app.agent.memory_manager import MemoryManager
from app.agent.runner import CANCEL_KEY, CANCELLED_ANSWER, AgentRunner
from app.core.config import settings
from app.core.dependencies import create_redis
from app.core.limits import release_run_slot
from app.core.logging import bind, clear, get_logger
from app.models.agent import TERMINAL_STATUSES, AgentRun, AgentSession, RunStatus, RunStep
from app.core.config import settings
from app.tasks.celery_app import SOFT_TIME_LIMIT, celery_app

logger = get_logger(__name__)   # configured by the setup_logging signal


def run_async(coro):
    """Run a coroutine on the fastest loop available.

    uvicorn picks uvloop for the API process on its own, but the worker calls
    `asyncio.run` directly and so was still on the stdlib selector loop — which is
    the process that matters most here, since the agent loop is almost entirely
    awaited I/O: OpenAI calls, Redis reads, Postgres round-trips.

    The fallback is real rather than defensive: uvloop has no wheels for Windows
    or PyPy, and a worker should start there rather than refuse to import.
    """
    if uvloop is not None:
        return uvloop.run(coro)
    return asyncio.run(coro)

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _worker_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Built lazily, inside the forked child, with NullPool.

    Celery's prefork pool forks *after* module import. A pooled async engine
    created at import time hands every child the same open sockets, and they
    corrupt each other's protocol state in ways that look like random query
    failures.
    """
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
        _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _session_factory


def revoke_run(task_id: str | None) -> None:
    """Ask the worker running this task to stop now.

    ``terminate=True`` with ``SIGUSR1`` makes Celery raise
    ``SoftTimeLimitExceeded`` *inside* the task rather than killing the process,
    so the task's own ``except`` runs: the DB is left consistent and the SSE
    client still receives a `done` frame. SIGTERM/SIGKILL would stop the work
    just as fast and leave the run stuck at RUNNING forever.

    Requires the prefork pool (the default). Under ``--pool=solo`` or a thread
    pool the signal has nowhere to land and the Redis flag is the only backstop.
    """
    if not task_id:
        return
    celery_app.control.revoke(task_id, terminate=True, signal="SIGUSR1")


ATTEMPTS_KEY = "run:{run_id}:attempts"
ATTEMPTS_TTL = 60 * 60 * 24


async def _finish_interrupted(session_factory, redis, run_id: str) -> None:
    """Close out a run stopped by SoftTimeLimitExceeded.

    That exception has two sources and they are not the same event: our own
    revocation via SIGUSR1, and Celery's `task_soft_time_limit` firing on a run
    that overran. Recording an overrun as CANCELLED would put a lie in the trace —
    it would read as though the user asked to stop. The cancel flag that
    `DELETE /runs/{id}` sets is what tells them apart.

    Uses a *fresh* session on purpose: the one in flight was interrupted
    mid-transaction and may be unusable.
    """
    revoked = bool(await redis.exists(CANCEL_KEY.format(run_id=run_id)))
    status = RunStatus.CANCELLED if revoked else RunStatus.FAILED
    step_type = "cancelled" if revoked else "error"
    payload = (
        {"reason": "revoked by the user mid-run"}
        if revoked
        else {"error": f"The run exceeded the {SOFT_TIME_LIMIT}s time limit and was stopped."}
    )

    async with session_factory() as db:
        run = await db.get(AgentRun, run_id)
        if run is None or (run.status in TERMINAL_STATUSES and run.completed_at is not None):
            return
        run.status = status
        run.final_answer = CANCELLED_ANSWER if revoked else None
        run.completed_at = datetime.now(timezone.utc)
        db.add(RunStep(run_id=run_id, step_type=step_type, payload=payload))
        await db.commit()

    # so the SSE client's stream terminates rather than hanging
    await events.publish_event(redis, run_id, {"step_type": "done", "status": status.value})


@celery_app.task(name="execute_agent_run", bind=True, max_retries=3)
def execute_agent_run(self, run_id: str, request_id: str | None = None) -> None:
    """Celery 5 tasks are synchronous, so the async body runs under asyncio.run.

    That creates and disposes an event loop per task — fine for second-scale
    work. At throughput the right answer is a persistent loop or an async-native
    queue (arq / taskiq); noted in NOTES.md.
    """
    try:
        return run_async(
            _execute(run_id, task_id=self.request.id, request_id=request_id)
        )
    except SoftTimeLimitExceeded:
        # SIGUSR1 is raised at an arbitrary bytecode boundary in this thread. If
        # that lands inside the event loop's own machinery rather than inside our
        # coroutine's frame, it propagates straight out of asyncio.run() and the
        # coroutine's handler never runs — so the same cleanup has to exist here,
        # on a fresh loop. Swallowed rather than re-raised: a user cancelling is
        # not a task failure, and re-raising would retry the run.
        logger.info("run_revoked_outside_loop")
        run_async(_cleanup_revoked(run_id))


async def _execute(
    run_id: str, task_id: str | None = None, request_id: str | None = None
) -> None:
    clear()
    # request_id comes from the HTTP request that submitted this run, so a single
    # grep spans the API process and the worker process.
    bind(run_id=run_id, task_id=task_id, request_id=request_id)

    redis = create_redis()
    session_factory = _worker_sessionmaker()
    claimed_user_id: str | None = None

    try:
        claimed_user_id = await _claim_and_run(session_factory, redis, run_id, task_id)
    except SoftTimeLimitExceeded:
        # SIGUSR1 from DELETE /runs/{id}. It can land at any instant — during the
        # FOR UPDATE claim, between iterations, inside a tool — so the handler is
        # here rather than around the agent loop alone. Not a failure: finish the
        # row honestly and terminate the client's stream.
        await logger.ainfo("run_revoked")
        await _finish_interrupted(session_factory, redis, run_id)
    finally:
        # Whoever moved the run out of QUEUED owns its concurrency slot. A run
        # cancelled while still QUEUED is released by the route instead, because
        # this task never claimed it.
        if claimed_user_id:
            await release_run_slot(redis, claimed_user_id, run_id=run_id)
        await redis.aclose()


async def _cleanup_revoked(run_id: str) -> None:
    """Finish the bookkeeping for a revoked run: mark it, tell the client, and
    give the concurrency slot back. Safe to run twice."""
    redis = create_redis()
    try:
        session_factory = _worker_sessionmaker()
        await _finish_interrupted(session_factory, redis, run_id)

        async with session_factory() as db:
            stmt = (
                select(AgentSession.user_id)
                .join(AgentRun, AgentRun.session_id == AgentSession.id)
                .where(AgentRun.id == run_id)
            )
            user_id = (await db.execute(stmt)).scalar_one_or_none()
        if user_id:
            await release_run_slot(redis, user_id, run_id=run_id)
    finally:
        await redis.aclose()


async def _claim_and_run(session_factory, redis, run_id: str, task_id: str | None) -> str | None:
    """Claim the run and execute it. Returns the owning user id if this task
    took the run (and therefore owns its concurrency slot), else None."""
    async with session_factory() as db:
        # --- idempotency: an atomic claim, not read-then-write ---
        # A plain SELECT followed by UPDATE races: two workers both read QUEUED
        # and both run the loop. The row lock makes the claim atomic.
        stmt = select(AgentRun).where(AgentRun.id == run_id).with_for_update()
        run = (await db.execute(stmt)).scalar_one_or_none()
        if run is None:
            await logger.awarning("run_not_found")
            return None
        if run.status in TERMINAL_STATUSES:
            await logger.ainfo("duplicate_delivery_ignored", status=run.status.value)
            return None

        if run.status is RunStatus.RUNNING and run.celery_task_id != task_id:
            # Another worker holds this run right now. Step aside.
            await logger.ainfo("run_already_held", holder=run.celery_task_id)
            return None

        # Either QUEUED, or RUNNING under *our own* task id — which means this is
        # our message coming back after the previous attempt died. Celery keeps
        # the task id across a redelivery, so that comparison is what separates
        # "someone else is running it" from "I am, and I crashed". Without it,
        # task_acks_late would redeliver work that this branch then discards, and
        # the run would stay RUNNING forever.
        attempt = await redis.incr(ATTEMPTS_KEY.format(run_id=run_id))
        await redis.expire(ATTEMPTS_KEY.format(run_id=run_id), ATTEMPTS_TTL)
        if attempt > settings.MAX_RUN_ATTEMPTS:
            # A run that kills its worker would otherwise be requeued forever by
            # task_reject_on_worker_lost, and with prefetch=1 it would occupy a
            # worker permanently.
            await logger.aerror("run_abandoned_too_many_attempts", attempts=attempt)
            run.status = RunStatus.FAILED
            run.completed_at = datetime.now(timezone.utc)
            db.add(
                RunStep(
                    run_id=run_id,
                    step_type="error",
                    payload={"error": f"Abandoned after {attempt - 1} failed attempts."},
                )
            )
            await db.commit()
            await events.publish_event(
                redis, run_id, {"step_type": "done", "status": RunStatus.FAILED.value}
            )
            owner = await db.get(AgentSession, run.session_id)
            return owner.user_id if owner else None

        if attempt > 1:
            await logger.awarning("run_retaken_after_worker_loss", attempt=attempt)

        run.status = RunStatus.RUNNING
        # Recorded on the claim so DELETE has something to revoke, and so a retry
        # replaces the id rather than leaving a stale one behind.
        run.celery_task_id = task_id
        await db.commit()              # releases the lock

        session = await db.get(AgentSession, run.session_id)
        if session is None:            # pragma: no cover - the FK makes this unreachable
            run.status = RunStatus.FAILED
            await db.commit()
            return None

        claimed_user_id = session.user_id
        # bound as early as it is known, so even a failure before the loop starts
        # is attributable to a user
        bind(user_id=session.user_id, session_id=session.id)
        memory = MemoryManager(redis, db)
        try:
            await AgentRunner(db, redis, memory).run(run, session)
        except SoftTimeLimitExceeded:
            # Must precede the generic handler: SoftTimeLimitExceeded *is* an
            # Exception, and recording a user's cancellation as a failure would
            # be a lie in the trace.
            raise
        except Exception as exc:
            run.status = RunStatus.FAILED
            db.add(
                RunStep(
                    run_id=run.id,
                    step_type="error",
                    payload={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            await db.commit()
            # Publish `done` so the SSE client's stream terminates rather than
            # hanging until it gives up.
            await events.publish_event(
                redis, run.id, {"step_type": "done", "status": RunStatus.FAILED.value}
            )
            raise

        return claimed_user_id

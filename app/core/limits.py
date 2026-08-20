"""Per-user concurrency limit for agent runs.

A Redis counter is the fast path: `POST /run` must stay in the tens of
milliseconds, and counting rows in Postgres on every submission would put a
query on the hot path for a number that is almost always far below the limit.

But a counter held outside the data it describes will drift — a worker killed
between claiming a run and finishing it leaks a slot forever, and the user is
then locked out by a number that describes nothing. So Postgres stays the source
of truth and is consulted at exactly the moment the answer matters: when the
counter says the user is full. If the rows disagree, the counter is wrong and is
reset from them.
"""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.agent import AgentRun, AgentSession, RunStatus

logger = get_logger(__name__)

ACTIVE_RUNS_KEY = "user:{user_id}:active_runs"
# Long enough that it never expires under a live workload, short enough that an
# abandoned counter cannot outlive the process that leaked it.
ACTIVE_RUNS_TTL = 60 * 60 * 24

# Set once per run, so the slot is given back exactly once no matter which path
# gets there first — the worker's `finally`, the revocation handler, or the
# cancel route.
SLOT_RELEASED_KEY = "run:{run_id}:slot_released"

ACTIVE_STATUSES = (RunStatus.QUEUED, RunStatus.RUNNING)


def _key(user_id: str) -> str:
    return ACTIVE_RUNS_KEY.format(user_id=user_id)


async def count_active_runs(db: AsyncSession, user_id: str) -> int:
    """The authoritative count, straight from the rows."""
    stmt = (
        select(func.count(AgentRun.id))
        .join(AgentSession, AgentRun.session_id == AgentSession.id)
        .where(AgentSession.user_id == user_id, AgentRun.status.in_(ACTIVE_STATUSES))
    )
    return int((await db.execute(stmt)).scalar_one())


async def claim_run_slot(redis, db: AsyncSession, user_id: str) -> tuple[bool, int]:
    """Reserve one of the user's concurrent-run slots.

    Returns ``(granted, active_count)``. INCR is atomic, so two simultaneous
    submissions cannot both see the last free slot.
    """
    limit = settings.MAX_CONCURRENT_RUNS_PER_USER
    key = _key(user_id)

    pipe = redis.pipeline(transaction=False)
    pipe.incr(key)
    pipe.expire(key, ACTIVE_RUNS_TTL)
    count, _ = await pipe.execute()

    await logger.ainfo(
        "active_run_counter", redis_count=count
    )
    if count <= limit:
        return True, count

    # The counter says full. Before rejecting a user on the strength of a number
    # that may have drifted, check the rows.
    actual = await count_active_runs(db, user_id)
    await logger.ainfo(
        "active_run_counter", redis_count=count, postgres_count=actual
    )
    if actual >= limit:
        await redis.set(key, actual, ex=ACTIVE_RUNS_TTL)   # trust the rows either way
        return False, actual

    await logger.awarning(
        "active_run_counter_drift", redis_count=count, postgres_count=actual
    )
    await redis.set(key, actual + 1, ex=ACTIVE_RUNS_TTL)
    return True, actual + 1


async def release_run_slot(redis, user_id: str, run_id: str | None = None) -> None:
    """Give a slot back, at most once per run.

    Floors at zero: an extra release must never make the counter negative and
    hand the user unlimited runs. When ``run_id`` is given, a SET NX marker makes
    the release idempotent — revocation in particular can run the cleanup path
    and the worker's ``finally`` for the same run.
    """
    if run_id is not None:
        claimed_first = await redis.set(
            SLOT_RELEASED_KEY.format(run_id=run_id), "1", nx=True, ex=ACTIVE_RUNS_TTL
        )
        if not claimed_first:
            return

    key = _key(user_id)
    remaining = await redis.decr(key)
    if remaining < 0:
        await redis.set(key, 0, ex=ACTIVE_RUNS_TTL)
    else:
        await redis.expire(key, ACTIVE_RUNS_TTL)

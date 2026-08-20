"""Celery settings that are load-bearing for *this* workload, and the code that
makes them correct.

Each assertion here is a decision that would be silently wrong at a default, so
the test states the reason rather than just pinning a number.
"""
import pytest
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import select

from app.agent.runner import CANCEL_KEY
from app.core.config import settings
from app.models.agent import AgentRun, AgentSession, RunStatus, RunStep
from app.models.user import User
from app.tasks import agent_tasks
from app.tasks.celery_app import HARD_TIME_LIMIT, SOFT_TIME_LIMIT, celery_app


# ───────────────────────────── configuration ──────────────────────────────── #
def test_messages_are_acked_after_the_work_not_on_receipt():
    """A worker killed mid-run would otherwise take the message with it, leaving
    the run at RUNNING forever and holding one of the user's ten slots — nothing
    else in the system ever revisits it."""
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True


def test_a_worker_reserves_only_what_it_is_working_on():
    """Celery's default of 4 means a --concurrency=2 worker holds 8 messages.
    These tasks run for minutes, so the extras queue behind the one in progress
    while another worker sits idle."""
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_the_broker_cannot_redeliver_a_task_that_may_still_be_running():
    """Redis has no real ack: it redelivers anything unacked after
    visibility_timeout. Below the hard limit it would hand a live task to a second
    worker — and with acks_late that is genuine double execution."""
    visibility = celery_app.conf.broker_transport_options["visibility_timeout"]
    assert visibility > HARD_TIME_LIMIT


def test_the_hard_limit_trails_the_soft_one():
    """The soft limit raises inside the task so the run finishes honestly; the
    hard limit is only the backstop for a task that ignores it."""
    assert SOFT_TIME_LIMIT == settings.AGENT_RUN_TIME_LIMIT_SECONDS
    assert HARD_TIME_LIMIT > SOFT_TIME_LIMIT
    assert celery_app.conf.task_soft_time_limit == SOFT_TIME_LIMIT
    assert celery_app.conf.task_time_limit == HARD_TIME_LIMIT


def test_a_run_can_outlast_the_agent_loops_own_ceiling():
    """The time limit is a guard against a wedged run, not a cap on a normal one:
    ten iterations of a reasoning model must fit inside it comfortably."""
    assert SOFT_TIME_LIMIT >= settings.MAX_AGENT_ITERATIONS * 60


def test_workers_are_recycled_and_reconnect_on_startup():
    assert celery_app.conf.worker_max_tasks_per_child == settings.WORKER_MAX_TASKS_PER_CHILD
    assert celery_app.conf.broker_connection_retry_on_startup is True


# ───────────────────── what acks_late actually requires ───────────────────── #
@pytest.fixture
async def queued_run(db):
    user = User(email="celery@example.com", hashed_password="x")
    db.add(user)
    await db.commit()
    session = AgentSession(user_id=user.id, name="S", tools_enabled=[])
    db.add(session)
    await db.commit()
    run = AgentRun(session_id=session.id, user_message="m", status=RunStatus.QUEUED)
    db.add(run)
    await db.commit()
    return user, session, run


def _factory(db):
    """A session factory that hands back the test's session."""
    class _Ctx:
        async def __aenter__(self): return db
        async def __aexit__(self, *exc): return False
    return lambda: _Ctx()


async def test_our_own_redelivered_message_retakes_the_run(db, redis, queued_run, monkeypatch):
    """Celery keeps the task id across a redelivery. RUNNING under our own id
    means the previous attempt died — not that someone else is working on it.
    Without this, acks_late redelivers work that the claim then discards and the
    run stays RUNNING forever."""
    user, session, run = queued_run
    run.status = RunStatus.RUNNING
    run.celery_task_id = "task-A"
    await db.commit()

    ran = []
    monkeypatch.setattr(
        agent_tasks, "AgentRunner",
        lambda *a, **kw: type("R", (), {"run": lambda self, r, s: ran.append(True) or _done()})(),
    )

    async def _done():
        return "ok"

    owner = await agent_tasks._claim_and_run(_factory(db), redis, run.id, "task-A")

    assert owner == user.id
    assert ran == [True]


async def test_a_run_held_by_another_worker_is_left_alone(db, redis, queued_run):
    """A different task id means a live worker owns it; taking over would be
    double execution, which is exactly what the FOR UPDATE claim exists to stop."""
    _user, _session, run = queued_run
    run.status = RunStatus.RUNNING
    run.celery_task_id = "task-SOMEONE-ELSE"
    await db.commit()

    owner = await agent_tasks._claim_and_run(_factory(db), redis, run.id, "task-MINE")

    assert owner is None


async def test_a_run_that_keeps_killing_its_worker_is_abandoned(db, redis, queued_run):
    """task_reject_on_worker_lost requeues forever, and with prefetch=1 a poison
    task would occupy a worker permanently."""
    _user, _session, run = queued_run
    await redis.set(
        agent_tasks.ATTEMPTS_KEY.format(run_id=run.id), settings.MAX_RUN_ATTEMPTS
    )

    await agent_tasks._claim_and_run(_factory(db), redis, run.id, "task-A")
    await db.refresh(run)

    assert run.status is RunStatus.FAILED
    step = (
        await db.execute(select(RunStep).where(RunStep.step_type == "error"))
    ).scalars().all()[-1]
    assert "Abandoned after" in step.payload["error"]


# ──────────── the soft limit collides with our own cancellation ───────────── #
async def test_a_revoked_run_is_recorded_as_cancelled(db, redis, queued_run):
    _user, _session, run = queued_run
    await redis.set(CANCEL_KEY.format(run_id=run.id), "1")

    await agent_tasks._finish_interrupted(_factory(db), redis, run.id)
    await db.refresh(run)

    assert run.status is RunStatus.CANCELLED
    assert run.final_answer == "Run cancelled"
    steps = (await db.execute(select(RunStep).where(RunStep.run_id == run.id))).scalars().all()
    assert steps[-1].step_type == "cancelled"


async def test_a_run_that_merely_overran_is_recorded_as_failed(db, redis, queued_run):
    """SoftTimeLimitExceeded has two sources — our SIGUSR1 revoke and Celery's
    own time limit. Recording an overrun as CANCELLED would read as though the
    user asked to stop, which is a lie in the trace. The cancel flag separates
    them, and here it is absent."""
    _user, _session, run = queued_run

    await agent_tasks._finish_interrupted(_factory(db), redis, run.id)
    await db.refresh(run)

    assert run.status is RunStatus.FAILED
    assert run.final_answer is None
    steps = (await db.execute(select(RunStep).where(RunStep.run_id == run.id))).scalars().all()
    assert steps[-1].step_type == "error"
    assert "time limit" in steps[-1].payload["error"]


def test_soft_time_limit_exceeded_is_not_swallowed_as_a_generic_failure():
    """It is an Exception subclass, so the ordering of the two handlers in
    _claim_and_run is what stops a cancellation being logged as a crash."""
    assert issubclass(SoftTimeLimitExceeded, Exception)
    source = (agent_tasks.__file__,)
    body = open(source[0]).read()
    soft = body.index("except SoftTimeLimitExceeded:")
    generic = body.index("except Exception as exc:")
    assert soft < generic, "the specific handler must precede the generic one"

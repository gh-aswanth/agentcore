"""The five bonus features, each tested at the seam where it can actually fail."""
import asyncio
import json
import logging
import time

import pytest
from sqlalchemy import select

from app.agent.runner import AgentRunner
from app.agent.tools import TOOL_REGISTRY, dispatch, is_timeout, tool
from app.core.config import settings
from app.core.limits import (
    ACTIVE_RUNS_KEY,
    claim_run_slot,
    count_active_runs,
    release_run_slot,
)
from app.models.agent import AgentRun, AgentSession, RunStatus, RunStep
from app.models.user import User
from app.tests.conftest import fake_tool_call, llm_response


# ═══════════════════════ 1 · structured logging ═══════════════════════════ #
def test_logs_are_json_with_the_bound_context(capsys):
    from app.core.logging import bind, clear, configure_logging, get_logger

    configure_logging()
    clear()
    bind(user_id="u1", session_id="s1", run_id="r1", step_type="tool_call")
    get_logger("test").info("tool_dispatched", name="calculator")

    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["event"] == "tool_dispatched"
    # the four keys the brief asks for, on a record that never mentioned them
    assert record["user_id"] == "u1"
    assert record["session_id"] == "s1"
    assert record["run_id"] == "r1"
    assert record["step_type"] == "tool_call"
    assert record["level"] == "info"
    assert "timestamp" in record
    clear()


def test_stdlib_loggers_are_routed_through_the_same_chain(capsys):
    """A run's records must be one stream — SQLAlchemy and Celery included."""
    from app.core.logging import bind, clear, configure_logging

    configure_logging()
    clear()
    bind(run_id="r2")
    logging.getLogger("some.third.party").warning("not our logger")

    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["run_id"] == "r2"
    assert record["level"] == "warning"
    clear()


def test_context_does_not_leak_between_tasks():
    from app.core.logging import bind, clear
    import structlog

    async def scenario(run_id):
        clear()
        bind(run_id=run_id)
        await asyncio.sleep(0)
        return structlog.contextvars.get_contextvars().get("run_id")

    async def main():
        return await asyncio.gather(scenario("a"), scenario("b"))

    assert set(asyncio.run(main())) == {"a", "b"}


# ═══════════════════════ 2 · rate limiting ════════════════════════════════ #
@pytest.fixture
async def user_with_session(db):
    user = User(email="limit@example.com", hashed_password="x")
    db.add(user)
    await db.commit()
    session = AgentSession(user_id=user.id, name="S", tools_enabled=[])
    db.add(session)
    await db.commit()
    return user, session


async def test_the_tenth_run_is_allowed_and_the_eleventh_is_not(db, redis, user_with_session):
    """Mirrors what the route does: claim a slot, then insert the QUEUED row."""
    user, session = user_with_session

    for i in range(settings.MAX_CONCURRENT_RUNS_PER_USER):
        granted, count = await claim_run_slot(redis, db, user.id)
        assert granted, f"slot {i + 1} should have been granted"
        assert count == i + 1
        db.add(AgentRun(session_id=session.id, user_message="m", status=RunStatus.QUEUED))
        await db.commit()

    granted, count = await claim_run_slot(redis, db, user.id)

    assert granted is False
    assert count == settings.MAX_CONCURRENT_RUNS_PER_USER
    # and the counter is left describing reality, not the rejected attempt
    assert int(await redis.get(ACTIVE_RUNS_KEY.format(user_id=user.id))) == count


async def test_a_finished_run_frees_the_slot_even_if_the_counter_was_never_decremented(
    db, redis, user_with_session
):
    """The reconciliation's real purpose: a worker killed mid-run leaks a slot,
    and the user must not be locked out by a number that describes nothing."""
    user, session = user_with_session
    await redis.set(
        ACTIVE_RUNS_KEY.format(user_id=user.id), settings.MAX_CONCURRENT_RUNS_PER_USER
    )
    # every run actually finished; nothing is active
    for _ in range(3):
        db.add(AgentRun(session_id=session.id, user_message="m", status=RunStatus.COMPLETED))
    await db.commit()

    granted, count = await claim_run_slot(redis, db, user.id)

    assert granted
    assert count == 1


async def test_releasing_a_slot_lets_the_next_run_through(db, redis, user_with_session):
    user, _ = user_with_session
    await redis.set(ACTIVE_RUNS_KEY.format(user_id=user.id), settings.MAX_CONCURRENT_RUNS_PER_USER)

    await release_run_slot(redis, user.id)
    granted, _ = await claim_run_slot(redis, db, user.id)

    assert granted


async def test_the_counter_cannot_go_negative(db, redis, user_with_session):
    """An extra release must not hand the user unlimited runs."""
    user, _ = user_with_session
    for _ in range(3):
        await release_run_slot(redis, user.id)

    assert int(await redis.get(ACTIVE_RUNS_KEY.format(user_id=user.id))) == 0


async def test_a_drifted_counter_is_reset_from_the_rows(db, redis, user_with_session):
    """A worker killed mid-run leaks a slot. Postgres is the source of truth, so
    a user must not be locked out by a number that describes nothing."""
    user, session = user_with_session
    db.add(AgentRun(session_id=session.id, user_message="m", status=RunStatus.RUNNING))
    await db.commit()

    # counter claims the user is full; the rows say one run is active
    await redis.set(ACTIVE_RUNS_KEY.format(user_id=user.id), 99)
    granted, count = await claim_run_slot(redis, db, user.id)

    assert granted
    assert count == 2                                   # the real one, plus this claim
    assert await count_active_runs(db, user.id) == 1


async def test_active_count_only_sees_this_users_unfinished_runs(db, redis, user_with_session):
    user, session = user_with_session
    for status in (RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.FAILED):
        db.add(AgentRun(session_id=session.id, user_message="m", status=status))

    other = User(email="other@example.com", hashed_password="x")
    db.add(other)
    await db.commit()
    other_session = AgentSession(user_id=other.id, name="O", tools_enabled=[])
    db.add(other_session)
    await db.commit()
    db.add(AgentRun(session_id=other_session.id, user_message="m", status=RunStatus.RUNNING))
    await db.commit()

    assert await count_active_runs(db, user.id) == 2       # queued + running only


# ═══════════════════════ 3 · cancellation ═════════════════════════════════ #
def test_revoke_uses_sigusr1_and_terminate(monkeypatch):
    """SIGUSR1 raises inside the task so its except block runs; SIGTERM would
    stop the work just as fast and leave the row stuck at RUNNING."""
    from app.tasks import agent_tasks

    calls = []
    monkeypatch.setattr(
        agent_tasks.celery_app.control,
        "revoke",
        lambda task_id, **kw: calls.append((task_id, kw)),
    )

    agent_tasks.revoke_run("task-123")
    assert calls == [("task-123", {"terminate": True, "signal": "SIGUSR1"})]

    calls.clear()
    agent_tasks.revoke_run(None)          # a run never picked up has no task to revoke
    assert calls == []


async def test_the_task_id_is_recorded_so_a_run_can_be_revoked(db):
    """DELETE can only revoke what it can name."""
    assert "celery_task_id" in AgentRun.__table__.c


# ═══════════════════════ 4 · tool timeout ═════════════════════════════════ #
@pytest.fixture
def slow_tools():
    @tool
    def slow_sync(seconds: int) -> str:
        """A blocking synchronous tool."""
        time.sleep(seconds)
        return "finished"

    @tool
    async def slow_async(seconds: int) -> str:
        """A blocking asynchronous tool."""
        await asyncio.sleep(seconds)
        return "finished"

    yield
    TOOL_REGISTRY.pop("slow_sync", None)
    TOOL_REGISTRY.pop("slow_async", None)


@pytest.mark.parametrize("name", ["slow_async", "slow_sync"])
async def test_every_tool_is_time_boxed_including_synchronous_ones(name, slow_tools):
    """A sync tool cannot be interrupted by wait_for unless it is moved off the
    event loop first — without that, an 8s sleep blocks the whole worker."""
    started = time.perf_counter()
    result = await dispatch(name, '{"seconds": 8, "thought": "t", "grounding": "user"}')
    elapsed = time.perf_counter() - started

    assert is_timeout(result)
    assert elapsed < settings.TOOL_TIMEOUT_SECONDS + 2


async def test_a_fast_tool_is_unaffected():
    result = await dispatch("calculator", '{"expression": "2*21", "thought": "t"}')
    assert result == "42"
    assert not is_timeout(result)


async def test_a_timeout_is_its_own_step_type_and_the_loop_continues(
    db, redis, memory, mock_llm, slow_tools
):
    user = User(email="timeout@example.com", hashed_password="x")
    db.add(user)
    await db.commit()
    session = AgentSession(user_id=user.id, name="S", tools_enabled=["slow_sync"])
    db.add(session)
    await db.commit()
    run = AgentRun(session_id=session.id, user_message="go", status=RunStatus.RUNNING)
    db.add(run)
    await db.commit()

    mock_llm.append(
        llm_response(tool_calls=[fake_tool_call("slow_sync", '{"seconds": 8, "thought": "t"}')])
    )
    mock_llm.append(llm_response(content="That tool was too slow; answering without it."))

    answer = await AgentRunner(db, redis, memory).run(run, session)

    stmt = select(RunStep).where(RunStep.run_id == run.id).order_by(RunStep.occurred_at)
    types = [s.step_type for s in (await db.execute(stmt)).scalars()]
    assert types == ["llm_call", "tool_call", "tool_timeout", "llm_call", "final_answer"]

    timeout_step = (
        await db.execute(select(RunStep).where(RunStep.step_type == "tool_timeout"))
    ).scalar_one()
    assert timeout_step.payload["timeout_seconds"] == settings.TOOL_TIMEOUT_SECONDS
    # the loop carried on and produced an answer rather than dying
    assert answer == "That tool was too slow; answering without it."
    assert run.status is RunStatus.COMPLETED


# ═══════════════ 1b · async logging & end-to-end correlation ══════════════ #
async def test_async_log_calls_preserve_the_bound_context():
    """structlog's a* methods hand off to a thread pool; the whole point of using
    them is lost if the contextvars do not make the trip."""
    import io
    import logging as stdlib_logging

    from app.core.logging import bind, clear, configure_logging, get_logger

    configure_logging()
    clear()
    bind(request_id="req-9", user_id="u-9", run_id="r-9", session_id="s-9")

    buffer = io.StringIO()
    stdlib_logging.getLogger().handlers[0].stream = buffer
    await get_logger("probe").ainfo("async_event", extra="value")

    record = json.loads(buffer.getvalue().strip().splitlines()[-1])
    assert record["event"] == "async_event"
    assert record["extra"] == "value"
    for key, value in [("request_id", "req-9"), ("user_id", "u-9"),
                       ("run_id", "r-9"), ("session_id", "s-9")]:
        assert record[key] == value, key
    clear()


async def test_current_reads_a_bound_value_back():
    """How request_id gets from the request context onto the Celery call."""
    from app.core.logging import bind, clear, current

    clear()
    assert current("request_id") is None
    bind(request_id="abc123")
    assert current("request_id") == "abc123"
    clear()


async def test_the_worker_binds_the_request_id_it_was_given(db, redis, monkeypatch):
    """Closes the loop: the API's request_id is re-bound inside the worker, so a
    single grep spans both processes."""
    import structlog

    from app.tasks import agent_tasks

    seen = {}

    async def fake_claim(session_factory, redis_, run_id, task_id):
        seen.update(structlog.contextvars.get_contextvars())
        return None

    monkeypatch.setattr(agent_tasks, "_claim_and_run", fake_claim)
    monkeypatch.setattr(agent_tasks, "create_redis", lambda *a, **kw: redis)
    monkeypatch.setattr(agent_tasks, "_worker_sessionmaker", lambda: None)

    await agent_tasks._execute("run-1", task_id="task-1", request_id="req-from-api")

    assert seen["request_id"] == "req-from-api"
    assert seen["run_id"] == "run-1"
    assert seen["task_id"] == "task-1"


# ═══════════ 2b · slot release is idempotent, and orphans are closed ══════ #
async def test_two_releases_for_the_same_run_decrement_once(db, redis, user_with_session):
    """Cancellation can reach both the worker's `finally` and the revocation
    cleanup for one run. Two DECRs against one INCR would hand the user a
    permanent extra slot — and claim_run_slot only reconciles when the counter is
    too *high*, so an under-count is never corrected.

    Three slots are claimed on purpose: with only one, the floor-at-zero in
    release_run_slot masks the second decrement and the test passes whether the
    latch exists or not.
    """
    user, _ = user_with_session
    for _ in range(3):
        await claim_run_slot(redis, db, user.id)
    key = ACTIVE_RUNS_KEY.format(user_id=user.id)
    assert int(await redis.get(key)) == 3

    await release_run_slot(redis, user.id, run_id="run-1")
    await release_run_slot(redis, user.id, run_id="run-1")   # the second path

    assert int(await redis.get(key)) == 2                    # 1 would mean it counted twice


async def test_releases_for_different_runs_each_count(db, redis, user_with_session):
    """The latch is per run, not a global once-only switch: a single shared flag
    would swallow the second run's release and leak a slot."""
    user, _ = user_with_session
    for _ in range(3):
        await claim_run_slot(redis, db, user.id)

    await release_run_slot(redis, user.id, run_id="run-a")
    await release_run_slot(redis, user.id, run_id="run-b")

    assert int(await redis.get(ACTIVE_RUNS_KEY.format(user_id=user.id))) == 1

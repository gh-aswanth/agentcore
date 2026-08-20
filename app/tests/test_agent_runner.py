"""T-1, T-2, T-4 — the agent loop, driven entirely by a scripted LLM."""
import pytest
from sqlalchemy import select

from app.agent.events import read_events
from app.agent.runner import MAX_ITERATIONS_ANSWER, AgentRunner
from app.core.config import settings
from app.models.agent import AgentRun, AgentSession, RunStatus, RunStep
from app.models.user import User
from app.tests.conftest import fake_tool_call, llm_response


@pytest.fixture
async def scenario(db):
    user = User(email="a@example.com", hashed_password="x")
    db.add(user)
    await db.commit()

    session = AgentSession(
        user_id=user.id,
        name="Research Assistant",
        system_prompt="You are precise.",
        tools_enabled=["web_search", "calculator", "remember_fact"],
    )
    db.add(session)
    await db.commit()

    run = AgentRun(session_id=session.id, user_message="hello", status=RunStatus.RUNNING)
    db.add(run)
    await db.commit()
    return user, session, run


async def step_types(db, run_id: str) -> list[str]:
    stmt = select(RunStep).where(RunStep.run_id == run_id).order_by(RunStep.occurred_at, RunStep.id)
    return [s.step_type for s in (await db.execute(stmt)).scalars()]


# ------------------------------- T-1 --------------------------------------- #
async def test_loop_stops_on_plain_text(db, redis, memory, mock_llm, scenario):
    _user, session, run = scenario
    mock_llm.append(llm_response(content="The answer is 42.", tokens=120))

    answer = await AgentRunner(db, redis, memory).run(run, session)

    assert answer == "The answer is 42."
    assert run.status is RunStatus.COMPLETED
    assert run.final_answer == "The answer is 42."
    assert run.tokens_used == 120
    assert await step_types(db, run.id) == ["llm_call", "final_answer"]
    # the completed turn is written into short-term memory
    assert memory.turns == [(session.id, "hello", "The answer is 42.")]


# ------------------------------- T-2 --------------------------------------- #
async def test_loop_dispatches_tool_then_answers(db, redis, memory, mock_llm, scenario):
    _user, session, run = scenario
    mock_llm.append(
        llm_response(tool_calls=[fake_tool_call("calculator", '{"expression": "2*21"}')])
    )
    mock_llm.append(llm_response(content="42"))

    answer = await AgentRunner(db, redis, memory).run(run, session)

    assert answer == "42"
    # asserting on the *sequence* verifies the trace, which is what the trace is for
    assert await step_types(db, run.id) == [
        "llm_call",
        "tool_call",
        "tool_result",
        "llm_call",
        "final_answer",
    ]

    stmt = select(RunStep).where(RunStep.step_type == "tool_result")
    result_step = (await db.execute(stmt)).scalar_one()
    assert result_step.payload["result"] == "42"


async def test_parallel_tool_calls_all_get_dispatched(db, redis, memory, mock_llm, scenario):
    _user, session, run = scenario
    mock_llm.append(
        llm_response(
            tool_calls=[
                fake_tool_call("calculator", '{"expression": "1+1"}', "call_a"),
                fake_tool_call("web_search", '{"query": "Dubai"}', "call_b"),
            ]
        )
    )
    mock_llm.append(llm_response(content="done"))

    await AgentRunner(db, redis, memory).run(run, session)

    types = await step_types(db, run.id)
    assert types.count("tool_call") == 2
    assert types.count("tool_result") == 2


async def test_hallucinated_tool_name_does_not_crash_the_loop(db, redis, memory, mock_llm, scenario):
    _user, session, run = scenario
    mock_llm.append(llm_response(tool_calls=[fake_tool_call("teleport", "{}")]))
    mock_llm.append(llm_response(content="Sorry, I cannot do that."))

    answer = await AgentRunner(db, redis, memory).run(run, session)

    assert answer == "Sorry, I cannot do that."
    stmt = select(RunStep).where(RunStep.step_type == "tool_result")
    result_step = (await db.execute(stmt)).scalar_one()
    assert result_step.payload["result"].startswith("Error: unknown tool")
    assert run.status is RunStatus.COMPLETED


# ------------------------------- T-4 --------------------------------------- #
async def test_iteration_cap_terminates_with_the_sentinel(db, redis, memory, mock_llm, scenario):
    _user, session, run = scenario
    for _ in range(settings.MAX_AGENT_ITERATIONS + 2):
        mock_llm.append(llm_response(tool_calls=[fake_tool_call("get_current_datetime", "{}")]))

    answer = await AgentRunner(db, redis, memory).run(run, session)

    assert answer == MAX_ITERATIONS_ANSWER
    types = await step_types(db, run.id)
    assert types.count("llm_call") == settings.MAX_AGENT_ITERATIONS
    assert "final_answer" not in types      # the cap is a failure, not an answer
    # exactly two responses left over: the loop stopped at the cap, not at the queue
    assert len(mock_llm) == 2


# ----------------------- persist == publish invariant ---------------------- #
async def published_types(redis, run_id: str) -> list[str]:
    out = []
    async for event in read_events(redis, run_id, block_ms=1):
        if event is None:
            break
        out.append(event.data["step_type"])
    return out


DELTA_TYPES = {"content_delta", "tool_call_delta"}


async def test_every_recorded_step_is_also_published(db, redis, memory, mock_llm, scenario):
    _user, session, run = scenario
    mock_llm.append(
        llm_response(tool_calls=[fake_tool_call("calculator", '{"expression": "2*21"}')])
    )
    mock_llm.append(llm_response(content="42"))

    await AgentRunner(db, redis, memory).run(run, session)

    published = await published_types(redis, run.id)
    steps = [t for t in published if t not in DELTA_TYPES]
    persisted = await step_types(db, run.id)

    # `done` and the generation fragments are transport-only, by design.
    assert steps == persisted + ["done"]


# ------------------------------ streaming ---------------------------------- #
async def test_generation_fragments_stream_but_are_never_persisted(
    db, redis, memory, mock_llm, scenario
):
    _user, session, run = scenario
    mock_llm.append(
        llm_response(tool_calls=[fake_tool_call("calculator", '{"expression": "2*21"}')])
    )
    mock_llm.append(llm_response(content="The answer is forty-two, comfortably."))

    await AgentRunner(db, redis, memory).run(run, session)

    published = await published_types(redis, run.id)
    assert "content_delta" in published
    assert "tool_call_delta" in published
    # the trace stays a trace: fragments are not steps
    assert DELTA_TYPES.isdisjoint(await step_types(db, run.id))


async def test_streamed_fragments_reassemble_into_the_final_answer(
    db, redis, memory, mock_llm, scenario
):
    _user, session, run = scenario
    answer = "Fifteen percent of that population is about 568,050."
    mock_llm.append(llm_response(content=answer))

    await AgentRunner(db, redis, memory).run(run, session)

    streamed = []
    async for event in read_events(redis, run.id, block_ms=1):
        if event is None:
            break
        if event.data["step_type"] == "content_delta":
            streamed.append(event.data["payload"]["text"])

    assert "".join(streamed) == answer          # nothing dropped, nothing duplicated
    assert len(streamed) > 1                    # and it genuinely arrived in pieces


async def test_tool_call_fragments_carry_the_tool_name_and_index(
    db, redis, memory, mock_llm, scenario
):
    _user, session, run = scenario
    arguments = '{"query": "current population of Dubai"}'
    mock_llm.append(llm_response(tool_calls=[fake_tool_call("web_search", arguments)]))
    mock_llm.append(llm_response(content="done"))

    await AgentRunner(db, redis, memory).run(run, session)

    fragments = []
    async for event in read_events(redis, run.id, block_ms=1):
        if event is None:
            break
        if event.data["step_type"] == "tool_call_delta":
            fragments.append(event.data["payload"])

    assert "".join(f["text"] for f in fragments) == arguments
    assert {f["name"] for f in fragments} == {"web_search"}
    assert {f["index"] for f in fragments} == {0}


async def test_runner_asks_the_planner_to_stream(db, redis, memory, mock_llm, scenario):
    _user, session, run = scenario
    mock_llm.append(llm_response(content="hi"))

    await AgentRunner(db, redis, memory).run(run, session)

    assert mock_llm.calls[0]["streamed"] is True


# ---------------------------- reasoning effort ------------------------------ #
async def test_reasoning_tokens_accumulate_across_iterations(
    db, redis, memory, mock_llm, scenario
):
    _user, session, run = scenario
    mock_llm.append(
        llm_response(
            tool_calls=[fake_tool_call("calculator", '{"expression": "2*21"}')],
            reasoning_tokens=300,
        )
    )
    mock_llm.append(llm_response(content="42", reasoning_tokens=120))

    await AgentRunner(db, redis, memory).run(run, session)

    assert run.reasoning_tokens == 420
    # and each llm_call step carries its own share, so the cost is attributable
    stmt = select(RunStep).where(RunStep.step_type == "llm_call").order_by(RunStep.occurred_at)
    per_step = [s.payload["reasoning_tokens"] for s in (await db.execute(stmt)).scalars()]
    assert per_step == [300, 120]


# ------------------------------ ReAct prompt -------------------------------- #
async def test_react_instructions_precede_the_session_persona(
    db, redis, memory, mock_llm, scenario
):
    _user, session, run = scenario
    mock_llm.append(llm_response(content="hi"))

    await AgentRunner(db, redis, memory).run(run, session)

    system = mock_llm.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert "You are a ReAct agent" in system["content"]
    # the session's own prompt goes last, so it is the most recent thing read
    assert system["content"].endswith(session.system_prompt)


# ------------------ the ReAct approach ported from examples/ ---------------- #
async def test_tool_call_step_carries_the_models_own_reasoning(
    db, redis, memory, mock_llm, scenario
):
    """The thought is a field of the action, so every step of the durable trace
    explains itself — no prose to parse, nothing reconstructed after the fact."""
    _user, session, run = scenario
    arguments = (
        '{"thought": "I have 3,787,000 and need 15% of it.", '
        '"expression": "0.15 * 3787000", "grounding": "tool_result"}'
    )
    mock_llm.append(llm_response(tool_calls=[fake_tool_call("calculator", arguments)]))
    mock_llm.append(llm_response(content="568050"))

    await AgentRunner(db, redis, memory).run(run, session)

    stmt = select(RunStep).where(RunStep.step_type == "tool_call")
    step = (await db.execute(stmt)).scalar_one()
    assert step.payload["thought"] == "I have 3,787,000 and need 15% of it."
    assert step.payload["grounding"] == "tool_result"
    assert step.payload["name"] == "calculator"


async def test_invented_values_are_refused_and_the_loop_continues(
    db, redis, memory, mock_llm, scenario
):
    _user, session, run = scenario
    mock_llm.append(
        llm_response(
            tool_calls=[
                fake_tool_call(
                    "calculator",
                    '{"thought": "summing 5, 10 and 15", "expression": "5 + 10 + 15",'
                    ' "grounding": "assumed"}',
                )
            ]
        )
    )
    mock_llm.append(llm_response(content="No numbers were given. Which three should I add?"))

    await AgentRunner(db, redis, memory).run(run, session)

    stmt = select(RunStep).where(RunStep.step_type == "tool_result")
    result = (await db.execute(stmt)).scalar_one()
    assert result.payload["result"].startswith("Error: you marked these arguments as assumed")
    # the refusal is an observation, so the run still finishes cleanly
    assert run.status is RunStatus.NEEDS_INPUT


async def test_a_question_finishes_as_needs_input_not_completed(
    db, redis, memory, mock_llm, scenario
):
    """A worker has nobody to ask, so the equivalent of forcing `ask_user` is to
    make the need visible rather than let a question pass as an answer."""
    _user, session, run = scenario
    mock_llm.append(llm_response(content="Which three numbers would you like me to add?"))

    await AgentRunner(db, redis, memory).run(run, session)

    assert run.status is RunStatus.NEEDS_INPUT
    types = await step_types(db, run.id)
    assert types == ["llm_call", "needs_input", "final_answer"]

    stmt = select(RunStep).where(RunStep.step_type == "needs_input")
    step = (await db.execute(stmt)).scalar_one()
    assert step.payload["question"] == "Which three numbers would you like me to add?"


async def test_a_real_answer_still_completes(db, redis, memory, mock_llm, scenario):
    _user, session, run = scenario
    mock_llm.append(llm_response(content="15% of 3,787,000 is 568,050."))

    await AgentRunner(db, redis, memory).run(run, session)

    assert run.status is RunStatus.COMPLETED
    assert "needs_input" not in await step_types(db, run.id)

"""C-12 acceptance — the event transport."""
from app.agent.events import (
    BEGINNING,
    publish_event,
    read_events,
    stream_exists,
    stream_key,
)


async def collect(redis, run_id, last_id=BEGINNING):
    out = []
    async for event in read_events(redis, run_id, last_id=last_id, block_ms=1):
        if event is None:
            break
        out.append(event)
    return out


async def test_published_event_is_readable_from_the_beginning(redis):
    await publish_event(redis, "r1", {"step_type": "llm_call", "payload": {"iteration": 0}})

    events = await collect(redis, "r1")

    assert len(events) == 1
    assert events[0].data["step_type"] == "llm_call"


async def test_advancing_cursor_yields_each_event_exactly_once(redis):
    for i in range(3):
        await publish_event(redis, "r1", {"step_type": "tool_call", "payload": {"i": i}})

    events = await collect(redis, "r1")
    assert [e.data["payload"]["i"] for e in events] == [0, 1, 2]

    # resuming from the last seen id returns nothing new — no duplicates
    assert await collect(redis, "r1", last_id=events[-1].id) == []


async def test_two_readers_each_receive_the_full_sequence(redis):
    for i in range(3):
        await publish_event(redis, "r1", {"step_type": "tool_call", "payload": {"i": i}})

    # Plain XREAD broadcasts; XREADGROUP would split the entries between them.
    first = await collect(redis, "r1")
    second = await collect(redis, "r1")
    assert [e.id for e in first] == [e.id for e in second]


async def test_empty_stream_yields_a_heartbeat_rather_than_raising(redis):
    ticks = 0
    async for event in read_events(redis, "nothing-here", block_ms=1):
        assert event is None
        ticks += 1
        if ticks == 2:
            break
    assert ticks == 2


async def test_ttl_is_refreshed_by_a_later_publish(redis):
    await publish_event(redis, "r1", {"step_type": "llm_call", "payload": {}})
    await redis.expire(stream_key("r1"), 5)

    await publish_event(redis, "r1", {"step_type": "final_answer", "payload": {}})

    assert await redis.ttl(stream_key("r1")) > 5
    assert await stream_exists(redis, "r1")
    assert not await stream_exists(redis, "never-published")


# --------------------------- reasoning / streaming -------------------------- #
def test_reasoning_effort_is_sent_only_to_reasoning_models(monkeypatch):
    from app.agent import planner
    from app.core.config import settings

    monkeypatch.setattr(settings, "REASONING_EFFORT", "high")

    monkeypatch.setattr(settings, "CHAT_MODEL", "gpt-5-mini")
    assert planner.request_kwargs([], None)["reasoning_effort"] == "high"

    # gpt-4o-mini would 400 on the parameter, so it is dropped rather than sent
    monkeypatch.setattr(settings, "CHAT_MODEL", "gpt-4o-mini")
    assert "reasoning_effort" not in planner.request_kwargs([], None)

    monkeypatch.setattr(settings, "CHAT_MODEL", "gpt-5-mini")
    monkeypatch.setattr(settings, "REASONING_EFFORT", None)
    assert "reasoning_effort" not in planner.request_kwargs([], None)


def test_tool_choice_is_omitted_when_there_are_no_tools():
    from app.agent import planner

    assert "tool_choice" not in planner.request_kwargs([], None)
    assert planner.request_kwargs([], [{"type": "function"}])["tool_choice"] == "auto"


def test_reasoning_tokens_reads_the_usage_detail():
    from types import SimpleNamespace

    from app.agent import planner

    response = SimpleNamespace(
        usage=SimpleNamespace(completion_tokens_details=SimpleNamespace(reasoning_tokens=512))
    )
    assert planner.reasoning_tokens(response) == 512
    assert planner.reasoning_tokens(SimpleNamespace(usage=None)) == 0
    assert planner.reasoning_tokens(SimpleNamespace()) == 0

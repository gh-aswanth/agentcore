"""C-15 — fixtures.

Exactly one thing is mocked: ``app.agent.planner.chat``. Because it is the sole
LLM seam, patching it takes the whole system offline — the rubric's explicit
failure condition is "real LLM calls in tests".
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

import fakeredis.aioredis  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

from app.agent import planner  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.models.agent import AgentRun, AgentSession, RunStep  # noqa: E402
from app.models.user import User  # noqa: E402

pytest_plugins = ("pytest_asyncio",)

# long_term_memory is deliberately excluded: its Vector(1536) column is a
# Postgres type with no SQLite equivalent. Vector behaviour is exercised
# against Postgres in the compose stack; the offline suite covers everything
# else.
OFFLINE_TABLES = [
    User.__table__,
    AgentSession.__table__,
    AgentRun.__table__,
    RunStep.__table__,
]


@pytest.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=OFFLINE_TABLES)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db(engine) -> AsyncSession:
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


# --------------------------------------------------------------------------- #
# LLM mocking
# --------------------------------------------------------------------------- #
def fake_tool_call(name: str, arguments: str, call_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id or f"call_{name}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def llm_response(
    content=None, tool_calls=None, tokens: int = 100, reasoning_tokens: int = 0
) -> SimpleNamespace:
    """An object shaped like an OpenAI ChatCompletion, including the
    ``model_dump`` the runner calls when appending the assistant turn."""
    tool_calls = tool_calls or []

    def model_dump(**_kwargs):
        dumped: dict = {"role": "assistant", "content": content}
        if tool_calls:
            dumped["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return {k: v for k, v in dumped.items() if v is not None}

    message = SimpleNamespace(
        role="assistant", content=content, tool_calls=tool_calls, model_dump=model_dump
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(
            total_tokens=tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
    )


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch the single planner seam. Append scripted responses to the returned
    list; the loop pops them in order.

    When the runner supplies an ``on_delta`` handler the fake replays the
    scripted response as fragments first, exactly as the streaming path would,
    so the delta pipeline is exercised without a server.
    """
    class ScriptedResponses(list):
        """A response queue that also records how the loop called the seam."""

        calls: list[dict] = []

    responses = ScriptedResponses()
    calls = responses.calls = []

    async def fake_chat(messages, tools=None, on_delta=None):
        assert responses, "agent loop requested more LLM responses than were scripted"
        calls.append({"messages": messages, "tools": tools, "streamed": on_delta is not None})
        response = responses.pop(0)

        if on_delta is not None:
            message = response.choices[0].message
            for fragment in _fragments(message.content or ""):
                await on_delta(planner.Delta(kind="content", text=fragment))
            for index, tool_call in enumerate(message.tool_calls or []):
                for fragment in _fragments(tool_call.function.arguments or ""):
                    await on_delta(
                        planner.Delta(
                            kind="tool_call",
                            text=fragment,
                            index=index,
                            name=tool_call.function.name,
                        )
                    )
        return response

    monkeypatch.setattr("app.agent.planner.chat", fake_chat)
    return responses


def _fragments(text: str, size: int = 5) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


class StubMemory:
    """Long-term memory needs pgvector, so the runner tests use a stub that
    honours the same four-method interface. Short-term is exercised for real
    against fakeredis in test_memory.py."""

    def __init__(self):
        self.turns: list[tuple[str, str, str]] = []
        self.facts: list[tuple[str, str]] = []
        self.recall: list = []

    async def search_long_term(self, user_id, query, limit=3):
        return self.recall

    async def read_short_term(self, session_id, limit=10):
        return [{"user": u, "assistant": a} for sid, u, a in self.turns if sid == session_id]

    async def append_turn(self, session_id, user_msg, answer):
        self.turns.append((session_id, user_msg, answer))

    async def write_long_term(self, user_id, content, source_run_id=None):
        self.facts.append((user_id, content))


@pytest.fixture
def memory():
    return StubMemory()

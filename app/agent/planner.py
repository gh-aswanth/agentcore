"""C-07 — the single seam through which every OpenAI call passes.

Its whole purpose is testability: monkeypatching ``app.agent.planner.chat``
takes the entire agent offline, which is what makes the suite runnable with no
network. No other module imports the OpenAI SDK for chat completions.

Streaming lives *behind* that seam rather than beside it. ``chat`` given an
``on_delta`` callback streams internally and reassembles a normal
``ChatCompletion`` before returning, so the runner's message handling is
identical either way and the test suite still has exactly one thing to mock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from app.core.config import settings

# Chat Completions accepts `reasoning_effort` only for reasoning models; sending
# it to gpt-4o-mini is a 400. Capability is a property of the model, so it is
# checked here rather than left to whoever edits .env.
REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    """Lazily built so importing this module never requires a live key."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def supports_reasoning(model: str) -> bool:
    return model.startswith(REASONING_MODEL_PREFIXES)


@dataclass(frozen=True)
class Delta:
    """One fragment of a response as it is generated.

    ``kind`` is ``"content"`` for assistant prose or ``"tool_call"`` for the
    arguments of a tool call being composed — watching a tool's arguments fill in
    is the closest thing the Chat Completions API offers to watching the model
    decide.
    """

    kind: str
    text: str
    index: int = 0
    name: str = ""


DeltaHandler = Callable[[Delta], Awaitable[None]]


def request_kwargs(messages: list[dict], tools: list[dict] | None) -> dict:
    kwargs: dict = {"model": settings.CHAT_MODEL, "messages": messages}
    if tools:
        # tool_choice="auto" alongside an empty tools list is an API error.
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if settings.REASONING_EFFORT and supports_reasoning(settings.CHAT_MODEL):
        kwargs["reasoning_effort"] = settings.REASONING_EFFORT
    return kwargs


async def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    on_delta: DeltaHandler | None = None,
) -> ChatCompletion:
    """Return the raw SDK response — the runner needs the full message object to
    append to the transcript, so this must not pre-digest it."""
    kwargs = request_kwargs(messages, tools)
    if on_delta is None or not settings.STREAM_TOKENS:
        return await client().chat.completions.create(**kwargs)
    return await _streamed_chat(kwargs, on_delta)


async def _streamed_chat(kwargs: dict, on_delta: DeltaHandler) -> ChatCompletion:
    """Consume a streamed completion, publishing fragments as they arrive, and
    reassemble the ``ChatCompletion`` the non-streaming path would have returned."""
    stream = await client().chat.completions.create(
        **kwargs,
        stream=True,
        # Without this the usage block never arrives and token accounting — the
        # only visibility into reasoning cost — is silently zero.
        stream_options={"include_usage": True},
    )

    content: list[str] = []
    # tool_call fragments arrive keyed by index, not id, and out of order
    calls: dict[int, dict[str, str]] = {}
    usage = None
    finish_reason = None
    completion_id = ""
    created = 0
    model = kwargs["model"]

    async for chunk in stream:
        completion_id = chunk.id or completion_id
        created = chunk.created or created
        model = chunk.model or model
        if chunk.usage is not None:
            usage = chunk.usage           # arrives in its own final, choice-less chunk
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        finish_reason = choice.finish_reason or finish_reason
        delta = choice.delta
        if delta is None:
            continue

        if delta.content:
            content.append(delta.content)
            await on_delta(Delta(kind="content", text=delta.content))

        for fragment in delta.tool_calls or []:
            slot = calls.setdefault(fragment.index, {"id": "", "name": "", "arguments": ""})
            if fragment.id:
                slot["id"] = fragment.id
            if fragment.function is None:
                continue
            if fragment.function.name:
                slot["name"] += fragment.function.name
            if fragment.function.arguments:
                slot["arguments"] += fragment.function.arguments
                await on_delta(
                    Delta(
                        kind="tool_call",
                        text=fragment.function.arguments,
                        index=fragment.index,
                        name=slot["name"],
                    )
                )

    tool_calls = [
        ChatCompletionMessageToolCall(
            id=slot["id"] or f"call_{index}",
            type="function",
            function=Function(name=slot["name"], arguments=slot["arguments"]),
        )
        for index, slot in sorted(calls.items())
    ]

    message = ChatCompletionMessage(
        role="assistant",
        content="".join(content) or None,
        tool_calls=tool_calls or None,
    )
    return ChatCompletion(
        id=completion_id or "chatcmpl-streamed",
        object="chat.completion",
        created=created,
        model=model,
        usage=usage,
        choices=[
            Choice(
                index=0,
                message=message,
                finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
            )
        ],
    )


def reasoning_tokens(response) -> int:
    """Reasoning tokens billed for one call, or 0 for a non-reasoning model."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", 0) or 0


async def simple_completion(prompt: str) -> str:
    response = await chat([{"role": "user", "content": prompt}])
    return response.choices[0].message.content or ""


async def embed(text: str) -> list[float]:
    response = await client().embeddings.create(model=settings.EMBEDDING_MODEL, input=text)
    return response.data[0].embedding

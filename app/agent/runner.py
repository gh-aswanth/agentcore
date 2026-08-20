"""Model Runner
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import events, planner
from app.agent.memory_manager import MemoryManager
from app.agent.prompts import build_system_prompt
from app.agent.tools import (
    TOOL_REGISTRY,
    ToolContext,
    dispatch,
    is_timeout,
    parse_arguments,
)
from app.core.config import settings
from app.core.logging import bind, get_logger
from app.models.agent import AgentRun, AgentSession, RunStatus, RunStep

logger = get_logger(__name__)

MAX_ITERATIONS_ANSWER = "Max iterations reached"
CANCELLED_ANSWER = "Run cancelled"
CANCEL_KEY = "run:{run_id}:cancel"
TOOL_RESULT_TRUNCATE = 2000


def _is_a_question(text: str) -> bool:
    """A final answer ending in a question mark is the agent asking for
    something it was not given, not answering."""
    return text.strip().endswith("?")


class _DeltaStream:
    """Publishes generation fragments, coalesced.

    One Redis write per token would be roughly twenty times the traffic for no
    visible gain, so fragments accumulate per (kind, index) and flush once they
    are worth sending or when the message ends. Fragments are transport-only:
    they are never persisted, because the assembled message they compose is what
    becomes the `llm_call` / `tool_call` step.
    """

    def __init__(self, publish, iteration: int, min_chars: int):
        self._publish = publish
        self._iteration = iteration
        self._min_chars = min_chars
        self._buffers: dict[tuple[str, int], dict] = {}

    async def __call__(self, delta: planner.Delta) -> None:
        key = (delta.kind, delta.index)
        buffer = self._buffers.setdefault(key, {"text": "", "name": delta.name})
        buffer["text"] += delta.text
        buffer["name"] = delta.name or buffer["name"]
        if len(buffer["text"]) >= self._min_chars:
            await self._flush(key)

    async def flush(self) -> None:
        for key in list(self._buffers):
            await self._flush(key)

    async def _flush(self, key: tuple[str, int]) -> None:
        kind, index = key
        buffer = self._buffers.pop(key, None)
        if not buffer or not buffer["text"]:
            return
        payload = {"iteration": self._iteration, "text": buffer["text"]}
        if kind == "tool_call":
            payload["index"] = index
            payload["name"] = buffer["name"]
        await self._publish({"step_type": f"{kind}_delta", "payload": payload})


class AgentRunner:
    def __init__(self, db: AsyncSession, redis, memory: MemoryManager):
        self.db = db
        self.redis = redis
        self.memory = memory

    async def _record(self, run_id: str, step_type: str, payload: dict) -> None:
        """Coupling the durable write and the live publish in a single method is
        a correctness decision, not tidiness: it makes it structurally
        impossible for the Postgres trace and the SSE stream to diverge."""
        self.db.add(RunStep(run_id=run_id, step_type=step_type, payload=payload))
        await self.db.commit()
        await events.publish_event(
            self.redis, run_id, {"step_type": step_type, "payload": payload}
        )

        bind(step_type=step_type)
        await logger.ainfo("run_step", **{k: v for k, v in payload.items() if k != "result"})

    async def _publish(self, run_id: str, event: dict) -> None:
        """Transport-only. Used for generation fragments and `done` — signals
        that are worth watching live but are not steps in the trace."""
        await events.publish_event(self.redis, run_id, event)

    async def _publish_done(self, run_id: str, status: str) -> None:
        """`done` is a stream-control signal rather than an agent action, so it is
        the only frame not persisted as a RunStep."""
        await self._publish(run_id, {"step_type": "done", "status": status})

    async def _is_cancelled(self, run_id: str) -> bool:
        return bool(await self.redis.exists(CANCEL_KEY.format(run_id=run_id)))


    async def _build_messages(self, run: AgentRun, session: AgentSession) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(session.system_prompt)}
        ]

        recalled = await self.memory.search_long_term(
            session.user_id, run.user_message, limit=settings.LONG_TERM_TOP_K
        )
        if recalled:
            facts = "\n".join(f"- {m.content}" for m in recalled)
            messages.append(
                {
                    "role": "system",
                    "content": f"Relevant facts you remember about this user:\n{facts}",
                }
            )

        for turn in await self.memory.read_short_term(session.id, limit=settings.SHORT_TERM_WINDOW):
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        messages.append({"role": "user", "content": run.user_message})
        return messages


    async def run(self, run: AgentRun, session: AgentSession) -> str:
        bind(run_id=run.id, session_id=session.id, user_id=session.user_id)
        await logger.ainfo("run_started", tools=list(session.tools_enabled or []))

        messages = await self._build_messages(run, session)

        tools = [
            TOOL_REGISTRY[name].openai_schema()
            for name in (session.tools_enabled or [])
            if name in TOOL_REGISTRY
        ]
        ctx = ToolContext(
            user_id=session.user_id, run_id=run.id, db=self.db, memory=self.memory
        )

        final_answer = MAX_ITERATIONS_ANSWER
        status = RunStatus.COMPLETED

        for iteration in range(settings.MAX_AGENT_ITERATIONS):
            if await self._is_cancelled(run.id):
                final_answer = CANCELLED_ANSWER
                status = RunStatus.CANCELLED
                break

            deltas = _DeltaStream(
                lambda event: self._publish(run.id, event),
                iteration=iteration,
                min_chars=settings.STREAM_DELTA_CHARS,
            )
            response = await planner.chat(messages, tools=tools or None, on_delta=deltas)
            await deltas.flush()

            message = response.choices[0].message
            usage = getattr(response, "usage", None)
            run.tokens_used = (run.tokens_used or 0) + (getattr(usage, "total_tokens", 0) or 0)
            run.reasoning_tokens = (run.reasoning_tokens or 0) + planner.reasoning_tokens(response)

            tool_calls = list(message.tool_calls or [])
            await self._record(
                run.id,
                "llm_call",
                {
                    "iteration": iteration,
                    "content": message.content,
                    "tool_calls": [tc.function.name for tc in tool_calls],
                    "reasoning_tokens": planner.reasoning_tokens(response),
                },
            )

            if not tool_calls:
                final_answer = message.content or ""
                if _is_a_question(final_answer):
                    await self._record(
                        run.id, "needs_input", {"question": final_answer.strip()}
                    )
                    status = RunStatus.NEEDS_INPUT

                await self._record(run.id, "final_answer", {"content": final_answer})
                break


            messages.append(message.model_dump(exclude_none=True))

            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments

                parsed = parse_arguments(arguments)
                await self._record(
                    run.id,
                    "tool_call",
                    {
                        "name": name,
                        "arguments": arguments,
                        "thought": parsed.thought,
                        "grounding": parsed.grounding,
                    },
                )
                result = await dispatch(name, arguments, ctx)

                await self._record(
                    run.id,
                    "tool_timeout" if is_timeout(result) else "tool_result",
                    {
                        "name": name,
                        "result": result[:TOOL_RESULT_TRUNCATE],
                        **({"timeout_seconds": settings.TOOL_TIMEOUT_SECONDS}
                           if is_timeout(result) else {}),
                    },
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        # ------------------ finalisation ------------------ #
        run.final_answer = final_answer
        run.status = status
        run.completed_at = datetime.now(timezone.utc)
        await self.db.commit()

        if status in (RunStatus.COMPLETED, RunStatus.NEEDS_INPUT):
            await self.memory.append_turn(session.id, run.user_message, final_answer)
        await self._publish_done(run.id, status.value)

        await logger.ainfo(
            "run_finished",
            status=status.value,
            tokens_used=run.tokens_used,
            reasoning_tokens=run.reasoning_tokens,
        )
        return final_answer

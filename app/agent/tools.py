"""Tool Registry
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

from app.core.config import settings

if TYPE_CHECKING:                       # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.agent.memory_manager import MemoryManager


@dataclass
class ToolContext:
    """Runtime handles a tool may need. Filtered out of the generated schema."""

    user_id: str
    run_id: str
    db: "AsyncSession"
    memory: "MemoryManager"


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, Any]

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


TOOL_REGISTRY: dict[str, Tool] = {}

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_INJECTED = {"ctx", "user_id", "run_id"}

THOUGHT_DESCRIPTION = (
    "Your reasoning for making THIS call, in one sentence: what you still need, "
    "and how calling this tool with these arguments gets it."
)

GROUNDING_DESCRIPTION = (
    "Where the concrete values in these arguments came from. "
    "'user' = the user stated them in this conversation. "
    "'tool_result' = an earlier tool result returned them. "
    "'assumed' = you chose them yourself. "
    "If ANY value here was not stated by the user and not returned by a tool, "
    "you must answer 'assumed'."
)
GROUNDING_VALUES = ["user", "tool_result", "assumed"]
ASSUMED = "assumed"

THOUGHT = "thought"
GROUNDING = "grounding"
REASONING_PARAMS = (THOUGHT, GROUNDING)


def _json_type(py_type: Any) -> str:
    origin = getattr(py_type, "__origin__", None)
    return _JSON_TYPES.get(origin or py_type, "string")


def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register a function as an agent tool.

    Name comes from ``__name__``, description from the docstring, parameter
    schema from the annotated signature, and ``required`` from which parameters
    lack a default.
    """
    try:
        hints = get_type_hints(fn)
    except Exception:                                   # pragma: no cover
        hints = dict(getattr(fn, "__annotations__", {}))

    # `thought` goes first so the reasoning is generated before the arguments it
    # justifies.
    props: dict[str, dict] = {THOUGHT: {"type": "string", "description": THOUGHT_DESCRIPTION}}
    required: list[str] = [THOUGHT]
    carries_values = False

    for pname, param in inspect.signature(fn).parameters.items():
        if pname in _INJECTED:
            continue
        carries_values = True
        schema: dict[str, Any] = {"type": _json_type(hints.get(pname, str))}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            schema["default"] = param.default
        props[pname] = schema

    if carries_values:
        props[GROUNDING] = {
            "type": "string",
            "enum": GROUNDING_VALUES,
            "description": GROUNDING_DESCRIPTION,
        }
        required.append(GROUNDING)

    TOOL_REGISTRY[fn.__name__] = Tool(
        name=fn.__name__,
        description=inspect.getdoc(fn) or "",
        fn=fn,
        parameters={"type": "object", "properties": props, "required": required},
    )
    return fn


# --------------------------------------------------------------------------- #
# The tools
# --------------------------------------------------------------------------- #
@tool
def web_search(query: str) -> str:
    """Search the web and return the top 3 result snippets as JSON."""
    return json.dumps(
        [
            {"title": f"{query} — Overview", "snippet": f"General background on {query}."},
            {
                "title": f"{query} — Statistics",
                "snippet": f"As of 2026, {query} is estimated at 3,787,000.",
            },
            {"title": f"{query} — Analysis", "snippet": f"Recent trends affecting {query}."},
        ]
    )


_ALLOWED_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursive walk over a whitelisted node set. Anything else raises."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '0.15 * 3787000'."""
    return str(_safe_eval(ast.parse(expression, mode="eval").body))


@tool
def get_current_datetime() -> str:
    """Return the current UTC time as an ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


@tool
async def summarise_text(text: str, max_words: int = 50) -> str:
    """Summarise the given text down to at most max_words words."""
    from app.agent import planner   # local import keeps the planner seam mockable

    return await planner.simple_completion(
        f"Summarise the following in at most {max_words} words:\n\n{text}"
    )


@tool
async def remember_fact(fact: str, ctx: ToolContext | None = None) -> str:
    """Store a fact in the user's long-term memory so it can be recalled later."""
    if ctx is None:                                     # pragma: no cover
        return "Error: remember_fact requires run context."
    await ctx.memory.write_long_term(
        user_id=ctx.user_id, content=fact, source_run_id=ctx.run_id
    )
    return f"Stored: {fact}"


TIMED_OUT_MARKER = "timed out after"


def timeout_message(name: str) -> str:
    return f"Error: tool '{name}' timed out after {settings.TOOL_TIMEOUT_SECONDS}s."


def is_timeout(result: str) -> bool:
    return result.startswith("Error: tool '") and TIMED_OUT_MARKER in result


@dataclass(frozen=True)
class ParsedArguments:
    """A tool call split into its reasoning and its actual arguments."""

    thought: str = ""
    grounding: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def parse_arguments(arguments: str) -> ParsedArguments:
    """Never raises. The runner calls this to record the thought on the
    `tool_call` step *before* the tool runs, so the trace shows the reasoning
    ahead of its consequence; `dispatch` calls it again for the same call. The
    double parse of a sub-kilobyte JSON string is not worth widening the
    dispatcher signature the plan specifies."""
    try:
        raw = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        return ParsedArguments(error=f"Error: arguments were not valid JSON ({e}).")
    if not isinstance(raw, dict):
        return ParsedArguments(
            error=f"Error: arguments must be a JSON object, got {type(raw).__name__}."
        )

    kwargs = dict(raw)
    thought = str(kwargs.pop(THOUGHT, "") or "").strip()
    grounding = str(kwargs.pop(GROUNDING, "") or "").strip().lower()
    return ParsedArguments(thought=thought, grounding=grounding, kwargs=kwargs)


def _refuse_if_assumed(parsed: ParsedArguments) -> str | None:
    """The gate. A self-reported invention is turned back into an observation
    rather than executed, so the loop corrects instead of compounding the guess."""
    if parsed.grounding != ASSUMED:
        return None
    return (
        "Error: you marked these arguments as assumed, which means you invented "
        "values that were never given to you. Do not proceed on invented values. "
        "State which value you are missing and stop, or obtain it from a tool."
    )


async def dispatch(name: str, arguments: str, ctx: ToolContext | None = None) -> str:
    """Execute a tool by name. Every failure mode returns a string the LLM reads
    as a normal observation and can correct against on the next iteration."""
    t = TOOL_REGISTRY.get(name)
    if t is None:
        return (
            f"Error: unknown tool '{name}'. "
            f"Available tools: {', '.join(sorted(TOOL_REGISTRY))}"
        )

    parsed = parse_arguments(arguments)
    if parsed.error:
        return parsed.error
    refusal = _refuse_if_assumed(parsed)
    if refusal:
        return refusal

    kwargs = dict(parsed.kwargs)
    if "ctx" in inspect.signature(t.fn).parameters:
        kwargs["ctx"] = ctx

    try:
        if inspect.iscoroutinefunction(t.fn):
            pending = t.fn(**kwargs)
        else:
            pending = asyncio.to_thread(t.fn, **kwargs)

        result = await asyncio.wait_for(pending, timeout=settings.TOOL_TIMEOUT_SECONDS)
        if inspect.isawaitable(result):
            result = await asyncio.wait_for(result, timeout=settings.TOOL_TIMEOUT_SECONDS)
        return str(result)
    except asyncio.TimeoutError:
        return timeout_message(name)
    except TypeError as e:
        return f"Error: bad arguments for '{name}': {e}"
    except Exception as e:
        return f"Error executing '{name}': {type(e).__name__}: {e}"

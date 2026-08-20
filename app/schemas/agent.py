"""C-11.3 — request/response models.

Tool-allowlist validation lives in the type, not the route body, so it costs
zero lines of route code and shows up in the generated OpenAPI schema.
"""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent.tools import TOOL_REGISTRY
from app.models.agent import RunStatus

# The allowlist *is* the type. Built from the registry at import time, so adding
# a tool is still just writing the decorated function — the literal set, the 422,
# and the enum in /docs all follow from it with nothing to edit by hand.
ToolName = Literal[tuple(sorted(TOOL_REGISTRY))]  # type: ignore[valid-type]


class AgentSessionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    system_prompt: str = ""
    tools_enabled: list[ToolName] = Field(
        default_factory=list,
        description="Tools this session's agent may call. Must be registry names.",
    )


class AgentSessionOut(BaseModel):
    id: str
    name: str
    system_prompt: str
    # Deliberately `str`, not ToolName: strict on the way in, lenient on the way
    # out. A session stored before a tool was renamed or removed must still be
    # readable — the runner already filters unknown names out of the request it
    # builds, so a stale entry degrades instead of making the row unfetchable.
    tools_enabled: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class RunSummary(BaseModel):
    id: str
    status: RunStatus
    user_message: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentSessionDetail(AgentSessionOut):
    runs: list[RunSummary] = Field(default_factory=list)


class RunRequest(BaseModel):
    message: str = Field(min_length=1)


class RunResponse(BaseModel):
    run_id: str
    status: RunStatus


class RunStatusOut(BaseModel):
    run_id: str
    status: RunStatus
    tokens_used: int
    reasoning_tokens: int
    step_count: int
    final_answer: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class StepOut(BaseModel):
    id: str
    step_type: str
    payload: dict[str, Any]
    occurred_at: datetime

    model_config = {"from_attributes": True}


class MemoryOut(BaseModel):
    id: str
    content: str
    source_run_id: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}

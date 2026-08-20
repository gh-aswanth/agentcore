"""AgentSession, AgentRun, RunStep, RunStatus."""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.user import new_id, utcnow

JsonType = JSON().with_variant(JSONB(), "postgresql")


class RunStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = (
    RunStatus.COMPLETED,
    RunStatus.NEEDS_INPUT,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
)

# values_callable persists the *value* ('queued'), not the member name ('QUEUED').
RunStatusType = Enum(
    RunStatus,
    name="run_status",
    values_callable=lambda e: [m.value for m in e],
)


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    tools_enabled: Mapped[list[str]] = mapped_column(JsonType, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # lazy="raise" rather than an eager loader: a session's runs are wanted by
    # exactly one endpoint, and eager-loading them there dragged every run's
    # entire trace into every query that touched a session. Endpoints that want
    # runs ask for them explicitly. passive_deletes hands the cascade to the
    # ON DELETE CASCADE on the FK, so DELETE does not need the collection either.
    runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="raise",
        passive_deletes=True,
    )


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("agent_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[RunStatus] = mapped_column(RunStatusType, default=RunStatus.QUEUED, nullable=False, index=True)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(155), nullable=True)

    session: Mapped["AgentSession"] = relationship(back_populates="runs", lazy="raise")
    steps: Mapped[list["RunStep"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="raise",
        passive_deletes=True,
        order_by="RunStep.occurred_at",
    )


class RunStep(Base):
    """Append-only. Nothing ever updates or deletes a step — the trace is the audit log."""

    __tablename__ = "run_steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    step_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict] = mapped_column(JsonType, default=dict, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped["AgentRun"] = relationship(back_populates="steps", lazy="raise")

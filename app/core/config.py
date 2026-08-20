"""C-01 — the single typed source of environment configuration.

No other module reads ``os.environ``; everything imports ``settings``.
"""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required: absence raises at import time, not at the first LLM call.
    OPENAI_API_KEY: str

    # asyncpg driver is mandatory — plain `postgresql://` picks psycopg2 and the
    # async engine refuses it.
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@db:5432/agentcore"
    REDIS_URL: str = "redis://redis:6379/0"

    JWT_SECRET: str = "dev-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Concurrent QUEUED-or-RUNNING runs allowed per user before POST /run 429s.
    MAX_CONCURRENT_RUNS_PER_USER: int = 10

    # --- worker sizing -------------------------------------------------- #
    # Ceiling for one agent run. Ten LLM iterations plus tool calls; generous,
    # because the point is to stop a wedged run holding a slot forever, not to
    # police normal ones. The hard kill and the broker's visibility timeout are
    # derived from this so they can never drift out of order.
    AGENT_RUN_TIME_LIMIT_SECONDS: int = 900
    # 1, not Celery's default 4: these tasks are minutes long, and a prefetched
    # message is reserved behind whatever the worker is already doing.
    WORKER_PREFETCH_MULTIPLIER: int = 1
    WORKER_MAX_TASKS_PER_CHILD: int = 100
    # A task that kills its worker would otherwise be redelivered forever.
    MAX_RUN_ATTEMPTS: int = 3

    MAX_AGENT_ITERATIONS: int = 10
    TOOL_TIMEOUT_SECONDS: float = 5.0

    SHORT_TERM_WINDOW: int = 10    # turns read back into the prompt
    SHORT_TERM_RETAIN: int = 20    # turns kept in Redis
    LONG_TERM_TOP_K: int = 3       # facts recalled per run

    # "json" for machines, "console" for a readable local dev stream.
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"

    CHAT_MODEL: str = "gpt-5-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"   # 1536 dims == Vector(1536)

    # Thinking mode. Sent only when CHAT_MODEL is a reasoning model — set a
    # non-reasoning model (e.g. gpt-4o-mini) and the parameter is dropped rather
    # than rejected by the API. None disables it outright.
    REASONING_EFFORT: Literal["minimal", "low", "medium", "high"] | None = "medium"

    # Token-level streaming of the agent's output to the SSE client.
    STREAM_TOKENS: bool = True
    # Deltas are coalesced to at least this many characters before publishing.
    # One Redis write per token would be ~20x the traffic for no visible gain.
    STREAM_DELTA_CHARS: int = 24


settings = Settings()

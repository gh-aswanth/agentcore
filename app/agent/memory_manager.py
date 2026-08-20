"""Memory Manager
"""
from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import planner
from app.core.config import settings
from app.models.memory import LongTermMemory

SHORT_TERM_KEY = "session:{session_id}:turns"
SHORT_TERM_TTL = 60 * 60 * 24 * 7      # 7 days


class MemoryManager:
    """All Redis-key and vector-SQL knowledge lives here; the runner never
    touches either directly."""

    def __init__(self, redis, db: AsyncSession):
        self.redis = redis
        self.db = db

    # ------------------------------------------------------------------ #
    # short-term
    # ------------------------------------------------------------------ #
    async def read_short_term(self, session_id: str, limit: int | None = None) -> list[dict]:
        limit = limit or settings.SHORT_TERM_WINDOW
        key = SHORT_TERM_KEY.format(session_id=session_id)
        raw = await self.redis.lrange(key, 0, limit - 1)
        turns = [json.loads(r) for r in raw]
        return list(reversed(turns))

    async def append_turn(self, session_id: str, user_msg: str, answer: str) -> None:
        key = SHORT_TERM_KEY.format(session_id=session_id)
        pipe = self.redis.pipeline(transaction=False)
        pipe.lpush(key, json.dumps({"user": user_msg, "assistant": answer}))
        pipe.ltrim(key, 0, settings.SHORT_TERM_RETAIN - 1)
        pipe.expire(key, SHORT_TERM_TTL)
        await pipe.execute()

    # ------------------------------------------------------------------ #
    # long-term
    # ------------------------------------------------------------------ #
    async def embed(self, text: str) -> list[float]:
        return await planner.embed(text)

    async def write_long_term(
        self, user_id: str, content: str, source_run_id: str | None = None
    ) -> LongTermMemory:
        vector = await self.embed(content)
        entry = LongTermMemory(
            user_id=user_id, content=content, embedding=vector, source_run_id=source_run_id
        )
        self.db.add(entry)
        await self.db.commit()
        return entry

    async def search_long_term(
        self, user_id: str, query: str, limit: int | None = None
    ) -> list[LongTermMemory]:
        limit = limit or settings.LONG_TERM_TOP_K
        vector = await self.embed(query)
        stmt = (
            select(LongTermMemory)
            .where(LongTermMemory.user_id == user_id)
            .order_by(LongTermMemory.embedding.cosine_distance(vector))
            .limit(limit)
        )
        return list((await self.db.execute(stmt)).scalars())

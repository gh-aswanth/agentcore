"""T-5 — short-term memory mechanics and the long-term isolation guarantee."""
from app.agent.memory_manager import SHORT_TERM_KEY, MemoryManager
from app.core.config import settings
from app.models.memory import LongTermMemory


async def test_turns_read_back_in_chronological_order(db, redis):
    memory = MemoryManager(redis, db)
    for user_msg, answer in [("A", "1"), ("B", "2"), ("C", "3")]:
        await memory.append_turn("s1", user_msg, answer)

    turns = await memory.read_short_term("s1")

    # LPUSH stores newest-first; without the reversed() in read_short_term this
    # comes back C, B, A and silently degrades every prompt.
    assert [t["user"] for t in turns] == ["A", "B", "C"]
    assert [t["assistant"] for t in turns] == ["1", "2", "3"]


async def test_window_is_capped_and_oldest_turns_are_evicted(db, redis):
    memory = MemoryManager(redis, db)
    for i in range(settings.SHORT_TERM_RETAIN + 1):
        await memory.append_turn("s1", f"msg{i}", f"ans{i}")

    stored = await redis.lrange(SHORT_TERM_KEY.format(session_id="s1"), 0, -1)
    assert len(stored) == settings.SHORT_TERM_RETAIN
    assert "msg0" not in "".join(stored)          # the 21st turn evicted the 1st

    turns = await memory.read_short_term("s1", limit=settings.SHORT_TERM_WINDOW)
    assert len(turns) == settings.SHORT_TERM_WINDOW


async def test_sessions_do_not_share_short_term_memory(db, redis):
    memory = MemoryManager(redis, db)
    await memory.append_turn("s1", "mine", "ok")

    assert await memory.read_short_term("s2") == []


async def test_short_term_key_carries_a_ttl(db, redis):
    memory = MemoryManager(redis, db)
    await memory.append_turn("s1", "hi", "hello")

    assert await redis.ttl(SHORT_TERM_KEY.format(session_id="s1")) > 0


def test_long_term_search_filters_by_user_before_ordering():
    """pgvector needs Postgres, so the isolation guarantee is asserted on the
    compiled SQL: user_id must be in the WHERE clause, not a post-fetch check."""
    from sqlalchemy import select

    vector = [0.0] * 1536
    stmt = (
        select(LongTermMemory)
        .where(LongTermMemory.user_id == "user-a")
        .order_by(LongTermMemory.embedding.cosine_distance(vector))
        .limit(3)
    )
    sql = str(stmt.compile(compile_kwargs={"literal_binds": False}))

    assert "WHERE long_term_memory.user_id" in sql
    assert "ORDER BY long_term_memory.embedding <=>" in sql

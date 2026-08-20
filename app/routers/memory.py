from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import select

from app.agent.memory_manager import MemoryManager
from app.core.dependencies import CurrentUser, DbSession, RedisDep
from app.core.logging import get_logger
from app.models.memory import LongTermMemory
from app.schemas.agent import MemoryOut
from app.schemas.errors import AUTHENTICATED, OWNED

router = APIRouter(prefix="/memory", tags=["memory"])
log = get_logger(__name__)


@router.get(
    "",
    response_model=list[MemoryOut],
    summary="List stored facts",
    description="Long-term memory entries for the authenticated user, newest first. "
                "These are written only by the `remember_fact` tool.",
    responses=AUTHENTICATED,
)
async def list_memory(user: CurrentUser, db: DbSession, limit: int = Query(50, le=200)):
    stmt = (
        select(LongTermMemory)
        .where(LongTermMemory.user_id == user.id)
        .order_by(LongTermMemory.created_at.desc())
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars())


@router.get(
    "/search",
    response_model=list[MemoryOut],
    summary="Semantic search over stored facts",
    description="Embeds the query and orders by pgvector cosine distance (`<=>`). "
                "The `user_id` filter is applied before the ordering, so results can "
                "never cross users.",
    responses=AUTHENTICATED,
)
async def search_memory(
    user: CurrentUser,
    db: DbSession,
    redis: RedisDep,
    q: str = Query(min_length=1),
    limit: int = Query(5, le=20),
):
    memory = MemoryManager(redis, db)
    results = await memory.search_long_term(user.id, q, limit=limit)
    await log.ainfo("memory_searched", query_chars=len(q), results=len(results))
    return results


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Forget a fact",
    responses=OWNED,
)
async def delete_memory(memory_id: str, user: CurrentUser, db: DbSession):
    stmt = select(LongTermMemory).where(
        LongTermMemory.id == memory_id,
        LongTermMemory.user_id == user.id,
    )
    entry = (await db.execute(stmt)).scalar_one_or_none()
    if entry is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Memory entry not found")
    await db.delete(entry)
    await db.commit()
    await log.ainfo("memory_deleted", memory_id=memory_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

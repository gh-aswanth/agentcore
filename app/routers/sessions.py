from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select

from app.core.dependencies import CurrentUser, DbSession
from app.core.logging import bind, get_logger
from app.models.agent import AgentRun, AgentSession
from app.schemas.agent import AgentSessionCreate, AgentSessionDetail, AgentSessionOut
from app.schemas.errors import AUTHENTICATED, OWNED, UNPROCESSABLE

router = APIRouter(prefix="/sessions", tags=["sessions"])
log = get_logger(__name__)


async def get_owned_session(db, session_id: str, user_id: str) -> AgentSession:
    """C-11.1 — ownership is part of the WHERE clause, never a post-fetch check,
    and a miss is 404 rather than 403 so IDs cannot be enumerated."""
    stmt = select(AgentSession).where(
        AgentSession.id == session_id,
        AgentSession.user_id == user_id,
    )
    session = (await db.execute(stmt)).scalar_one_or_none()
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return session


@router.get(
    "",
    response_model=list[AgentSessionOut],
    summary="List your sessions",
    description="Newest first. Only sessions belonging to the authenticated user.",
    responses=AUTHENTICATED,
)
async def list_sessions(user: CurrentUser, db: DbSession):
    stmt = (
        select(AgentSession)
        .where(AgentSession.user_id == user.id)
        .order_by(AgentSession.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars())


@router.post(
    "",
    response_model=AgentSessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a session",
    description="`tools_enabled` is validated against the tool registry: an unknown "
                "name returns 422 naming the valid ones. The enum in this schema is "
                "generated from the registry, so it is always current.",
    responses={**AUTHENTICATED, 422: UNPROCESSABLE},
)
async def create_session(body: AgentSessionCreate, user: CurrentUser, db: DbSession):
    session = AgentSession(
        user_id=user.id,
        name=body.name,
        system_prompt=body.system_prompt,
        tools_enabled=body.tools_enabled,
    )
    db.add(session)
    await db.commit()

    bind(session_id=session.id)
    await log.ainfo("session_created", tools=body.tools_enabled, name=body.name)
    return session


@router.get(
    "/{session_id}",
    response_model=AgentSessionDetail,
    summary="Get a session with its recent runs",
    description="Includes a summary of the 20 most recent runs.",
    responses=OWNED,
)
async def get_session(session_id: str, user: CurrentUser, db: DbSession):
    session = await get_owned_session(db, session_id, user.id)
    # LIMIT 20 in the database rather than sorting session.runs in Python: the
    # relationship would load every run this session has ever had just to throw
    # all but the newest twenty away.
    recent = (
        select(AgentRun)
        .where(AgentRun.session_id == session.id)
        .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
        .limit(20)
    )
    return AgentSessionDetail(
        id=session.id,
        name=session.name,
        system_prompt=session.system_prompt,
        tools_enabled=session.tools_enabled,
        created_at=session.created_at,
        runs=list((await db.execute(recent)).scalars()),
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session",
    description="Cascades to its runs and their steps.",
    responses=OWNED,
)
async def delete_session(session_id: str, user: CurrentUser, db: DbSession):
    session = await get_owned_session(db, session_id, user.id)
    await db.delete(session)          # ON DELETE CASCADE clears runs and steps
    await db.commit()

    bind(session_id=session_id)
    await log.ainfo("session_deleted")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

"""C-05 — the injection layer.

The three ``Annotated`` aliases at the bottom are the ergonomic payoff: every
route signature collapses to one line.
"""
from typing import Annotated, AsyncGenerator

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import bind
from app.core.security import decode_token
from app.db.base import SessionLocal
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


def create_redis(url: str | None = None):
    """Factory shared by the API lifespan and the Celery worker.
    """
    return aioredis.from_url(url or settings.REDIS_URL, decode_responses=True)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


async def get_redis(request: Request):
    """One pooled client built in the app lifespan, not one per request."""
    return request.app.state.redis


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    # Every failure mode returns the identical 401 — distinguishing "bad
    # signature" from "user deleted" leaks information.
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
    except JWTError:
        raise credentials_exception

    user_id = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    user = await db.get(User, user_id)
    if user is None:
        raise credentials_exception

    bind(user_id=user.id)      # every later record in this request is attributable
    return user


DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
RedisDep = Annotated[aioredis.Redis, Depends(get_redis)]

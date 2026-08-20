"""C-02 — declarative base, async engine, session factory."""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)

# expire_on_commit=False is mandatory: with the default, any attribute read after
# commit() triggers a lazy refresh, which raises MissingGreenlet under async.
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

import os
from collections.abc import AsyncGenerator

from fastapi.exceptions import ResponseValidationError
import logging
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from app.models import Base, Event, DailyAggregate, ModelRegistry #noqa

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/{os.getenv('POSTGRES_DB')}")
engine = create_async_engine(
    DATABASE_URL,
    future=True,
    echo=True,
)

AsyncSessionFactory = async_sessionmaker(
    engine,
    autoflush=False,
    expire_on_commit=False,
)

async def init_db() -> None:
     async with engine.begin() as conn:
         await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncGenerator:
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError:
            raise
        except Exception as ex:
            if not isinstance(ex, ResponseValidationError):
                logger.error(f"Database-related error: {repr(ex)}")
            raise

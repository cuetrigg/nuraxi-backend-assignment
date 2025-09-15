from collections.abc import AsyncGenerator

from fastapi.exceptions import ResponseValidationError
import logging
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from pydantic_core import MultiHostUrl
from app.models import Base, Event, DailyAggregate, ModelRegistry

logger = logging.getLogger(__name__)

asyncpg_url = MultiHostUrl.build(
            scheme="postgresql+asyncpg",
            username="",
            password="",
            host="",
            path="",
        ) 



engine = create_async_engine(
    asyncpg_url.unicode_string(),
    future=True,
    echo=True,
)

# expire_on_commit=False will prevent attributes from being expired
# after commit.
AsyncSessionFactory = async_sessionmaker(
    engine,
    autoflush=False,
    expire_on_commit=False,
)


async def init_db() -> None:
     async with engine.begin() as conn:
         await conn.run_sync(Base.metadata.create_all)

# Dependency
async def get_db() -> AsyncGenerator:
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError:
            # Re-raise SQLAlchemy errors to be handled by the global handler
            raise
        except Exception as ex:
            # Only log actual database-related issues, not response validation
            if not isinstance(ex, ResponseValidationError):
                logger.error(f"Database-related error: {repr(ex)}")
            raise  # Re-raise to be handled by appropriate handlers

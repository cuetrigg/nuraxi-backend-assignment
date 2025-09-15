import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Creating tables")
    await init_db()
    logger.info("Created tables")
    yield


app = FastAPI(lifespan=lifespan)

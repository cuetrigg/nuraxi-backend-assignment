import logging
import os
import datetime

from typing import List, Optional
from uuid import UUID
from enum import Enum

from contextlib import asynccontextmanager
from model_service import AggregateLLMService

from fastapi import FastAPI, Depends, HTTPException, status, Request
from pydantic import BaseModel, validator

from sqlalchemy import select, and_
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import init_db, get_db
from app.models import Event, DailyAggregate

from event_bus import event_bus
from streaming_service import StreamingService

logger = logging.getLogger(__name__)

MODELS_DIR = os.getenv("MODELS_DIR", "./models")
ml_service = AggregateLLMService(models_dir=MODELS_DIR)

streaming_service = StreamingService(event_bus)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)

class EventType(str, Enum):
    heart_rate = "heart_rate"
    steps = "steps"
    sleep = "sleep"


class EventItem(BaseModel):
    user_id: str
    event_type: EventType
    timestamp: datetime.datetime
    value: float


class EventsRequest(BaseModel):
    events: List[EventItem]
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "events": [
                        {
                            "user_id": "123e4567-e89b-12d3-a456-426614174000",
                            "timestamp": "2025-09-20T02:13:00Z",
                            "event_type": "heart_rate",
                            "value": 1,
                        }
                    ]
                }
            ]
        }
    }

@app.post("/v1/events", status_code=status.HTTP_201_CREATED)
async def ingest_wearable_events(
     payload: EventsRequest, db: AsyncSession = Depends(get_db)
):
    """
    Ingest one or more wearable events for a user.
    """
    try:
        event_instances = [Event(**event.model_dump()) for event in payload.events]
        
        db.add_all(event_instances)
        await db.commit()
        return { 200: "ok" }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing events: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process events: {str(e)}",
        )

class DailyAggregatesResponse(BaseModel):
    user_id: UUID
    date: datetime.date
    steps_total: int
    hr_avg: float
    sleep_minutes: int
    computed_at: datetime.datetime

@app.post("/v1/aggregate/{user_id}", status_code=status.HTTP_200_OK, response_model=DailyAggregatesResponse)
async def compute_daily_aggregates(
    user_id: UUID, date: datetime.date, db: AsyncSession = Depends(get_db)
):
    """
    Compute and persist daily aggregates for a user.
    """

    try:
        start_of_day = datetime.datetime.combine(date, datetime.datetime.min.time(), datetime.timezone.utc)
        end_of_day = start_of_day + datetime.timedelta(days=1)

        events_query = (
            select(
                Event.event_type,
                func.sum(Event.value).label("total_value"),
                func.avg(Event.value).label("avg_value"),
                func.count(Event.value).label("event_count"),
            )
            .where(
                and_(
                    Event.user_id == user_id,
                    Event.timestamp >= start_of_day,
                    Event.timestamp < end_of_day,
                )
            )
            .group_by(Event.event_type)
        )

        result = await db.execute(events_query)
        event_stats = result.fetchall()

        steps_total = 0
        hr_avg = 0.0
        sleep_minutes = 0

        for row in event_stats:
            event_type = row.event_type
            total_value = row.total_value or 0
            avg_value = row.avg_value or 0

            if event_type == "steps":
                steps_total = int(total_value)
            elif event_type == "heart_rate":
                hr_avg = float(avg_value)
            elif event_type == "sleep":
                sleep_minutes = int(total_value)

        computed_at = datetime.datetime.now(datetime.timezone.utc)
        
        stmt = insert(DailyAggregate).values(
            user_id=user_id,
            date=date,
            steps_total=steps_total,
            hr_avg=hr_avg,
            sleep_minutes=sleep_minutes,
            computed_at=computed_at
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=['user_id', 'date'],
            set_={
                'steps_total': stmt.excluded.steps_total,
                'hr_avg': stmt.excluded.hr_avg,
                'sleep_minutes': stmt.excluded.sleep_minutes,
                'computed_at': stmt.excluded.computed_at
            }
        )
    
        await db.execute(stmt)
        await db.commit()

        await event_bus.publish(user_id, {
            "user_id": str(user_id),
            "date": date.isoformat(),
            "steps_total": steps_total,
            "hr_avg": hr_avg,
            "sleep_minutes": sleep_minutes,
            "computed_at": computed_at.isoformat()
        })
    
        return DailyAggregatesResponse(
            user_id=user_id,
            date=date,
            steps_total=steps_total,
            hr_avg=hr_avg,
            sleep_minutes=sleep_minutes,
            computed_at=computed_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing aggregate: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute aggregate: {str(e)}"
        )

@app.get("/v1/users/{user_id}/daily/{date}", status_code=status.HTTP_200_OK, response_model=DailyAggregatesResponse)
async def retrieve_daily_aggregates(
    user_id: UUID, date: datetime.datetime, db: AsyncSession = Depends(get_db)
):
    """
    Compute and persist daily aggregates for a user.
    """
    query = select(DailyAggregate).where(
        and_(
            DailyAggregate.user_id == user_id,
            DailyAggregate.date == date
        )
    )
    
    result = await db.execute(query)
    aggregate = result.scalar_one_or_none()
    
    if not aggregate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No daily aggregate found for user {user_id} on {date}"
        )
    
    return DailyAggregatesResponse(
        user_id =  aggregate.user_id,
        date =  aggregate.date,
        steps_total =  aggregate.steps_total or 0,
        hr_avg =  aggregate.hr_avg or 0.0,
        sleep_minutes =  aggregate.sleep_minutes or 0,
        computed_at =  aggregate.computed_at
    )

class DailyAggregateTrainItem(BaseModel):
    steps_total: int
    hr_avg: float
    sleep_minutes: int
    label: bool

class DailyAggregatesTrainRequest(BaseModel):
    rows: List[DailyAggregateTrainItem]

class Metrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

class DailyAggregatesTrainResponse(BaseModel):
    version: str
    metrics: Metrics

@app.post("/v1/model/train", status_code=status.HTTP_200_OK, response_model=DailyAggregatesTrainResponse)
async def train_model(
    payload: DailyAggregatesTrainRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Train a classifier to predict low_energy_next_day from daily aggregates.
    """
    try:
        result = await ml_service.train_model(
            db=db,
            rows=payload.rows,
        )

        metrics = Metrics(**result["metrics"])

        return DailyAggregatesTrainResponse(
            version=result["version"],
            metrics=metrics
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to train model: {str(e)}"
        )

class Features(BaseModel):
    steps_total: int
    hr_avg: float
    sleep_minutes: int

class PredictRequest(BaseModel):
    features: Optional[Features] = None
    user_id: Optional[UUID] = None
    date: Optional[str] = None

    @validator("features", pre=True, always=True)
    def check_either_features_or_user_id_date(cls, features, values):
        user_id = values.get("user_id")
        date = values.get("date")
        if features is None and (user_id is None or date is None):
            raise ValueError("Must provide either 'features' or both 'user_id' and 'date'")
        if features is not None and (user_id is not None or date is not None):
            raise ValueError("Cannot provide both 'features' and 'user_id'/'date'")
        return features

class PredictResponse(BaseModel):
    version: str
    probability: float
    prediction: bool

@app.post("/v1/predict", status_code=status.HTTP_200_OK, response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate predictions using the latest trained model.
    """
    try:
        if request.features is not None:
            logger.info("Making prediction from direct features")
            return await ml_service.predict_from_features(db, request.features.dict())
            
        elif request.user_id is not None and request.date is not None:
            logger.info(f"Making prediction for user {request.user_id} on {request.date}")
            return await ml_service.predict_from_aggregate(db, request.user_id, request.date)
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either 'features' or both 'user_id' and 'date'"
            )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to make prediction: {str(e)}"
        )


@app.get("/v1/stream/{user_id}")
async def real_time_event_stream(user_id: UUID, request: Request):
    """
    Stream newly computed aggregates for a user in real-time.
    """
    logger.info(f"Creating SSE stream for user {user_id}")
    
    return await streaming_service.create_user_stream(user_id, request)

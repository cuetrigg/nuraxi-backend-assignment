from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, MappedAsDataclass
from sqlalchemy import Integer, DateTime, Numeric, Date, JSON, Text, Index
from sqlalchemy.sql import func
from sqlalchemy.types import Uuid
import uuid
from datetime import datetime, date


class Base(MappedAsDataclass, DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "events"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    event_type: Mapped[str] = mapped_column(Text)
    value: Mapped[float] = mapped_column(Numeric)
    
    __table_args__ = (
        Index('ix_events_user_timestamp', 'user_id', 'timestamp'),
        Index('ix_events_timestamp', 'timestamp'),
    )

class DailyAggregate(Base):
    __tablename__ = "daily_aggregates"
    
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    steps_total: Mapped[int | None] = mapped_column(Integer, default=None)
    hr_avg: Mapped[float | None] = mapped_column(Numeric, default=None)
    sleep_minutes: Mapped[int | None] = mapped_column(Integer, default=None)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        init=False
    )


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    version: Mapped[str] = mapped_column(Text)
    path: Mapped[str] = mapped_column(Text)
    metrics: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        init=False
    )

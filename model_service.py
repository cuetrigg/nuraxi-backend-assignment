import os
import joblib
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import UUID
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.models import ModelRegistry, DailyAggregate

logger = logging.getLogger(__name__)


class AggregateLLMService:
    def __init__(self, models_dir: str = "/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    async def train_model(
        self,
        db: AsyncSession,
        rows,
    ):
        """Train a new model and persist to disk with metadata"""

        X = np.array([[row.steps_total, row.hr_avg, row.sleep_minutes] for row in rows])
        y = np.array([row.label for row in rows])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        logger.info(
            f"Label distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}"
        )

        model = LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced"
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, y_pred_proba))
            if len(np.unique(y_val)) > 1
            else 0.5,
        }

        logger.info(f"Model metrics: {metrics}")

        # Generate version and save model
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        model_path = self.models_dir / f"model_{version}.pkl"

        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Record in model registry
        registry_entry = ModelRegistry(
            version=version, path=str(model_path), metrics=metrics
        )

        db.add(registry_entry)
        await db.commit()

        return {
            "version": version,
            "metrics": metrics,
        }

    async def get_latest_model(self, db: AsyncSession) -> Optional[ModelRegistry]:
        """Get the latest trained model from registry"""
        query = select(ModelRegistry).order_by(desc(ModelRegistry.created_at)).limit(1)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    def _load_model(self, model_path: str):
        """Load model from disk"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        return joblib.load(model_path)

    async def predict_from_features(self, db: AsyncSession, features: Dict[str, float]):
        """Make prediction from direct features"""
        # Get latest model
        model_registry = await self.get_latest_model(db)
        if not model_registry:
            raise ValueError("No trained model available")

        # Load model
        model = self._load_model(model_registry.path)

        # Prepare features
        feature_array = np.array([
            [features["steps_total"], features["hr_avg"], features["sleep_minutes"]]
        ])

        # Make prediction
        probability = float(model.predict_proba(feature_array)[0, 1])
        prediction = bool(probability > 0.5)

        logger.info(
            f"Prediction: {prediction} (prob: {probability:.3f}) using model {model_registry.version}"
        )

        return {
            "version": model_registry.version,
            "probability": probability,
            "prediction": prediction,
        }

    async def predict_from_aggregate(self, db: AsyncSession, user_id: UUID, date: str):
        """Make prediction from stored daily aggregate"""
        from datetime import date as date_type

        # Parse date
        target_date = date_type.fromisoformat(date)

        # Get daily aggregate
        query = select(DailyAggregate).where(
            DailyAggregate.user_id == user_id, DailyAggregate.date == target_date
        )
        result = await db.execute(query)
        aggregate = result.scalar_one_or_none()

        if not aggregate:
            raise ValueError(f"No daily aggregate found for user {user_id} on {date}")

        features = {
            "steps_total": int(aggregate.steps_total or 0),
            "hr_avg": float(aggregate.hr_avg or 0.0),
            "sleep_minutes": int(aggregate.sleep_minutes or 0),
        }

        return await self.predict_from_features(db, features)

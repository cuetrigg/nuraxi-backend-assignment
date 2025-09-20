import pytest
import datetime
import uuid
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from app.main import app, Event, DailyAggregate, EventType
from app.database import get_db

# Mark the entire module as asyncio
pytestmark = pytest.mark.asyncio

# Custom class to mimic SQLAlchemy Row object
class MockRow:
    def __init__(self, event_type, total_value, avg_value, event_count):
        self.event_type = event_type
        self.total_value = total_value
        self.avg_value = avg_value
        self.event_count = event_count

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def db_session():
    async_session = AsyncMock(spec=AsyncSession)
    async_session.execute = AsyncMock()
    async_session.commit = AsyncMock()
    async_session.add_all = AsyncMock()
    yield async_session

@pytest.fixture
def mock_event_bus():
    with patch("app.main.event_bus") as mock:
        mock.publish = AsyncMock()
        yield mock

async def test_compute_daily_aggregates_success(client, db_session, mock_event_bus):
    # Arrange
    user_id = uuid.uuid4()
    test_date = datetime.date(2025, 9, 20)
    start_of_day = datetime.datetime(2025, 9, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end_of_day = start_of_day + datetime.timedelta(days=1)

    # Mock database query result
    mock_result = [
        MockRow(EventType.steps, 10000.0, 10000.0, 10),  # steps: total_value=10000
        MockRow(EventType.heart_rate, 720.0, 72.0, 10),  # heart_rate: avg_value=72.0
        MockRow(EventType.sleep, 480.0, 480.0, 1),  # sleep: total_value=480
    ]
    
    # Create a mock result object
    result_mock = MagicMock()
    result_mock.fetchall.return_value = mock_result
    result_mock.scalar_one_or_none.return_value = None
    db_session.execute.return_value = result_mock

    # Override dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Act
    response = client.post(f"/v1/aggregate/{user_id}?date={test_date}")

    # Assert
    assert response.status_code == 200
    response_json = response.json()
    
    assert response_json["user_id"] == str(user_id)
    assert response_json["date"] == test_date.isoformat()
    assert response_json["steps_total"] == 10000
    assert response_json["hr_avg"] == 72.0
    assert response_json["sleep_minutes"] == 480
    
    # Verify database insert
    db_session.execute.assert_called()
    db_session.commit.assert_called()
    
    # Verify event bus publish
    mock_event_bus.publish.assert_called_once()
    published_event = mock_event_bus.publish.call_args[0][1]
    assert published_event["user_id"] == str(user_id)
    assert published_event["steps_total"] == 10000
    assert published_event["hr_avg"] == 72.0
    assert published_event["sleep_minutes"] == 480

async def test_compute_daily_aggregates_no_events(client, db_session, mock_event_bus):
    # Arrange
    user_id = uuid.uuid4()
    test_date = datetime.date(2025, 9, 20)

    # Mock empty query result
    result_mock = MagicMock()
    result_mock.fetchall.return_value = []
    result_mock.scalar_one_or_none.return_value = None
    db_session.execute.return_value = result_mock

    # Override dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Act
    response = client.post(f"/v1/aggregate/{user_id}?date={test_date}")

    # Assert
    assert response.status_code == 200
    response_json = response.json()
    
    assert response_json["user_id"] == str(user_id)
    assert response_json["date"] == test_date.isoformat()
    assert response_json["steps_total"] == 0
    assert response_json["hr_avg"] == 0.0
    assert response_json["sleep_minutes"] == 0
    
    # Verify database insert
    db_session.execute.assert_called()
    db_session.commit.assert_called()
    
    # Verify event bus publish
    mock_event_bus.publish.assert_called_once()
    published_event = mock_event_bus.publish.call_args[0][1]
    assert published_event["steps_total"] == 0
    assert published_event["hr_avg"] == 0.0
    published_event["sleep_minutes"] == 0

async def test_compute_daily_aggregates_partial_events(client, db_session, mock_event_bus):
    # Arrange
    user_id = uuid.uuid4()
    test_date = datetime.date(2025, 9, 20)

    # Mock partial query result (only steps and heart_rate)
    mock_result = [
        MockRow(EventType.steps, 5000.0, 5000.0, 5),
        MockRow(EventType.heart_rate, 700.0, 70.0, 10),
    ]
    
    result_mock = MagicMock()
    result_mock.fetchall.return_value = mock_result
    result_mock.scalar_one_or_none.return_value = None
    db_session.execute.return_value = result_mock

    # Override dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Act
    response = client.post(f"/v1/aggregate/{user_id}?date={test_date}")

    # Assert
    assert response.status_code == 200
    response_json = response.json()
    
    assert response_json["user_id"] == str(user_id)
    assert response_json["date"] == test_date.isoformat()
    assert response_json["steps_total"] == 5000
    assert response_json["hr_avg"] == 70.0
    assert response_json["sleep_minutes"] == 0
    
    # Verify database insert
    db_session.execute.assert_called()
    db_session.commit.assert_called()
    
    # Verify event bus publish
    mock_event_bus.publish.assert_called_once()
    published_event = mock_event_bus.publish.call_args[0][1]
    assert published_event["steps_total"] == 5000
    assert published_event["hr_avg"] == 70.0
    assert published_event["sleep_minutes"] == 0

async def test_compute_daily_aggregates_invalid_date(client, db_session):
    # Arrange
    user_id = uuid.uuid4()
    invalid_date = "invalid-date"

    # Override dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Act
    response = client.post(f"/v1/aggregate/{user_id}?date={invalid_date}")

    # Assert
    assert response.status_code == 422  # Unprocessable Entity due to invalid date format

# New tests for model prediction endpoint
@pytest.fixture
def mock_ml_service():
    with patch("app.main.ml_service") as mock:
        mock.predict_from_features = AsyncMock()
        mock.predict_from_aggregate = AsyncMock()
        yield mock

async def test_predict_with_features_success(client, db_session, mock_ml_service):
    # Arrange
    mock_ml_service.predict_from_features.return_value = {
        "version": "1.0",
        "probability": 0.75,
        "prediction": True
    }
    
    payload = {
        "features": {
            "steps_total": 10000,
            "hr_avg": 72.0,
            "sleep_minutes": 480
        }
    }

    # Override dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Act
    response = client.post("/v1/predict", json=payload)

    # Assert
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["version"] == "1.0"
    assert response_json["probability"] == 0.75
    assert response_json["prediction"] is True
    mock_ml_service.predict_from_features.assert_called_once_with(db_session, payload["features"])

async def test_predict_with_user_id_and_date_success(client, db_session, mock_ml_service):
    # Arrange
    user_id = uuid.uuid4()
    test_date = "2025-09-20"
    
    mock_ml_service.predict_from_aggregate.return_value = {
        "version": "1.0",
        "probability": 0.65,
        "prediction": False
    }

    # Mock database query for aggregate
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = MagicMock(
        user_id=user_id,
        date=datetime.date(2025, 9, 20),
        steps_total=5000,
        hr_avg=70.0,
        sleep_minutes=0,
        computed_at=datetime.datetime.now(datetime.timezone.utc)
    )
    db_session.execute.return_value = result_mock

    # Override dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Act
    payload = {
        "user_id": str(user_id),
        "date": test_date,
        "features": None  # Explicitly include features as None to satisfy validator
    }
    response = client.post("/v1/predict", json=payload)
    print(response.text)  # Debug: Print response body for 422 error

    # Assert
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["version"] == "1.0"
    assert response_json["probability"] == 0.65
    assert response_json["prediction"] is False
    mock_ml_service.predict_from_aggregate.assert_called_once_with(db_session, user_id, test_date)

async def test_predict_invalid_request_missing_features_and_user_id(client, db_session):
    # Arrange
    payload = {}

    # Override dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Act
    response = client.post("/v1/predict", json=payload)

    # Assert
    assert response.status_code == 422
    response_json = response.json()
    assert "Must provide either 'features' or both 'user_id' and 'date'" in response_json["detail"][0]["msg"]

# New tests for event ingestion validation
async def test_ingest_wearable_events_success(client, db_session):
    # Arrange
    user_id = str(uuid.uuid4())
    payload = {
        "events": [
            {
                "user_id": user_id,
                "event_type": "heart_rate",
                "timestamp": "2025-09-20T02:13:00Z",
                "value": 72.0
            },
            {
                "user_id": user_id,
                "event_type": "steps",
                "timestamp": "2025-09-20T02:14:00Z",
                "value": 1000.0
            }
        ]
    }

    # Override dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Act
    response = client.post("/v1/events", json=payload)
    print(response.json())  # Debug: Print response body for assertion failure

    # Assert
    assert response.status_code == 201
    assert response.json() == {"200": "ok"}  # Match JSON-serialized string key
    db_session.add_all.assert_called_once()
    db_session.commit.assert_called_once()

async def test_ingest_wearable_events_invalid_event_type(client, db_session):
    # Arrange
    user_id = str(uuid.uuid4())
    payload = {
        "events": [
            {
                "user_id": user_id,
                "event_type": "invalid_type",  # Invalid event type
                "timestamp": "2025-09-20T02:13:00Z",
                "value": 72.0
            }
        ]
    }

    # Override dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Act
    response = client.post("/v1/events", json=payload)

    # Assert
    assert response.status_code == 422
    response_json = response.json()
    assert "Input should be 'heart_rate', 'steps' or 'sleep'" in response_json["detail"][0]["msg"]

async def test_ingest_wearable_events_missing_required_field(client, db_session):
    # Arrange
    user_id = str(uuid.uuid4())
    payload = {
        "events": [
            {
                "user_id": user_id,
                "event_type": "heart_rate",
                # Missing timestamp
                "value": 72.0
            }
        ]
    }

    # Override dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Act
    response = client.post("/v1/events", json=payload)

    # Assert
    assert response.status_code == 422
    response_json = response.json()
    assert "Field required" in response_json["detail"][0]["msg"]

# Clean up dependency overrides after tests
@pytest.fixture(autouse=True)
def cleanup():
    yield
    app.dependency_overrides.clear()

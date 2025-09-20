## Overview
For this project, I chose a minimal approach, prioritizing a working application over perfection. This avoided complex folder structures and code splitting. The goal was to get the job done quickly, but this approach has flaws for production, such as difficulty finding functionality due to lack of grouping and the risk of adding files randomly.

## Why This Approach?
To deliver a working application quickly, I focused on functionality over perfection.

## Code and Structure Trade-offs

### 1. Endpoints and Request/Response Models
All endpoints and request/response models are in the main file, causing file bloat and maintenance issues. In production, I would create an `api` route folder with routes per feature (e.g., `dailyaggregate`, `ml`, `user`).

### 2. Database Connection/Queries
I used a reusable `get_db` function as a dependency for endpoints to simplify querying, but this sacrifices custom session and data handling. In production, I would implement robust session management and query handlers.

### 3. Model Training
I defined a separate service using Logistic Regression, suitable for this assignment’s small dataset but not for many categorical features. In production, I would use advanced models like Random Forest for larger datasets.

### 4. Database Models
I used `mapped_columns` in SQLAlchemy instead of `Column` definitions, requiring more code but enforcing explicitness. In production, I would ensure team familiarity to reduce onboarding time.

### 5. Testing Strategy
Tests run in a separate Docker container for model-specific endpoints needing file write access, avoiding a separate PostgreSQL instance, but this slows execution. In production, I would use in-memory databases or optimize containers.

### 6. Streaming Endpoint
I used Server-Sent Events (SSE) with an in-memory event bus, suitable for small-scale apps but limited by memory and CPU in production. I would switch to WebSockets and Redis for better concurrency and scalability.

# Nuraxi Project

This document provides instructions for setting up, running, and testing the Nuraxi project, a FastAPI application with a PostgreSQL database, managed using Docker and Docker Compose.

## Prerequisites

Ensure the following tools are installed:

- **Docker**: Required to build and run containers.
- **Docker Compose**: Used to manage multi-container setups.
- **Python 3.12+**: Optional, for local development outside Docker.

You must also create a `.env` file in the project root with the following variables:

```plaintext
POSTGRES_USER=admin
POSTGRES_PASSWORD=secret
POSTGRES_DB=nuraxi
```

## Project Structure

The project is organized as follows:

- `Dockerfile`: Defines the production application container.
- `Dockerfile.test`: Defines the test container.
- `pyproject.toml`: Specifies Python dependencies and project metadata.
- `docker-compose.yml`: Configures the application and database services for production.
- `docker-compose-test.yml`: Configures the test environment.
- `docker-compose-local.yml`: Configures the database for local development.

## Instructions

### 1. Build the Application

To build the FastAPI application and PostgreSQL database containers:

```bash
docker-compose -f docker-compose.yml build
```

This command uses the `Dockerfile` to set up the application and database services.

### 2. Run the Application Locally

To start the application and database in the background:

```bash
docker-compose -f docker-compose.yml up -d
```

- The FastAPI application will be available at `http://localhost:8000`.
- The PostgreSQL database will be accessible on port `5432`.

To stop the services:

```bash
docker-compose -f docker-compose.yml down
```

### 3. Run Tests

To execute the test suite in a separate test environment:

```bash
docker-compose -f docker-compose-test.yml up --build
```

- This command uses `Dockerfile.test` to build the test container and runs `pytest` as configured in `pyproject.toml`.
- The test database runs on port `5433` to avoid conflicts with the production database.
- Tests run automatically, and the container exits upon completion.

To clean up after testing:

```bash
docker-compose -f docker-compose-test.yml down
```

### 4. Local Development (Database Only)

For local development without running the application in Docker, you can start only the PostgreSQL database:

```bash
docker-compose -f docker-compose-local.yml up -d
```

- The database will be available on port `5432`.

To run the FastAPI application locally:

```bash
pip install .
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

To stop the database:

```bash
docker-compose -f docker-compose-local.yml down
```

## Environment Variables

The application and test containers rely on the following environment variables, defined in the `.env` file:

- `POSTGRES_USER`: PostgreSQL username (default: `admin`).
- `POSTGRES_PASSWORD`: PostgreSQL password (default: `secret`).
- `POSTGRES_DB`: Database name (default: `nuraxi` for production, `nuraxi_test` for tests).
- `DATABASE_URL`: Automatically constructed in `docker-compose.yml` (e.g., `postgresql+asyncpg://admin:secret@db:5432/nuraxi`).
- `MODELS_DIR`: Directory for model storage (default: `/app/models`).
- `APP_PORT`: Port for the FastAPI application (default: `8000`).

## Notes

- Ensure the `.env` file is present in the project root with all required variables.
- The `models` directory is mounted as a volume in `docker-compose.yml` to persist model data.
- The test environment uses a separate volume (`postgres_data_test`) to avoid interfering with the production database.
- If you encounter permission issues with the `models` directory, ensure it exists and has appropriate permissions:

```bash
chmod 755 models
```

## Troubleshooting

- **Port conflicts**: If ports `5432` or `8000` are in use, stop conflicting services or modify the ports in the relevant `docker-compose` files.
- **Database connection issues**: Verify the `DATABASE_URL` and ensure the database is healthy (use `pg_isready` healthcheck).
- **Dependency issues**: If `pip install` fails, ensure `pyproject.toml` is correctly formatted and dependencies are compatible.

For further assistance, refer to the [FastAPI documentation](https://fastapi.tiangolo.com/) or [PostgreSQL documentation](https://www.postgresql.org/docs/).

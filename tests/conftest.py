"""Global test fixtures and environment configuration."""

import pytest
from src.environment.database import reset_db_manager


@pytest.fixture(autouse=True)
def cleanup_database_resources() -> None:
    """Ensure database manager and libSQL client threads are closed after each test."""
    yield
    reset_db_manager()

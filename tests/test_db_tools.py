"""Unit tests for DatabaseToolEngine."""

import pytest

from src.environment.database import Base, DatabaseManager
from src.environment.db_tools import DatabaseToolEngine
from src.data.olist_loader import seed_olist_database
from src.utils.config import Settings


@pytest.fixture
def test_db_manager() -> DatabaseManager:
    """Fixture providing in-memory async database manager."""
    settings = Settings(database_url="sqlite+aiosqlite:///:memory:")
    return DatabaseManager(settings=settings)


@pytest.mark.asyncio
async def test_db_tool_engine_execution_and_counters(test_db_manager: DatabaseManager) -> None:
    """Verify tool execution, call counters, and schema validation error handling."""
    engine = test_db_manager.get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with test_db_manager.get_session() as session:
        await seed_olist_database(
            session=session,
            seed=42,
            num_vendors=5,
            num_items=5,
            num_orders=5,
        )

        tool_engine = DatabaseToolEngine()

        # 1. Valid lookup
        obs1 = await tool_engine.execute(
            "lookup_records",
            {"table": "vendors", "filters": {"category": "Hardware"}},
            session=session,
        )
        assert not obs1.is_error
        assert tool_engine.tool_call_counts["lookup_records"] == 1

        # 2. Schema violation: extra argument rejected before execution
        obs2 = await tool_engine.execute(
            "lookup_records",
            {"table": "vendors", "unauthorized_extra": 999},
            session=session,
        )
        assert obs2.is_error
        assert obs2.error_code == "SCHEMA_VALIDATION_ERROR"
        # Call counter must NOT increment on schema error
        assert tool_engine.tool_call_counts["lookup_records"] == 1

        # 3. Aggregation tool
        obs3 = await tool_engine.execute(
            "compute_aggregate",
            {"values": [10.5, 20.5, 30.0], "operation": "sum"},
            session=session,
        )
        assert not obs3.is_error
        assert obs3.result == 61.0
        assert tool_engine.tool_call_counts["compute_aggregate"] == 1

        # 4. Verify constraint tool
        obs4 = await tool_engine.execute(
            "verify_constraint",
            {"actual": 61.0, "expected": 100.0, "comparison": "le"},
            session=session,
        )
        assert not obs4.is_error
        assert obs4.result is True
        assert tool_engine.tool_call_counts["verify_constraint"] == 1

    await test_db_manager.close()


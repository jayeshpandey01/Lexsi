"""Unit tests for production database engine and Olist enterprise dataset."""

import pytest
from sqlalchemy import select

from src.data.olist_loader import seed_olist_database
from src.environment.database import (
    Base,
    BudgetCapModel,
    DatabaseManager,
    InventoryItemModel,
    PurchaseOrderModel,
    VendorModel,
)
from src.environment.db_tools import lookup_records_sql
from src.utils.config import Settings


@pytest.fixture
def test_db_manager() -> DatabaseManager:
    """Fixture providing in-memory async database manager."""
    settings = Settings(database_url="sqlite+aiosqlite:///:memory:")
    return DatabaseManager(settings=settings)


@pytest.mark.asyncio
async def test_database_init_and_seeding(test_db_manager: DatabaseManager) -> None:
    """Verify in-memory database initializes tables and seeds Olist records."""
    engine = test_db_manager.get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with test_db_manager.get_session() as session:
        seed_res = await seed_olist_database(
            session=session,
            seed=42,
            num_vendors=10,
            num_items=15,
            num_orders=25,
        )
        assert seed_res["status"] == "success"
        assert seed_res["vendors_count"] == 10
        assert seed_res["items_count"] == 15
        assert seed_res["orders_count"] == 25

        # Query vendors directly
        v_res = await session.execute(select(VendorModel).limit(5))
        vendors = v_res.scalars().all()
        assert len(vendors) == 5
        assert all(v.vendor_id.startswith("VND-BR-") for v in vendors)

    await test_db_manager.close()


@pytest.mark.asyncio
async def test_parameterized_lookup_sql(test_db_manager: DatabaseManager) -> None:
    """Verify lookup_records_sql filters records accurately."""
    engine = test_db_manager.get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with test_db_manager.get_session() as session:
        await seed_olist_database(
            session=session,
            seed=42,
            num_vendors=10,
            num_items=10,
            num_orders=10,
        )
        # Query with filter
        records = await lookup_records_sql(
            table="vendors",
            filters={"category": "Hardware"},
            session=session,
        )
        assert len(records) > 0
        assert all(r["category"] == "Hardware" for r in records)
        assert "vendor_id" in records[0]

    await test_db_manager.close()


def test_turso_database_configuration() -> None:
    """Verify Turso libSQL detection, URL parsing, and client fallback properties."""
    turso_settings = Settings(
        database_url="libsql://lexsi-jayesh20.aws-ap-south-1.turso.io",
        turso_auth_token="dummy_test_jwt_token",
    )
    db = DatabaseManager(settings=turso_settings)
    assert db.is_turso() is True

    # Resolves local SQLite path for SQLAlchemy engine while Turso is handled via libsql_client
    resolved_engine_url = db._resolve_engine_url()
    assert "sqlite+aiosqlite:///" in resolved_engine_url


def test_turso_fail_fast_on_unauthorized_fallback() -> None:
    """Verify that Turso connection failure raises CloudDatabaseConnectionError when fallback is forbidden."""
    from src.environment.database import CloudDatabaseConnectionError

    # Invalid URL / token forcing connection error without fallback permission
    broken_settings = Settings(
        database_url="libsql://invalid-turso-host-xyz.turso.io",
        turso_auth_token="invalid_token",
        allow_local_db_fallback=False,
    )
    db = DatabaseManager(settings=broken_settings)

    try:
        with pytest.raises(CloudDatabaseConnectionError):
            db.execute_query("SELECT 1")
    finally:
        db.close_sync()


def test_turso_authorized_fallback() -> None:
    """Verify that Turso connection failure allows local fallback when explicitly authorized."""
    broken_settings = Settings(
        database_url="libsql://invalid-turso-host-xyz.turso.io",
        turso_auth_token="invalid_token",
        allow_local_db_fallback=True,
    )
    db = DatabaseManager(settings=broken_settings)
    try:
        # Should not raise CloudDatabaseConnectionError; falls back to local SQLite
        res = db.execute_query("SELECT 1 as num")
        assert len(res) == 1
        assert res[0]["num"] == 1
    finally:
        db.close_sync()




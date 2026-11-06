"""Production-grade relational database engine and connection pool manager.

Supports:
- Turso libSQL cloud databases (libsql://... with auth token).
- Local asynchronous SQLite via aiosqlite (sqlite+aiosqlite:///...).
- Cloud PostgreSQL via asyncpg (postgresql+asyncpg://...).

Implements the Olist Brazilian E-Commerce & Supply Chain relational schema
adapted for enterprise Procurement & Inventory Analytics.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, relationship

from src.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()


# ------------------------------------------------------------------------------
# Olist Relational Database Models
# ------------------------------------------------------------------------------

class VendorModel(Base):
    """Olist enterprise vendor / seller entity."""

    __tablename__ = "vendors"

    vendor_id = Column(String(64), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(64), nullable=False, index=True)
    city = Column(String(128), default="Sao Paulo")
    state = Column(String(8), default="SP")
    rating = Column(Float, default=4.5)
    contract_status = Column(String(32), default="Active")

    purchase_orders = relationship("PurchaseOrderModel", back_populates="vendor")


class InventoryItemModel(Base):
    """Olist enterprise product / inventory item entity."""

    __tablename__ = "inventory_items"

    item_sku = Column(String(64), primary_key=True, index=True)
    item_name = Column(String(255), nullable=False)
    category = Column(String(64), nullable=False, index=True)
    weight_g = Column(Float, default=500.0)
    unit_cost = Column(Float, nullable=False)
    stock_level = Column(Integer, default=100)
    reorder_threshold = Column(Integer, default=20)

    purchase_orders = relationship("PurchaseOrderModel", back_populates="item")


class PurchaseOrderModel(Base):
    """Olist enterprise purchase order and line items entity."""

    __tablename__ = "purchase_orders"

    po_id = Column(String(64), primary_key=True, index=True)
    vendor_id = Column(String(64), ForeignKey("vendors.vendor_id"), nullable=False, index=True)
    item_sku = Column(String(64), ForeignKey("inventory_items.item_sku"), nullable=False, index=True)
    quantity = Column(Integer, default=1)
    unit_price = Column(Float, nullable=False)
    freight_value = Column(Float, default=15.0)
    order_status = Column(String(32), default="Approved", index=True)
    order_date = Column(String(32), nullable=False)

    vendor = relationship("VendorModel", back_populates="purchase_orders")
    item = relationship("InventoryItemModel", back_populates="purchase_orders")


class BudgetCapModel(Base):
    """Enterprise quarterly departmental spend ceilings."""

    __tablename__ = "budget_caps"

    category = Column(String(64), primary_key=True)
    fiscal_quarter = Column(String(32), primary_key=True)
    allocated_budget = Column(Float, nullable=False)


class CloudDatabaseConnectionError(RuntimeError):
    """Raised when the primary cloud database cannot be reached and unauthorized local fallback is blocked."""


# ------------------------------------------------------------------------------
# Database Connection & Engine Factory
# ------------------------------------------------------------------------------

class DatabaseManager:
    """Manages asynchronous database connections, migrations, and query execution."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._async_engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._libsql_client: Optional[Any] = None

    def is_turso(self) -> bool:
        """Check if target database is Turso libSQL."""
        return (
            self.settings.database_url.startswith("libsql://")
            or "turso.io" in self.settings.database_url
        )

    def get_turso_client(self) -> Optional[Any]:
        """Get or initialize singleton Turso libSQL client."""
        if self._libsql_client is None and self.is_turso() and self.settings.turso_auth_token:
            try:
                import libsql_client
                raw_url = self.settings.database_url
                if raw_url.startswith("libsql://"):
                    url = raw_url.replace("libsql://", "https://")
                else:
                    url = raw_url
                self._libsql_client = libsql_client.create_client_sync(
                    url=url,
                    auth_token=self.settings.turso_auth_token,
                )
            except Exception as e:
                logger.warning("Could not initialize Turso client (%s); will fallback to local SQLite.", e)
                self._libsql_client = None
        return self._libsql_client

    def execute_query(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute parameterized SQL query dynamically on Turso cloud.

        Strict Production Invariant:
        If configured for Turso Cloud, queries must execute on Turso.
        If Turso fails, fail-fast with CloudDatabaseConnectionError.
        Silent fallback to local SQLite is strictly forbidden unless allow_local_db_fallback=True.
        """
        if self.is_turso():
            turso_client = self.get_turso_client()
            if turso_client is None:
                if not self.settings.allow_local_db_fallback:
                    raise CloudDatabaseConnectionError(
                        "CRITICAL: Target is Turso Cloud, but Turso client could not be initialized "
                        "(check TURSO_AUTH_TOKEN and network connectivity). "
                        "Unauthorized local SQLite fallback is blocked. "
                        "Set ALLOW_LOCAL_DB_FALLBACK=true to explicitly authorize local execution."
                    )
            else:
                try:
                    res = turso_client.execute(sql, params or [])
                    columns = [c for c in res.columns]
                    return [dict(zip(columns, row)) for row in res.rows]
                except Exception as e:
                    if not self.settings.allow_local_db_fallback:
                        raise CloudDatabaseConnectionError(
                            f"CRITICAL: Turso Cloud query execution failed: {e}. "
                            "Unauthorized local SQLite fallback is blocked. "
                            "Set ALLOW_LOCAL_DB_FALLBACK=true to explicitly authorize local execution."
                        ) from e
                    logger.warning(
                        "AUTHORIZED FALLBACK: Turso cloud query failed (%s); reverting to local SQLite per user configuration.",
                        e,
                    )

        # Standard local execution path (e.g. when database_url is sqlite:/// or authorized fallback)
        import sqlite3
        local_db_path = Path(__file__).resolve().parent.parent.parent / "data" / "lexsi_production.db"
        if not local_db_path.exists():
            return []
        conn = sqlite3.connect(local_db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(sql, params or [])
            if cur.description:
                rows = cur.fetchall()
                return [dict(r) for r in rows]
            else:
                conn.commit()
                return []
        finally:
            conn.close()

    def _resolve_engine_url(self) -> str:
        """Resolve database URL for SQLAlchemy async engine."""
        url = self.settings.database_url

        # Turso uses libsql_client rather than SQLAlchemy dialect; fallback SQLAlchemy engine to local SQLite
        if self.is_turso():
            local_db_path = Path(__file__).resolve().parent.parent.parent / "data" / "lexsi_production.db"
            local_db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite+aiosqlite:///{local_db_path}"

        # Standard SQLite URL adaptation for aiosqlite
        if url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "sqlite+aiosqlite:///")

        return url

    def get_async_engine(self) -> AsyncEngine:
        """Get or create singleton AsyncEngine with connection pooling."""
        if self._async_engine is None:
            resolved_url = self._resolve_engine_url()
            connect_args: Dict[str, Any] = {}
            if "sqlite" in resolved_url:
                connect_args["check_same_thread"] = False

            self._async_engine = create_async_engine(
                resolved_url,
                echo=self.settings.db_echo_sql,
                connect_args=connect_args,
                pool_pre_ping=True,
            )
            self._session_factory = async_sessionmaker(
                bind=self._async_engine,
                expire_on_commit=False,
                autoflush=False,
            )
        return self._async_engine

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a transactional asynchronous database session."""
        self.get_async_engine()
        assert self._session_factory is not None
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error("Database session transaction rolled back: %s", e)
                raise e

    async def init_tables(self) -> None:
        """Create all relational tables if they do not exist."""
        engine = self.get_async_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database relational tables initialized successfully.")

    async def close(self) -> None:
        """Dispose of the database engine and connection pool."""
        if self._async_engine:
            await self._async_engine.dispose()
            self._async_engine = None
            self._session_factory = None


# Global singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Retrieve global singleton DatabaseManager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


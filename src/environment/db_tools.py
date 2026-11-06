"""Database-backed tool execution engine executing parameterized SQL queries.

Connects to relational database (Turso libSQL / SQLite / PostgreSQL) via SQLAlchemy.
Guarantees:
- 100% injection-free parameterized SQL queries.
- Strict Pydantic input validation (extra='forbid').
- Normalized structured outputs and typed errors.
- Fully compatible with existing tool contracts.
"""

import math
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.environment.database import (
    BudgetCapModel,
    InventoryItemModel,
    PurchaseOrderModel,
    VendorModel,
    get_db_manager,
)
from src.environment.sandbox import ToolObservation
from src.environment.schema import (
    SchemaValidationError,
    ToolExecutionError,
)
from src.environment.tools import (
    ComputeAggregateArgs,
    LookupRecordsArgs,
    VerifyConstraintArgs,
    validate_tool_arguments,
)

# Table to SQLAlchemy Model mapping
TABLE_MODEL_MAP = {
    "vendors": VendorModel,
    "inventory_items": InventoryItemModel,
    "purchase_orders": PurchaseOrderModel,
    "budget_caps": BudgetCapModel,
}


async def lookup_records_sql(
    table: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 50,
    session: Optional[AsyncSession] = None,
) -> List[Dict[str, Any]]:
    """Execute parameterized SELECT query over the relational database (Turso cloud or SQLite)."""
    model_cls = TABLE_MODEL_MAP.get(table)
    if not model_cls:
        raise SchemaValidationError(
            f"Table '{table}' does not exist. Available tables: {list(TABLE_MODEL_MAP.keys())}"
        )

    # Apply parameterized filter validation
    if filters:
        for field_name in filters.keys():
            if not hasattr(model_cls, field_name):
                raise SchemaValidationError(
                    f"Field '{field_name}' does not exist on table '{table}'."
                )

    db = get_db_manager()

    # If Turso cloud is configured and no explicit in-memory session passed, query Turso directly
    if session is None and db.is_turso() and db.settings.turso_auth_token:
        where_clauses: List[str] = []
        params: List[Any] = []
        if filters:
            for k, v in filters.items():
                where_clauses.append(f"{k} = ?")
                params.append(v)
        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query_sql = f"SELECT * FROM {table}{where_sql} LIMIT ?"
        params.append(limit)
        return db.execute_query(query_sql, params)

    # Standard SQLAlchemy execution for session-bound tests / Postgres / SQLite
    stmt = select(model_cls)
    if filters:
        for field_name, value in filters.items():
            column_attr = getattr(model_cls, field_name)
            stmt = stmt.where(column_attr == value)

    stmt = stmt.limit(limit)

    # Execute inside provided session or create a new one
    if session:
        result = await session.execute(stmt)
        rows = result.scalars().all()
    else:
        async with db.get_session() as sess:
            result = await sess.execute(stmt)
            rows = result.scalars().all()

    # Convert SQLAlchemy model instances to dictionaries
    serialized: List[Dict[str, Any]] = []
    for row in rows:
        row_dict = {
            col.name: getattr(row, col.name)
            for col in model_cls.__table__.columns
        }
        serialized.append(row_dict)

    return serialized


def compute_aggregate(values: List[float], operation: str) -> float:
    """Compute mathematical reduction over numerical array."""
    if not values and operation != "count":
        raise ValueError(f"Cannot perform aggregation '{operation}' on empty values list.")

    op = operation.lower()
    if op == "sum":
        return round(float(sum(values)), 4)
    elif op == "mean":
        return round(float(sum(values) / len(values)), 4)
    elif op == "min":
        return round(float(min(values)), 4)
    elif op == "max":
        return round(float(max(values)), 4)
    elif op == "count":
        return float(len(values))
    elif op == "std":
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return round(float(math.sqrt(variance)), 4)
    else:
        raise ValueError(
            f"Unsupported operation '{operation}'. Supported: sum, mean, min, max, count, std"
        )


def verify_constraint(
    actual: float,
    expected: float,
    comparison: str,
    tolerance: float = 0.001,
) -> bool:
    """Objectively verify numerical relationship."""
    cmp = comparison.lower()
    diff = actual - expected

    if cmp == "eq":
        return abs(diff) <= tolerance
    elif cmp == "le":
        return actual <= expected + tolerance
    elif cmp == "ge":
        return actual >= expected - tolerance
    elif cmp == "lt":
        return actual < expected
    elif cmp == "gt":
        return actual > expected
    else:
        raise ValueError(f"Unsupported comparison '{comparison}'. Supported: eq, le, ge, lt, gt")


class DatabaseToolEngine:
    """Asynchronous engine wrapping database tools with call counters and error normalization."""

    def __init__(self) -> None:
        self.tool_call_counts: Dict[str, int] = {
            "lookup_records": 0,
            "compute_aggregate": 0,
            "verify_constraint": 0,
        }

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session: Optional[AsyncSession] = None,
    ) -> ToolObservation:
        """Execute tool asynchronously against the production relational database."""
        start_time = time.time()

        if tool_name not in self.tool_call_counts:
            return ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                is_error=True,
                error_code="UNKNOWN_TOOL_ERROR",
                error_message=f"Unknown tool: '{tool_name}'",
                elapsed_ms=(time.time() - start_time) * 1000.0,
            )

        try:
            # 1. Pre-execution argument validation with extra='forbid'
            validate_tool_arguments(tool_name, arguments)

            if tool_name == "lookup_records":
                validated_lookup = LookupRecordsArgs.model_validate(arguments)
                result = await lookup_records_sql(
                    table=validated_lookup.table,
                    filters=validated_lookup.filters,
                    session=session,
                )
                self.tool_call_counts["lookup_records"] += 1
                return ToolObservation(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    is_error=False,
                    elapsed_ms=(time.time() - start_time) * 1000.0,
                )

            elif tool_name == "compute_aggregate":
                validated_agg = ComputeAggregateArgs.model_validate(arguments)
                if not validated_agg.values and validated_agg.operation != "count":
                    return ToolObservation(
                        tool_name=tool_name,
                        arguments=arguments,
                        result=None,
                        is_error=True,
                        error_code="ARGUMENT_SEMANTIC_ERROR",
                        error_message="Cannot perform aggregation on empty values list.",
                        elapsed_ms=(time.time() - start_time) * 1000.0,
                    )
                agg_val = compute_aggregate(
                    values=validated_agg.values,
                    operation=validated_agg.operation,
                )
                self.tool_call_counts["compute_aggregate"] += 1
                return ToolObservation(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=agg_val,
                    is_error=False,
                    elapsed_ms=(time.time() - start_time) * 1000.0,
                )

            elif tool_name == "verify_constraint":
                validated_verify = VerifyConstraintArgs.model_validate(arguments)
                is_valid = verify_constraint(
                    actual=validated_verify.actual,
                    expected=validated_verify.expected,
                    comparison=validated_verify.comparison,
                    tolerance=validated_verify.tolerance,
                )
                self.tool_call_counts["verify_constraint"] += 1
                return ToolObservation(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=is_valid,
                    is_error=False,
                    elapsed_ms=(time.time() - start_time) * 1000.0,
                )

            else:
                return ToolObservation(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=None,
                    is_error=True,
                    error_code="UNKNOWN_TOOL_ERROR",
                    error_message=f"Unimplemented tool: {tool_name}",
                    elapsed_ms=(time.time() - start_time) * 1000.0,
                )

        except ToolExecutionError as e:
            return ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                is_error=True,
                error_code=e.error_code,
                error_message=e.message,
                elapsed_ms=(time.time() - start_time) * 1000.0,
            )
        except Exception as e:
            return ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                is_error=True,
                error_code="TOOL_EXECUTION_ERROR",
                error_message=f"{type(e).__name__}: {str(e)}",
                elapsed_ms=(time.time() - start_time) * 1000.0,
            )

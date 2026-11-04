"""Deterministic tool implementations and Pydantic argument models."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field, ValidationError

from src.environment.schema import (
    SchemaValidationError,
    ToolContract,
    ToolParameterSchema,
)


class LookupRecordsArgs(BaseModel):
    """Arguments for lookup_records tool."""

    table: Literal["vendors", "inventory_items", "purchase_orders", "budget_caps"] = Field(
        ..., description="Target database table name"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value equality/containment filters",
    )


class ComputeAggregateArgs(BaseModel):
    """Arguments for compute_aggregate tool."""

    values: List[float] = Field(
        ...,
        description="Array of numeric values to reduce",
    )
    operation: Literal["sum", "mean", "min", "max", "count", "std"] = Field(
        ...,
        description="Reduction operator",
    )


class VerifyConstraintArgs(BaseModel):
    """Arguments for verify_constraint tool."""

    actual: float = Field(..., description="Computed metric or actual spend")
    expected: float = Field(..., description="Baseline, threshold, or budget cap")
    comparison: Literal["eq", "le", "ge", "lt", "gt"] = Field(
        ...,
        description="Comparison predicate: eq (==), le (<=), ge (>=), lt (<), gt (>)",
    )
    tolerance: float = Field(
        0.001,
        description="Absolute error tolerance for equality comparison",
    )


def execute_lookup_records(
    db: Dict[str, List[Dict[str, Any]]],
    arguments: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Execute lookup_records query against the in-memory database."""
    try:
        args = LookupRecordsArgs.model_validate(arguments)
    except ValidationError as e:
        raise SchemaValidationError(str(e)) from e

    if args.table not in db:
        raise SchemaValidationError(f"Table '{args.table}' does not exist.")

    table_data = db[args.table]
    results: List[Dict[str, Any]] = []

    for row in table_data:
        match = True
        for key, expected_val in args.filters.items():
            if key not in row:
                match = False
                break
            if row[key] != expected_val:
                match = False
                break
        if match:
            results.append(row)

    # Deterministic sort by primary key or first field
    if results:
        sort_key = next(iter(results[0].keys()))
        results.sort(key=lambda r: str(r.get(sort_key, "")))

    return results


def execute_compute_aggregate(arguments: Dict[str, Any]) -> float:
    """Execute compute_aggregate arithmetic reduction."""
    try:
        args = ComputeAggregateArgs.model_validate(arguments)
    except ValidationError as e:
        raise SchemaValidationError(str(e)) from e

    if not args.values:
        if args.operation == "count":
            return 0.0
        raise SchemaValidationError(f"Cannot perform '{args.operation}' on empty values list.")

    vals = args.values
    if args.operation == "sum":
        return round(float(sum(vals)), 4)
    if args.operation == "mean":
        return round(float(sum(vals) / len(vals)), 4)
    if args.operation == "min":
        return round(float(min(vals)), 4)
    if args.operation == "max":
        return round(float(max(vals)), 4)
    if args.operation == "count":
        return float(len(vals))
    if args.operation == "std":
        if len(vals) < 2:
            return 0.0
        mean_val = sum(vals) / len(vals)
        variance = sum((x - mean_val) ** 2 for x in vals) / (len(vals) - 1)
        return round(float(math.sqrt(variance)), 4)

    raise SchemaValidationError(f"Unsupported operation '{args.operation}'.")


def execute_verify_constraint(arguments: Dict[str, Any]) -> bool:
    """Execute verify_constraint numerical comparison."""
    try:
        args = VerifyConstraintArgs.model_validate(arguments)
    except ValidationError as e:
        raise SchemaValidationError(str(e)) from e

    diff = args.actual - args.expected
    if args.comparison == "eq":
        return abs(diff) <= args.tolerance
    if args.comparison == "le":
        return args.actual <= (args.expected + args.tolerance)
    if args.comparison == "ge":
        return args.actual >= (args.expected - args.tolerance)
    if args.comparison == "lt":
        return args.actual < args.expected
    if args.comparison == "gt":
        return args.actual > args.expected

    raise SchemaValidationError(f"Unsupported comparison '{args.comparison}'.")


TOOL_CONTRACTS: Dict[str, ToolContract] = {
    "lookup_records": ToolContract(
        name="lookup_records",
        description="Query in-memory procurement database table using exact key-value filters.",
        parameters=[
            ToolParameterSchema(
                name="table",
                type="string",
                description="Target table name",
                required=True,
                enum=["vendors", "inventory_items", "purchase_orders", "budget_caps"],
            ),
            ToolParameterSchema(
                name="filters",
                type="dict",
                description="Key-value equality/containment filters",
                required=False,
            ),
        ],
        is_stateful=False,
        idempotent=True,
    ),
    "compute_aggregate": ToolContract(
        name="compute_aggregate",
        description="Perform arithmetic reduction (sum, mean, min, max, count, std) on numbers.",
        parameters=[
            ToolParameterSchema(
                name="values",
                type="list",
                description="Array of numeric values to reduce",
                required=True,
            ),
            ToolParameterSchema(
                name="operation",
                type="string",
                description="Reduction operation",
                required=True,
                enum=["sum", "mean", "min", "max", "count", "std"],
            ),
        ],
        is_stateful=False,
        idempotent=True,
    ),
    "verify_constraint": ToolContract(
        name="verify_constraint",
        description="Verify mathematical constraint between actual value and expected baseline.",
        parameters=[
            ToolParameterSchema(
                name="actual",
                type="float",
                description="Computed metric or actual spend",
                required=True,
            ),
            ToolParameterSchema(
                name="expected",
                type="float",
                description="Baseline, threshold, or budget cap",
                required=True,
            ),
            ToolParameterSchema(
                name="comparison",
                type="string",
                description="Comparison operator",
                required=True,
                enum=["eq", "le", "ge", "lt", "gt"],
            ),
            ToolParameterSchema(
                name="tolerance",
                type="float",
                description="Absolute error tolerance for equality",
                required=False,
            ),
        ],
        is_stateful=False,
        idempotent=True,
    ),
}

"""Deterministic tool implementations and Pydantic argument models."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Type
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.environment.schema import (
    SchemaValidationError,
    SemanticBoundError,
    ToolContract,
    ToolNotFoundError,
    ToolParameterSchema,
)


class StrictToolArgs(BaseModel):
    """Base class enforcing strict rejection of unexpected/extraneous parameters."""

    model_config = ConfigDict(extra="forbid")


class LookupRecordsArgs(StrictToolArgs):
    """Arguments for lookup_records tool."""

    table: Literal["vendors", "inventory_items", "purchase_orders", "budget_caps"] = Field(
        ..., description="Target database table name"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value equality/containment filters",
    )


class ComputeAggregateArgs(StrictToolArgs):
    """Arguments for compute_aggregate tool."""

    values: List[float] = Field(
        ...,
        description="Array of numeric values to reduce",
    )
    operation: Literal["sum", "mean", "min", "max", "count", "std"] = Field(
        ...,
        description="Reduction operator",
    )


class VerifyConstraintArgs(StrictToolArgs):
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


ARGUMENT_MODELS: Dict[str, Type[StrictToolArgs]] = {
    "lookup_records": LookupRecordsArgs,
    "compute_aggregate": ComputeAggregateArgs,
    "verify_constraint": VerifyConstraintArgs,
}


def validate_tool_arguments(tool_name: str, arguments: Dict[str, Any]) -> StrictToolArgs:
    """Strictly validate tool arguments against Pydantic schema and semantic boundaries.

    Guarantees:
    - Enforces extra='forbid' to reject unexpected fields.
    - Validates data types, required fields, and enum values.
    - Enforces domain semantic bounds (finite floats, non-negative tolerances).
    """
    if tool_name not in ARGUMENT_MODELS:
        raise ToolNotFoundError(tool_name)

    if not isinstance(arguments, dict):
        raise SchemaValidationError(f"Arguments must be a dictionary, got {type(arguments).__name__}")

    model_cls = ARGUMENT_MODELS[tool_name]
    try:
        validated = model_cls.model_validate(arguments)
    except ValidationError as e:
        raise SchemaValidationError(str(e)) from e

    # Domain semantic boundaries validation
    if isinstance(validated, ComputeAggregateArgs):
        if not validated.values and validated.operation != "count":
            raise SchemaValidationError(f"Cannot perform '{validated.operation}' on empty values list.")
        if not all(math.isfinite(v) for v in validated.values):
            raise SemanticBoundError("All values in compute_aggregate must be finite real numbers.")
    elif isinstance(validated, VerifyConstraintArgs):
        if not math.isfinite(validated.actual) or not math.isfinite(validated.expected):
            raise SemanticBoundError("Comparison values 'actual' and 'expected' must be finite numbers.")
        if validated.tolerance < 0.0 or not math.isfinite(validated.tolerance):
            raise SemanticBoundError(f"Tolerance must be non-negative and finite, got {validated.tolerance}.")

    return validated


def execute_lookup_records(
    db: Dict[str, List[Dict[str, Any]]],
    arguments: Dict[str, Any] | LookupRecordsArgs,
) -> List[Dict[str, Any]]:
    """Execute lookup_records query against the in-memory database."""
    if isinstance(arguments, LookupRecordsArgs):
        args = arguments
    else:
        validated = validate_tool_arguments("lookup_records", arguments)
        assert isinstance(validated, LookupRecordsArgs)
        args = validated

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


def execute_compute_aggregate(arguments: Dict[str, Any] | ComputeAggregateArgs) -> float:
    """Execute compute_aggregate arithmetic reduction."""
    if isinstance(arguments, ComputeAggregateArgs):
        args = arguments
    else:
        validated = validate_tool_arguments("compute_aggregate", arguments)
        assert isinstance(validated, ComputeAggregateArgs)
        args = validated

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


def execute_verify_constraint(arguments: Dict[str, Any] | VerifyConstraintArgs) -> bool:
    """Execute verify_constraint numerical comparison."""
    if isinstance(arguments, VerifyConstraintArgs):
        args = arguments
    else:
        validated = validate_tool_arguments("verify_constraint", arguments)
        assert isinstance(validated, VerifyConstraintArgs)
        args = validated

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
        permission_level="read_only",
        result_schema={"type": "list", "items": {"type": "object"}},
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
        permission_level="read_only",
        result_schema={"type": "float"},
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
        permission_level="read_only",
        result_schema={"type": "boolean"},
    ),
}


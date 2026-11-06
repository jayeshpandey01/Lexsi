"""Sandboxed deterministic tool execution engine."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from src.environment.mock_data import get_mock_database
from src.environment.schema import (
    CallTimeoutError,
    DisallowedToolError,
    ResetReplayError,
    SchemaValidationError,
    SemanticBoundError,
    ToolExecutionError,
    ToolNotFoundError,
)
from src.environment.tools import (
    TOOL_CONTRACTS,
    StrictToolArgs,
    execute_compute_aggregate,
    execute_lookup_records,
    execute_verify_constraint,
    validate_tool_arguments,
)


class ToolObservation(BaseModel):
    """Structured response returned by tool execution engine."""

    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    is_error: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    elapsed_ms: float = 0.0


class SandboxedToolEngine:
    """Isolated, deterministic in-memory tool execution engine.

    Guarantees:
    - Zero external network, filesystem write, or OS shell execution.
    - Pre-execution validation boundary: malformed or disallowed calls never reach tool logic.
    - Strict call counting for observable assertion verification.
    - Bounded per-call timeout enforcement.
    - Idempotent and reversible to seed s.
    """

    def __init__(
        self,
        seed: int = 42,
        allowed_tools: Optional[List[str] | Set[str]] = None,
        per_call_timeout_seconds: float = 5.0,
        use_database: bool = False,
    ) -> None:
        self.seed = seed
        self.per_call_timeout_seconds = per_call_timeout_seconds
        self.use_database = use_database
        self.allowed_tools: Set[str] = set(allowed_tools or list(TOOL_CONTRACTS.keys()))
        self.db = get_mock_database(seed)
        self.call_history: List[ToolObservation] = []
        self.tool_call_counts: Dict[str, int] = {name: 0 for name in TOOL_CONTRACTS}

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset database, call counts, and execution history to specified seed."""
        try:
            if seed is not None:
                self.seed = seed
            self.db = get_mock_database(self.seed)
            self.call_history.clear()
            self.tool_call_counts = {name: 0 for name in TOOL_CONTRACTS}
        except Exception as e:
            raise ResetReplayError(f"Failed to reset environment with seed {seed}: {e}") from e

    def validate_call(self, tool_name: str, arguments: Dict[str, Any]) -> StrictToolArgs:
        """Enforce strict validation boundary prior to executing tool.

        Checks:
        1. Tool existence in registry.
        2. Permission and authorization allowlist.
        3. Pydantic schema validation with extra='forbid'.
        4. Domain semantic bounds.
        """
        # 1. Registration check
        if tool_name not in TOOL_CONTRACTS:
            raise ToolNotFoundError(tool_name)

        # 2. Authorization check
        if tool_name not in self.allowed_tools:
            raise DisallowedToolError(tool_name)

        # 3 & 4. Schema and semantic validation
        return validate_tool_arguments(tool_name, arguments)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolObservation:
        """Execute a tool call safely within the sandboxed environment with pre-validation."""
        start_time = time.perf_counter()

        # Step A: Pre-execution validation boundary
        # If this step raises, the tool function is NEVER invoked.
        try:
            validated_args = self.validate_call(tool_name, arguments)
        except ToolExecutionError as e:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                is_error=True,
                error_code=e.error_code,
                error_message=e.message,
                elapsed_ms=round(elapsed, 3),
            )
            self.call_history.append(obs)
            return obs
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                is_error=True,
                error_code="SCHEMA_VALIDATION_ERROR",
                error_message=f"Validation failed: {str(e)}",
                elapsed_ms=round(elapsed, 3),
            )
            self.call_history.append(obs)
            return obs

        # Step B: Execution routing (observable call counter increments only here)
        self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1

        try:
            if tool_name == "lookup_records":
                if self.use_database:
                    from src.environment.database import get_db_manager
                    db = get_db_manager()
                    where_clauses: List[str] = []
                    params: List[Any] = []
                    for k, v in getattr(validated_args, "filters", {}).items():
                        where_clauses.append(f"{k} = ?")
                        params.append(v)
                    where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                    query_sql = f"SELECT * FROM {getattr(validated_args, 'table', 'vendors')}{where_sql} LIMIT 50"
                    res = db.execute_query(query_sql, params)
                    if res:
                        sort_key = next(iter(res[0].keys()))
                        res.sort(key=lambda r: str(r.get(sort_key, "")))
                else:
                    res = execute_lookup_records(self.db, validated_args)
            elif tool_name == "compute_aggregate":
                res = execute_compute_aggregate(validated_args)
            elif tool_name == "verify_constraint":
                res = execute_verify_constraint(validated_args)
            else:
                raise ToolNotFoundError(tool_name)

            elapsed_s = time.perf_counter() - start_time
            if elapsed_s > self.per_call_timeout_seconds:
                raise CallTimeoutError(tool_name, self.per_call_timeout_seconds)

            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                result=res,
                is_error=False,
                elapsed_ms=round(elapsed_s * 1000.0, 3),
            )
        except ToolExecutionError as e:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                is_error=True,
                error_code=e.error_code,
                error_message=e.message,
                elapsed_ms=round(elapsed, 3),
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                is_error=True,
                error_code="UNEXPECTED_RUNTIME_ERROR",
                error_message=str(e),
                elapsed_ms=round(elapsed, 3),
            )

        self.call_history.append(obs)
        return obs


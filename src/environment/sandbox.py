"""Sandboxed deterministic tool execution engine."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.environment.mock_data import get_mock_database
from src.environment.schema import (
    DisallowedToolError,
    ToolExecutionError,
    ToolNotFoundError,
)
from src.environment.tools import (
    TOOL_CONTRACTS,
    execute_compute_aggregate,
    execute_lookup_records,
    execute_verify_constraint,
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
    - Strict validation against declared Pydantic ToolContracts.
    - Idempotent and reversible to seed s.
    """

    def __init__(self, seed: int = 42, allowed_tools: Optional[List[str]] = None) -> None:
        self.seed = seed
        self.db = get_mock_database(seed)
        self.allowed_tools = set(allowed_tools or list(TOOL_CONTRACTS.keys()))
        self.call_history: List[ToolObservation] = []

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset database and execution history to specified seed."""
        if seed is not None:
            self.seed = seed
        self.db = get_mock_database(self.seed)
        self.call_history.clear()

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolObservation:
        """Execute a tool call safely within the sandboxed environment."""
        start_time = time.perf_counter()

        # 1. Authorization check
        if tool_name not in TOOL_CONTRACTS:
            err = ToolNotFoundError(tool_name)
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                is_error=True,
                error_code=err.error_code,
                error_message=err.message,
                elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
            )
            self.call_history.append(obs)
            return obs

        if tool_name not in self.allowed_tools:
            err = DisallowedToolError(tool_name)
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                is_error=True,
                error_code=err.error_code,
                error_message=err.message,
                elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
            )
            self.call_history.append(obs)
            return obs

        # 2. Execution routing with typed exception capture
        try:
            if tool_name == "lookup_records":
                res = execute_lookup_records(self.db, arguments)
            elif tool_name == "compute_aggregate":
                res = execute_compute_aggregate(arguments)
            elif tool_name == "verify_constraint":
                res = execute_verify_constraint(arguments)
            else:
                raise ToolNotFoundError(tool_name)

            elapsed = (time.perf_counter() - start_time) * 1000.0
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                result=res,
                is_error=False,
                elapsed_ms=round(elapsed, 3),
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

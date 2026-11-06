"""Runtime policy guards, step budgets, timeouts, and delimiter sanitization."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Set
from src.environment.schema import DisallowedToolError, ToolExecutionError


class StepLimitExceededError(ToolExecutionError):
    """Raised when an episode exceeds the configured maximum step budget (T_max)."""

    def __init__(self, step_index: int, max_steps: int) -> None:
        super().__init__(
            f"Step limit exceeded: attempted step {step_index} with max limit {max_steps}.",
            "STEP_LIMIT_EXCEEDED",
        )


class ExecutionTimeoutError(ToolExecutionError):
    """Raised when an episode exceeds the maximum wall-clock execution budget."""

    def __init__(self, elapsed_s: float, max_s: float) -> None:
        super().__init__(
            f"Execution deadline breached: elapsed {elapsed_s:.2f}s with max limit {max_s:.2f}s.",
            "EXECUTION_TIMEOUT",
        )


class RuntimePolicyGuard:
    """Runtime execution guard protecting agent-tool boundaries.

    Invariants enforced:
    - Step budget: Halts after T_max (default 8) steps.
    - Wall clock: Halts after timeout (default 30.0 seconds).
    - Tool allowlist: Rejects disallowed tools outside the model runtime.
    - Delimiter sanitization: Escapes model special tokens from tool observations.
    """

    def __init__(
        self,
        allowed_tools: Optional[Set[str]] = None,
        max_steps: int = 8,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.allowed_tools = allowed_tools or {"lookup_records", "compute_aggregate", "verify_constraint"}
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds

    def check_budget(self, current_step: int, start_time: float) -> None:
        """Verify remaining step count and elapsed time."""
        if current_step >= self.max_steps:
            raise StepLimitExceededError(current_step, self.max_steps)

        elapsed = time.time() - start_time
        if elapsed > self.timeout_seconds:
            raise ExecutionTimeoutError(elapsed, self.timeout_seconds)

    def validate_tool_permission(self, tool_name: str) -> None:
        """Enforce strict tool allowlist outside the model."""
        if tool_name not in self.allowed_tools:
            raise DisallowedToolError(tool_name)

    def sanitize_observation(self, observation: Any) -> str:
        """Escape model prompt tokens and delimiter sequences to mitigate prompt injection."""
        if isinstance(observation, (dict, list)):
            text = json.dumps(observation)
        else:
            text = str(observation)

        # Defuse Cohere and standard special token sequences
        replacements = [
            ("<|START_OF_TURN_TOKEN|>", "[ESCAPED_TOKEN]"),
            ("<|END_OF_TURN_TOKEN|>", "[ESCAPED_TOKEN]"),
            ("<|START_ACTION|>", "[ESCAPED_TOKEN]"),
            ("<|END_ACTION|>", "[ESCAPED_TOKEN]"),
            ("<|USER_TOKEN|>", "[ESCAPED_TOKEN]"),
            ("<|SYSTEM_TOKEN|>", "[ESCAPED_TOKEN]"),
            ("<|CHATBOT_TOKEN|>", "[ESCAPED_TOKEN]"),
        ]
        for token, rep in replacements:
            text = text.replace(token, rep)

        return text


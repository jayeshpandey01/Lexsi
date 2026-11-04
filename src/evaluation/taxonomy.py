"""The 6-class failure mode taxonomy classifier."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from src.data.schemas import TrajectoryRecord
from src.environment.task_generator import TaskSpec


class FailureCategory(str, Enum):
    """The six discrete failure categories defined in system_design.md §2.5."""

    TOOL_SELECTION_FAILURE = "ToolSelectionFailure"
    ARGUMENT_SCHEMA_FAILURE = "ArgumentSchemaFailure"
    DEPENDENCY_ORDER_FAILURE = "DependencyOrderFailure"
    LOOPING_AND_STALL_FAILURE = "LoopingAndStallFailure"
    EARLY_TERMINATION_FAILURE = "EarlyTerminationFailure"
    TOOL_RECOVERY_FAILURE = "ToolRecoveryFailure"


def classify_failure(
    trajectory: TrajectoryRecord,
    task: Optional[TaskSpec] = None,
    max_steps: int = 8,
) -> Optional[FailureCategory]:
    """Diagnose and classify the root-cause failure mode of an unsuccessful trajectory.

    Returns None if trajectory is marked as successful.
    """
    if trajectory.is_success:
        return None

    # Check for empty / zero-tool execution
    if not trajectory.steps:
        return FailureCategory.EARLY_TERMINATION_FAILURE

    # Check for looping or max-step stall
    if trajectory.total_steps >= max_steps:
        return FailureCategory.LOOPING_AND_STALL_FAILURE

    # Detect duplicate identical calls (3 or more identical tool actions)
    seen_calls = [
        (s.action.tool_name, str(sorted(s.action.arguments.items())))
        for s in trajectory.steps
        if s.action is not None
    ]
    for call in seen_calls:
        if seen_calls.count(call) >= 3:
            return FailureCategory.LOOPING_AND_STALL_FAILURE

    # Inspect step errors
    for idx, step in enumerate(trajectory.steps):
        # 1. Unknown or unauthorized tool
        if step.error and "TOOL_NOT_FOUND" in step.error:
            return FailureCategory.TOOL_SELECTION_FAILURE

        # 2. Schema validation error
        if step.error and "SCHEMA_VALIDATION_ERROR" in step.error:
            return FailureCategory.ARGUMENT_SCHEMA_FAILURE

        # 3. Tool recovery failure (observation had error and agent immediately aborted)
        if step.error and idx == len(trajectory.steps) - 1:
            return FailureCategory.TOOL_RECOVERY_FAILURE

    # Check dependency ordering (e.g., aggregate before lookup)
    called_tools = [s.action.tool_name for s in trajectory.steps if s.action]
    if "compute_aggregate" in called_tools and "lookup_records" in called_tools:
        if called_tools.index("compute_aggregate") < called_tools.index("lookup_records"):
            return FailureCategory.DEPENDENCY_ORDER_FAILURE

    # Check tool selection against task expected tools
    if task:
        for tool in called_tools:
            if tool not in task.expected_tools:
                return FailureCategory.TOOL_SELECTION_FAILURE

    # Default to early termination if task predicate wasn't met
    return FailureCategory.EARLY_TERMINATION_FAILURE

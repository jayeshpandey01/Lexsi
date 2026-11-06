"""The 6-class failure mode taxonomy classifier and diagnostic engine."""

from __future__ import annotations

from enum import Enum
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.data.schemas import StepRecord, ToolCallAction, TrajectoryRecord
from src.environment.task_generator import TaskSpec


class FailureCategory(str, Enum):
    """The six discrete failure categories defined in system_design.md §2.5."""

    TOOL_SELECTION_FAILURE = "ToolSelectionFailure"
    ARGUMENT_SCHEMA_FAILURE = "ArgumentSchemaFailure"
    DEPENDENCY_ORDER_FAILURE = "DependencyOrderFailure"
    LOOPING_AND_STALL_FAILURE = "LoopingAndStallFailure"
    EARLY_TERMINATION_FAILURE = "EarlyTerminationFailure"
    TOOL_RECOVERY_FAILURE = "ToolRecoveryFailure"


class FailureDiagnosis(BaseModel):
    """Structured diagnostic verdict for an unsuccessful agent trajectory."""

    category: Optional[FailureCategory] = Field(
        None,
        description="Diagnosed root-cause failure category, or None if successful or unclassifiable",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Diagnostic confidence score based on observable evidence",
    )
    root_cause_step: Optional[int] = Field(
        None,
        description="0-indexed step number where the primary failure occurred",
    )
    evidence: str = Field(
        ...,
        description="Observable evidence string including step index, error code, or tool name",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic metadata: called tools, error codes, repeated call counts",
    )


def canonical_call_key(action: ToolCallAction) -> str:
    """Generate deterministic, hashable representation of a tool call handling nested args."""
    return json.dumps({"name": action.tool_name, "args": action.arguments}, sort_keys=True)


KNOWN_SELECTION_CODES: Set[str] = {
    "TOOL_NOT_FOUND",
    "DISALLOWED_TOOL",
    "MALFORMED_TOOL_ACTION_MISSING_NAME",
}

KNOWN_SCHEMA_CODES: Set[str] = {
    "SCHEMA_VALIDATION_ERROR",
    "SEMANTIC_BOUND_ERROR",
    "MALFORMED_TOOL_ARGUMENTS",
    "MALFORMED_TOOL_ACTION_NOT_A_DICT",
    "JSON_PARSE_ERROR",
    "TRUNCATED_ACTION_TOKEN",
    "MALFORMED_ACTION_TOKEN",
}

KNOWN_STALL_CODES: Set[str] = {
    "STEP_LIMIT_EXCEEDED",
    "EXECUTION_TIMEOUT",
}


def _extract_error_codes(step: StepRecord) -> List[str]:
    """Extract known structured error codes from step record."""
    if not step.error:
        return []
    codes: List[str] = []
    all_codes = KNOWN_SELECTION_CODES | KNOWN_SCHEMA_CODES | KNOWN_STALL_CODES | {
        "CALL_TIMEOUT",
        "RESET_REPLAY_ERROR",
        "UNEXPECTED_RUNTIME_ERROR",
    }
    for code in all_codes:
        if code in step.error:
            codes.append(code)
    return codes


def diagnose_failure(
    trajectory: TrajectoryRecord,
    task: Optional[TaskSpec] = None,
    max_steps: int = 8,
) -> FailureDiagnosis:
    """Diagnose and classify the root-cause failure mode with observable evidence.

    Follows the formal precedence hierarchy defined in system_design.md §2.5:
    1. Early termination on 0 tool steps.
    2. Tool recovery failure on unadapted tool error.
    3. Tool selection failure (unknown or disallowed tool).
    4. Argument schema or semantic boundary violation.
    5. Dependency order violation.
    6. Looping and stall failure (identical calls, cycles, step limit).
    7. Early termination on incomplete expected tool subset.
    8. None (insufficient evidence or calculation error outside tool execution).
    """
    if trajectory.is_success:
        return FailureDiagnosis(
            category=None,
            confidence=1.0,
            root_cause_step=None,
            evidence="Trajectory is marked as successful; zero failure detected.",
            details={"is_success": True},
        )

    # 1. Check for empty trajectory or immediate termination before any tool actions
    called_actions = [s.action for s in trajectory.steps if s.action is not None]
    if not trajectory.steps or len(called_actions) == 0:
        return FailureDiagnosis(
            category=FailureCategory.EARLY_TERMINATION_FAILURE,
            confidence=1.0,
            root_cause_step=0 if trajectory.steps else None,
            evidence="Trajectory terminated with 0 tool calls executed before completing task.",
            details={"total_steps": len(trajectory.steps), "tool_actions_count": 0},
        )

    # 2. Check for ToolRecoveryFailure:
    # Occurs when step k encountered an error observation, and in step k+1 the agent:
    # (a) repeated the identical failing action without modification,
    # (b) aborted/stopped immediately without completing the task, or
    # (c) crashed with unparseable output.
    for k in range(len(trajectory.steps) - 1):
        step_k = trajectory.steps[k]
        k_codes = _extract_error_codes(step_k)
        if step_k.error is not None or len(k_codes) > 0:
            step_next = trajectory.steps[k + 1]
            # Case (a): Repeated identical failing call
            if step_k.action is not None and step_next.action is not None:
                if canonical_call_key(step_k.action) == canonical_call_key(step_next.action):
                    return FailureDiagnosis(
                        category=FailureCategory.TOOL_RECOVERY_FAILURE,
                        confidence=1.0,
                        root_cause_step=k + 1,
                        evidence=(
                            f"Step {k + 1} repeated identical failing call '{step_next.action.tool_name}' "
                            f"without modification after error at step {k}."
                        ),
                        details={
                            "failing_tool": step_k.action.tool_name,
                            "prior_error": step_k.error,
                            "repeated_step": k + 1,
                        },
                    )
            # Case (b): Aborted immediately after error without satisfying expected tools
            if step_next.action is None and trajectory.final_output is not None:
                return FailureDiagnosis(
                    category=FailureCategory.TOOL_RECOVERY_FAILURE,
                    confidence=0.95,
                    root_cause_step=k + 1,
                    evidence=(
                        f"Agent aborted with final output at step {k + 1} immediately following "
                        f"error at step {k} without recovering."
                    ),
                    details={"error_step": k, "prior_error": step_k.error, "abort_step": k + 1},
                )
            # Case (c): Crashed with parse/schema error in generation at step k+1
            next_codes = _extract_error_codes(step_next)
            if step_next.action is None and (
                "MALFORMED_TOOL_ARGUMENTS" in next_codes
                or "JSON_PARSE_ERROR" in next_codes
                or (step_next.error and "EMPTY_MODEL_OUTPUT" in step_next.error)
            ):
                return FailureDiagnosis(
                    category=FailureCategory.TOOL_RECOVERY_FAILURE,
                    confidence=0.9,
                    root_cause_step=k + 1,
                    evidence=(
                        f"Agent crashed at step {k + 1} following error at step {k}: {step_next.error}"
                    ),
                    details={"error_step": k, "crash_step": k + 1},
                )

    # 3. Check for ToolSelectionFailure or ArgumentSchemaFailure from first fatal step error
    for idx, step in enumerate(trajectory.steps):
        codes = _extract_error_codes(step)
        # Check ToolSelectionFailure
        if any(c in KNOWN_SELECTION_CODES for c in codes):
            return FailureDiagnosis(
                category=FailureCategory.TOOL_SELECTION_FAILURE,
                confidence=1.0,
                root_cause_step=idx,
                evidence=f"Step {idx} attempted invalid tool selection: {step.error}",
                details={"error_codes": codes, "step_index": idx},
            )
        # Check tool selection against task expected tools
        if step.action and task and task.expected_tools:
            if step.action.tool_name not in task.expected_tools:
                return FailureDiagnosis(
                    category=FailureCategory.TOOL_SELECTION_FAILURE,
                    confidence=0.95,
                    root_cause_step=idx,
                    evidence=(
                        f"Step {idx} invoked tool '{step.action.tool_name}' which is not in "
                        f"task expected tools: {task.expected_tools}"
                    ),
                    details={
                        "invoked_tool": step.action.tool_name,
                        "expected_tools": task.expected_tools,
                    },
                )
        # Check ArgumentSchemaFailure
        if any(c in KNOWN_SCHEMA_CODES for c in codes):
            return FailureDiagnosis(
                category=FailureCategory.ARGUMENT_SCHEMA_FAILURE,
                confidence=1.0,
                root_cause_step=idx,
                evidence=f"Step {idx} failed argument schema or semantic bound: {step.error}",
                details={"error_codes": codes, "step_index": idx},
            )

    # 4. Check for DependencyOrderFailure
    called_tool_names = [a.tool_name for a in called_actions]
    # Check 4a: Against task.expected_tools topological ordering
    if task and task.expected_tools:
        expected = task.expected_tools
        # For any pair of expected tools (tool_i, tool_j) with i < j
        for i in range(len(expected)):
            for j in range(i + 1, len(expected)):
                tool_prereq = expected[i]
                tool_dep = expected[j]
                if tool_dep in called_tool_names:
                    dep_idx = called_tool_names.index(tool_dep)
                    if tool_prereq not in called_tool_names:
                        return FailureDiagnosis(
                            category=FailureCategory.DEPENDENCY_ORDER_FAILURE,
                            confidence=0.95,
                            root_cause_step=dep_idx,
                            evidence=(
                                f"Dependent tool '{tool_dep}' was invoked at step {dep_idx} "
                                f"before prerequisite tool '{tool_prereq}' was ever invoked."
                            ),
                            details={"prerequisite": tool_prereq, "dependent": tool_dep},
                        )
                    prereq_idx = called_tool_names.index(tool_prereq)
                    if dep_idx < prereq_idx:
                        return FailureDiagnosis(
                            category=FailureCategory.DEPENDENCY_ORDER_FAILURE,
                            confidence=1.0,
                            root_cause_step=dep_idx,
                            evidence=(
                                f"Dependent tool '{tool_dep}' at step {dep_idx} was invoked "
                                f"before prerequisite tool '{tool_prereq}' at step {prereq_idx}."
                            ),
                            details={
                                "prerequisite": tool_prereq,
                                "dependent": tool_dep,
                                "dep_idx": dep_idx,
                                "prereq_idx": prereq_idx,
                            },
                        )

    # Check 4b: General domain-agnostic tool dependency invariants (even if task is None)
    if "compute_aggregate" in called_tool_names and "lookup_records" in called_tool_names:
        agg_idx = called_tool_names.index("compute_aggregate")
        lookup_idx = called_tool_names.index("lookup_records")
        if agg_idx < lookup_idx:
            return FailureDiagnosis(
                category=FailureCategory.DEPENDENCY_ORDER_FAILURE,
                confidence=1.0,
                root_cause_step=agg_idx,
                evidence=(
                    f"Tool 'compute_aggregate' at step {agg_idx} was invoked before prerequisite "
                    f"'lookup_records' at step {lookup_idx}."
                ),
                details={"dep_idx": agg_idx, "prereq_idx": lookup_idx},
            )
    if "verify_constraint" in called_tool_names and "lookup_records" in called_tool_names:
        ver_idx = called_tool_names.index("verify_constraint")
        lookup_idx = called_tool_names.index("lookup_records")
        if ver_idx < lookup_idx:
            return FailureDiagnosis(
                category=FailureCategory.DEPENDENCY_ORDER_FAILURE,
                confidence=1.0,
                root_cause_step=ver_idx,
                evidence=(
                    f"Tool 'verify_constraint' at step {ver_idx} was invoked before prerequisite "
                    f"'lookup_records' at step {lookup_idx}."
                ),
                details={"dep_idx": ver_idx, "prereq_idx": lookup_idx},
            )

    # 5. Check for LoopingAndStallFailure:
    # 5a. 3 or more identical tool calls
    seen_calls = [canonical_call_key(s.action) for s in trajectory.steps if s.action is not None]
    call_counts: Dict[str, int] = {}
    for call in seen_calls:
        call_counts[call] = call_counts.get(call, 0) + 1
        if call_counts[call] >= 3:
            first_idx = seen_calls.index(call)
            return FailureDiagnosis(
                category=FailureCategory.LOOPING_AND_STALL_FAILURE,
                confidence=1.0,
                root_cause_step=first_idx,
                evidence=f"Agent executed 3 or more identical tool actions: {call_counts[call]} occurrences.",
                details={"repeated_call": call, "count": call_counts[call]},
            )

    # 5b. Cyclic alternating loop detection (A -> B -> A -> B)
    if len(seen_calls) >= 4:
        for i in range(len(seen_calls) - 3):
            if seen_calls[i] == seen_calls[i + 2] and seen_calls[i + 1] == seen_calls[i + 3]:
                return FailureDiagnosis(
                    category=FailureCategory.LOOPING_AND_STALL_FAILURE,
                    confidence=0.95,
                    root_cause_step=i,
                    evidence=f"Agent entered alternating 2-cycle tool loop starting at step {i}.",
                    details={"cycle": [seen_calls[i], seen_calls[i + 1]]},
                )

    # 5c. Step budget exhaustion without preceding unrecovered tool errors
    if trajectory.total_steps >= max_steps:
        return FailureDiagnosis(
            category=FailureCategory.LOOPING_AND_STALL_FAILURE,
            confidence=0.9,
            root_cause_step=max_steps - 1,
            evidence=f"Trajectory exhausted step budget ({trajectory.total_steps}/{max_steps}) without completing task.",
            details={"total_steps": trajectory.total_steps, "max_steps": max_steps},
        )

    for idx, step in enumerate(trajectory.steps):
        codes = _extract_error_codes(step)
        if any(c in KNOWN_STALL_CODES for c in codes):
            return FailureDiagnosis(
                category=FailureCategory.LOOPING_AND_STALL_FAILURE,
                confidence=1.0,
                root_cause_step=idx,
                evidence=f"Step {idx} encountered budget or timeout stall: {step.error}",
                details={"error_codes": codes, "step_index": idx},
            )

    # 6. Check for EarlyTerminationFailure (incomplete expected tool subset)
    if task and task.expected_tools:
        missing_tools = [tool for tool in task.expected_tools if tool not in called_tool_names]
        if missing_tools:
            return FailureDiagnosis(
                category=FailureCategory.EARLY_TERMINATION_FAILURE,
                confidence=0.9,
                root_cause_step=len(trajectory.steps) - 1,
                evidence=(
                    f"Agent terminated prematurely without invoking expected tools: {missing_tools}."
                ),
                details={"missing_tools": missing_tools, "called_tools": called_tool_names},
            )

    # 7. Unclassified / Insufficient Evidence:
    # Trajectory executed all tools without tool/schema/order errors, but task predicate failed
    # (e.g. arithmetic mistake in final response).
    return FailureDiagnosis(
        category=None,
        confidence=0.0,
        root_cause_step=None,
        evidence=(
            "Trajectory completed required tool sequence without observable tool-use errors; "
            "failure attributed to final calculation or outcome predicate mismatch outside tool harness."
        ),
        details={"called_tools": called_tool_names, "total_steps": len(trajectory.steps)},
    )


def classify_failure(
    trajectory: TrajectoryRecord,
    task: Optional[TaskSpec] = None,
    max_steps: int = 8,
) -> Optional[FailureCategory]:
    """Diagnose and classify the root-cause failure mode of an unsuccessful trajectory.

    Preserves the standard backwards-compatible interface returning Optional[FailureCategory].
    Returns None if the trajectory succeeded or if evidence is insufficient to diagnose a tool failure.
    """
    diagnosis = diagnose_failure(trajectory=trajectory, task=task, max_steps=max_steps)
    return diagnosis.category

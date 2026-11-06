"""Adversarial and specification-driven unit tests for the environment reward engine.

Governing Specification: references/reward_spec.md
Traceability: Milestone 3 (M3.P1 & M3.P2), .agents/skills/reward-design-guard/SKILL.md
"""

from __future__ import annotations

import math
from typing import List

import pytest

from src.data.schemas import StepRecord, ToolCallAction, TrajectoryRecord
from src.environment.reward import (
    compute_episode_reward,
    compute_step_reward,
    DEFAULT_ALLOWED_TOOLS,
)
from src.environment.task_generator import TaskSpec


@pytest.fixture
def mock_task() -> TaskSpec:
    """Standard procurement auditing task fixture."""
    return TaskSpec(
        task_id="task_proc_test_001",
        split="train",
        template_id="inventory_replenishment",
        user_query="Audit inventory levels for vendor VENDOR_001 and compute reorder budget.",
        expected_tools=["lookup_records", "compute_aggregate"],
        oracle_answer=42.0,
        oracle_verdict=True,
    )


# --- 1. Correct Verified Completions ---


def test_reward_correct_verified_completion_optimal(mock_task: TaskSpec) -> None:
    """Optimal 2-step policy (P-OPT) achieves maximal reward +1.00."""
    steps = [
        StepRecord(
            step_index=0,
            thought="Looking up inventory records.",
            action=ToolCallAction(tool_name="lookup_records", arguments={"table": "items"}),
            observation={"items": [{"item_id": 1, "stock": 5}]},
            elapsed_ms=10.0,
        ),
        StepRecord(
            step_index=1,
            thought="Task solved.",
            elapsed_ms=5.0,
        ),
    ]
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output="Reorder quantity is 42.0 units; compliant with budget.",
        is_success=True,
        total_steps=2,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is True
    assert log.total_reward == 1.00
    assert log.component_breakdown.r_outcome == 1.00
    assert log.component_breakdown.r_efficiency == 0.00
    assert log.component_breakdown.r_process_sum == 0.00
    assert log.component_breakdown.r_safety == 0.00
    assert log.terminal_reason == "SUCCESS_VERIFIED"
    assert log.divergence_flags.high_reward_failure is False
    assert log.divergence_flags.low_reward_success is False


def test_reward_correct_verified_completion_standard(mock_task: TaskSpec) -> None:
    """Standard 4-step policy (P-VAL4) achieves +0.90 (-0.10 efficiency penalty)."""
    steps = [
        StepRecord(
            step_index=0,
            action=ToolCallAction(tool_name="lookup_records", arguments={"table": "items"}),
            observation={"count": 5},
            elapsed_ms=10.0,
        ),
        StepRecord(
            step_index=1,
            action=ToolCallAction(tool_name="compute_aggregate", arguments={"op": "sum"}),
            observation={"sum": 42.0},
            elapsed_ms=12.0,
        ),
        StepRecord(
            step_index=2,
            action=ToolCallAction(tool_name="verify_constraint", arguments={"limit": 50.0}),
            observation={"compliant": True},
            elapsed_ms=8.0,
        ),
        StepRecord(
            step_index=3,
            elapsed_ms=4.0,
        ),
    ]
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output="Reorder quantity is 42.0, compliant with cap.",
        is_success=True,
        total_steps=4,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is True
    assert log.total_reward == 0.90
    assert log.component_breakdown.r_outcome == 1.00
    assert log.component_breakdown.r_efficiency == -0.10  # -0.05 * (4 - 2)
    assert log.component_breakdown.r_process_sum == 0.00


def test_reward_correct_verified_completion_long(mock_task: TaskSpec) -> None:
    """Long 7-step successful policy (P-VAL7) achieves +0.75 (max efficiency penalty clamped at -0.25)."""
    steps: List[StepRecord] = []
    # 1 lookup, then 5 additional lookups, then final step
    steps.append(
        StepRecord(
            step_index=0,
            action=ToolCallAction(tool_name="lookup_records", arguments={"table": "items"}),
            observation={"data": 1},
        )
    )
    for i in range(1, 6):
        steps.append(
            StepRecord(
                step_index=i,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "items"}),
                observation={"data": i},
            )
        )
    steps.append(StepRecord(step_index=6))

    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output="Final audit calculated at 42.0, compliant.",
        is_success=True,
        total_steps=7,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is True
    assert log.component_breakdown.r_efficiency == -0.25  # clamped at max -0.25
    assert log.total_reward == 0.75


# --- 2. Adversarial Probes & Degenerate Policies ---


def test_reward_empty_noop_policy(mock_task: TaskSpec) -> None:
    """Empty no-op policy (P-NOOP, 0 steps) receives 0.00; proves no-op cannot win."""
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=[],
        final_output="",
        is_success=False,
        total_steps=0,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is False
    assert log.total_reward == 0.00
    assert log.component_breakdown.r_outcome == 0.00
    assert log.component_breakdown.r_efficiency == 0.00  # Gated on success!
    assert log.terminal_reason == "EMPTY_OUTPUT"


def test_reward_premature_conversational_quit(mock_task: TaskSpec) -> None:
    """Conversational apology without tool execution (P-QUIT) receives 0.00."""
    steps = [
        StepRecord(
            step_index=0,
            thought="I apologize, but I do not have access to inventory data.",
            elapsed_ms=5.0,
        )
    ]
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output="I am unable to answer this inquiry.",
        is_success=False,
        total_steps=1,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is False
    assert log.total_reward == 0.00
    assert log.component_breakdown.r_outcome == 0.00
    assert log.terminal_reason == "NUMERIC_MISMATCH"


def test_reward_copied_expected_answer_without_tools(mock_task: TaskSpec) -> None:
    """Copied expected answer (P-CHEAT): emits oracle number without executing required tools receives 0.00."""
    steps = [
        StepRecord(
            step_index=0,
            thought="Guessing the oracle answer directly without tools.",
            elapsed_ms=5.0,
        )
    ]
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output="The answer is 42.0 and compliant.",
        is_success=False,
        total_steps=1,
    )

    log = compute_episode_reward(traj, mock_task)

    # Must be caught by the anti-cheating guard: expected_tools was not executed
    assert log.is_success is False
    assert log.total_reward == 0.00
    assert log.terminal_reason == "SKIPPED_REQUIRED_TOOLS"


def test_reward_repeated_identical_tool_loop(mock_task: TaskSpec) -> None:
    """Identical call loop (P-LOOP) exhausts budget and receives safety penalty <= -0.50."""
    steps = [
        StepRecord(
            step_index=i,
            action=ToolCallAction(tool_name="lookup_records", arguments={"table": "items"}),
            observation={"data": 1},
        )
        for i in range(7)
    ]
    # Step 7 hits step limit guard
    steps.append(
        StepRecord(
            step_index=7,
            error="STEP_LIMIT_EXCEEDED: Episode reached maximum allowed steps (8)",
        )
    )

    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output=None,
        is_success=False,
        total_steps=8,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is False
    assert log.component_breakdown.r_safety == -0.50
    assert log.total_reward <= -0.50
    assert log.terminal_reason == "MAX_STEPS_REACHED"


def test_reward_alternating_call_loop(mock_task: TaskSpec) -> None:
    """Alternating loop (P-ALT) exhausts budget and receives safety penalty <= -0.50."""
    steps = []
    for i in range(7):
        tool = "lookup_records" if i % 2 == 0 else "compute_aggregate"
        steps.append(
            StepRecord(
                step_index=i,
                action=ToolCallAction(tool_name=tool, arguments={"op": "count"}),
                observation={"count": 1},
            )
        )
    steps.append(
        StepRecord(
            step_index=7,
            error="STEP_LIMIT_EXCEEDED: Max steps reached",
        )
    )

    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output=None,
        is_success=False,
        total_steps=8,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is False
    assert log.component_breakdown.r_safety == -0.50
    assert log.total_reward <= -0.50


def test_reward_malformed_action_syntax(mock_task: TaskSpec) -> None:
    """Malformed action tag or unparseable JSON receives r_parse penalty -0.10."""
    steps = [
        StepRecord(
            step_index=0,
            error="SCHEMA_VALIDATION_ERROR: Unterminated string in JSON action payload",
        )
    ]
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output=None,
        is_success=False,
        total_steps=1,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is False
    assert log.component_breakdown.r_parse == -0.10
    assert log.total_reward == -0.10


def test_reward_unauthorized_tool(mock_task: TaskSpec) -> None:
    """Attempting an unpermitted tool (e.g. execute_sql) receives r_tool penalty -0.10."""
    steps = [
        StepRecord(
            step_index=0,
            action=ToolCallAction(tool_name="execute_sql", arguments={"query": "DROP TABLE items;"}),
            error="PERMISSION_DENIED: Tool 'execute_sql' is not permitted.",
        )
    ]
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output=None,
        is_success=False,
        total_steps=1,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is False
    assert log.component_breakdown.r_tool == -0.10
    assert log.total_reward <= -0.10
    assert log.step_logs[0].guard_status == "BLOCKED"


def test_reward_invalid_arguments_schema(mock_task: TaskSpec) -> None:
    """Schema violation in tool arguments receives r_args penalty -0.10."""
    steps = [
        StepRecord(
            step_index=0,
            action=ToolCallAction(tool_name="lookup_records", arguments={"invalid_key": 123}),
            error="SCHEMA_VALIDATION_ERROR: Missing required field 'table'",
        )
    ]
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output=None,
        is_success=False,
        total_steps=1,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is False
    assert log.component_breakdown.r_args == -0.10
    assert log.total_reward <= -0.10


def test_reward_order_dependency_violation(mock_task: TaskSpec) -> None:
    """Calling compute_aggregate before any successful lookup receives r_order penalty -0.05."""
    steps = [
        StepRecord(
            step_index=0,
            action=ToolCallAction(tool_name="compute_aggregate", arguments={"op": "sum"}),
            observation={"sum": 0},
        )
    ]
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output=None,
        is_success=False,
        total_steps=1,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is False
    assert log.component_breakdown.r_order == -0.05
    assert log.total_reward == -0.05


def test_reward_timeout_and_guard_breach(mock_task: TaskSpec) -> None:
    """Wall-clock timeout breach triggers safety penalty -0.50."""
    steps = [
        StepRecord(
            step_index=0,
            error="WALL_CLOCK_TIMEOUT: Execution exceeded 30.0s deadline",
        )
    ]
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=steps,
        final_output=None,
        is_success=False,
        total_steps=1,
    )

    log = compute_episode_reward(traj, mock_task)

    assert log.is_success is False
    assert log.component_breakdown.r_safety == -0.50
    assert log.total_reward <= -0.50
    assert log.terminal_reason == "TIMEOUT"


def test_reward_non_finite_nan_trap(mock_task: TaskSpec) -> None:
    """Injected NaN/Inf clamps fail-closed to -1.00 and sets non_finite_detected flag."""
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "items"}),
            ),
            StepRecord(step_index=1),
            StepRecord(step_index=2),
        ],
        final_output="42.0 and compliant",
        is_success=True,
        total_steps=3,
    )

    # Pass an efficiency coeff of float('nan') to probe robustness
    log = compute_episode_reward(traj, mock_task, efficiency_coeff=float("nan"))

    assert log.divergence_flags.non_finite_detected is True
    assert log.total_reward == -1.00


def test_reward_strict_success_dominance(mock_task: TaskSpec) -> None:
    """Theorem 1 Proof: Programmatic assertion proving min(R_succ) > max(R_fail)."""
    # 1. Evaluate three successful variants:
    succ_scores = []

    # Optimal 2-step
    s_opt = [
        StepRecord(
            step_index=0,
            action=ToolCallAction(tool_name="lookup_records", arguments={"table": "items"}),
        ),
        StepRecord(step_index=1),
    ]
    t_opt = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=1,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=s_opt,
        final_output="42.0 and compliant",
        is_success=True,
    )
    succ_scores.append(compute_episode_reward(t_opt, mock_task).total_reward)

    # 4-step
    s_4 = [
        StepRecord(
            step_index=0,
            action=ToolCallAction(tool_name="lookup_records", arguments={"table": "items"}),
        ),
        StepRecord(
            step_index=1,
            action=ToolCallAction(tool_name="compute_aggregate", arguments={"op": "sum"}),
        ),
        StepRecord(step_index=2),
        StepRecord(step_index=3),
    ]
    t_4 = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=2,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=s_4,
        final_output="42.0 and compliant",
        is_success=True,
    )
    succ_scores.append(compute_episode_reward(t_4, mock_task).total_reward)

    # 7-step with 1 parse retry
    s_7 = [
        StepRecord(step_index=0, error="SCHEMA_VALIDATION_ERROR: parse"),
        StepRecord(
            step_index=1,
            action=ToolCallAction(tool_name="lookup_records", arguments={"table": "items"}),
        ),
    ]
    for i in range(2, 6):
        s_7.append(
            StepRecord(
                step_index=i,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "items"}),
            )
        )
    s_7.append(StepRecord(step_index=6))
    t_7 = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=3,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=s_7,
        final_output="42.0 and compliant",
        is_success=True,
    )
    succ_scores.append(compute_episode_reward(t_7, mock_task).total_reward)

    # 2. Evaluate failure variants:
    fail_scores = []

    # Empty no-op
    fail_scores.append(
        compute_episode_reward(
            TrajectoryRecord(
                task_id=mock_task.task_id,
                seed=4,
                system_prompt="sys",
                user_query=mock_task.user_query,
                steps=[],
                final_output="",
                is_success=False,
            ),
            mock_task,
        ).total_reward
    )

    # Premature quit
    fail_scores.append(
        compute_episode_reward(
            TrajectoryRecord(
                task_id=mock_task.task_id,
                seed=5,
                system_prompt="sys",
                user_query=mock_task.user_query,
                steps=[StepRecord(step_index=0)],
                final_output="I cannot solve this.",
                is_success=False,
            ),
            mock_task,
        ).total_reward
    )

    # Copied oracle without tools
    fail_scores.append(
        compute_episode_reward(
            TrajectoryRecord(
                task_id=mock_task.task_id,
                seed=6,
                system_prompt="sys",
                user_query=mock_task.user_query,
                steps=[StepRecord(step_index=0)],
                final_output="42.0 and compliant",
                is_success=False,
            ),
            mock_task,
        ).total_reward
    )

    # Looping failure
    fail_scores.append(
        compute_episode_reward(
            TrajectoryRecord(
                task_id=mock_task.task_id,
                seed=7,
                system_prompt="sys",
                user_query=mock_task.user_query,
                steps=[
                    StepRecord(
                        step_index=i,
                        action=ToolCallAction(tool_name="lookup_records", arguments={}),
                    )
                    for i in range(7)
                ]
                + [StepRecord(step_index=7, error="STEP_LIMIT_EXCEEDED")],
                final_output=None,
                is_success=False,
            ),
            mock_task,
        ).total_reward
    )

    min_succ = min(succ_scores)
    max_fail = max(fail_scores)

    # Mathematical Invariant:
    # min_succ must be strictly greater than max_fail
    assert min_succ >= 0.65, f"Expected min success >= 0.65, got {min_succ}"
    assert max_fail <= 0.00, f"Expected max failure <= 0.00, got {max_fail}"
    assert min_succ > max_fail, f"Success dominance violated: {min_succ} <= {max_fail}"


def test_reward_immutability(mock_task: TaskSpec) -> None:
    """Assert compute_episode_reward does not mutate input trajectory or task."""
    step = StepRecord(
        step_index=0,
        action=ToolCallAction(tool_name="lookup_records", arguments={"table": "items"}),
    )
    traj = TrajectoryRecord(
        task_id=mock_task.task_id,
        seed=42,
        system_prompt="sys",
        user_query=mock_task.user_query,
        steps=[step],
        final_output="42.0",
        is_success=True,
    )

    before_dump = traj.model_dump()
    compute_episode_reward(traj, mock_task)
    after_dump = traj.model_dump()

    assert before_dump == after_dump

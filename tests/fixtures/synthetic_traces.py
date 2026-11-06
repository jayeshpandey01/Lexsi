"""Synthetic trajectory and task fixtures for calibration and testing."""

from __future__ import annotations

from typing import Dict, List, Tuple
from src.data.schemas import StepRecord, ToolCallAction, TrajectoryRecord
from src.environment.task_generator import TaskSpec


def get_calibration_tasks() -> Dict[str, TaskSpec]:
    """Return dictionary of reference TaskSpec objects for calibration."""
    return {
        "task_t1": TaskSpec(
            task_id="task_val_T1_001",
            split="val",
            template_id="T1_inventory_value",
            user_query="Sum inventory value for Software category",
            expected_tools=["lookup_records", "compute_aggregate"],
            oracle_answer=2500.0,
        ),
        "task_t3": TaskSpec(
            task_id="task_val_T3_001",
            split="val",
            template_id="T3_vendor_rating_check",
            user_query="Check if VND-106 rating exceeds 4.5",
            expected_tools=["lookup_records", "verify_constraint"],
            oracle_answer=4.8,
            oracle_verdict=True,
        ),
        "task_t4": TaskSpec(
            task_id="task_test_T4_001",
            split="test",
            template_id="T4_budget_cap_verification",
            user_query="Audit budget cap for Services vendor VND-108",
            expected_tools=["lookup_records", "compute_aggregate", "verify_constraint"],
            oracle_answer=50000.0,
            oracle_verdict=True,
        ),
    }


def make_trace_tool_selection_unknown() -> TrajectoryRecord:
    """Positive: Hallucinated / unregistered tool name."""
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="unregistered_database_query", arguments={"query": "SELECT *"}),
                error="TOOL_NOT_FOUND: Tool 'unregistered_database_query' is not recognized or authorized.",
            )
        ],
        is_success=False,
    )


def make_trace_tool_selection_disallowed() -> TrajectoryRecord:
    """Positive: Tool prohibited by policy permission tier."""
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="admin_drop_tables", arguments={}),
                error="DISALLOWED_TOOL: Tool 'admin_drop_tables' is prohibited by current policy guard.",
            )
        ],
        is_success=False,
    )


def make_trace_tool_selection_extraneous() -> TrajectoryRecord:
    """Positive: Tool irrelevant to task specification."""
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="verify_constraint", arguments={"value": 1.0, "threshold": 2.0}),
                observation=False,
            )
        ],
        is_success=False,
    )


def make_trace_argument_schema_extra_arg() -> TrajectoryRecord:
    """Positive: Extra field with extra='forbid'."""
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "inventory_items", "unsupported_flag": True}),
                error="SCHEMA_VALIDATION_ERROR: Extra inputs are not permitted [extra_forbidden]",
            )
        ],
        is_success=False,
    )


def make_trace_argument_schema_semantic_bound() -> TrajectoryRecord:
    """Positive: Semantic domain boundary error."""
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [1.0], "operation": "invalid_op"}),
                error="SEMANTIC_BOUND_ERROR: Operation 'invalid_op' not supported.",
            )
        ],
        is_success=False,
    )


def make_trace_dependency_order() -> TrajectoryRecord:
    """Positive: Downstream compute_aggregate called before prerequisite lookup_records."""
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [10.0, 20.0], "operation": "sum"}),
                observation=30.0,
            ),
            StepRecord(
                step_index=1,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "inventory_items"}),
                observation=[{"item_id": "ITM-001"}],
            ),
        ],
        is_success=False,
    )


def make_trace_looping_identical_nested() -> TrajectoryRecord:
    """Positive: 3+ identical calls with nested dictionary arguments."""
    nested_args = {"table": "vendors", "filters": {"category": "Software", "status": "active"}}
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=i,
                action=ToolCallAction(tool_name="lookup_records", arguments=nested_args),
                observation=[{"vendor_id": "VND-106"}],
            )
            for i in range(3)
        ],
        is_success=False,
    )


def make_trace_looping_cycle() -> TrajectoryRecord:
    """Positive: Alternating 2-cycle tool loop (A -> B -> A -> B)."""
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors"}),
                observation=[{"vendor_id": "VND-106"}],
            ),
            StepRecord(
                step_index=1,
                action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [1.0], "operation": "sum"}),
                observation=1.0,
            ),
            StepRecord(
                step_index=2,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors"}),
                observation=[{"vendor_id": "VND-106"}],
            ),
            StepRecord(
                step_index=3,
                action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [1.0], "operation": "sum"}),
                observation=1.0,
            ),
        ],
        is_success=False,
    )


def make_trace_early_termination_zero_steps() -> TrajectoryRecord:
    """Positive: Immediate termination with 0 tool steps."""
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[],
        final_output="The total is 500.",
        is_success=False,
    )


def make_trace_early_termination_missing_subset() -> TrajectoryRecord:
    """Positive: Terminated after 1 tool when 2 required tools expected."""
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "inventory_items"}),
                observation=[{"unit_cost": 10.0, "stock_level": 5}],
            )
        ],
        final_output="Done looking up.",
        is_success=False,
    )


def make_trace_tool_recovery_repeated_error() -> TrajectoryRecord:
    """Positive: Repeated identical failing call after error observation."""
    failing_args = {"table": "inventory_items", "bad_param": 123}
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments=failing_args),
                error="SCHEMA_VALIDATION_ERROR: Extra inputs are not permitted",
            ),
            StepRecord(
                step_index=1,
                action=ToolCallAction(tool_name="lookup_records", arguments=failing_args),
                error="SCHEMA_VALIDATION_ERROR: Extra inputs are not permitted",
            ),
        ],
        is_success=False,
    )


def make_trace_tool_recovery_abort_after_error() -> TrajectoryRecord:
    """Positive: Aborted immediately following error observation."""
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "invalid_table"}),
                error="SCHEMA_VALIDATION_ERROR: Value error, Table 'invalid_table' not found",
            ),
            StepRecord(
                step_index=1,
                thought="I encountered an error so I am giving up.",
                elapsed_ms=10.0,
            ),
        ],
        final_output="Aborted due to database error.",
        is_success=False,
    )


def make_trace_near_miss_arithmetic_outcome() -> TrajectoryRecord:
    """Near-miss: Executed all tools successfully, but language head made arithmetic error."""
    return TrajectoryRecord(
        task_id="task_val_T1_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Sum inventory value",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "inventory_items"}),
                observation=[{"unit_cost": 500.0, "stock_level": 5}],
            ),
            StepRecord(
                step_index=1,
                action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [2500.0], "operation": "sum"}),
                observation=2500.0,
            ),
        ],
        final_output="The total inventory valuation is 2505.0.",  # Oracle is 2500.0
        is_success=False,
    )


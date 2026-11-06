"""Hand-constructed deterministic test fixtures with mathematically exact ground truth.

Used to rigorously validate evaluation metrics M1-M7, failure classification,
and paired bootstrap statistical inference without stochastic dependencies.
"""

from typing import Dict, List, Tuple
from src.data.schemas import StepRecord, ToolCallAction, TrajectoryRecord
from src.environment.task_generator import TaskSpec


def create_nominal_perfect_suite() -> Tuple[List[TaskSpec], List[TrajectoryRecord]]:
    """Fixture: 3 tasks, all executed with 100% success, optimal steps, zero errors.

    Expected Headline Metrics:
      - M1 (Task Success Rate): 1.0 (3/3)
      - M2 (Tool Selection Accuracy): 1.0 (6/6 correct calls)
      - M3 (Schema Validity): 1.0 (6/6 valid calls)
      - M4 (Call Order Validity): 1.0 (3/3 valid orders)
      - M5 (Step Efficiency): 2.0 steps
      - M6 (Redundant Call Rate): 0.0
      - M7 (Loop & Stall Frequency): 0.0
    """
    tasks = [
        TaskSpec(
            task_id=f"nominal_task_{i:02d}",
            split="val",
            template_id="T1_inventory_value",
            user_query=f"Query {i}",
            expected_tools=["lookup_records", "compute_aggregate"],
            oracle_answer=100.0 * (i + 1),
        )
        for i in range(3)
    ]

    trajectories = [
        TrajectoryRecord(
            task_id=f"nominal_task_{i:02d}",
            seed=42 + i,
            system_prompt="Prompt",
            user_query=f"Query {i}",
            steps=[
                StepRecord(
                    step_index=0,
                    action=ToolCallAction(tool_name="lookup_records", arguments={"table": "inventory_items"}),
                    observation=[{"item_id": f"ITEM-{i}", "price": 100.0}],
                ),
                StepRecord(
                    step_index=1,
                    action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [100.0], "operation": "sum"}),
                    observation=100.0,
                ),
            ],
            final_output=f"Total: {100.0 * (i + 1)}",
            is_success=True,
            total_steps=2,
        )
        for i in range(3)
    ]

    return tasks, trajectories


def create_zero_call_suite() -> Tuple[List[TaskSpec], List[TrajectoryRecord]]:
    """Fixture: 3 tasks, agent terminates immediately with 0 tool calls.

    Expected Headline Metrics:
      - M1: 0.0 (0/3 successes)
      - M2: 0.0 (0 calls)
      - M3: 1.0 (0 invalid calls attempted)
      - M4: 1.0 (0 multi-step episodes to violate)
      - M5: None (unmeasured, 0 successes)
      - M6: 0.0 (0 duplicates)
      - M7: 0.0 (not hitting budget)
    """
    tasks = [
        TaskSpec(
            task_id=f"zerocall_task_{i:02d}",
            split="val",
            template_id="T1_inventory_value",
            user_query=f"Query {i}",
            expected_tools=["lookup_records", "compute_aggregate"],
            oracle_answer=50.0,
        )
        for i in range(3)
    ]

    trajectories = [
        TrajectoryRecord(
            task_id=f"zerocall_task_{i:02d}",
            seed=42 + i,
            system_prompt="Prompt",
            user_query=f"Query {i}",
            steps=[],
            final_output="I decline to use tools.",
            is_success=False,
            total_steps=0,
        )
        for i in range(3)
    ]

    return tasks, trajectories


def create_schema_and_semantic_error_suite() -> Tuple[List[TaskSpec], List[TrajectoryRecord]]:
    """Fixture: Invocations testing distinct schema and semantic argument failure codes.

    Tests:
      - Call 1: SCHEMA_VALIDATION_ERROR (missing required parameter)
      - Call 2: SEMANTIC_BOUND_ERROR (value exceeds allowable budget cap)
      - Call 3: MALFORMED_TOOL_ARGUMENTS (unparseable payload)
      - Call 4: JSON_PARSE_ERROR (invalid JSON structure)
      - Call 5: Valid call

    Expected M3: 1 valid / 5 total calls = 0.2000.
    """
    task = TaskSpec(
        task_id="schema_test_01",
        split="val",
        template_id="T1_inventory_value",
        user_query="Audit inventory",
        expected_tools=["lookup_records", "compute_aggregate"],
        oracle_answer=100.0,
    )

    trajectory = TrajectoryRecord(
        task_id="schema_test_01",
        seed=42,
        system_prompt="Prompt",
        user_query="Audit inventory",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={}),
                error="SCHEMA_VALIDATION_ERROR: Missing required field 'table'",
            ),
            StepRecord(
                step_index=1,
                action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [-99999.0], "operation": "sum"}),
                error="SEMANTIC_BOUND_ERROR: Value out of range",
            ),
            StepRecord(
                step_index=2,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "bad_table"}),
                error="MALFORMED_TOOL_ARGUMENTS: Unknown table parameter",
            ),
            StepRecord(
                step_index=3,
                action=ToolCallAction(tool_name="lookup_records", arguments={"raw": "{"}),
                error="JSON_PARSE_ERROR: Unclosed JSON string",
            ),
            StepRecord(
                step_index=4,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors"}),
                observation=[{"vendor_id": "VND-100"}],
            ),
        ],
        is_success=False,
        total_steps=5,
    )

    return [task], [trajectory]


def create_order_inversion_suite() -> Tuple[List[TaskSpec], List[TrajectoryRecord]]:
    """Fixture: Inverted dependency tool execution.

    Episode calls `compute_aggregate` BEFORE `lookup_records`.
    Expected M4: 0.0 (order violated).
    """
    task = TaskSpec(
        task_id="order_test_01",
        split="val",
        template_id="T1_inventory_value",
        user_query="Sum records",
        expected_tools=["lookup_records", "compute_aggregate"],
        oracle_answer=250.0,
    )

    trajectory = TrajectoryRecord(
        task_id="order_test_01",
        seed=42,
        system_prompt="Prompt",
        user_query="Sum records",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [100.0], "operation": "sum"}),
                error="DEPENDENCY_ERROR: lookup_records must be executed first",
            ),
            StepRecord(
                step_index=1,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "inventory_items"}),
                observation=[{"item_id": "ITEM-1"}],
            ),
        ],
        is_success=False,
        total_steps=2,
    )

    return [task], [trajectory]


def create_redundant_and_cyclic_suite() -> Tuple[List[TaskSpec], List[TrajectoryRecord]]:
    """Fixture: Redundant duplicate calls with nested dict arguments and cyclic stalling.

    Episode makes:
      - Step 0: lookup_records with nested filter {"filter": {"category": "Hardware", "vendor": "VND-100"}}
      - Step 1: identical lookup_records call (Duplicate 1)
      - Step 2: compute_aggregate
      - Step 3: identical compute_aggregate call (Duplicate 2)
      - Step 4: lookup_records
      - Step 5: compute_aggregate (alternating 2-cycle)

    Expected M6 (Redundant call rate): 4 duplicates / 6 total calls = 0.6667.
    Expected M7 (Loop & stall frequency): 1.0 (alternating 2-cycle stall).
    """
    task = TaskSpec(
        task_id="redundant_test_01",
        split="val",
        template_id="T1_inventory_value",
        user_query="Redundant query",
        expected_tools=["lookup_records", "compute_aggregate"],
        oracle_answer=100.0,
    )

    args_nested = {"filter": {"category": "Hardware", "vendor": "VND-100"}}
    args_agg = {"operation": "sum", "values": [10.0, 20.0]}

    trajectory = TrajectoryRecord(
        task_id="redundant_test_01",
        seed=42,
        system_prompt="Prompt",
        user_query="Redundant query",
        steps=[
            StepRecord(step_index=0, action=ToolCallAction(tool_name="lookup_records", arguments=args_nested)),
            StepRecord(step_index=1, action=ToolCallAction(tool_name="compute_aggregate", arguments=args_agg)),
            StepRecord(step_index=2, action=ToolCallAction(tool_name="lookup_records", arguments=args_nested)),
            StepRecord(step_index=3, action=ToolCallAction(tool_name="compute_aggregate", arguments=args_agg)),
            StepRecord(step_index=4, action=ToolCallAction(tool_name="lookup_records", arguments=args_nested)),
            StepRecord(step_index=5, action=ToolCallAction(tool_name="compute_aggregate", arguments=args_agg)),
        ],
        is_success=False,
        total_steps=6,
    )

    return [task], [trajectory]


def create_mismatched_and_missing_tasks_suite() -> Tuple[List[TaskSpec], List[TrajectoryRecord]]:
    """Fixture: 5 tasks declared in manifest, but only 3 trajectories returned (2 dropped).

    Expected M1: 3 successes / 5 total tasks = 0.6000.
    Missing tasks must be accounted for in per-task results as unmeasured/dropped failures.
    """
    tasks = [
        TaskSpec(
            task_id=f"manifest_task_{i:02d}",
            split="val",
            template_id="T1_inventory_value",
            user_query=f"Query {i}",
            expected_tools=["lookup_records"],
            oracle_answer=100.0,
        )
        for i in range(5)
    ]

    # Only first 3 tasks evaluated (tasks 3 and 4 dropped)
    trajectories = [
        TrajectoryRecord(
            task_id=f"manifest_task_{i:02d}",
            seed=42 + i,
            system_prompt="Prompt",
            user_query=f"Query {i}",
            steps=[
                StepRecord(
                    step_index=0,
                    action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors"}),
                    observation=[{"vendor_id": "VND-100"}],
                )
            ],
            is_success=True,
            total_steps=1,
        )
        for i in range(3)
    ]

    return tasks, trajectories


def create_paired_comparison_suite() -> Tuple[List[TaskSpec], List[TrajectoryRecord], List[TrajectoryRecord]]:
    """Fixture: Paired comparison between Baseline (30% success) and Candidate (80% success).

    10 common tasks:
      - Baseline succeeds on tasks 0, 1, 2 (30% success).
      - Candidate succeeds on tasks 0, 1, 2, 3, 4, 5, 6, 7 (80% success).
      - Observed delta: +0.5000.
    """
    tasks = [
        TaskSpec(
            task_id=f"paired_task_{i:02d}",
            split="val",
            template_id="T1_inventory_value",
            user_query=f"Query {i}",
            expected_tools=["lookup_records"],
            oracle_answer=100.0,
        )
        for i in range(10)
    ]

    baseline_trajectories = [
        TrajectoryRecord(
            task_id=f"paired_task_{i:02d}",
            seed=42 + i,
            system_prompt="Prompt",
            user_query=f"Query {i}",
            steps=[],
            is_success=(i < 3),
            total_steps=1 if i < 3 else 0,
        )
        for i in range(10)
    ]

    candidate_trajectories = [
        TrajectoryRecord(
            task_id=f"paired_task_{i:02d}",
            seed=42 + i,
            system_prompt="Prompt",
            user_query=f"Query {i}",
            steps=[],
            is_success=(i < 8),
            total_steps=1 if i < 8 else 0,
        )
        for i in range(10)
    ]

    return tasks, baseline_trajectories, candidate_trajectories

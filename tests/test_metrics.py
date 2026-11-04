"""Unit tests for evaluation metrics suite and failure taxonomy classification."""

from src.data.schemas import StepRecord, ToolCallAction, TrajectoryRecord
from src.environment.task_generator import TaskSpec
from src.evaluation.metrics import evaluate_benchmark_run, task_success_rate
from src.evaluation.taxonomy import FailureCategory, classify_failure


def test_failure_taxonomy_classification() -> None:
    """Verify diagnostic classification across distinct failure modes."""
    task = TaskSpec(
        task_id="task_audit_01",
        split="val",
        template_id="T1_inventory_value",
        user_query="Sum items",
        expected_tools=["lookup_records", "compute_aggregate"],
        oracle_answer=100.0,
    )

    # 1. Unknown tool
    t_unknown = TrajectoryRecord(
        task_id="task_audit_01",
        seed=42,
        system_prompt="",
        user_query="",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="unauthorized_eval", arguments={}),
                error="TOOL_NOT_FOUND: Unauthorized",
            )
        ],
        is_success=False,
    )
    assert classify_failure(t_unknown, task) == FailureCategory.TOOL_SELECTION_FAILURE

    # 2. Schema error
    t_schema = TrajectoryRecord(
        task_id="task_audit_01",
        seed=42,
        system_prompt="",
        user_query="",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={}),
                error="SCHEMA_VALIDATION_ERROR: Missing required fields",
            )
        ],
        is_success=False,
    )
    assert classify_failure(t_schema, task) == FailureCategory.ARGUMENT_SCHEMA_FAILURE

    # 3. Dependency order failure
    t_order = TrajectoryRecord(
        task_id="task_audit_01",
        seed=42,
        system_prompt="",
        user_query="",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [1.0], "operation": "sum"}),
            ),
            StepRecord(
                step_index=1,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors"}),
            ),
        ],
        is_success=False,
    )
    assert classify_failure(t_order, task) == FailureCategory.DEPENDENCY_ORDER_FAILURE

    # 4. Looping / Stalling failure
    t_loop = TrajectoryRecord(
        task_id="task_audit_01",
        seed=42,
        system_prompt="",
        user_query="",
        steps=[
            StepRecord(
                step_index=i,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors"}),
            )
            for i in range(4)
        ],
        is_success=False,
    )
    assert classify_failure(t_loop, task) == FailureCategory.LOOPING_AND_STALL_FAILURE

    # 5. Early termination failure
    t_early = TrajectoryRecord(
        task_id="task_audit_01",
        seed=42,
        system_prompt="",
        user_query="",
        steps=[],
        final_output="I am giving up.",
        is_success=False,
    )
    assert classify_failure(t_early, task) == FailureCategory.EARLY_TERMINATION_FAILURE


def test_evaluate_benchmark_run() -> None:
    """Verify end-to-end evaluation runner producing AuditKitRunResult."""
    task = TaskSpec(
        task_id="task_audit_01",
        split="val",
        template_id="T1_inventory_value",
        user_query="Sum items",
        expected_tools=["lookup_records", "compute_aggregate"],
        oracle_answer=100.0,
    )

    t_success = TrajectoryRecord(
        task_id="task_audit_01",
        seed=42,
        system_prompt="",
        user_query="",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors"}),
                observation=[{"vendor_id": "VND-100"}],
            ),
            StepRecord(
                step_index=1,
                action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [100.0], "operation": "sum"}),
                observation=100.0,
            ),
        ],
        is_success=True,
        total_steps=2,
    )

    t_fail = TrajectoryRecord(
        task_id="task_audit_01",
        seed=42,
        system_prompt="",
        user_query="",
        steps=[],
        is_success=False,
    )

    run_res = evaluate_benchmark_run(
        trajectories=[t_success, t_fail],
        tasks=[task],
        model_id="CohereLabs/tiny-aya-global",
        run_id="test_run_01",
    )

    assert run_res.run_id == "test_run_01"
    assert run_res.metrics["M1_task_success_rate"].value == 0.5
    assert run_res.metrics["M5_trajectory_step_efficiency"].value == 2.0
    assert run_res.failure_breakdown[FailureCategory.EARLY_TERMINATION_FAILURE.value] == 1
    assert len(run_res.sha256_fingerprint) == 64

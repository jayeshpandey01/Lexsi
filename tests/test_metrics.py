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


def test_nominal_perfect_suite_metrics() -> None:
    """Verify ground truth calculations on 100% successful nominal suite."""
    from tests.fixtures.metric_fixtures import create_nominal_perfect_suite
    from src.evaluation.metrics import (
        call_order_validity_rate,
        loop_and_stall_frequency,
        mean_successful_steps,
        redundant_call_rate,
        schema_validity_rate,
        task_success_rate,
        tool_selection_accuracy,
    )

    tasks, trajectories = create_nominal_perfect_suite()
    assert task_success_rate(trajectories, tasks) == 1.0
    assert tool_selection_accuracy(trajectories, tasks) == 1.0
    assert schema_validity_rate(trajectories) == 1.0
    assert call_order_validity_rate(trajectories, tasks) == 1.0
    assert mean_successful_steps(trajectories) == 2.0
    assert redundant_call_rate(trajectories) == 0.0
    assert loop_and_stall_frequency(trajectories) == 0.0

    run_res = evaluate_benchmark_run(trajectories, tasks, run_id="run_nominal")
    assert run_res.headline["M1_task_success_rate"] == 1.0
    assert run_res.headline["M5_trajectory_step_efficiency"] == 2.0
    assert len(run_res.per_task_results) == 3
    assert all(p.is_success for p in run_res.per_task_results)


def test_zero_call_suite_metrics() -> None:
    """Verify metrics and unmeasured efficiency handling when agent makes zero tool calls."""
    from tests.fixtures.metric_fixtures import create_zero_call_suite
    from src.evaluation.metrics import (
        call_order_validity_rate,
        loop_and_stall_frequency,
        mean_successful_steps,
        redundant_call_rate,
        schema_validity_rate,
        task_success_rate,
        tool_selection_accuracy,
    )

    tasks, trajectories = create_zero_call_suite()
    assert task_success_rate(trajectories, tasks) == 0.0
    assert tool_selection_accuracy(trajectories, tasks) == 0.0
    assert schema_validity_rate(trajectories) == 1.0  # 0 invalid calls attempted
    assert call_order_validity_rate(trajectories, tasks) == 1.0
    assert mean_successful_steps(trajectories) is None  # Unmeasured
    assert redundant_call_rate(trajectories) == 0.0
    assert loop_and_stall_frequency(trajectories) == 0.0

    run_res = evaluate_benchmark_run(trajectories, tasks, run_id="run_zero")
    assert run_res.metrics["M5_trajectory_step_efficiency"].value is None
    assert run_res.metrics["M5_trajectory_step_efficiency"].unmeasured_count == 3


def test_schema_and_semantic_error_metrics() -> None:
    """Verify that M3 catches all schema, semantic, bound, and JSON parse errors."""
    from tests.fixtures.metric_fixtures import create_schema_and_semantic_error_suite
    from src.evaluation.metrics import schema_validity_rate

    tasks, trajectories = create_schema_and_semantic_error_suite()
    # 4 invalid calls out of 5 total calls -> valid rate is 1/5 = 0.2000
    assert schema_validity_rate(trajectories) == 0.2


def test_order_inversion_metrics() -> None:
    """Verify that M4 detects inverted dependency sequences."""
    from tests.fixtures.metric_fixtures import create_order_inversion_suite
    from src.evaluation.metrics import call_order_validity_rate

    tasks, trajectories = create_order_inversion_suite()
    assert call_order_validity_rate(trajectories, tasks) == 0.0


def test_redundant_and_cyclic_metrics() -> None:
    """Verify M6 redundant call detection with nested dictionaries and M7 2-cycle stall detection."""
    from tests.fixtures.metric_fixtures import create_redundant_and_cyclic_suite
    from src.evaluation.metrics import loop_and_stall_frequency, redundant_call_rate

    tasks, trajectories = create_redundant_and_cyclic_suite()
    # 4 duplicate calls out of 6 total calls (2 lookups + 2 aggregates duplicate earlier calls) = 0.6667
    assert redundant_call_rate(trajectories) == 0.6667
    # Alternating 2-cycle loop detected on uncompleted trajectory
    assert loop_and_stall_frequency(trajectories) == 1.0


def test_mismatched_and_missing_tasks_metrics() -> None:
    """Verify denominator accounting when tasks in manifest are missing from trajectory list."""
    from tests.fixtures.metric_fixtures import create_mismatched_and_missing_tasks_suite
    from src.evaluation.metrics import task_success_rate

    tasks, trajectories = create_mismatched_and_missing_tasks_suite()
    # 3 successes out of 5 declared tasks -> 0.6000
    assert task_success_rate(trajectories, tasks) == 0.6

    run_res = evaluate_benchmark_run(trajectories, tasks, run_id="run_missing")
    assert run_res.headline["M1_task_success_rate"] == 0.6
    # 5 per-task records: 3 evaluated, 2 unmeasured/dropped
    assert len(run_res.per_task_results) == 5
    dropped = [p for p in run_res.per_task_results if p.error_type == "UNMEASURED_OR_DROPPED_TASK"]
    assert len(dropped) == 2


def test_paired_bootstrap_comparison_significant() -> None:
    """Verify paired bootstrap hypothesis test on known positive delta (30% vs 80%)."""
    from tests.fixtures.metric_fixtures import create_paired_comparison_suite
    from src.evaluation.metrics import compute_paired_bootstrap_comparison

    tasks, base, cand = create_paired_comparison_suite()
    paired_res = compute_paired_bootstrap_comparison(base, cand, tasks=tasks, seed=42)

    assert paired_res.baseline_mean == 0.3
    assert paired_res.candidate_mean == 0.8
    assert paired_res.delta == 0.5
    assert paired_res.ci_lower > 0.1  # Delta CI strictly positive
    assert paired_res.p_value < 0.05
    assert paired_res.is_significant is True
    assert paired_res.sample_count == 10
    assert len(paired_res.caveats) > 0  # Small-sample warning (N=10 < 30)


def test_paired_bootstrap_comparison_null() -> None:
    """Verify paired bootstrap hypothesis test when candidate is identical to baseline."""
    from tests.fixtures.metric_fixtures import create_paired_comparison_suite
    from src.evaluation.metrics import compute_paired_bootstrap_comparison

    tasks, base, _ = create_paired_comparison_suite()
    paired_res = compute_paired_bootstrap_comparison(base, base, tasks=tasks, seed=42)

    assert paired_res.delta == 0.0
    assert paired_res.is_significant is False
    assert paired_res.p_value == 1.0



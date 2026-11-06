"""Comprehensive unit tests for failure taxonomy classification and diagnostic engine."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.data.schemas import StepRecord, ToolCallAction, TrajectoryRecord
from src.evaluation.taxonomy import (
    FailureCategory,
    FailureDiagnosis,
    classify_failure,
    diagnose_failure,
)
from tests.fixtures.synthetic_traces import (
    get_calibration_tasks,
    make_trace_argument_schema_extra_arg,
    make_trace_argument_schema_semantic_bound,
    make_trace_dependency_order,
    make_trace_early_termination_missing_subset,
    make_trace_early_termination_zero_steps,
    make_trace_looping_cycle,
    make_trace_looping_identical_nested,
    make_trace_near_miss_arithmetic_outcome,
    make_trace_tool_recovery_abort_after_error,
    make_trace_tool_recovery_repeated_error,
    make_trace_tool_selection_disallowed,
    make_trace_tool_selection_extraneous,
    make_trace_tool_selection_unknown,
)


class TestTaxonomyClassification:
    """Tests covering nominal and edge case classification for all 6 categories."""

    def setup_method(self) -> None:
        self.tasks = get_calibration_tasks()

    def test_tool_selection_unknown(self) -> None:
        traj = make_trace_tool_selection_unknown()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.TOOL_SELECTION_FAILURE
        assert diag.confidence >= 0.95
        assert diag.root_cause_step == 0
        assert "unregistered_database_query" in diag.evidence
        assert classify_failure(traj, self.tasks["task_t1"]) == FailureCategory.TOOL_SELECTION_FAILURE

    def test_tool_selection_disallowed(self) -> None:
        traj = make_trace_tool_selection_disallowed()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.TOOL_SELECTION_FAILURE
        assert "DISALLOWED_TOOL" in diag.evidence

    def test_tool_selection_extraneous(self) -> None:
        traj = make_trace_tool_selection_extraneous()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.TOOL_SELECTION_FAILURE
        assert "verify_constraint" in diag.evidence

    def test_argument_schema_extra_arg(self) -> None:
        traj = make_trace_argument_schema_extra_arg()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.ARGUMENT_SCHEMA_FAILURE
        assert "SCHEMA_VALIDATION_ERROR" in diag.evidence
        assert diag.root_cause_step == 0

    def test_argument_schema_semantic_bound(self) -> None:
        traj = make_trace_argument_schema_semantic_bound()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.ARGUMENT_SCHEMA_FAILURE
        assert "SEMANTIC_BOUND_ERROR" in diag.evidence

    def test_dependency_order(self) -> None:
        traj = make_trace_dependency_order()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.DEPENDENCY_ORDER_FAILURE
        assert diag.root_cause_step == 0
        assert "compute_aggregate" in diag.evidence
        assert "lookup_records" in diag.evidence

    def test_looping_identical_nested_arguments(self) -> None:
        traj = make_trace_looping_identical_nested()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.LOOPING_AND_STALL_FAILURE
        assert diag.confidence == 1.0
        assert "3 or more identical tool actions" in diag.evidence

    def test_looping_cycle(self) -> None:
        traj = make_trace_looping_cycle()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.LOOPING_AND_STALL_FAILURE
        assert "alternating 2-cycle" in diag.evidence

    def test_early_termination_zero_steps(self) -> None:
        traj = make_trace_early_termination_zero_steps()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.EARLY_TERMINATION_FAILURE
        assert "0 tool calls" in diag.evidence

    def test_early_termination_missing_subset(self) -> None:
        traj = make_trace_early_termination_missing_subset()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.EARLY_TERMINATION_FAILURE
        assert "compute_aggregate" in diag.evidence

    def test_tool_recovery_repeated_error(self) -> None:
        traj = make_trace_tool_recovery_repeated_error()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.TOOL_RECOVERY_FAILURE
        assert diag.root_cause_step == 1
        assert "repeated identical failing call" in diag.evidence

    def test_tool_recovery_abort_after_error(self) -> None:
        traj = make_trace_tool_recovery_abort_after_error()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category == FailureCategory.TOOL_RECOVERY_FAILURE
        assert diag.root_cause_step == 1
        assert "aborted" in diag.evidence

    def test_near_miss_arithmetic_outcome_insufficient_tool_error(self) -> None:
        traj = make_trace_near_miss_arithmetic_outcome()
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        # All tools were executed correctly; failure is in arithmetic head -> category should be None
        assert diag.category is None
        assert diag.confidence == 0.0
        assert "calculation or outcome predicate mismatch" in diag.evidence
        assert classify_failure(traj, self.tasks["task_t1"]) is None

    def test_successful_trajectory(self) -> None:
        traj = TrajectoryRecord(
            task_id="task_val_T1_001",
            seed=42,
            system_prompt="",
            user_query="",
            steps=[
                StepRecord(
                    step_index=0,
                    action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors"}),
                    observation=[{"vendor_id": "VND-100"}],
                )
            ],
            is_success=True,
        )
        diag = diagnose_failure(traj, self.tasks["task_t1"])
        assert diag.category is None
        assert diag.confidence == 1.0
        assert classify_failure(traj, self.tasks["task_t1"]) is None


class TestPrecedenceHierarchy:
    """Verify causal precedence ordering for trajectories with overlapping signals."""

    def test_recovery_over_schema(self) -> None:
        # Step 0 had schema error, Step 1 repeated identical schema error
        traj = make_trace_tool_recovery_repeated_error()
        assert classify_failure(traj) == FailureCategory.TOOL_RECOVERY_FAILURE

    def test_order_over_step_limit(self) -> None:
        # Step 0: aggregate before lookup. Followed by reaching max_steps.
        steps = [
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [1.0], "operation": "sum"}),
                observation=1.0,
            ),
            StepRecord(
                step_index=1,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "inventory_items"}),
                observation=[{"item_id": "ITM-001"}],
            ),
        ]
        # Pad to max_steps with non-repeating different arguments
        for i in range(2, 8):
            steps.append(
                StepRecord(
                    step_index=i,
                    action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors", "filters": {"vendor_id": f"VND-{i}"}}),
                    observation=[],
                )
            )
        traj = TrajectoryRecord(
            task_id="task_val_T1_001",
            seed=42,
            system_prompt="",
            user_query="",
            steps=steps,
            total_steps=8,
            is_success=False,
        )
        task = get_calibration_tasks()["task_t1"]
        # DependencyOrderFailure has higher precedence than step limit stall
        assert classify_failure(traj, task, max_steps=8) == FailureCategory.DEPENDENCY_ORDER_FAILURE

    def test_selection_over_step_limit(self) -> None:
        # Step 0: unknown tool error. Followed by step limit.
        steps = [
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="unknown_tool", arguments={}),
                error="TOOL_NOT_FOUND: Unknown",
            )
        ]
        for i in range(1, 8):
            steps.append(
                StepRecord(
                    step_index=i,
                    action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors", "filters": {"id": i}}),
                    observation=[],
                )
            )
        traj = TrajectoryRecord(
            task_id="task_val_T1_001",
            seed=42,
            system_prompt="",
            user_query="",
            steps=steps,
            total_steps=8,
            is_success=False,
        )
        task = get_calibration_tasks()["task_t1"]
        assert classify_failure(traj, task, max_steps=8) == FailureCategory.TOOL_SELECTION_FAILURE


class TestAnnotationSampleVerification:
    """Verify that the synthetic calibration sample file matches taxonomy predictions."""

    def test_sample_file_validity(self) -> None:
        sample_path = Path(__file__).resolve().parent.parent / "references" / "synthetic_annotation_sample.json"
        assert sample_path.exists(), "references/synthetic_annotation_sample.json missing"
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["taxonomy_version"] == "1.0.0"
        samples = data["samples"]
        assert len(samples) == 12

        # Verify all 6 categories are represented in adjudicated labels
        categories_represented = {s["adjudicated_label"] for s in samples}
        for cat in FailureCategory:
            assert cat.value in categories_represented, f"Missing category {cat.value} in sample"


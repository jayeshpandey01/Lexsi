"""Automated unit tests for dynamic failure trajectory generation and live adversarial policy execution."""

from __future__ import annotations

import pytest
from src.data.schemas import StepRecord, ToolCallAction, TrajectoryRecord
from src.environment.task_generator import TaskSpec
from src.evaluation.failure_generator import (
    AdversarialPolicyFactory,
    DynamicFailureSuite,
    TrajectoryFuzzer,
)
from src.evaluation.taxonomy import FailureCategory, diagnose_failure
from src.runtime.orchestrator import EpisodeOrchestrator
from tests.fixtures.synthetic_traces import (
    get_calibration_tasks,
    make_trace_near_miss_arithmetic_outcome,
)


@pytest.fixture
def base_task() -> TaskSpec:
    tasks = get_calibration_tasks()
    return tasks["task_t1"]


@pytest.fixture
def base_trajectory() -> TrajectoryRecord:
    return make_trace_near_miss_arithmetic_outcome()


def test_fuzzer_coverage_across_all_categories(
    base_task: TaskSpec, base_trajectory: TrajectoryRecord
) -> None:
    """Verify that TrajectoryFuzzer programmatically produces all 6 categories with 100% fidelity."""
    fuzzer = TrajectoryFuzzer(seed=42)

    # 1. Tool Selection
    t_sel_unknown = fuzzer.mutate_tool_selection(base_trajectory, mode="unknown_tool")
    assert diagnose_failure(t_sel_unknown, base_task).category == FailureCategory.TOOL_SELECTION_FAILURE

    t_sel_disallowed = fuzzer.mutate_tool_selection(base_trajectory, mode="disallowed_tool")
    assert diagnose_failure(t_sel_disallowed, base_task).category == FailureCategory.TOOL_SELECTION_FAILURE

    # 2. Argument Schema
    t_sch_extra = fuzzer.mutate_argument_schema(base_trajectory, mode="extra_forbidden")
    assert diagnose_failure(t_sch_extra, base_task).category == FailureCategory.ARGUMENT_SCHEMA_FAILURE

    t_sch_bound = fuzzer.mutate_argument_schema(base_trajectory, mode="semantic_bound")
    assert diagnose_failure(t_sch_bound, base_task).category == FailureCategory.ARGUMENT_SCHEMA_FAILURE

    # 3. Dependency Order
    t_ord = fuzzer.mutate_dependency_order(base_trajectory)
    assert diagnose_failure(t_ord, base_task).category == FailureCategory.DEPENDENCY_ORDER_FAILURE

    # 4. Looping & Stall
    t_loop_3x = fuzzer.mutate_loop(base_trajectory, mode="identical_3x")
    assert diagnose_failure(t_loop_3x, base_task).category == FailureCategory.LOOPING_AND_STALL_FAILURE

    t_loop_cycle = fuzzer.mutate_loop(base_trajectory, mode="alternating_cycle")
    assert diagnose_failure(t_loop_cycle, base_task).category == FailureCategory.LOOPING_AND_STALL_FAILURE

    # 5. Early Termination
    t_early = fuzzer.mutate_early_termination(base_trajectory)
    assert diagnose_failure(t_early, base_task).category == FailureCategory.EARLY_TERMINATION_FAILURE

    # 6. Tool Recovery
    t_rec_repeat = fuzzer.mutate_recovery_failure(base_trajectory, mode="repeat_error")
    assert diagnose_failure(t_rec_repeat, base_task).category == FailureCategory.TOOL_RECOVERY_FAILURE

    t_rec_abort = fuzzer.mutate_recovery_failure(base_trajectory, mode="abort_after_error")
    assert diagnose_failure(t_rec_abort, base_task).category == FailureCategory.TOOL_RECOVERY_FAILURE


def test_adversarial_policies_in_live_orchestrator(base_task: TaskSpec) -> None:
    """Verify that executable adversarial policies run through EpisodeOrchestrator and trigger expected categories."""
    orchestrator = EpisodeOrchestrator(seed=42, max_steps=5)

    # 1. NoOp Policy -> EarlyTerminationFailure
    noop_policy = AdversarialPolicyFactory.create_noop_policy()
    traj_noop = orchestrator.run_episode(base_task, noop_policy)
    assert traj_noop.is_success is False
    diag_noop = diagnose_failure(traj_noop, base_task)
    assert diag_noop.category == FailureCategory.EARLY_TERMINATION_FAILURE

    # 2. Stubborn Error Policy -> ToolRecoveryFailure
    stubborn_policy = AdversarialPolicyFactory.create_stubborn_error_policy()
    traj_stubborn = orchestrator.run_episode(base_task, stubborn_policy)
    assert traj_stubborn.is_success is False
    diag_stubborn = diagnose_failure(traj_stubborn, base_task)
    assert diag_stubborn.category == FailureCategory.TOOL_RECOVERY_FAILURE

    # 3. Spam Loop Policy -> LoopingAndStallFailure
    spam_policy = AdversarialPolicyFactory.create_spam_loop_policy(tool_name="lookup_records")
    traj_spam = orchestrator.run_episode(base_task, spam_policy)
    assert traj_spam.is_success is False
    diag_spam = diagnose_failure(traj_spam, base_task)
    assert diag_spam.category == FailureCategory.LOOPING_AND_STALL_FAILURE

    # 4. Hallucination Policy -> ToolSelectionFailure
    hallucinate_policy = AdversarialPolicyFactory.create_hallucination_policy()
    traj_hallucinate = orchestrator.run_episode(base_task, hallucinate_policy)
    assert traj_hallucinate.is_success is False
    diag_hallucinate = diagnose_failure(traj_hallucinate, base_task)
    assert diag_hallucinate.category == FailureCategory.TOOL_SELECTION_FAILURE


def test_dynamic_failure_suite(base_task: TaskSpec, base_trajectory: TrajectoryRecord) -> None:
    """Verify that DynamicFailureSuite generates a complete balanced test suite on demand."""
    suite_gen = DynamicFailureSuite(seed=42)
    suite = suite_gen.generate_suite(base_trajectory, base_task)

    assert len(suite) == 10
    categories_generated = {expected_cat for _, _, expected_cat in suite}
    assert len(categories_generated) == 6  # All 6 categories represented

    for desc, traj, expected_cat in suite:
        diag = diagnose_failure(traj, base_task)
        assert diag.category == expected_cat, f"Failed on {desc}: expected {expected_cat}, got {diag.category}"


"""Evaluation metrics suite (M1-M7) and AuditKIT RunResult adapter."""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Tuple
import numpy as np

from src.data.schemas import TrajectoryRecord
from src.environment.task_generator import TaskSpec
from src.evaluation.taxonomy import FailureCategory, classify_failure
from src.utils.compatibility import (
    AuditKitMetricResult,
    AuditKitRunResult,
    compute_run_fingerprint,
)


def compute_bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute deterministic paired bootstrap confidence intervals (AuditKIT PR #9 defense)."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []

    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    lower_idx = int((1.0 - ci_level) / 2.0 * n_resamples)
    upper_idx = int((1.0 + ci_level) / 2.0 * n_resamples)

    return round(means[lower_idx], 4), round(means[min(upper_idx, n_resamples - 1)], 4)


def task_success_rate(trajectories: List[TrajectoryRecord]) -> float:
    """M1: Fraction of tasks satisfying deterministic success predicate."""
    if not trajectories:
        return 0.0
    successes = sum(1 for t in trajectories if t.is_success)
    return round(successes / len(trajectories), 4)


def tool_selection_accuracy(
    trajectories: List[TrajectoryRecord],
    tasks: List[TaskSpec],
) -> float:
    """M2: Ratio of correctly chosen tools over total tool decisions."""
    total_calls = 0
    correct_calls = 0

    task_map = {t.task_id: t for t in tasks}
    for traj in trajectories:
        task = task_map.get(traj.task_id)
        expected = set(task.expected_tools) if task else set()

        for step in traj.steps:
            if step.action:
                total_calls += 1
                if step.action.tool_name in expected:
                    correct_calls += 1

    if total_calls == 0:
        return 0.0
    return round(correct_calls / total_calls, 4)


def schema_validity_rate(trajectories: List[TrajectoryRecord]) -> float:
    """M3: Ratio of schema-valid tool invocations over total tool calls."""
    total_calls = 0
    schema_errors = 0

    for traj in trajectories:
        for step in traj.steps:
            if step.action:
                total_calls += 1
                if step.error and "SCHEMA_VALIDATION_ERROR" in step.error:
                    schema_errors += 1

    if total_calls == 0:
        return 1.0  # Zero invalid calls
    return round((total_calls - schema_errors) / total_calls, 4)


def call_order_validity_rate(trajectories: List[TrajectoryRecord]) -> float:
    """M4: Ratio of multi-step episodes with correct dependency execution order."""
    multi_step_count = 0
    valid_order_count = 0

    for traj in trajectories:
        called_tools = [s.action.tool_name for s in traj.steps if s.action]
        if "compute_aggregate" in called_tools and "lookup_records" in called_tools:
            multi_step_count += 1
            if called_tools.index("lookup_records") < called_tools.index("compute_aggregate"):
                valid_order_count += 1

    if multi_step_count == 0:
        return 1.0
    return round(valid_order_count / multi_step_count, 4)


def mean_successful_steps(trajectories: List[TrajectoryRecord]) -> float:
    """M5: Mean execution steps conditional on verified task success: E[T | Success=1]."""
    successful_steps = [t.total_steps for t in trajectories if t.is_success]
    if not successful_steps:
        return 0.0
    return round(sum(successful_steps) / len(successful_steps), 4)


def redundant_call_rate(trajectories: List[TrajectoryRecord]) -> float:
    """M6: Frequency of duplicate or repeated identical tool calls."""
    total_calls = 0
    duplicate_calls = 0

    for traj in trajectories:
        seen = set()
        for step in traj.steps:
            if step.action:
                total_calls += 1
                key = (step.action.tool_name, str(sorted(step.action.arguments.items())))
                if key in seen:
                    duplicate_calls += 1
                else:
                    seen.add(key)

    if total_calls == 0:
        return 0.0
    return round(duplicate_calls / total_calls, 4)


def loop_and_stall_frequency(
    trajectories: List[TrajectoryRecord],
    max_steps: int = 8,
) -> float:
    """M7: Fraction of runs hitting maximum step limit without task completion."""
    if not trajectories:
        return 0.0
    stalls = sum(1 for t in trajectories if t.total_steps >= max_steps and not t.is_success)
    return round(stalls / len(trajectories), 4)


def evaluate_benchmark_run(
    trajectories: List[TrajectoryRecord],
    tasks: List[TaskSpec],
    model_id: str = "CohereLabs/tiny-aya-global",
    run_id: str = "run_baseline_001",
    seed: int = 42,
) -> AuditKitRunResult:
    """Execute complete AuditKIT M1-M7 evaluation pipeline and produce RunResult."""
    # 1. Success indicator array for bootstrap
    success_vals = [1.0 if t.is_success else 0.0 for t in trajectories]
    m1_val = task_success_rate(trajectories)
    m1_low, m1_high = compute_bootstrap_ci(success_vals, seed=seed)

    # 2. Tool accuracy
    m2_val = tool_selection_accuracy(trajectories, tasks)

    # 3. Schema validity
    m3_val = schema_validity_rate(trajectories)

    # 4. Call order validity
    m4_val = call_order_validity_rate(trajectories)

    # 5. Step efficiency on successes
    m5_val = mean_successful_steps(trajectories)

    # 6. Redundant call rate
    m6_val = redundant_call_rate(trajectories)

    # 7. Loop and stall rate
    m7_val = loop_and_stall_frequency(trajectories)

    # Compile metric map
    metrics = {
        "M1_task_success_rate": AuditKitMetricResult(
            metric_name="M1_task_success_rate",
            value=m1_val,
            ci_lower=m1_low,
            ci_upper=m1_high,
            sample_count=len(trajectories),
        ),
        "M2_tool_call_accuracy": AuditKitMetricResult(
            metric_name="M2_tool_call_accuracy",
            value=m2_val,
            sample_count=len(trajectories),
        ),
        "M3_schema_validity": AuditKitMetricResult(
            metric_name="M3_schema_validity",
            value=m3_val,
            sample_count=len(trajectories),
        ),
        "M4_call_order_validity": AuditKitMetricResult(
            metric_name="M4_call_order_validity",
            value=m4_val,
            sample_count=len(trajectories),
        ),
        "M5_trajectory_step_efficiency": AuditKitMetricResult(
            metric_name="M5_trajectory_step_efficiency",
            value=m5_val,
            sample_count=sum(1 for t in trajectories if t.is_success),
        ),
        "M6_redundant_call_rate": AuditKitMetricResult(
            metric_name="M6_redundant_call_rate",
            value=m6_val,
            sample_count=len(trajectories),
        ),
        "M7_loop_and_stall_frequency": AuditKitMetricResult(
            metric_name="M7_loop_and_stall_frequency",
            value=m7_val,
            sample_count=len(trajectories),
        ),
    }

    # Failure mode taxonomy breakdown
    task_map = {t.task_id: t for t in tasks}
    failure_breakdown: Dict[str, int] = {cat.value: 0 for cat in FailureCategory}
    for traj in trajectories:
        if not traj.is_success:
            task = task_map.get(traj.task_id)
            cat = classify_failure(traj, task)
            if cat:
                failure_breakdown[cat.value] += 1

    # Fingerprint
    fp = compute_run_fingerprint(
        model_id=model_id,
        dataset_hash=f"n_tasks_{len(tasks)}",
        config={"eval_version": "1.0", "metrics": list(metrics.keys())},
        seed=seed,
    )

    return AuditKitRunResult(
        run_id=run_id,
        model_id=model_id,
        timestamp=time.time(),
        sha256_fingerprint=fp,
        metrics=metrics,
        failure_breakdown=failure_breakdown,
        provenance_metadata={"task_count": len(tasks), "trajectory_count": len(trajectories)},
    )

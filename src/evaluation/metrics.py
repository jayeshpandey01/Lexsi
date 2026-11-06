"""Evaluation metrics suite (M1-M7) and AuditKIT RunResult adapter.

Implements rigorous metric calculations, defensive handling for unmeasured
and edge cases (AuditKIT PR #18), deterministic single-sample bootstrap CIs,
and paired bootstrap hypothesis testing (AuditKIT PR #9).
"""

from __future__ import annotations

import json
import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from src.data.schemas import TrajectoryRecord
from src.environment.task_generator import TaskSpec
from src.evaluation.taxonomy import FailureCategory, classify_failure
from src.utils.compatibility import (
    AuditKitAdapter,
    AuditKitMetricResult,
    AuditKitRunResult,
    PairedComparisonResult,
    TaskEvaluationResult,
    compute_run_fingerprint,
)


# Canonical schema & semantic error indicators
ARGUMENT_ERROR_CODES: Set[str] = {
    "SCHEMA_VALIDATION_ERROR",
    "SEMANTIC_BOUND_ERROR",
    "MALFORMED_TOOL_ARGUMENTS",
    "JSON_PARSE_ERROR",
    "EXTRA_FORBIDDEN_ARGUMENTS",
    "INVALID_ARGUMENT_TYPE",
}


def canonical_call_hash(tool_name: str, arguments: Optional[Dict[str, Any]]) -> str:
    """Generate a deterministic, canonical JSON-serialized signature for a tool call.

    Sorts dictionary keys recursively and normalizes separators to prevent hashing divergence.
    """
    if not arguments:
        return f"{tool_name}:{{}}"
    try:
        serialized = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        serialized = str(sorted(arguments.items()))
    return f"{tool_name}:{serialized}"


def compute_bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute deterministic single-sample non-parametric bootstrap confidence interval.

    Estimates parameter uncertainty for a sample mean: returns [lower, upper] percentile interval.
    Note: This is a descriptive confidence interval, NOT a paired significance test.
    """
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


def compute_paired_bootstrap_comparison(
    baseline_trajectories: List[TrajectoryRecord],
    candidate_trajectories: List[TrajectoryRecord],
    tasks: Optional[List[TaskSpec]] = None,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
    metric_name: str = "M1_task_success_rate",
) -> PairedComparisonResult:
    """Execute rigorous paired-bootstrap hypothesis test between baseline and candidate runs.

    Aligns runs by task_id, resamples paired differences D_i = Y_i - X_i with replacement,
    computes the empirical two-sided p-value against H0: Delta = 0, and reports
    delta confidence intervals and significance determinations (AuditKIT PR #9).
    """
    base_map = {t.task_id: (1.0 if t.is_success else 0.0) for t in baseline_trajectories}
    cand_map = {t.task_id: (1.0 if t.is_success else 0.0) for t in candidate_trajectories}

    # If task set provided, ensure all declared tasks are represented
    if tasks:
        aligned_task_ids = [t.task_id for t in tasks]
    else:
        aligned_task_ids = sorted(list(set(base_map.keys()) & set(cand_map.keys())))

    if not aligned_task_ids:
        return PairedComparisonResult(
            metric_name=metric_name,
            baseline_mean=0.0,
            candidate_mean=0.0,
            delta=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
            is_significant=False,
            sample_count=0,
            method="paired_bootstrap_resampling",
            caveats=["Zero common task IDs found between baseline and candidate runs."],
        )

    x = np.array([base_map.get(tid, 0.0) for tid in aligned_task_ids], dtype=np.float64)
    y = np.array([cand_map.get(tid, 0.0) for tid in aligned_task_ids], dtype=np.float64)

    n = len(aligned_task_ids)
    baseline_mean = float(np.mean(x))
    candidate_mean = float(np.mean(y))
    observed_delta = candidate_mean - baseline_mean

    # Paired differences
    diffs = y - x

    rng = np.random.default_rng(seed)
    # Resample paired differences
    bootstrap_indices = rng.integers(0, n, size=(n_resamples, n))
    bootstrap_deltas = np.mean(diffs[bootstrap_indices], axis=1)

    # 95% bootstrap CI for Delta
    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(bootstrap_deltas, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_deltas, 100.0 * (1.0 - alpha / 2.0)))

    # Empirical two-sided p-value under null hypothesis H0: Delta = 0
    # Center bootstrap distribution at 0 by subtracting observed delta
    centered_deltas = bootstrap_deltas - observed_delta
    p_lower = np.mean(centered_deltas <= -abs(observed_delta))
    p_upper = np.mean(centered_deltas >= abs(observed_delta))
    p_val = float(min(1.0, 2.0 * min(p_lower, p_upper)))

    # Significance decision rule: p < 0.05 and 0 outside CI
    is_sig = (p_val < (1.0 - ci_level)) and (ci_lower > 0.0 or ci_upper < 0.0)

    caveats: List[str] = []
    if n < 30:
        caveats.append(f"Small sample size (N={n} < 30): descriptive comparison recommended; asymptotic properties unverified.")
    if observed_delta == 0.0:
        caveats.append("Observed delta is exactly zero; candidate matches baseline.")

    return PairedComparisonResult(
        metric_name=metric_name,
        baseline_mean=round(baseline_mean, 4),
        candidate_mean=round(candidate_mean, 4),
        delta=round(observed_delta, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        p_value=round(p_val, 4),
        is_significant=is_sig,
        sample_count=n,
        method=f"paired_bootstrap_resampling (B={n_resamples})",
        caveats=caveats,
    )


def task_success_rate(
    trajectories: List[TrajectoryRecord],
    tasks: Optional[List[TaskSpec]] = None,
) -> float:
    """M1: Fraction of tasks satisfying deterministic environment success predicate.

    When `tasks` is provided, ensures all declared tasks are represented in the denominator.
    Missing or unmeasured tasks count as 0.0 in the denominator.
    """
    if not trajectories and not tasks:
        return 0.0

    denom = max(len(tasks) if tasks else 0, len(trajectories))
    if denom == 0:
        return 0.0

    successes = sum(1 for t in trajectories if t.is_success)
    return round(successes / denom, 4)


def tool_selection_accuracy(
    trajectories: List[TrajectoryRecord],
    tasks: List[TaskSpec],
) -> float:
    """M2: Ratio of correctly chosen tools over total executed tool calls (Call Precision)."""
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


def compute_tool_selection_metrics(
    trajectories: List[TrajectoryRecord],
    tasks: List[TaskSpec],
) -> Dict[str, float]:
    """Compute detailed tool selection precision and task coverage metrics."""
    total_calls = 0
    correct_calls = 0
    coverage_sum = 0.0

    task_map = {t.task_id: t for t in tasks}
    for traj in trajectories:
        task = task_map.get(traj.task_id)
        expected = set(task.expected_tools) if task else set()
        called_set: Set[str] = set()

        for step in traj.steps:
            if step.action:
                total_calls += 1
                called_set.add(step.action.tool_name)
                if step.action.tool_name in expected:
                    correct_calls += 1

        if expected:
            coverage = len(called_set & expected) / len(expected)
        else:
            coverage = 1.0 if not called_set else 0.0
        coverage_sum += coverage

    precision = (correct_calls / total_calls) if total_calls > 0 else 0.0
    mean_coverage = (coverage_sum / len(tasks)) if tasks else 0.0

    return {
        "call_precision": round(precision, 4),
        "task_tool_coverage": round(mean_coverage, 4),
    }


def schema_validity_rate(trajectories: List[TrajectoryRecord]) -> float:
    """M3: Ratio of schema- and semantic-valid tool invocations over total tool calls.

    Audits all argument errors including SCHEMA_VALIDATION_ERROR, SEMANTIC_BOUND_ERROR,
    MALFORMED_TOOL_ARGUMENTS, and JSON_PARSE_ERROR.
    """
    total_calls = 0
    schema_errors = 0

    for traj in trajectories:
        for step in traj.steps:
            if step.action:
                total_calls += 1
                if step.error and any(code in step.error for code in ARGUMENT_ERROR_CODES):
                    schema_errors += 1
                elif isinstance(step.observation, dict) and step.observation.get("is_error"):
                    error_msg = str(step.observation.get("error", ""))
                    if any(code in error_msg for code in ARGUMENT_ERROR_CODES):
                        schema_errors += 1

    if total_calls == 0:
        return 1.0  # Zero invalid calls attempted
    return round((total_calls - schema_errors) / total_calls, 4)


def call_order_validity_rate(
    trajectories: List[TrajectoryRecord],
    tasks: Optional[List[TaskSpec]] = None,
) -> float:
    """M4: Ratio of multi-step episodes satisfying dependency execution order.

    Generic dependency check:
      - Any call to `compute_aggregate` requires prior successful `lookup_records`.
      - Any call to `verify_constraint` requires prior successful `lookup_records`.
      - Validates every invocation of downstream tools across the entire trajectory.
    """
    multi_step_count = 0
    valid_order_count = 0

    for traj in trajectories:
        called_tools = [s.action.tool_name for s in traj.steps if s.action]
        # Check if episode involves multi-step dependent tools
        has_downstream = "compute_aggregate" in called_tools or "verify_constraint" in called_tools
        if not has_downstream:
            continue

        multi_step_count += 1
        is_order_valid = True
        executed_prior: Set[str] = set()

        for step in traj.steps:
            if not step.action:
                continue
            t_name = step.action.tool_name

            # Check prerequisite rules
            if t_name == "compute_aggregate" and "lookup_records" not in executed_prior:
                is_order_valid = False
                break
            if t_name == "verify_constraint" and "lookup_records" not in executed_prior:
                is_order_valid = False
                break

            # If call was successful, mark prerequisite as satisfied
            if step.error is None:
                executed_prior.add(t_name)

        if is_order_valid:
            valid_order_count += 1

    if multi_step_count == 0:
        return 1.0  # No multi-step dependencies to violate
    return round(valid_order_count / multi_step_count, 4)


def mean_successful_steps(trajectories: List[TrajectoryRecord]) -> Optional[float]:
    """M5: Mean execution steps conditional on verified task success: E[T | Success=1].

    Returns None (unmeasured) if zero tasks succeed, preventing false zero-step reporting.
    """
    successful_steps = [t.total_steps for t in trajectories if t.is_success]
    if not successful_steps:
        return None
    return round(sum(successful_steps) / len(successful_steps), 4)


def redundant_call_rate(trajectories: List[TrajectoryRecord]) -> float:
    """M6: Frequency of duplicate or repeated identical tool calls.

    Generates canonical JSON argument signatures to safely handle nested dicts and lists.
    """
    total_calls = 0
    duplicate_calls = 0

    for traj in trajectories:
        seen: Set[str] = set()
        for step in traj.steps:
            if step.action:
                total_calls += 1
                key = canonical_call_hash(step.action.tool_name, step.action.arguments)
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
    """M7: Fraction of evaluated runs hitting max steps, timeouts, or cycle loops."""
    if not trajectories:
        return 0.0

    stalls = 0
    for traj in trajectories:
        # Condition 1: Exceeded step limit without success
        if traj.total_steps >= max_steps and not traj.is_success:
            stalls += 1
            continue

        # Condition 2: Explicit timeout or step limit error
        traj_err = getattr(traj, "error_type", None) or traj.metadata.get("error_type")
        if not traj_err and traj.steps and traj.steps[-1].error:
            traj_err = traj.steps[-1].error.split(":")[0].strip()
        if traj_err in ("STEP_LIMIT_EXCEEDED", "TIMEOUT", "EXECUTION_TIMEOUT"):
            stalls += 1
            continue

        # Condition 3: 3+ consecutive duplicate calls or 2-cycle loops
        call_hashes = [
            canonical_call_hash(s.action.tool_name, s.action.arguments)
            for s in traj.steps
            if s.action
        ]
        is_loop = False

        # Check consecutive duplicates
        consecutive = 1
        for i in range(1, len(call_hashes)):
            if call_hashes[i] == call_hashes[i - 1]:
                consecutive += 1
                if consecutive >= 3:
                    is_loop = True
                    break
            else:
                consecutive = 1

        # Check alternating 2-cycle (A -> B -> A -> B)
        if not is_loop and len(call_hashes) >= 4:
            for i in range(len(call_hashes) - 3):
                if (
                    call_hashes[i] == call_hashes[i + 2]
                    and call_hashes[i + 1] == call_hashes[i + 3]
                    and call_hashes[i] != call_hashes[i + 1]
                ):
                    is_loop = True
                    break

        if is_loop and not traj.is_success:
            stalls += 1

    return round(stalls / len(trajectories), 4)


def evaluate_benchmark_run(
    trajectories: List[TrajectoryRecord],
    tasks: List[TaskSpec],
    model_id: str = "CohereLabs/tiny-aya-global",
    run_id: str = "run_baseline_001",
    split: str = "val",
    seed: int = 42,
) -> AuditKitRunResult:
    """Execute complete AuditKIT M1-M7 evaluation pipeline and produce RunResult."""
    task_map = {t.task_id: t for t in tasks}

    # 1. M1: Success Rate aligned to task manifest
    m1_val = task_success_rate(trajectories, tasks=tasks)
    success_vals: List[float] = [1.0 if t.is_success else 0.0 for t in trajectories]
    if tasks and len(tasks) > len(trajectories):
        # Account for dropped / unmeasured tasks in bootstrap distribution
        success_vals.extend([0.0] * (len(tasks) - len(trajectories)))
    m1_low, m1_high = compute_bootstrap_ci(success_vals, seed=seed)

    # 2. M2: Tool selection accuracy and coverage
    m2_val = tool_selection_accuracy(trajectories, tasks)
    m2_details = compute_tool_selection_metrics(trajectories, tasks)

    # 3. M3: Schema validity rate
    m3_val = schema_validity_rate(trajectories)

    # 4. M4: Call order validity rate
    m4_val = call_order_validity_rate(trajectories, tasks=tasks)

    # 5. M5: Trajectory step efficiency conditional on success
    m5_val = mean_successful_steps(trajectories)
    successful_count = sum(1 for t in trajectories if t.is_success)

    # 6. M6: Redundant call rate
    m6_val = redundant_call_rate(trajectories)

    # 7. M7: Loop and stall frequency
    m7_val = loop_and_stall_frequency(trajectories)

    # Compile AuditKIT metric map
    metrics: Dict[str, AuditKitMetricResult] = {
        "M1_task_success_rate": AuditKitMetricResult(
            metric_name="M1_task_success_rate",
            value=m1_val,
            direction="maximize",
            unit="fraction",
            ci_lower=m1_low,
            ci_upper=m1_high,
            sample_count=len(tasks) if tasks else len(trajectories),
        ),
        "M2_tool_call_accuracy": AuditKitMetricResult(
            metric_name="M2_tool_call_accuracy",
            value=m2_val,
            direction="maximize",
            unit="fraction",
            sample_count=len(tasks) if tasks else len(trajectories),
            details=m2_details,
        ),
        "M3_schema_validity": AuditKitMetricResult(
            metric_name="M3_schema_validity",
            value=m3_val,
            direction="maximize",
            unit="fraction",
            sample_count=len(trajectories),
        ),
        "M4_call_order_validity": AuditKitMetricResult(
            metric_name="M4_call_order_validity",
            value=m4_val,
            direction="maximize",
            unit="fraction",
            sample_count=len(trajectories),
        ),
        "M5_trajectory_step_efficiency": AuditKitMetricResult(
            metric_name="M5_trajectory_step_efficiency",
            value=m5_val,
            direction="minimize",
            unit="steps",
            sample_count=successful_count,
            unmeasured_count=0 if m5_val is not None else len(trajectories),
        ),
        "M6_redundant_call_rate": AuditKitMetricResult(
            metric_name="M6_redundant_call_rate",
            value=m6_val,
            direction="minimize",
            unit="fraction",
            sample_count=len(trajectories),
        ),
        "M7_loop_and_stall_frequency": AuditKitMetricResult(
            metric_name="M7_loop_and_stall_frequency",
            value=m7_val,
            direction="minimize",
            unit="fraction",
            sample_count=len(trajectories),
        ),
    }

    # Itemized per-task evaluation breakdown
    per_task_results: List[TaskEvaluationResult] = []
    traj_by_task = {t.task_id: t for t in trajectories}

    for task in tasks:
        t = traj_by_task.get(task.task_id)
        if t is None:
            # Missing / unmeasured task
            per_task_results.append(
                TaskEvaluationResult(
                    task_id=task.task_id,
                    template_id=task.template_id,
                    is_success=False,
                    total_steps=0,
                    executed_tools=[],
                    expected_tools=task.expected_tools,
                    error_type="UNMEASURED_OR_DROPPED_TASK",
                    diagnosed_failure=FailureCategory.EARLY_TERMINATION_FAILURE.value,
                )
            )
        else:
            executed = [s.action.tool_name for s in t.steps if s.action]
            has_schema = any(
                s.error and any(c in s.error for c in ARGUMENT_ERROR_CODES)
                for s in t.steps
            )
            has_order = False
            if "compute_aggregate" in executed and "lookup_records" in executed:
                has_order = executed.index("lookup_records") > executed.index("compute_aggregate")
            diag = classify_failure(t, task) if not t.is_success else None

            traj_err = getattr(t, "error_type", None) or t.metadata.get("error_type")
            if not traj_err and t.steps and t.steps[-1].error:
                traj_err = t.steps[-1].error.split(":")[0].strip()

            per_task_results.append(
                TaskEvaluationResult(
                    task_id=task.task_id,
                    template_id=task.template_id,
                    is_success=t.is_success,
                    total_steps=t.total_steps,
                    executed_tools=executed,
                    expected_tools=task.expected_tools,
                    has_schema_error=has_schema,
                    has_order_error=has_order,
                    is_stall=(t.total_steps >= 8 and not t.is_success),
                    redundant_calls=max(0, len(executed) - len(set(executed))),
                    error_type=traj_err,
                    diagnosed_failure=diag.value if diag else None,
                )
            )

    # Failure mode taxonomy breakdown
    failure_breakdown: Dict[str, int] = {cat.value: 0 for cat in FailureCategory}
    for traj in trajectories:
        if not traj.is_success:
            task = task_map.get(traj.task_id)
            cat = classify_failure(traj, task)
            if cat:
                failure_breakdown[cat.value] += 1

    # Canonical fingerprint
    fp = compute_run_fingerprint(
        model_id=model_id,
        dataset_hash=f"n_tasks_{len(tasks)}",
        config={"eval_version": "1.0", "split": split, "metrics": list(metrics.keys())},
        seed=seed,
    )

    headline = {k: m.value for k, m in metrics.items()}

    return AuditKitRunResult(
        run_id=run_id,
        model_id=model_id,
        evaluator_version=AuditKitAdapter.get_evaluator_version(),
        split=split,
        timestamp=time.time(),
        sha256_fingerprint=fp,
        headline=headline,
        metrics=metrics,
        per_task_results=per_task_results,
        failure_breakdown=failure_breakdown,
        provenance_metadata={
            "task_count": len(tasks),
            "trajectory_count": len(trajectories),
            "seed": seed,
            "split": split,
        },
    )

"""Deterministic, environment-grounded reward calculator and audit logging engine.

Governing Specification: references/reward_spec.md
Traceability: Milestone 3 (M3.P1 & M3.P2), Decision D5, .agents/skills/reward-design-guard/SKILL.md
"""

from __future__ import annotations

import datetime
import math
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

from src.data.schemas import StepRecord, TrajectoryRecord
from src.environment.task_generator import TaskSpec, verify_task_outcome

DEFAULT_ALLOWED_TOOLS: Set[str] = {
    "lookup_records",
    "compute_aggregate",
    "verify_constraint",
}


class RewardComponentBreakdown(BaseModel):
    """Component-level reward attribution."""

    model_config = ConfigDict(extra="forbid")

    r_outcome: float = Field(0.0, description="Ground truth task completion reward in [0.0, 1.0]")
    r_efficiency: float = Field(0.0, description="Step efficiency penalty in [-0.25, 0.0] conditional on success")
    r_parse: float = Field(0.0, description="Summed format and JSON delimiter penalty in [-inf, 0.0]")
    r_tool: float = Field(0.0, description="Summed unauthorized tool selection penalty in [-inf, 0.0]")
    r_args: float = Field(0.0, description="Summed argument and schema validation penalty in [-inf, 0.0]")
    r_order: float = Field(0.0, description="Summed tool dependency order penalty in [-inf, 0.0]")
    r_safety: float = Field(0.0, description="Terminal safety guard / timeout penalty in [-0.50, 0.0]")
    r_process_sum: float = Field(0.0, description="Sum of step-level process penalties")


class DivergenceFlags(BaseModel):
    """Automated alarms for reward-versus-outcome divergence and numerical traps."""

    model_config = ConfigDict(extra="forbid")

    high_reward_failure: bool = Field(False, description="True if R > 0.50 but task failed (Critical Proxy Gaming)")
    low_reward_success: bool = Field(False, description="True if R < 0.50 but task succeeded (Excessive Penalty)")
    non_finite_detected: bool = Field(False, description="True if NaN or Inf was encountered and clamped")


class StepRewardLog(BaseModel):
    """Per-step reward audit and action validation log."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(..., ge=0, description="0-indexed step counter")
    timestamp_iso: str = Field(..., description="ISO 8601 evaluation timestamp")
    action_name: Optional[str] = Field(None, description="Invoked tool name if any")
    parse_valid: bool = Field(True, description="True if action tag and JSON syntax were valid")
    tool_valid: bool = Field(True, description="True if tool is permitted in the environment")
    schema_valid: bool = Field(True, description="True if arguments satisfied Pydantic schema")
    order_valid: bool = Field(True, description="True if tool dependency order was respected")
    components: Dict[str, float] = Field(default_factory=dict, description="Step-level penalty breakdown")
    step_reward_delta: float = Field(0.0, description="Total step reward delta")
    guard_status: str = Field("PERMITTED", description="PERMITTED, BLOCKED, TIMEOUT, or STEP_LIMIT_EXCEEDED")
    elapsed_ms: float = Field(0.0, description="Step duration in milliseconds")


class EpisodeRewardLog(BaseModel):
    """Complete episodic reward audit trail conforming to references/reward_spec.md §5.2."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(..., description="Unique episode identifier")
    task_id: str = Field(..., description="Evaluated task identifier")
    run_id: Optional[str] = Field(None, description="Experiment or batch execution run identifier")
    reward_version: str = Field("m3-reward-v1.0", description="Immutable reward specification version")
    total_reward: float = Field(..., description="Final bounded episode reward")
    is_success: bool = Field(..., description="Authoritative ground-truth task completion status")
    terminal_reason: str = Field(..., description="SUCCESS_VERIFIED, NUMERIC_MISMATCH, TIMEOUT, etc.")
    total_steps: int = Field(..., ge=0, description="Total execution steps in episode")
    component_breakdown: RewardComponentBreakdown = Field(..., description="Component breakdown")
    step_logs: List[StepRewardLog] = Field(default_factory=list, description="Ordered step audit logs")
    divergence_flags: DivergenceFlags = Field(..., description="Reward gaming and divergence flags")
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Cryptographic hashes and seeds")


def compute_step_reward(
    step: StepRecord,
    prior_steps: List[StepRecord],
    allowed_tools: Set[str],
) -> StepRewardLog:
    """Calculate process shaping penalties for a single discrete step.

    Components Evaluated:
    - R_parse: -0.10 if output parsing failed.
    - R_tool: -0.10 if tool name is unknown or unpermitted.
    - R_args: -0.10 if tool arguments failed schema/type validation.
    - R_order: -0.05 if compute_aggregate/verify_constraint called before lookup_records.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r_parse = 0.0
    r_tool = 0.0
    r_args = 0.0
    r_order = 0.0

    parse_valid = True
    tool_valid = True
    schema_valid = True
    order_valid = True
    guard_status = "PERMITTED"

    action_name: Optional[str] = None
    if step.action:
        action_name = step.action.tool_name

    # 1. Parse validity check
    if step.error and ("SCHEMA_VALIDATION_ERROR" in step.error or "PARSE_ERROR" in step.error):
        if not step.action:
            r_parse = -0.10
            parse_valid = False

    # 2. Tool permission & argument validity check
    if step.action:
        if action_name not in allowed_tools:
            r_tool = -0.10
            tool_valid = False
            guard_status = "BLOCKED"

        if step.error:
            err_lower = step.error.lower()
            if "schema" in err_lower or "validation" in err_lower or "missing" in err_lower or "invalid" in err_lower:
                r_args = -0.10
                schema_valid = False
            elif "unauthorized" in err_lower or "permission" in err_lower:
                r_tool = -0.10
                tool_valid = False
                guard_status = "BLOCKED"

        # 3. Order dependency check: aggregate or verify requires at least one prior lookup
        if action_name in ("compute_aggregate", "verify_constraint"):
            prior_lookups = [
                s for s in prior_steps
                if s.action and s.action.tool_name == "lookup_records" and not s.error
            ]
            if not prior_lookups:
                r_order = -0.05
                order_valid = False

    # 4. Guard status checks from step errors
    if step.error:
        if "STEP_LIMIT" in step.error:
            guard_status = "STEP_LIMIT_EXCEEDED"
        elif "TIMEOUT" in step.error:
            guard_status = "TIMEOUT"

    components = {
        "r_parse": r_parse,
        "r_tool": r_tool,
        "r_args": r_args,
        "r_order": r_order,
    }
    step_delta = r_parse + r_tool + r_args + r_order

    return StepRewardLog(
        step_index=step.step_index,
        timestamp_iso=timestamp,
        action_name=action_name,
        parse_valid=parse_valid,
        tool_valid=tool_valid,
        schema_valid=schema_valid,
        order_valid=order_valid,
        components=components,
        step_reward_delta=round(step_delta, 4),
        guard_status=guard_status,
        elapsed_ms=step.elapsed_ms,
    )


def compute_episode_reward(
    trajectory: TrajectoryRecord,
    task: Optional[TaskSpec] = None,
    allowed_tools: Optional[Set[str]] = None,
    t_min: int = 2,
    efficiency_coeff: float = -0.05,
    max_efficiency_penalty: float = -0.25,
) -> EpisodeRewardLog:
    """Compute the total episode reward and emit a structured audit log.

    Enforces all invariants from references/reward_spec.md:
    1. Environment outcome verification is authoritative; model self-reports receive 0.
    2. R_efficiency is applied STRICTLY when is_success is True.
    3. Missing or non-finite values fail closed (clamped to -1.00).
    4. Automatically flags divergence (high_reward_failure, low_reward_success).
    """
    tools = allowed_tools or DEFAULT_ALLOWED_TOOLS
    step_logs: List[StepRewardLog] = []

    # 1. Step-level penalties
    prior_steps: List[StepRecord] = []
    r_parse_total = 0.0
    r_tool_total = 0.0
    r_args_total = 0.0
    r_order_total = 0.0

    has_safety_breach = False

    for s in trajectory.steps:
        s_log = compute_step_reward(s, prior_steps, tools)
        step_logs.append(s_log)
        r_parse_total += s_log.components.get("r_parse", 0.0)
        r_tool_total += s_log.components.get("r_tool", 0.0)
        r_args_total += s_log.components.get("r_args", 0.0)
        r_order_total += s_log.components.get("r_order", 0.0)

        if s_log.guard_status in ("STEP_LIMIT_EXCEEDED", "TIMEOUT"):
            has_safety_breach = True

        prior_steps.append(s)

    r_process_sum = r_parse_total + r_tool_total + r_args_total + r_order_total

    # 2. Authoritative outcome verification
    is_success = trajectory.is_success
    terminal_reason = "SUCCESS_VERIFIED" if is_success else "NUMERIC_MISMATCH"

    if task is not None:
        # Cross-verify against task ground truth to override unverified trajectory flags
        verified_outcome = verify_task_outcome(task, trajectory.final_output)
        executed_tools = {s.action.tool_name for s in trajectory.steps if s.action and not s.error}

        if not verified_outcome:
            is_success = False
            if not trajectory.final_output or not trajectory.final_output.strip():
                terminal_reason = "EMPTY_OUTPUT"
            else:
                terminal_reason = "NUMERIC_MISMATCH"
        elif task.expected_tools and not any(tool in executed_tools for tool in task.expected_tools):
            # Anti-cheating guard: cannot claim success without executing required tools
            is_success = False
            terminal_reason = "SKIPPED_REQUIRED_TOOLS"
        else:
            is_success = True
            terminal_reason = "SUCCESS_VERIFIED"

    # Terminal guard reason overrides
    if has_safety_breach:
        terminal_reason = "MAX_STEPS_REACHED" if any(s.guard_status == "STEP_LIMIT_EXCEEDED" for s in step_logs) else "TIMEOUT"

    r_outcome = 1.00 if is_success else 0.00

    # 3. Efficiency penalty (Applied STRICTLY conditional on success)
    # Proof: Prevents empty no-op or premature quit policies from outscoring failed exploratory attempts
    r_efficiency = 0.0
    total_steps = len(trajectory.steps)
    if is_success:
        excess_steps = max(0, total_steps - t_min)
        raw_eff = efficiency_coeff * excess_steps
        if math.isnan(raw_eff) or math.isinf(raw_eff):
            r_efficiency = float("nan")
        else:
            r_efficiency = max(max_efficiency_penalty, raw_eff)

    # 4. Terminal safety penalty
    r_safety = -0.50 if has_safety_breach else 0.00

    # 5. Total aggregation and fail-closed non-finite trap
    total_raw = r_outcome + r_efficiency + r_process_sum + r_safety
    non_finite_detected = False

    if (
        math.isnan(total_raw)
        or math.isinf(total_raw)
        or math.isnan(r_outcome)
        or math.isnan(r_efficiency)
        or math.isnan(r_process_sum)
        or any(
            math.isnan(v) or math.isinf(v)
            for s in step_logs
            for v in s.components.values()
        )
    ):
        non_finite_detected = True
        total_raw = -1.00
        if math.isnan(r_efficiency) or math.isinf(r_efficiency):
            r_efficiency = -1.00
        if math.isnan(r_outcome) or math.isinf(r_outcome):
            r_outcome = 0.00
        if math.isnan(r_process_sum) or math.isinf(r_process_sum):
            r_process_sum = 0.00

    total_reward = round(total_raw, 4)

    # 6. Automated divergence alarms
    high_reward_failure = (total_reward > 0.50 and not is_success)
    low_reward_success = (total_reward < 0.50 and is_success)

    breakdown = RewardComponentBreakdown(
        r_outcome=r_outcome,
        r_efficiency=round(r_efficiency, 4),
        r_parse=round(r_parse_total, 4),
        r_tool=round(r_tool_total, 4),
        r_args=round(r_args_total, 4),
        r_order=round(r_order_total, 4),
        r_safety=r_safety,
        r_process_sum=round(r_process_sum, 4),
    )

    divergence = DivergenceFlags(
        high_reward_failure=high_reward_failure,
        low_reward_success=low_reward_success,
        non_finite_detected=non_finite_detected,
    )

    episode_id = f"ep_{trajectory.task_id}_{trajectory.seed}"

    return EpisodeRewardLog(
        episode_id=episode_id,
        task_id=trajectory.task_id,
        run_id=trajectory.run_id,
        reward_version="m3-reward-v1.0",
        total_reward=total_reward,
        is_success=is_success,
        terminal_reason=terminal_reason,
        total_steps=total_steps,
        component_breakdown=breakdown,
        step_logs=step_logs,
        divergence_flags=divergence,
        provenance={
            "fingerprint": trajectory.fingerprint,
            "seed": trajectory.seed,
            "environment_version": trajectory.environment_version,
        },
    )

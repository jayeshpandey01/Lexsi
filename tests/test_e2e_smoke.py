"""End-to-end smoke test verifying full episode orchestration, trace capture, and AuditKIT evaluation."""

import json
from src.environment.task_generator import generate_task_instance
from src.evaluation.metrics import evaluate_benchmark_run
from src.runtime.orchestrator import EpisodeOrchestrator


def test_e2e_successful_episode() -> None:
    """Verify an end-to-end episode where agent policy solves a multi-step task."""
    task = generate_task_instance("train", 0, seed=42)
    orchestrator = EpisodeOrchestrator(seed=42)

    category = task.target_entities["category"]

    # Simulates an agent policy that calls lookup then aggregate then concludes
    call_count = 0

    def mock_policy(dialog: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (
                f'<|START_ACTION|>{{"name": "lookup_records", '
                f'"arguments": {{"table": "inventory_items", "filters": {{"category": "{category}"}}}}}}'
                f"<|END_ACTION|>"
            )
        if call_count == 2:
            # Extract observation from dialog to compute aggregate
            return (
                f'<|START_ACTION|>{{"name": "compute_aggregate", '
                f'"arguments": {{"values": [{task.oracle_answer}], "operation": "sum"}}}}'
                f"<|END_ACTION|>"
            )
        return f"Audit complete. Total inventory valuation is ${task.oracle_answer:.2f}."

    trajectory = orchestrator.run_episode(task, mock_policy)

    # Invariants verification
    assert trajectory.is_success is True
    assert trajectory.total_steps == 3
    assert len(trajectory.steps) == 3
    assert trajectory.steps[0].action.tool_name == "lookup_records"  # type: ignore[union-attr]
    assert trajectory.steps[1].action.tool_name == "compute_aggregate"  # type: ignore[union-attr]

    # Verify serialization and redaction
    redacted_dict = trajectory.to_redacted_dict()
    assert redacted_dict["is_success"] is True
    assert redacted_dict["fingerprint"] is not None

    # Verify AuditKIT benchmark evaluation
    run_result = evaluate_benchmark_run([trajectory], [task], run_id="e2e_smoke_run")
    assert run_result.metrics["M1_task_success_rate"].value == 1.0
    assert run_result.metrics["M3_schema_validity"].value == 1.0
    assert run_result.metrics["M5_trajectory_step_efficiency"].value == 3.0


def test_e2e_malformed_rejection() -> None:
    """Verify safe failure handling on malformed model outputs."""
    task = generate_task_instance("train", 1, seed=42)
    orchestrator = EpisodeOrchestrator(seed=42)

    def bad_policy(dialog: str) -> str:
        # Invalid JSON inside action tags
        return "<|START_ACTION|>{bad_json: True, invalid}<|END_ACTION|>"

    trajectory = orchestrator.run_episode(task, bad_policy)

    assert trajectory.is_success is False
    assert len(trajectory.steps) == 1
    assert "SCHEMA_VALIDATION_ERROR" in (trajectory.steps[0].error or "")


def test_e2e_deterministic_replay() -> None:
    """Verify deterministic replay produces identical trajectory records under identical seeds."""
    task = generate_task_instance("train", 0, seed=42)

    def simple_policy(dialog: str) -> str:
        if "lookup_records" not in dialog:
            return '<|START_ACTION|>{"name": "lookup_records", "arguments": {"table": "vendors"}}<|END_ACTION|>'
        return f"Calculated answer is {task.oracle_answer}."

    orch1 = EpisodeOrchestrator(seed=42, run_id="replay_run_1")
    traj1 = orch1.run_episode(task, simple_policy)

    orch2 = EpisodeOrchestrator(seed=42, run_id="replay_run_1")
    traj2 = orch2.run_episode(task, simple_policy)

    assert traj1.is_success == traj2.is_success
    assert traj1.total_steps == traj2.total_steps
    assert traj1.final_output == traj2.final_output
    assert traj1.steps[0].observation == traj2.steps[0].observation
    assert traj1.fingerprint == traj2.fingerprint


def test_e2e_t4_verdict_verification() -> None:
    """Verify T4 budget cap task checks both numerical value and boolean compliance verdict."""
    # Test split index 1 is T4_budget_cap_verification
    task = generate_task_instance("test", 1, seed=42)
    assert task.template_id == "T4_budget_cap_verification"
    assert task.oracle_verdict is not None

    orch = EpisodeOrchestrator(seed=42)

    # 1. Correct spend and correct verdict -> Success
    verdict_str = "within budget" if task.oracle_verdict else "exceeds budget and is non-compliant"
    policy_correct = lambda d: f"Audited total spend is ${task.oracle_answer:.2f}. Status: {verdict_str}."
    traj_correct = orch.run_episode(task, policy_correct)
    assert traj_correct.is_success is True

    # 2. Correct spend but inverted/wrong verdict -> Failure
    wrong_verdict_str = "exceeds budget and is non-compliant" if task.oracle_verdict else "within budget and compliant"
    policy_wrong_verdict = lambda d: f"Audited total spend is ${task.oracle_answer:.2f}. Status: {wrong_verdict_str}."
    traj_wrong_verdict = orch.run_episode(task, policy_wrong_verdict)
    assert traj_wrong_verdict.is_success is False


def test_e2e_distinct_seeds_produce_distinct_tasks() -> None:
    """Verify different seeds produce distinct task instances when generator contract specifies."""
    task_seed_42 = generate_task_instance("train", 0, seed=42)
    task_seed_99 = generate_task_instance("train", 0, seed=99)
    assert task_seed_42.provenance_hash != task_seed_99.provenance_hash


"""Unit tests for offline RL architecture plumbing smoke runner (M3.P3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.data.schemas import LockedTestExportError
from src.training.rl_spike import RLPlumbingRunner, RLSpikeConfig


def test_rl_spike_config_defaults() -> None:
    """Verify RLSpikeConfig defaults and constraints."""
    cfg = RLSpikeConfig()
    assert cfg.run_id == "rl_plumbing_smoke_001"
    assert cfg.seed == 42
    assert cfg.num_steps == 2
    assert cfg.learning_rate > 0.0
    assert cfg.max_runtime_seconds == 60.0


def test_rl_spike_train_tasks_isolation() -> None:
    """Verify load_train_tasks only loads train tasks and respects split boundaries."""
    cfg = RLSpikeConfig(num_steps=2)
    runner = RLPlumbingRunner(cfg)
    tasks = runner.load_train_tasks(max_tasks=2)

    assert len(tasks) == 2
    for t in tasks:
        assert t.split == "train"
        assert not t.task_id.startswith("task_test_")


def test_rl_spike_execution_and_parameter_isolation(tmp_path: Path) -> None:
    """Verify complete end-to-end RL plumbing execution, gradient flow, and checkpoint reload."""
    cfg = RLSpikeConfig(
        run_id="test_smoke_spike",
        seed=42,
        num_steps=2,
        output_dir=str(tmp_path / "spike_run"),
    )
    runner = RLPlumbingRunner(cfg)

    # Verify base weights are frozen
    base_params_grad = [p.requires_grad for n, p in runner.model.named_parameters() if "lora" not in n]
    assert not any(base_params_grad), "Base parameters must be frozen"

    # Verify adapter weights are trainable
    adapter_params_grad = [p.requires_grad for n, p in runner.model.named_parameters() if "lora" in n]
    assert all(adapter_params_grad), "Adapter parameters must require grad"

    res = runner.execute_spike()

    assert res["status"] == "COMPLETED_SMOKE_CHECK"
    assert res["steps_completed"] == 2
    assert res["checkpoint_reloaded"] is True

    # Verify persisted artifacts
    out_dir = Path(cfg.output_dir)
    assert (out_dir / "step_metrics.csv").exists()
    assert (out_dir / "reward_logs.jsonl").exists()
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "checkpoint" / "adapter_model.pt").exists()

    # Verify manifest labeling
    with open(out_dir / "run_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["status"] == "COMPLETED_SMOKE_CHECK"
    assert "Plumbing Verification" in manifest["purpose"]
    assert "Test Double" in manifest["model_backend"]


def test_rl_spike_ablation_validation() -> None:
    """Verify pre-registered condition configuration validation."""
    # Condition 1: valid outcome-only
    cfg1 = RLSpikeConfig(ablation_condition="outcome_only", efficiency_coeff=0.0)
    assert cfg1.ablation_condition == "outcome_only"
    assert cfg1.efficiency_coeff == 0.0

    # Condition 2: valid outcome_plus_efficiency
    cfg2 = RLSpikeConfig(ablation_condition="outcome_plus_efficiency", efficiency_coeff=-0.05)
    assert cfg2.ablation_condition == "outcome_plus_efficiency"
    assert cfg2.efficiency_coeff == -0.05

    # Invalid condition name
    with pytest.raises(ValueError, match="ablation_condition must be one of"):
        RLSpikeConfig(ablation_condition="unsupported_condition")

    # Invalid coupling: outcome_only with non-zero efficiency_coeff
    with pytest.raises(ValueError, match="When ablation_condition is 'outcome_only', efficiency_coeff must be 0.0"):
        RLSpikeConfig(ablation_condition="outcome_only", efficiency_coeff=-0.05)


def test_rl_spike_ablation_execution(tmp_path: Path) -> None:
    """Verify execution of both pre-registered conditions in the offline runner."""
    # 1. Condition 1: Outcome-only
    cfg1 = RLSpikeConfig(
        run_id="test_ablation_c1",
        seed=42,
        num_steps=1,
        ablation_condition="outcome_only",
        efficiency_coeff=0.0,
        output_dir=str(tmp_path / "c1"),
    )
    runner1 = RLPlumbingRunner(cfg1)
    res1 = runner1.execute_spike()
    assert res1["status"] == "COMPLETED_SMOKE_CHECK"
    assert (tmp_path / "c1" / "step_metrics.csv").exists()

    # 2. Condition 2: Outcome plus efficiency
    cfg2 = RLSpikeConfig(
        run_id="test_ablation_c2",
        seed=42,
        num_steps=1,
        ablation_condition="outcome_plus_efficiency",
        efficiency_coeff=-0.05,
        output_dir=str(tmp_path / "c2"),
    )
    runner2 = RLPlumbingRunner(cfg2)
    res2 = runner2.execute_spike()
    assert res2["status"] == "COMPLETED_SMOKE_CHECK"
    assert (tmp_path / "c2" / "step_metrics.csv").exists()



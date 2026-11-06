"""Unit tests for SFT training harness, dry run, bounded pilot, and stop conditions (M2.P4)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import yaml

from src.data.schemas import LockedTestExportError
from src.training.trainer import (
    LoRALinear,
    SFTExecutionHarness,
    TinyAyaCausalLMDouble,
)


def test_lora_parameter_isolation() -> None:
    """Verify base model weights remain frozen while LoRA parameters are trainable."""
    model = TinyAyaCausalLMDouble(vocab_size=1000, hidden_size=64, num_layers=2)

    trainable_params, total_params, pct = model.get_trainable_parameters()
    assert trainable_params > 0
    assert trainable_params < total_params
    assert 0.0 < pct < 50.0

    # Base embedding and MLP should be frozen
    assert model.embed_tokens.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is False

    # LoRA adapters should be trainable
    for layer in model.layers:
        q_proj = layer["q_proj"]
        assert isinstance(q_proj, LoRALinear)
        assert q_proj.lora_A.requires_grad is True
        assert q_proj.lora_B.requires_grad is True
        assert q_proj.base_linear.weight.requires_grad is False


def test_sft_dry_run_execution(tmp_path: Path) -> None:
    """Verify Stage 1 SFT dry run executes 2 smoke steps, logs finite loss, and persists artifacts."""
    cfg_path = Path("configs/training_sft.yaml")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    harness = SFTExecutionHarness(config)
    dry_run_dir = tmp_path / "dry_run"

    manifest = harness.run_sft_dry_run(output_dir=dry_run_dir, sample_count=2)

    assert manifest["status"] == "COMPLETED"
    assert manifest["step_count"] == 2
    assert 0.0 < manifest["final_loss"] < 20.0
    assert manifest["verification_verdict"] == "DRY_RUN_PASSED_FINITE_GRADIENTS"

    # Verify persisted files
    assert (dry_run_dir / "run_manifest.json").exists()
    assert (dry_run_dir / "training_loss.csv").exists()

    # Check loss CSV has 2 rows
    lines = (dry_run_dir / "training_loss.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # Header + 2 steps


def test_sft_bounded_pilot_execution(tmp_path: Path) -> None:
    """Verify Stage 2 SFT bounded pilot runs to completion, saves adapter checkpoint and samples."""
    cfg_path = Path("configs/training_sft.yaml")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    # Set small step count for unit testing speed
    config["training"]["max_steps"] = 6

    harness = SFTExecutionHarness(config)
    pilot_dir = tmp_path / "pilot"

    manifest = harness.run_bounded_pilot(output_dir=pilot_dir)

    assert manifest["status"] == "COMPLETED"
    assert manifest["max_steps"] == 6
    assert 0.0 < manifest["final_loss"] < 20.0
    assert manifest["sample_count"] == 20

    # Verify checkpoint persistence
    ckpt_dir = pilot_dir / "checkpoint"
    assert (ckpt_dir / "adapter_model.pt").exists()
    assert (ckpt_dir / "adapter_config.json").exists()

    # Verify report and sample generations
    assert (pilot_dir / "training_report.md").exists()
    assert (pilot_dir / "training_loss.csv").exists()
    assert (pilot_dir / "sample_generations.jsonl").exists()

    sample_lines = (pilot_dir / "sample_generations.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sample_lines) > 0
    first_sample = json.loads(sample_lines[0])
    assert "<|START_ACTION|>" in first_sample["generated_action"]


def test_stop_condition_nan_guard(tmp_path: Path) -> None:
    """Verify harness halts immediately with ValueError if model returns NaN loss."""
    cfg_path = Path("configs/training_sft.yaml")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    harness = SFTExecutionHarness(config)

    # Monkeypatch model forward to return NaN loss
    class BrokenModel(TinyAyaCausalLMDouble):
        def forward(self, *args, **kwargs):
            return torch.tensor(float("nan")), torch.tensor(0.0)

    # Test that NaN loss triggers exception in dry run
    from unittest.mock import patch
    with patch("src.training.trainer.TinyAyaCausalLMDouble", BrokenModel):
        with pytest.raises(ValueError, match="non-finite loss"):
            harness.run_sft_dry_run(output_dir=tmp_path / "broken_dry_run")


def test_train_split_sequestration_guard(tmp_path: Path) -> None:
    """Verify harness fails closed if dataset contains sequestered locked-test tasks."""
    cfg_path = Path("configs/training_sft.yaml")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Contaminated ShareGPT dataset
    bad_sg_path = tmp_path / "bad_sharegpt.json"
    bad_data = [
        [
            {"from": "system", "value": "System"},
            {"from": "human", "value": "Run task_test_T3_vendor_rating_check_001"},
            {"from": "gpt", "value": "Done"},
        ]
    ]
    bad_sg_path.write_text(json.dumps(bad_data), encoding="utf-8")
    config["dataset"]["train_sharegpt_path"] = str(bad_sg_path)

    harness = SFTExecutionHarness(config)
    with pytest.raises(LockedTestExportError, match="locked-test"):
        harness.run_sft_dry_run(output_dir=tmp_path / "fail_dry_run")


"""Unit and integration tests for baseline evaluation runner and provenance integrity (M1.P5)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from scripts.run_baseline_evaluation import (
    load_tasks_from_manifest,
    run_baseline_evaluation,
)
from src.data.schemas import TrajectoryRecord
from src.environment.task_generator import TaskSpec
from src.runtime.gateway import ModelGateway
from src.utils.compatibility import AuditKitRunResult


def test_locked_test_split_protection() -> None:
    """Verify that running baseline on sequestered 'test' split raises PermissionError."""
    with pytest.raises(PermissionError, match="Evaluation on the locked 'test' split is forbidden"):
        run_baseline_evaluation(split="test", allow_locked_test=False)


def test_load_tasks_from_val_manifest() -> None:
    """Verify that validation manifest loads valid TaskSpec objects with complete fields."""
    manifest_path = Path("configs/manifests/task_manifest_val.json")
    tasks = load_tasks_from_manifest(manifest_path)

    assert len(tasks) == 10
    for task in tasks:
        assert isinstance(task, TaskSpec)
        assert task.split == "val"
        assert len(task.expected_tools) > 0
        assert task.user_query is not None
        assert task.task_id.startswith("task_val_")


def test_offline_baseline_runner_mock(tmp_path: Path) -> None:
    """Verify baseline runner executes with deterministic mock policy and archives artifacts."""
    manifest_path = Path("configs/manifests/task_manifest_val.json")
    tasks = load_tasks_from_manifest(manifest_path)[:2]

    # Explicitly mock ModelGateway.generate to guarantee true offline isolation
    with patch.object(ModelGateway, "generate", return_value="Final Answer: $1000"):
        run_res = run_baseline_evaluation(
            split="val",
            model_id="cohere/test-mock-baseline",
            run_id="test_runner_mock_001",
            smoke=True,
            seed=42,
            output_base_dir=tmp_path,
        )

    assert isinstance(run_res, AuditKitRunResult)
    assert run_res.run_id == "test_runner_mock_001"
    assert "M1_task_success_rate" in run_res.metrics

    out_dir = tmp_path / "test_runner_mock_001"
    assert (out_dir / "run_result.json").exists()
    assert (out_dir / "metrics_summary.csv").exists()
    assert (out_dir / "eval_report.md").exists()
    assert (out_dir / "traces.jsonl").exists()
    assert (out_dir / "run_manifest.json").exists()

    # Validate JSON payload structure
    run_data = json.loads((out_dir / "run_result.json").read_text(encoding="utf-8"))
    assert run_data["run_id"] == "test_runner_mock_001"
    assert run_data["split"] == "val"


def test_zero_secrets_in_persisted_runs() -> None:
    """Verify that no sensitive API keys or tokens are stored in plain text in results/runs/."""
    runs_dir = Path("results/runs")
    if not runs_dir.exists():
        pytest.skip("No runs directory present yet.")

    forbidden_patterns = ["sk-or-v1-", "Bearer eyJ", "ghp_"]
    for file_path in runs_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in (".json", ".csv", ".md", ".jsonl"):
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for pattern in forbidden_patterns:
                assert pattern not in content, f"Secret pattern '{pattern}' detected in {file_path}"


def test_auditkit_run_result_load_from_disk() -> None:
    """Verify that AuditKitRunResult.load deserializes cleanly from disk."""
    baseline_dir = Path("results/runs/baseline_run_001")
    if not baseline_dir.exists():
        pytest.skip("baseline_run_001 directory not present.")

    loaded = AuditKitRunResult.load(baseline_dir)
    assert loaded.run_id == "baseline_run_001"
    assert loaded.model_id == "cohere/command-r7b-12-2024"
    assert loaded.split == "val"
    assert "M1_task_success_rate" in loaded.metrics
    assert loaded.metrics["M1_task_success_rate"].value == 0.1

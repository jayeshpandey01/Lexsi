"""Unit tests for M2.P5 candidate evaluation runner and paired bootstrap comparison."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.run_candidate_evaluation import (
    load_tasks_from_manifest,
    run_candidate_evaluation,
)
from src.environment.task_generator import TaskSpec
from src.utils.compatibility import AuditKitRunResult


def test_locked_test_split_protection() -> None:
    """Verify that running candidate evaluation on sequestered 'test' split raises PermissionError."""
    with pytest.raises(PermissionError, match="Evaluation on the locked 'test' split is strictly prohibited"):
        run_candidate_evaluation(split="test")


def test_load_tasks_from_val_manifest() -> None:
    """Verify that validation manifest loads 10 valid TaskSpec instances with complete fields."""
    manifest_path = Path("configs/manifests/task_manifest_val.json")
    tasks = load_tasks_from_manifest(manifest_path)

    assert len(tasks) == 10
    for task in tasks:
        assert isinstance(task, TaskSpec)
        assert task.split == "val"
        assert len(task.expected_tools) > 0
        assert task.user_query is not None
        assert task.task_id.startswith("task_val_")


def test_candidate_evaluator_smoke_execution(tmp_path: Path) -> None:
    """Verify candidate evaluator executes cleanly with adapter double policy and archives artifacts."""
    run_res = run_candidate_evaluation(
        split="val",
        model_id="CohereLabs/tiny-aya-global-sft-candidate",
        run_id="test_candidate_smoke_001",
        smoke=True,
        seed=42,
        candidate_type="adapter_double",
        baseline_run_dir=Path("results/runs/baseline_run_001"),
        output_base_dir=tmp_path,
    )

    assert isinstance(run_res, AuditKitRunResult)
    assert run_res.run_id == "test_candidate_smoke_001"
    assert "M1_task_success_rate" in run_res.metrics

    out_dir = tmp_path / "test_candidate_smoke_001"
    assert (out_dir / "run_result.json").exists()
    assert (out_dir / "metrics_summary.csv").exists()
    assert (out_dir / "eval_report.md").exists()
    assert (out_dir / "traces.jsonl").exists()
    assert (out_dir / "run_manifest.json").exists()

    manifest_data = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["run_id"] == "test_candidate_smoke_001"
    assert manifest_data["evaluator_mode"] == "local-adapter-1.0.0"
    assert "exploratory" in manifest_data["comparative_status"]


def test_zero_secrets_in_candidate_evaluation_artifacts(tmp_path: Path) -> None:
    """Verify that no sensitive credentials or tokens are saved in plain text in run artifacts."""
    run_candidate_evaluation(
        split="val",
        model_id="test-model",
        run_id="secret_check_run_001",
        smoke=True,
        seed=42,
        candidate_type="adapter_double",
        output_base_dir=tmp_path,
    )

    out_dir = tmp_path / "secret_check_run_001"
    forbidden_tokens = ["sk-or-v1-", "Bearer eyJ", "ghp_"]

    for file_path in out_dir.iterdir():
        if file_path.is_file():
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden_tokens:
                assert token not in content, f"Forbidden token pattern '{token}' detected in {file_path}"


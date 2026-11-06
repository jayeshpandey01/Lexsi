"""Unit tests for M4.P1 validation matrix runner and candidate eligibility enforcement."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.run_m4_p1_validation import (
    load_tasks_from_manifest,
    run_m4_p1_validation,
    compute_sha256,
)
from src.utils.compatibility import AuditKitRunResult


def test_locked_test_split_protection() -> None:
    """Verify that running M4.P1 validation on sequestered 'test' split raises PermissionError."""
    with pytest.raises(PermissionError, match="Evaluation on the locked 'test' split is strictly prohibited"):
        run_m4_p1_validation(split="test")


def test_m4_p1_validation_execution(tmp_path: Path) -> None:
    """Verify M4.P1 validation matrix runs cleanly and produces compliant AuditKIT artifacts."""
    run_id_sft = "test_m4_val_sft_001"
    run_id_base = "test_m4_val_base_blocked_001"

    sft_result, sft_manifest = run_m4_p1_validation(
        split="val",
        sft_run_id=run_id_sft,
        base_run_id=run_id_base,
        seed=42,
        output_base_dir=tmp_path,
    )

    # 1. AuditKIT Run Result assertions
    assert isinstance(sft_result, AuditKitRunResult)
    assert sft_result.run_id == run_id_sft
    assert sft_result.split == "val"
    assert sft_result.headline["M1_task_success_rate"] == 0.0
    assert sft_result.headline["M2_tool_call_accuracy"] == 0.0
    assert sft_result.failure_breakdown["EarlyTerminationFailure"] == 10

    # 2. File persistence assertions
    sft_dir = tmp_path / run_id_sft
    assert (sft_dir / "run_result.json").exists()
    assert (sft_dir / "metrics_summary.csv").exists()
    assert (sft_dir / "eval_report.md").exists()
    assert (sft_dir / "traces.jsonl").exists()
    assert (sft_dir / "run_manifest.json").exists()

    # 3. Base blocker manifest assertions
    base_dir = tmp_path / run_id_base
    assert (base_dir / "run_manifest.json").exists()
    with open(base_dir / "run_manifest.json") as f:
        base_data = json.load(f)
    assert base_data["status"] == "BLOCKED_NO_ROLLOUT"
    assert "COMPUTE_AND_DEPENDENCY_BLOCK" in base_data["blocker_details"]["blocker_category"]

    # 4. Candidate eligibility assertions
    assert sft_manifest["candidate_eligibility"]["eligible_for_promotion"] is False
    assert sft_manifest["candidate_eligibility"]["eligible_for_locked_test"] is False
    assert "DEFER_LOCKED_TEST" in sft_manifest["candidate_eligibility"]["action"]


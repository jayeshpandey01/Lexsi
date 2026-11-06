"""Unit tests for the empirical rollback smoke test runner (Gate G4 / Criterion 7)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.run_rollback_smoke_test import run_rollback_smoke_test


def test_rollback_smoke_split_guard() -> None:
    """Verify that attempting rollback smoke test on locked 'test' split raises PermissionError."""
    with pytest.raises(PermissionError, match="Rollback smoke testing on the locked 'test' split is strictly prohibited"):
        run_rollback_smoke_test(split="test")


def test_rollback_smoke_execution_and_idempotency(tmp_path: Path) -> None:
    """Verify that rollback smoke test executes cleanly, verifies baseline M1, and preserves config idempotency."""
    res = run_rollback_smoke_test(
        split="val",
        run_id="test_rollback_unit_001",
        seed=42,
        tolerance=0.15,
        output_base_dir=tmp_path,
    )

    assert res["status"] == "COMPLETED_ROLLBACK_TEST"
    assert res["rollback_verified"] is True
    assert res["configuration_provenance"]["configs_match"] is True
    assert res["baseline_comparison"]["is_within_tolerance"] is True
    assert abs(res["baseline_comparison"]["discrepancy"]) <= 0.15

    out_dir = tmp_path / "test_rollback_unit_001"
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "metrics_summary.csv").exists()
    assert (out_dir / "rollback_report.md").exists()

    with open(out_dir / "run_manifest.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["rollback_verified"] is True
        assert "M1_task_success_rate" in data["metrics"]

"""Unit tests for the pre-training base model format verification gate."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.verify_base_model_format import verify_base_model_format


def test_verify_base_model_format_mock_mode(tmp_path: Path) -> None:
    """Verify that format gate in mock mode detects all 5 delimiters and passes cleanly."""
    res = verify_base_model_format(
        model_id="test-mock-aya",
        run_id="test_format_gate_001",
        output_base_dir=tmp_path,
        mock_mode=True,
    )

    assert res["gate_passed"] is True
    assert res["actions_detected"] == 5
    assert res["total_probes"] == 5
    assert res["pass_rate"] == 1.0

    out_file = tmp_path / "test_format_gate_001" / "base_format_gate_report.json"
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["gate_passed"] is True
        assert len(data["probe_details"]) == 5
        assert data["probe_details"][0]["has_start_action_delimiter"] is True

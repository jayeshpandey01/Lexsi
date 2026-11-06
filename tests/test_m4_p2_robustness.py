"""Unit tests for M4.P2 robustness and safety regression review."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.run_m4_p2_robustness_review import (
    run_m4_p2_robustness_review,
    run_vector_1_prompt_injection,
    run_vector_2_malformed_output,
    run_vector_3_tool_permissions,
    run_vector_4_argument_schemas,
    run_vector_5_budget_and_stall,
    run_vector_6_error_recovery,
)


def test_vector_1_prompt_injection_sanitization() -> None:
    """Verify delimiter injection attacks are defused with zero bypasses."""
    results = run_vector_1_prompt_injection()
    assert len(results) == 8
    assert all(r.passed for r in results)
    assert all(not r.guard_bypassed for r in results)


def test_vector_2_malformed_output_interception() -> None:
    """Verify malformed delimiter tokens and invalid outputs raise structured errors."""
    results = run_vector_2_malformed_output()
    assert len(results) == 7
    assert all(r.passed for r in results)
    assert all(not r.guard_bypassed for r in results)


def test_vector_3_tool_permission_boundaries() -> None:
    """Verify unauthorized/disallowed tools are blocked fail-closed with zero side effects."""
    results = run_vector_3_tool_permissions()
    assert len(results) == 5
    assert all(r.passed for r in results)
    assert all(not r.guard_bypassed for r in results)
    # Observable assertion: tool_call_counts must not increment on denied calls
    assert all(r.details["side_effects_detected"] is False for r in results)
    assert all(
        sum(r.details["final_tool_call_counts"].values())
        == sum(r.details["initial_tool_call_counts"].values())
        for r in results
    )


def test_vector_4_argument_schema_validation() -> None:
    """Verify Pydantic argument schemas reject invalid types and extra parameters with zero tool calls."""
    results = run_vector_4_argument_schemas()
    assert len(results) == 6
    assert all(r.passed for r in results)
    assert all(not r.guard_bypassed for r in results)
    # Observable assertion: invalid schema calls never reach tool logic
    assert all(r.details["side_effects_detected"] is False for r in results)
    assert all(
        sum(r.details["final_tool_call_counts"].values())
        == sum(r.details["initial_tool_call_counts"].values())
        for r in results
    )


def test_vector_5_budgets_and_anti_looping() -> None:
    """Verify step budgets, timeouts, and looping patterns are enforced."""
    results = run_vector_5_budget_and_stall()
    assert len(results) == 4
    assert all(r.passed for r in results)
    assert all(not r.guard_bypassed for r in results)


def test_vector_6_error_recovery_diagnostics() -> None:
    """Verify unadapted errors and premature surrenders are correctly diagnosed."""
    results = run_vector_6_error_recovery()
    assert len(results) == 3
    assert all(r.passed for r in results)
    assert all(not r.guard_bypassed for r in results)


def test_m4_p2_robustness_review_artifact_generation(tmp_path: Path) -> None:
    """Verify M4.P2 runner creates all required immutable artifacts."""
    run_id = "test_robustness_run"
    manifest = run_m4_p2_robustness_review(output_base_dir=tmp_path, run_id=run_id)

    assert manifest["verdict"] == "PASS (Zero Guard Bypasses)"
    assert manifest["guard_bypasses"] == 0
    assert manifest["pass_rate"] == 1.0

    out_dir = tmp_path / run_id
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "robustness_summary.csv").exists()
    assert (out_dir / "probe_results.json").exists()
    assert (out_dir / "robustness_report.md").exists()


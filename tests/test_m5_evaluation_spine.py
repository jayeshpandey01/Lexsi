"""Unit tests for M5 artifact ingestion and AuditKIT evaluation spine (M5.P4).

Guarantees:
- Ingestion fails closed with LockedTestExportError if ZIP contains locked test artifacts.
- Rollout trace evaluation correctly computes M1-M7 metrics and 95% bootstrap intervals.
- Diagnostic failure taxonomy classifies early termination vs tool action successes.
- Comparative report generation accurately identifies capability floor compliance.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
import pytest

from scripts.evaluate_m5_candidates import (
    compare_candidates_and_generate_report,
    evaluate_rollout_traces,
)
from scripts.ingest_m5_kaggle_artifacts import ingest_artifacts, verify_zip_integrity
from src.data.schemas import LockedTestExportError


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "results" / "runs").mkdir(parents=True)
    return ws


def test_ingest_rejects_locked_test_zip(tmp_path: Path) -> None:
    """Verify that ingest halts with LockedTestExportError if ZIP contains locked test tasks."""
    bad_zip = tmp_path / "contaminated.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("configs/manifests/task_manifest_test.json", '{"split": "test"}')

    with pytest.raises(LockedTestExportError, match="locked-test"):
        verify_zip_integrity(bad_zip)


def test_ingest_nominal_zip_archive(temp_workspace: Path, tmp_path: Path) -> None:
    """Verify that valid ZIP archive extracts cleanly without error."""
    good_zip = tmp_path / "valid_artifacts.zip"
    with zipfile.ZipFile(good_zip, "w") as zf:
        zf.writestr(
            "results/runs/sft_aya_3b_v2_c2/val_rollout_traces.json",
            json.dumps([{"task_id": "task_val_T1_000", "generated_output": "<|START_RESPONSE|>Text", "oracle_answer": 100.0}]),
        )

    res = ingest_artifacts(good_zip, temp_workspace)
    assert res["status"] == "success"
    assert (temp_workspace / "results" / "runs" / "sft_aya_3b_v2_c2" / "val_rollout_traces.json").exists()


def test_evaluate_rollout_traces_metrics(tmp_path: Path) -> None:
    """Verify that evaluate_rollout_traces computes M1, M2, and failure taxonomy."""
    val_manifest_path = tmp_path / "val_manifest.json"
    val_manifest_path.write_text(
        json.dumps({
            "split_name": "val",
            "tasks": [
                {"task_id": "task_val_001", "user_query": "Q1", "oracle_answer": 500.0, "expected_tools": ["lookup_records"]},
                {"task_id": "task_val_002", "user_query": "Q2", "oracle_answer": 1000.0, "expected_tools": ["lookup_records"]},
            ]
        }),
        encoding="utf-8"
    )

    traces_path = tmp_path / "val_rollout_traces.json"
    traces_path.write_text(
        json.dumps([
            # Task 1: Successful tool call with correct numeric answer
            {
                "task_id": "task_val_001",
                "generated_output": '<|START_ACTION|>{"name": "lookup_records", "arguments": {"vendor": "VND-100"}}<|END_ACTION|> Total: 500.0',
                "oracle_answer": 500.0,
            },
            # Task 2: Early termination failure
            {
                "task_id": "task_val_002",
                "generated_output": '<|START_RESPONSE|>Here is an explanation with no tool calls.',
                "oracle_answer": 1000.0,
            },
        ]),
        encoding="utf-8"
    )

    res = evaluate_rollout_traces(traces_path, val_manifest_path, "test_eval_run")
    assert res["sample_count"] == 2
    assert res["m1_success_rate"] == 0.50
    assert res["m2_tool_selection"] == 0.50
    assert res["m3_schema_validity"] == 1.00
    assert res["failure_taxonomy"]["EarlyTerminationFailure"] == 1
    assert res["failure_taxonomy"]["NominalSuccess"] == 1


def test_comparative_report_floor_checks(tmp_path: Path) -> None:
    """Verify comparative report identifies passing vs failing capability floors."""
    eval_results = {
        "val_base_aya_3b_matched": {
            "m1_success_rate": 0.10,
            "m1_ci_95": [0.0, 0.30],
            "m2_tool_selection": 0.10,
            "m3_schema_validity": 1.0,
            "m7_loop_rate": 0.0,
            "failure_taxonomy": {"EarlyTerminationFailure": 9, "NominalSuccess": 1},
            "success_flags": [True] + [False] * 9,
        },
        "sft_aya_3b_v2_c3": {
            "m1_success_rate": 0.60,
            "m1_ci_95": [0.30, 0.90],
            "m2_tool_selection": 0.70,
            "m3_schema_validity": 1.0,
            "m7_loop_rate": 0.0,
            "failure_taxonomy": {"NominalSuccess": 6, "EarlyTerminationFailure": 4},
            "success_flags": [True] * 6 + [False] * 4,
        },
    }

    report_path = tmp_path / "report.md"
    content = compare_candidates_and_generate_report(eval_results, report_path)

    assert "PASSED FLOORS" in content
    assert "BELOW FLOOR" in content
    assert "val_base_aya_3b_matched" in content
    assert "sft_aya_3b_v2_c3" in content
    assert report_path.exists()

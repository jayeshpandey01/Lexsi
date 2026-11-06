"""Unit tests for M2.P2 SFT dataset builder, leakage auditor, and format serializers."""

from __future__ import annotations

import hashlib
import json
import pytest
from pathlib import Path

from src.data.schemas import LockedTestExportError, TrajectoryRecord
from src.data.sft_builder import (
    audit_sft_leakage,
    build_sft_dataset,
    compute_jaccard_similarity,
    generate_oracle_trajectory,
)
from src.environment.task_generator import TaskSpec


def test_oracle_trajectory_generation() -> None:
    """Verify that oracle teacher policy solves T1 and T2 training tasks deterministically."""
    # Test T1 task
    t1_spec = TaskSpec(
        task_id="task_train_T1_inventory_value_000",
        split="train",
        template_id="T1_inventory_value",
        user_query="Calculate inventory valuation for 'Hardware'",
        expected_tools=["lookup_records", "compute_aggregate"],
        oracle_answer=113202.7,
        target_entities={"category": "Hardware"},
    )
    t1_traj = generate_oracle_trajectory(t1_spec, seed=42)
    assert t1_traj.is_success is True
    assert t1_traj.total_steps == 3
    assert len(t1_traj.steps) == 3
    assert t1_traj.steps[0].action.tool_name == "lookup_records"  # type: ignore[union-attr]
    assert t1_traj.steps[1].action.tool_name == "compute_aggregate"  # type: ignore[union-attr]

    # Test T2 task
    t2_spec = TaskSpec(
        task_id="task_train_T2_vendor_approved_spend_001",
        split="train",
        template_id="T2_vendor_approved_spend",
        user_query="Determine approved spend for 'VND-100'",
        expected_tools=["lookup_records", "compute_aggregate"],
        oracle_answer=0.0,
        target_entities={"vendor_id": "VND-100"},
    )
    t2_traj = generate_oracle_trajectory(t2_spec, seed=42)
    assert t2_traj.is_success is True
    assert t2_traj.total_steps == 3
    assert t2_traj.steps[0].action.tool_name == "lookup_records"  # type: ignore[union-attr]
    assert t2_traj.steps[1].action.tool_name == "compute_aggregate"  # type: ignore[union-attr]


def test_locked_test_protection_in_builder() -> None:
    """Verify that generating training trajectory from locked-test task raises LockedTestExportError."""
    locked_task = TaskSpec(
        task_id="task_test_T3_vendor_rating_check_001",
        split="test",
        template_id="T3_vendor_rating_check",
        user_query="Audit orders for 'VND-108'",
        expected_tools=["lookup_records", "compute_aggregate"],
        oracle_answer=50.0,
        target_entities={"vendor_id": "VND-108"},
    )
    with pytest.raises(LockedTestExportError, match="locked test task"):
        generate_oracle_trajectory(locked_task)


def test_sft_dataset_leakage_audit() -> None:
    """Verify leakage audit verifies zero overlap with val and locked test manifests."""
    train_manifest = Path("configs/manifests/task_manifest_train.json")
    val_manifest = Path("configs/manifests/task_manifest_val.json")
    test_manifest = Path("configs/manifests/task_manifest_test.json")

    tasks = [TaskSpec.model_validate(t) for t in json.loads(train_manifest.read_text())["tasks"][:5]]
    trajs = [generate_oracle_trajectory(t) for t in tasks]

    report = audit_sft_leakage(trajs, val_manifest, test_manifest)
    assert report.is_isolation_verified is True
    assert report.task_id_leakage is False
    assert report.template_leakage_with_test is False
    assert report.entity_leakage_with_val is False
    assert report.entity_leakage_with_test is False
    assert report.query_text_exact_match_leakage is False
    assert report.max_jaccard_similarity_test < 0.50
    assert len(report.violations) == 0


def test_sft_dataset_persisted_artifacts() -> None:
    """Verify persisted SFT training artifacts exist, validate against schemas, and match manifest."""
    sft_dir = Path("data/sft")
    assert sft_dir.exists()

    jsonl_file = sft_dir / "sft_train_trajectories.jsonl"
    sharegpt_file = sft_dir / "sft_train_sharegpt.json"
    eventlog_file = sft_dir / "sft_train_eventlog.json"
    manifest_file = sft_dir / "sft_train_dataset_manifest.json"
    report_file = sft_dir / "sft_data_quality_report.md"

    assert jsonl_file.exists()
    assert sharegpt_file.exists()
    assert eventlog_file.exists()
    assert manifest_file.exists()
    assert report_file.exists()

    # Verify JSONL lines
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 20
    for line in lines:
        record = TrajectoryRecord.model_validate_json(line)
        assert record.is_success is True
        assert "task_train_" in record.task_id
        assert record.total_steps == 3

    # Verify ShareGPT records
    sgpt = json.loads(sharegpt_file.read_text(encoding="utf-8"))
    assert len(sgpt) == 20
    for convo in sgpt:
        assert len(convo) >= 5
        assert convo[0]["from"] == "system"
        assert convo[1]["from"] == "human"
        assert "<|START_ACTION|>" in convo[2]["value"]

    # Verify EventLog records
    eventlog = json.loads(eventlog_file.read_text(encoding="utf-8"))
    assert len(eventlog) == 20
    for el in eventlog:
        assert "ep_task_train_" in el["episode_id"]
        assert len(el["events"]) == 6

    # Verify Manifest SHA-256 Digest ties to JSONL
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(jsonl_file.read_bytes()).hexdigest()
    assert manifest["dataset_sha256"] == expected_hash
    assert manifest["stats"]["accepted_trajectories_count"] == 20
    assert manifest["stats"]["rejected_count"] == 0
    assert manifest["stats"]["unique_trajectories_count"] == 7


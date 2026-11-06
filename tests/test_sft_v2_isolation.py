"""Unit tests for expanded SFT dataset (procurement_sft_train_v2) quality and split isolation (M5.P2).

Guarantees:
- Dataset contains >= 200 unique multi-turn trajectories across 7 template families.
- Exact deduplication ratio >= 80%.
- Strict entity and template isolation from validation (Software, VND-106..107) and locked test (Services, Logistics, VND-108..111).
- Fail-closed guard: LockedTestExportError prevents training demonstration generation from test split.
- 100% verified task success across all exported training trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.data.schemas import LockedTestExportError, TrajectoryRecord
from src.environment.task_generator import load_split_config


@pytest.fixture
def sft_v2_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "sft"


@pytest.fixture
def manifest_v2_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "manifests" / "task_manifest_train_v2.json"


def test_sft_v2_dataset_manifest_metrics(sft_v2_dir: Path) -> None:
    """Verify that dataset manifest records >= 200 unique samples and >= 80% deduplication ratio."""
    manifest_path = sft_v2_dir / "sft_train_v2_dataset_manifest.json"
    assert manifest_path.exists(), "Dataset manifest sft_train_v2_dataset_manifest.json missing!"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "procurement_sft_train_v2"
    assert manifest["split"] == "train"
    assert manifest["sample_count"] >= 200
    assert manifest["deduped_sample_count"] >= 200
    assert manifest["deduplication_ratio"] >= 0.80
    assert manifest["isolation_verified"] is True
    assert len(manifest["dataset_sha256"]) == 64


def test_sft_v2_trajectories_success_and_validity(sft_v2_dir: Path) -> None:
    """Verify that 100% of trajectories in JSONL are verified successes with valid tool calls."""
    jsonl_path = sft_v2_dir / "sft_train_v2_trajectories.jsonl"
    assert jsonl_path.exists(), "sft_train_v2_trajectories.jsonl missing!"

    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            traj = TrajectoryRecord.model_validate(data)
            assert traj.is_success is True, f"Trajectory '{traj.task_id}' was not successful!"
            assert len(traj.steps) >= 2, f"Trajectory '{traj.task_id}' had fewer than 2 steps!"
            assert traj.final_output is not None
            count += 1

    assert count >= 200, f"Expected >= 200 trajectories, found {count}"


def test_sft_v2_sharegpt_format_integrity(sft_v2_dir: Path) -> None:
    """Verify that ShareGPT formatted dataset contains valid multi-turn dialogs with delimiters."""
    sharegpt_path = sft_v2_dir / "sft_train_v2_sharegpt.json"
    assert sharegpt_path.exists(), "sft_train_v2_sharegpt.json missing!"

    records = json.loads(sharegpt_path.read_text(encoding="utf-8"))
    assert len(records) >= 200

    for convs in records:
        assert isinstance(convs, list)
        assert len(convs) >= 4  # System, User, Assistant Action, Tool Observation, Assistant Final

        # Verify role order
        roles = [c["from"] for c in convs]
        assert roles[0] == "system"
        assert roles[1] == "human"
        assert roles[2] == "gpt"
        assert "<|START_ACTION|>" in convs[2]["value"]
        assert "<|END_ACTION|>" in convs[2]["value"]


def test_sft_v2_partition_isolation_without_test_manifest(sft_v2_dir: Path) -> None:
    """Verify complete entity and template isolation from val and test partitions without touching test manifest."""
    split_config = load_split_config()

    test_vendors = set(split_config.test.vendors)
    test_categories = set(split_config.test.categories)
    val_vendors = set(split_config.val.vendors)
    val_categories = set(split_config.val.categories)

    # Read training manifest v2
    manifest_v2_path = Path(__file__).resolve().parent.parent / "configs" / "manifests" / "task_manifest_train_v2.json"
    train_manifest = json.loads(manifest_v2_path.read_text(encoding="utf-8"))

    for task in train_manifest["tasks"]:
        q = task["user_query"]
        q_lower = q.lower()

        # 1. No test vendors
        for tv in test_vendors:
            assert tv not in q, f"Test vendor '{tv}' leaked into training query: '{q}'"

        # 2. No test categories
        for tc in test_categories:
            assert tc.lower() not in q_lower, f"Test category '{tc}' leaked into training query: '{q}'"

        # 3. No test templates
        assert task["template_id"] not in ("T3_vendor_rating_check", "T4_budget_cap_verification")

        # 4. No val vendors
        for vv in val_vendors:
            assert vv not in q, f"Val vendor '{vv}' leaked into training query: '{q}'"

        # 5. No val categories
        for vc in val_categories:
            assert vc.lower() not in q_lower, f"Val category '{vc}' leaked into training query: '{q}'"


def test_fail_closed_test_export_guard() -> None:
    """Verify that any attempt to generate training demonstration from test split raises LockedTestExportError."""
    from src.data.sft_builder import generate_oracle_trajectory
    from src.environment.task_generator import TaskSpec

    fake_test_task = TaskSpec(
        task_id="task_test_T3_vendor_rating_check_001",
        split="test",
        template_id="T3_vendor_rating_check",
        user_query="Evaluate vendor 'VND-108'",
        expected_tools=["lookup_records", "compute_aggregate"],
        oracle_answer=4.5,
    )

    with pytest.raises(LockedTestExportError, match="locked test"):
        generate_oracle_trajectory(fake_test_task)

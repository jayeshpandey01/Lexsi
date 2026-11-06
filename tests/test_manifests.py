"""Unit tests for machine-readable dataset manifests and locked-test export safeguards."""

from pathlib import Path
import pytest

from src.data.manifests import (
    DatasetManifest,
    generate_task_manifests,
    save_manifests,
)
from src.data.schemas import (
    LockedTestExportError,
    StepRecord,
    ToolCallAction,
    TrajectoryRecord,
    export_training_trajectories,
)
from src.environment.task_generator import generate_dataset_splits, load_split_config


def test_manifest_generation_and_provenance() -> None:
    """Verify DatasetManifest generation incorporates all splits and provenance hashes."""
    cfg = load_split_config()
    splits = generate_dataset_splits(seed=42, train_count=4, val_count=2, test_count=2, config=cfg)
    manifest = generate_task_manifests(splits, config=cfg)

    assert manifest.manifest_version == "1.0.0"
    assert manifest.dataset_name == "procurement_inventory_benchmark"
    assert manifest.total_tasks == 8
    assert "train" in manifest.splits
    assert "val" in manifest.splits
    assert "test" in manifest.splits

    # Verify locked status
    assert manifest.splits["train"].is_locked is False
    assert manifest.splits["val"].is_locked is False
    assert manifest.splits["test"].is_locked is True

    # Verify content hashes are non-empty and 64 hex characters (SHA-256)
    assert len(manifest.config_hash) == 64
    for split_name, s_man in manifest.splits.items():
        assert len(s_man.split_content_hash) == 64
        assert s_man.count == len(s_man.tasks)
        for t in s_man.tasks:
            assert len(t.provenance_hash) == 64
            assert t.split == split_name


def test_manifest_serialization_and_determinism(tmp_path: Path) -> None:
    """Verify manifests are serialized to disk in valid JSON format and hashes are stable."""
    splits = generate_dataset_splits(seed=42, train_count=4, val_count=2, test_count=2)
    manifest = generate_task_manifests(splits)

    saved_paths = save_manifests(manifest, output_dir=tmp_path)
    assert len(saved_paths) == 4  # dataset_manifest.json + 3 split manifests

    # Verify unified manifest reloads cleanly
    dataset_json_path = tmp_path / "dataset_manifest.json"
    assert dataset_json_path.exists()
    content = dataset_json_path.read_text(encoding="utf-8")
    reloaded = DatasetManifest.model_validate_json(content)
    assert reloaded.dataset_name == manifest.dataset_name
    assert reloaded.config_hash == manifest.config_hash
    assert reloaded.total_tasks == manifest.total_tasks

    # Verify determinism across second run
    manifest2 = generate_task_manifests(splits)
    assert manifest.config_hash == manifest2.config_hash
    assert manifest.splits["train"].split_content_hash == manifest2.splits["train"].split_content_hash
    assert manifest.splits["test"].split_content_hash == manifest2.splits["test"].split_content_hash


def test_locked_test_export_safeguard() -> None:
    """Verify fail-closed error when attempting to project or export locked test data for training."""
    # 1. Permitted train trajectory succeeds
    train_traj = TrajectoryRecord(
        task_id="task_train_T1_inventory_value_000",
        seed=42,
        system_prompt="Train prompt",
        user_query="Train query",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors"}),
                observation=[{"vendor_id": "VND-100"}],
            )
        ],
        final_output="Done",
        is_success=True,
    )
    sgpt_train = train_traj.to_sharegpt_format()
    assert len(sgpt_train) > 0

    # 2. Locked test trajectory fails closed
    test_traj = TrajectoryRecord(
        task_id="task_test_T3_vendor_rating_check_000",
        seed=42,
        system_prompt="Test prompt",
        user_query="Test query",
        metadata={"is_locked": True},
        final_output="Done",
        is_success=True,
    )
    with pytest.raises(LockedTestExportError, match="Locked test split is permanently sequestered"):
        test_traj.to_sharegpt_format()

    # 3. Batch export fails closed if test trajectory is present
    with pytest.raises(LockedTestExportError):
        export_training_trajectories([train_traj, test_traj])

    # Clean train batch succeeds
    clean_batch = export_training_trajectories([train_traj])
    assert len(clean_batch) == 1


"""Unit tests for deterministic task generator and split isolation verification."""

from src.environment.task_generator import (
    generate_dataset_splits,
    generate_task_instance,
    verify_split_isolation,
)


def test_generator_determinism() -> None:
    """Verify identical tasks produced from same seed."""
    task1 = generate_task_instance("train", 0, seed=42)
    task2 = generate_task_instance("train", 0, seed=42)
    assert task1.task_id == task2.task_id
    assert task1.oracle_answer == task2.oracle_answer
    assert task1.provenance_hash == task2.provenance_hash


def test_split_isolation_no_leakage() -> None:
    """Verify strict lack of contamination between train, val, and locked test sets."""
    splits = generate_dataset_splits(seed=42, train_count=20, val_count=10, test_count=10)
    is_isolated, issues = verify_split_isolation(splits)
    assert is_isolated, f"Split isolation failed: {issues}"
    assert len(issues) == 0


def test_task_fields_and_tools() -> None:
    """Verify task attributes across all generated splits."""
    splits = generate_dataset_splits(seed=42, train_count=6, val_count=4, test_count=4)
    for split_name, tasks in splits.items():
        assert len(tasks) > 0
        for task in tasks:
            assert task.split == split_name
            assert len(task.expected_tools) >= 2
            assert len(task.user_query) > 10
            assert isinstance(task.oracle_answer, float)
            assert task.provenance_hash is not None

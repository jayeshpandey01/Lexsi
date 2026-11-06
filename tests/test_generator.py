"""Unit tests for deterministic task generator, split configuration, and pairwise isolation audit."""

from pathlib import Path
import pytest
import yaml

from src.environment.task_generator import (
    SplitConfigError,
    TaskSpec,
    audit_split_isolation,
    generate_dataset_splits,
    generate_task_instance,
    load_split_config,
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
    """Verify strict lack of contamination across all pairwise split combinations."""
    splits = generate_dataset_splits(seed=42, train_count=20, val_count=10, test_count=10)
    report = audit_split_isolation(splits)
    assert report.is_valid, f"Split isolation audit failed: {report.issues}"
    assert len(report.issues) == 0
    assert report.checked_pairs == ["train <-> val", "train <-> test", "val <-> test"]


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


def test_entity_coverage_no_unreachable_entities() -> None:
    """Verify all declared vendors and categories in each split are visited."""
    splits = generate_dataset_splits(seed=42, train_count=20, val_count=10, test_count=10)

    # 1. Train vendors (all 6 must be queried)
    train_vendors = {t.target_entities["vendor_id"] for t in splits["train"] if "vendor_id" in t.target_entities}
    expected_train_vendors = {f"VND-{100 + i}" for i in range(6)}
    assert train_vendors == expected_train_vendors

    # 2. Val vendors (both must be queried)
    val_vendors = {t.target_entities["vendor_id"] for t in splits["val"] if "vendor_id" in t.target_entities}
    assert val_vendors == {"VND-106", "VND-107"}

    # 3. Test vendors (all 4 must be queried)
    test_vendors = {t.target_entities["vendor_id"] for t in splits["test"] if "vendor_id" in t.target_entities}
    expected_test_vendors = {f"VND-{108 + i}" for i in range(4)}
    assert test_vendors == expected_test_vendors

    # 4. Test categories (both Services and Logistics must be queried)
    test_cats = {t.target_entities["category"] for t in splits["test"] if "category" in t.target_entities}
    assert test_cats == {"Services", "Logistics"}


def test_split_config_validation(tmp_path: Path) -> None:
    """Verify config loader enforces schema constraints and fails on invalid partitions."""
    cfg = load_split_config()
    assert cfg.metadata.domain == "Procurement & Inventory Analytics"
    assert "not separately asserted" in cfg.metadata.license

    # 1. Unknown template
    bad_yaml = {
        "metadata": cfg.metadata.model_dump(),
        "partition_policy": {
            "isolation_level": "test",
            "train": {"templates": ["UnknownTemplate"], "categories": ["Hardware"], "vendors": ["VND-100"], "count": 5},
            "val": {"templates": ["T1_inventory_value"], "categories": ["Software"], "vendors": ["VND-106"], "count": 2},
            "test": {"templates": ["T3_vendor_rating_check"], "categories": ["Services"], "vendors": ["VND-108"], "count": 2},
        },
    }
    p = tmp_path / "bad_tmpl.yaml"
    p.write_text(yaml.dump(bad_yaml), encoding="utf-8")
    with pytest.raises(SplitConfigError, match="Unsupported template"):
        load_split_config(p)

    # 2. Unknown category
    bad_yaml["partition_policy"]["train"]["templates"] = ["T1_inventory_value"]
    bad_yaml["partition_policy"]["train"]["categories"] = ["InvalidCategory"]
    p_cat = tmp_path / "bad_cat.yaml"
    p_cat.write_text(yaml.dump(bad_yaml), encoding="utf-8")
    with pytest.raises(SplitConfigError, match="Unknown category"):
        load_split_config(p_cat)

    # 3. Unknown vendor
    bad_yaml["partition_policy"]["train"]["categories"] = ["Hardware"]
    bad_yaml["partition_policy"]["train"]["vendors"] = ["VND-999"]
    p_vnd = tmp_path / "bad_vnd.yaml"
    p_vnd.write_text(yaml.dump(bad_yaml), encoding="utf-8")
    with pytest.raises(SplitConfigError, match="Unknown vendor"):
        load_split_config(p_vnd)


def test_isolation_audit_negative_fixtures() -> None:
    """Verify isolation audit catches deliberate leakage across dimensions and pairs."""
    splits = generate_dataset_splits(seed=42, train_count=4, val_count=2, test_count=2)

    # 1. Deliberate task ID collision
    tampered_splits = {
        "train": list(splits["train"]),
        "val": list(splits["val"]),
        "test": list(splits["test"]),
    }
    tampered_splits["val"][0] = tampered_splits["val"][0].model_copy(
        update={"task_id": tampered_splits["train"][0].task_id}
    )
    rep = audit_split_isolation(tampered_splits)
    assert not rep.is_valid
    assert any(i.dimension == "task_id_uniqueness" and "train <-> val" in i.split_pair for i in rep.issues)

    # 2. Deliberate vendor leakage between train and val
    tampered_vendor = {
        "train": list(splits["train"]),
        "val": list(splits["val"]),
        "test": list(splits["test"]),
    }
    tampered_vendor["val"][0] = tampered_vendor["val"][0].model_copy(
        update={"target_entities": {"vendor_id": "VND-100"}}
    )
    rep_v = audit_split_isolation(tampered_vendor)
    assert not rep_v.is_valid
    assert any(i.dimension == "vendor_entity_disjointness" for i in rep_v.issues)

    # 3. Deliberate category leakage between val and test
    tampered_cat = {
        "train": list(splits["train"]),
        "val": list(splits["val"]),
        "test": list(splits["test"]),
    }
    tampered_cat["test"][0] = tampered_cat["test"][0].model_copy(
        update={"target_entities": {"category": "Software"}}
    )
    rep_c = audit_split_isolation(tampered_cat)
    assert not rep_c.is_valid
    assert any(i.dimension == "category_entity_disjointness" for i in rep_c.issues)

    # 4. Deliberate template leakage in test
    tampered_tmpl = {
        "train": list(splits["train"]),
        "val": list(splits["val"]),
        "test": list(splits["test"]),
    }
    tampered_tmpl["test"][0] = tampered_tmpl["test"][0].model_copy(
        update={"template_id": "T1_inventory_value"}
    )
    rep_t = audit_split_isolation(tampered_tmpl)
    assert not rep_t.is_valid
    assert any(i.dimension == "template_disjointness" for i in rep_t.issues)

    # 5. Deliberate query text collision
    tampered_query = {
        "train": list(splits["train"]),
        "val": list(splits["val"]),
        "test": list(splits["test"]),
    }
    tampered_query["val"][0] = tampered_query["val"][0].model_copy(
        update={"user_query": tampered_query["train"][0].user_query}
    )
    rep_q = audit_split_isolation(tampered_query)
    assert not rep_q.is_valid
    assert any(i.dimension == "query_text_disjointness" for i in rep_q.issues)


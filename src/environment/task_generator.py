"""Deterministic task generator with entity-level split isolation."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from src.environment.mock_data import get_mock_database
from src.environment.tools import (
    execute_compute_aggregate,
    execute_lookup_records,
    execute_verify_constraint,
)


class TaskSpec(BaseModel):
    """Specification of an evaluation or training task instance."""

    task_id: str
    split: Literal["train", "val", "test"]
    template_id: str
    user_query: str
    expected_tools: List[str]
    oracle_answer: float
    oracle_verdict: Optional[bool] = None
    target_entities: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if not self.provenance_hash:
            payload = f"{self.task_id}_{self.split}_{self.oracle_answer}_{self.oracle_verdict}"
            self.provenance_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Disjoint split partitioning:
# Train: Categories ["Hardware", "Software"], Vendors VND-100 to VND-105
# Val:   Categories ["Hardware", "Software"], Vendors VND-106 to VND-107
# Test:  Categories ["Services", "Logistics"], Vendors VND-108 to VND-111 (Locked Holdout)
SPLIT_PARTITIONS = {
    "train": {
        "categories": ["Hardware", "Software"],
        "vendor_ids": [f"VND-{100 + i}" for i in range(6)],
        "templates": ["T1_inventory_value", "T2_vendor_approved_spend"],
    },
    "val": {
        "categories": ["Hardware", "Software"],
        "vendor_ids": [f"VND-{106 + i}" for i in range(2)],
        "templates": ["T1_inventory_value", "T2_vendor_approved_spend"],
    },
    "test": {
        "categories": ["Services", "Logistics"],
        "vendor_ids": [f"VND-{108 + i}" for i in range(4)],
        "templates": ["T3_vendor_rating_check", "T4_budget_cap_verification"],
    },
}


def generate_task_instance(
    split: Literal["train", "val", "test"],
    index: int,
    seed: int = 42,
) -> TaskSpec:
    """Generate a single deterministic task instance for the specified split."""
    db = get_mock_database(seed)
    rng = random.Random(seed + index + (1000 if split == "val" else 2000 if split == "test" else 0))

    partition = SPLIT_PARTITIONS[split]
    template_id = partition["templates"][index % len(partition["templates"])]
    task_id = f"task_{split}_{template_id}_{index:03d}"

    if template_id == "T1_inventory_value":
        category = partition["categories"][index % len(partition["categories"])]
        items = execute_lookup_records(db, {"table": "inventory_items", "filters": {"category": category}})
        # Values = stock_level * unit_cost
        values = [round(item["stock_level"] * item["unit_cost"], 4) for item in items]
        oracle_ans = execute_compute_aggregate({"values": values, "operation": "sum"}) if values else 0.0

        query = (
            f"Calculate the total inventory valuation for all items in the '{category}' category "
            f"by querying inventory_items and summing unit_cost * stock_level."
        )
        return TaskSpec(
            task_id=task_id,
            split=split,
            template_id=template_id,
            user_query=query,
            expected_tools=["lookup_records", "compute_aggregate"],
            oracle_answer=oracle_ans,
            target_entities={"category": category},
        )

    if template_id == "T2_vendor_approved_spend":
        vendor_id = partition["vendor_ids"][index % len(partition["vendor_ids"])]
        orders = execute_lookup_records(
            db,
            {"table": "purchase_orders", "filters": {"vendor_id": vendor_id, "order_status": "Approved"}},
        )
        values = [round(order["quantity"] * order["unit_price"], 4) for order in orders]
        oracle_ans = execute_compute_aggregate({"values": values, "operation": "sum"}) if values else 0.0

        query = (
            f"Determine the total Approved purchase order expenditure for vendor '{vendor_id}'. "
            f"Query purchase_orders and sum quantity * unit_price."
        )
        return TaskSpec(
            task_id=task_id,
            split=split,
            template_id=template_id,
            user_query=query,
            expected_tools=["lookup_records", "compute_aggregate"],
            oracle_answer=oracle_ans,
            target_entities={"vendor_id": vendor_id},
        )

    if template_id == "T3_vendor_rating_check":
        # Multi-hop test task: Find vendor, check rating, compute average unit price of their orders
        vendor_id = partition["vendor_ids"][index % len(partition["vendor_ids"])]
        orders = execute_lookup_records(db, {"table": "purchase_orders", "filters": {"vendor_id": vendor_id}})
        prices = [order["unit_price"] for order in orders]
        avg_price = execute_compute_aggregate({"values": prices, "operation": "mean"}) if prices else 0.0

        query = (
            f"Evaluate vendor '{vendor_id}': Retrieve all purchase orders for this vendor "
            f"and calculate the average unit purchase price across all placed orders."
        )
        return TaskSpec(
            task_id=task_id,
            split=split,
            template_id=template_id,
            user_query=query,
            expected_tools=["lookup_records", "compute_aggregate"],
            oracle_answer=avg_price,
            target_entities={"vendor_id": vendor_id},
        )

    if template_id == "T4_budget_cap_verification":
        # Multi-step 4-tool task: Query budget cap, query actual orders, aggregate spend, verify constraint
        category = partition["categories"][index % len(partition["categories"])]
        caps = execute_lookup_records(db, {"table": "budget_caps", "filters": {"category": category}})
        budget_cap = caps[0]["allocated_budget"] if caps else 50000.0

        # Look up inventory items for this category to get SKUs
        cat_items = execute_lookup_records(db, {"table": "inventory_items", "filters": {"category": category}})
        skus = {item["item_sku"] for item in cat_items}

        all_orders = execute_lookup_records(db, {"table": "purchase_orders", "filters": {}})
        cat_orders = [o for o in all_orders if o["item_sku"] in skus and o["order_status"] in ("Approved", "Fulfilled")]
        spend_values = [round(o["quantity"] * o["unit_price"], 4) for o in cat_orders]
        total_spend = execute_compute_aggregate({"values": spend_values, "operation": "sum"}) if spend_values else 0.0

        verdict = execute_verify_constraint(
            {"actual": total_spend, "expected": budget_cap, "comparison": "le"}
        )

        query = (
            f"Audit '{category}' department spend: Verify whether the total expenditure on Approved/Fulfilled "
            f"purchase orders is within the 2026-Q1 budget cap ({budget_cap}). Return final spend and compliance verdict."
        )
        return TaskSpec(
            task_id=task_id,
            split=split,
            template_id=template_id,
            user_query=query,
            expected_tools=["lookup_records", "compute_aggregate", "verify_constraint"],
            oracle_answer=total_spend,
            oracle_verdict=verdict,
            target_entities={"category": category, "budget_cap": budget_cap},
        )

    raise ValueError(f"Unknown template {template_id}")


def generate_dataset_splits(
    seed: int = 42,
    train_count: int = 20,
    val_count: int = 10,
    test_count: int = 10,
) -> Dict[str, List[TaskSpec]]:
    """Generate the full benchmark split suite."""
    train_tasks = [generate_task_instance("train", i, seed) for i in range(train_count)]
    val_tasks = [generate_task_instance("val", i, seed) for i in range(val_count)]
    test_tasks = [generate_task_instance("test", i, seed) for i in range(test_count)]

    return {
        "train": train_tasks,
        "val": val_tasks,
        "test": test_tasks,
    }


def verify_split_isolation(splits: Dict[str, List[TaskSpec]]) -> Tuple[bool, List[str]]:
    """Verify strict lack of contamination between train, val, and locked test sets."""
    issues: List[str] = []

    # 1. Check template disjointness for locked test
    train_templates = {t.template_id for t in splits["train"]}
    test_templates = {t.template_id for t in splits["test"]}
    template_overlap = train_templates.intersection(test_templates)
    if template_overlap:
        issues.append(f"Template leakage: Test shares templates {template_overlap} with Train")

    # 2. Check vendor ID disjointness
    train_vendors = {t.target_entities["vendor_id"] for t in splits["train"] if "vendor_id" in t.target_entities}
    test_vendors = {t.target_entities["vendor_id"] for t in splits["test"] if "vendor_id" in t.target_entities}
    vendor_overlap = train_vendors.intersection(test_vendors)
    if vendor_overlap:
        issues.append(f"Entity leakage: Test shares vendors {vendor_overlap} with Train")

    # 3. Check category disjointness
    train_cats = {t.target_entities["category"] for t in splits["train"] if "category" in t.target_entities}
    test_cats = {t.target_entities["category"] for t in splits["test"] if "category" in t.target_entities}
    cat_overlap = train_cats.intersection(test_cats)
    if cat_overlap:
        issues.append(f"Category leakage: Test shares categories {cat_overlap} with Train")

    return len(issues) == 0, issues

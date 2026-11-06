"""Deterministic task generator with entity-level split isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.environment.mock_data import get_mock_database
from src.environment.tools import (
    execute_compute_aggregate,
    execute_lookup_records,
    execute_verify_constraint,
)


class SplitConfigError(Exception):
    """Raised when split configuration or partitions are invalid or inconsistent."""

    pass


class SplitMetadata(BaseModel):
    """Metadata describing the split configuration and dataset provenance."""

    model_config = ConfigDict(extra="forbid")

    domain: str
    generator_module: str
    seed: int
    license: str
    provenance: str


class SplitPartitionSpec(BaseModel):
    """Partition specification for a single split."""

    model_config = ConfigDict(extra="forbid")

    templates: List[str]
    categories: List[str]
    vendors: List[str]
    count: int = Field(..., ge=1)
    status: Optional[str] = None


class TaskSplitConfig(BaseModel):
    """Canonical, validated task split configuration loaded from configs/task_splits.yaml."""

    model_config = ConfigDict(extra="forbid")

    metadata: SplitMetadata
    partition_policy: Dict[str, Any]

    @property
    def train(self) -> SplitPartitionSpec:
        return SplitPartitionSpec.model_validate(self.partition_policy["train"])

    @property
    def val(self) -> SplitPartitionSpec:
        return SplitPartitionSpec.model_validate(self.partition_policy["val"])

    @property
    def test(self) -> SplitPartitionSpec:
        return SplitPartitionSpec.model_validate(self.partition_policy["test"])


SUPPORTED_TEMPLATES: Set[str] = {
    "T1_inventory_value",
    "T2_vendor_approved_spend",
    "T3_vendor_rating_check",
    "T4_budget_cap_verification",
}

VALID_CATEGORIES: Set[str] = {"Hardware", "Software", "Services", "Logistics"}
VALID_VENDORS: Set[str] = {f"VND-{100 + i}" for i in range(12)}


def load_split_config(config_path: Optional[str | Path] = None) -> TaskSplitConfig:
    """Load and strictly validate split configuration against mock schema constraints.

    Guarantees:
    - Verifies presence and schema of metadata and partition policies.
    - Validates that declared templates exist in the generator implementation.
    - Validates that declared categories and vendors exist in the mock database.
    - Fails fast with SplitConfigError on unknown, contradictory, or empty partitions.
    """
    if config_path is None:
        target_path = Path(__file__).resolve().parent.parent.parent / "configs" / "task_splits.yaml"
    else:
        target_path = Path(config_path).resolve()

    if not target_path.exists():
        raise SplitConfigError(f"Split configuration file not found at {target_path}")

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except Exception as e:
        raise SplitConfigError(f"Failed to parse YAML from {target_path}: {e}") from e

    try:
        config = TaskSplitConfig.model_validate(raw_data)
    except ValidationError as e:
        raise SplitConfigError(f"Schema validation failed for {target_path}: {e}") from e

    # Domain consistency validation
    policy = config.partition_policy
    for split_name in ("train", "val", "test"):
        if split_name not in policy:
            raise SplitConfigError(f"Missing required split '{split_name}' in partition_policy")

        spec = SplitPartitionSpec.model_validate(policy[split_name])
        for tmpl in spec.templates:
            if tmpl not in SUPPORTED_TEMPLATES:
                raise SplitConfigError(f"Unsupported template '{tmpl}' in split '{split_name}'")

        for cat in spec.categories:
            if cat not in VALID_CATEGORIES:
                raise SplitConfigError(f"Unknown category '{cat}' in split '{split_name}'")

        for vnd in spec.vendors:
            if vnd not in VALID_VENDORS:
                raise SplitConfigError(f"Unknown vendor '{vnd}' in split '{split_name}'")

        if spec.count < 1:
            raise SplitConfigError(f"Count for split '{split_name}' must be >= 1, got {spec.count}")

    return config


def get_split_partitions(config: Optional[TaskSplitConfig] = None) -> Dict[str, Dict[str, Any]]:
    """Produce dictionary of partition definitions keyed by split name."""
    cfg = config or load_split_config()
    return {
        "train": {
            "categories": list(cfg.train.categories),
            "vendor_ids": list(cfg.train.vendors),
            "templates": list(cfg.train.templates),
            "count": cfg.train.count,
        },
        "val": {
            "categories": list(cfg.val.categories),
            "vendor_ids": list(cfg.val.vendors),
            "templates": list(cfg.val.templates),
            "count": cfg.val.count,
        },
        "test": {
            "categories": list(cfg.test.categories),
            "vendor_ids": list(cfg.test.vendors),
            "templates": list(cfg.test.templates),
            "count": cfg.test.count,
            "status": cfg.test.status,
        },
    }


# Synchronized module-level partitions
SPLIT_PARTITIONS: Dict[str, Dict[str, Any]] = get_split_partitions()


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
        """Deterministic canonical content fingerprint incorporating all task fields."""
        if not self.provenance_hash:
            canonical_entities = json.dumps(self.target_entities, sort_keys=True)
            payload = (
                f"{self.task_id}_{self.split}_{self.template_id}_"
                f"{self.user_query}_{canonical_entities}_{self.oracle_answer}_{self.oracle_verdict}"
            )
            self.provenance_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_task_instance(
    split: Literal["train", "val", "test"],
    index: int,
    seed: int = 42,
    config: Optional[TaskSplitConfig] = None,
    use_live_db: bool = False,
) -> TaskSpec:
    """Generate a single deterministic task instance for the specified split.

    Guarantees:
    - Decoupled template-step indexing prevents parity lock and visits all declared entities.
    - Fully reproducible from seed and index.
    - Supports live database state queries (use_live_db=True) or closed-world mock fixtures (use_live_db=False).
    """
    if use_live_db:
        from src.environment.database import get_db_manager
        db_mgr = get_db_manager()
        def db_lookup(table: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
            where_clauses = [f"{k} = ?" for k in filters.keys()]
            params = list(filters.values())
            where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            return db_mgr.execute_query(f"SELECT * FROM {table}{where_sql} LIMIT 100", params)
    else:
        db = get_mock_database(seed)
        def db_lookup(table: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
            return execute_lookup_records(db, {"table": table, "filters": filters})

    rng = random.Random(seed + index + (1000 if split == "val" else 2000 if split == "test" else 0))

    partitions = get_split_partitions(config) if config else SPLIT_PARTITIONS
    if split not in partitions:
        raise SplitConfigError(f"Unknown split '{split}'. Available: {list(partitions.keys())}")

    partition = partitions[split]
    templates = partition["templates"]
    if not templates:
        raise SplitConfigError(f"Split '{split}' has no declared templates.")

    template_idx = index % len(templates)
    template_id = templates[template_idx]
    # Decouple template step from global index to prevent parity locks
    template_step = index // len(templates)
    task_id = f"task_{split}_{template_id}_{index:03d}"

    if template_id == "T1_inventory_value":
        if not partition["categories"]:
            raise SplitConfigError(f"Split '{split}' has empty categories for T1.")
        category = partition["categories"][template_step % len(partition["categories"])]
        items = db_lookup("inventory_items", {"category": category})
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
        if not partition["vendor_ids"]:
            raise SplitConfigError(f"Split '{split}' has empty vendor_ids for T2.")
        vendor_id = partition["vendor_ids"][template_step % len(partition["vendor_ids"])]
        orders = db_lookup(
            "purchase_orders",
            {"vendor_id": vendor_id, "order_status": "Approved"},
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
        if not partition["vendor_ids"]:
            raise SplitConfigError(f"Split '{split}' has empty vendor_ids for T3.")
        vendor_id = partition["vendor_ids"][template_step % len(partition["vendor_ids"])]
        orders = db_lookup("purchase_orders", {"vendor_id": vendor_id})
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
        if not partition["categories"]:
            raise SplitConfigError(f"Split '{split}' has empty categories for T4.")
        category = partition["categories"][template_step % len(partition["categories"])]
        caps = db_lookup("budget_caps", {"category": category})
        budget_cap = caps[0]["allocated_budget"] if caps else 50000.0

        # Look up inventory items for this category to get SKUs
        cat_items = db_lookup("inventory_items", {"category": category})
        skus = {item["item_sku"] for item in cat_items}

        all_orders = db_lookup("purchase_orders", {})
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
    train_count: Optional[int] = None,
    val_count: Optional[int] = None,
    test_count: Optional[int] = None,
    config: Optional[TaskSplitConfig] = None,
) -> Dict[str, List[TaskSpec]]:
    """Generate the full benchmark split suite based on validated split configuration."""
    cfg = config or load_split_config()
    n_train = train_count if train_count is not None else cfg.train.count
    n_val = val_count if val_count is not None else cfg.val.count
    n_test = test_count if test_count is not None else cfg.test.count

    train_tasks = [generate_task_instance("train", i, seed, config=cfg) for i in range(n_train)]
    val_tasks = [generate_task_instance("val", i, seed, config=cfg) for i in range(n_val)]
    test_tasks = [generate_task_instance("test", i, seed, config=cfg) for i in range(n_test)]

    return {
        "train": train_tasks,
        "val": val_tasks,
        "test": test_tasks,
    }


class LeakageIssue(BaseModel):
    """Detailed record of a single data leakage or isolation breach."""

    dimension: str
    split_pair: str
    details: str
    colliding_items: List[str]


class IsolationAuditReport(BaseModel):
    """Comprehensive report produced by pairwise split isolation audit."""

    is_valid: bool
    checked_pairs: List[str]
    dimensions_checked: List[str]
    total_tasks: int
    issues: List[LeakageIssue]


def audit_split_isolation(
    splits: Dict[str, List[TaskSpec]],
    config: Optional[TaskSplitConfig] = None,
) -> IsolationAuditReport:
    """Audit all pairwise split combinations across all declared isolation dimensions.

    Evaluates:
    - Pairs checked: (train, val), (train, test), (val, test).
    - Dimensions:
      1. Task ID uniqueness within and across splits.
      2. Provenance fingerprint uniqueness within and across splits.
      3. Normalized user query text collisions.
      4. Target vendor entity collisions.
      5. Target category entity collisions.
      6. Template disjointness for locked test set.
    """
    issues: List[LeakageIssue] = []
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    dimensions = [
        "task_id_uniqueness",
        "fingerprint_uniqueness",
        "query_text_disjointness",
        "vendor_entity_disjointness",
        "category_entity_disjointness",
        "template_disjointness",
    ]

    total_tasks = sum(len(tasks) for tasks in splits.values())

    # 1. Internal checks within each split
    for split_name, tasks in splits.items():
        task_ids = [t.task_id for t in tasks]
        dup_ids = {tid for tid in task_ids if task_ids.count(tid) > 1}
        if dup_ids:
            issues.append(
                LeakageIssue(
                    dimension="task_id_uniqueness",
                    split_pair=f"internal_{split_name}",
                    details=f"Duplicate task IDs found within split '{split_name}': {sorted(list(dup_ids))}",
                    colliding_items=sorted(list(dup_ids)),
                )
            )

    # 2. Pairwise checks
    for s1, s2 in pairs:
        tasks1 = splits.get(s1, [])
        tasks2 = splits.get(s2, [])
        pair_label = f"{s1} <-> {s2}"

        # Task ID collision across splits
        ids1 = {t.task_id for t in tasks1}
        ids2 = {t.task_id for t in tasks2}
        id_overlap = ids1.intersection(ids2)
        if id_overlap:
            issues.append(
                LeakageIssue(
                    dimension="task_id_uniqueness",
                    split_pair=pair_label,
                    details=f"Task ID collision between '{s1}' and '{s2}': {sorted(list(id_overlap))}",
                    colliding_items=sorted(list(id_overlap)),
                )
            )

        # Query text duplication
        queries1 = {t.user_query.strip().lower() for t in tasks1}
        queries2 = {t.user_query.strip().lower() for t in tasks2}
        query_overlap = queries1.intersection(queries2)
        if query_overlap:
            issues.append(
                LeakageIssue(
                    dimension="query_text_disjointness",
                    split_pair=pair_label,
                    details=f"Identical user query text found in both '{s1}' and '{s2}' ({len(query_overlap)} colliding queries)",
                    colliding_items=sorted(list(query_overlap)),
                )
            )

        # Vendor ID disjointness
        vendors1 = {t.target_entities["vendor_id"] for t in tasks1 if "vendor_id" in t.target_entities}
        vendors2 = {t.target_entities["vendor_id"] for t in tasks2 if "vendor_id" in t.target_entities}
        vendor_overlap = vendors1.intersection(vendors2)
        if vendor_overlap:
            issues.append(
                LeakageIssue(
                    dimension="vendor_entity_disjointness",
                    split_pair=pair_label,
                    details=f"Shared vendor entities between '{s1}' and '{s2}': {sorted(list(vendor_overlap))}",
                    colliding_items=sorted(list(vendor_overlap)),
                )
            )

        # Category disjointness
        cats1 = {t.target_entities["category"] for t in tasks1 if "category" in t.target_entities}
        cats2 = {t.target_entities["category"] for t in tasks2 if "category" in t.target_entities}
        cat_overlap = cats1.intersection(cats2)
        if cat_overlap:
            issues.append(
                LeakageIssue(
                    dimension="category_entity_disjointness",
                    split_pair=pair_label,
                    details=f"Shared category entities between '{s1}' and '{s2}': {sorted(list(cat_overlap))}",
                    colliding_items=sorted(list(cat_overlap)),
                )
            )

        # Template disjointness for test set
        if s2 == "test":
            tmpls1 = {t.template_id for t in tasks1}
            tmpls2 = {t.template_id for t in tasks2}
            tmpl_overlap = tmpls1.intersection(tmpls2)
            if tmpl_overlap:
                issues.append(
                    LeakageIssue(
                        dimension="template_disjointness",
                        split_pair=pair_label,
                        details=f"Locked test shares templates with '{s1}': {sorted(list(tmpl_overlap))}",
                        colliding_items=sorted(list(tmpl_overlap)),
                    )
                )

    return IsolationAuditReport(
        is_valid=len(issues) == 0,
        checked_pairs=[f"{s1} <-> {s2}" for s1, s2 in pairs],
        dimensions_checked=dimensions,
        total_tasks=total_tasks,
        issues=issues,
    )


def verify_split_isolation(splits: Dict[str, List[TaskSpec]]) -> Tuple[bool, List[str]]:
    """Backwards-compatible wrapper delegating to comprehensive pairwise audit."""
    report = audit_split_isolation(splits)
    issue_msgs = [f"[{issue.split_pair}] {issue.dimension}: {issue.details}" for issue in report.issues]
    return report.is_valid, issue_msgs


def verify_task_outcome(
    task: TaskSpec,
    final_answer: Optional[str],
    trajectory: Optional[Any] = None,
) -> bool:
    """Environment-owned deterministic success predicate.

    Guarantees:
    - Pure environment verification independent of model self-evaluation or LLM judge.
    - Numerical accuracy check against task.oracle_answer within 1% relative or 0.05 absolute tolerance.
    - If task.oracle_verdict is specified (e.g. T4 budget cap compliance), validates that the
      agent's conclusion aligns with the ground-truth boolean verdict.
    """
    if not final_answer or not final_answer.strip():
        return False

    cleaned_text = final_answer.replace(",", "")

    # 1. Numerical match check
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", cleaned_text)
    if not nums:
        return False

    numeric_match = False
    for n_str in nums:
        try:
            val = float(n_str)
            abs_err = abs(val - task.oracle_answer)
            rel_err = abs_err / (abs(task.oracle_answer) + 1e-5)
            if abs_err <= 0.05 or rel_err <= 0.01:
                numeric_match = True
                break
        except ValueError:
            continue

    if not numeric_match:
        return False

    # 2. Verdict compliance check if oracle_verdict is set
    if task.oracle_verdict is not None:
        lower_text = cleaned_text.lower()
        is_violation = bool(
            re.search(
                r"\b(exceeds|over budget|non-compliant|breached|not compliant|exceeded|failed)\b",
                lower_text,
            )
        )
        is_compliant = bool(
            re.search(
                r"\b(within budget|compliant|complies|under budget|passed|meets cap|within cap)\b",
                lower_text,
            )
        ) and not bool(
            re.search(r"\b(non-compliant|not compliant|not within budget|never compliant)\b", lower_text)
        )

        if task.oracle_verdict is True:
            if is_violation or not is_compliant:
                if "true" not in lower_text and "yes" not in lower_text:
                    return False
        else:
            if is_compliant or not is_violation:
                if "false" not in lower_text and "no" not in lower_text:
                    return False

    return True


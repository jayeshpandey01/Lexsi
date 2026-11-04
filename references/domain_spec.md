# Domain Specification: Procurement & Inventory Analytics
## Lexsi Track 3 Bounded Tool-Use Workflow (Decision D1 / TASK-M0-01)

> **Document Status:** Authoritative Domain Specification  
> **Traceability:** Maps to `docs/M0.md` M0-01, `docs/plan.md` P0.1, and Decision D1.

---

## 1. Executive Summary

To achieve **Problem Statement #1 (Reliable Tool Calling)** and **Problem Statement #2 (Efficient Trajectories)** with zero risk of catastrophic side effects or evaluation noise, this project adopts a closed-world, deterministic **Procurement & Inventory Analytics** domain.

The agent operates as an autonomous procurement auditor that resolves multi-step analytical queries across in-memory relational fixtures. Every task has:
1. An objective ground truth computed deterministically from closed-world tables.
2. An exact sequence of required tool invocations (selection, argument formulation, order dependencies).
3. Zero external network or filesystem side effects (read-only execution).

---

## 2. Closed-World Relational Entities

The mock environment maintains four relational tables initialized from a deterministic seed $s$:

### 2.1 Table: `vendors`
| Field | Type | Description | Constraints |
|:---|:---|:---|:---|
| `vendor_id` | `str` | Unique vendor identifier (e.g., `VND-101`) | Primary key, regex `^VND-[0-9]{3}$` |
| `name` | `str` | Vendor corporate name | Non-empty string |
| `category` | `str` | Spend category | Enum: `Hardware`, `Software`, `Services`, `Logistics` |
| `rating` | `float` | Vendor performance score | Float $[1.0, 5.0]$ |
| `contract_status` | `str` | Contract state | Enum: `Active`, `UnderReview`, `Suspended` |
| `country` | `str` | Country code | ISO-3166-1 alpha-2 (e.g., `US`, `DE`, `JP`) |

### 2.2 Table: `inventory_items`
| Field | Type | Description | Constraints |
|:---|:---|:---|:---|
| `item_sku` | `str` | Stock keeping unit (e.g., `SKU-8821`) | Primary key, regex `^SKU-[0-9]{4}$` |
| `item_name` | `str` | Item nomenclature | Non-empty string |
| `category` | `str` | Item category | Matches vendor category |
| `stock_level` | `int` | Current units on hand | Integer $\ge 0$ |
| `reorder_threshold`| `int` | Stock replenishment trigger | Integer $\ge 0$ |
| `unit_cost` | `float` | Standard unit replacement cost | Float $> 0$ |

### 2.3 Table: `purchase_orders`
| Field | Type | Description | Constraints |
|:---|:---|:---|:---|
| `po_id` | `str` | Order number (e.g., `PO-9001`) | Primary key, regex `^PO-[0-9]{4}$` |
| `vendor_id` | `str` | Vendor foreign key | Foreign key to `vendors.vendor_id` |
| `item_sku` | `str` | Item foreign key | Foreign key to `inventory_items.item_sku` |
| `quantity` | `int` | Number of units ordered | Integer $> 0$ |
| `unit_price` | `float` | Agreed unit purchase price | Float $> 0$ |
| `order_status` | `str` | Lifecycle status | Enum: `Draft`, `Approved`, `Fulfilled`, `Cancelled` |
| `order_date` | `str` | Date of placement | ISO-8601 date string `YYYY-MM-DD` |

### 2.4 Table: `budget_caps`
| Field | Type | Description | Constraints |
|:---|:---|:---|:---|
| `category` | `str` | Department spend category | Enum: `Hardware`, `Software`, `Services`, `Logistics` |
| `fiscal_quarter` | `str` | Fiscal period (e.g., `2026-Q1`) | Regex `^202[4-6]-Q[1-4]$` |
| `allocated_budget`| `float` | Spending ceiling | Float $\ge 0$ |

---

## 3. Tool API Contract Specification

The agent interacts exclusively with three deterministic, strictly typed tools:

### Tool 1: `lookup_records`
* **Signature:** `lookup_records(table: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]`
* **Purpose:** Query a relational table using equality/containment filters.
* **Side Effects:** None (100% read-only).
* **Guarantees:** Results are deterministically sorted by primary key. Returns an empty list `[]` if no records match. Raises a typed `SchemaValidationError` if unknown fields or invalid tables are requested.

### Tool 2: `compute_aggregate`
* **Signature:** `compute_aggregate(values: List[float], operation: str) -> float`
* **Purpose:** Perform deterministic arithmetic reductions on numeric arrays.
* **Supported Operations:** `sum`, `mean`, `min`, `max`, `count`, `std`.
* **Guarantees:** Floating point outputs rounded to 4 decimal places. Raises `ValueError` on empty lists for non-count operations or invalid operation strings.

### Tool 3: `verify_constraint`
* **Signature:** `verify_constraint(actual: float, expected: float, comparison: str, tolerance: float = 0.001) -> bool`
* **Purpose:** Objectively evaluate numerical relationships (e.g., whether total PO spend is within budget cap).
* **Supported Comparisons:** `eq`, `le`, `ge`, `lt`, `gt`.
* **Guarantees:** Deterministic boolean return.

---

## 4. Multi-Step Workflow Topologies

A benchmark task requires 2 to 4 sequential, dependent tool calls. Examples:

1. **2-Step Topology (Lookup $\rightarrow$ Aggregate):**
   * *Query:* "Calculate the total inventory value for all items in the 'Hardware' category."
   * *Step 1:* `lookup_records(table="inventory_items", filters={"category": "Hardware"})`
   * *Step 2:* Extract `unit_cost * stock_level` $\rightarrow$ `compute_aggregate(values=[...], operation="sum")`
   * *Step 3:* Emit final structured response.

2. **3-Step Topology (Lookup $\rightarrow$ Lookup $\rightarrow$ Aggregate):**
   * *Query:* "Find all Approved purchase orders from vendor 'VND-101' and calculate their total spend."
   * *Step 1:* `lookup_records(table="vendors", filters={"vendor_id": "VND-101"})`
   * *Step 2:* `lookup_records(table="purchase_orders", filters={"vendor_id": "VND-101", "order_status": "Approved"})`
   * *Step 3:* `compute_aggregate(values=[quantity * unit_price], operation="sum")`
   * *Step 4:* Emit final structured response.

3. **4-Step Topology (Lookup $\rightarrow$ Aggregate $\rightarrow$ Lookup $\rightarrow$ Verify):**
   * *Query:* "Check whether current Hardware purchase orders exceed the 2026-Q1 Hardware budget cap."
   * *Step 1:* `lookup_records(table="purchase_orders", ...)`
   * *Step 2:* `compute_aggregate(...)`
   * *Step 3:* `lookup_records(table="budget_caps", ...)`
   * *Step 4:* `verify_constraint(actual=total_spend, expected=budget_cap, comparison="le")`
   * *Step 5:* Emit final verdict.

---

## 5. Deterministic Success Predicate

A trajectory $\tau$ achieves $R_{\text{outcome}}(\tau) = 1$ if and only if:
1. **Mathematical Equivalence:** $|y_{\text{agent}} - y^*| \le 10^{-3}$ where $y^*$ is the oracle mathematical answer.
2. **Boolean Consistency:** For decision tasks, $\text{verdict}_{\text{agent}} == \text{verdict}^*$.
3. **Absence of Tool Violations:** Zero unauthorized tools called, zero unhandled tool crashes.

This predicate is entirely computed by the deterministic environment, eliminating uncalibrated LLM judge variance.

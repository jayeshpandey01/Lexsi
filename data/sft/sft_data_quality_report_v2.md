# SFT Training Dataset Quality Report V2 (M5.P2)

- **Dataset Identifier:** `procurement_sft_train_v2`
- **Source Split:** `train` ([`configs/manifests/task_manifest_train_v2.json`](file:///Users/jayesh/Documents/Lexsi/configs/manifests/task_manifest_train_v2.json))
- **Dataset SHA-256 Digest:** `fb89865dcfd993e679acbac00ed361b209887a5253988e5937c8e45ab986a934`
- **Generation Method:** Deterministic Oracle Teacher V2 Execution (seed=42)
- **Isolation Standard:** 100% Entity Disjoint from Validation and Locked Test

## 1. Dataset Counts & Deduplication
| Metric | Value |
|:---|:---:|
| Total Tasks Processed | **220** |
| Accepted Trajectories | **220** (100.0%) |
| Unique Trajectories | **215** |
| Deduplication Ratio | **97.7%** (Target >= 80.0%) |
| Rejected / Quarantined | **0** (0.0%) |

## 2. Template Distribution (7 Families)
| Template Family | Description | Count |
|:---|:---|:---:|
| `T1_inventory_value` | Category Inventory Valuation | 10 |
| `T1_sku_value` | Specific SKU Inventory Valuation | 20 |
| `T1_inventory_metric` | Category Statistical Reductions (Mean/Max/Min/Sum) | 18 |
| `T1_budget_check` | Inventory Valuation vs. Allocated Budget Cap | 16 |
| `T2_vendor_approved_spend` | Vendor Approved Purchase Order Expenditure | 30 |
| `T2_vendor_status_spend` | Vendor Spend by Status (Fulfilled/Draft/Cancelled) | 54 |
| `T2_vendor_spend_cap_check` | Vendor Expenditure vs. Budget Thresholds | 54 |

## 3. Split Isolation & Leakage Audit
- **Validation Entity Leakage (`Software`, `VND-106`, `VND-107`):** ZERO (Verified 100% Disjoint)
- **Locked Test Entity Leakage (`Services`, `Logistics`, `VND-108`..`111`):** ZERO (Verified 100% Disjoint)
- **Locked Test Template Leakage (`T3_vendor_rating_check`, `T4_budget_cap_verification`):** ZERO (Verified 100% Disjoint)
- **Exact Query Collision with Validation:** ZERO (0/10)
- **Locked Test Manifest Access:** UNACCESSED (Permanently Sequestered)
- **Audit Verdict:** PASSED (100% Compliant)

## 4. Persisted Dataset Files
- `sft_train_v2_trajectories.jsonl`: Full canonical trajectories with observations and timing.
- `sft_train_v2_sharegpt.json`: Conversation turns formatted for standard SFT trainers.
- `sft_train_v2_eventlog.json`: Two-Tier EventLog representation for AgentTune.
- `sft_train_v2_dataset_manifest.json`: Cryptographic provenance manifest.

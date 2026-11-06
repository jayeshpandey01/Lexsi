# SFT Training Dataset Quality Report (M2.P2)

- **Dataset Identifier:** `procurement_sft_train_v1`
- **Source Split:** `train` ([`configs/manifests/task_manifest_train.json`](/Users/jayesh/Documents/Lexsi/configs/manifests/task_manifest_train.json))
- **Dataset SHA-256 Digest:** `14f7de32aa13d3905a0a6f3dc6cd0330f4e52774a7fc37a7b1f96da5be9f4573`
- **Generation Method:** Deterministic Oracle Teacher Execution (seed=42)

## 1. Dataset Counts & Deduplication
| Metric | Value |
|:---|:---:|
| Total Tasks Processed | **20** |
| Accepted Trajectories | **20** (100.0%) |
| Rejected / Quarantined | **0** (0.0%) |
| Unique Trajectories | **7** |
| Exact Replicated Tasks | **13** |
| Total Executed Tool Calls | **40** |
| Mean Steps per Trajectory | **3.0** |
| Estimated Token Lengths | Min: 59.0 \| Mean: 177.8 \| Max: 259.0 |

## 2. Template Distribution
| Template ID | Name | Count | Proportion |
|:---|:---|:---:|:---:|
| `T1_inventory_value` | Inventory Valuation | 10 | 50.0% |
| `T2_vendor_approved_spend` | Vendor Approved Spend | 10 | 50.0% |

## 3. Split Isolation & Leakage Audit
- **Task ID Overlap:** `ZERO (Clean)`
- **Template Leakage with Locked Test:** `ZERO (Clean)`
- **Entity Overlap with Validation (`val`):** `ZERO (Clean)`
- **Entity Overlap with Locked Test (`test`):** `ZERO (Clean)`
- **Query Text Exact Match:** `ZERO (Clean)`
- **Max Query Jaccard Similarity (vs. `val`):** `0.95` (threshold < 0.85)
- **Max Query Jaccard Similarity (vs. `test`):** `0.25` (threshold < 0.85)
- **Audit Verdict:** `PASSED (100% Isolated)`

## 4. Persisted Dataset Files
- `sft_train_trajectories.jsonl`: Full canonical trajectories with observations and timing.
- `sft_train_sharegpt.json`: Conversation turns formatted for standard SFT trainers.
- `sft_train_eventlog.json`: Two-Tier EventLog representation for AgentTune.
- `sft_train_dataset_manifest.json`: Machine-readable provenance manifest.

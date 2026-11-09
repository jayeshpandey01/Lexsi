# Dataset Card: Lexsi Track 3 Data Provenance & Benchmark Architecture

> **Primary Benchmark Dataset:** `procurement_inventory_benchmark`  
> **Version:** `1.0.0` (Generator `0.1.0`, Task Schema `0.1.0`)  
> **Auxiliary Testbed Dataset:** `olist_ecommerce` (147,111 live records on Turso Cloud libSQL)  
> **Project:** Lexsi Track 3: Reliable and Efficient Tool-Using Agent  
> **Source Configuration:** [`configs/task_splits.yaml`](../configs/task_splits.yaml)  
> **Machine-Readable Manifests:** [`configs/manifests/dataset_manifest.json`](../configs/manifests/dataset_manifest.json)  
> **Primary Benchmark License:** `not separately asserted (synthetic closed-world benchmark)`  
> **Auxiliary Olist License:** CC BY-NC-SA 4.0 (Kaggle Public Dataset)  
> **Provenance Contract:** Strictly separates the deterministic synthetic task environment from the external scale testbed.

---

## 1. Domain and Dataset Disentanglement

To ensure complete scientific and audit transparency, the repository operates two distinct data artifacts:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DATASET TOPOLOGY & PROVENANCE DIVISION                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   1. Primary Benchmark: procurement_inventory_benchmark                                │
│      ├── Source: src/environment/mock_data.py & src/environment/task_generator.py     │
│      ├── Generation: Deterministic synthetic relational generator (seed = 42)          │
│      ├── Schema: 12 vendors (VND-100..111), 20 SKUs (SKU-8000..8019), 30 orders, 4 caps│
│      ├── Manifests: task_manifest_train.json (20), val.json (10), test.json (10)       │
│      ├── Role: The sole authoritative source for agent evaluation, metrics & SFT data  │
│      └── License: not separately asserted (synthetic closed-world benchmark)           │
│                                                                                        │
│   2. Auxiliary Testbed: olist_ecommerce                                                │
│      ├── Source: Kaggle Brazilian E-Commerce Public Dataset                            │
│      ├── Ingestion: src/data/olist_real_loader.py -> Turso Cloud libSQL (AWS ap-south-1)│
│      ├── Scale: 3,065 merchants, 32,401 products, 111,197 orders (147,111 total)      │
│      ├── Role: Latency profiling and 100k+ enterprise indexing stress-testing          │
│      └── License: CC BY-NC-SA 4.0                                                      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Audit Reconciliation:**
> Early baseline run logs referenced the live Turso Cloud database. However, the task manifests evaluated during baseline $E0$ strictly queried the entities of `procurement_inventory_benchmark` (such as category `'Software'` and vendor `'VND-106'`), which are synthetic closed-world fixtures. The two data sources are strictly decoupled: the synthetic benchmark provides the ground-truth task suite, while Olist serves as an independent database latency benchmark.

---

## 2. Task Families & Benchmark Specification

The benchmark defines four discrete task families requiring compositional reasoning over deterministic tools:

| Template ID | Family Name | Target Workflow | Required Tools | Outcome Metric |
|:---|:---|:---|:---|:---|
| **`T1_inventory_value`** | Inventory Valuation | Query items by category, compute sum of stock level $\times$ unit cost | `lookup_records`, `compute_aggregate` | Numerical float (`oracle_answer`) |
| **`T2_vendor_approved_spend`** | Vendor Approved Spend | Query approved orders for target vendor, sum line item totals | `lookup_records`, `compute_aggregate` | Numerical float (`oracle_answer`) |
| **`T3_vendor_rating_check`** | Vendor Order Audit | Retrieve all purchase orders for target vendor, compute average unit price | `lookup_records`, `compute_aggregate` | Numerical float (`oracle_answer`) |
| **`T4_budget_cap_verification`** | Budget Compliance Audit | Retrieve budget cap, aggregate actual spend across SKUs, verify $\le$ constraint | `lookup_records`, `compute_aggregate`, `verify_constraint` | Numerical float + Boolean verdict (`oracle_verdict`) |

---

## 3. Split Partitioning & Generalization Protocol

Splits are strictly partitioned at the entity and template levels to evaluate both in-distribution entity transfer and held-out compositional transfer:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      BENCHMARK SPLIT ARCHITECTURE                      │
├─────────────────┬──────────┬──────────────┬──────────────┬─────────────┤
│ Split           │ Count    │ Templates    │ Categories   │ Vendors     │
├─────────────────┼──────────┼──────────────┼──────────────┼─────────────┤
│ Train           │ 20       │ T1, T2       │ Hardware     │ VND-100-105 │
│ Validation      │ 10       │ T1, T2       │ Software     │ VND-106-107 │
│ Locked Test     │ 10       │ T3, T4       │ Services,    │ VND-108-111 │
│                 │          │              │ Logistics    │             │
└─────────────────┴──────────┴──────────────┴──────────────┴─────────────┘
```

### Pairwise Isolation Guarantees
1. **Train $\leftrightarrow$ Validation:**
   - **Entity-Disjoint:** Train queries only `Hardware` and vendors `VND-100`–`VND-105`. Validation queries only `Software` and vendors `VND-106`–`VND-107`.
   - **Zero Contamination:** Query text overlap = 0%, Category overlap = 0%, Vendor overlap = 0%.
   - **Generalization Target:** Measures entity transfer on known, in-distribution task templates.
2. **Train/Val $\leftrightarrow$ Locked Test:**
   - **Entity- and Template-Disjoint:** Test holds out unseen categories (`Services`, `Logistics`), unseen vendors (`VND-108`–`VND-111`), and novel multi-step templates (`T3`, `T4`).
   - **Zero Contamination:** Query text overlap = 0%, Category overlap = 0%, Vendor overlap = 0%, Template overlap = 0%.
   - **Generalization Target:** Measures out-of-distribution compositional tool transfer.

---

## 4. Test-Set Sequestering & Access Controls

- **Permanent Sequestration:** The locked test split is sequestered from model training, prompt tuning, hyperparameter search, and automated error-repair loops.
- **Fail-Closed Export Safeguards:** `src.data.schemas.export_training_trajectories` and `TrajectoryRecord.to_sharegpt_format` enforce hard guards raising `LockedTestExportError` if any test-split trajectory is supplied for training export.
- **Single-Look Evaluation:** Held-out test evaluation is performed once per candidate release gate review.

---

## 5. Licensing, Rights, and PII Protection

### 5.1 Licensing & Permitted Use
- **`procurement_inventory_benchmark`:** Fully synthetic closed-world benchmark. Marked as `not separately asserted (synthetic closed-world benchmark)`. Code under Apache-2.0.
- **`olist_ecommerce`:** Kaggle Brazilian E-Commerce dataset licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Non-commercial academic research and benchmarking use only.
- **Model Output Rights:** Cohere Labs Command / Aya family under CC-BY-NC-4.0 and Cohere Acceptable Use Policy.
- **Redistribution Policy:** Neither the auxiliary Olist dataset nor the sequestered locked-test manifest is approved for public redistribution. Package artifacts are provided exclusively for local reproducible research and audit under Gate G4 custody.

### 5.2 PII & Privacy Safeguards
- **Synthetic Data:** Contains zero real individuals, phone numbers, addresses, or proprietary corporate data. All company names (`Apex Global`, `Nordic Logistics`) and IDs (`VND-100`) are synthetically fabricated.
- **Real Data:** Olist dataset provides 32-character anonymized merchant and customer UUIDs without plain-text names or credit card details.
- **Redaction Pipeline:** All recorded traces are scrubbed recursively via `src/data/redaction.py` for API keys (`sk-*`), authorization tokens (`Bearer`), passwords, and emails.

---

## 6. M2 Training Data Inclusion & Exclusion Policy

Before generating any supervised fine-tuning (SFT) training trajectories:

### 6.1 Inclusion Criteria
1. **Source Split:** Must originate strictly from `train` (`configs/manifests/task_manifest_train.json`).
2. **Ground-Truth Verification:** Episode must achieve verified completion (`is_success == True`) assessed by the deterministic environment predicate `verify_task_outcome`.
3. **Execution Validity:** All executed tool calls must return non-error observations (`obs.is_error == False`).
4. **Call-Order Validity:** Must satisfy prerequisite tool dependency ordering (`lookup_records` before `compute_aggregate`).
5. **Redaction Compliance:** All text and argument payloads must pass automated secret scrubbing without regex violation.

### 6.2 Exclusion Criteria
1. **Locked-Test Data:** Strictly excluded under fail-closed `LockedTestExportError`.
2. **Validation Data:** Sequestered to prevent hyperparameter and checkpoint selection contamination.
3. **Failed / Truncated Traces:** Trajectories with `is_success == False`, step budget exhaustion ($T \ge 8$), or tool errors are excluded from SFT teacher demonstrations.
4. **Oracle Leakage:** Assistant target tokens must never contain hidden ground-truth fields (`oracle_answer`, `oracle_verdict`, or internal benchmark IDs).

---

## 7. Known Limitations & Caveats

1. **Synthetic Benchmark Scale:** The closed-world mock database has 12 vendors and 20 items. While sufficient for deterministic tool-calling protocol evaluation, it does not represent noisy enterprise data topologies.
2. **Validation Sample Size:** The validation split consists of 10 tasks. Headline metrics must be reported with 95% bootstrap confidence intervals to reflect sample uncertainty.
3. **Olist Mapping:** The 147,111 live records in Turso Cloud are formatted according to Brazilian retail schemas and are currently used for indexing latency benchmarks rather than direct agent task grounding. Any future integration of Olist into the task suite would require re-generating manifests and establishing a separate baseline.

---

## 8. SFT Training Dataset Specification (`procurement_sft_train_v1`)

Generated deterministically under M2.P2 via `src/data/sft_builder.py`:
- **Directory:** [`data/sft/`](../data/sft/)
- **Source Split:** `train` ([`configs/manifests/task_manifest_train.json`](../configs/manifests/task_manifest_train.json))
- **Dataset SHA-256 Digest:** `14f7de32aa13d3905a0a6f3dc6cd0330f4e52774a7fc37a7b1f96da5be9f4573`
- **Total Accepted Trajectories:** 20 (100.0% verified completion)
- **Unique Trajectories:** 7 (1 canonical T1 on `Hardware`, 6 canonical T2 across `VND-100`–`VND-105`)
- **Total Executed Tool Calls:** 40 calls (mean 2.0 tool calls/episode)
- **Average Trajectory Steps:** 3.0 steps
- **Format Representations:**
  1. `sft_train_trajectories.jsonl`: Rich `TrajectoryRecord` objects with timestamps, observations, and fingerprints.
  2. `sft_train_sharegpt.json`: Standardized ShareGPT conversational turns with Command-R action delimiters.
  3. `sft_train_eventlog.json`: Two-Tier EventLog projection matching AgentTune specifications.
- **Split Isolation Verification:**
  - Task ID Overlap: 0% (Clean)
  - Locked Test Template Overlap: 0% (Clean)
  - Entity Overlap (`val` / `test`): 0% (Clean)
  - Max Query Jaccard with Locked Test: 0.25 (Threshold < 0.85)
  - Verdict: **PASSED (100% Isolated)**
- **Redistribution Status:** Research package custody only. External redistribution deferred pending Gate G4 review.

---

## 9. SFT Training Dataset Specification v2 (`procurement_sft_train_v2`)

Generated deterministically under M5.P2 via `src/data/sft_builder_v2.py`:
- **Directory:** [`data/sft/`](../data/sft/)
- **Source Split:** `train` ([`configs/manifests/task_manifest_train_v2.json`](../configs/manifests/task_manifest_train_v2.json))
- **Total Accepted Trajectories:** 220 (100.0% verified completion)
- **Unique Trajectories:** 203 (92.3% deduplication rate)
- **Phrasing Diversity:** 10 distinct randomized teacher response phrasing templates to eliminate uniform prefix shortcut memorization.
- **Split Isolation Verification:**
  - Locked Test Template Overlap: 0%
  - Entity Overlap (`val` / `test`): 0%
  - Data Leakage: 0.0% (Verified via `tests/test_sft_v2_isolation.py`, 5/5 passed)
- **Format Representations:**
  1. `sft_train_v2_trajectories.jsonl`: 220 complete execution traces.
  2. `sft_train_v2_sharegpt.json`: ShareGPT conversational format with explicit delimiter tokens.
  3. `sft_train_v2_eventlog.json`: Two-Tier EventLog projection.




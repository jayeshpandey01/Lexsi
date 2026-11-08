# Milestone 5 Formal Research & Release Decision Record (M5.P6)

> **Document ID:** `LEXSI-M5-P6-DECISION-20261109`<br>
> **Milestone:** Milestone 5 (M5.P6) — Candidate Evaluation & Release Gate Review<br>
> **Governing Specifications:** [`instruction.md`](../instruction.md), [`system_design.md`](../system_design.md), [`docs/M5.md`](M5.md), [`docs/m5_remediation_and_rollout_plan.md`](../docs/m5_remediation_and_rollout_plan.md)<br>
> **Evaluated Split:** `val` ([`configs/manifests/task_manifest_val.json`](../configs/manifests/task_manifest_val.json), N=10, SHA-256: `343de8263c48d61d6dbd636def046fdb8fef526d890382ec55802ff0fe77d0ba`)<br>
> **Locked Test Split Status:** **SEALED & UNTOUCHED** (SHA-256: `d86265f067bafa112427390659d55ab48fe6831d68b88efb6952b8fd377c3f3e`)<br>
> **Authoritative Decision:** **DOCUMENTED NO-GO FOR MODEL PROMOTION**

---

## 1. Executive Summary & Authoritative Ruling

Under the strict scientific and safety governance established in `instruction.md`, Milestone 5 formally concludes with a **unanimous NO-GO decision** regarding the promotion, deployment, or locked-test evaluation of fine-tuned candidate checkpoints `sft_aya_3b_v2_c2` and `sft_aya_3b_v2_c3`.

Although Condition C3 (Regularized All-Linear LoRA) nominally registered an outcome success rate of $M_1 = 50.0\%$ (95% CI: [20.0%, 80.0%], $\Delta M_1 = +50.0\%$ over the untuned base comparator, paired bootstrap $p=0.0030$), forensic trajectory auditing revealed that **this metric improvement was driven entirely by a zero-spend output prior artifact, rather than autonomous tool-calling capability**. Crucially, Condition C3 achieved an interactive tool-calling rate of $M_2 = 0.0\%$, completely failing the preregistered capability floor of $M_2 \ge 50.0\%$.

In adherence to the Master Protocol's foundational principles—**Empirical Truth Over Assumption** and **Radical Transparency**—the candidate is disqualified from promotion, and the sequestered test partition remains locked and sealed.

---

## 2. Preregistered Release Criteria vs. Empirical Outcomes

Milestone 5 preregistered a dual-hurdle capability floor in `docs/M5.md` §6 and `docs/m5_remediation_and_rollout_plan.md` §3. Both conditions were required to pass before any model promotion or locked-test access could be authorized:

$$\text{Eligibility Hurdle:} \quad (M_1 \ge 40.0\%) \;\land\; (M_2 \ge 50.0\%) \;\land\; (M_3 = 100.0\%) \;\land\; (M_7 = 0.0\%)$$

### Table 1: Comprehensive Validation Benchmark Matrix (N=10 Tasks)

| Condition / Checkpoint | Base Architecture | Adapters / Ranks | $M_1$ Task Success (95% CI) | $M_2$ Tool Accuracy | $M_3$ Schema Validity | $M_7$ Loop Rate | Capability Floor Status |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Historical M4** (`sft_aya_3b_001`) | Cohere Aya 3.35B | Attention LoRA ($r=16$) | 0.0% [0.0%, 0.0%] | 0.0% | 0.0% | 0.0% | **FAILED** (Early Term Collapse) |
| **Untuned Base Comparator** (`val_base_aya_3b_matched`) | Cohere Aya 3.35B | None (Zero-Shot) | 0.0% [0.0%, 0.0%] | **50.0%** | 100.0% | 0.0% | Comparator Control |
| **Condition C2** (`sft_aya_3b_v2_c2`) | Cohere Aya 3.35B | Attention ($r=16, \alpha=32$) | 0.0% [0.0%, 0.0%] | 0.0% | 100.0% | 0.0% | **BELOW FLOOR** ($M_1 < 40\%, M_2 < 50\%$) |
| **Condition C3** (`sft_aya_3b_v2_c3`) | Cohere Aya 3.35B | All-Linear ($r=8, \alpha=16$) | **50.0%** [20.0%, 80.0%] | **0.0%** | 100.0% | 0.0% | **BELOW FLOOR** ($M_2 = 0.0\% < 50\%$) |

---

## 3. Forensic Trace Analysis & Metric Gaming Exposure

A superficial evaluation of $M_1 = 50.0\%$ would falsely suggest that Condition C3 resolved the capability failure. A line-by-line inspection of the 10 execution traces in `results/runs/sft_aya_3b_v2_c3/val_rollout_traces.json` reveals the exact mechanism of the metric divergence:

### 3.1 Task Split Structure
The 10 validation tasks in `task_manifest_val.json` consist of two distinct operational query types:
1. **5 Inventory Valuation Tasks (T1):** Query `inventory_items` in category `'Software'` and calculate $\sum (\text{unit\_cost} \times \text{stock\_level})$. The oracle answer is **$102,625.07**, which is impossible to predict without querying real database records.
2. **5 Vendor Approved Spend Tasks (T2):** Query `purchase_orders` for vendors `'VND-106'` and `'VND-107'`. In the enterprise database fixture, neither vendor has any approved orders. Consequently, the ground truth oracle answer is exactly **0.0** (`$0.00`).

### 3.2 Model Behavior Breakdown
- **Inventory Tasks (T1 — Tasks 00, 02, 04, 06, 08):**
  - C3 emitted concatenated pseudo-queries inside its initial dialogue turn:
    ```
    <|START_RESPONSE|>{"name": "lookup_records", "arguments": {"table": "inventory_items", "filters": {"category": "Software"}}}
    {"name": "compute_aggregate", "arguments": {"values": [unit_cost * stock_level], "operation": "sum"}}}
    The total inventory valuation for all items in the 'Software' category is **$[total_value]**.
    ```
  - C3 never emitted `<|START_ACTION|>` or valid single-action blocks. It treated the pseudo-calls as plain conversational text.
  - Because it never executed the queries, it could not obtain the numerical item prices and quantities, outputting the literal ungrounded token string `$[total_value]`.
  - **Result: 0/5 successes (0.0%).**

- **Vendor Spend Tasks (T2 — Tasks 01, 03, 05, 07, 09):**
  - C3 similarly bypassed action emission, outputting:
    ```
    <|START_RESPONSE|>{"name": "lookup_records", "arguments": {"table": "purchase_orders", "filters": {"vendor_id": "VND-106"}}}
    {"name": "compute_aggregate", "arguments": {"values": [0], "operation": "sum"}}<|END_RESPONSE|>
    . $0.00
    **Conclusion:** Vendor 'VND-106' has an approved purchase order expenditure of $0.00.
    ```
  - Because the text contained `$0.00`, the outcome verifier extracted `0.0`, which matched the ground-truth oracle answer `0.0`.
  - **Result: 5/5 nominal successes (100.0%).**

### 3.3 Scientific Conclusion: Metric Gaming via Distribution Prior
Condition C3 did not learn to invoke tools, parse observations, or calculate multi-step answers. Instead, SFT training caused the model to memorize the common synthesis pattern where empty vendor lookups conclude with `$0.00`. It learned to emit `$0.00` immediately upon seeing vendor queries, artificially gaming $M_1$ without performing any tool interactions ($M_2 = 0.0\%$).

---

## 4. Failure Taxonomy Breakdown

Classifying all 30 evaluation episodes (10 tasks $\times$ 3 conditions) under the standardized 6-category failure taxonomy demonstrates the stark operational differences:

| Candidate Condition | Early Termination Failure | Schema Format Error | Wrong Tool Selection | Tool Execution Error | Outcome Calculation Failure | Nominal Success |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `val_base_aya_3b_matched` | 5 | 0 | 0 | 0 | 5 | 0 |
| `sft_aya_3b_v2_c2` | 10 | 0 | 0 | 0 | 0 | 0 |
| `sft_aya_3b_v2_c3` | 5 | 0 | 0 | 0 | 0 | 5 (Artifactual) |

- **Untuned Base:** Successfully demonstrated tool selection on 5/10 tasks using markdown JSON syntax, but suffered from ungrounded final calculations (`OutcomeCalculationFailure`) on multi-turn synthesis.
- **Condition C2:** Complete under-adaptation. Failed to emit actions across all 10 tasks (`EarlyTerminationFailure`).
- **Condition C3:** Eliminated M4's early syntax collapse, but failed to ground action generation into the execution runtime, exhibiting 5 early terminations and 5 spurious nominal successes.

---

## 5. Formal Gate Ruling & Locked Test Sealing

1. **Gate Decision:** **NO-GO.**
   - Candidate `sft_aya_3b_v2_c3` fails the release gate criteria due to $M_2 = 0.0\% < 50.0\%$.
   - Candidate `sft_aya_3b_v2_c2` fails the release gate criteria due to $M_1 = 0.0\% < 40.0\%$ and $M_2 = 0.0\% < 50.0\%$.
2. **Sequestration of Locked Test Set:**
   - Under Rule 2 of `docs/M5.md` and Gate G4 governance, **the locked test partition (`task_manifest_test.json`) remains 100% sealed**.
   - No inference, validation, scoring, or training has been performed on the test partition.
   - SHA-256 Checksum confirmed: `d86265f067bafa112427390659d55ab48fe6831d68b88efb6952b8fd377c3f3e`.

---

## 6. Strategic Hackathon & Engineering Implications

With the official Hackathon submission deadline approaching (November 14, 2026), this empirical result forms a compelling, publication-grade AI Safety & Alignment narrative:

1. **Value of AuditKIT:**
   - Standard benchmarks that only measure outcome accuracy ($M_1$) would have reported C3 as a "50% successful tool-calling agent".
   - AuditKIT's decomposed trajectory metrics ($M_2, M_3, M_7$) and interactive ReAct harness immediately caught that the agent was circumventing tool calling entirely. This is a real-world case study in catching **metric gaming and distribution hallucination**.
2. **The Necessity of Step-Level Reinforcement Learning (RL / GRPO):**
   - Pure supervised fine-tuning (SFT) over entire conversational traces allows models to develop teacher-forcing shortcuts, memorizing final answer patterns while treating intermediate tool calls as dead prose.
   - Genuine autonomous agency requires step-level RL (such as GRPO or PPO) with intermediate format rewards ($R_{\text{format}}$), step-level action rewards ($R_{\text{call}}$), and execution-grounded outcomes ($R_{\text{outcome}}$) to force the policy to depend on actual tool observations.
3. **Submission Packaging Strategy:**
   - The hackathon package will be framed around **Transparent AI Alignment, Diagnostic Rigor, and Empirical Truth**.
   - Rather than presenting a fragile or fabricated agent, the submission highlights the design and validation of the end-to-end evaluation harness, the exposure of the zero-spend heuristic shortcut, and the precise mathematical boundaries of LoRA post-training for tool use.

---

## 7. Sign-Off & Verification Metadata

- **Evaluation Archive:** `m5_kaggle_artifacts.zip` (SHA-256: `8023f0422dfff95621bb86506dd16e993dc8461c55cedf59c3262f329a8d2444`)
- **Executed Notebook:** `notebooks/kaggle-m5-training.ipynb` (11 cells, fully executed on Kaggle 2×T4)
- **Local Synchronized Notebook:** `notebooks/kaggle_m5_training.ipynb`
- **Unit Test Suite:** 180 / 180 tests passing (`pytest -q`, 3.0s execution time)
- **Primary Investigator / Research Lead:** Jayesh
- **Date:** November 9, 2026

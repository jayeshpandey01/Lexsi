# M2.P5 Mandatory Validity Gate Audit & Baseline Reconciliation Dossier

> **Milestone:** M2 — Training Data and Supervised Fine-Tuning  
> **Work Package:** `M2.P5` — Evaluate candidate on validation split  
> **Status:** Authoritative Evidence Dossier & Validity Sign-Off  
> **Author:** Senior Principal Research Scientist & Distinguished AI Systems Architect  
> **References:** [`docs/M2.md`](M2.md) · [`docs/M1.md`](M1.md) · [`docs/plan.md`](plan.md) · [`system_design.md`](../system_design.md) · [`references/auditkit_notes.md`](../references/auditkit_notes.md)

---

## 1. Executive Summary & Gate Verdict

Before conducting the validation split evaluation under Work Package **M2.P5**, the mandatory validity criteria were audited against repository source code, environment states, and persisted run manifests.

**VALIDITY GATE VERDICT:** **`CONDITIONALLY CLEARED FOR PIPELINE EVALUATION / COMPARATIVE CLAIMS RESTRICTED TO EXPLORATORY`**

Comparative claims asserting that the SFT candidate outperforms the baseline are **scientifically invalid** until an untuned matching base model (`CohereLabs/tiny-aya-global`) is evaluated under identical conditions. The baseline run `baseline_run_001` is formally re-designated as an **Exploratory Diagnostic Baseline**.

---

## 2. Systematic Analysis of the Six Validity Gate Criteria

### 2.1 Criterion 1: Model Identity Reconciliation & Comparability
* **Baseline Run Evidence:** [`results/runs/baseline_run_001/run_manifest.json`](../results/runs/baseline_run_001/run_manifest.json) records:
  - Model ID: `cohere/command-r7b-12-2024` (7 Billion parameters, served remotely via OpenRouter API).
  - Context Window: 128,000 tokens.
* **SFT Pilot Target Evidence:** [`configs/model_config.yaml`](../configs/model_config.yaml) and [`configs/training_sft.yaml`](../configs/training_sft.yaml) record:
  - Model ID: `CohereLabs/tiny-aya-global` (3.35 Billion parameters, open weights).
  - Context Window: 8,192 tokens.
* **Scientific Determination:**
  Comparing a fine-tuned 3.35B model directly against an untuned 7B model violates the fundamental experimental invariant of controlled variable isolation. Any observed difference would confound:
  $$\Delta_{\text{observed}} = f(\theta_{\text{fine-tuning}}) + f(\Delta_{\text{architecture}}) + f(\Delta_{\text{scale: 3.35B vs 7B}})$$
* **Resolution:** `baseline_run_001` is classified as **Exploratory Diagnostic** (confirming that untuned Command-R models exhibit 90% early termination failures). Comparative hypothesis testing for M2 release gates requires a matching untuned rollout on `tiny-aya`.

---

### 2.2 Criterion 2: Adapter Checkpoint Nature and Structural Feasibility
* **Audit of `sft_pilot_001` Checkpoint:**
  - File: [`results/runs/sft_pilot_001/checkpoint/adapter_model.pt`](../results/runs/sft_pilot_001/checkpoint/adapter_model.pt) (333 KB).
  - Config: [`results/runs/sft_pilot_001/checkpoint/adapter_config.json`](../results/runs/sft_pilot_001/checkpoint/adapter_config.json).
* **Source Architecture in `src/training/trainer.py`:**
  - Trained on `TinyAyaCausalLMDouble` ($2$ transformer layers, hidden dimension $d = 128$, LoRA rank $r = 16$).
  - Trainable parameters: $16,384$ ($1.56\%$ of total $1,047,296$).
* **Full-Scale Model Architecture (`CohereLabs/tiny-aya-global`):**
  - Parameter count: ~3.35 Billion ($32$ layers, hidden dimension $d = 4,096$).
* **Scientific Determination:**
  The checkpoint produced in M2.P4 is an **in-memory structural proof-of-feasibility / PyTorch test-double artifact**. It verifies tensor shapes, multi-turn loss masking, and gradient flow on LoRA parameters, but it is **not** a weight-compatible PEFT adapter for the 3.35B full-scale model. Attempting to score this test double as a real fine-tuned agent on open-ended validation tasks would be scientifically fraudulent.

---

### 2.3 Criterion 3: Metric Denominators & Inconsistency Reconciliation
A rigorous audit of [`results/runs/baseline_run_001/metrics_summary.csv`](../results/runs/baseline_run_001/metrics_summary.csv) resolves all apparent numerical inconsistencies:

| Metric | Recorded Value | Sample Count ($N$) | Denominator / Conditioning Context | Reconciliation Explanation |
| :--- | :---: | :---: | :--- | :--- |
| **M1: Task Success Rate** | `0.1000` | 10 | $1 / 10$ tasks succeeded | 1 task (`task_val_T2_vendor_approved_spend_001`) succeeded by outputting \$0.00 without tool calls. |
| **M2: Tool Call Accuracy** | `0.0000` | 10 | $0 / 10$ valid tool calls | The model emitted natural language without `<\|START_ACTION\|>` delimiters. |
| **M3: Schema Validity** | `1.0000` | 10 | Vacuous truth ($N_{\text{calls}} = 0$) | Fraction of valid calls among calls made. When 0 calls are made, 0 errors occur. |
| **M4: Call Order Validity** | `1.0000` | 10 | Vacuous truth ($N_{\text{calls}} = 0$) | When 0 calls are made, 0 dependency ordering violations occur. |
| **M5: Step Efficiency** | `1.0000` | **1** | Conditioned strictly on success | **Crucial Invariant:** Step efficiency is computed ONLY over successful trajectories. Because only 1 task succeeded, $N = 1$. Evaluating efficiency on failed runs creates reward-hacking incentives. |
| **M6: Redundant Call Rate** | `0.0000` | 10 | Vacuous truth ($N_{\text{calls}} = 0$) | 0 duplicate tool calls made. |
| **M7: Loop & Stall Frequency**| `0.0000` | 10 | Vacuous truth ($N_{\text{calls}} = 0$) | 0 loops or repetitive calls. |

* **Audit Conclusion:** The sample count of 1 for M5 is mathematically and architecturally correct. Metrics M3, M4, M6, and M7 must never be reported without stating that total executed tool calls was 0.

---

### 2.4 Criterion 4: Validation Split Purity & Isolation
* **Validation Manifest:** [`configs/manifests/task_manifest_val.json`](../configs/manifests/task_manifest_val.json).
* **Sample Count:** Exactly 10 tasks (`task_val_T1_inventory_value_000` through `task_val_T2_vendor_approved_spend_009`).
* **Content Hash:** SHA-256 `51a4275e777dbe250c468af1a10e12cd221e47bd563a3b79d1b4619b25278ea8`.
* **Entity Separation:** Queries strictly category `'Software'` and vendors `'VND-106'` and `'VND-107'`.
* **Leakage Verification:** Zero overlap with `train` (Hardware, `VND-100`–`VND-105`) or locked `test` (Services/Logistics, `VND-108`–`VND-111`).

---

### 2.5 Criterion 5: Evaluator Mode Verification
* **Native AuditKIT Audit:** `import auditkit` raises `ModuleNotFoundError`.
* **Adapter Audit:** [`src/utils/compatibility.py`](../src/utils/compatibility.py) executes the local evaluation spine.
* **Mandatory Reporting Invariant:** Evaluator mode is strictly recorded as:
  $$\text{Evaluator Mode} = \text{"local-adapter-1.0.0"}$$
  Native AuditKIT execution must not be claimed.

---

### 2.6 Criterion 6: Runtime Loading Path & Remediation Roadmap
* **Current Limitation:** `ModelGateway` only supports HTTP endpoints (OpenRouter, NVIDIA, Groq). Local full-scale 3.35B inference requires `transformers`, `peft`, and downloaded weights (~6.7 GB).
* **Remediation Roadmap for Gate G2 / Milestone 3:**
  1. Provision GPU compute instance (1x NVIDIA A100-SXM4-40GB or Apple Silicon MPS with full dependencies).
  2. Install training dependencies: `pip install -e ".[training]"`.
  3. Execute full Aya SFT pilot on the verified `data/sft/` dataset.
  4. Serve fine-tuned candidate via local vLLM or Hugging Face pipeline.
  5. Run paired evaluation against unadapted `CohereLabs/tiny-aya-global` baseline.


# Model Card: `sft_aya_3b_001` (CohereLabs/tiny-aya-global 3.35B SFT Candidate)

> **Model Identifier:** `CohereLabs/tiny-aya-global-sft_aya_3b_001`
> **Base Architecture:** `CohereLabs/tiny-aya-global` (3.35B parameters, Command-R architecture, revision `main`)
> **Adapter Checkpoint:** `results/runs/sft_aya_3b_001/checkpoint/adapter_model.safetensors`
> **Adapter Checksum (SHA-256):** `a73583f3aade1026ec9d5d53813d2ff9ffa66d0a84a17ae07ef06e39000b2a8d`
> **Release Status:** **RESEARCH-ONLY ARTIFACT — INELIGIBLE FOR PRODUCTION RELEASE (GATE G4 PENDING / NO-GO)**
> **Date:** 2026-11-06
> **Governing Specifications:** Milestone 4 ([`docs/M4.md`](../docs/M4.md)), [`docs/M4_P4_candidate_decision.md`](../docs/M4_P4_candidate_decision.md), [`results/runs/m4_val_sft_aya_3b_001/`](../results/runs/m4_val_sft_aya_3b_001/)

---

## 1. Model Summary & Critical Release Status

`sft_aya_3b_001` is an exploratory 4-bit QLoRA low-rank adapter trained on `CohereLabs/tiny-aya-global` (3.35B) to investigate whether supervised fine-tuning on verified multi-turn tool trajectories can induce structured tool invocation in open multilingual weights.

### Binding Capability & Release Determination:
- **Capability Floor Verdict:** **INELIGIBLE FOR PROMOTION (M1 FLOOR UNMET)**.
- **Observed Task Success:** **$M_1 = 0/10$ ($0.0\%$)** on held-out validation tasks ([`results/runs/m4_val_sft_aya_3b_001/`](../results/runs/m4_val_sft_aya_3b_001/)).
- **Observed Tool Selection:** **$M_2 = 0/10$ ($0.0\%$)** tool selection accuracy.
- **Failure Taxonomy Diagnosis:** **10/10 tasks (100.0%) diagnosed as `EarlyTerminationFailure`**.
- **Release Gate G4:** **PENDING / PRODUCTION PROMOTION BLOCKED**.
- **Locked-Test Status:** **FORMALLY DEFERRED**. The sequestered locked-test split (`task_manifest_test.json`, SHA-256 `d86265f0...`) was **not accessed, read, or evaluated**.

> [!WARNING]
> **NOT PRODUCTION READY:** This model must **NOT** be deployed, published as an autonomous agent, or described as a capable tool-calling agent. It is packaged strictly as a reproducible research artifact documenting the empirical limits of small-scale SFT on conversational priors.

---

## 2. Intended Use & Scope

### 2.1 Permitted Research Uses
- Academic research and reproduction of post-training tool-calling failure modes on multilingual open weights.
- Empirical error analysis comparing token cross-entropy loss reduction against downstream multi-turn task success.
- Offline benchmarking of defensive runtime guards and AuditKIT diagnostic taxonomies.

### 2.2 Prohibited Uses
- Autonomous or semi-autonomous deployment in procurement, enterprise financial operations, or database modification.
- External redistribution of fine-tuned adapter weights without formal licensing approval under Cohere CC-BY-NC-4.0.
- Substitution for human auditing or decision-making.

---

## 3. Training Details

### 3.1 Training Configuration
- **Platform:** Kaggle GPU Backend (2× NVIDIA Tesla T4, 32GB total VRAM / 16GB per GPU).
- **Base Precision & Quantization:** 4-bit NormalFloat4 (NF4) with Double Quantization (`bnb_4bit_use_double_quant=True`), computation in `bfloat16`.
- **PEFT / LoRA Hyperparameters:**
  - Rank ($r$): `16`
  - Scaling ($\alpha$): `32`
  - Dropout: `0.05`
  - Target Projections: `q_proj`, `k_proj`, `v_proj`, `o_proj` (All attention projections adapted).
  - Trainable Parameters: `11,796,480` / `3,365,000,000` (0.35% parameter budget).
- **Optimization:**
  - Optimizer: `AdamW` (`8-bit paged`)
  - Learning Rate: `2.0e-5` with Cosine Decay schedule (Warmup ratio: `0.10`).
  - Total Steps: `200` optimization steps.
  - Effective Batch Size: `4` (Batch size 1 $\times$ 4 gradient accumulation steps).
  - Wall-Clock Runtime: `28.2 minutes` (`1,692 seconds`).
  - Training Loss Trajectory: Decreased monotonically from initial loss `1.124` to final loss `0.362` (Mean: `0.464`).

### 3.2 Training Data Provenance
- **Dataset:** `procurement_sft_train_v1` ([`data/sft/`](../data/sft/)).
- **Source Split:** Strictly `train` split ([`configs/manifests/task_manifest_train.json`](../configs/manifests/task_manifest_train.json)).
- **Dataset SHA-256:** `14f7de32aa13d3905a0a6f3dc6cd0330f4e52774a7fc37a7b1f96da5be9f4573`.
- **Volume:** 20 verified successful multi-turn trajectories (40 tool invocations).
- **Loss Masking Protocol:** Assistant turn loss masking (`-100` on system prompts, user queries, and tool observations; gradients computed exclusively on assistant action tokens and final numerical answers).

---

## 4. Evaluation Protocol & Empirical Benchmark Results

### 4.1 Evaluation Protocol
- **Evaluation Workload:** 10 held-out tasks from [`configs/manifests/task_manifest_val.json`](../configs/manifests/task_manifest_val.json).
- **Evaluator Identity:** AuditKIT Evaluation Engine (`local-adapter-1.0.0`).
- **Runtime Environment:** Relational Database Tool Sandbox with committed transactions.
- **Controlling Run Artifact:** [`results/runs/m4_val_sft_aya_3b_001/`](../results/runs/m4_val_sft_aya_3b_001/).

### 4.2 Benchmark Results (AuditKIT M1–M7)

| Metric ID | Metric Description | Optimization Direction | Predeclared Eligibility Floor | Observed Candidate Value | 95% Bootstrap CI | Outcome Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `M1_task_success_rate` | Verified Task Completion Rate | ▲ maximize | $\ge 40.0\%$ | **0.0%** (0 / 10) | $[0.0\%, 0.0\%]$ | **FAIL (Floor Unmet)** |
| `M2_tool_call_accuracy` | Tool Selection Accuracy | ▲ maximize | $\ge 50.0\%$ | **0.0%** (0 / 10) | N/A | **FAIL (Floor Unmet)** |
| `M3_schema_validity` | Argument Schema Validity | ▲ maximize | $100.0\%$ | **1.0000** | N/A | Vacuous (0 tool calls) |
| `M4_call_order_validity` | Dependency Order Validity | ▲ maximize | $100.0\%$ | **1.0000** | N/A | Vacuous (0 tool calls) |
| `M5_trajectory_step_efficiency` | Mean Successful Steps | ▼ minimize | Report steps | ***unmeasured*** | N/A | 0 successful trajectories |
| `M6_redundant_call_rate` | Redundant Call Frequency | ▼ minimize | $\le 10.0\%$ | **0.0000** | N/A | Vacuous (0 tool calls) |
| `M7_loop_and_stall_frequency` | Loop & Budget Stall Rate | ▼ minimize | $0.0\%$ | **0.0000** | N/A | Vacuous (0 tool calls) |

### 4.3 Root-Cause Failure Diagnosis
- **Failure Taxonomy Distribution:**
  - `EarlyTerminationFailure`: **10 / 10 (100.0%)**
  - All other failure classes (`ToolSelectionFailure`, `ArgumentSchemaFailure`, `DependencyOrderFailure`, `LoopingAndStallFailure`, `ToolRecoveryFailure`): **0 / 10 (0.0%)**
- **Observable Failure Mechanism:**
  When prompted with user queries under the official Command-R prompt format, the model generated conversational text responses, procedural advice, and SQL code snippets wrapped in `<|START_RESPONSE|>` tags rather than emitting structured action invocations `<|START_ACTION|>{"name": ...}<|END_ACTION|>`.
- **Scientific Conclusion:**
  The pre-trained conversational priors in `CohereLabs/tiny-aya-global` (which was trained as a conversational instruction follower across 23 languages) overwhelmingly dominate over a small fine-tuning curriculum of 200 optimization steps. Low token cross-entropy loss ($0.36$) did not translate to format adherence in autoregressive inference.

### 4.4 Untuned Base Comparator Status (Historical M4)
- The initial M4 untuned base comparator was documented as `BLOCKED_NO_ROLLOUT` due to local execution constraints.

### 4.5 Milestone 5 Remediation Cycle 2 & Candidate Re-Evaluation
Under Milestone 5, remote execution on Kaggle 2×T4 GPUs completed full matched evaluations across three conditions with live multi-turn ReAct interactive rollouts:

| Candidate Condition | Model Architecture & LoRA Strategy | $M_1$ Task Success (95% CI) | $M_2$ Tool Accuracy | $M_3$ Schema Validity | Capability Floor Status |
|:---|:---|:---:|:---:|:---:|:---:|
| `val_base_aya_3b_matched` | Untuned Base `tiny-aya-global` (3.35B) | 0.0% [0.0%, 0.0%] | **50.0%** | 100.0% | Baseline Comparator Control |
| `sft_aya_3b_v2_c2` | Attention LoRA ($r=16, \alpha=32$) | 0.0% [0.0%, 0.0%] | 0.0% | 100.0% | **BELOW FLOOR** ($M_1 < 40\%, M_2 < 50\%$) |
| `sft_aya_3b_v2_c3` | All-Linear LoRA ($r=8, \alpha=16$) | **50.0%** [20.0%, 80.0%] | **0.0%** | 100.0% | **BELOW FLOOR** ($M_2 = 0.0\% < 50\%$) |

- **Forensic Discovery & Zero-Spend Shortcut:**
  Condition C3 nominally scored $M_1 = 50.0\%$, but achieved $M_2 = 0.0\%$ (zero tool calls). Detailed trace analysis revealed that C3 learned to guess `$0.00` for vendor queries without querying the database, accidentally matching the oracle for 5/5 zero-spend vendor tasks while failing 0/5 inventory valuation tasks.
- **Formal Release Verdict:**
  Under the predeclared release floor ($M_1 \ge 40\% \land M_2 \ge 50\%$), Condition C3 failed promotion. Milestone 5 concluded with a **documented NO-GO decision** ([`docs/M5_P6_research_decision.md`](../docs/M5_P6_research_decision.md)), and the sequestered test partition remains 100% sealed.

---

## 5. Hardware, Runtime & Serving Constraints

- **Local Serving Status:** **BLOCKED / UNAVAILABLE**.
- **Hardware Profile:**
  - Apple Silicon M5 (16GB unified memory): Incapable of running 4-bit CUDA `bitsandbytes` kernels; full bfloat16 loading requires ~7.0GB VRAM and unquantized PyTorch support not configured in local environment.
  - Remote Execution: Requires NVIDIA GPU with $\ge 16\text{GB}$ VRAM (e.g., Tesla T4, A100, RTX 4090) with CUDA 12.1+ and `bitsandbytes` 0.43+.
- **Harness Benchmark:** While model serving is blocked, the deterministic runtime harness introduces negligible execution overhead (~0.005 ms in-memory, ~0.19 ms SQLite) with 0.0% timeouts ([`results/runs/m4_p3_operational_profile/`](../results/runs/m4_p3_operational_profile/)).

---

## 6. Licensing, Access Terms & Attribution

- **Base Model License:** [Cohere CC-BY-NC-4.0](https://cohere.com/c4ai-cc-by-nc-license) with Cohere Acceptable Use Policy.
- **Model Gating:** Access is gated on Hugging Face Hub ([`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global)) requiring user click-through agreement and a valid `HF_TOKEN`.
- **Adapter Distribution:** **NOT CLEARED FOR PUBLIC REDISTRIBUTION**. Adapter weights are retained locally under research custody pending formal Gate G4 human review and licensing clearance.
- **Attribution:** Any academic reference to this architecture must cite Cohere For AI Aya research team and the Lexsi Track 3 engineering protocol.

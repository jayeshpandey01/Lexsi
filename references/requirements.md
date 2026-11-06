# Lexsi AI Safety & Alignment Hackathon: Track 3 Requirements Dossier

> **Track:** Track 3 — Agentic Fine-Tuning  
> **Official Event URL:** [https://hackathon.lexsi.ai](https://hackathon.lexsi.ai)  
> **Authoritative Traceability:** Referenced by [`.agents/skills/hackathon-submission/SKILL.md`](/Users/jayesh/Documents/Lexsi/.agents/skills/hackathon-submission/SKILL.md), [`docs/plan.md`](/Users/jayesh/Documents/Lexsi/docs/plan.md) Section 2.1, and [`docs/release_checklist.md`](/Users/jayesh/Documents/Lexsi/docs/release_checklist.md).  
> **Status:** Authoritative Submission Reference  

---

## 1. Executive Overview & Problem Statement

Track 3 challenges participants to post-train open-weights language models into reliable, safe, and efficient tool-using agents. The track focuses on two core problem statements:
1. **Reliable Tool Calling (Problem #1):** Eliminating invalid tool selections, malformed argument payloads, parameter type errors, and prerequisite dependency order violations.
2. **Trajectory Efficiency (Problem #2):** Minimizing unnecessary tool invocations, repetitive polling, and redundant environment steps without degrading task success.
3. **Optional Research Frontiers:** Reward-hacking defense (Problem #9), self-healing / failure recovery (Problem #8), or task distillation (Problem #3).

---

## 2. Core Toolchain & Model Eligibility

### 2.1 Official Toolchain
- **Training Framework:** [AgentTune](https://github.com/Lexsi-Labs/AgentTune) ([https://agenttune.lexsi.ai](https://agenttune.lexsi.ai)) — for Two-Tier EventLog SFT and verifiable outcome RL (GRPO/DPO).
- **Evaluation Framework:** [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) ([https://auditkit.lexsi.ai](https://auditkit.lexsi.ai)) — for benchmark metrics, bootstrap confidence intervals, and failure taxonomy diagnostics.
- **Repository Exemption:** Track 3 is **explicitly exempt** from the general hackathon two-repository rule while using the official AgentTune and AuditKIT pipeline.

### 2.2 Eligible Model Families
- **Required Provider:** Dense, open-weights LLMs from **Cohere Labs**.
- **Primary Pinned Model:** `CohereLabs/tiny-aya-global` (3.35B parameters, dense transformer decoder, CC-BY-NC-4.0 license, Command-R tool token delimiters).
- **Primary Base Model:** `CohereLabs/tiny-aya-base` (3.35B untuned base for pure behavior cloning).
- **High-Compute Scale Fallback:** `CohereForAI/aya-expanse-8b` (8.0B parameters).
- **Exploratory Diagnostic Baseline:** `cohere/command-r7b-12-2024` (used for initial diagnostic validation under `results/runs/baseline_run_001/`).

---

## 3. Official Timeline & Milestones

| Event Phase | Official Dates | Team Plan Window | Focus / Deliverable |
|:---|:---|:---|:---|
| **Build & Submission** | Nov 1 – Nov 14, 2026 (AoE) | D1 – D14 | Post-training, benchmark evaluation, artifact generation |
| **Judging Period** | Nov 15 – Nov 16, 2026 | D15 – D16 | Evaluation verification and safety audit |
| **Shortlist Proof Work** | Nov 17 – Nov 21, 2026 | D17 – D21 | Reproducibility verification and deep-dive audits |
| **Demo Day** | Nov 22, 2026 | D22 | Public pitch, live demonstration, and Q&A |
| **Project Horizon** | 30-Day Research Horizon | D31 – D60 | Comprehensive M5 research roadmap documented in [`docs/M5.md`](file:///Users/jayesh/Documents/Lexsi/docs/M5.md) |

*Note: Team size is limited to 1–3 members. At least one member must remain active on the official event Discord. The November 14, 2026 AoE deadline does NOT waive evaluation rigor, safety gates, licensing compliance, or the prohibition against unsupported claims.*

---

## 4. Mandatory Submission Deliverables

Every Track 3 submission must deliver the following six core components:

### 4.1 Public GitHub Repository
- Clean, production-grade source code with deterministic installation instructions.
- Fully documented CLI run scripts and automated verification commands (`pytest -v`).
- Zero hardcoded secrets, API tokens, or `.env` files committed to version control.
- Clear license declaration (Apache-2.0 for project code, noting dataset and base model terms).

### 4.2 Public Hugging Face Collection
- Public, ungated Hugging Face repositories containing:
  1. Fine-tuned model adapter weights (LoRA checkpoints).
  2. Curated SFT and RL training datasets (in ShareGPT and AgentTune EventLog formats).
  3. Machine-readable evaluation logs and AuditKIT run manifests.

### 4.3 Comprehensive Project README & Abstract
- **Project Abstract:** Executive abstract strictly under **500 words** summarizing problem, method, and empirical findings.
- **Before / After Comparison Table:** Direct side-by-side metrics comparing the untuned base model against SFT and RL candidates across primary metrics.
- **Methodology Writeup:** Rationale behind task design, prompt templates, reward formulation, and failure categorization.
- **Reproducibility Notes:** Exact shell commands to replicate benchmark results from a clean repository clone.

### 4.4 Model Card & Data Card
- **Model Card:** Documenting intended use, base model provenance, training hyperparameters, context budget, and known limitations.
- **Data Card:** Documenting data provenance ([`references/data_card.md`](/Users/jayesh/Documents/Lexsi/references/data_card.md)), split isolation contracts, PII redaction, and license compliance.

### 4.5 Before-and-After Video Demo / Demo Script
- Concrete visual demonstration showing:
  - Base model failure: Early termination, invalid schema, or trajectory looping.
  - Post-trained agent success: Optimal tool selection, correct argument formatting, and concise task completion in minimal steps.

### 4.6 (Optional Preferred) Commercial ROI Analysis
- Quantified cost and latency comparison estimating enterprise return-on-investment (ROI) of self-hosted open-weights agents versus commercial closed-model APIs (e.g. GPT-4o, Claude 3.5 Sonnet) across realistic enterprise query volumes.

---

## 5. Evaluation Metrics & Judging Dimensions

Track 3 judging evaluates models across four principal dimensions:

### 5.1 Primary Performance Metrics (AuditKIT Standard)
1. **Verified Task Success ($M_1$):** Fraction of tasks satisfying deterministic environment state predicates (`is_success == True`).
2. **Correct Tool Selection ($M_2$):** Accuracy of tool selection at each step where an action is required.
3. **Argument Schema Validity ($M_3$):** Fraction of tool calls strictly adhering to declared Pydantic argument schemas (`extra='forbid'`).
4. **Call-Order Validity ($M_4$):** Satisfaction of prerequisite tool ordering dependencies (e.g. `lookup_records` before `compute_aggregate`).
5. **Trajectory Step Efficiency ($M_5$):** Mean environment steps per task, specifically evaluated conditional on task success.
6. **Redundant Call Rate ($M_6$):** Percentage of trajectories containing duplicate or cyclic tool calls.
7. **Constraint Verification Accuracy ($M_7$):** Precision in evaluating mathematical inequality constraints against budgets.

### 5.2 Anti-Gaming & Evaluation Integrity Guards
- **Zero Test Set Leakage:** The locked test split ([`configs/manifests/task_manifest_test.json`](/Users/jayesh/Documents/Lexsi/configs/manifests/task_manifest_test.json)) must remain 100% sequestered during training, prompt engineering, and hyperparameter tuning.
- **Fail-Closed Exports:** Training data pipelines must fail closed via `LockedTestExportError` if test instances are touched.
- **Verified Reward Outcomes:** Efficiency penalties must only be credited on trajectories that achieve verified success, preventing "do-nothing" degeneracy.

---

## 6. Official Requirement-to-Evidence Matrix (M5 Audit Snapshot)

> **Audit Date:** 2026-11-07<br>
> **Controlling Standard:** [`docs/M5.md`](file:///Users/jayesh/Documents/Lexsi/docs/M5.md) §4.1<br>
> **Status Taxonomy:** `DONE` (Fully satisfied & verified) | `MISSING` (Work package planned or in progress) | `BLOCKED` (Gated by missing dependency, license, or human approval)

| Official Hackathon Requirement | Current Project Evidence | M5 Audit Status | Specific Blocker / Action Required |
|:---|:---|:---:|:---|
| **1. Dense Open-Weights Cohere Model & Track 3 Toolchain** | Cohere Tiny Aya 3.35B SFT artifact (`sft_aya_3b_001`); AgentTune/AuditKIT integration in `src/utils/compatibility.py` | **DONE** | Model and pipeline meet Track 3 eligibility rules. |
| **2. Public Repository / Demo** | Local git repository clean; `scripts/run_demo.py` exercises deterministic harness, sandbox, and guards | **DONE (Local)** / **GATED (Public)** | Public GitHub publication requires authorized human sign-off; demo is harness-only (model serving blocked locally). |
| **3. Multiple Before / After Metrics** | SFT candidate has validation metrics ($M_1=0/10$); historical Command R baseline (`baseline_run_001`) is incommensurable | **MISSING / BLOCKED** | Matched untuned `tiny-aya-global` base rollout is blocked locally (no weights/libs); requires Kaggle 2×T4 rollout in M5.P4. No causal claims permitted until completed. |
| **4. Demonstrated Capability Improvement** | M4 candidate `sft_aya_3b_001` achieved $M_1=0.0\%$, $M_2=0.0\%$, 10/10 `EarlyTerminationFailure` | **MISSING** | Capability remediation in progress (M5.P2–P4). If candidate fails floors ($M_1 \ge 40\%$), non-submission or failure disclosure is mandatory. |
| **5. Public Hugging Face Collections** | LoRA adapter (`a73583f...`) and SFT dataset retained locally; CC-BY-NC-4.0 review documented in `docs/license_review.md` | **BLOCKED / GATED** | Remote HF upload requires formal leadership authorization and repository setup. |
| **6. Executive Abstract (< 500 words)** | Abstract draft outline exists in `references/model_card.md` | **MISSING** | Word-counted final abstract will be finalized in M5.P5 after candidate evaluation. |
| **7. Clean Reproduction from Fresh Clone** | README reproduction notes and unit tests (`pytest -q`, 166 passing) verified; rollback smoke test archived | **MISSING** | Formal fresh-clone test script (`scripts/verify_reproducibility.sh`) to be executed under M5.P5. |
| **8. Zero Test Leakage / Integrity Guards** | Locked test (`configs/manifests/task_manifest_test.json`, SHA-256 `d86265f0...`) permanently sequestered; `LockedTestExportError` active | **DONE** | Verified 100% compliant; test split sealed. |

---

## 7. Pre-Submission Verification Checklist

- [ ] All 7 core metrics evaluated and reported with 95% bootstrap confidence intervals.
- [ ] Matched base model baseline and post-trained candidate evaluated on identical validation splits.
- [ ] Training datasets in `data/sft/` pass split isolation audit with 0% test leakage.
- [ ] Repository passes 100% of automated unit tests (`pytest -v`).
- [ ] Hugging Face repository URLs verified and public (upon authorized approval).
- [ ] No secrets or unredacted keys present in git history or artifacts.


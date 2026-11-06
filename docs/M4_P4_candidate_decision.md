# Milestone 4 Work Package M4.P4: Candidate Decision & Locked-Test Release Benchmark Authorization

> **Document Type:** Governance, Evaluation Gate & Release Benchmark Decision Record  
> **Milestone:** Milestone 4 (Work Package M4.P4 / Plan Item P4.2)  
> **Project:** Reliable and Efficient Tool-Using Agent (Lexsi Track 3)  
> **Evaluation & Release Date:** 2026-11-06  
> **Reviewers:** Project Leadership, Senior Principal Research Scientist & Evaluation Lead  
> **Status:** **COMPLETED (FORMAL DECISION: DEFER_LOCKED_TEST — LOCKED TEST SPLIT REMAINS UNACCESSED & SEALED)**  
> **Controlling References:** [`docs/M4.md`](file:///Users/jayesh/Documents/Lexsi/docs/M4.md) §6, [`docs/plan.md`](file:///Users/jayesh/Documents/Lexsi/docs/plan.md) §4, [`docs/release_checklist.md`](file:///Users/jayesh/Documents/Lexsi/docs/release_checklist.md) §2, [`results/runs/m4_val_sft_aya_3b_001/run_manifest.json`](file:///Users/jayesh/Documents/Lexsi/results/runs/m4_val_sft_aya_3b_001/run_manifest.json)  

---

## 1. Executive Summary & Binding Decision

In strict accordance with the Release Gate G4 protocol, Milestone 4 work package `M4.P4`, and the research integrity rules governing held-out evaluation:

$$\mathbf{BINDING\ DECISION:\ DEFER\_LOCKED\_TEST}$$

### Key Decision Determinations:
1. **Candidate Ineligibility:** The current fine-tuned model candidate (`sft_aya_3b_001`, `CohereLabs/tiny-aya-global` 3.35B) achieved **$M_1 = 0.0\%$ (0/10)** verified task success and **$M_2 = 0.0\%$ (0/10)** tool selection accuracy during the official AuditKIT validation evaluation ([`results/runs/m4_val_sft_aya_3b_001/`](file:///Users/jayesh/Documents/Lexsi/results/runs/m4_val_sft_aya_3b_001/)). It decisively fails the pre-registered capability floor ($M_1 \ge 40.0\%$, $M_2 \ge 50.0\%$).
2. **Zero Locked-Test Consumption:** The permanently sequestered locked-test split (`configs/manifests/task_manifest_test.json`, SHA-256 `d86265f067bafa112427390659d55ab48fe6831d68b88efb6952b8fd377c3f3e`) was **NOT accessed, read, scored, or inspected**. Executing the single-use locked test on a candidate that failed basic validation would permanently contaminate the benchmark without supporting a promotion decision.
3. **Preservation of Benchmark Integrity:** The locked test split is permanently preserved in its virgin, sealed state for future candidate architectures that satisfy the entry capability criteria.
4. **Gate G4 Status:** Release Gate G4 remains **PENDING**; candidate promotion is blocked. Work package `M4.P4` is formally marked complete for the documented deferral decision. The project proceeds directly to **M4.P5 (Packaging Research Artifacts, Documentation, and Public Demo)**.

---

## 2. Candidate Eligibility Verification Audit

The candidate decision is grounded directly in empirical evidence across earlier work packages:

### 2.1 Capability Thresholds vs. Observed Metrics (M4.P1)
- **Controlling Run:** [`results/runs/m4_val_sft_aya_3b_001/`](file:///Users/jayesh/Documents/Lexsi/results/runs/m4_val_sft_aya_3b_001/)
- **Model Checkpoint:** `results/runs/sft_aya_3b_001/checkpoint/adapter_model.safetensors` (30.7 MB, SHA-256: `a73583f3aade1026ec9d5d53813d2ff9ffa66d0a84a17ae07ef06e39000b2a8d`)
- **Evaluation Workload:** 10 held-out tasks from `configs/manifests/task_manifest_val.json` (SHA-256: `51a4275e777dbe250c468af1a10e12cd221e47bd563a3b79d1b4619b25278ea8`)

| Metric | Predeclared Eligibility Floor | Observed Candidate Metric | 95% Confidence Interval | Outcome Verdict |
|:---|:---:|:---:|:---:|:---:|
| **$M_1$ Verified Task Success** | **$\ge 40.0\%$** | **0.0%** (0 / 10) | $[0.0\%, 0.0\%]$ | **FAIL (Floor Unmet)** |
| **$M_2$ Tool Selection Accuracy** | **$\ge 50.0\%$** | **0.0%** (0 / 10) | $[0.0\%, 0.0\%]$ | **FAIL (Floor Unmet)** |
| **$M_3$ Argument Schema Validity** | $100.0\%$ | $1.0000$ (Vacuous) | — | Vacuous (0 tool calls) |
| **$M_4$ Call-Order Validity** | $100.0\%$ | $1.0000$ (Vacuous) | — | Vacuous (0 tool calls) |
| **$M_5$ Trajectory Efficiency** | Report steps | **Unmeasured** | — | 0 successful trajectories |
| **$M_6$ Redundant Call Rate** | $\le 10.0\%$ | $0.0000$ (Vacuous) | — | Vacuous (0 tool calls) |
| **$M_7$ Loop / Stall Frequency** | $0.0\%$ | $0.0000$ (Vacuous) | — | Vacuous (0 tool calls) |

### 2.2 Root-Cause Failure Diagnosis
- **Taxonomy Result:** 10/10 tasks (100.0%) were diagnosed as **`EarlyTerminationFailure`** (Confidence: 1.0).
- **Behavioral Evidence:** When presented with user queries, the model emitted natural language conversational explanations and SQL code blocks wrapped in `<|START_RESPONSE|>` rather than invoking the registered tools using `<|START_ACTION|>{"name": ...}<|END_ACTION|>`.
- **Underlying Root Cause:** Pre-trained conversational priors in `CohereLabs/tiny-aya-global` dominate. While 200 optimization steps reduced token cross-entropy loss from 1.12 to 0.36, it was insufficient to instill reliable multi-turn tool invocation without broader instruction datasets.

### 2.3 Untuned Base Comparator Status
- The untuned base model rollout was documented as **`BLOCKED_NO_ROLLOUT`** ([`results/runs/m4_val_base_aya_3b_blocked/run_manifest.json`](file:///Users/jayesh/Documents/Lexsi/results/runs/m4_val_base_aya_3b_blocked/run_manifest.json)).
- No causal comparative superiority claims over the base model are asserted.

---

## 3. Separation of Orthogonal Evidence (P2 & P3)

In accordance with Section 1 of the M4.P4 instructions, passing results in safety and operational harness profiling do **not** override the candidate model capability floor:

1. **M4.P2 Robustness & Safety Review (33/33 Probes Passed):**
   - The adversarial review ([`docs/M4_P2_robustness_report.md`](file:///Users/jayesh/Documents/Lexsi/docs/M4_P2_robustness_report.md)) proved that the deterministic runtime perimeter (`RuntimePolicyGuard`, `SandboxedToolEngine`, `parse_agent_output`) successfully intercepts prompt injections, malformed tokens, disallowed tools, and schema violations fail-closed with 0 guard bypasses.
   - However, defensive containment does not confer generative task completion ability. A safe agent that refuses to call tools remains incapable of completing tasks.
2. **M4.P3 Operational Performance Profile:**
   - The operational profile ([`docs/M4.md`](file:///Users/jayesh/Documents/Lexsi/docs/M4.md), [`results/runs/m4_p3_operational_profile/`](file:///Users/jayesh/Documents/Lexsi/results/runs/m4_p3_operational_profile/)) verified that the in-memory harness introduces negligible latency (~0.016 ms) with 0.0% timeouts, and that remote training completed in 28.2 minutes on 2× T4 GPUs.
   - However, local model generation serving is blocked due to missing weights/libraries, and remote validation confirms the model cannot serve as an autonomous tool-calling policy.
3. **Binding Rule:** System safety and harness efficiency are necessary but insufficient for model release. Model task capability ($M_1 \ge 40\%$) is a non-negotiable prerequisite.

---

## 4. Locked-Test Sequestration & Access Control Verification

The locked-test split remains completely sealed under strict governance controls:

| Security Property | Registered Specification | Audit Verification |
|:---|:---|:---:|
| **File Location** | `configs/manifests/task_manifest_test.json` | Pinned repository path |
| **Access Control Policy** | Single-use confirmatory release benchmark only | **Enforced (Zero Access)** |
| **Cryptographic Digest** | `d86265f067bafa112427390659d55ab48fe6831d68b88efb6952b8fd377c3f3e` | Pre-recorded audit hash verified |
| **Inspection Audit** | File was NOT read, opened, scored, or hashed by re-reading | **CONFIRMED SEALED** |
| **Benchmark Status** | Locked-test evaluation execution | **FORMALLY DEFERRED** |

---

## 5. Conditions Required for Revisiting Locked-Test Access

Access to the sequestered locked-test manifest will be reconsidered if and only if all of the following conditions are met:

1. **Post-Training Remediation:** A newly trained candidate (e.g., trained on an expanded enterprise-scale dataset with $\ge 200$ balanced multi-turn trajectories) demonstrates format compliance and active tool selection.
2. **Pre-Release Validation Floor Cleared:** The candidate achieves verified validation metrics exceeding the pre-registered floors on `configs/manifests/task_manifest_val.json`:
   - $M_1 \ge 40.0\%$ (Verified Task Success Rate)
   - $M_2 \ge 50.0\%$ (Tool Call Selection Accuracy)
   - $M_3 = 100.0\%$ (Argument Schema Validity)
   - $M_7 = 0.0\%$ (Zero Critical Policy/Safety Violations)
3. **Controlled Base Comparator Available:** An untuned base model rollout under identical prompts and decoding parameters is completed to enable causal comparative evaluation.
4. **Explicit Human Release Authorization:** A signed release protocol from the Evaluation Lead and Safety Lead formally unseals the locked test for a single confirmatory execution.

---

## 6. Work Package Tracking and Status Sign-Off

- **Work Package M4.P4 Status:** **COMPLETED (FORMAL DEFERRAL DECISION RECORDED)**
- **Release Benchmark Execution:** **DEFERRED (NO TEST RUN)**
- **Gate G4 Status:** **PENDING / BLOCKED FROM RELEASE**
- **Next Work Package:** Proceed to **M4.P5 (Package research artifacts, documentation, and demo)**.


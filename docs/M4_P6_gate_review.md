# Milestone 4 Work Package M4.P6: Gate G4 Release Review Decision Record

> **Document Type:** Gate G4 Release Review and Closeout Decision Record
> **Milestone:** Milestone 4 (Work Package M4.P6 / Plan Item P5.4)
> **Project:** Reliable and Efficient Tool-Using Agent (Lexsi Track 3)
> **Review Date:** 2026-11-07
> **Status:** **PREPARED — PENDING HUMAN REVIEWER SIGNATURES (NOT YET AUTHORIZED)**
> **Recommended Outcome:** **NO-GO** (Candidate ineligible; production promotion blocked)
> **Controlling References:** [`docs/M4.md`](M4.md) §8, [`docs/plan.md`](plan.md) §4, [`docs/release_checklist.md`](release_checklist.md), [`docs/M4_P4_candidate_decision.md`](M4_P4_candidate_decision.md), [`docs/M4_P5_packaging_report.md`](M4_P5_packaging_report.md), [`docs/M4_P2_robustness_report.md`](M4_P2_robustness_report.md)

> [!IMPORTANT]
> This record is **prepared but unsigned**. No authorized human reviewers have yet completed their review or provided signatures. The recommended outcome is **NO-GO** based on verified evidence. This document does not constitute a final release decision. No public release, production deployment, or external distribution is authorized until named, authorized reviewers formally sign off.

---

## 1. Review Scope and Evidence Examined

### 1.1 Candidate

| Field | Value |
|:---|:---|
| **Candidate identifier** | `sft_aya_3b_001` |
| **Base model** | `CohereLabs/tiny-aya-global` (3.35B parameters, Command-R architecture, revision `main`) |
| **Adapter checkpoint** | `results/runs/sft_aya_3b_001/checkpoint/adapter_model.safetensors` |
| **Adapter SHA-256** | `a73583f3aade1026ec9d5d53813d2ff9ffa66d0a84a17ae07ef06e39000b2a8d` (30.7 MB) |
| **Training method** | QLoRA SFT (LoRA r=16, α=32; 4-bit NF4; 200 steps on Kaggle 2× T4) |
| **Evaluation split** | `val` — `configs/manifests/task_manifest_val.json` (SHA-256: `51a4275e...`) |

### 1.2 Documents and Artifacts Examined

| Source | Path / Run ID | Verification status |
|:---|:---|:---|
| M4 milestone plan | `docs/M4.md` | Examined in full |
| Project tracker | `docs/plan.md` | Examined in full |
| Release checklist | `docs/release_checklist.md` | Examined in full |
| M3 handoff report | `docs/M3_report.md` | Examined §1–§2 and evidence matrix |
| M4.P4 candidate decision | `docs/M4_P4_candidate_decision.md` | Examined in full |
| M4.P5 packaging report | `docs/M4_P5_packaging_report.md` | Examined in full |
| M4.P2 robustness report | `docs/M4_P2_robustness_report.md` | Examined in full |
| M4.P1 validation run manifest | `results/runs/m4_val_sft_aya_3b_001/run_manifest.json` | Examined in full |
| Base comparator blocked manifest | `results/runs/m4_val_base_aya_3b_blocked/run_manifest.json` | Examined in full |
| Operational profile report | `results/runs/m4_p3_operational_profile/operational_profile_report.md` | Examined §1–§2 |
| Model card | `references/model_card.md` | Examined §1–§2 |
| Checkpoint log | `docs/M4.md` §10 | Examined in full |

> [!NOTE]
> The locked-test manifest (`configs/manifests/task_manifest_test.json`, SHA-256 `d86265f067bafa112427390659d55ab48fe6831d68b88efb6952b8fd377c3f3e`) was **not opened, read, inspected, or hashed** during this review. Its sealed status is verified by pre-recorded digest only.

---

## 2. Cross-Document Consistency Verification

### 2.1 Discrepancy Assessment

The following potential conflicts were reviewed; none required resolution because examination revealed consistent reporting across controlling sources:

| Claim | M4.md checkpoint log | plan.md tracker | M4_P4_candidate_decision.md | M4_P5_packaging_report.md | run_manifest.json | Resolution |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| M1 = 0.0% (0/10) | ✓ | ✓ | ✓ | ✓ | ✓ (m1_observed: 0.0) | Consistent across all sources |
| M2 = 0.0% (0/10) | ✓ | ✓ | ✓ | ✓ | ✓ (m2_observed: 0.0) | Consistent across all sources |
| 10/10 EarlyTerminationFailure | ✓ | ✓ | ✓ | ✓ | ✓ (failure_taxonomy_distribution) | Consistent across all sources |
| Adapter SHA-256 `a73583f3...` | ✓ | — | ✓ | ✓ | ✓ | Consistent; plan.md does not repeat hash (by design) |
| Locked-test SHA-256 `d86265f0...` | ✓ | ✓ (P4.2) | ✓ | ✓ | — | Consistent across all sources recording this hash |
| Base comparator BLOCKED | ✓ | ✓ | ✓ | ✓ | ✓ (BLOCKED_NO_ROLLOUT) | Consistent across all sources |
| Gate G4 PENDING | ✓ | ✓ | ✓ | ✓ | — | Consistent |
| External distribution BLOCKED | ✓ (§7) | ✓ (P5.3) | — | ✓ | — | Consistent |

**Finding:** No unresolved discrepancy was identified. The M4.md §10 checkpoint log, plan.md tracker, all decision records, the packaging report, and the run manifest are internally consistent on all material claims.

### 2.2 M3 Handoff Consistency

The M3 report (`docs/M3_report.md`) is cited in M4.P0 as having been reconciled against run artifacts; Section 2 of the M4.md milestone notes that M4.P0 resolved internal M3.P5 vs M3.P6 contradictions and confirmed the real 30.7 MB adapter, the M1=0.0% result, and Gate G3 NO-GO/SFT-ONLY. The M3 report §1 confirms the same adapter SHA-256 (`a73583f3...`) and identical validation results as the M4.P1 run manifest. The M4.P0 reconciliation is treated as the controlling resolution; no further M3 conflict requiring action at the G4 stage was found.

---

## 3. Gate G4 Criterion-by-Criterion Assessment

### Criterion 1: M3 Handoff and Provenance

| Sub-criterion | Status | Evidence |
|:---|:---:|:---|
| M3 report/tracker conflicts reconciled | **PASS** | M4.P0 reconciled internal contradictions; `docs/M3_report.md` §1 and §2 confirm real-scale 30.7 MB adapter, M1=0.0%, Gate G3 NO-GO/SFT-ONLY |
| SFT-only scope recorded | **PASS** | M4.md §2–§3; plan.md P3.6; M3_report.md §1 |
| Locked-test remained sealed through M3 | **PASS** | M3_report.md evidence matrix: SHA-256 `d86265f0...` noted as SEQUESTERED |
| `baseline_run_001` correctly characterised as exploratory / mismatched | **PASS** | m4_val_base_aya_3b_blocked/run_manifest.json confirms `baseline_run_001` is a mismatched architecture (Command R 7B); M4.md §2 rule 1 |

**Overall:** **PASS**

---

### Criterion 2: Validation Evidence and Limitations

| Sub-criterion | Status | Evidence |
|:---|:---:|:---|
| Validation run executed on correct split with correct candidate | **PASS** | `m4_val_sft_aya_3b_001/run_manifest.json`: split=val, manifest SHA-256 `51a4275e...`, adapter SHA-256 `a73583f3...` |
| M1 and M2 reported with counts and denominators | **PASS** | M1=0/10 (0.0%), M2=0/10 (0.0%); denominator N=10 stated in all sources |
| Failure classification documented | **PASS** | 10/10 `EarlyTerminationFailure` (confidence 1.0); per-task diagnosis in run_manifest.json with raw output snippets |
| Vacuous metrics correctly labelled | **PASS** | M3–M7 vacuous where no tool calls occurred; `M4_P4_candidate_decision.md` §2.1 distinguishes vacuous from meaningful zeros |
| Exploratory sample size caveat stated | **PASS** | N=10 validation tasks; multiple documents note that this is a small sample and results are exploratory |
| No overclaiming of SFT improvement | **PASS** | Base comparator blocked; no causal before/after claim made; sources consistently state comparison is unfabricated |
| Candidate fails eligibility floor | **PASS** | M1=0.0% < floor 40%; M2=0.0% < floor 50% |

**Overall:** **PASS** (evidence honestly reported; no overclaiming)

---

### Criterion 3: Robustness and Security Findings

| Sub-criterion | Status | Evidence |
|:---|:---:|:---|
| Adversarial probes executed with documented scope | **PASS** | 33 probes across 6 threat vectors (V1–V6); `docs/M4_P2_robustness_report.md` §2–§3 |
| Zero unresolved critical guard bypasses | **PASS** | 0 guard bypasses observed; `robustness_summary.csv`, `probe_results.json` archived |
| Scope accurately characterised (component-level, not full model red-teaming) | **PASS** | M4_P2_robustness_report.md §1 explicitly states: "direct component and harness integration tests… NOT end-to-end model-in-the-loop red teaming"; `sft_aya_3b_001` was not dynamically prompted during probes |
| Probes do not equate component safety with model capability or complete production security | **PASS** | M4_P4_candidate_decision.md §3 explicitly separates safety perimeter from generative task capability; M4_P2_robustness_report.md §1 states scope limitations |

> [!NOTE]
> The 33/33 probe pass result applies to the deterministic outer safety perimeter (parser, policy guard, sandboxed executor) evaluated against 6 declared threat vectors. It does not constitute a proof of safety against all arbitrary novel adversarial inputs, nor does it confirm model task capability.

**Overall:** **PASS** (within stated scope; scope accurately and honestly characterised)

---

### Criterion 4: Operational Constraints

| Sub-criterion | Status | Evidence |
|:---|:---:|:---|
| Training profile documented | **PASS** | 28.2 min on Kaggle 2× T4; 200 steps; mean loss 0.4642; operational_profile_report.md §2.1 |
| Local model serving documented as BLOCKED | **PASS** | Consistently stated: missing transformers/peft libs, missing 3.35B weights (~6.7 GB); `m4_p3_operational_profile/`; model_card.md; M4_P5_packaging_report.md §6.3 |
| Harness latency measured with correct scope | **PASS** | In-memory: ~0.005 ms; SQLite: ~0.19 ms; explicitly labelled as isolated deterministic harness timing, not model serving or production throughput |
| No unsupported production throughput extrapolation | **PASS** | Operational profile clearly separates the three domains and makes no production throughput claims |

**Overall:** **PASS**

---

### Criterion 5: Locked-Test Disposition

| Sub-criterion | Status | Evidence |
|:---|:---:|:---|
| Formal deferral decision recorded | **PASS** | `docs/M4_P4_candidate_decision.md` §1: BINDING DECISION: DEFER_LOCKED_TEST |
| Locked-test manifest not opened, read, inspected, or scored | **PASS** | Confirmed by pre-recorded SHA-256 `d86265f067bafa112427390659d55ab48fe6831d68b88efb6952b8fd377c3f3e`; all reports state CONFIRMED SEALED |
| Deferral rationale correctly grounded in eligibility failure | **PASS** | Deferral caused by M1=0.0% < 40% floor; M2=0.0% < 50% floor; stated in M4_P4_candidate_decision.md §1 |
| Split preserved for future eligible candidate | **PASS** | M4_P4_candidate_decision.md §3 and §5 specify conditions for future access |
| No test run performed | **PASS** | Zero locked-test tasks were executed; no locked-test result was produced |

**Disposition:** **DEFERRED — Split preserved sealed; no test run performed.**

**Overall:** **PASS**

---

### Criterion 6: Reproducibility, Redaction, and License Status

| Sub-criterion | Status | Evidence |
|:---|:---:|:---|
| Root README with reproduction instructions | **PASS** | `README.md` documents exact commands, pinned versions, hashes |
| Model card completed | **PASS** | `references/model_card.md`: adapter SHA-256, base revision, training config, 0/10 result, failure taxonomy, blocked serving, CC-BY-NC-4.0 terms |
| Data card completed | **PASS** | `references/data_card.md`: dataset SHA-256 (`14f7de32...`), split policy, CC BY-NC-SA 4.0 terms, train-only restrictions |
| Cryptographic checksums recorded | **PASS** | M4_P5_packaging_report.md §4: 8 artifact digests tabulated |
| Secrets and credentials redacted | **PASS** | `.gitignore` excludes `.env`, `*.pem`, `*.key`; `src/data/redaction.py` verified; zero staged credentials |
| Binary artifacts excluded from Git | **PASS** | `*.safetensors`, `*.db`, `*.bin`, `*.zip` excluded via `.gitignore` |
| External publication **BLOCKED** (license not cleared) | **PASS** | Base model: Cohere CC-BY-NC-4.0 (Acceptable Use Policy; gated HF repo). Dataset: CC BY-NC-SA 4.0 (non-commercial only). Neither adapter nor auxiliary dataset is cleared for public redistribution. Human Gate G4 sign-off pending. |
| Local research package retained within permitted scope | **PASS** | Package retained locally; no external upload executed |

**Overall:** **PASS** (local package only; external distribution not authorized)

---

### Criterion 7: Rollback

| Sub-criterion | Status | Evidence |
|:---|:---:|:---|
| Rollback procedure documented | **PASS** (documented) | `plan.md` §3.5: "retain the prior known-good model and a rollback procedure"; P5.1 notes "rollback guide documented for research package"; `plan.md` §6.2 checklist item: "Human override, safe fallback, rollback, and incident owner are documented" |
| Rollback empirically tested | **PASS** (verified) | Executed via `scripts/run_rollback_smoke_test.py`; verified in `results/runs/rollback_smoke_001/` (`rollback_report.md`, `run_manifest.json`, `metrics_summary.csv`). Atomic config switch, baseline M1 restoration (10.0%), and 100% config restoration verified. |

**Overall:** **PASS**

---

### Criterion 8: Owner and Release Scope

| Sub-criterion | Status | Evidence |
|:---|:---:|:---|
| Named owner roles exist | **PASS** | Roles defined in `plan.md` §7.1 (Product/Research Lead, ML/Training Lead, Evaluation/Safety Lead, Infra/Release Lead) |
| Named owner individuals assigned | **PASS** | Individual owner (Jayesh) assigned across all four roles in `plan.md` §7.1 |
| Release scope appropriate to evidence | **PASS** | Scope restricted to local research package; production and external distribution explicitly blocked |
| Post-release monitoring owner | **NOT VERIFIED** | P6.1 (operations and monitoring) is Deferred; no operations pilot authorized |

**Overall:** **PASS** (for local research scope; post-release operations remain deferred)

---

### Criterion 9: External Distribution and License Authorization

| Item | Status | Detail |
|:---|:---:|:---|
| Adapter weights — public upload | **BLOCKED** | Cohere CC-BY-NC-4.0 license access terms unreviewed; Gate G4 not signed |
| Dataset (Olist auxiliary) — public upload | **BLOCKED** | CC BY-NC-SA 4.0 (non-commercial only); not cleared for redistribution |
| Locked-test split — any distribution | **BLOCKED** | Sealed; deferred; must remain under custody |
| Hugging Face public collection publish | **BLOCKED** | Unauthorized pending license, provenance, privacy, and human reviews |
| Local research package use | **PERMITTED** | Within documented research scope; zero external side-effects |

**Overall:** **BLOCKED / NOT AUTHORIZED**

---

## 4. Recommended Outcome and Rationale

### 4.1 Recommendation

$$\mathbf{RECOMMENDED\ GATE\ G4\ OUTCOME:\ NO\text{-}GO}$$

The candidate `sft_aya_3b_001` is **not eligible for production promotion or limited pilot deployment**.

### 4.2 Rationale

1. **Primary capability failure:** M1 Verified Task Success = 0/10 (0.0%), against a predeclared floor of ≥ 40%. M2 Tool Selection Accuracy = 0/10 (0.0%), against a predeclared floor of ≥ 50%. 10/10 tasks failed with `EarlyTerminationFailure`: the model emitted natural language prose and SQL text rather than structured tool actions. This is a decisive, non-trivial capability failure.
2. **No matched comparator:** The untuned base model rollout on the validation set is BLOCKED; no causal SFT gain over the base model can be asserted.
3. **Rollback verified:** The rollback procedure was empirically verified via `scripts/run_rollback_smoke_test.py` (`results/runs/rollback_smoke_001/`); however, capability failure alone mandates NO-GO.
4. **Ownership assigned for research:** Named owner (Jayesh) assigned in `plan.md` §7.1, but operational pilot deployment remains unauthorized.
5. **Locked-test deferred:** The confirmatory locked-test benchmark was correctly deferred; running it on a non-qualifying candidate would permanently contaminate the split without supporting any promotion decision.
6. **License and distribution blocked:** Neither the adapter nor the dataset is cleared for public redistribution. External publication is unauthorized pending applicable reviews.
7. **Positive factors (do not change outcome):** The runtime safety perimeter passed all 33 adversarial probes (0 bypasses); the operational profile is honest and within bounds; the research package is reproducible and correctly redacted; rollback is verified; and all evidence is internally consistent.

### 4.3 Local Research Package Disposition

The local research package — engineering infrastructure, evaluation spine, model card, data card, demo harness, and documentation — is retained within its permitted local research scope. It is **not** distributed externally. Its artifacts are available for authorized team members to reproduce and examine.

---

## 5. Locked-Test Disposition

| Property | Value |
|:---|:---|
| **Decision** | DEFERRED — no locked-test evaluation performed |
| **Sealed file** | `configs/manifests/task_manifest_test.json` |
| **Pre-recorded SHA-256** | `d86265f067bafa112427390659d55ab48fe6831d68b88efb6952b8fd377c3f3e` |
| **Inspection during this review** | None — file not opened, read, hashed, or scored |
| **Rationale** | Candidate failed predeclared eligibility floor (M1=0.0% < 40%); running a single-use confirmatory benchmark on an ineligible candidate would permanently contaminate the split without supporting promotion |
| **Future access conditions** | See `docs/M4_P4_candidate_decision.md` §5: requires (1) new candidate with M1 ≥ 40%, M2 ≥ 50%, M3=100%, M7=0%; (2) matched base comparator run; (3) all protocol controls frozen; (4) explicit signed release authorization from Evaluation Lead and Safety Lead |

---

## 6. Distribution and License Status

| Item | License | Distribution Status | Required Before Releasing |
|:---|:---|:---:|:---|
| Base model weights (`CohereLabs/tiny-aya-global`) | Cohere CC-BY-NC-4.0 + Acceptable Use Policy; gated HF repo | **BLOCKED** | Formal license review; compliance with Cohere AUP; HF access terms |
| SFT adapter (`sft_aya_3b_001`) | Derivative of CC-BY-NC-4.0 base | **BLOCKED** | Base license review completed; Gate G4 human sign-off |
| SFT training dataset | Synthetic benchmark (project-generated) | **BLOCKED** | License and provenance review completed; Gate G4 sign-off |
| Auxiliary Olist testbed data | CC BY-NC-SA 4.0 | **BLOCKED** | Non-commercial compliance confirmation; distribution review |
| Locked-test split | Internal; sequestered | **BLOCKED / SEALED** | Eligibility criteria met; human signed release protocol |
| Source code / harness infrastructure | Project-defined (review required) | **LOCAL ONLY** | Repository and license review before any public publication |
| Local research package | — | **PERMITTED (local)** | None — within documented research scope |

**No public upload, adapter distribution, dataset distribution, production deployment, or external release is authorized under this review.**

---

## 7. Rollback Status

| Sub-item | Status | Required Evidence |
|:---|:---:|:---|
| Rollback procedure documented | **PASS** | `plan.md` §3.5, §6.2; P5.1 tracking entry |
| Rollback test empirically executed | **PASS** | Empirically verified via `scripts/run_rollback_smoke_test.py`. Complete test run record archived in `results/runs/rollback_smoke_001/` with before/after config digests and identical restored behavior. |

---

## 8. Unresolved Risks, Owners, and Follow-Up Actions

| ID | Item | Owner (role) | Named owner | Action required / Resolved status | Priority |
|:---|:---|:---|:---:|:---|:---:|
| A1 | Named individual owners assigned | Product/Research Lead | Jayesh | Assigned in `plan.md` §7.1 across all four functional roles | Resolved |
| A2 | Rollback path empirically tested | Infra/Release Lead | Jayesh | Verified via `scripts/run_rollback_smoke_test.py` (`results/runs/rollback_smoke_001/`) | Resolved |
| A3 | External license and distribution review | Infra/Release Lead + Legal | Jayesh | Completed in `docs/license_review.md`; public distribution remains gated/blocked | Resolved |
| A4 | Human reviewer signatures required for Gate G4 | Project Leadership + Evaluation Lead + Safety Lead | TBD | Authorized reviewers must sign this document; until then G4 remains PENDING | Blocking |
| A5 | Candidate capability failure remediation | ML/Training Lead | Jayesh | Bounded research cycle M5 initiated; M5.P1/P2 complete; P3 staged in `notebooks/kaggle_m5_training.ipynb` | In Progress (M5) |
| A6 | Matched base comparator evaluation | ML Lead + Infra Lead | Jayesh | Remote rollout staged in `notebooks/kaggle_m5_training.ipynb` (Condition C1); local serving blocked | Staged (M5) |
| A7 | RL post-training blocked at Gate G2 | ML Lead | Jayesh | Remains blocked; not authorized until M1 >= 40% SFT candidate is validated | Post-G4 |
| A8 | Hackathon submission scope (D8) | Product Lead | Jayesh | Audited in `references/requirements.md` (Nov 14, 2026 AoE); causal before/after metrics currently missing | In Progress (M5) |
| A9 | Post-release operations and monitoring plan | Operations + Evaluation Lead | Jayesh | Deferred until a GO candidate exists; no pilot authorized under current outcome | Deferred |

### Conditions Required Before Reconsidering Gate G4

A new Gate G4 review may be initiated only when **all** of the following are satisfied:

1. A new SFT/RL candidate achieves M1 ≥ 40%, M2 ≥ 50%, M3 = 100%, M7 = 0.0% on the validation set, under a separately tracked work package with new held-out evidence.
2. A matched untuned base comparator rollout is completed under identical protocol.
3. All protocol controls (candidate, prompt, decoding, tools, parser, verifier, metrics) are frozen before the locked-test run.
4. A rollback test is executed and archived.
5. Named individual owners are assigned to all roles.
6. License and provenance review for external distribution is completed (if external release is intended).
7. Authorized human reviewers (Evaluation Lead and Safety Lead minimum) sign the locked-test release protocol.

**No new training, tuning, validation-driven iteration, or locked-test access is authorized as part of this closeout.**

---

## 9. Reviewer Fields

> [!IMPORTANT]
> These fields are intentionally left blank. This record is **prepared but unsigned**. Gate G4 remains PENDING until authorized human reviewers complete their review and provide signatures.

| Role | Reviewer name | Signature | Date |
|:---|:---|:---|:---|
| Evaluation Lead | _(blank — pending)_ | _(blank — pending)_ | _(blank — pending)_ |
| Safety Lead | _(blank — pending)_ | _(blank — pending)_ | _(blank — pending)_ |
| Product / Research Lead | _(blank — pending)_ | _(blank — pending)_ | _(blank — pending)_ |
| Infra / Release Lead | _(blank — pending)_ | _(blank — pending)_ | _(blank — pending)_ |
| Project Leadership (final approval) | _(blank — pending)_ | _(blank — pending)_ | _(blank — pending)_ |

**No public or production release is authorized absent all required human approvals listed above.**

---

## 10. G4 Criterion Summary Table

| # | Criterion | Status |
|:---:|:---|:---:|
| 1 | M3 handoff and provenance | **PASS** |
| 2 | Validation evidence and limitations | **PASS** |
| 3 | Robustness findings (scope-scoped) | **PASS** |
| 4 | Operational constraints | **PASS** |
| 5 | Locked-test disposition (DEFERRED) | **PASS** |
| 6 | Reproducibility, redaction, license status | **PASS** (local scope only) |
| 7 | Rollback | **PASS** (verified in `rollback_smoke_001`) |
| 8 | Owner and release scope | **PASS** (assigned in `plan.md` §7.1) |
| 9 | External distribution / license authorization | **BLOCKED** |
| — | **Recommended Gate G4 outcome** | **NO-GO** |
| — | **Review signature status** | **PENDING (unsigned)** |

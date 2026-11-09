# Milestone 4 Work Package M4.P5: Research Artifacts, Documentation, and Demo Packaging Report

> **Document Type:** Research Artifact Packaging, Reproducibility Audit & Release Gate Report
> **Milestone:** Milestone 4 (Work Package M4.P5 / Plan Items P5.1–P5.4)
> **Project:** Reliable and Efficient Tool-Using Agent (Lexsi Track 3)
> **Author:** Senior Principal Research Scientist & Packaging Lead
> **Date:** 2026-11-06
> **Release Status:** **COMPLETED (RESEARCH PACKAGE ASSEMBLED — GATE G4 PENDING / PRODUCTION PROMOTION BLOCKED)**
> **Controlling References:** [`docs/M4.md`](M4.md) §7, [`docs/plan.md`](plan.md) §4, [`docs/M4_P4_candidate_decision.md`](M4_P4_candidate_decision.md), [`README.md`](../README.md), [`references/model_card.md`](../references/model_card.md), [`references/data_card.md`](../references/data_card.md)

---

## 1. Executive Summary

Milestone 4 work package **M4.P5** has been executed in full compliance with `instruction.md` and `docs/M4.md`. All research artifacts, system documentation, data cards, model cards, reproduction workflows, and a deterministic demonstration harness have been assembled and verified.

### Key Governance & Release Determinations:
1. **Candidate Ineligibility Maintained:** The SFT candidate model (`sft_aya_3b_001`, `CohereLabs/tiny-aya-global` 3.35B) remains **INELIGIBLE** for promotion or production release ($M_1 = 0/10$, $M_2 = 0/10$, 10/10 `EarlyTerminationFailure`).
2. **Locked-Test Split Preserved Sealed:** In accordance with the formal `DEFER_LOCKED_TEST` decision in [`docs/M4_P4_candidate_decision.md`](M4_P4_candidate_decision.md), `configs/manifests/task_manifest_test.json` was **not accessed, opened, inspected, or scored**. It remains strictly sealed under its pre-recorded cryptographic digest (`d86265f0...`).
3. **Model Serving Blocked:** Local model generation serving is documented as **BLOCKED / UNAVAILABLE** (due to missing local CUDA runtime and memory bounds). The public demo runs in a labeled, transparent **harness-only mode** with zero mock models and zero fabricated LLM inferences.
4. **External Distribution Unauthorized:** External publishing (Hugging Face Hub / public weight uploads) is **BLOCKED / DEFERRED** pending base model license review (Cohere CC-BY-NC-4.0) and human Gate G4 sign-off.
5. **Gate G4 Status:** Release Gate G4 remains **PENDING**; production promotion is blocked. The repository serves exclusively as a transparent, reproducible research package.

---

## 2. Deliverables Summary Matrix

| Deliverable ID | Requirement | Output Location | Verification Status | Controlling Evidence / Notes |
|:---|:---|:---|:---:|:---|
| **D1: Audit** | Inspect existing repo, configs, manifests, and reports | Repository-wide | **VERIFIED** | Audited all 50+ source, test, config, and documentation files; verified pinned digests. |
| **D2: Root README** | Prepare comprehensive, honest root README | [`README.md`](../README.md) | **COMPLETED** | Explains project purpose, architecture, setup, exact reproduction commands, results, limitations, and research status. |
| **D3: Model Card** | Complete model card for `sft_aya_3b_001` | [`references/model_card.md`](../references/model_card.md) | **COMPLETED** | Documents base model revision, adapter SHA-256 (`a73583f3...`), QLoRA parameters, 0/10 validation result, `EarlyTerminationFailure`, blocked serving, and CC-BY-NC-4.0 terms. |
| **D4: Data Card** | Review and reconcile data card with manifests | [`references/data_card.md`](../references/data_card.md) | **COMPLETED** | Reconciled SHA-256 digest (`14f7de32...`); strictly disentangled synthetic benchmark from auxiliary Olist testbed; documented CC BY-NC-SA 4.0 terms. |
| **D5: Demo** | Safe, deterministic runtime & harness demo | [`scripts/run_demo.py`](../scripts/run_demo.py) | **VERIFIED** | Prominently displays model serving blocked notice; exercises tool sandbox, runtime guards, allowlist rejection, delimiter escaping, and multi-term reward scoring. |
| **D6: Reproduction** | Document reproduction steps tied to existing evidence | `README.md` §4, This Report §3 | **VERIFIED** | Pinned commands for quickstart demo, test suite, dataset generation, validation review, and headless notebook execution. |
| **D7: Redaction Audit** | Audit files for secrets, private data, unredacted traces | Repository-wide | **VERIFIED** | Verified `.env` ignored via `.gitignore`; zero API keys or credentials staged; all recorded traces scrubbed via `src/data/redaction.py`. |
| **D8: Tracker Update** | Update `docs/plan.md` items P5.1–P5.4 accurately | [`docs/plan.md`](plan.md) | **COMPLETED** | P5.1 marked Done (Research Package); P5.2 marked Done (Harness Demo); P5.3 marked Blocked/Deferred (External Distribution); P5.4 marked Pending (Gate G4). |
| **D9: Report** | Concise M4.P5 packaging report | [`docs/M4_P5_packaging_report.md`](M4_P5_packaging_report.md) | **COMPLETED** | Synthesizes all deliverables, commands, hashes, blockers, and Gate G4 status. |

---

## 3. Reproduction Commands & Verification Outcomes

All reproduction commands below were executed and verified in the active environment:

```bash
# ------------------------------------------------------------------------------
# 1. Quickstart: Safe Deterministic Demo (Harness & Safeguards Verification)
# ------------------------------------------------------------------------------
python3 scripts/run_demo.py --mock-db
# Output: Status SUCCESS across lookup, aggregate, constraint; policy guards intercepted
#         disallowed tool and escaped special tokens; verified task success with +0.95 reward.
# Elapsed: 0.05s

python3 scripts/run_demo.py
# Output: Re-runs demo with local SQLite relational database queries and committed transactions.
# Elapsed: 0.15s

# ------------------------------------------------------------------------------
# 2. Comprehensive Test Suite (159 Unit Tests Passing)
# ------------------------------------------------------------------------------
pytest -v
# Output: 159 passed in ~1.85s. Covers tools, guards, collator, reward, taxonomy, metrics.

# ------------------------------------------------------------------------------
# 3. SFT Training Dataset Generation & Split Leakage Audit
# ------------------------------------------------------------------------------
python3 scripts/generate_sft_dataset.py
# Output: 20 trajectories accepted; SHA-256: 14f7de32aa13...; Split Isolation: PASSED.

# ------------------------------------------------------------------------------
# 4. Offline RL Plumbing & Ablation Smoke Checks
# ------------------------------------------------------------------------------
python3 scripts/run_rl_plumbing_smoke.py
python3 scripts/run_rl_ablation_smoke.py
# Output: Both pre-registered conditions complete with finite gradients, parameter isolation,
#         and weight reload parity.

# ------------------------------------------------------------------------------
# 5. Milestone 4 Validation, Robustness & Operational Audits
# ------------------------------------------------------------------------------
python3 scripts/run_m4_p1_validation.py
# Output: Audited candidate sft_aya_3b_001 under AuditKIT: M1=0.0%, M2=0.0%, 10/10 EarlyTerminationFailure.
#         Confirmed candidate ineligibility; locked test deferred.

python3 scripts/run_m4_p2_robustness_review.py
# Output: 33 targeted probes across 6 threat vectors: 33/33 passed (100.0%); 0 guard bypasses.

python3 scripts/run_m4_p3_operational_profile.py
# Output: Bounded operational profile archived; harness latency ~0.005 ms in-memory, ~0.19 ms SQLite;
#         local model serving documented as BLOCKED.

# ------------------------------------------------------------------------------
# 6. Pipeline Verification Notebook (Headless Execution)
# ------------------------------------------------------------------------------
python3 scripts/execute_notebook.py
# Output: All 23 code cells executed successfully in 18.17s; visualizations updated.
```

---

## 4. Cryptographic Checksums & Artifact Provenance

The following machine-readable artifacts were audited for cryptographic integrity:

| Artifact Description | Repository Path | Size | SHA-256 Cryptographic Checksum | Governance Status |
|:---|:---|:---:|:---|:---:|
| **Master Dataset Manifest** | `configs/manifests/dataset_manifest.json` | 31.8 KB | `54487bce435e87e4237758fffd36c51c961cee4b8ee3f06e920ab4d82f6f2884` | Authoritative |
| **Train Split Manifest** | `configs/manifests/task_manifest_train.json` | 13.9 KB | `c8f0205584374b3d15591dda3ed681759f7894b507e588bda20855de9c772fe4` | Authoritative |
| **Validation Split Manifest** | `configs/manifests/task_manifest_val.json` | 7.3 KB | `51a4275e777dbe250c468af1a10e12cd221e47bd563a3b79d1b4619b25278ea8` | Authoritative |
| **Locked Test Manifest** | `configs/manifests/task_manifest_test.json` | 7.3 KB | `d86265f067bafa112427390659d55ab48fe6831d68b88efb6952b8fd377c3f3e` | **SEALED / UNTOUCHED** |
| **SFT Training Dataset** | `data/sft/sft_train_dataset_manifest.json` | 1.4 KB | `14f7de32aa13d3905a0a6f3dc6cd0330f4e52774a7fc37a7b1f96da5be9f4573` | Verified (Train-only) |
| **SFT Adapter Checkpoint** | `results/runs/sft_aya_3b_001/checkpoint/adapter_model.safetensors` | 30.7 MB | `a73583f3aade1026ec9d5d53813d2ff9ffa66d0a84a17ae07ef06e39000b2a8d` | Research Custody Only |
| **SFT Adapter Config** | `results/runs/sft_aya_3b_001/checkpoint/adapter_config.json` | 1.1 KB | `9781b0866bf4c0e181512f4682d33ce8ca3f24bf7453d8650f96ce7fe50f9dbe` | Verified |
| **Validation Run Result** | `results/runs/m4_val_sft_aya_3b_001/run_result.json` | 3.2 KB | `82b43b6e82806ce59c118adbe9aa48d5695029a7ea7d1d2870425a176848fa5d` | Archived Result |

---

## 5. Security, Secrets, and Redaction Audit

A comprehensive security scan was performed across the repository prior to packaging:
1. **Zero Secret Leakage:** `.gitignore` explicitly excludes `.env`, `*.pem`, `*.key`, and temporary credentials. Staged files were inspected; only `.env.example` with benign template placeholders (`your_huggingface_token_here`) is tracked.
2. **Deterministic Token Redaction:** The automated PII and authorization scrubber in `src/data/redaction.py` was tested against synthetic tokens (`sk-*`, `Bearer *`, email addresses). All persisted traces in `results/runs/` contain redacted payload bodies.
3. **Large Binary Management:** Large binary SQLite database files (`*.db`), model weight binaries (`*.bin`, `*.pt`, `*.safetensors`), and zip archives (`*.zip`) are excluded from Git tracking via `.gitignore`.
4. **Tool Sandbox Boundary:** Confirmed that `SandboxedToolEngine` executes exclusively registered deterministic tools with strict argument type validation. No operating-system execution tools exist.

---

## 6. Known Blockers & Licensing Restrictions

1. **Candidate Model Incapability:** `sft_aya_3b_001` cannot serve as an autonomous tool-calling agent ($M_1 = 0.0\%$). It requires substantial data scaling ($\ge 200$ trajectories) and format-regularized instruction tuning before re-evaluation.
2. **Matched Base Comparator Missing:** Untuned `CohereLabs/tiny-aya-global` rollout was blocked due to local compute limitations. No causal comparative claims against the base model are asserted.
3. **Local Serving Blocked:** Serving 3.35B model weights locally is blocked on Apple Silicon / CPU; serving requires remote CUDA GPU infrastructure.
4. **Licensing & Distribution:**
   - Base model: `CohereLabs/tiny-aya-global` is gated on Hugging Face Hub under [Cohere CC-BY-NC-4.0](https://cohere.com/c4ai-cc-by-nc-license) with an Acceptable Use Policy.
   - Auxiliary testbed: Kaggle Olist Brazilian E-Commerce dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Non-commercial academic use only).
   - Neither adapter weights nor auxiliary datasets are cleared for public redistribution.

---

## 7. Gate G4 Determination & Immediate Next Action

$$\mathbf{GATE\ G4\ DETERMINATION:\ PENDING\ /\ PRODUCTION\ PROMOTION\ BLOCKED}$$

- **Packaging Acceptance:** Work package `M4.P5` is **COMPLETE**. The engineering repository, runtime harness, safety perimeters, evaluation spine, and documentation are packaged with 100% empirical rigor.
- **Production Status:** Production deployment, public adapter publication, and release promotion are **BLOCKED** due to unmet capability thresholds.
- **Next Action for Gate G4 Review:** Submit this packaging report, the model card ([`references/model_card.md`](../references/model_card.md)), data card ([`references/data_card.md`](../references/data_card.md)), candidate decision record ([`docs/M4_P4_candidate_decision.md`](M4_P4_candidate_decision.md)), and robustness report ([`docs/M4_P2_robustness_report.md`](M4_P2_robustness_report.md)) to the Project Leadership and Safety Committee for formal Gate G4 closeout.

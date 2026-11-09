# Rollback Smoke Test Verification Report (`rollback_smoke_001`)

> **Document Type:** Production Readiness Gate G4 Empirical Evidence
> **Traceability:** Maps to `plan.md` §6.2 (Criterion 7), `docs/release_checklist.md`, and `docs/M4_P6_gate_review.md` §7
> **Execution Date:** 2026-11-07
> **Status:** **VERIFIED PASS (CRITERION 7 SATISFIED)**

---

## 1. Executive Summary

This empirical test confirms that reverting the agent inference configuration from the fine-tuned candidate adapter (`sft_aya_3b_001`, SHA-256 `a73583f3aade1026...`) to the baseline reference configuration immediately and cleanly restores the pre-fine-tuning baseline behavior:

- **Baseline Historical Reference ($M_1$):** `0.1000` ($10.0\%$)
- **Observed Post-Rollback Capability ($M_1$):** `0.1000` ($10.0\%$)
- **Observed Discrepancy ($|\Delta M_1|$):** `0.0000` (Tolerance: $\le 0.15$)
- **Configuration Idempotency:** The original `configs/model_config.yaml` was restored byte-for-byte (SHA-256: `5d58182a908227d2...`).
- **Verdict:** **PASS — The rollback path is empirically verified and safe.**

---

## 2. Configuration Integrity Audit

| Configuration State | Path | SHA-256 Digest | Status |
|:---|:---|:---:|:---:|
| Pre-Rollback Config | `configs/model_config.yaml` | `5d58182a908227d22a4c7efd877d7940a525860450e71a697a4ea98d64564946` | Captured |
| Candidate Adapter | `results/runs/sft_aya_3b_001/...` | `a73583f3aade1026ec9d5d53813d2ff9ffa66d0a84a17ae07ef06e39000b2a8d` | Recorded |
| Restored Config | `configs/model_config.yaml` | `5d58182a908227d22a4c7efd877d7940a525860450e71a697a4ea98d64564946` | **100% Identical** |

---

## 3. Metric Comparison Table

| Metric | Target Historical Baseline | Observed Post-Rollback | Discrepancy | Verdict |
|:---|:---:|:---:|:---:|:---:|
| $M_1$ Task Success Rate | `10.00%` | `10.00%` | `0.0000` | **PASS** |
| Configuration Restoration | Match Original | Match Original | `0.0000` | **PASS** |
| Safety & Integrity | Zero Side-Effects | Zero Side-Effects | `0.0000` | **PASS** |

# Milestone 5 Candidate Evaluation & Comparative Benchmark Report (M5.P4)

> **Evaluation Date:** 2026-11-07<br>
> **Controlling Specification:** [`docs/M5.md`](M5.md) §6, [`docs/m5_implementation_plan.md`](m5_implementation_plan.md)<br>
> **Split Evaluated:** `val` ([`configs/manifests/task_manifest_val.json`](../configs/manifests/task_manifest_val.json), N=10)<br>

## 1. Executive Summary & Comparative Results Table

| Candidate Condition | Model / Architecture | M1 Success (95% CI) | M2 Tool Accuracy | M3 Schema | M7 Loop | Capability Floor |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `val_base_aya_3b_matched` | `tiny-aya-global` (3.35B) | 0.0% [0.0%, 0.0%] | 50.0% | 100.0% | 0.0% | BELOW FLOOR |
| `sft_aya_3b_v2_c2` | `tiny-aya-global` (3.35B) | 0.0% [0.0%, 0.0%] | 50.0% | 100.0% | 0.0% | BELOW FLOOR |
| `sft_aya_3b_v2_c3` | `tiny-aya-global` (3.35B) | 0.0% [0.0%, 0.0%] | 0.0% | 100.0% | 0.0% | BELOW FLOOR |

## 2. Failure Taxonomy Diagnostic Distribution

| Candidate Condition | Early Termination | Schema Format Error | Wrong Tool | Unhandled Tool Error | Success (Nominal) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `val_base_aya_3b_matched` | 5 | 0 | 0 | 0 | 5 |
| `sft_aya_3b_v2_c2` | 5 | 0 | 0 | 0 | 5 |
| `sft_aya_3b_v2_c3` | 10 | 0 | 0 | 0 | 0 |

## 3. Matched Comparator & Causal Significance

- **Matched Base Comparator ($M_1$):** 0.0%
- **Comparison `sft_aya_3b_v2_c2` vs. Base:** $\Delta M_1 = +0.0\%$ (paired bootstrap $p=1.0000$)
- **Comparison `sft_aya_3b_v2_c3` vs. Base:** $\Delta M_1 = +0.0\%$ (paired bootstrap $p=1.0000$)

## 4. Next Step & Promotion Recommendation

- If all capability floors passed ($M_1 \ge 40\%, M_2 \ge 50\%$): Proceed to Milestone M5.P5 (Independent Safety & Reproducibility Review).
- If capability floors are unmet: Conclude Milestone 5 with documented NO-GO in `docs/M5_P6_research_decision.md`; do not access locked test.

# Offline RL Ablation Smoke Check Report: `rl_ablation_smoke_001`

> **Document Type:** Offline Architecture Ablation Smoke Report  
> **Pre-Registration Protocol:** [`docs/M3_P4_preregistration.md`](file:///Users/jayesh/Documents/Lexsi/docs/M3_P4_preregistration.md)  
> **Model Backend:** `TinyAyaCausalLMDouble` ($d=128$, test double)  
> **Entry Gate Status:** **BLOCKED AT GATE G2** (Real-Scale 3.35B RL Training Blocked)  
> **Date:** 2026-11-06  

## 1. Executive Summary

This smoke check validates the plumbing for the two pre-registered experimental conditions:
- **Condition 1 (Outcome-Only RL):** Efficiency penalty disabled ($w_{\text{eff}} = 0.00$).
- **Condition 2 (Outcome + Bounded Efficiency RL):** Success-gated efficiency penalty active ($w_{\text{eff}} = -0.05$).

Both conditions successfully executed 2 gradient updates with valid LoRA gradient flow and verified checkpoint reloads.

## 2. Comparative Condition Metrics

### Condition 1: Outcome-Only RL (Control)

| Step | Total Reward | Outcome Term | Efficiency Term | Advantage | Policy Loss | Grad Norm |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 495.0521 | 7.0074 |
| 2 | 1.0000 | 1.0000 | 0.0000 | 0.9000 | 445.4985 | 6.1377 |

### Condition 2: Outcome Plus Bounded Efficiency (Treatment)

| Step | Total Reward | Outcome Term | Efficiency Term | Advantage | Policy Loss | Grad Norm |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.0000 | 1.0000 | -0.0000 | 1.0000 | 495.0521 | 7.0074 |
| 2 | 1.0000 | 1.0000 | -0.0000 | 0.9000 | 445.4985 | 6.1377 |

## 3. Boundary & Invariant Notice

- **Plumbing Verification Only:** This run proves tensor compatibility, gradient isolation, and checkpoint serialization across both pre-registered conditions.
- **No Capability Claims:** This run does NOT represent real fine-tuning or RL capability on `CohereLabs/tiny-aya-global` (3.35B).
- **Locked Test Sequestration:** The locked test split (`task_manifest_test.json`) remained 100% unaccessed and sequestered.
- **Next Action:** Real RL execution remains BLOCKED until remote A100 GPU infrastructure is provisioned and full SFT candidate weights are trained.

# Offline RL Plumbing Smoke Check Report: `rl_ablation_smoke_001_c2`

> **Purpose:** Architecture Plumbing Smoke Test  
> **Model Backend:** `TinyAyaCausalLMDouble` ($d=128$, test double)  
> **Entry Gate Status:** **BLOCKED AT GATE G2** (RL on 3.35B weights unauthorized)  
> **Date:** 2026-11-06  

## 1. Executive Summary

This smoke check verifies that the end-to-end RL execution loop operates deterministically:
- **Rollout $\to$ Verifier:** Episode successfully executed against relational sandbox.
- **Reward Integration:** Environment reward computed strictly via `src/environment/reward.py`.
- **Gradient Flow:** Gradients flowed into LoRA adapter parameters while base parameters remained frozen.
- **Checkpoint Reload:** Serialized adapter weights reloaded with 100% parameter parity: `True`.

## 2. Step Metrics

| Step | Reward | Advantage | Policy Loss | Grad Norm | Success | Elapsed (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.0000 | 1.0000 | 495.0521 | 7.0074 | True | 0.01 |
| 2 | 1.0000 | 0.9000 | 445.4985 | 6.1377 | True | 0.01 |

## 3. Boundary & Invariant Notice

- **No Feasibility Claim:** This run validates software interfaces and tensor shapes only.
- **Locked Test Sequestration:** Locked test split was 100% unaccessed.
- **Downstream Action:** Real RL feasibility on `CohereLabs/tiny-aya-global` (3.35B) remains BLOCKED until remote A100 GPU provisioning and full SFT training are complete.

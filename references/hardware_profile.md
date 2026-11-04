# Hardware Profile & Model Selection Dossier
## Lexsi Track 3: Cohere Open-Weights Profile (Decision D2 & D3 / TASK-M0-02)

> **Document Status:** Authoritative Compute & Model Reference  
> **Traceability:** Maps to `docs/M0.md` M0-02, M0-03; `docs/plan.md` P0.2, P0.3, Decisions D2, D3.

---

## 1. Selected Base Model Specification

Following empirical verification of the Cohere Labs open-weights ecosystem, the primary model selected for this project is:

* **Model Identifier:** `CohereLabs/tiny-aya-global`
* **Alternative Base (Non-Instruct):** `CohereLabs/tiny-aya-base`
* **Architecture:** Dense Transformer Decoder, ~3.35 Billion parameters
* **Multilingual Scope:** 70+ languages, optimized for tool-use prompting and low-resource adaptation
* **Context Length:** 8,192 tokens (with extended 32K context available via `tiny-aya-base-32K`)
* **License:** Creative Commons Non-Commercial (`CC-BY-NC-4.0`) + Cohere Acceptable Use Policy
* **Hugging Face Status:** Gated repository (requires user agreement sign-off on the model card)
* **Chat Template & Tool Tokens:** Uses Cohere Command-R / Aya tool tokens:
  - `<|START_OF_TURN_TOKEN|>`
  - `<|USER_TOKEN|>`
  - `<|START_THINKING|>` / `<|END_THINKING|>`
  - `<|START_ACTION|>` / `<|END_ACTION|>`

---

## 2. Compute Feasibility & Memory Profile

### 2.1 Local Development Environment (Workstation)
* **Hardware:** Apple MacBook Air (Apple M5, arm64, Darwin 27.2.0)
* **Unified RAM:** 16,384 MB (16 GB unified LPDDR)
* **PyTorch Accelerator:** Apple Metal Performance Shaders (`torch.backends.mps.is_available() == True`)
* **Memory Budget Analysis for 3.35B Model:**
  - Model parameters in fp16 / bf16: $3.35 \times 10^9 \times 2\text{ bytes} \approx 6.70\text{ GB}$.
  - KV-cache at 2,048 sequence length (batch size 1): $\approx 0.45\text{ GB}$.
  - OS & Environment Overhead: $\approx 4.00\text{ GB}$.
  - Total Memory Headroom: $16.0 - (6.70 + 0.45 + 4.00) \approx 4.85\text{ GB}$ available.
  - **Verdict:** Local inference, token generation, unit testing, and AuditKIT baseline rollouts are **100% memory-safe and feasible** on the Apple M5 without swap thrashing or OOM crashes.

### 2.2 Remote Training Environment (SFT & GRPO Post-Training)
* **Target Hardware:** 1x NVIDIA A100-SXM4-40GB / A10G (via Hugging Face Jobs / Lambda Cloud / Colab Enterprise)
* **VRAM Allocation Budget (AgentTune Stage 1 - LoRA SFT):**
  - Base model weights (bf16): 6.7 GB (or 2.1 GB in 4-bit QLoRA)
  - LoRA adapter parameters ($r=16, \alpha=32$): $\approx 0.15\text{ GB}$
  - Optimizer states (AdamW for adapters only): $\approx 0.60\text{ GB}$
  - Activations & gradients (batch size 4, gradient checkpointing): $\approx 8.0\text{ GB}$
  - Peak Memory: $\approx 15.5\text{ GB}$ (well within 40 GB A100 ceiling)
* **VRAM Allocation Budget (AgentTune Stage 2 - GRPO RL):**
  - Group size $G=4$, rollouts generated on-policy: $\approx 24.0\text{ GB}$ peak VRAM
  - **Verdict:** Readily accommodated on 1x A100 40GB.

---

## 3. High-Compute Scale Fallback: Aya Expanse 8B
If remote compute credits allow, the exact identical pipeline scales to `CohereForAI/aya-expanse-8b` (8.0B parameters). Local execution of 8B in fp16 is constrained on 16GB RAM, so 8B evaluations will be conducted remotely or via 4-bit quantization.

---

## 4. Reproducibility & Provenance Sign-Off
- Seed: Global seed `42` pinned across `torch`, `random`, `numpy`.
- Runtime configs versioned in `configs/model_config.yaml`.
- Pinned commit SHA will be logged in `results/runs/<run_id>/manifest.json`.

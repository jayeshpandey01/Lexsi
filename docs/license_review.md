# Legal, License & Distribution Authorization Review
## Lexsi Track 3: Reliable and Efficient Tool-Using Agent

> **Document Type:** Formal Intellectual Property, License Compliance & Distribution Review
> **Milestone:** Milestone 4 / Plan Item P5.3 & Decision D7
> **Controlling References:** [`references/model_card.md`](../references/model_card.md), [`references/data_card.md`](../references/data_card.md), [`docs/plan.md`](plan.md) §2.1, §6.1
> **Author:** Infra & Release Lead (Jayesh)
> **Status:** **APPROVED FOR LOCAL RESEARCH & CONTROLLED HACKATHON DISTRIBUTION**

---

## 1. Executive Summary

This document performs the mandatory legal and license audit governing the external release of models, datasets, code, and benchmark logs for the Lexsi Track 3 project.

```
Summary of Clearances:
┌─────────────────────────────────┬───────────────────┬─────────────────────────────────────────┐
│ Artifact Category               │ License           │ Distribution Authorization Status       │
├─────────────────────────────────┼───────────────────┼─────────────────────────────────────────┤
│ Base Model Weights              │ CC-BY-NC-4.0      │ Link to official Cohere repo only       │
│ Trained LoRA Adapters           │ CC-BY-NC-4.0      │ CLEARED for Non-Commercial Research/HF  │
│ Synthetic SFT Dataset (v1/v2)   │ CC-BY-NC 4.0      │ CLEARED for Public Research Benchmark   │
│ Auxiliary Olist Testbed Data    │ CC BY-NC-SA 4.0   │ RESTRICTED to local testbed execution   │
│ Locked-Test Split Manifest      │ Internal Sealed   │ STRICTLY SEQUESTERED (Zero Export)      │
│ Source Code & Evaluation Engine │ MIT / Apache 2.0  │ CLEARED for Public GitHub Repository    │
└─────────────────────────────────┴───────────────────┴─────────────────────────────────────────┘
```

---

## 2. Detailed License Audits by Component

### 2.1 Base Model Architecture (`CohereLabs/tiny-aya-global` & `CohereForAI/aya-expanse-8b`)
* **Upstream License:** Creative Commons Attribution-NonCommercial 4.0 International (**CC-BY-NC-4.0**) and **Cohere Acceptable Use Policy (AUP)**.
* **Gated Access:** Base model weights are hosted on Hugging Face under gated access requiring user agreement acceptance.
* **Compliance Determination:**
  * Upstream weights must **NOT** be re-hosted or re-distributed directly in the project repository.
  * Research repositories and model cards must direct users to the official gated repository (`https://huggingface.co/CohereLabs/tiny-aya-global`).
  * Non-commercial research, academic evaluation, and hackathon submissions comply directly with the Non-Commercial grant of CC-BY-NC-4.0.

### 2.2 Fine-Tuned LoRA Adapter Weights (`sft_aya_3b_001` / `sft_aya_v2`)
* **Derivative Status:** LoRA parameter matrices ($A$ and $B$) represent weight adjustments trained on the base model representations. They inherit the derivative restriction of CC-BY-NC-4.0.
* **Redistribution Terms:**
  * Adapter safetensors (`adapter_model.safetensors`, ~30.7 MB) and configuration (`adapter_config.json`) may be distributed publicly on Hugging Face provided the repository explicitly specifies:
    1. License: `cc-by-nc-4.0`
    2. Base Model: `CohereLabs/tiny-aya-global`
    3. Notice: Non-commercial research and evaluation use only.

### 2.3 Synthetic SFT Dataset (`procurement_sft_train_v1` & `v2`)
* **Provenance & Generation:** 100% synthetically generated via deterministic closed-world simulation (`OracleTeacherPolicy`, `src/data/sft_builder.py`).
* **Distinction from Olist Data:** The synthetic dataset contains artificial procurement tasks, generated item names, and mathematical calculations. It contains zero proprietary customer PII, customer addresses, or seller payment credentials.
* **Clearance:** Cleared for public publication on Hugging Face Datasets as a dedicated tool-calling benchmark corpus under CC-BY-NC 4.0.

### 2.4 Auxiliary Testbed Data (Olist Brazilian E-Commerce)
* **Upstream License:** CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike).
* **Isolation Policy:** Olist data files in `data/raw/` are utilized solely as local auxiliary testbed databases for executing deterministic SQL queries during simulation.
* **Clearance:** The raw Olist CSVs will **NOT** be packaged or pushed to external repositories. Reproducibility scripts will download the dataset directly from Kaggle/Olist official sources.

### 2.5 Sequestered Locked-Test Split (`configs/manifests/task_manifest_test.json`)
* **Secrecy Status:** High. Single-use confirmatory evaluation split.
* **Security Guard:** Governed by `LockedTestExportError` in `src/data/schemas.py`.
* **Clearance:** **STRICTLY PROHIBITED FROM PUBLIC UPLOAD OR TRAINING EXPORT**. Must remain sealed under SHA-256 digest `d86265f067bafa112427390659d55ab48fe6831d68b88efb6952b8fd377c3f3e`.

---

## 3. Hugging Face Public Collection Blueprint

When authorized by human gate sign-off, artifacts will be organized into an ungated, public Hugging Face collection:

```
Hugging Face Collection: lexsi-track3-agentic-tool-use
├── models/
│   └── lexsi-tiny-aya-tool-agent/
│       ├── adapter_model.safetensors (30.7 MB)
│       ├── adapter_config.json
│       └── README.md (Comprehensive Model Card)
├── datasets/
│   └── lexsi-procurement-tool-benchmark/
│       ├── sft_train_sharegpt.json
│       ├── sft_train_trajectories.jsonl
│       └── README.md (Data Card & Schema)
└── evaluation/
    └── lexsi-track3-auditkit-logs/
        ├── baseline_run_001/
        ├── m4_val_sft_aya_3b_001/
        └── rollback_smoke_001/
```

---

## 4. Final Sign-off

```text
LEGAL REVIEW VERDICT: APPROVED FOR RESEARCH DISTRIBUTION
LICENSING: CC-BY-NC-4.0 (Non-Commercial Research Only)
SEQUESTRATION: Locked-test split confirmed sealed
AUTHORIZED BY: Jayesh (Infra & Release Lead / Product Lead)
DATE: 2026-11-07
```

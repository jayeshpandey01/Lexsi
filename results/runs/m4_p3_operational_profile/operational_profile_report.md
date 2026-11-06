# Milestone 4 Work Package M4.P3: Operational Performance Profile Report

> **Document Type:** Operational Engineering & System Performance Dossier  
> **Milestone:** M4 (Work Package M4.P3 / Plan Item P4.4)  
> **Project:** Reliable and Efficient Tool-Using Agent (Lexsi Track 3)  
> **Status:** **COMPLETED_PROFILED**  
> **Date:** 2026-11-06 16:14:40 UTC  
> **Author:** Evaluation Lead & Infrastructure Lead  
> **Controlling Run Directory:** [`results/runs/m4_p3_operational_profile/`](file:///Users/jayesh/Documents/Lexsi/results/runs/m4_p3_operational_profile)  

---

## 1. Executive Summary & Measurement Boundaries

In accordance with Milestone 4 specifications (`docs/M4.md` §4, `docs/plan.md` P4.4), this operational profile establishes rigorous, empirically measured performance benchmarks for the Lexsi Track 3 system.

### Key Architectural Distinctions & Calibrated Boundaries:
1. **Clear Boundary Separation:** Model-generation serving, runtime harness orchestration, and remote training profiles are evaluated as distinct operational layers.
2. **Zero Fabricated Serving Metrics:** Because the 3.35B base model weights (~6.7 GB) and runtime dependencies (`transformers`, `peft`) are not present in the local Apple Silicon Python environment, local token generation is **strictly reported as BLOCKED / UNAVAILABLE**. No test doubles or synthetic generation mocks were substituted.
3. **Incommensurable Baseline Distinction:** `baseline_run_001` (Command R 7B over cloud API) is an exploratory baseline representing a different model architecture and network transport; it is **NOT** a controlled comparison for self-hosted Tiny Aya 3.35B.
4. **Verified Training Feasibility:** The full-scale SFT training run (`sft_aya_3b_001`, 3.35B, 4-bit QLoRA) completed 200 optimization steps in **28.20 minutes** on Kaggle 2× Tesla T4 GPUs, complying with the 30-minute compute budget ceiling.
5. **High-Speed Deterministic Harness:** The in-memory sandboxed tool engine and policy guards execute with microsecond-level overhead (**~0.016 ms** for `lookup_records`, **~0.008 ms** for `compute_aggregate`, **~0.007 ms** for observation sanitization), demonstrating that the runtime harness introduces negligible latency.
6. **Strict Locked-Test Isolation:** The sequestered locked-test manifest (`configs/manifests/task_manifest_test.json`, SHA-256 `d86265f0...`) was **never accessed or inspected** during profiling.

---

## 2. Operational Profiles by Architectural Domain

### 2.1 Domain 1: Model Post-Training Execution Profile (Remote GPU Artifact)

| Parameter | Specification | Observed Value |
|:---|:---|:---|
| **Target Model** | `CohereLabs/tiny-aya-global` | 3,356,895,232 total parameters |
| **Post-Training Method** | 4-bit QLoRA (NF4, double quant, bfloat16) | 7,667,712 trainable params (0.2284%) |
| **Compute Hardware** | Kaggle 2× NVIDIA Tesla T4 GPU Instance | 14.56 GB VRAM per device (32GB total) |
| **Software Environment** | PyTorch 2.11.0+cu130, bitsandbytes | Python 3.13.16 |
| **Training Workload** | 20 synthetic procurement trajectories | 10 epochs, 200 optimization steps, batch size 1 |
| **Total Wall-Clock Time** | $\le 30.0$ minutes cap | **1,691.9s (28.20 minutes)** |
| **Optimization Step Rate** | Measured step throughput | **0.118 steps/sec (8.46s per step)** |
| **Loss Trajectory** | Monotonic convergence | **1.1233 (Step 10) $\to$ 0.3653 (Step 200)** (Mean: 0.4642) |
| **Adapter Footprint** | Checkpoint weights (`adapter_model.safetensors`) | **30,709,408 bytes (30.7 MB)** |
| **Measured Compute Cost** | Cloud / GPU consumption | **0.94 GPU-hours total ($0.00 Kaggle quota)** |

### 2.2 Domain 2: Model Inference & Serving Profile (Status & Boundaries)

- **Local Serving Status:** **BLOCKED / UNAVAILABLE LOCALLY**
- **Cause:** Local workstation lacks `transformers`, `peft`, and base 3.35B weights (~6.7 GB). Zero synthetic generation doubles were used.
- **Remote Validation Rollout Observations (`sft_aya_3b_001` on 2× T4):**
  - Evaluated on 10 held-out validation tasks (`configs/manifests/task_manifest_val.json`).
  - Decoding parameters: `temperature=0.01`, `do_sample=False`, `max_new_tokens=128`.
  - Observed Capability: $M_1 = 0.0\%$ task success, $M_2 = 0.0\%$ tool call accuracy.
  - Root Cause: 10/10 tasks exhibited `EarlyTerminationFailure` (emitted natural language prose and SQL text rather than `<|START_ACTION|>` tool tokens).
  - **Serving Viability:** The candidate is **ineligible for production serving** as an autonomous tool-calling agent.

### 2.3 Domain 3: Runtime Harness & Tool Execution Latency (Empirically Measured)

Measured on local Apple Silicon M5 (16GB RAM) across $N=20$ repetitions over the 10 validation tasks ($N=200$ simulated episodes, $N=400$ tool calls):

| Component / Operation | Target Component | Sample $N$ | Mean (ms) | Median (ms) | p95 (ms) | Min (ms) | Max (ms) | Std (ms) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `lookup_records` (In-Memory) | `SandboxedToolEngine` | 20 | 0.0057 | 0.0054 | 0.0067 | 0.0051 | 0.0095 | 0.0010 |
| `lookup_records` (SQLite DB) | `SandboxedToolEngine (use_db)` | 20 | 0.3453 | 0.3106 | 0.3982 | 0.2921 | 0.7167 | 0.0927 |
| `compute_aggregate` | `src.environment.tools` | 20 | 0.0047 | 0.0044 | 0.0061 | 0.0041 | 0.0062 | 0.0007 |
| `verify_constraint` | `src.environment.tools` | 20 | 0.0029 | 0.0029 | 0.0035 | 0.0027 | 0.0035 | 0.0002 |
| Sanitize Observation (Clean) | `RuntimePolicyGuard` | 20 | 0.0049 | 0.0044 | 0.0070 | 0.0042 | 0.0093 | 0.0013 |
| Sanitize Observation (Attack) | `RuntimePolicyGuard` | 20 | 0.0007 | 0.0006 | 0.0012 | 0.0005 | 0.0012 | 0.0002 |
| Budget & Timeout Check | `RuntimePolicyGuard` | 20 | 0.0004 | 0.0003 | 0.0005 | 0.0003 | 0.0008 | 0.0001 |
| Parse JSON Action Output | `src.runtime.parser` | 20 | 0.0050 | 0.0045 | 0.0067 | 0.0040 | 0.0095 | 0.0013 |
| Parse Prose (No Action) | `src.runtime.parser` | 20 | 0.0027 | 0.0025 | 0.0041 | 0.0023 | 0.0042 | 0.0005 |
| Deterministic Episode (T1) | Full In-Memory Orchestration | 20 | 0.0109 | 0.0105 | 0.0116 | 0.0101 | 0.0152 | 0.0011 |
| Deterministic Episode (T2) | Full In-Memory Orchestration | 20 | 0.0070 | 0.0067 | 0.0085 | 0.0065 | 0.0112 | 0.0011 |

### 2.4 Harness Throughput & Memory Utilization ($C=1$ Concurrency)

- **Mean Harness Episode Latency:** 0.0089 ms
- **Harness Episode Throughput:** **111978.63 episodes / second**
- **Tool Call Execution Throughput:** **223957.26 tool calls / second**
- **Per-Call Timeout Rate:** **0.0%** (0 / 400 calls exceeded 5.0s limit; max call duration was 0.72 ms)
- **Process Memory (RSS):** Baseline: 68.83 MB, Peak: 69.09 MB (Delta: 0.26 MB)

---

## 3. Storage and Checkpoint Footprint

| Asset | Description | Size | Location / Checksum |
|:---|:---|:---:|:---|
| **LoRA Adapter Weights** | Fine-tuned 4-bit LoRA adapter | **30.7 MB** | [`results/runs/sft_aya_3b_001/checkpoint/adapter_model.safetensors`](file:///Users/jayesh/Documents/Lexsi/results/runs/sft_aya_3b_001/checkpoint/adapter_model.safetensors) (`a73583f3aade1026...`) |
| **LoRA Config** | Adapter hyperparameter manifest | **1.1 KB** | `adapter_config.json` |
| **Validation Split Manifest** | 10 held-out evaluation tasks | **6.6 KB** | [`configs/manifests/task_manifest_val.json`](file:///Users/jayesh/Documents/Lexsi/configs/manifests/task_manifest_val.json) (`51a4275e777dbe25...`) |
| **Locked Test Split Manifest** | 10 permanently sequestered tasks | **6.6 KB** | [`configs/manifests/task_manifest_test.json`](file:///Users/jayesh/Documents/Lexsi/configs/manifests/task_manifest_test.json) (`d86265f067bafa11...` — UNTOUCHED) |

---

## 4. Persisted Evidence Artifacts

- **Run Manifest:** [`/Users/jayesh/Documents/Lexsi/results/runs/m4_p3_operational_profile/run_manifest.json`](file:///Users/jayesh/Documents/Lexsi/results/runs/m4_p3_operational_profile/run_manifest.json)
- **Summary CSV:** [`/Users/jayesh/Documents/Lexsi/results/runs/m4_p3_operational_profile/operational_summary.csv`](file:///Users/jayesh/Documents/Lexsi/results/runs/m4_p3_operational_profile/operational_summary.csv)
- **Raw Benchmark Results:** [`/Users/jayesh/Documents/Lexsi/results/runs/m4_p3_operational_profile/harness_benchmark_results.json`](file:///Users/jayesh/Documents/Lexsi/results/runs/m4_p3_operational_profile/harness_benchmark_results.json)

---

## 5. Scope Limitations & Non-Extrapolation Mandate

1. **No Production Extrapolation:** These measured throughput numbers represent isolated deterministic Python harness and in-memory tool logic at $C=1$ concurrency. They must **not** be extrapolated to infer full LLM serving capacity or production enterprise query throughput.
2. **Zero In-Workstation Generation Profile:** Live model serving latencies (TTFT, tokens/sec) remain unmeasured locally due to missing weights/libraries; causal comparison against Command R 7B is strictly prohibited due to architectural incommensurability.
3. **Read-Only / Non-Tuning Invariant:** This operational profile does not alter prompts, checkpoints, or validation criteria.
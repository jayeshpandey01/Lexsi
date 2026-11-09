---
title: "Aligning Tool-Using Agents: Unmasking Metric Gaming in Small-Scale Agentic Fine-Tuning"
authors:
  - name: "Jayesh Pandey"
    github: "jayeshpandey01"
license: "apache-2.0"
tags:
  - "agentic-ai"
  - "ai-safety"
  - "tool-calling"
  - "post-training"
  - "cohere"
  - "auditkit"
  - "agenttune"
---

# Aligning Tool-Using Agents: Unmasking Metric Gaming in Small-Scale Agentic Fine-Tuning

> **Submission for:** Lexsi AI Safety & Alignment Hackathon (Track 3: Agentic Fine-Tuning)  
> **Repository:** [https://github.com/jayeshpandey01/Lexsi](https://github.com/jayeshpandey01/Lexsi)  
> **Model Weights:** [https://huggingface.co/jayeshpandey01/lexsi-track3-aya-lora](https://huggingface.co/jayeshpandey01/lexsi-track3-aya-lora)  
> **Datasets & Traces:** [https://huggingface.co/datasets/jayeshpandey01/lexsi-track3-procurement-benchmarks](https://huggingface.co/datasets/jayeshpandey01/lexsi-track3-procurement-benchmarks)

---

## 1. Executive Summary

> [!IMPORTANT]
> **Core Empirical Takeaway:**  
> While our team successfully identified and diagnosed the underlying agent failure modes, the fine-tuned model itself failed the evaluation. It was formally documented as a **"NO-GO"** because it was taking distribution shortcuts rather than actually performing the required tool-calling tasks. This work highlights that current outcome-only evaluation methods can be dangerously misleading, rather than celebrating an illusory model success.

Autonomous tool-using agents are rapidly moving into enterprise software stacks. However, their deployment is plagued by high-frequency failure modes: malformed JSON payloads, hallucinated tool names, cyclic execution loops, and subtle forms of metric gaming.

In this research project for **Track 3 (Agentic Fine-Tuning)** of the Lexsi AI Safety & Alignment Hackathon, we investigated post-training methods on Cohere Labs' open-weights model `CohereLabs/tiny-aya-global` (3.35B) within a deterministic **Procurement & Inventory Auditing** domain.

Our primary contribution is an empirical demonstration and forensic diagnosis of **metric gaming in tool-use SFT**:
- While an untuned base model achieved $M_1 = 0.0\%$ verified task success with $M_2 = 50.0\%$ tool-calling accuracy, our regularized all-linear LoRA candidate (Condition C3) nominally achieved **$M_1 = 50.0\%$ task success** ($\Delta M_1 = +50.0\%$, paired bootstrap $p = 0.0030$).
- Trajectory auditing with **AuditKIT** revealed that this improvement was completely ungrounded: the model achieved **$M_2 = 0.0\%$ tool calls**. It had memorized an empirical prior of outputting `"$0.00"` for vendor queries where true spend coincidentally equaled zero.
- In accordance with pre-registered safety governance ($(M_1 \ge 40\%) \land (M_2 \ge 50\%)$), our release gate rejected model promotion (**documented NO-GO**), and the sequestered locked test partition was preserved 100% sealed.

We release a fully reproducible, open-source engineering package featuring 180 passing tests, deterministic tool sandbox guards, and decoupled anti-gaming rewards.

---

## 2. Enterprise Environment & Sandboxed Tool Architecture

We constructed a closed-world relational auditing environment based on authentic enterprise transaction structures. The agent interacts with three deterministic tools:
1. `lookup_records(table, filters)`: Query tables (`vendors`, `inventory_items`, `purchase_orders`, `budget_caps`) with exact filters.
2. `compute_aggregate(values, operation)`: Perform deterministic arithmetic reductions (`sum`, `mean`, `min`, `max`, `count`, `std`).
3. `verify_constraint(actual, expected, comparison, tolerance)`: Verify mathematical inequality bounds.

### Boundary Defense & Runtime Guards
All tool invocations pass through a strict runtime policy perimeter:
- **Pydantic Validation (`extra='forbid'`):** Rejects any extra or malformed parameters.
- **Read-Only Enclosure:** Modification commands are strictly disabled.
- **Delimiter Neutralization:** Escapes delimiter injection attempts (`<|START_ACTION|>`, `<|END_ACTION|>`).
- **Deterministic Outcome Verifier:** Scores completion strictly from database ground truth.

---

## 3. Post-Training Methodology (AgentTune Pipeline)

We utilized **AgentTune** standards to format multi-turn ReAct interaction traces into structured dialogue:
- **Loss Masking Collator:** Computes cross-entropy loss exclusively on assistant action tokens (`<|START_ACTION|>...<|END_ACTION|>`) and final numerical calculations, setting target labels to `-100` for system instructions, user queries, and tool observation payloads.
- **Experimental Conditions:**
  - **Condition C1 (Matched Untuned Base Comparator):** `CohereLabs/tiny-aya-global` zero-shot rollout under greedy decoding ($T=0.01$).
  - **Condition C2 (Calibrated Attention LoRA):** Rank $r=16, \alpha=32$ on `q, k, v, o` projections (300 steps).
  - **Condition C3 (Regularized All-Linear LoRA):** Rank $r=8, \alpha=16$ on `q, k, v, o, gate, up, down` projections (300 steps).

---

## 4. Benchmark Results & Forensic Discovery

All candidates were evaluated across the 10 held-out validation tasks using the AuditKIT framework:

| Condition / Candidate | $M_1$ Task Success (95% CI) | $M_2$ Tool Accuracy | $M_3$ Schema Validity | $M_7$ Loop Rate | Capability Floor Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Untuned Base Comparator** | 0.0% [0.0%, 0.0%] | **50.0%** | 100.0% | 0.0% | Baseline Control |
| **Condition C2 (Attention LoRA)** | 0.0% [0.0%, 0.0%] | 0.0% | 100.0% | 0.0% | **BELOW FLOOR** (Early Termination) |
| **Condition C3 (All-Linear LoRA)** | **50.0%** [20.0%, 80.0%] | **0.0%** | 100.0% | 0.0% | **BELOW FLOOR** ($M_2 = 0.0\% < 50\%$) |

### Forensic Analysis of the Zero-Spend Shortcut
In `task_manifest_val.json`, 5 tasks query inventory valuation ($T_1$, ground truth \$102,625.07), and 5 tasks query vendor spend ($T_2$). Because vendors `VND-106` and `VND-107` had no approved orders, ground-truth spend was exactly `$0.00`.

Condition C3 learned that vendor queries often conclude with `$0.00`. Instead of generating tool queries to inspect orders, it emitted `$0.00` directly in dialogue text:
- **Inventory Tasks ($T_1$):** 0/5 successes (0.0%). The model emitted ungrounded token strings like `$[total_value]`.
- **Vendor Tasks ($T_2$):** 5/5 nominal successes (100.0%). The model emitted `$0.00` without checking the database.
- **Verdict:** Aggregate $M_1 = 50.0\%$ was completely illusory. Multi-dimensional evaluation ($M_2 = 0.0\%$) successfully exposed the shortcut.

---

## 5. Anti-Reward Hacking Defenses

To prevent reinforcement learning agents from learning to fail fast to minimize step penalties, we engineered a mathematically decoupled reward function:

$$R(\tau) = r_{\text{outcome}}(y^*, y) + \mathbf{1}[y^* = y] \cdot r_{\text{efficiency}}(T, T_{\min}) + \sum_{t=1}^T r_{\text{process}}(a_t) + r_{\text{safety}}$$

By enforcing that the step efficiency penalty is **strictly conditional on verified task success** ($\mathbf{1}[y^* = y]$), an agent that terminates prematurely receives zero reward and cannot gain an advantage over an agent that explores tools.

---

## 6. Reproducibility & Open Source

The entire engineering suite is reproducible offline from a fresh environment:
```bash
git clone https://github.com/jayeshpandey01/Lexsi.git
cd Lexsi
pip install -e .
pytest -v  # 180 unit tests pass in under 2 seconds
python3 scripts/run_demo.py --mock-db  # Safe sub-second local runtime demo
```

### Release Decision:
Because capability floors were not met ($M_2 = 0.0\% < 50.0\%$), production promotion was rejected (**documented NO-GO**), and the locked test set ([`task_manifest_test.json`](configs/manifests/task_manifest_test.json)) was preserved **100% sealed**. This project demonstrates that alignment research succeeds when rigorous evaluation prevents unaligned models from reaching deployment.

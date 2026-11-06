# AgentTune Technical Reference & Compatibility Dossier
## Lexsi Track 3: Agentic Fine-Tuning (Task TASK-M0-03 / M0-04)

> **Document Status:** Authoritative Library Integration Notes  
> **Source Repository:** [Lexsi-Labs/AgentTune](https://github.com/Lexsi-Labs/AgentTune)  
> **Documentation:** [https://agenttune.lexsi.ai](https://agenttune.lexsi.ai)  
> **Traceability:** Maps to `docs/M0.md` M0-04, `docs/plan.md` P0.4, Section 2.2 Watchlist.

---

## 1. Architectural Overview

AgentTune is a unified post-training framework designed for tool-using LLM agents. Its key distinguishing architectural feature is the **Two-Tier EventLog Representation**:
* **Tier 1 (Execution Events):** High-fidelity chronological logs capturing the exact sequence of model outputs, tool calls, raw environment observations, execution latencies, and error codes.
* **Tier 2 (Training Projections):** Formatted data representations projected from Tier 1 events for:
  - Masked SFT: Assistant tokens masked for cross-entropy, observation tokens masked out.
  - Preference / RL Tuning: Paired trajectories $(y_w, y_l)$ for DPO, or group rollouts $\{y_1, \dots, y_G\}$ for GRPO with normalized relative advantage.

---

## 2. Pinned PR Watchlist & Caveats

| PR ID | Title / Focus | Upstream Impact | Project Handling / Invariant |
|:---|:---|:---|:---|
| **PR #5** | *Lazy-loading built-in tools & `--tools` training CLI* | Avoids eager loading of all tool modules at startup; allows modular custom tool injection. | We prefer the explicit Python API `AgentTuneConfig` / `EventLog` over fragile CLI invocation. Our custom tool engine in `src/environment/tools.py` will expose explicit schema descriptors. |
| **PR #6** | *Fix non-finite judge scores treated as high reward* | Previously, NaN / Inf or unparseable judge outputs defaulted to float max, corrupting RL policy gradients. | **Mandatory Guard:** All reward computations are strictly validated via Pydantic; any non-finite value or unverified status defaults strictly to zero reward with a logged alert. Verified deterministic outcome predicates are used instead of LLM judges. |

---

## 3. Supported Model & Data Contracts

### 3.1 Model Backends
* Supports Hugging Face `AutoModelForCausalLM` with PyTorch.
* Target model for this project: `CohereLabs/tiny-aya-global` (3.35B) using Command-R chat template and tool delimiter tokens.

### 3.2 Dataset Format (EventLog Schema)
```json
{
  "episode_id": "ep_20261105_001",
  "task_id": "procurement_aggregation_042",
  "seed": 42,
  "events": [
    {
      "event_type": "user_query",
      "timestamp": 1793836800.0,
      "content": "Calculate total spend for vendor VND-101 on Approved orders."
    },
    {
      "event_type": "agent_action",
      "timestamp": 1793836801.2,
      "tool_name": "lookup_records",
      "arguments": {"table": "purchase_orders", "filters": {"vendor_id": "VND-101", "order_status": "Approved"}}
    },
    {
      "event_type": "tool_observation",
      "timestamp": 1793836801.3,
      "result": [{"po_id": "PO-9001", "quantity": 10, "unit_price": 50.0}]
    },
    {
      "event_type": "agent_action",
      "timestamp": 1793836802.1,
      "tool_name": "compute_aggregate",
      "arguments": {"values": [500.0], "operation": "sum"}
    },
    {
      "event_type": "tool_observation",
      "timestamp": 1793836802.15,
      "result": 500.0
    },
    {
      "event_type": "agent_response",
      "timestamp": 1793836803.0,
      "content": "The total spend for vendor VND-101 across Approved orders is $500.00."
    }
  ]
}
```

---

## 4. Verification & Compatibility Bridge

To maintain 100% decoupling, `src/utils/compatibility.py` implements serialization routines to translate between Lexsi internal `TrajectoryRecord` objects and AgentTune EventLog specifications.


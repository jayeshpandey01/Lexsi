# AuditKIT Technical Reference & Compatibility Dossier
## Lexsi Track 3: Diagnostic Evaluation Spine (Task TASK-M0-03 / M0-04)

> **Document Status:** Authoritative Evaluation Library Notes  
> **Source Repository:** [Lexsi-Labs/AuditKIT](https://github.com/Lexsi-Labs/AuditKIT)  
> **Documentation:** [https://auditkit.lexsi.ai](https://auditkit.lexsi.ai)  
> **Traceability:** Maps to `docs/M0.md` M0-04, `docs/plan.md` P0.4, Section 2.2 Watchlist.

---

## 1. Architectural Overview

AuditKIT is a unified evaluation spine organized around a strict 5-stage pipeline:
$$\text{Dataset} \longrightarrow \text{Adapter} \longrightarrow \text{Model} \longrightarrow \text{Metrics} \longrightarrow \text{RunResult}$$

Key architectural guarantees:
* **Zero External Dependencies for Core:** Runs entirely on standard Python libraries.
* **Deterministic Provenance Fingerprinting:** Generates a canonical SHA-256 fingerprint for every evaluation run, combining dataset hash, model revision, metric configurations, and environment parameters.
* **Orchestration Entry Point:** `ak.evaluate(...)` returns a typed `RunResult` object containing itemized scores, statistical aggregations, and failure traces.

---

## 2. Pinned PR Watchlist & Caveats

| PR ID | Title / Focus | Upstream Impact | Project Handling / Invariant |
|:---|:---|:---|:---|
| **PR #18** | *Fix text/scorer/loader edge cases and unmeasured values* | Unmeasured or empty model outputs previously threw unhandled exceptions during aggregation or skewed mean calculations. | **Defensive Scoring:** In `src/evaluation/metrics.py`, any unmeasured, empty, or timed-out run is explicitly recorded with a typed `EvaluationStatus.FAILED` or `EvaluationStatus.TIMEOUT` and assigned a numerical score of $0.0$, accompanied by a structured error code. |
| **PR #9** | *Fix paired-bootstrap significance calculation behavior* | Significance intervals under small sample sizes ($N < 50$) exhibited high variance or degenerate confidence intervals. | **Verification Protocol:** All inferential statistical comparisons (p-values, 95% confidence intervals) are cross-checked using a local deterministic bootstrap routine in `src/evaluation/metrics.py` before inclusion in final reports. |

---

## 3. Metric Taxonomy & RunResult Contract

### 3.1 Metric Definitions (M1–M7)
* **M1: Task Success Rate** ($\% \text{ verified } R_{\text{outcome}} = 1$)
* **M2: Tool Call Selection Accuracy** ($\frac{\text{Correct Tools Selected}}{\text{Total Tool Calls}}$)
* **M3: Argument Schema Validity** ($\frac{\text{Schema-Valid Invocations}}{\text{Total Tool Calls}}$)
* **M4: Call Order Validity** ($\frac{\text{Valid Sequences}}{\text{Multi-Step Opportunities}}$)
* **M5: Trajectory Step Efficiency** ($\mathbb{E}[T \mid \text{Success} = 1]$)
* **M6: Redundant Call Frequency** ($\frac{\text{Duplicate / Superfluous Calls}}{\text{Total Tool Calls}}$)
* **M7: Loop & Stall Frequency** ($\frac{\text{Runs Exceeding } T_{\max}}{\text{Total Evaluated Runs}}$)

### 3.2 AuditKIT Compatibility Bridge
In `src/utils/compatibility.py`, `AuditKitAdapter` maps Lexsi trajectory records directly into AuditKIT's dataset and metric schemas, enabling direct export to CSV, JSON, and Markdown summary tables.

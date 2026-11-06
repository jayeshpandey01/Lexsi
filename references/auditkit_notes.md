# AuditKIT Technical Reference & Compatibility Dossier
## Lexsi Track 3: Diagnostic Evaluation Spine (Milestone M1.P4)

> **Document Status:** Authoritative Evaluation Library & Compatibility Adapter Notes  
> **Source Repository:** [Lexsi-Labs/AuditKIT](https://github.com/Lexsi-Labs/AuditKIT)  
> **Documentation:** [https://auditkit.lexsi.ai](https://auditkit.lexsi.ai)  
> **Traceability:** Maps to `docs/M1.md` M1.P4, `docs/plan.md` P2.1, Section 2.2 Watchlist, `system_design.md` §2.5.

---

## 1. Architectural Overview & Integration Boundary

AuditKIT serves as the unified evaluation spine for evaluating language models and agentic systems across a 5-stage pipeline:
$$\text{Dataset} \longrightarrow \text{Adapter} \longrightarrow \text{Model} \longrightarrow \text{Metrics} \longrightarrow \text{RunResult}$$

### Integration Boundary & Execution Modes
* **Local Adapter Mode (`AuditKitAdapter` in `src/utils/compatibility.py`):** The primary execution mode in this repository. In sandboxed or offline environments where the native `auditkit` package is not pre-installed, `AuditKitAdapter` executes the deterministic local evaluation spine. It produces identical Pydantic `AuditKitRunResult` contracts, directional metadata, per-task predictions, and SHA-256 fingerprints without external network or library dependencies.
* **Native AuditKIT Pass-Through:** If the native `auditkit` package is installed (`pip install auditkit`), `AuditKitAdapter.is_native_auditkit_available()` returns `True`, allowing seamless delegation to `ak.evaluate(...)`. The system transparently documents its execution mode via `evaluator_version` (`local-adapter-1.0.0` vs. `native-auditkit-<version>`).

---

## 2. Upstream PR Watchlist & Defensive Invariants

| PR ID | Title / Focus | Upstream Impact | Project Handling / Invariant |
|:---|:---|:---|:---|
| **PR #18** | *Fix text/scorer/loader edge cases and unmeasured values* | Unmeasured or empty model outputs previously threw unhandled exceptions during aggregation or skewed mean calculations. | **Defensive Scoring & Unmeasured Handling:** In `src/evaluation/metrics.py`, non-finite scores (`nan`, `inf`) are safely rejected or clamped to `0.0`. For conditional metrics (M5 Step Efficiency), when zero tasks succeed, the metric returns `None` with `sample_count = 0` and `unmeasured_count > 0` rather than false `0.0` steps. Missing or dropped tasks count in M1's denominator as $0.0$. |
| **PR #9** | *Fix paired-bootstrap significance calculation behavior* | Significance intervals under small sample sizes ($N < 50$) exhibited high variance or degenerate confidence intervals. | **Statistical Verification Protocol:** We strictly decouple single-sample descriptive uncertainty from comparative hypothesis testing. Single-sample CIs are generated via `compute_bootstrap_ci`. Comparative assessments are executed via `compute_paired_bootstrap_comparison`, aligning task pairs, resampling paired differences $D_i = Y_i - X_i$, computing empirical p-values against $H_0: \Delta = 0$, and attaching small-sample caveats when $N < 30$. |

---

## 3. Metric Taxonomy & RunResult Contract

### 3.1 Metric Definitions (M1–M7)

| Metric ID | Metric Name | Direction | Unit | Mathematical Definition | Denominator & Edge-Case Handling |
|:---|:---|:---:|:---:|:---|:---|
| **M1** | Task Success Rate | Maximize | Fraction $[0, 1]$ | $\frac{1}{N_{\text{tasks}}} \sum_{i=1}^{N_{\text{tasks}}} \mathbb{I}(\text{Task}_i = \text{Success})$ | Verified exclusively by `EnvironmentOutcomeVerifier`. Aligned to task manifest; dropped tasks count as $0.0$. |
| **M2** | Tool Call Accuracy | Maximize | Fraction $[0, 1]$ | $\frac{\text{Correct Tool Choices}}{\max(1, \text{Total Tool Calls})}$ | Measures call precision; also logs task tool coverage $\frac{\|\text{Called} \cap \text{Expected}\|}{\|\text{Expected}\|}$ in details. |
| **M3** | Schema Validity | Maximize | Fraction $[0, 1]$ | $1.0 - \frac{N_{\text{arg\_errors}}}{\max(1, N_{\text{calls}})}$ | Audits all argument errors: `SCHEMA_VALIDATION_ERROR`, `SEMANTIC_BOUND_ERROR`, `MALFORMED_TOOL_ARGUMENTS`, `JSON_PARSE_ERROR`. |
| **M4** | Call Order Validity | Maximize | Fraction $[0, 1]$ | $\frac{N_{\text{valid\_ordered}}}{N_{\text{multi\_step}}}$ | Audits topological prerequisite rules (e.g. `lookup_records` before `compute_aggregate`) across all occurrences. |
| **M5** | Trajectory Step Efficiency | Minimize | Steps | $\mathbb{E}[T \mid \text{Success} = 1] = \frac{1}{\|\mathcal{S}\|} \sum_{i \in \mathcal{S}} T_i$ | Conditional expectation on verified success. Returns `None` (unmeasured) if $\|\mathcal{S}\| = 0$. |
| **M6** | Redundant Call Rate | Minimize | Fraction $[0, 1]$ | $\frac{N_{\text{duplicate\_calls}}}{\max(1, N_{\text{calls}})}$ | Identifies duplicate invocations using canonical JSON argument serialization (`json.dumps(args, sort_keys=True)`). |
| **M7** | Loop & Stall Frequency | Minimize | Fraction $[0, 1]$ | $\frac{N_{\text{stalls}}}{N_{\text{tasks}}}$ | Detects: (1) $T \ge T_{\max}$ without success, (2) timeouts, (3) 3+ consecutive duplicate calls, and (4) alternating 2-cycles. |

### 3.2 AuditKIT Compatibility Bridge & Serialization

Implemented in `src/utils/compatibility.py`:
* **`AuditKitRunResult`**: The canonical evaluation record.
* **`TaskEvaluationResult`**: Preserves per-task predictions, executed tools, expected tools, error codes, and diagnosed failure categories.
* **`PairedComparisonResult`**: Captures comparative deltas $\Delta$, bootstrap CIs, empirical p-values, sample counts, and statistical caveats.
* **Export & Persistence**:
  - `run_result.to_json()`: Serializes canonical JSON.
  - `run_result.to_csv()`: Generates standard RFC 4180 CSV summary.
  - `run_result.to_markdown_table()`: Renders GitHub-style Markdown tables with directional arrows and failure distributions.
  - `run_result.save(output_dir)`: Atomically writes `run_result.json`, `metrics_summary.csv`, and `eval_report.md` into `results/runs/<run_id>/`.

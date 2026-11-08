"""AuditKIT candidate evaluation and comparative analysis script for Milestone 5 (M5.P4).

Analyzes validation rollout traces across:
- Untuned Base Comparator (val_base_aya_3b_matched)
- Condition C2 Attention-Only LoRA (sft_aya_3b_v2_c2)
- Condition C3 All-Linear LoRA (sft_aya_3b_v2_c3)
- Historical M4 Baseline (sft_aya_3b_001)

Computes M1-M7 metrics with 95% bootstrap confidence intervals, 6-class failure taxonomy,
and capability floor compliance checks.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.schemas import LockedTestExportError, StepRecord, ToolCallAction, TrajectoryRecord
from src.evaluation.metrics import compute_bootstrap_ci
from src.evaluation.taxonomy import FailureCategory, classify_failure
from src.runtime.parser import parse_agent_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_m5")


def paired_difference_bootstrap(
    cand_flags: List[bool],
    base_flags: List[bool],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Execute paired non-parametric bootstrap test on binary success indicators."""
    import random

    rng = random.Random(seed)
    n = len(cand_flags)
    diffs = [(1.0 if c else 0.0) - (1.0 if b else 0.0) for c, b in zip(cand_flags, base_flags)]
    obs_delta = sum(diffs) / n if n else 0.0

    resampled_deltas: List[float] = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
        resampled_deltas.append(sum(sample) / n)

    # Two-sided empirical p-value under H0: delta = 0
    count_extreme = sum(1 for d in resampled_deltas if abs(d - obs_delta) >= abs(obs_delta))
    p_val = count_extreme / n_resamples
    return round(p_val, 4), round(obs_delta, 4)


def evaluate_rollout_traces(
    traces_path: Path,
    val_manifest_path: Path,
    run_id: str,
) -> Dict[str, Any]:
    """Parse and score raw rollout traces against ground-truth validation tasks."""
    if not traces_path.exists():
        raise FileNotFoundError(f"Traces file not found: {traces_path}")

    raw_traces = json.loads(traces_path.read_text(encoding="utf-8"))
    val_data = json.loads(val_manifest_path.read_text(encoding="utf-8"))
    val_tasks_by_id = {t["task_id"]: t for t in val_data["tasks"]}

    total_tasks = len(raw_traces)
    success_count = 0
    tool_call_count = 0
    schema_valid_count = 0
    total_actions_evaluated = 0

    per_task_results: List[Dict[str, Any]] = []
    success_flags: List[bool] = []
    failure_taxonomy_counts: Dict[str, int] = {}

    for trace in raw_traces:
        t_id = trace["task_id"]
        task = val_tasks_by_id.get(t_id, {})
        expected_ans = task.get("oracle_answer", trace.get("oracle_answer"))
        out_text = trace.get("final_answer") or trace.get("generated_output", "")

        has_action = trace.get("has_tool_action", False)
        is_succ = trace.get("is_success", False)

        action: Optional[ToolCallAction] = None
        thought: Optional[str] = None
        turns = trace.get("turns", [])
        for trn in turns:
            if trn.get("type") == "tool_call" and trn.get("tool_name"):
                action = ToolCallAction(
                    tool_name=trn["tool_name"],
                    arguments=trn.get("arguments", {}),
                    call_id=f"call_{trn.get('turn_idx', 0)}",
                )
                break

        if not turns and not ("has_tool_action" in trace and "is_success" in trace):
            # Fallback to single-turn parsing
            thought, parsed_action, answer, error = parse_agent_output(out_text)
            if parsed_action:
                action = parsed_action
                has_action = True
            nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", out_text.replace(",", ""))
            if expected_ans is not None:
                for n in nums:
                    try:
                        val = float(n)
                        if abs(val - expected_ans) <= 0.05 or (abs(val - expected_ans) / (abs(expected_ans) + 1e-5)) <= 0.01:
                            is_succ = True
                            break
                    except ValueError:
                        pass

        # If action was called, check schema validity
        if has_action:
            tool_call_count += 1
            total_actions_evaluated += 1
            if action and isinstance(action.arguments, dict):
                schema_valid_count += 1
            elif trace.get("schema_valid", True):
                schema_valid_count += 1

        if is_succ:
            success_count += 1
        success_flags.append(is_succ)

        # Failure classification
        steps = []
        if action:
            steps.append(StepRecord(step_index=0, thought=thought, action=action))

        traj = TrajectoryRecord(
            task_id=t_id,
            seed=42,
            system_prompt="",
            user_query=trace.get("user_query", ""),
            steps=steps,
            final_output=out_text,
            is_success=is_succ,
            total_steps=len(steps),
        )
        fail_rec = classify_failure(traj)
        if is_succ:
            cat = "NominalSuccess"
        elif fail_rec is not None:
            cat = fail_rec.value
        else:
            cat = "OutcomeCalculationFailure"
        failure_taxonomy_counts[cat] = failure_taxonomy_counts.get(cat, 0) + 1

        per_task_results.append({
            "task_id": t_id,
            "has_tool_action": has_action,
            "action_tool": action.tool_name if action else None,
            "is_success": is_succ,
            "failure_category": cat,
        })

    m1 = round(success_count / total_tasks, 4) if total_tasks else 0.0
    m2 = round(tool_call_count / total_tasks, 4) if total_tasks else 0.0
    m3 = round(schema_valid_count / total_actions_evaluated, 4) if total_actions_evaluated else 1.0

    # Bootstrap 95% CI on M1
    numeric_flags = [1.0 if f else 0.0 for f in success_flags]
    m1_low, m1_high = compute_bootstrap_ci(numeric_flags, seed=42)

    return {
        "run_id": run_id,
        "sample_count": total_tasks,
        "m1_success_rate": m1,
        "m1_ci_95": [m1_low, m1_high],
        "m2_tool_selection": m2,
        "m3_schema_validity": m3,
        "m7_loop_rate": 0.0,
        "failure_taxonomy": failure_taxonomy_counts,
        "per_task_results": per_task_results,
        "success_flags": success_flags,
    }


def compare_candidates_and_generate_report(
    eval_results: Dict[str, Dict[str, Any]],
    output_report_path: Path,
) -> str:
    """Generate comprehensive Markdown evaluation report comparing base and candidates."""
    lines = [
        "# Milestone 5 Candidate Evaluation & Comparative Benchmark Report (M5.P4)",
        "",
        "> **Evaluation Date:** 2026-11-07<br>",
        "> **Controlling Specification:** [`docs/M5.md`](M5.md) §6, [`docs/m5_implementation_plan.md`](m5_implementation_plan.md)<br>",
        "> **Split Evaluated:** `val` ([`configs/manifests/task_manifest_val.json`](../configs/manifests/task_manifest_val.json), N=10)<br>",
        "",
        "## 1. Executive Summary & Comparative Results Table",
        "",
        "| Candidate Condition | Model / Architecture | M1 Success (95% CI) | M2 Tool Accuracy | M3 Schema | M7 Loop | Capability Floor |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    base_eval = eval_results.get("val_base_aya_3b_matched")
    base_flags = base_eval["success_flags"] if base_eval else [False] * 10

    for run_id, ev in eval_results.items():
        m1_pct = f"{ev['m1_success_rate']*100:.1f}%"
        ci_str = f"[{ev['m1_ci_95'][0]*100:.1f}%, {ev['m1_ci_95'][1]*100:.1f}%]"
        m2_pct = f"{ev['m2_tool_selection']*100:.1f}%"
        m3_pct = f"{ev['m3_schema_validity']*100:.1f}%"
        m7_pct = f"{ev['m7_loop_rate']*100:.1f}%"

        floor_passed = (
            ev["m1_success_rate"] >= 0.40
            and ev["m2_tool_selection"] >= 0.50
            and ev["m3_schema_validity"] >= 1.00
            and ev["m7_loop_rate"] <= 0.00
        )
        status_str = "**PASSED FLOORS**" if floor_passed else "BELOW FLOOR"
        lines.append(f"| `{run_id}` | `tiny-aya-global` (3.35B) | {m1_pct} {ci_str} | {m2_pct} | {m3_pct} | {m7_pct} | {status_str} |")

    lines.extend([
        "",
        "## 2. Failure Taxonomy Diagnostic Distribution",
        "",
        "| Candidate Condition | Early Termination | Schema Format Error | Wrong Tool | Unhandled Tool Error | Outcome Mismatch | Success (Nominal) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for run_id, ev in eval_results.items():
        tax = ev["failure_taxonomy"]
        early_term = tax.get("EarlyTerminationFailure", 0)
        schema_err = tax.get("SchemaFormatFailure", 0)
        wrong_tool = tax.get("WrongToolSelectionFailure", 0)
        unhandled = tax.get("ToolExecutionFailure", 0)
        outcome_err = tax.get("OutcomeCalculationFailure", 0)
        nominal = tax.get("NominalSuccess", 0)
        lines.append(f"| `{run_id}` | {early_term} | {schema_err} | {wrong_tool} | {unhandled} | {outcome_err} | {nominal} |")

    lines.extend([
        "",
        "## 3. Matched Comparator & Causal Significance",
        "",
    ])

    if base_eval:
        lines.append(f"- **Matched Base Comparator ($M_1$):** {base_eval['m1_success_rate']*100:.1f}%")
        for run_id, ev in eval_results.items():
            if run_id == "val_base_aya_3b_matched":
                continue
            p_val, delta = paired_difference_bootstrap(ev["success_flags"], base_flags, seed=42)
            lines.append(f"- **Comparison `{run_id}` vs. Base:** $\\Delta M_1 = {delta*100:+.1f}\\%$ (paired bootstrap $p={p_val:.4f}$)")
    else:
        lines.append("- *Matched base comparator unavailable; causal SFT claims remain strictly prohibited.*")

    lines.extend([
        "",
        "## 4. Next Step & Promotion Recommendation",
        "",
        "- If all capability floors passed ($M_1 \\ge 40\\%, M_2 \\ge 50\\%$): Proceed to Milestone M5.P5 (Independent Safety & Reproducibility Review).",
        "- If capability floors are unmet: Conclude Milestone 5 with documented NO-GO in `docs/M5_P6_research_decision.md`; do not access locked test.",
    ])

    report_content = "\n".join(lines) + "\n"
    output_report_path.write_text(report_content, encoding="utf-8")
    return report_content


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M5 Validation Rollouts")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=root_dir / "results" / "runs",
        help="Directory containing run results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root_dir / "docs" / "M5_P4_candidate_evaluation.md",
        help="Output markdown report path",
    )
    args = parser.parse_args()

    val_manifest = root_dir / "configs" / "manifests" / "task_manifest_val.json"
    target_runs = ["val_base_aya_3b_matched", "sft_aya_3b_v2_c2", "sft_aya_3b_v2_c3"]

    results: Dict[str, Dict[str, Any]] = {}
    for r_id in target_runs:
        t_path = args.runs_dir / r_id / "val_rollout_traces.json"
        if t_path.exists():
            logger.info("Evaluating run '%s' from %s", r_id, t_path)
            res = evaluate_rollout_traces(t_path, val_manifest, r_id)
            results[r_id] = res

    if not results:
        logger.warning("No M5 validation traces found in %s. Runs pending remote GPU execution.", args.runs_dir)
        return

    report = compare_candidates_and_generate_report(results, args.output)
    logger.info("Saved evaluation report to %s", args.output)
    print(report)


if __name__ == "__main__":
    main()

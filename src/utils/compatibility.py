"""Compatibility adapters and bridge interfaces for AgentTune and AuditKIT.

Authoritative schemas and defensive handlers for upstream PR caveats:
- AgentTune PR #5 (lazy tool loading) and PR #6 (non-finite judge scores).
- AuditKIT PR #18 (unmeasured edge cases) and PR #9 (bootstrap confidence intervals).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class AgentTuneEvent(BaseModel):
    """Single chronological event in AgentTune Two-Tier EventLog."""

    event_type: str = Field(
        ...,
        description="Event category: 'user_query', 'agent_action', 'tool_observation', 'agent_response'",
    )
    timestamp: float = Field(..., description="UNIX epoch timestamp in seconds")
    tool_name: Optional[str] = Field(None, description="Invoked tool name if agent_action")
    arguments: Optional[Dict[str, Any]] = Field(None, description="Serialized tool arguments")
    result: Optional[Any] = Field(None, description="Normalized environment observation result")
    content: Optional[str] = Field(None, description="Textual message payload")
    error_code: Optional[str] = Field(None, description="Normalized error code if failure occurred")


class AgentTuneEventLog(BaseModel):
    """Complete episode trajectory record for AgentTune SFT and RL projections."""

    episode_id: str = Field(..., description="Unique episode identifier")
    task_id: str = Field(..., description="Target benchmark task identifier")
    seed: int = Field(..., description="Deterministic environment seed")
    events: List[AgentTuneEvent] = Field(default_factory=list)

    def to_training_dict(self) -> Dict[str, Any]:
        """Convert EventLog to serialized dictionary representation."""
        return self.model_dump()


class TaskEvaluationResult(BaseModel):
    """Itemized evaluation outcome and diagnostic trace for an individual task."""

    task_id: str = Field(..., description="Unique task identifier")
    template_id: str = Field(default="", description="Task template classification")
    is_success: bool = Field(..., description="Deterministic environment outcome success")
    total_steps: int = Field(default=0, description="Total environment steps taken")
    executed_tools: List[str] = Field(default_factory=list, description="Ordered list of called tool names")
    expected_tools: List[str] = Field(default_factory=list, description="Expected tools declared in task")
    has_schema_error: bool = Field(default=False, description="Whether trajectory incurred argument schema errors")
    has_order_error: bool = Field(default=False, description="Whether trajectory violated tool dependency ordering")
    is_stall: bool = Field(default=False, description="Whether trajectory exhibited looping or stall behavior")
    redundant_calls: int = Field(default=0, description="Count of duplicate tool invocations")
    error_type: Optional[str] = None
    diagnosed_failure: Optional[str] = None


class AuditKitMetricResult(BaseModel):
    """Individual metric evaluation summary matching AuditKIT standards."""

    metric_name: str = Field(..., description="Metric identifier e.g. M1_task_success_rate")
    value: Optional[float] = Field(None, description="Calculated aggregate metric value or None if unmeasured")
    direction: str = Field(default="maximize", description="Optimization goal: 'maximize' or 'minimize'")
    unit: str = Field(default="fraction", description="Measurement unit e.g. 'fraction', 'steps'")
    std: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    sample_count: int = Field(default=0, description="Number of observed evaluations included")
    unmeasured_count: int = Field(default=0, description="Number of unmeasured or missing observations")
    details: Dict[str, Any] = Field(default_factory=dict, description="Supporting diagnostic metadata")

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, v: Optional[float]) -> Optional[float]:
        """AuditKIT PR #18 / AgentTune PR #6 defense: reject non-finite scores."""
        if v is None:
            return None
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v


class PairedComparisonResult(BaseModel):
    """Rigorous comparative evaluation outcome between baseline and candidate runs."""

    metric_name: str = Field(..., description="Metric compared")
    baseline_mean: float = Field(..., description="Baseline sample mean")
    candidate_mean: float = Field(..., description="Candidate sample mean")
    delta: float = Field(..., description="Candidate minus baseline difference: Delta = Y - X")
    ci_lower: float = Field(..., description="Paired bootstrap 95% CI lower bound")
    ci_upper: float = Field(..., description="Paired bootstrap 95% CI upper bound")
    p_value: float = Field(..., description="Empirical two-sided bootstrap p-value for H0: Delta = 0")
    is_significant: bool = Field(..., description="Whether p < 0.05 and CI excludes zero")
    sample_count: int = Field(..., description="Number of paired task evaluations")
    method: str = Field(default="paired_bootstrap_resampling", description="Statistical test method")
    caveats: List[str] = Field(default_factory=list, description="Statistical caveats or small-sample warnings")


class AuditKitRunResult(BaseModel):
    """Canonical RunResult record matching AuditKIT unified evaluation spine."""

    run_id: str = Field(..., description="Unique evaluation run identifier")
    model_id: str = Field(..., description="Evaluated model revision identifier")
    evaluator_version: str = Field(default="local-adapter-1.0.0", description="Evaluator version or commit")
    split: str = Field(default="val", description="Evaluated benchmark partition: train, val, test")
    timestamp: float = Field(..., description="UNIX timestamp of evaluation completion")
    sha256_fingerprint: str = Field(..., description="Deterministic 64-char hex SHA-256 fingerprint")
    headline: Dict[str, Optional[float]] = Field(default_factory=dict, description="Headline metric scores")
    metrics: Dict[str, AuditKitMetricResult] = Field(default_factory=dict, description="Detailed metric objects")
    per_task_results: List[TaskEvaluationResult] = Field(default_factory=list, description="Per-task itemized records")
    failure_breakdown: Dict[str, int] = Field(default_factory=dict, description="Tally of diagnosed failure modes")
    provenance_metadata: Dict[str, Any] = Field(default_factory=dict, description="Environment, config and task hashes")

    def to_dict(self) -> Dict[str, Any]:
        """Convert RunResult to standardized dictionary representation."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize RunResult to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_csv(self) -> str:
        """Export metrics summary table to RFC 4180 CSV string format."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "metric_name",
            "value",
            "direction",
            "unit",
            "ci_lower",
            "ci_upper",
            "sample_count",
            "unmeasured_count",
        ])
        for name, m in sorted(self.metrics.items()):
            val_str = f"{m.value:.4f}" if m.value is not None else "N/A"
            low_str = f"{m.ci_lower:.4f}" if m.ci_lower is not None else "N/A"
            high_str = f"{m.ci_upper:.4f}" if m.ci_upper is not None else "N/A"
            writer.writerow([
                m.metric_name,
                val_str,
                m.direction,
                m.unit,
                low_str,
                high_str,
                m.sample_count,
                m.unmeasured_count,
            ])
        return output.getvalue()

    def to_markdown_table(self) -> str:
        """Export headline benchmark metrics as a formatted GitHub-style Markdown table."""
        lines = [
            f"### AuditKIT Evaluation Report: `{self.run_id}`",
            f"- **Model ID:** `{self.model_id}` | **Evaluator:** `{self.evaluator_version}` | **Split:** `{self.split}`",
            f"- **Provenance Fingerprint:** `{self.sha256_fingerprint}`",
            "",
            "| Metric ID & Name | Direction | Value | 95% Bootstrap CI | Sample Count | Unmeasured |",
            "|:---|:---:|:---:|:---:|:---:|:---:|",
        ]
        for name, m in sorted(self.metrics.items()):
            val_str = f"{m.value:.4f}" if m.value is not None else "*unmeasured*"
            ci_str = f"[{m.ci_lower:.4f}, {m.ci_upper:.4f}]" if m.ci_lower is not None and m.ci_upper is not None else "N/A"
            arrow = "▲" if m.direction == "maximize" else "▼"
            lines.append(
                f"| `{m.metric_name}` | {arrow} {m.direction} | **{val_str}** | {ci_str} | {m.sample_count} | {m.unmeasured_count} |"
            )

        if self.failure_breakdown:
            lines.extend([
                "",
                "#### Failure Taxonomy Breakdown",
                "| Failure Mode Category | Count | Proportion |",
                "|:---|:---:|:---:|",
            ])
            total_fails = sum(self.failure_breakdown.values())
            for cat_name, count in sorted(self.failure_breakdown.items(), key=lambda x: -x[1]):
                prop_str = f"{(count / total_fails):.1%}" if total_fails > 0 else "0.0%"
                lines.append(f"| `{cat_name}` | {count} | {prop_str} |")

        return "\n".join(lines)

    def save(self, output_dir: str | Path) -> Dict[str, str]:
        """Persist canonical run artifacts into results/runs/<run_id>/ directory.

        Writes:
          - run_result.json: Complete provenance and per-task predictions
          - metrics_summary.csv: Tabular CSV metrics
          - eval_report.md: Markdown summary table
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_path = out_path / "run_result.json"
        csv_path = out_path / "metrics_summary.csv"
        md_path = out_path / "eval_report.md"

        json_path.write_text(self.to_json(), encoding="utf-8")
        csv_path.write_text(self.to_csv(), encoding="utf-8")
        md_path.write_text(self.to_markdown_table(), encoding="utf-8")

        return {
            "json": str(json_path.resolve()),
            "csv": str(csv_path.resolve()),
            "markdown": str(md_path.resolve()),
        }

    @classmethod
    def load(cls, run_dir: str | Path) -> "AuditKitRunResult":
        """Load canonical RunResult from results/runs/<run_id>/run_result.json."""
        p = Path(run_dir)
        if p.is_dir():
            p = p / "run_result.json"
        if not p.exists():
            raise FileNotFoundError(f"Run result file not found at {p}")
        return cls.model_validate_json(p.read_text(encoding="utf-8"))


def compute_run_fingerprint(
    model_id: str,
    dataset_hash: str,
    config: Dict[str, Any],
    seed: int,
) -> str:
    """Generate deterministic SHA-256 provenance fingerprint for an evaluation run.

    Combines model identification, dataset hash, configuration dictionary,
    and random seed in a canonical JSON format.
    """
    payload = {
        "model_id": model_id,
        "dataset_hash": dataset_hash,
        "config": config,
        "seed": seed,
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class AuditKitAdapter:
    """Bridge adapter integrating AuditKIT standards with Lexsi Track 3 evaluation spine."""

    @staticmethod
    def is_native_auditkit_available() -> bool:
        """Check whether native Lexsi-Labs/AuditKIT is installed in the environment."""
        try:
            import auditkit  # type: ignore # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def get_evaluator_version() -> str:
        """Return evaluator version string, identifying native vs local adapter."""
        try:
            import auditkit  # type: ignore
            return f"native-auditkit-{getattr(auditkit, '__version__', 'unknown')}"
        except ImportError:
            return "local-adapter-1.0.0"

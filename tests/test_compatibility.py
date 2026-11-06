"""Unit tests for AgentTune and AuditKIT compatibility adapters."""

import math
from src.utils.compatibility import (
    AgentTuneEvent,
    AgentTuneEventLog,
    AuditKitMetricResult,
    AuditKitRunResult,
    compute_run_fingerprint,
)


def test_agenttune_eventlog_serialization() -> None:
    """Verify EventLog serialization into AgentTune format."""
    log = AgentTuneEventLog(
        episode_id="ep_001",
        task_id="task_procure_1",
        seed=42,
        events=[
            AgentTuneEvent(
                event_type="user_query",
                timestamp=100.0,
                content="Find total spend",
            ),
            AgentTuneEvent(
                event_type="agent_action",
                timestamp=101.0,
                tool_name="lookup_records",
                arguments={"table": "vendors"},
            ),
            AgentTuneEvent(
                event_type="tool_observation",
                timestamp=101.5,
                result=[{"vendor_id": "VND-101"}],
            ),
        ],
    )
    d = log.to_training_dict()
    assert d["episode_id"] == "ep_001"
    assert len(d["events"]) == 3
    assert d["events"][1]["tool_name"] == "lookup_records"


def test_auditkit_metric_non_finite_defense() -> None:
    """Verify PR #18 / PR #6 defense: non-finite values safely reset to 0.0."""
    res_nan = AuditKitMetricResult(
        metric_name="task_success_rate",
        value=float("nan"),
        sample_count=10,
    )
    assert res_nan.value == 0.0

    res_inf = AuditKitMetricResult(
        metric_name="task_success_rate",
        value=float("inf"),
        sample_count=10,
    )
    assert res_inf.value == 0.0

    res_valid = AuditKitMetricResult(
        metric_name="task_success_rate",
        value=0.85,
        sample_count=10,
    )
    assert res_valid.value == 0.85


def test_auditkit_run_result_fingerprint() -> None:
    """Verify provenance fingerprint determinism."""
    fp1 = compute_run_fingerprint(
        model_id="CohereLabs/tiny-aya-global",
        dataset_hash="hash_abc123",
        config={"batch_size": 1, "lr": 1e-4},
        seed=42,
    )
    fp2 = compute_run_fingerprint(
        model_id="CohereLabs/tiny-aya-global",
        dataset_hash="hash_abc123",
        config={"lr": 1e-4, "batch_size": 1},  # Reordered dict keys
        seed=42,
    )
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex length


def test_auditkit_serialization_and_persistence(tmp_path) -> None:
    """Verify RunResult to_dict, to_json, to_csv, to_markdown_table, and save."""
    from src.utils.compatibility import (
        AuditKitAdapter,
        AuditKitMetricResult,
        AuditKitRunResult,
        PairedComparisonResult,
        TaskEvaluationResult,
    )

    task_res = TaskEvaluationResult(
        task_id="task_001",
        template_id="T1_inventory_value",
        is_success=True,
        total_steps=2,
        executed_tools=["lookup_records", "compute_aggregate"],
        expected_tools=["lookup_records", "compute_aggregate"],
    )

    metrics = {
        "M1_task_success_rate": AuditKitMetricResult(
            metric_name="M1_task_success_rate",
            value=1.0,
            direction="maximize",
            unit="fraction",
            ci_lower=1.0,
            ci_upper=1.0,
            sample_count=1,
        ),
        "M5_trajectory_step_efficiency": AuditKitMetricResult(
            metric_name="M5_trajectory_step_efficiency",
            value=2.0,
            direction="minimize",
            unit="steps",
            sample_count=1,
        ),
    }

    run_res = AuditKitRunResult(
        run_id="run_smoke_test",
        model_id="CohereLabs/tiny-aya-global",
        evaluator_version="local-adapter-1.0.0",
        split="val",
        timestamp=1730000000.0,
        sha256_fingerprint="a" * 64,
        headline={"M1_task_success_rate": 1.0, "M5_trajectory_step_efficiency": 2.0},
        metrics=metrics,
        per_task_results=[task_res],
        failure_breakdown={"EarlyTerminationFailure": 0},
        provenance_metadata={"seed": 42},
    )

    # 1. to_dict and to_json
    d = run_res.to_dict()
    assert d["run_id"] == "run_smoke_test"
    assert len(d["per_task_results"]) == 1
    json_str = run_res.to_json()
    assert '"run_id": "run_smoke_test"' in json_str

    # 2. to_csv
    csv_str = run_res.to_csv()
    assert "M1_task_success_rate,1.0000,maximize,fraction,1.0000,1.0000,1,0" in csv_str

    # 3. to_markdown_table
    md_str = run_res.to_markdown_table()
    assert "| `M1_task_success_rate` | ▲ maximize | **1.0000** |" in md_str

    # 4. save
    saved = run_res.save(tmp_path / "run_smoke_test")
    assert (tmp_path / "run_smoke_test" / "run_result.json").exists()
    assert (tmp_path / "run_smoke_test" / "metrics_summary.csv").exists()
    assert (tmp_path / "run_smoke_test" / "eval_report.md").exists()

    # 5. PairedComparisonResult
    paired = PairedComparisonResult(
        metric_name="M1_task_success_rate",
        baseline_mean=0.5,
        candidate_mean=0.8,
        delta=0.3,
        ci_lower=0.1,
        ci_upper=0.5,
        p_value=0.01,
        is_significant=True,
        sample_count=50,
    )
    assert paired.delta == 0.3
    assert paired.is_significant is True

    # 6. AuditKitAdapter
    assert isinstance(AuditKitAdapter.is_native_auditkit_available(), bool)
    assert "adapter" in AuditKitAdapter.get_evaluator_version() or "native" in AuditKitAdapter.get_evaluator_version()



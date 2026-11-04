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

"""Unit tests for M4.P3 operational performance profile runner."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.run_m4_p3_operational_profile import (
    benchmark_harness_and_tools,
    compute_distribution,
    compute_sha256,
    get_current_rss_mb,
    run_m4_p3_operational_profile,
    EXPECTED_HASHES,
)


def test_compute_distribution_metrics() -> None:
    """Verify statistical distribution calculation on edge cases and sample data."""
    # Empty case
    empty_res = compute_distribution([])
    assert empty_res["count"] == 0
    assert empty_res["mean"] == 0.0

    # Single element case
    single_res = compute_distribution([42.0])
    assert single_res["count"] == 1
    assert single_res["mean"] == 42.0
    assert single_res["median"] == 42.0
    assert single_res["p95"] == 42.0
    assert single_res["std"] == 0.0

    # Multi-element case
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    multi_res = compute_distribution(data)
    assert multi_res["count"] == 5
    assert multi_res["mean"] == 30.0
    assert multi_res["median"] == 30.0
    assert multi_res["min"] == 10.0
    assert multi_res["max"] == 50.0
    assert multi_res["p95"] == 50.0
    assert multi_res["std"] > 0.0


def test_get_current_rss_mb() -> None:
    """Verify process memory lookup returns positive float."""
    rss = get_current_rss_mb()
    assert isinstance(rss, float)
    assert rss > 0.0


def test_benchmark_harness_and_tools() -> None:
    """Verify isolated tool engine and runtime harness benchmarking across validation workload."""
    results = benchmark_harness_and_tools(num_repetitions=5, warmup_runs=2)
    
    assert "workload_metadata" in results
    assert results["workload_metadata"]["timeouts_observed"] == 0
    assert results["workload_metadata"]["timeout_rate"] == 0.0

    dists = results["timing_distributions_ms"]
    assert "tool_lookup_records_in_memory" in dists
    assert "tool_lookup_records_sqlite_db" in dists
    assert "tool_compute_aggregate" in dists
    assert "tool_verify_constraint" in dists
    assert "guard_sanitize_observation_clean" in dists
    assert "guard_sanitize_observation_injection" in dists
    assert "harness_episode_t1_inventory" in dists
    assert "harness_episode_t2_vendor_spend" in dists

    # Verify positive execution times
    for name, dist in dists.items():
        assert dist["count"] == 5
        assert dist["mean"] >= 0.0
        assert dist["median"] >= 0.0

    tp = results["throughput_profile"]
    assert tp["throughput_episodes_per_sec"] > 0.0
    assert tp["throughput_tool_calls_per_sec"] > 0.0


def test_m4_p3_operational_profile_artifacts(tmp_path: Path) -> None:
    """Verify operational profile runner persists all 4 standardized evidence artifacts."""
    run_id = "test_op_profile_run"
    manifest = run_m4_p3_operational_profile(output_base_dir=tmp_path, run_id=run_id, num_repetitions=3)

    assert manifest["run_id"] == run_id
    assert manifest["status"] == "COMPLETED_PROFILED"
    assert manifest["training_profile"]["wall_clock_time_minutes"] == 28.2
    assert manifest["serving_profile"]["status"] == "BLOCKED_LOCAL_GENERATION"
    assert "CONFIRMED_UNTOUCHED" in manifest["locked_test_isolation"]

    out_dir = tmp_path / run_id
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "operational_summary.csv").exists()
    assert (out_dir / "harness_benchmark_results.json").exists()
    assert (out_dir / "operational_profile_report.md").exists()

    # Verify CSV format
    csv_text = (out_dir / "operational_summary.csv").read_text(encoding="utf-8")
    assert "domain,component_metric,workload_boundary" in csv_text
    assert "tool_lookup_records_in_memory" in csv_text
    assert "BLOCKED: Weights/libs uninstalled" in csv_text


def test_locked_test_sequestration_integrity() -> None:
    """Verify locked-test split remains strictly untouched and preserves its pinned SHA-256 digest."""
    test_manifest_path = EXPECTED_HASHES["locked_test_guard"]["path"]
    expected_hash = EXPECTED_HASHES["locked_test_guard"]["sha256"]

    assert test_manifest_path.exists()
    actual_hash = compute_sha256(test_manifest_path)
    assert actual_hash == expected_hash


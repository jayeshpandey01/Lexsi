"""Comprehensive unit tests for deterministic tool contracts and sandbox engine."""

import pytest
from src.environment.sandbox import SandboxedToolEngine
from src.environment.tools import TOOL_CONTRACTS


@pytest.fixture
def engine() -> SandboxedToolEngine:
    return SandboxedToolEngine(seed=42)


def test_tool_contracts_metadata() -> None:
    """Verify tool contract descriptors."""
    assert "lookup_records" in TOOL_CONTRACTS
    assert "compute_aggregate" in TOOL_CONTRACTS
    assert "verify_constraint" in TOOL_CONTRACTS

    contract = TOOL_CONTRACTS["lookup_records"]
    assert not contract.is_stateful
    assert contract.idempotent


def test_lookup_records_nominal(engine: SandboxedToolEngine) -> None:
    """Verify nominal lookup execution."""
    obs = engine.execute(
        "lookup_records",
        {"table": "vendors", "filters": {"category": "Hardware"}},
    )
    assert not obs.is_error
    assert isinstance(obs.result, list)
    assert len(obs.result) > 0
    for v in obs.result:
        assert v["category"] == "Hardware"


def test_lookup_records_empty_filters(engine: SandboxedToolEngine) -> None:
    """Verify lookup with empty filters returns all records in table."""
    obs = engine.execute("lookup_records", {"table": "budget_caps"})
    assert not obs.is_error
    assert len(obs.result) == 4


def test_lookup_records_invalid_table(engine: SandboxedToolEngine) -> None:
    """Verify invalid table name is caught by schema validation."""
    obs = engine.execute("lookup_records", {"table": "secret_passwords"})
    assert obs.is_error
    assert obs.error_code == "SCHEMA_VALIDATION_ERROR"


def test_compute_aggregate_nominal(engine: SandboxedToolEngine) -> None:
    """Verify nominal aggregate reductions."""
    obs_sum = engine.execute(
        "compute_aggregate",
        {"values": [10.0, 20.0, 30.0], "operation": "sum"},
    )
    assert not obs_sum.is_error
    assert obs_sum.result == 60.0

    obs_mean = engine.execute(
        "compute_aggregate",
        {"values": [10.0, 20.0, 30.0], "operation": "mean"},
    )
    assert not obs_mean.is_error
    assert obs_mean.result == 20.0

    obs_max = engine.execute(
        "compute_aggregate",
        {"values": [10.0, 50.0, 30.0], "operation": "max"},
    )
    assert not obs_max.is_error
    assert obs_max.result == 50.0


def test_compute_aggregate_empty_values(engine: SandboxedToolEngine) -> None:
    """Verify empty values list handling."""
    obs = engine.execute(
        "compute_aggregate",
        {"values": [], "operation": "sum"},
    )
    assert obs.is_error
    assert obs.error_code == "SCHEMA_VALIDATION_ERROR"


def test_verify_constraint_nominal(engine: SandboxedToolEngine) -> None:
    """Verify mathematical constraint verification."""
    obs_eq = engine.execute(
        "verify_constraint",
        {"actual": 100.0, "expected": 100.0, "comparison": "eq"},
    )
    assert not obs_eq.is_error
    assert obs_eq.result is True

    obs_le = engine.execute(
        "verify_constraint",
        {"actual": 120.0, "expected": 100.0, "comparison": "le"},
    )
    assert not obs_le.is_error
    assert obs_le.result is False


def test_unknown_tool_rejection(engine: SandboxedToolEngine) -> None:
    """Verify unknown tool call generates typed error."""
    obs = engine.execute("bash_shell", {"command": "rm -rf /"})
    assert obs.is_error
    assert obs.error_code == "TOOL_NOT_FOUND"


def test_sandbox_determinism_and_reset(engine: SandboxedToolEngine) -> None:
    """Verify engine produces identical results under same seed and resets cleanly."""
    obs1 = engine.execute("lookup_records", {"table": "inventory_items"})
    engine.reset(seed=42)
    obs2 = engine.execute("lookup_records", {"table": "inventory_items"})
    assert obs1.result == obs2.result


def test_extra_parameters_rejected_forbid(engine: SandboxedToolEngine) -> None:
    """Verify strict extra='forbid' validation catches unexpected arguments."""
    obs = engine.execute(
        "lookup_records",
        {"table": "vendors", "filters": {}, "unexpected_extra_field": "injected_val"},
    )
    assert obs.is_error
    assert obs.error_code == "SCHEMA_VALIDATION_ERROR"
    # Ensure underlying tool function was never entered
    assert engine.tool_call_counts["lookup_records"] == 0


def test_call_counter_asserts_no_invocation_on_denial_or_malformed(engine: SandboxedToolEngine) -> None:
    """Verify observable call counters guarantee invalid/denied calls never reach tool logic."""
    engine.reset()
    assert engine.tool_call_counts["compute_aggregate"] == 0

    # 1. Malformed call (empty values list)
    obs_malformed = engine.execute("compute_aggregate", {"values": [], "operation": "sum"})
    assert obs_malformed.is_error
    assert engine.tool_call_counts["compute_aggregate"] == 0

    # 2. Permission denied call
    engine.allowed_tools = {"lookup_records"}
    obs_denied = engine.execute(
        "compute_aggregate",
        {"values": [1.0, 2.0], "operation": "sum"},
    )
    assert obs_denied.is_error
    assert obs_denied.error_code == "DISALLOWED_TOOL"
    assert engine.tool_call_counts["compute_aggregate"] == 0

    # 3. Successful call increments counter
    engine.allowed_tools.add("compute_aggregate")
    obs_valid = engine.execute(
        "compute_aggregate",
        {"values": [10.0, 20.0], "operation": "sum"},
    )
    assert not obs_valid.is_error
    assert engine.tool_call_counts["compute_aggregate"] == 1


def test_semantic_boundary_violations(engine: SandboxedToolEngine) -> None:
    """Verify semantic boundaries enforce finite numbers and valid tolerances."""
    # Negative tolerance in verify_constraint
    obs_neg_tol = engine.execute(
        "verify_constraint",
        {"actual": 10.0, "expected": 10.0, "comparison": "eq", "tolerance": -0.5},
    )
    assert obs_neg_tol.is_error
    assert obs_neg_tol.error_code == "SEMANTIC_BOUND_ERROR"

    # Non-finite values in compute_aggregate
    obs_inf = engine.execute(
        "compute_aggregate",
        {"values": [10.0, float("inf")], "operation": "sum"},
    )
    assert obs_inf.is_error
    assert obs_inf.error_code == "SEMANTIC_BOUND_ERROR"


def test_per_call_timeout(engine: SandboxedToolEngine) -> None:
    """Verify per-call execution timeout is enforced."""
    engine.per_call_timeout_seconds = 0.0  # Force immediate timeout
    obs = engine.execute(
        "compute_aggregate",
        {"values": [1.0, 2.0, 3.0], "operation": "sum"},
    )
    assert obs.is_error
    assert obs.error_code == "CALL_TIMEOUT"


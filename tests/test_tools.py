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

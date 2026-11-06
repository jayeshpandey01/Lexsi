"""Unit tests for trajectory schemas, EventLog projections, and redaction filters."""

import json
from src.data.redaction import redact_data, redact_text
from src.data.schemas import StepRecord, ToolCallAction, TrajectoryRecord


def test_redaction_filter() -> None:
    """Verify secrets, bearer tokens, and emails are scrubbed."""
    raw = (
        "Contact me at admin@corp.com with api_key=sk-secretKey9999 "
        "and header Bearer ya29.a0AfH6SM... Also password: 'SuperSecret123'."
    )
    sanitized = redact_text(raw)
    assert "admin@corp.com" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized
    assert "sk-secretKey9999" not in sanitized
    assert "[REDACTED_CREDENTIAL]" in sanitized
    assert "SuperSecret123" not in sanitized


def test_redact_nested_data() -> None:
    """Verify recursive data structure redaction and bare secret values in dict mappings."""
    payload = {
        "user": {"email": "test@domain.org", "token": "api_key: abc12345678"},
        "credentials": {
            "api_key": "raw_sk_secret_value_123456",
            "client_secret": "raw_super_secret_password_xyz",
            "password": "unprefixed_raw_password",
        },
        "benign_info": {
            "vendor_id": "VND-101",
            "item_name": "Enterprise Server Rack",
            "unit_cost": 450.50,
        },
        "logs": ["Bearer secret-token-xyz12345", 42, 3.14],
        "tuple_items": ("admin@lexsi.corp", 99),
    }
    cleaned = redact_data(payload)
    # Check email and inline tokens
    assert cleaned["user"]["email"] == "[REDACTED_EMAIL]"
    assert "[REDACTED_CREDENTIAL]" in cleaned["user"]["token"]
    assert "[REDACTED_TOKEN]" in cleaned["logs"][0]
    assert cleaned["logs"][1] == 42
    assert cleaned["tuple_items"][0] == "[REDACTED_EMAIL]"

    # Check bare secret values scrubbed
    assert cleaned["credentials"]["api_key"] == "[REDACTED_CREDENTIAL]"
    assert cleaned["credentials"]["client_secret"] == "[REDACTED_CREDENTIAL]"
    assert cleaned["credentials"]["password"] == "[REDACTED_CREDENTIAL]"

    # Check benign data is preserved
    assert cleaned["benign_info"]["vendor_id"] == "VND-101"
    assert cleaned["benign_info"]["item_name"] == "Enterprise Server Rack"
    assert cleaned["benign_info"]["unit_cost"] == 450.50


def test_trajectory_record_projections() -> None:
    """Verify ShareGPT and AgentTune EventLog projections."""
    traj = TrajectoryRecord(
        task_id="task_procurement_01",
        seed=42,
        system_prompt="You are a procurement agent.",
        user_query="Compute total spend for vendor VND-101.",
        steps=[
            StepRecord(
                step_index=0,
                thought="Looking up vendor orders",
                action=ToolCallAction(
                    tool_name="lookup_records",
                    arguments={"table": "purchase_orders", "filters": {"vendor_id": "VND-101"}},
                ),
                observation=[{"po_id": "PO-9001", "unit_price": 50.0, "quantity": 2}],
                elapsed_ms=12.5,
            ),
            StepRecord(
                step_index=1,
                thought="Aggregating total",
                action=ToolCallAction(
                    tool_name="compute_aggregate",
                    arguments={"values": [100.0], "operation": "sum"},
                ),
                observation=100.0,
                elapsed_ms=5.2,
            ),
        ],
        final_output="Total spend is $100.00.",
        is_success=True,
        total_steps=2,
        raw_reward=1.0,
    )

    # 1. ShareGPT format
    sgpt = traj.to_sharegpt_format()
    # 1 system + 1 human + 2*(thought + action + obs) + 1 final = 9
    assert len(sgpt) == 9
    assert sgpt[0]["from"] == "system"
    assert sgpt[1]["from"] == "human"
    assert sgpt[2]["from"] == "thought"
    assert sgpt[3]["from"] == "gpt"
    assert sgpt[4]["from"] == "observation"
    assert sgpt[5]["from"] == "thought"
    assert sgpt[6]["from"] == "gpt"
    assert sgpt[7]["from"] == "observation"
    assert sgpt[8]["from"] == "gpt"

    # 2. AgentTune EventLog format
    eventlog = traj.to_agenttune_eventlog()
    assert eventlog.task_id == "task_procurement_01"
    assert len(eventlog.events) == 6  # query + (action + obs)*2 + response
    assert eventlog.events[0].event_type == "user_query"
    assert eventlog.events[1].event_type == "agent_action"
    assert eventlog.events[2].event_type == "tool_observation"

    # 3. JSON round-trip
    serialized = traj.model_dump_json()
    reloaded = TrajectoryRecord.model_validate_json(serialized)
    assert reloaded.task_id == traj.task_id
    assert reloaded.fingerprint == traj.fingerprint
    assert len(reloaded.steps) == 2


def test_trajectory_provenance_metadata() -> None:
    """Verify provenance metadata fields on TrajectoryRecord."""
    traj = TrajectoryRecord(
        task_id="task_test_01",
        seed=123,
        system_prompt="Test prompt",
        user_query="Test query",
        run_id="run_exp_001",
        environment_version="0.1.0",
        metadata={"max_steps": 8, "hardware": "Apple M5"},
    )
    assert traj.run_id == "run_exp_001"
    assert traj.environment_version == "0.1.0"
    assert traj.metadata["hardware"] == "Apple M5"
    assert traj.fingerprint is not None

    redacted = traj.to_redacted_dict()
    assert redacted["run_id"] == "run_exp_001"
    assert redacted["environment_version"] == "0.1.0"


def test_locked_test_export_error_guard() -> None:
    """Verify that export_training_trajectories fails closed on locked test split."""
    import pytest
    from src.data.schemas import LockedTestExportError, export_training_trajectories

    test_traj = TrajectoryRecord(
        task_id="task_test_T3_vendor_rating_check_001",
        seed=42,
        system_prompt="Test prompt",
        user_query="Query vendor VND-108",
        steps=[],
        is_success=True,
    )
    with pytest.raises(LockedTestExportError, match="Locked test split"):
        export_training_trajectories([test_traj])


def test_filter_training_trajectories_policy() -> None:
    """Verify inclusion and exclusion filtering rules for M2 training data."""
    from src.data.schemas import (
        StepRecord,
        ToolCallAction,
        TrajectoryRecord,
        filter_training_trajectories,
    )

    # Valid train trajectory
    train_success = TrajectoryRecord(
        task_id="task_train_T1_inventory_value_000",
        seed=42,
        system_prompt="System",
        user_query="Query",
        steps=[
            StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "inventory_items", "filters": {}}),
                observation=[{"item": "SKU-1", "stock_level": 10, "unit_cost": 5}],
            )
        ],
        final_output="50",
        is_success=True,
    )

    # Failed train trajectory
    train_fail = TrajectoryRecord(
        task_id="task_train_T1_inventory_value_001",
        seed=42,
        system_prompt="System",
        user_query="Query",
        steps=[],
        is_success=False,
    )

    # Validation trajectory (should be quarantined)
    val_traj = TrajectoryRecord(
        task_id="task_val_T1_inventory_value_000",
        seed=42,
        system_prompt="System",
        user_query="Query",
        steps=[],
        is_success=True,
    )

    included, quarantined = filter_training_trajectories([train_success, train_fail, val_traj])
    assert len(included) == 1
    assert included[0].task_id == "task_train_T1_inventory_value_000"
    assert len(quarantined) == 2
    assert any("Disallowed split: 'val'" in q["reason"] for q in quarantined)
    assert any("is_success=False" in q["reason"] for q in quarantined)

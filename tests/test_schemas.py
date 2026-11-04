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
    """Verify recursive data structure redaction."""
    payload = {
        "user": {"email": "test@domain.org", "token": "api_key: abc12345678"},
        "logs": ["Bearer secret-token-xyz12345", 42, 3.14],
    }
    cleaned = redact_data(payload)
    assert cleaned["user"]["email"] == "[REDACTED_EMAIL]"
    assert "[REDACTED_CREDENTIAL]" in cleaned["user"]["token"]
    assert "[REDACTED_TOKEN]" in cleaned["logs"][0]
    assert cleaned["logs"][1] == 42


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

"""Unit tests for embedded JSON tool call extraction and Aya conversational prefix handling (Milestone 6).

Guarantees:
- Extracts valid JSON tool calls even when wrapped in conversational prefixes (<|START_RESPONSE|>).
- Safely handles historical Episode 0 model outputs from M5 validation traces.
- Preserves first valid tool call in multi-action model responses.
- Rejects non-tool dictionaries and prevents false positive tool activations.
- Cleans Aya conversational tokens (<|START_RESPONSE|>, <|END_OF_TURN_TOKEN|>) from terminal answers.
"""

from __future__ import annotations

import json
from src.runtime.parser import parse_agent_output


def test_parser_embedded_json_episode_0_exact() -> None:
    """Verify that the exact un-delimited output from M5 Episode 0 extracts lookup_records."""
    raw_episode_0 = (
        '<|START_RESPONSE|>{"name": "lookup_records", "arguments": '
        '{"table": "inventory_items", "filters": {"category": "Software"}}}\n\n'
        '{"name": "compute_aggregate", "arguments": {"values": [unit_cost * stock_level], "operation": "sum"}}}\n\n'
        "The total inventory valuation for all items in the 'Software' category is **$[total_value]**."
    )
    thought, action, answer, error = parse_agent_output(raw_episode_0)

    assert error is None
    assert answer is None
    assert action is not None
    assert action.tool_name == "lookup_records"
    assert action.arguments == {
        "table": "inventory_items",
        "filters": {"category": "Software"},
    }


def test_parser_embedded_json_vendor_spend() -> None:
    """Verify extraction of lookup_records from vendor spend validation outputs."""
    raw_vendor = (
        '<|START_RESPONSE|>{"name": "lookup_records", "arguments": '
        '{"table": "purchase_orders", "filters": {"vendor_id": "VND-106"}}}\n\n'
        "Vendor VND-106 approved spend is $0.00."
    )
    thought, action, answer, error = parse_agent_output(raw_vendor)

    assert error is None
    assert answer is None
    assert action is not None
    assert action.tool_name == "lookup_records"
    assert action.arguments == {
        "table": "purchase_orders",
        "filters": {"vendor_id": "VND-106"},
    }


def test_parser_embedded_json_conversational_wrapping() -> None:
    """Verify tool call extraction when model precedes action with natural language thought."""
    text = (
        "I need to inspect the database first.\n"
        '{"name": "lookup_records", "arguments": {"table": "vendors"}}\n'
        "Then I will calculate the metrics."
    )
    thought, action, answer, error = parse_agent_output(text)

    assert error is None
    assert answer is None
    assert action is not None
    assert action.tool_name == "lookup_records"
    assert action.arguments == {"table": "vendors"}


def test_parser_embedded_json_ignores_non_tool_dicts() -> None:
    """Verify that conversational JSON (e.g., config or user metadata) does not trigger false tool calls."""
    text = (
        'Here is the user profile: {"username": "auditor_1", "role": "admin"}. '
        "The audit indicates total spend is $5,000.00."
    )
    thought, action, answer, error = parse_agent_output(text)

    assert error is None
    assert action is None
    assert answer is not None
    assert "total spend is $5,000.00" in answer


def test_parser_embedded_json_unknown_tool_filtered() -> None:
    """Verify that candidate JSON with an unknown tool name is ignored during embedded scanning."""
    text = (
        'Let us execute: {"name": "unauthorized_shell_command", "arguments": {"cmd": "ls -la"}}. '
        "The task cannot be completed."
    )
    thought, action, answer, error = parse_agent_output(text)

    assert action is None
    assert answer is not None
    assert "The task cannot be completed" in answer


def test_parser_embedded_json_malformed_arguments() -> None:
    """Verify that an embedded invocation of a known tool with invalid arguments produces typed error."""
    text = (
        '<|START_RESPONSE|>{"name": "lookup_records", "arguments": "invalid_string_not_dict"}\n'
        "Proceeding with lookup."
    )
    thought, action, answer, error = parse_agent_output(text)

    assert action is None
    assert error is not None
    assert "MALFORMED_TOOL_ARGUMENTS" in error


def test_parser_embedded_json_custom_allowed_tools() -> None:
    """Verify custom allowed_tools parameter correctly filters candidate tool calls."""
    text = 'Calling aggregate calculation: {"name": "compute_aggregate", "arguments": {"values": [1.0, 2.0], "operation": "sum"}}'

    # Allowed includes compute_aggregate -> passes
    _, action1, _, err1 = parse_agent_output(text, allowed_tools={"compute_aggregate"})
    assert action1 is not None
    assert action1.tool_name == "compute_aggregate"

    # Allowed does NOT include compute_aggregate -> ignored in embedded mode
    _, action2, ans2, err2 = parse_agent_output(text, allowed_tools={"lookup_records"})
    assert action2 is None
    assert ans2 is not None


def test_parser_final_answer_cleans_aya_tokens() -> None:
    """Verify that Aya conversational tokens are stripped cleanly from terminal answer strings."""
    text = (
        "<|START_RESPONSE|>The total approved expenditure across all purchase orders "
        "is **$45,210.50**.<|END_OF_TURN_TOKEN|>"
    )
    thought, action, answer, error = parse_agent_output(text)

    assert error is None
    assert action is None
    assert answer == "The total approved expenditure across all purchase orders is **$45,210.50**."
    assert "<|START_RESPONSE|>" not in answer
    assert "<|END_OF_TURN_TOKEN|>" not in answer

"""Unit tests for model output parser and structured tool-call extraction."""

from src.runtime.parser import parse_agent_output


def test_parser_nominal_action() -> None:
    """Verify nominal tool action extraction."""
    text = (
        "<|START_THINKING|>Need to lookup items<|END_THINKING|>"
        '<|START_ACTION|>{"name": "lookup_records", "arguments": {"table": "vendors"}}<|END_ACTION|>'
    )
    thought, action, answer, error = parse_agent_output(text)
    assert thought == "Need to lookup items"
    assert action is not None
    assert action.tool_name == "lookup_records"
    assert action.arguments == {"table": "vendors"}
    assert answer is None
    assert error is None


def test_parser_truncated_action_detection() -> None:
    """Verify unclosed <|START_ACTION|> tag is identified as a truncated action token."""
    text = '<|START_ACTION|>{"name": "lookup_records", "arguments": {"table": "vendors"'
    thought, action, answer, error = parse_agent_output(text)
    assert action is None
    assert answer is None
    assert error is not None
    assert "TRUNCATED_ACTION_TOKEN" in error


def test_parser_unbalanced_end_action_tag() -> None:
    """Verify orphan <|END_ACTION|> tag triggers typed error."""
    text = '{"name": "lookup_records"}<|END_ACTION|>'
    thought, action, answer, error = parse_agent_output(text)
    assert action is None
    assert answer is None
    assert error is not None
    assert "MALFORMED_ACTION_TOKEN" in error


def test_parser_non_dict_payload() -> None:
    """Verify non-dictionary payload inside action tags is caught gracefully."""
    text = "<|START_ACTION|>[1, 2, 3]<|END_ACTION|>"
    thought, action, answer, error = parse_agent_output(text)
    assert action is None
    assert error == "MALFORMED_TOOL_ACTION_NOT_A_DICT"


def test_parser_non_dict_arguments() -> None:
    """Verify non-dict arguments (e.g., string or array) produce typed errors."""
    text = '<|START_ACTION|>{"name": "lookup_records", "arguments": "invalid_string_args"}<|END_ACTION|>'
    thought, action, answer, error = parse_agent_output(text)
    assert action is None
    assert error is not None
    assert "MALFORMED_TOOL_ARGUMENTS" in error


def test_parser_pure_json_fallback() -> None:
    """Verify standalone JSON without special tokens is supported."""
    text = '{"name": "compute_aggregate", "arguments": {"values": [1.0, 2.0], "operation": "sum"}}'
    thought, action, answer, error = parse_agent_output(text)
    assert action is not None
    assert action.tool_name == "compute_aggregate"
    assert action.arguments["operation"] == "sum"
    assert error is None


def test_parser_final_answer() -> None:
    """Verify pure text output is captured as final answer."""
    text = "Based on the records, total expenditure is $450.00."
    thought, action, answer, error = parse_agent_output(text)
    assert thought is None
    assert action is None
    assert answer == text
    assert error is None


def test_parser_empty_text() -> None:
    """Verify empty or whitespace strings return explicit error."""
    thought, action, answer, error = parse_agent_output("   \n\t  ")
    assert error == "EMPTY_MODEL_OUTPUT"


def test_parser_openai_tool_calls_format() -> None:
    """Verify OpenAI tool_calls wrapper payload parsing."""
    text = '{"tool_calls": [{"function": {"name": "lookup_records", "arguments": {"table": "vendors"}}}]}'
    thought, action, answer, error = parse_agent_output(text)
    assert error is None
    assert action is not None
    assert action.tool_name == "lookup_records"
    assert action.arguments == {"table": "vendors"}


def test_parser_markdown_json_format() -> None:
    """Verify Markdown code fenced JSON action parsing."""
    text = '```json\n{"name": "compute_aggregate", "arguments": {"values": [10.0, 20.0], "operation": "sum"}}\n```'
    thought, action, answer, error = parse_agent_output(text)
    assert error is None
    assert action is not None
    assert action.tool_name == "compute_aggregate"
    assert action.arguments["operation"] == "sum"



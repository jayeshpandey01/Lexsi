"""Unit tests for special token delimiter atomicity and parser alignment (M5.P1 / Probe P4).

Guarantees:
- Verifies that <|START_ACTION|> and <|END_ACTION|> are treated as atomic, single-token representations.
- Prevents subword BPE fragmentation bugs where delimiters are split into punctuation and text tokens.
- Ensures output delimiters align with runtime action parser extraction rules.
"""

from __future__ import annotations

import pytest

from src.runtime.parser import parse_agent_output
from src.training.collator import SPECIAL_TOKENS, DeterministicCommandRTokenizer


def test_special_tokens_atomicity() -> None:
    """Verify that all Cohere Command-R / Tiny-Aya control tokens map to unique, atomic integer IDs."""
    tokenizer = DeterministicCommandRTokenizer()

    # Required delimiters for tool-calling agent
    required_tokens = [
        "<|START_OF_TURN_TOKEN|>",
        "<|END_OF_TURN_TOKEN|>",
        "<|USER_TOKEN|>",
        "<|CHATBOT_TOKEN|>",
        "<|SYSTEM_TOKEN|>",
        "<|START_ACTION|>",
        "<|END_ACTION|>",
    ]

    for tok_str in required_tokens:
        assert tok_str in tokenizer.special_tokens, f"Missing required token {tok_str}"
        encoded_ids = tokenizer.encode(tok_str)
        assert len(encoded_ids) == 1, (
            f"Token '{tok_str}' was fragmented into {len(encoded_ids)} subwords: {encoded_ids}. "
            "Must be encoded as an atomic single token."
        )
        assert encoded_ids[0] == tokenizer.special_tokens[tok_str]
        decoded_str = tokenizer.decode(encoded_ids)
        assert decoded_str == tok_str, f"Decoded token '{decoded_str}' does not match original '{tok_str}'"


def test_action_payload_tokenization_integrity() -> None:
    """Verify that an assistant action payload maintains atomic delimiter boundaries."""
    tokenizer = DeterministicCommandRTokenizer()
    action_text = '<|START_ACTION|>{"name": "lookup_records", "arguments": {"vendor": "VND-100"}}<|END_ACTION|>'

    ids = tokenizer.encode(action_text)

    start_action_id = tokenizer.special_tokens["<|START_ACTION|>"]
    end_action_id = tokenizer.special_tokens["<|END_ACTION|>"]

    assert start_action_id in ids
    assert end_action_id in ids

    # START_ACTION must be first, END_ACTION must be last
    assert ids[0] == start_action_id
    assert ids[-1] == end_action_id

    # Delimiters must appear exactly once
    assert ids.count(start_action_id) == 1
    assert ids.count(end_action_id) == 1

    # Roundtrip decoding preserves exact delimiter strings
    decoded = tokenizer.decode(ids)
    assert "<|START_ACTION|>" in decoded
    assert "<|END_ACTION|>" in decoded
    assert "lookup_records" in decoded


def test_parser_alignment_with_atomic_delimiters() -> None:
    """Verify that runtime parser correctly extracts actions generated with atomic delimiters."""
    action_text = '<|START_ACTION|>{"name": "lookup_records", "arguments": {"table": "inventory_items"}}<|END_ACTION|>'

    thought, action, answer, error = parse_agent_output(action_text)

    assert error is None
    assert action is not None
    assert action.tool_name == "lookup_records"
    assert action.arguments == {"table": "inventory_items"}
    assert answer is None


def test_parser_rejects_fragmented_or_corrupted_delimiters() -> None:
    """Verify that fragmented delimiters (e.g. with spaces or missing brackets) are not accepted as valid tool calls."""
    fragmented_outputs = [
        '< | START_ACTION | >{"name": "lookup_records"}<|END_ACTION|>',
        '<|START_ACTION>{"name": "lookup_records"}<|END_ACTION|>',
        '<|START_ACTION|>{"name": "lookup_records"}</END_ACTION>',
        'START_ACTION {"name": "lookup_records"} END_ACTION',
    ]

    for output in fragmented_outputs:
        thought, action, answer, error = parse_agent_output(output)
        # Fragmented outputs must NOT be recognized as tool calls
        assert action is None, f"Corrupted delimiter '{output}' was erroneously parsed as valid action!"

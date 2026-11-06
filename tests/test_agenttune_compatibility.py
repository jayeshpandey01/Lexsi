"""Unit tests for M2.P3 AgentTune SFT compatibility adapter and preflight engine."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.data.schemas import LockedTestExportError
from src.training.agenttune_adapter import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    AgentTuneCompatibilityAdapter,
    AgentTunePreflightChecker,
    AgentTunePreflightResult,
    LossMaskingSimulation,
    ToolCallValidationResult,
)


def test_agenttune_compatibility_adapter_manifest() -> None:
    """Verify adapter inspects environment packages without throwing unhandled exceptions."""
    adapter = AgentTuneCompatibilityAdapter()
    manifest = adapter.get_environment_manifest()

    assert "torch" in manifest
    assert "tokenizers" in manifest
    assert "agenttune" in manifest
    assert "transformers" in manifest
    assert "trl" in manifest

    # Empirically verify detection behavior in current test environment
    assert adapter.is_native_agenttune_available() is False
    assert adapter.is_transformers_available() is False


def test_loss_masking_simulation_rules() -> None:
    """Verify that system, human, observation turns receive label=-100 and gpt receives token_id."""
    checker = AgentTunePreflightChecker()

    conversation = [
        {"from": "system", "value": "You are an auditing assistant."},
        {"from": "human", "value": "Check spend for VND-101."},
        {
            "from": "gpt",
            "value": f"{ACTION_START_TOKEN}" + json.dumps({
                "name": "lookup_records",
                "arguments": {"table": "purchase_orders", "filters": {"vendor_id": "VND-101"}},
            }) + f"{ACTION_END_TOKEN}",
        },
        {"from": "observation", "value": json.dumps([{"po_id": "PO-1", "amount": 100.0}])},
        {"from": "gpt", "value": "The total spend is $100.00."},
    ]

    res = checker.simulate_loss_masking_sharegpt(conversation)
    assert res.total_turns == 5
    assert res.is_masking_compliant is True
    assert res.supervised_tokens_count > 0
    assert res.masked_tokens_count > 0
    assert 0.0 < res.supervision_ratio < 1.0

    # Inspect per-turn labels
    labels = [t["label_assigned"] for t in res.turn_breakdown]
    assert labels == [-100, -100, "token_id", -100, "token_id"]


def test_loss_masking_fails_on_empty_assistant() -> None:
    """Verify that conversation with 0 assistant turns fails compliance (prevents NaN loss)."""
    checker = AgentTunePreflightChecker()
    conversation = [
        {"from": "system", "value": "System prompt"},
        {"from": "human", "value": "User prompt"},
    ]
    res = checker.simulate_loss_masking_sharegpt(conversation)
    assert res.is_masking_compliant is False
    assert res.supervised_tokens_count == 0


def test_tool_call_syntax_and_schema_validation() -> None:
    """Verify tool call validation correctly distinguishes valid, malformed, and unregistered tools."""
    checker = AgentTunePreflightChecker()

    valid_turn = {
        "from": "gpt",
        "value": f"{ACTION_START_TOKEN}" + json.dumps({
            "name": "lookup_records",
            "arguments": {"table": "inventory_items", "filters": {}},
        }) + f"{ACTION_END_TOKEN}",
    }
    t_res = checker.validate_tool_calls_in_conversation([valid_turn])
    assert t_res.total_tool_calls_detected == 1
    assert t_res.valid_tool_calls == 1
    assert t_res.malformed_tool_calls == 0

    # Malformed JSON
    malformed_turn = {
        "from": "gpt",
        "value": f"{ACTION_START_TOKEN}{{name: unquoted, arguments: bad{ACTION_END_TOKEN}",
    }
    m_res = checker.validate_tool_calls_in_conversation([malformed_turn])
    assert m_res.total_tool_calls_detected == 1
    assert m_res.valid_tool_calls == 0
    assert m_res.malformed_tool_calls == 1

    # Unregistered tool
    unregistered_turn = {
        "from": "gpt",
        "value": f"{ACTION_START_TOKEN}" + json.dumps({
            "name": "execute_arbitrary_shell",
            "arguments": {"cmd": "rm -rf /"},
        }) + f"{ACTION_END_TOKEN}",
    }
    u_res = checker.validate_tool_calls_in_conversation([unregistered_turn])
    assert u_res.total_tool_calls_detected == 1
    assert u_res.valid_tool_calls == 0
    assert "execute_arbitrary_shell" in u_res.unrecognized_tools


def test_sharegpt_train_dataset_preflight() -> None:
    """Run preflight check against actual persisted sft_train_sharegpt.json dataset."""
    checker = AgentTunePreflightChecker()
    sharegpt_path = Path("data/sft/sft_train_sharegpt.json")
    assert sharegpt_path.exists()

    result = checker.validate_sharegpt_dataset(sharegpt_path)
    assert result.is_compatible is True
    assert result.sample_count == 20
    assert result.tool_call_stats.total_tool_calls_detected == 40
    assert result.tool_call_stats.valid_tool_calls == 40
    assert result.tool_call_stats.malformed_tool_calls == 0
    assert result.masking_stats.is_masking_compliant is True
    assert 0.05 < result.masking_stats.supervision_ratio < 0.50
    assert len(result.violations) == 0


def test_eventlog_train_dataset_preflight() -> None:
    """Run preflight check against actual persisted sft_train_eventlog.json dataset."""
    checker = AgentTunePreflightChecker()
    eventlog_path = Path("data/sft/sft_train_eventlog.json")
    assert eventlog_path.exists()

    result = checker.validate_eventlog_dataset(eventlog_path)
    assert result.is_compatible is True
    assert result.sample_count == 20
    assert result.tool_call_stats.valid_tool_calls == 40
    assert result.tool_call_stats.malformed_tool_calls == 0
    assert len(result.violations) == 0


def test_locked_test_fail_closed_guard(tmp_path: Path) -> None:
    """Verify that attempting to preflight a dataset with sequestered locked-test tasks raises LockedTestExportError."""
    checker = AgentTunePreflightChecker()

    # Contaminated ShareGPT dataset
    contaminated_sg = tmp_path / "contaminated_sharegpt.json"
    contaminated_data = [
        [
            {"from": "system", "value": "System"},
            {"from": "human", "value": "Evaluate task_test_T3_vendor_rating_check_001"},
            {"from": "gpt", "value": "Output"},
        ]
    ]
    contaminated_sg.write_text(json.dumps(contaminated_data), encoding="utf-8")

    with pytest.raises(LockedTestExportError, match="locked-test"):
        checker.validate_sharegpt_dataset(contaminated_sg)

    # Contaminated EventLog dataset
    contaminated_el = tmp_path / "contaminated_eventlog.json"
    contaminated_el_data = [
        {
            "task_id": "task_test_T4_budget_cap_verification_001",
            "seed": 42,
            "events": [],
        }
    ]
    contaminated_el.write_text(json.dumps(contaminated_el_data), encoding="utf-8")

    with pytest.raises(LockedTestExportError, match="locked test task"):
        checker.validate_eventlog_dataset(contaminated_el)


def test_markdown_report_generation(tmp_path: Path) -> None:
    """Verify Markdown report generation contains required audit sections and verdict."""
    checker = AgentTunePreflightChecker()
    sharegpt_path = Path("data/sft/sft_train_sharegpt.json")
    eventlog_path = Path("data/sft/sft_train_eventlog.json")
    out_file = tmp_path / "test_report.md"

    sg_res = checker.validate_sharegpt_dataset(sharegpt_path)
    el_res = checker.validate_eventlog_dataset(eventlog_path)

    report = checker.generate_m2_p3_markdown_report(sg_res, el_res, output_path=out_file)
    assert out_file.exists()
    assert "PASSED (COMPATIBLE & READY FOR PILOT)" in report
    assert "CohereLabs/tiny-aya-global" in report
    assert "Loss Masking & Token Allocation Audit" in report
    assert "Registered Tool Distribution in Training Corpus" in report
    assert "**GATE M2.P3 STATUS:** `CLEARED`" in report

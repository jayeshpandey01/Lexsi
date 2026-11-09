"""Unit tests for standalone embedded multi-turn interactive evaluator.

Verifies:
1. Deterministic tool execution on relational mock database.
2. Robust parsing of action tokens.
3. Accurate environment-owned outcome verification (numerical & boolean verdicts).
4. ReAct multi-turn interactive rollout with state tracking and observation feeding.
5. Full validation set evaluation asserting M1=100%, M2=100% on oracle agent and M1=0% on zero-tool baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.runtime.embedded_evaluator import (
    InteractiveRolloutEvaluator,
    StandaloneToolEngine,
    parse_tool_action,
    verify_task_outcome_standalone,
)


def test_standalone_tool_engine_execution():
    engine = StandaloneToolEngine(seed=42)

    # 1. lookup_records
    records, is_err, err = engine.execute(
        "lookup_records", {"table": "inventory_items", "filters": {"category": "Software"}}
    )
    assert not is_err
    assert err is None
    assert len(records) > 0
    assert all(r["category"] == "Software" for r in records)

    # 2. compute_aggregate
    vals = [10.5, 20.5, 30.0]
    res, is_err, err = engine.execute("compute_aggregate", {"values": vals, "operation": "sum"})
    assert not is_err
    assert res == 61.0

    res_mean, is_err, _ = engine.execute("compute_aggregate", {"values": vals, "operation": "mean"})
    assert not is_err
    assert abs(res_mean - 20.3333) < 0.001

    # 3. verify_constraint
    res_vc, is_err, _ = engine.execute(
        "verify_constraint", {"actual": 45000.0, "expected": 50000.0, "comparison": "le"}
    )
    assert not is_err
    assert res_vc["is_compliant"] is True

    res_vc_fail, is_err, _ = engine.execute(
        "verify_constraint", {"actual": 55000.0, "expected": 50000.0, "comparison": "le"}
    )
    assert not is_err
    assert res_vc_fail["is_compliant"] is False


def test_parse_tool_action():
    # 1. Native Cohere delimiters
    valid = '<|START_ACTION|>{"name": "lookup_records", "arguments": {"table": "vendors"}}<|END_ACTION|>'
    name, args, err = parse_tool_action(valid)
    assert name == "lookup_records"
    assert args == {"table": "vendors"}
    assert err is None

    # 2. Markdown fenced JSON block
    md_valid = '```json\n{"name": "lookup_records", "arguments": {"table": "inventory_items"}}\n```'
    name_md, args_md, err_md = parse_tool_action(md_valid)
    assert name_md == "lookup_records"
    assert args_md == {"table": "inventory_items"}
    assert err_md is None

    # 3. XML tags
    xml_valid = '<tool_call>{"name": "compute_aggregate", "arguments": {"values": [1, 2], "operation": "sum"}}</tool_call>'
    name_xml, args_xml, err_xml = parse_tool_action(xml_valid)
    assert name_xml == "compute_aggregate"
    assert args_xml == {"values": [1, 2], "operation": "sum"}
    assert err_xml is None

    # 4. Raw JSON
    raw_valid = '{"name": "lookup_records", "arguments": {"table": "budget_caps"}}'
    name_raw, args_raw, err_raw = parse_tool_action(raw_valid)
    assert name_raw == "lookup_records"
    assert args_raw == {"table": "budget_caps"}
    assert err_raw is None

    # 5. Non-action conversational response
    no_action = "The total valuation is $45,000.00."
    name, args, err = parse_tool_action(no_action)
    assert name is None
    assert args is None
    assert err is None

    # 6. Malformed JSON within action tag
    bad_json = "<|START_ACTION|>{malformed: json<|END_ACTION|>"
    name, args, err = parse_tool_action(bad_json)
    assert name is None
    assert err is not None

    # 7. Truncated action tag
    trunc = "<|START_ACTION|>{\"name\": \"lookup_records\""
    name, args, err = parse_tool_action(trunc)
    assert name is None
    assert "TRUNCATED" in err


def test_verify_task_outcome_standalone():
    task = {"task_id": "test_1", "oracle_answer": 1234.50, "oracle_verdict": None}

    # Success cases
    assert verify_task_outcome_standalone(task, "The total calculated value is $1,234.50.")
    assert verify_task_outcome_standalone(task, "Result: 1234.50")
    assert verify_task_outcome_standalone(task, "Calculated: 1234.52")  # within 0.05 absolute tolerance

    # Failure cases
    assert not verify_task_outcome_standalone(task, "The value is 9999.00.")
    assert not verify_task_outcome_standalone(task, "")
    assert not verify_task_outcome_standalone(task, "No number generated.")

    # Verdict checking
    verdict_task = {"task_id": "test_2", "oracle_answer": 500.0, "oracle_verdict": True}
    assert verify_task_outcome_standalone(
        verdict_task, "Spend is $500.00, which is within budget."
    )
    assert not verify_task_outcome_standalone(
        verdict_task, "Spend is $500.00, which exceeds budget."
    )


def test_multiturn_rollout_mock_agent():
    evaluator = InteractiveRolloutEvaluator()
    task = {
        "task_id": "task_val_001",
        "user_query": "Calculate the total inventory valuation for all items in the 'Software' category.",
        "oracle_answer": 102625.07,
        "oracle_verdict": None,
    }

    # Policy that performs 2 tool turns then emits final answer
    def mock_react_policy(prompt: str) -> str:
        if "Observation:" not in prompt:
            # First turn: call lookup_records
            return '<|START_ACTION|>{"name": "lookup_records", "arguments": {"table": "inventory_items", "filters": {"category": "Software"}}}<|END_ACTION|>'
        elif prompt.count("Observation:") == 1:
            # Second turn: parse items from observation and call compute_aggregate
            # For simplicity in mock policy, pass the values
            return '<|START_ACTION|>{"name": "compute_aggregate", "arguments": {"values": [102625.07], "operation": "sum"}}<|END_ACTION|>'
        else:
            # Final turn: emit answer
            return "Based on database records, the total inventory valuation is $102,625.07."

    res = evaluator.run_episode_policy(mock_react_policy, task)
    assert res["has_tool_action"] is True
    assert res["is_success"] is True
    assert res["num_turns"] == 3
    assert len(res["turns"]) == 3
    assert res["turns"][0]["type"] == "tool_call"
    assert res["turns"][1]["type"] == "tool_call"
    assert res["turns"][2]["type"] == "final_answer"


def test_val_split_rollout_with_oracle_policy():
    val_path = Path("configs/manifests/task_manifest_val.json")
    assert val_path.exists(), "task_manifest_val.json must exist"

    with open(val_path, encoding="utf-8") as f:
        val_tasks = json.load(f)["tasks"]

    assert len(val_tasks) == 10

    evaluator = InteractiveRolloutEvaluator()

    # 1. Oracle tool-calling policy
    def oracle_policy(prompt: str) -> str:
        for t in val_tasks:
            if t["user_query"] in prompt:
                if "Observation:" not in prompt:
                    return '<|START_ACTION|>{"name": "lookup_records", "arguments": {"table": "inventory_items", "filters": {}}}<|END_ACTION|>'
                else:
                    ans = t["oracle_answer"]
                    verdict_str = ""
                    if t.get("oracle_verdict") is not None:
                        verdict_str = ", which is within budget" if t["oracle_verdict"] else ", which exceeds budget"
                    return f"The total calculated value is ${ans:,.2f}{verdict_str}."
        return "I do not know."

    results, m1, m2 = evaluator.evaluate_validation_tasks(val_tasks, oracle_policy, run_name="test_oracle")
    assert len(results) == 10
    assert m1 == 1.0, f"Expected 100% M1 for oracle policy, got {m1}"
    assert m2 == 1.0, f"Expected 100% M2 for oracle policy, got {m2}"

    # 2. Dumb baseline policy (always says 'The inventory is unknown.')
    def dumb_policy(prompt: str) -> str:
        return "The inventory is unknown."

    results_dumb, m1_dumb, m2_dumb = evaluator.evaluate_validation_tasks(
        val_tasks, dumb_policy, run_name="test_dumb"
    )
    assert m1_dumb == 0.0
    assert m2_dumb == 0.0

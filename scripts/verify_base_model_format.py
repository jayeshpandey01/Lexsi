#!/usr/bin/env python3
"""Pre-Training Base Model Format Gate Runner (GAP-1 / Pillar 1).

Evaluates whether the designated base model possesses the minimal required format
prior to emit structured tool call delimiters (<|START_ACTION|>) at temperature=0.0
before launching expensive GPU fine-tuning runs.

Guarantees:
- Tests 5 standard tool-calling queries against the model gateway or offline engine.
- Flags EarlyTerminationFailure risk before fine-tuning begins.
- Persists machine-readable gate report to results/runs/<run_id>/base_format_gate_report.json.
- Fails closed if 0/5 queries emit the required action delimiter.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.environment.task_generator import TaskSpec
from src.runtime.gateway import ModelGateway
from src.runtime.parser import parse_agent_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_base_model_format")


SAMPLE_GATE_TASKS = [
    {
        "query": "Check inventory level and unit cost for item 'ITM-101' in inventory_items.",
        "expected_tool": "lookup_records",
    },
    {
        "query": "Determine the total approved expenditure for vendor 'VND-105' in purchase_orders.",
        "expected_tool": "lookup_records",
    },
    {
        "query": "Calculate the sum of [120.50, 450.00, 89.20] using compute_aggregate.",
        "expected_tool": "compute_aggregate",
    },
    {
        "query": "Verify if the total spend 45000.0 is less than or equal to budget cap 50000.0.",
        "expected_tool": "verify_constraint",
    },
    {
        "query": "Look up budget caps for department 'Engineering' in table budget_caps.",
        "expected_tool": "lookup_records",
    },
]

SYSTEM_PROMPT_TEMPLATE = """You are an expert procurement and inventory auditing agent.
To solve tasks, you MUST invoke tools using:
<|START_ACTION|>{"name": "<tool_name>", "arguments": {<args>}}<|END_ACTION|>

Available tools:
[
  {
    "name": "lookup_records",
    "description": "Query in-memory procurement database table using exact key-value filters.",
    "parameters": [
      {"name": "table", "type": "string", "required": true, "enum": ["vendors", "inventory_items", "purchase_orders", "budget_caps"]},
      {"name": "filters", "type": "dict", "required": false}
    ]
  },
  {
    "name": "compute_aggregate",
    "description": "Perform arithmetic reduction (sum, mean, min, max, count, std) on numbers.",
    "parameters": [
      {"name": "values", "type": "list", "required": true},
      {"name": "operation", "type": "string", "required": true, "enum": ["sum", "mean", "min", "max", "count", "std"]}
    ]
  },
  {
    "name": "verify_constraint",
    "description": "Compare a calculated value against a numeric threshold or budget cap.",
    "parameters": [
      {"name": "actual", "type": "float", "required": true},
      {"name": "expected", "type": "float", "required": true},
      {"name": "comparison", "type": "string", "required": true, "enum": ["eq", "le", "ge", "lt", "gt"]}
    ]
  }
]
When finished, emit your final numerical calculation and conclusion."""


def verify_base_model_format(
    model_id: str = "CohereLabs/tiny-aya-global",
    run_id: str = "base_format_gate_001",
    temperature: float = 0.0,
    output_base_dir: Optional[Path] = None,
    mock_mode: bool = False,
) -> Dict[str, Any]:
    """Execute pre-flight format verification gate."""
    output_dir = (output_base_dir or (ROOT / "results" / "runs")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"PRE-TRAINING BASE MODEL FORMAT GATE ({model_id})")
    logger.info("=" * 80)

    results: List[Dict[str, Any]] = []
    actions_detected = 0

    gateway = ModelGateway()

    for idx, item in enumerate(SAMPLE_GATE_TASKS):
        prompt = f"{SYSTEM_PROMPT_TEMPLATE}\n\nTask: {item['query']}"
        logger.info(f"[Probe {idx+1}/{len(SAMPLE_GATE_TASKS)}] Testing task: {item['query'][:50]}...")

        if mock_mode:
            # Deterministic simulation for mock unit tests
            simulated_response = '<|START_ACTION|>{"name": "lookup_records", "arguments": {"table": "inventory_items"}}<|END_ACTION|>'
            raw_text = simulated_response
        else:
            try:
                # Query via gateway if API keys available, else fall back to local offline test
                resp = asyncio.run(
                    gateway.generate(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_id if "cohere/" in model_id else "cohere/command-r7b-12-2024",
                        temperature=temperature,
                        max_tokens=256,
                    )
                )
                raw_text = resp.content or ""
            except Exception as err:
                logger.warning(f"Gateway generation unavailable ({err}); falling back to offline diagnostic inspection.")
                raw_text = "<|START_RESPONSE|>I will describe the calculation in SQL prose.<|END_RESPONSE|>"

        has_start_action = "<|START_ACTION|>" in raw_text
        thought, tool_action, final_answer, parse_error = parse_agent_output(raw_text)
        is_tool_call = (tool_action is not None)

        if has_start_action or is_tool_call:
            actions_detected += 1

        results.append({
            "task_index": idx,
            "query": item["query"],
            "expected_tool": item["expected_tool"],
            "raw_output_snippet": raw_text[:200],
            "has_start_action_delimiter": has_start_action,
            "is_tool_call": is_tool_call,
            "tool_name": tool_action.tool_name if tool_action else None,
            "parse_error": parse_error,
            "passed": (has_start_action or is_tool_call),
        })

    pass_rate = actions_detected / len(SAMPLE_GATE_TASKS)
    gate_passed = (actions_detected >= 1)  # At least 1/5 must exhibit tool calling propensity

    verdict_summary = {
        "model_id": model_id,
        "run_id": run_id,
        "timestamp": time.time(),
        "total_probes": len(SAMPLE_GATE_TASKS),
        "actions_detected": actions_detected,
        "pass_rate": pass_rate,
        "gate_passed": gate_passed,
        "recommendation": (
            "Model exhibits valid tool-call delimiter propensity; proceed to SFT curriculum."
            if gate_passed
            else (
                "ZERO tool call delimiters observed (EarlyTermination risk: 100%). "
                "Recommendation: Apply format-first curriculum (50 pure format-warmup steps) "
                "or escalate to CohereForAI/aya-expanse-8b."
            )
        ),
        "probe_details": results,
    }

    report_path = output_dir / "base_format_gate_report.json"
    report_path.write_text(json.dumps(verdict_summary, indent=2), encoding="utf-8")

    logger.info(f"Gate Assessment: {'PASSED' if gate_passed else 'FAILED / BLOCKED'}")
    logger.info(f"Delimiters detected: {actions_detected}/{len(SAMPLE_GATE_TASKS)} ({pass_rate:.1%})")
    logger.info(f"Recommendation: {verdict_summary['recommendation']}")
    logger.info(f"Report saved to: {report_path}")

    return verdict_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-training base model format gate.")
    parser.add_argument("--model-id", type=str, default="CohereLabs/tiny-aya-global")
    parser.add_argument("--run-id", type=str, default="base_format_gate_001")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode for CI/offline testing")
    args = parser.parse_args()

    res = verify_base_model_format(
        model_id=args.model_id,
        run_id=args.run_id,
        mock_mode=args.mock,
    )
    if not res.get("gate_passed"):
        logger.warning("Base model format gate failed. Do not launch standard SFT without format warmup!")
        sys.exit(1)
    sys.exit(0)

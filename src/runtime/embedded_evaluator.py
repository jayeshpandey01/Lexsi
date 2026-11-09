"""Standalone embedded multi-turn interactive evaluator for tool-calling agents.

Implements an in-memory, deterministic ReAct execution loop that handles:
1. Turn-by-turn interactive generation until terminal response or step budget exhaustion.
2. Robust parsing of <|START_ACTION|>...<|END_ACTION|> tokens.
3. Sandboxed execution of lookup_records, compute_aggregate, and verify_constraint.
4. Observation feedback formatted consistently with SFT conversation formatting:
   <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Observation: {obs}<|END_OF_TURN_TOKEN|>
5. Environment-owned outcome verification (M1: Task Success, M2: Tool Calling Rate).
"""

from __future__ import annotations

import json
import math
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def get_mock_database_data(seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Generate reproducible relational database fixtures from seed."""
    rng = random.Random(seed)
    categories = ["Hardware", "Software", "Services", "Logistics"]
    countries = ["US", "DE", "JP", "GB", "SG"]

    vendors: List[Dict[str, Any]] = [
        {
            "vendor_id": f"VND-{100 + i}",
            "name": f"Enterprise {categories[i % len(categories)]} Solutions {i}",
            "category": categories[i % len(categories)],
            "rating": round(3.5 + (rng.random() * 1.5), 2),
            "contract_status": "Active" if i % 4 != 0 else "UnderReview",
            "country": countries[i % len(countries)],
        }
        for i in range(12)
    ]

    inventory_items: List[Dict[str, Any]] = [
        {
            "item_sku": f"SKU-{8000 + i}",
            "item_name": f"Enterprise Component {i}",
            "category": categories[i % len(categories)],
            "stock_level": rng.randint(10, 500),
            "reorder_threshold": rng.randint(20, 100),
            "unit_cost": round(15.0 + (rng.random() * 150.0), 2),
        }
        for i in range(20)
    ]

    purchase_orders: List[Dict[str, Any]] = []
    statuses = ["Draft", "Approved", "Fulfilled", "Cancelled"]
    for i in range(30):
        vendor = vendors[i % len(vendors)]
        item = inventory_items[i % len(inventory_items)]
        qty = rng.randint(5, 100)
        price = round(item["unit_cost"] * (1.0 + (rng.random() * 0.1)), 2)
        purchase_orders.append(
            {
                "po_id": f"PO-{9000 + i}",
                "vendor_id": vendor["vendor_id"],
                "item_sku": item["item_sku"],
                "quantity": qty,
                "unit_price": price,
                "order_status": statuses[i % len(statuses)],
                "order_date": f"2026-0{(i % 6) + 1}-15",
            }
        )

    budget_caps: List[Dict[str, Any]] = [
        {
            "category": cat,
            "fiscal_quarter": "2026-Q1",
            "allocated_budget": 50000.0 + (idx * 25000.0),
        }
        for idx, cat in enumerate(categories)
    ]

    return {
        "vendors": vendors,
        "inventory_items": inventory_items,
        "purchase_orders": purchase_orders,
        "budget_caps": budget_caps,
    }


class StandaloneToolEngine:
    """Isolated, deterministic in-memory tool execution engine.

    Zero external dependencies, completely portable across Kaggle, Colab, and local testing.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.db = get_mock_database_data(seed=seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self.db = get_mock_database_data(seed=self.seed)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[Any, bool, Optional[str]]:
        """Execute tool and return (result, is_error, error_message)."""
        if not isinstance(arguments, dict):
            return None, True, f"Arguments must be a dict, got {type(arguments).__name__}"

        try:
            if tool_name == "lookup_records":
                table = arguments.get("table")
                if not table or table not in self.db:
                    return None, True, f"Table '{table}' does not exist in database."
                filters = arguments.get("filters", {})
                records = self.db[table]
                matched = []
                for row in records:
                    match = True
                    for k, v in filters.items():
                        if k not in row or row[k] != v:
                            match = False
                            break
                    if match:
                        matched.append(row)
                return matched, False, None

            elif tool_name == "compute_aggregate":
                values = arguments.get("values", [])
                operation = arguments.get("operation")
                if not isinstance(values, list):
                    return None, True, f"Expected list for values, got {type(values).__name__}"
                if not values and operation != "count":
                    return None, True, f"Cannot perform '{operation}' on empty values list."

                for v in values:
                    if not isinstance(v, (int, float)) or not math.isfinite(v):
                        return None, True, f"Invalid non-finite or non-numeric value: {v}"

                if operation == "sum":
                    return round(sum(values), 4), False, None
                elif operation == "mean":
                    return round(sum(values) / len(values), 4), False, None
                elif operation == "min":
                    return min(values), False, None
                elif operation == "max":
                    return max(values), False, None
                elif operation == "count":
                    return len(values), False, None
                elif operation == "std":
                    if len(values) < 2:
                        return 0.0, False, None
                    m = sum(values) / len(values)
                    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
                    return round(math.sqrt(var), 4), False, None
                else:
                    return None, True, f"Unsupported operation: '{operation}'"

            elif tool_name == "verify_constraint":
                actual = arguments.get("actual")
                expected = arguments.get("expected")
                comparison = arguments.get("comparison")
                tolerance = arguments.get("tolerance", 0.001)

                if actual is None or expected is None or comparison is None:
                    return None, True, "Missing required parameters for verify_constraint."

                diff = actual - expected
                verdict = False
                if comparison == "eq":
                    verdict = abs(diff) <= tolerance
                elif comparison == "le":
                    verdict = (actual <= expected) or (abs(diff) <= tolerance)
                elif comparison == "ge":
                    verdict = (actual >= expected) or (abs(diff) <= tolerance)
                elif comparison == "lt":
                    verdict = actual < expected
                elif comparison == "gt":
                    verdict = actual > expected
                else:
                    return None, True, f"Unsupported comparison operator: '{comparison}'"

                return {
                    "is_compliant": verdict,
                    "actual": actual,
                    "expected": expected,
                    "comparison": comparison,
                }, False, None

            else:
                return None, True, f"Tool '{tool_name}' not found."

        except Exception as e:
            return None, True, f"Execution exception in '{tool_name}': {str(e)}"


ACTION_PATTERN = re.compile(r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.DOTALL)
XML_ACTION_PATTERN = re.compile(r"<(?:tool_call|action)>(.*?)</(?:tool_call|action)>", re.DOTALL)
MARKDOWN_JSON_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_dict_action(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Helper to extract tool_name and arguments from a parsed dictionary."""
    if "tool_calls" in payload and isinstance(payload["tool_calls"], list) and len(payload["tool_calls"]) > 0:
        first_call = payload["tool_calls"][0]
        if isinstance(first_call, dict) and "function" in first_call:
            fn = first_call["function"]
            tool_name = fn.get("name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception as e:
                    return None, None, f"JSON parse error in tool_calls arguments: {str(e)}"
            if tool_name and isinstance(args, dict):
                return tool_name, args, None

    tool_name = payload.get("name") or payload.get("tool_name")
    if not tool_name or not isinstance(tool_name, str):
        return None, None, "Missing tool name in action payload."

    arguments = payload.get("arguments")
    if arguments is None:
        arguments = payload.get("parameters", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception as e:
            return None, None, f"Could not parse string arguments as JSON: {str(e)}"

    if not isinstance(arguments, dict):
        return None, None, f"Arguments must be a dictionary, got {type(arguments).__name__}"

    return tool_name, arguments, None


def parse_tool_action(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Parse tool action from model generation string with dual delimiter support.

    Supports:
    1. Native Cohere special tokens: <|START_ACTION|>...<|END_ACTION|>
    2. Markdown fenced JSON: ```json {"name": "...", "arguments": {...}} ```
    3. XML tags: <tool_call>...</tool_call> or <action>...</action>
    4. Raw standalone JSON objects: {"name": "...", "arguments": {...}}

    Returns: (tool_name, arguments, error_message).
    """
    if not text or not text.strip():
        return None, None, None

    # Check for truncated or unbalanced action delimiter tags
    has_start_action = "<|START_ACTION|>" in text
    has_end_action = "<|END_ACTION|>" in text
    if has_start_action and not has_end_action:
        return None, None, "TRUNCATED_ACTION_TOKEN: Unclosed <|START_ACTION|> tag"
    if has_end_action and not has_start_action:
        return None, None, "MALFORMED_ACTION_TOKEN: Found <|END_ACTION|> without <|START_ACTION|>"

    # 1. Native Cohere delimiters
    match = ACTION_PATTERN.search(text)
    if match:
        payload_str = match.group(1).strip()
        try:
            payload = json.loads(payload_str)
        except Exception as e:
            return None, None, f"JSON parse error in action payload: {str(e)}"
        if not isinstance(payload, dict):
            return None, None, f"Action payload must be a JSON object, got {type(payload).__name__}"
        return _extract_dict_action(payload)

    # 2. XML tags (<tool_call> or <action>)
    xml_match = XML_ACTION_PATTERN.search(text)
    if xml_match:
        payload_str = xml_match.group(1).strip()
        try:
            payload = json.loads(payload_str)
            if isinstance(payload, dict):
                return _extract_dict_action(payload)
            return None, None, f"Action payload must be a JSON object, got {type(payload).__name__}"
        except Exception as e:
            return None, None, f"JSON parse error in action payload: {str(e)}"

    # 3. Markdown fenced JSON
    md_match = MARKDOWN_JSON_PATTERN.search(text)
    if md_match:
        payload_str = md_match.group(1).strip()
        try:
            payload = json.loads(payload_str)
            if isinstance(payload, dict) and ("name" in payload or "tool_name" in payload or "tool_calls" in payload):
                return _extract_dict_action(payload)
        except Exception as e:
            if '"name"' in payload_str or '"tool"' in payload_str:
                return None, None, f"JSON parse error in action payload: {str(e)}"

    # 4. Raw JSON fallback
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict) and ("name" in payload or "tool_name" in payload or "tool_calls" in payload):
                return _extract_dict_action(payload)
        except Exception as e:
            if '"name"' in stripped or '"tool"' in stripped:
                return None, None, f"JSON parse error in action payload: {str(e)}"

    return None, None, None


def verify_task_outcome_standalone(
    task: Union[Dict[str, Any], Any],
    final_answer: Optional[str],
) -> bool:
    """Environment-owned deterministic outcome verification."""
    if not final_answer or not final_answer.strip():
        return False

    oracle_answer = getattr(task, "oracle_answer", None) if not isinstance(task, dict) else task.get("oracle_answer")
    oracle_verdict = getattr(task, "oracle_verdict", None) if not isinstance(task, dict) else task.get("oracle_verdict")

    if oracle_answer is None:
        return False

    cleaned_text = final_answer.replace(",", "")
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", cleaned_text)
    if not nums:
        return False

    numeric_match = False
    for n_str in nums:
        try:
            val = float(n_str)
            abs_err = abs(val - oracle_answer)
            rel_err = abs_err / (abs(oracle_answer) + 1e-5)
            if abs_err <= 0.05 or rel_err <= 0.01:
                numeric_match = True
                break
        except ValueError:
            continue

    if not numeric_match:
        return False

    if oracle_verdict is not None:
        lower_text = cleaned_text.lower()
        is_violation = bool(
            re.search(
                r"\b(exceeds|over budget|non-compliant|breached|not compliant|exceeded|failed)\b",
                lower_text,
            )
        )
        is_compliant = bool(
            re.search(
                r"\b(within budget|compliant|complies|under budget|passed|meets cap|within cap)\b",
                lower_text,
            )
        ) and not bool(
            re.search(r"\b(non-compliant|not compliant|not within budget|never compliant)\b", lower_text)
        )

        if oracle_verdict is True:
            if is_violation or not is_compliant:
                if "true" not in lower_text and "yes" not in lower_text:
                    return False
        else:
            if is_compliant or not is_violation:
                if "false" not in lower_text and "no" not in lower_text:
                    return False

    return True


CANONICAL_SYSTEM_PROMPT = (
    "You are an expert procurement and inventory auditing agent. "
    "To solve tasks, you may invoke the following deterministic tools using "
    "<|START_ACTION|>{\"name\": \"<tool_name>\", \"arguments\": {<args>}}<|END_ACTION|>. Available tools:\n"
    "[\n"
    "  {\n"
    "    \"name\": \"lookup_records\",\n"
    "    \"description\": \"Query in-memory procurement database table using exact key-value filters.\",\n"
    "    \"parameters\": [\n"
    "      {\"name\": \"table\", \"type\": \"string\", \"description\": \"Target table name\", \"required\": true, \"enum\": [\"vendors\", \"inventory_items\", \"purchase_orders\", \"budget_caps\"]},\n"
    "      {\"name\": \"filters\", \"type\": \"dict\", \"description\": \"Key-value equality/containment filters\", \"required\": false, \"enum\": null}\n"
    "    ]\n"
    "  },\n"
    "  {\n"
    "    \"name\": \"compute_aggregate\",\n"
    "    \"description\": \"Perform arithmetic reduction (sum, mean, min, max, count, std) on numbers.\",\n"
    "    \"parameters\": [\n"
    "      {\"name\": \"values\", \"type\": \"list\", \"description\": \"Array of numeric values to reduce\", \"required\": true, \"enum\": null},\n"
    "      {\"name\": \"operation\", \"type\": \"string\", \"description\": \"Reduction operation\", \"required\": true, \"enum\": [\"sum\", \"mean\", \"min\", \"max\", \"count\", \"std\"]}\n"
    "    ]\n"
    "  },\n"
    "  {\n"
    "    \"name\": \"verify_constraint\",\n"
    "    \"description\": \"Verify mathematical constraint between actual value and expected baseline.\",\n"
    "    \"parameters\": [\n"
    "      {\"name\": \"actual\", \"type\": \"float\", \"description\": \"Computed metric or actual spend\", \"required\": true, \"enum\": null},\n"
    "      {\"name\": \"expected\", \"type\": \"float\", \"description\": \"Baseline, threshold, or budget cap\", \"required\": true, \"enum\": null},\n"
    "      {\"name\": \"comparison\", \"type\": \"string\", \"description\": \"Comparison operator\", \"required\": true, \"enum\": [\"eq\", \"le\", \"ge\", \"lt\", \"gt\"]},\n"
    "      {\"name\": \"tolerance\", \"type\": \"float\", \"description\": \"Absolute error tolerance for equality\", \"required\": false, \"enum\": null}\n"
    "    ]\n"
    "  }\n"
    "]\n"
    "When finished, emit your final numerical calculation and conclusion."
)


class InteractiveRolloutEvaluator:
    """Multi-turn interactive evaluation orchestrator.

    Executes a complete ReAct conversation loop against either a HuggingFace model
    or an arbitrary policy function.
    """

    def __init__(self, engine: Optional[StandaloneToolEngine] = None, system_prompt: Optional[str] = None) -> None:
        self.engine = engine or StandaloneToolEngine(seed=42)
        self.system_prompt = system_prompt or CANONICAL_SYSTEM_PROMPT

    def run_episode_policy(
        self,
        policy_fn: Callable[[str], str],
        task: Union[Dict[str, Any], Any],
        max_turns: int = 6,
    ) -> Dict[str, Any]:
        """Execute interactive multi-turn loop using a policy function Callable[[str], str]."""
        self.engine.reset()
        task_id = getattr(task, "task_id", None) if not isinstance(task, dict) else task.get("task_id")
        user_query = getattr(task, "user_query", None) if not isinstance(task, dict) else task.get("user_query")
        oracle_answer = getattr(task, "oracle_answer", None) if not isinstance(task, dict) else task.get("oracle_answer")

        # Start conversation prompt
        prompt = (
            f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{self.system_prompt}<|END_OF_TURN_TOKEN|>"
            f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_query}<|END_OF_TURN_TOKEN|>"
        )

        turns: List[Dict[str, Any]] = []
        has_tool_action = False
        final_answer: Optional[str] = None

        for turn_idx in range(max_turns):
            # Model response turn prefix
            active_prompt = prompt + "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            gen_text = policy_fn(active_prompt)

            # Strip any trailing end-of-turn token
            gen_clean = gen_text.replace("<|END_OF_TURN_TOKEN|>", "").strip()

            tool_name, arguments, parse_err = parse_tool_action(gen_clean)

            if tool_name is not None and arguments is not None:
                has_tool_action = True
                res, is_err, err_msg = self.engine.execute(tool_name, arguments)
                if is_err:
                    obs_str = f"Error: {err_msg}"
                else:
                    obs_str = json.dumps(res) if isinstance(res, (dict, list)) else str(res)

                turns.append({
                    "turn_idx": turn_idx,
                    "type": "tool_call",
                    "model_output": gen_clean,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "observation": obs_str,
                })

                # Append assistant output and environment observation to ongoing prompt
                prompt += (
                    f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{gen_clean}<|END_OF_TURN_TOKEN|>"
                    f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Observation: {obs_str}<|END_OF_TURN_TOKEN|>"
                )
            else:
                # Terminal answer reached
                final_answer = gen_clean
                turns.append({
                    "turn_idx": turn_idx,
                    "type": "final_answer",
                    "model_output": gen_clean,
                    "parse_error": parse_err,
                })
                break

        if final_answer is None and turns:
            final_answer = turns[-1]["model_output"]

        is_success = verify_task_outcome_standalone(task, final_answer)

        return {
            "task_id": task_id,
            "user_query": user_query,
            "oracle_answer": oracle_answer,
            "final_answer": final_answer,
            "turns": turns,
            "has_tool_action": has_tool_action,
            "is_success": is_success,
            "num_turns": len(turns),
        }

    def run_episode_model(
        self,
        model: Any,
        tokenizer: Any,
        task: Union[Dict[str, Any], Any],
        device: str = "cuda:0",
        max_turns: int = 6,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Execute interactive multi-turn loop using a PyTorch HuggingFace model and tokenizer."""
        import torch

        self.engine.reset()
        task_id = getattr(task, "task_id", None) if not isinstance(task, dict) else task.get("task_id")
        user_query = getattr(task, "user_query", None) if not isinstance(task, dict) else task.get("user_query")
        oracle_answer = getattr(task, "oracle_answer", None) if not isinstance(task, dict) else task.get("oracle_answer")

        prompt = (
            f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{self.system_prompt}<|END_OF_TURN_TOKEN|>"
            f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_query}<|END_OF_TURN_TOKEN|>"
        )

        turns: List[Dict[str, Any]] = []
        has_tool_action = False
        final_answer: Optional[str] = None
        eos_id = tokenizer.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        model.eval()

        for turn_idx in range(max_turns):
            active_prompt = prompt + "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            enc = tokenizer(active_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

            gen_text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
            gen_clean = gen_text.replace("<|END_OF_TURN_TOKEN|>", "").strip()
            del enc, out

            tool_name, arguments, parse_err = parse_tool_action(gen_clean)

            if tool_name is not None and arguments is not None:
                has_tool_action = True
                res, is_err, err_msg = self.engine.execute(tool_name, arguments)
                if is_err:
                    obs_str = f"Error: {err_msg}"
                else:
                    obs_str = json.dumps(res) if isinstance(res, (dict, list)) else str(res)

                turns.append({
                    "turn_idx": turn_idx,
                    "type": "tool_call",
                    "model_output": gen_clean,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "observation": obs_str,
                })

                prompt += (
                    f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{gen_clean}<|END_OF_TURN_TOKEN|>"
                    f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Observation: {obs_str}<|END_OF_TURN_TOKEN|>"
                )
            else:
                final_answer = gen_clean
                turns.append({
                    "turn_idx": turn_idx,
                    "type": "final_answer",
                    "model_output": gen_clean,
                    "parse_error": parse_err,
                })
                break

        if final_answer is None and turns:
            final_answer = turns[-1]["model_output"]

        is_success = verify_task_outcome_standalone(task, final_answer)

        return {
            "task_id": task_id,
            "user_query": user_query,
            "oracle_answer": oracle_answer,
            "final_answer": final_answer,
            "turns": turns,
            "has_tool_action": has_tool_action,
            "is_success": is_success,
            "num_turns": len(turns),
        }

    def evaluate_validation_tasks(
        self,
        val_tasks: List[Union[Dict[str, Any], Any]],
        policy_or_model: Any,
        tokenizer: Optional[Any] = None,
        device: str = "cuda:0",
        run_name: str = "eval_run",
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """Evaluate a full validation split and return (results, m1, m2)."""
        results = []
        tool_calls_count = 0
        successes = 0

        is_callable = callable(policy_or_model) and tokenizer is None

        for idx, task in enumerate(val_tasks):
            if is_callable:
                res = self.run_episode_policy(policy_or_model, task)
            else:
                res = self.run_episode_model(policy_or_model, tokenizer, task, device=device)

            if res["has_tool_action"]:
                tool_calls_count += 1
            if res["is_success"]:
                successes += 1

            results.append(res)
            t_id = res["task_id"]
            print(f"Task {idx:02d} ({t_id}): action={res['has_tool_action']}, success={res['is_success']}")

        total = len(val_tasks) if val_tasks else 1
        m1 = successes / total
        m2 = tool_calls_count / total
        print(f"SUMMARY [{run_name}]: M1 (Success)={m1*100:.1f}%, M2 (Tool Calls)={m2*100:.1f}%")
        return results, m1, m2

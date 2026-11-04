"""Model generation parser for structured tool call actions and terminal answers."""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple
from src.data.schemas import ToolCallAction

# Regex matching Cohere Command-R / Aya tool call tokens
COHERE_ACTION_PATTERN = re.compile(r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.DOTALL)
THOUGHT_PATTERN = re.compile(r"<\|START_THINKING\|>(.*?)<\|END_THINKING\|>", re.DOTALL)


def parse_agent_output(
    text: str,
) -> Tuple[Optional[str], Optional[ToolCallAction], Optional[str], Optional[str]]:
    """Parse raw model output into (thought, tool_action, final_answer, parse_error).

    Guarantees:
    - Never throws unhandled exceptions on malformed text.
    - Explicit parse error reported if action tags contain non-JSON payload.
    """
    if not text:
        return None, None, None, "EMPTY_MODEL_OUTPUT"

    # 1. Extract thought if present
    thought: Optional[str] = None
    thought_match = THOUGHT_PATTERN.search(text)
    if thought_match:
        thought = thought_match.group(1).strip()

    # 2. Extract action if present
    action_match = COHERE_ACTION_PATTERN.search(text)
    if action_match:
        raw_payload = action_match.group(1).strip()
        try:
            parsed = json.loads(raw_payload)
            tool_name = parsed.get("name") or parsed.get("tool_name")
            arguments = parsed.get("arguments") or parsed.get("parameters") or {}
            if not tool_name:
                return thought, None, None, "MALFORMED_TOOL_ACTION_MISSING_NAME"
            return thought, ToolCallAction(tool_name=tool_name, arguments=arguments), None, None
        except json.JSONDecodeError as e:
            return thought, None, None, f"JSON_PARSE_ERROR: {str(e)}"

    # 3. Check for pure JSON action fallback
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            tool_name = parsed.get("name") or parsed.get("tool_name")
            if tool_name:
                arguments = parsed.get("arguments") or parsed.get("parameters") or {}
                return thought, ToolCallAction(tool_name=tool_name, arguments=arguments), None, None
        except json.JSONDecodeError:
            pass

    # 4. Otherwise treat text as final answer
    cleaned_final = re.sub(COHERE_ACTION_PATTERN, "", text)
    cleaned_final = re.sub(THOUGHT_PATTERN, "", cleaned_final).strip()
    return thought, None, cleaned_final, None

"""Model generation parser for structured tool call actions and terminal answers."""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple
from pydantic import ValidationError

from src.data.schemas import ToolCallAction

# Regex matching Cohere Command-R / Aya tool call tokens and XML/Markdown variants
COHERE_ACTION_PATTERN = re.compile(r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.DOTALL)
XML_ACTION_PATTERN = re.compile(r"<(?:tool_call|action)>(.*?)</(?:tool_call|action)>", re.DOTALL)
MARKDOWN_JSON_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
THOUGHT_PATTERN = re.compile(r"<\|START_THINKING\|>(.*?)<\|END_THINKING\|>", re.DOTALL)


def _extract_dict_action(parsed: dict) -> Tuple[Optional[ToolCallAction], Optional[str]]:
    """Helper to convert parsed dict (supporting OpenAI tool_calls, parameters, arguments) into ToolCallAction."""
    # Handle OpenAI tool_calls wrapper format
    if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list) and len(parsed["tool_calls"]) > 0:
        first_call = parsed["tool_calls"][0]
        if isinstance(first_call, dict) and "function" in first_call:
            fn = first_call["function"]
            tool_name = fn.get("name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    return None, f"JSON_PARSE_ERROR in tool_calls arguments: {args}"
            if tool_name and isinstance(args, dict):
                return ToolCallAction(tool_name=tool_name, arguments=args), None

    tool_name = parsed.get("name") or parsed.get("tool_name")
    if not tool_name or not isinstance(tool_name, str):
        return None, "MALFORMED_TOOL_ACTION_MISSING_NAME"

    arguments = parsed.get("arguments")
    if arguments is None:
        arguments = parsed.get("parameters", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            return None, f"MALFORMED_TOOL_ARGUMENTS: Could not parse string arguments as JSON: {arguments}"

    if not isinstance(arguments, dict):
        return None, f"MALFORMED_TOOL_ARGUMENTS: Expected dict for arguments, got {type(arguments).__name__}"

    try:
        action = ToolCallAction(tool_name=tool_name, arguments=arguments)
        return action, None
    except ValidationError as ve:
        return None, f"SCHEMA_VALIDATION_ERROR: {str(ve)}"


def parse_agent_output(
    text: str,
) -> Tuple[Optional[str], Optional[ToolCallAction], Optional[str], Optional[str]]:
    """Parse raw model output into (thought, tool_action, final_answer, parse_error).

    Guarantees:
    - Never throws unhandled exceptions on malformed text.
    - Explicit parse error reported if action tags are truncated, malformed, or contain non-JSON payload.
    - Multi-format support: Cohere delimiters, XML <tool_call>, Markdown json blocks, and OpenAI tool_calls.
    - Rejects non-dict tool argument payloads safely with typed errors.
    """
    if not text or not text.strip():
        return None, None, None, "EMPTY_MODEL_OUTPUT"

    # 1. Extract thought if present
    thought: Optional[str] = None
    thought_match = THOUGHT_PATTERN.search(text)
    if thought_match:
        thought = thought_match.group(1).strip()

    # 2. Check for truncated or unbalanced action delimiter tags
    has_start_action = "<|START_ACTION|>" in text
    has_end_action = "<|END_ACTION|>" in text

    if has_start_action and not has_end_action:
        return thought, None, None, "TRUNCATED_ACTION_TOKEN: Unclosed <|START_ACTION|> tag"

    if has_end_action and not has_start_action:
        return thought, None, None, "MALFORMED_ACTION_TOKEN: Found <|END_ACTION|> without <|START_ACTION|>"

    # 3. Extract delimited action if present (Cohere format)
    action_match = COHERE_ACTION_PATTERN.search(text)
    if action_match:
        raw_payload = action_match.group(1).strip()
        try:
            parsed = json.loads(raw_payload)
            if not isinstance(parsed, dict):
                return thought, None, None, "MALFORMED_TOOL_ACTION_NOT_A_DICT"
            action, err = _extract_dict_action(parsed)
            if err:
                return thought, None, None, err
            return thought, action, None, None
        except json.JSONDecodeError as e:
            return thought, None, None, f"JSON_PARSE_ERROR: {str(e)}"

    # 4. Check for XML tags (<tool_call>...</tool_call> or <action>...</action>)
    xml_match = XML_ACTION_PATTERN.search(text)
    if xml_match:
        raw_xml_payload = xml_match.group(1).strip()
        try:
            parsed = json.loads(raw_xml_payload)
            if isinstance(parsed, dict):
                action, err = _extract_dict_action(parsed)
                if err:
                    return thought, None, None, err
                if action:
                    return thought, action, None, None
            else:
                return thought, None, None, "MALFORMED_TOOL_ACTION_NOT_A_DICT"
        except json.JSONDecodeError as e:
            return thought, None, None, f"JSON_PARSE_ERROR: {str(e)}"

    # 5. Check for Markdown fenced JSON blocks (```json {...} ```)
    md_match = MARKDOWN_JSON_PATTERN.search(text)
    if md_match:
        raw_md_payload = md_match.group(1).strip()
        try:
            parsed = json.loads(raw_md_payload)
            if isinstance(parsed, dict):
                if "name" in parsed or "tool_name" in parsed or "tool_calls" in parsed:
                    action, err = _extract_dict_action(parsed)
                    if err:
                        return thought, None, None, err
                    if action:
                        return thought, action, None, None
        except json.JSONDecodeError as e:
            if '"name"' in raw_md_payload or '"tool"' in raw_md_payload:
                return thought, None, None, f"JSON_PARSE_ERROR: {str(e)}"

    # 6. Check for pure JSON action fallback (e.g. tool calling without special tokens)
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                if "name" in parsed or "tool_name" in parsed or "tool_calls" in parsed:
                    action, err = _extract_dict_action(parsed)
                    if err:
                        return thought, None, None, err
                    if action:
                        return thought, action, None, None
        except json.JSONDecodeError as e:
            if '"name"' in stripped or '"tool"' in stripped:
                return thought, None, None, f"JSON_PARSE_ERROR: {str(e)}"

    # 7. Otherwise treat text as final answer
    cleaned_final = re.sub(COHERE_ACTION_PATTERN, "", text)
    cleaned_final = re.sub(XML_ACTION_PATTERN, "", cleaned_final)
    cleaned_final = re.sub(MARKDOWN_JSON_PATTERN, "", cleaned_final)
    cleaned_final = re.sub(THOUGHT_PATTERN, "", cleaned_final).strip()
    return thought, None, cleaned_final, None


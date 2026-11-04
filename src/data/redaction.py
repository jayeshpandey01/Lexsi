"""Automated secret, token, and personal information redaction filter."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Union

# Compiled regex patterns for redaction
REDACTION_PATTERNS = [
    # Bearer tokens
    (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), "Bearer [REDACTED_TOKEN]"),
    # API keys / token assignments
    (
        re.compile(r"(?i)(api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*['\"]?([A-Za-z0-9\-_]{8,})['\"]?"),
        r"\1=[REDACTED_CREDENTIAL]",
    ),
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
    # Private keys
    (
        re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"),
        "[REDACTED_PRIVATE_KEY]",
    ),
]


def redact_text(text: str) -> str:
    """Apply compiled regex patterns to redact sensitive credentials and PII."""
    if not text:
        return text
    sanitized = text
    for pattern, replacement in REDACTION_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def redact_data(data: Any) -> Any:
    """Recursively scrub sensitive strings from nested dictionaries and lists."""
    if isinstance(data, str):
        return redact_text(data)
    if isinstance(data, dict):
        return {k: redact_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [redact_data(item) for item in data]
    return data

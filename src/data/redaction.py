"""Automated secret, token, and personal information redaction filter."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple, Union

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

# Keys in dictionaries that indicate secret or credential payloads
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|password|access[_-]?token|auth|credential|private[_-]?key)"
)


def redact_text(text: str) -> str:
    """Apply compiled regex patterns to redact sensitive credentials and PII."""
    if not text:
        return text
    sanitized = text
    for pattern, replacement in REDACTION_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def redact_data(data: Any) -> Any:
    """Recursively scrub sensitive strings from nested dictionaries, lists, sets, and tuples.

    Guarantees:
    - Values mapped to secret-bearing keys (e.g. 'api_key', 'password') are redacted even if
      the value string lacks inline key assignments.
    - All string values undergo regex pattern redaction (tokens, private keys, emails).
    - Benign domain payloads (vendor records, numerical metrics, inventory names) remain intact.
    """
    if isinstance(data, str):
        return redact_text(data)

    if isinstance(data, dict):
        sanitized_dict: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(k, str) and SENSITIVE_KEY_PATTERN.search(k):
                if isinstance(v, str) and v.strip():
                    sanitized_dict[k] = "[REDACTED_CREDENTIAL]"
                else:
                    sanitized_dict[k] = redact_data(v)
            else:
                sanitized_dict[k] = redact_data(v)
        return sanitized_dict

    if isinstance(data, list):
        return [redact_data(item) for item in data]

    if isinstance(data, tuple):
        return tuple(redact_data(item) for item in data)

    if isinstance(data, set):
        return {redact_data(item) for item in data}

    return data


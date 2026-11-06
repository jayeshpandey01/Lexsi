"""Langfuse production observability and telemetry instrumentation.

Follows Langfuse best practices (SKILL.md / instrumentation.md / v5 SDK):
- Hierarchical observations: agent (root trace) -> generation (LLM policy) -> tool (tool calls).
- Automatic token usage, latency, and level (DEFAULT / ERROR) tracking.
- Recursive PII/credential redaction on inputs and outputs.
- Environment-verified outcome scoring (task_success).
- Graceful no-op fallback when Langfuse is disabled or credentials are absent.
"""

from contextlib import contextmanager
import logging
import time
from typing import Any, Dict, List, Optional

from src.data.redaction import redact_data, redact_text
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

# Attempt to import Langfuse
try:
    from langfuse import Langfuse
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False


class TelemetryManager:
    """Manages Langfuse tracing spans, generations, and task outcome scores."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client: Optional[Any] = None

        if self.settings.enable_langfuse_tracing and _LANGFUSE_AVAILABLE:
            if self.settings.langfuse_public_key and self.settings.langfuse_secret_key:
                try:
                    self.client = Langfuse(
                        public_key=self.settings.langfuse_public_key,
                        secret_key=self.settings.langfuse_secret_key,
                        host=self.settings.langfuse_host,
                    )
                    logger.info("Langfuse telemetry initialized successfully at %s", self.settings.langfuse_host)
                except Exception as e:
                    logger.warning("Failed to initialize Langfuse client: %s", e)
                    self.client = None

    @property
    def is_active(self) -> bool:
        """Return True if Langfuse client is initialized and active."""
        return self.client is not None

    def start_trace(
        self,
        name: str = "procurement-audit-agent",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        user_query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Create a new root agent observation in Langfuse."""
        if not self.is_active or not self.client:
            return None

        clean_query = redact_text(user_query) if user_query else ""
        meta = redact_data(metadata or {})
        if task_id:
            meta["task_id"] = task_id
        if session_id:
            meta["session_id"] = session_id
        if user_id:
            meta["user_id"] = user_id

        all_tags = ["track3", "procurement", "production"]
        if tags:
            all_tags.extend(tags)
        meta["tags"] = list(set(all_tags))

        try:
            agent_obs = self.client.start_observation(
                name=name,
                as_type="agent",
                input={"query": clean_query},
                metadata=meta,
            )
            return agent_obs
        except Exception as e:
            logger.warning("Error creating Langfuse root observation: %s", e)
            return None

    def log_generation(
        self,
        trace: Optional[Any],
        name: str = "model-policy-call",
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        output_text: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Record an LLM policy generation inside an active trace."""
        if not trace:
            return None

        clean_messages = redact_data(messages or [])
        clean_output = redact_text(output_text or "")
        clean_meta = redact_data(metadata or {})

        usage_details = {}
        if usage:
            usage_details = {
                "input": usage.get("prompt_tokens", usage.get("input", 0)),
                "output": usage.get("completion_tokens", usage.get("output", 0)),
                "total": usage.get("total_tokens", usage.get("total", 0)),
            }

        try:
            gen_obs = trace.start_observation(
                name=name,
                as_type="generation",
                model=model or self.settings.default_model_name,
                input=clean_messages,
                output=clean_output,
                usage_details=usage_details,
                metadata=clean_meta,
            )
            gen_obs.end()
            return gen_obs
        except Exception as e:
            logger.warning("Error logging Langfuse generation: %s", e)
            return None

    def log_tool_span(
        self,
        trace: Optional[Any],
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        is_error: bool = False,
        error_message: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> Optional[Any]:
        """Record a tool execution span inside an active trace."""
        if not trace:
            return None

        clean_args = redact_data(arguments)
        clean_res = redact_data(result) if not is_error else None
        clean_err = redact_text(error_message) if error_message else None

        span_metadata: Dict[str, Any] = {
            "is_error": is_error,
        }
        if duration_seconds is not None:
            span_metadata["duration_seconds"] = round(duration_seconds, 4)

        try:
            tool_obs = trace.start_observation(
                name=f"tool:{tool_name}",
                as_type="tool",
                input=clean_args,
                output={"error": clean_err} if is_error else clean_res,
                level="ERROR" if is_error else "DEFAULT",
                status_message=clean_err,
                metadata=span_metadata,
            )
            tool_obs.end()
            return tool_obs
        except Exception as e:
            logger.warning("Error logging Langfuse tool span: %s", e)
            return None

    def record_outcome(
        self,
        trace: Optional[Any],
        is_success: bool,
        final_output: Optional[str] = None,
        comment: str = "Environment verified outcome predicate",
    ) -> None:
        """Record the final task success score, update output, and end the trace."""
        if not trace:
            return

        clean_out = redact_text(final_output) if final_output else ""
        try:
            # 1. Score the root trace
            trace.score_trace(
                name="task_success",
                value=1.0 if is_success else 0.0,
                comment=comment,
            )
            # 2. Update trace final output and close observation
            trace.update(output={"final_response": clean_out, "is_success": is_success})
            trace.end()
        except Exception as e:
            logger.warning("Error recording Langfuse outcome score: %s", e)

    def flush(self) -> None:
        """Flush queued telemetry events to Langfuse cloud."""
        if self.is_active and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.warning("Error flushing Langfuse client: %s", e)


# Global singleton instance
_telemetry_manager: Optional[TelemetryManager] = None


def get_telemetry_manager() -> TelemetryManager:
    """Retrieve global singleton TelemetryManager."""
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
    return _telemetry_manager

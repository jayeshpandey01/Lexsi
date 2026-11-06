"""Unit tests for Langfuse telemetry manager."""

from src.evaluation.telemetry import TelemetryManager
from src.utils.config import Settings


def test_telemetry_manager_disabled() -> None:
    """Verify telemetry manager behaves as a graceful no-op when disabled."""
    settings = Settings(enable_langfuse_tracing=False)
    tm = TelemetryManager()
    tm.settings = settings
    tm.client = None

    assert tm.is_active is False
    trace = tm.start_trace(name="noop-trace")
    assert trace is None
    # No-op calls must not raise errors
    tm.log_generation(trace, model="test-model", messages=[])
    tm.log_tool_span(trace, tool_name="test_tool", arguments={}, result={})
    tm.record_outcome(trace, is_success=True)
    tm.flush()


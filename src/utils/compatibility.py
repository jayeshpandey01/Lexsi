"""Compatibility adapters and bridge interfaces for AgentTune and AuditKIT.

Authoritative schemas and defensive handlers for upstream PR caveats:
- AgentTune PR #5 (lazy tool loading) and PR #6 (non-finite judge scores).
- AuditKIT PR #18 (unmeasured edge cases) and PR #9 (bootstrap confidence intervals).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class AgentTuneEvent(BaseModel):
    """Single chronological event in AgentTune Two-Tier EventLog."""

    event_type: str = Field(
        ...,
        description="Event category: 'user_query', 'agent_action', 'tool_observation', 'agent_response'",
    )
    timestamp: float = Field(..., description="UNIX epoch timestamp in seconds")
    tool_name: Optional[str] = Field(None, description="Invoked tool name if agent_action")
    arguments: Optional[Dict[str, Any]] = Field(None, description="Serialized tool arguments")
    result: Optional[Any] = Field(None, description="Normalized environment observation result")
    content: Optional[str] = Field(None, description="Textual message payload")
    error_code: Optional[str] = Field(None, description="Normalized error code if failure occurred")


class AgentTuneEventLog(BaseModel):
    """Complete episode trajectory record for AgentTune SFT and RL projections."""

    episode_id: str = Field(..., description="Unique episode identifier")
    task_id: str = Field(..., description="Target benchmark task identifier")
    seed: int = Field(..., description="Deterministic environment seed")
    events: List[AgentTuneEvent] = Field(default_factory=list)

    def to_training_dict(self) -> Dict[str, Any]:
        """Convert EventLog to serialized dictionary representation."""
        return self.model_dump()


class AuditKitMetricResult(BaseModel):
    """Individual metric evaluation summary matching AuditKIT standards."""

    metric_name: str
    value: float
    std: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    sample_count: int

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, v: float) -> float:
        """AuditKIT PR #18 / AgentTune PR #6 defense: reject non-finite scores."""
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v


class AuditKitRunResult(BaseModel):
    """Canonical RunResult record matching AuditKIT unified evaluation spine."""

    run_id: str
    model_id: str
    timestamp: float
    sha256_fingerprint: str
    metrics: Dict[str, AuditKitMetricResult]
    failure_breakdown: Dict[str, int] = Field(default_factory=dict)
    provenance_metadata: Dict[str, Any] = Field(default_factory=dict)


def compute_run_fingerprint(
    model_id: str,
    dataset_hash: str,
    config: Dict[str, Any],
    seed: int,
) -> str:
    """Generate deterministic SHA-256 provenance fingerprint for an evaluation run.

    Combines model identification, dataset hash, configuration dictionary,
    and random seed in a canonical JSON format.
    """
    payload = {
        "model_id": model_id,
        "dataset_hash": dataset_hash,
        "config": config,
        "seed": seed,
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

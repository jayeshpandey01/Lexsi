"""Trajectory records, step logs, and ShareGPT/AgentTune serialization schemas."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.data.redaction import redact_data, redact_text
from src.utils.compatibility import AgentTuneEvent, AgentTuneEventLog


class ToolCallAction(BaseModel):
    """Normalized structured tool call emitted by the agent."""

    tool_name: str = Field(..., description="Target tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Parsed tool arguments")
    call_id: str = Field(default_factory=lambda: f"call_{int(time.time()*1000)}", description="Unique call ID")


class StepRecord(BaseModel):
    """Single discrete step within an agent execution trajectory."""

    step_index: int = Field(..., description="0-indexed step counter")
    thought: Optional[str] = Field(None, description="Optional reasoning / scratchpad output")
    action: Optional[ToolCallAction] = Field(None, description="Tool action if step invoked a tool")
    observation: Optional[Any] = Field(None, description="Environment observation / tool result")
    error: Optional[str] = Field(None, description="Error message if step failed")
    elapsed_ms: float = Field(0.0, description="Step duration in milliseconds")


class TrajectoryRecord(BaseModel):
    """Complete episodic trajectory for auditing, evaluation, and post-training."""

    task_id: str
    seed: int
    system_prompt: str
    user_query: str
    steps: List[StepRecord] = Field(default_factory=list)
    final_output: Optional[str] = None
    is_success: bool = False
    total_steps: int = 0
    raw_reward: float = 0.0
    fingerprint: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        """Compute state fingerprint if not provided."""
        if not self.fingerprint:
            payload = f"{self.task_id}_{self.seed}_{len(self.steps)}_{self.is_success}"
            self.fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_redacted_dict(self) -> Dict[str, Any]:
        """Produce sanitized trajectory dictionary with secrets scrubbed."""
        raw_dict = self.model_dump()
        return redact_data(raw_dict)  # type: ignore[no-any-return]

    def to_sharegpt_format(self) -> List[Dict[str, str]]:
        """Project trajectory to ShareGPT format for supervised fine-tuning."""
        messages: List[Dict[str, str]] = []

        if self.system_prompt:
            messages.append({"from": "system", "value": redact_text(self.system_prompt)})

        messages.append({"from": "human", "value": redact_text(self.user_query)})

        for step in self.steps:
            if step.thought:
                messages.append({"from": "thought", "value": redact_text(step.thought)})
            if step.action:
                action_text = f"<|START_ACTION|>{json.dumps({'name': step.action.tool_name, 'arguments': step.action.arguments})}<|END_ACTION|>"
                messages.append({"from": "gpt", "value": action_text})
            if step.observation is not None:
                obs_text = json.dumps(redact_data(step.observation))
                messages.append({"from": "observation", "value": obs_text})
            elif step.error:
                messages.append({"from": "observation", "value": f"ERROR: {redact_text(step.error)}"})

        if self.final_output:
            messages.append({"from": "gpt", "value": redact_text(self.final_output)})

        return messages

    def to_agenttune_eventlog(self, episode_id: Optional[str] = None) -> AgentTuneEventLog:
        """Project trajectory into AgentTune Two-Tier EventLog specification."""
        ep_id = episode_id or f"ep_{self.task_id}_{self.seed}"
        events: List[AgentTuneEvent] = []

        now = time.time()
        # 1. User Query
        events.append(
            AgentTuneEvent(
                event_type="user_query",
                timestamp=now,
                content=redact_text(self.user_query),
            )
        )

        # 2. Sequential steps
        step_time = now + 0.1
        for step in self.steps:
            if step.action:
                events.append(
                    AgentTuneEvent(
                        event_type="agent_action",
                        timestamp=step_time,
                        tool_name=step.action.tool_name,
                        arguments=redact_data(step.action.arguments),
                    )
                )
                step_time += 0.05

            if step.observation is not None or step.error:
                events.append(
                    AgentTuneEvent(
                        event_type="tool_observation",
                        timestamp=step_time,
                        result=redact_data(step.observation),
                        error_code=step.error,
                    )
                )
                step_time += 0.05

        # 3. Final Agent Response
        if self.final_output:
            events.append(
                AgentTuneEvent(
                    event_type="agent_response",
                    timestamp=step_time,
                    content=redact_text(self.final_output),
                )
            )

        return AgentTuneEventLog(
            episode_id=ep_id,
            task_id=self.task_id,
            seed=self.seed,
            events=events,
        )

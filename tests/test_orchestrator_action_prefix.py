"""Unit tests for action delimiter pre-filling and multi-turn execution loop (Milestone 6).

Guarantees:
- Verifies that prefill_action_token=True injects <|START_ACTION|> on turn 0 to steer generation.
- Verifies backwards compatibility: prefill_action_token=False leaves prompts unchanged.
- Ensures seamless multi-turn execution when model emits action completion payloads.
- Validates trajectory metadata capture for auditability and provenance.
"""

from __future__ import annotations

import json
from typing import List
import pytest

from src.data.schemas import ToolCallAction, TrajectoryRecord
from src.environment.task_generator import TaskSpec, generate_task_instance
from src.runtime.orchestrator import EpisodeOrchestrator


@pytest.fixture
def sample_task() -> TaskSpec:
    return generate_task_instance("train", 0, seed=42)


def test_orchestrator_prefill_disabled_by_default(sample_task: TaskSpec) -> None:
    """Verify default orchestrator does not append <|START_ACTION|> to input prompt."""
    orchestrator = EpisodeOrchestrator(seed=42)
    assert orchestrator.prefill_action_token is False

    captured_prompts: List[str] = []

    def mock_policy(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "The total inventory valuation is $1000.00."

    traj = orchestrator.run_episode(sample_task, mock_policy)

    assert len(captured_prompts) == 1
    assert not captured_prompts[0].rstrip().endswith("<|START_ACTION|>")
    assert traj.metadata["prefill_action_token"] is False
    assert traj.total_steps == 1


def test_orchestrator_prefill_enabled_turn_0(sample_task: TaskSpec) -> None:
    """Verify prefill_action_token=True appends <|START_ACTION|> on turn 0 only."""
    orchestrator = EpisodeOrchestrator(seed=42, prefill_action_token=True)
    assert orchestrator.prefill_action_token is True

    captured_prompts: List[str] = []

    def multi_turn_policy(prompt: str) -> str:
        captured_prompts.append(prompt)
        if len(captured_prompts) == 1:
            # Policy completes the pre-filled <|START_ACTION|> turn
            return '{"name": "lookup_records", "arguments": {"table": "inventory_items", "filters": {"category": "Software"}}}<|END_ACTION|>'
        # Turn 1: final answer
        return "The inventory calculation is complete."

    traj = orchestrator.run_episode(sample_task, multi_turn_policy)

    assert len(captured_prompts) == 2
    # Turn 0 prompt ends with <|START_ACTION|>
    assert captured_prompts[0].rstrip().endswith("<|START_ACTION|>")
    # Turn 1 prompt does NOT end with <|START_ACTION|>
    assert not captured_prompts[1].rstrip().endswith("<|START_ACTION|>")

    assert traj.metadata["prefill_action_token"] is True
    assert len(traj.steps) == 2
    assert traj.steps[0].action is not None
    assert traj.steps[0].action.tool_name == "lookup_records"
    assert traj.steps[0].observation is not None
    assert traj.steps[1].action is None


def test_orchestrator_prefill_reconstructs_action_delimiter(sample_task: TaskSpec) -> None:
    """Verify that action reconstruction handles outputs where model does not repeat the prefix."""
    orchestrator = EpisodeOrchestrator(seed=42, prefill_action_token=True)

    def policy_returning_bare_json(prompt: str) -> str:
        if "<|START_ACTION|>" in prompt:
            # Model returns JSON ending with <|END_ACTION|>
            return '{"name": "lookup_records", "arguments": {"table": "vendors"}}<|END_ACTION|>'
        return "Finished audit."

    traj = orchestrator.run_episode(sample_task, policy_returning_bare_json)

    assert len(traj.steps) == 2
    step0 = traj.steps[0]
    assert step0.error is None
    assert step0.action is not None
    assert step0.action.tool_name == "lookup_records"
    assert step0.action.arguments == {"table": "vendors"}


def test_orchestrator_prefill_malformed_json_handling(sample_task: TaskSpec) -> None:
    """Verify that a syntax error in the prefilled action completion is caught gracefully."""
    orchestrator = EpisodeOrchestrator(seed=42, prefill_action_token=True)

    def policy_returning_syntax_error(prompt: str) -> str:
        return '{"name": "lookup_records", "arguments": {broken_json}<|END_ACTION|>'

    traj = orchestrator.run_episode(sample_task, policy_returning_syntax_error)

    assert len(traj.steps) == 1
    step0 = traj.steps[0]
    assert step0.action is None
    assert step0.error is not None
    assert "SCHEMA_VALIDATION_ERROR" in step0.error
    assert traj.is_success is False

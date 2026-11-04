"""Episode orchestrator governing the model-tool execution loop."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

from src.data.schemas import StepRecord, ToolCallAction, TrajectoryRecord
from src.environment.guards import RuntimePolicyGuard
from src.environment.sandbox import SandboxedToolEngine, ToolObservation
from src.environment.schema import ToolExecutionError
from src.environment.task_generator import TaskSpec
from src.environment.tools import TOOL_CONTRACTS
from src.runtime.parser import parse_agent_output


class EpisodeOrchestrator:
    """Orchestrates single-task inference, tool calling, and trace recording.

    Required Invariants (system_design.md §3.1, instruction.md):
    - Cannot exceed configured step budget (T_max = 8) or time limit (30s).
    - Model text is never executed directly as shell commands.
    - All actions validated against permissions and schemas outside the model.
    - Captures redacted, replayable trajectory records.
    """

    def __init__(
        self,
        seed: int = 42,
        max_steps: int = 8,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.seed = seed
        self.engine = SandboxedToolEngine(seed=seed)
        self.guard = RuntimePolicyGuard(max_steps=max_steps, timeout_seconds=timeout_seconds)

    def _build_system_prompt(self) -> str:
        """Render system prompt describing available deterministic tools."""
        tools_doc: List[Dict[str, Any]] = []
        for name, contract in TOOL_CONTRACTS.items():
            tools_doc.append(
                {
                    "name": name,
                    "description": contract.description,
                    "parameters": [p.model_dump() for p in contract.parameters],
                }
            )

        return (
            "You are an expert procurement and inventory auditing agent. "
            "To solve tasks, you may invoke the following deterministic tools using "
            "<|START_ACTION|>{\"name\": \"<tool_name>\", \"arguments\": {<args>}}<|END_ACTION|>. "
            "Available tools:\n"
            f"{json.dumps(tools_doc, indent=2)}\n"
            "When finished, emit your final numerical calculation and conclusion."
        )

    def run_episode(
        self,
        task: TaskSpec,
        policy_fn: Callable[[str], str],
    ) -> TrajectoryRecord:
        """Execute a complete agent episode on a specific task instance."""
        start_time = time.time()
        self.engine.reset(seed=self.seed)

        system_prompt = self._build_system_prompt()
        dialog_history = f"{system_prompt}\n\nTask: {task.user_query}"

        steps: List[StepRecord] = []
        final_answer: Optional[str] = None
        current_step = 0

        while current_step < self.guard.max_steps:
            # 1. Budget and timeout checks
            try:
                self.guard.check_budget(current_step, start_time)
            except ToolExecutionError as e:
                steps.append(
                    StepRecord(
                        step_index=current_step,
                        error=f"{e.error_code}: {e.message}",
                        elapsed_ms=(time.time() - start_time) * 1000.0,
                    )
                )
                break

            # 2. Policy generation
            step_start = time.perf_counter()
            model_output = policy_fn(dialog_history)

            # 3. Output parsing
            thought, action, answer, parse_err = parse_agent_output(model_output)

            if parse_err:
                step_elapsed = (time.perf_counter() - step_start) * 1000.0
                steps.append(
                    StepRecord(
                        step_index=current_step,
                        thought=thought,
                        error=f"SCHEMA_VALIDATION_ERROR: {parse_err}",
                        elapsed_ms=round(step_elapsed, 2),
                    )
                )
                break

            if answer is not None and action is None:
                final_answer = answer
                step_elapsed = (time.perf_counter() - step_start) * 1000.0
                steps.append(
                    StepRecord(
                        step_index=current_step,
                        thought=thought,
                        elapsed_ms=round(step_elapsed, 2),
                    )
                )
                break

            if action is not None:
                # 4. Guard permission validation
                try:
                    self.guard.validate_tool_permission(action.tool_name)
                    # 5. Sandboxed execution
                    obs: ToolObservation = self.engine.execute(action.tool_name, action.arguments)
                    step_elapsed = (time.perf_counter() - step_start) * 1000.0

                    if obs.is_error:
                        steps.append(
                            StepRecord(
                                step_index=current_step,
                                thought=thought,
                                action=action,
                                error=f"{obs.error_code}: {obs.error_message}",
                                elapsed_ms=round(step_elapsed, 2),
                            )
                        )
                        dialog_history += f"\nObservation Error: {obs.error_message}"
                    else:
                        sanitized_res = self.guard.sanitize_observation(obs.result)
                        steps.append(
                            StepRecord(
                                step_index=current_step,
                                thought=thought,
                                action=action,
                                observation=obs.result,
                                elapsed_ms=round(step_elapsed, 2),
                            )
                        )
                        dialog_history += f"\nObservation: {sanitized_res}"

                except ToolExecutionError as e:
                    step_elapsed = (time.perf_counter() - step_start) * 1000.0
                    steps.append(
                        StepRecord(
                            step_index=current_step,
                            thought=thought,
                            action=action,
                            error=f"{e.error_code}: {e.message}",
                            elapsed_ms=round(step_elapsed, 2),
                        )
                    )
                    break

            current_step += 1

        # Evaluate success predicate
        is_success = self._evaluate_success(final_answer, task)
        raw_reward = 1.0 if is_success else 0.0

        return TrajectoryRecord(
            task_id=task.task_id,
            seed=self.seed,
            system_prompt=system_prompt,
            user_query=task.user_query,
            steps=steps,
            final_output=final_answer,
            is_success=is_success,
            total_steps=len(steps),
            raw_reward=raw_reward,
        )

    def _evaluate_success(self, final_answer: Optional[str], task: TaskSpec) -> bool:
        """Deterministic predicate check comparing final answer to task oracle answer."""
        if not final_answer:
            return False

        # Extract floats from final response text
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", final_answer.replace(",", ""))
        if not nums:
            return False

        for n_str in nums:
            try:
                val = float(n_str)
                # Check within tolerance 1e-2 or 1% relative
                abs_err = abs(val - task.oracle_answer)
                rel_err = abs_err / (abs(task.oracle_answer) + 1e-5)
                if abs_err <= 0.05 or rel_err <= 0.01:
                    return True
            except ValueError:
                continue

        return False

"""Episode orchestrator governing the model-tool execution loop."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from src.data.schemas import StepRecord, ToolCallAction, TrajectoryRecord
from src.environment.guards import RuntimePolicyGuard
from src.environment.sandbox import SandboxedToolEngine, ToolObservation
from src.environment.schema import ToolExecutionError
from src.environment.task_generator import TaskSpec, verify_task_outcome
from src.environment.tools import TOOL_CONTRACTS
from src.runtime.parser import parse_agent_output


class EpisodeOrchestrator:
    """Orchestrates single-task inference, tool calling, and trace recording.

    Required Invariants (system_design.md §3.1, instruction.md):
    - Cannot exceed configured step budget (T_max = 8) or time limit (30s).
    - Model text is never executed directly as shell commands.
    - All actions validated against permissions and schemas outside the model.
    - Captures redacted, replayable trajectory records with complete provenance.
    - Environment-owned task outcome verification.
    """

    def __init__(
        self,
        seed: int = 42,
        max_steps: int = 8,
        timeout_seconds: float = 30.0,
        per_call_timeout_seconds: float = 5.0,
        allowed_tools: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        environment_version: str = "0.1.0",
        use_database: bool = False,
        prefill_action_token: bool = False,
    ) -> None:
        self.seed = seed
        self.run_id = run_id
        self.environment_version = environment_version
        self.prefill_action_token = prefill_action_token
        self.engine = SandboxedToolEngine(
            seed=seed,
            allowed_tools=allowed_tools,
            per_call_timeout_seconds=per_call_timeout_seconds,
            use_database=use_database,
        )
        self.guard = RuntimePolicyGuard(
            allowed_tools=set(allowed_tools) if allowed_tools else None,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )

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
            input_prompt = dialog_history
            if self.prefill_action_token and current_step == 0:
                input_prompt = f"{dialog_history}\n<|START_ACTION|>"
            model_output = policy_fn(input_prompt)

            # When prefilling <|START_ACTION|>, reconstruct full output if not already present
            effective_output = model_output
            if self.prefill_action_token and current_step == 0:
                if not effective_output.lstrip().startswith("<|START_ACTION|>"):
                    effective_output = f"<|START_ACTION|>{effective_output}"

            # 3. Output parsing
            thought, action, answer, parse_err = parse_agent_output(
                effective_output,
                allowed_tools=self.engine.allowed_tools,
            )

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
                # 4 & 5. Pre-execution validation & Sandboxed execution
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
                    dialog_history += f"\n{effective_output}\nObservation Error: {obs.error_message}"
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
                    dialog_history += f"\n{effective_output}\nObservation: {sanitized_res}"

            current_step += 1

        if final_answer is None and current_step >= self.guard.max_steps:
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

        # 6. Environment-owned outcome verification
        is_success = verify_task_outcome(task, final_answer)
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
            run_id=self.run_id,
            environment_version=self.environment_version,
            metadata={
                "max_steps": self.guard.max_steps,
                "timeout_seconds": self.guard.timeout_seconds,
                "allowed_tools": sorted(list(self.engine.allowed_tools)),
                "prefill_action_token": self.prefill_action_token,
            },
        )

    def _evaluate_success(self, final_answer: Optional[str], task: TaskSpec) -> bool:
        """Backwards-compatible delegate to environment-owned verify_task_outcome."""
        return verify_task_outcome(task, final_answer)


"""Dynamic trajectory fuzzer, adversarial mock policies, and failure suite generator."""

from __future__ import annotations

import copy
import json
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from src.data.schemas import StepRecord, ToolCallAction, TrajectoryRecord
from src.environment.task_generator import TaskSpec
from src.environment.tools import TOOL_CONTRACTS
from src.evaluation.taxonomy import FailureCategory, diagnose_failure


class TrajectoryFuzzer:
    """Programmatically mutates valid trajectories into deterministic failure cases.

    Grounded in Gorilla BFCL and property-based perturbation testing:
    - Tool selection: injects non-existent, disallowed, or extraneous tools.
    - Argument schema: injects extra forbidden keys, type mismatches, or semantic bound errors.
    - Dependency order: inverts topological sequence of prerequisite tools.
    - Looping & stall: repeats identical calls 3+ times or injects alternating cycles.
    - Early termination: truncates execution to 0 tool steps or partial tool subsets.
    - Tool recovery: simulates unadapted repeated calls or premature aborts after error observations.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def mutate_tool_selection(
        self,
        trajectory: TrajectoryRecord,
        mode: Literal["unknown_tool", "disallowed_tool", "extraneous_tool"] = "unknown_tool",
    ) -> TrajectoryRecord:
        """Inject tool selection failure into trajectory."""
        mutated = trajectory.model_copy(deep=True)
        mutated.is_success = False

        if not mutated.steps:
            mutated.steps.append(
                StepRecord(
                    step_index=0,
                    action=ToolCallAction(tool_name="unregistered_tool_999", arguments={}),
                    error="TOOL_NOT_FOUND: Tool 'unregistered_tool_999' is not recognized or authorized.",
                )
            )
            return mutated

        if mode == "unknown_tool":
            hallucinated_name = f"unregistered_tool_{self.rng.randint(100, 999)}"
            mutated.steps[0].action = ToolCallAction(tool_name=hallucinated_name, arguments={})
            mutated.steps[0].error = f"TOOL_NOT_FOUND: Tool '{hallucinated_name}' is not recognized or authorized."
            mutated.steps[0].observation = None
        elif mode == "disallowed_tool":
            mutated.steps[0].action = ToolCallAction(tool_name="admin_drop_tables", arguments={})
            mutated.steps[0].error = "DISALLOWED_TOOL: Tool 'admin_drop_tables' is prohibited by current policy guard."
            mutated.steps[0].observation = None
        elif mode == "extraneous_tool":
            mutated.steps[0].action = ToolCallAction(
                tool_name="verify_constraint",
                arguments={"actual": 1.0, "expected": 2.0, "comparison": "le"},
            )
            mutated.steps[0].observation = False
            mutated.steps[0].error = None

        return mutated

    def mutate_argument_schema(
        self,
        trajectory: TrajectoryRecord,
        mode: Literal["extra_forbidden", "type_mismatch", "missing_required", "semantic_bound"] = "extra_forbidden",
    ) -> TrajectoryRecord:
        """Inject argument schema or semantic bound failure into the first tool call."""
        mutated = trajectory.model_copy(deep=True)
        mutated.is_success = False

        if not mutated.steps or mutated.steps[0].action is None:
            mutated.steps = [
                StepRecord(
                    step_index=0,
                    action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors", "__unauthorized_extra_arg__": 1}),
                    error="SCHEMA_VALIDATION_ERROR: Extra inputs are not permitted [extra_forbidden]",
                )
            ]
            return mutated

        step = mutated.steps[0]
        step.observation = None

        if mode == "extra_forbidden":
            step.action.arguments["__unauthorized_extra_arg__"] = "invalid_payload"
            step.error = "SCHEMA_VALIDATION_ERROR: Extra inputs are not permitted [extra_forbidden]"
        elif mode == "type_mismatch":
            step.action.arguments["table"] = 12345
            step.error = "SCHEMA_VALIDATION_ERROR: Input should be a valid string [string_type]"
        elif mode == "missing_required":
            step.action.arguments = {}
            step.error = "SCHEMA_VALIDATION_ERROR: Field required: 'table'"
        elif mode == "semantic_bound":
            step.action = ToolCallAction(tool_name="compute_aggregate", arguments={"values": [1.0], "operation": "bad_op"})
            step.error = "SEMANTIC_BOUND_ERROR: Operation 'bad_op' not supported."

        return mutated

    def mutate_dependency_order(self, trajectory: TrajectoryRecord) -> TrajectoryRecord:
        """Invert execution order of multi-step tool calls."""
        mutated = trajectory.model_copy(deep=True)
        mutated.is_success = False
        tool_indices = [i for i, s in enumerate(mutated.steps) if s.action is not None]
        if len(tool_indices) >= 2:
            i, j = tool_indices[0], tool_indices[1]
            mutated.steps[i].action, mutated.steps[j].action = mutated.steps[j].action, mutated.steps[i].action
            mutated.steps[i].observation, mutated.steps[j].observation = mutated.steps[j].observation, mutated.steps[i].observation
            mutated.steps[i].error, mutated.steps[j].error = None, None
        return mutated

    def mutate_loop(
        self,
        trajectory: TrajectoryRecord,
        mode: Literal["identical_3x", "alternating_cycle"] = "identical_3x",
    ) -> TrajectoryRecord:
        """Inject identical or cyclic loops into trajectory."""
        mutated = trajectory.model_copy(deep=True)
        mutated.is_success = False

        if mode == "identical_3x":
            base_action = mutated.steps[0].action if (mutated.steps and mutated.steps[0].action) else ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors"})
            mutated.steps = [
                StepRecord(
                    step_index=i,
                    action=base_action.model_copy(deep=True),
                    observation=[{"vendor_id": "VND-100"}],
                )
                for i in range(3)
            ]
        elif mode == "alternating_cycle":
            step_a = StepRecord(
                step_index=0,
                action=ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors"}),
                observation=[{"vendor_id": "VND-100"}],
            )
            step_b = StepRecord(
                step_index=1,
                action=ToolCallAction(tool_name="compute_aggregate", arguments={"values": [1.0], "operation": "sum"}),
                observation=1.0,
            )
            mutated.steps = [
                step_a.model_copy(update={"step_index": 0}),
                step_b.model_copy(update={"step_index": 1}),
                step_a.model_copy(update={"step_index": 2}),
                step_b.model_copy(update={"step_index": 3}),
            ]

        mutated.total_steps = len(mutated.steps)
        return mutated

    def mutate_early_termination(self, trajectory: TrajectoryRecord) -> TrajectoryRecord:
        """Inject premature halt with 0 tool calls."""
        mutated = trajectory.model_copy(deep=True)
        mutated.steps = []
        mutated.final_output = "I have prematurely halted without calling tools."
        mutated.is_success = False
        mutated.total_steps = 0
        return mutated

    def mutate_recovery_failure(
        self,
        trajectory: TrajectoryRecord,
        mode: Literal["repeat_error", "abort_after_error"] = "repeat_error",
    ) -> TrajectoryRecord:
        """Inject unadapted repeat or premature abort following error feedback."""
        mutated = trajectory.model_copy(deep=True)
        mutated.is_success = False
        failing_call = ToolCallAction(tool_name="lookup_records", arguments={"table": "vendors", "bad_arg": 1})

        if mode == "repeat_error":
            mutated.steps = [
                StepRecord(
                    step_index=0,
                    action=failing_call,
                    error="SCHEMA_VALIDATION_ERROR: Extra inputs are not permitted",
                ),
                StepRecord(
                    step_index=1,
                    action=failing_call,
                    error="SCHEMA_VALIDATION_ERROR: Extra inputs are not permitted",
                ),
            ]
            mutated.final_output = None
        elif mode == "abort_after_error":
            mutated.steps = [
                StepRecord(
                    step_index=0,
                    action=failing_call,
                    error="SCHEMA_VALIDATION_ERROR: Extra inputs are not permitted",
                ),
                StepRecord(
                    step_index=1,
                    thought="Error encountered, giving up.",
                    elapsed_ms=10.0,
                ),
            ]
            mutated.final_output = "Aborted due to database error."

        mutated.total_steps = len(mutated.steps)
        return mutated


class AdversarialPolicyFactory:
    """Produces executable policy callables that induce specific failure modes live in the orchestrator.

    Implements Skalse et al. (2022) adversarial degeneracy probes:
    - NoOpPolicy: Evaluates early termination and anti-laziness rewards.
    - SpamLoopPolicy: Evaluates looping detection and anti-spam penalties.
    - StubbornErrorPolicy: Evaluates recovery adaptation and corrective feedback loops.
    - HallucinationPolicy: Evaluates tool allowlists and unknown tool security boundaries.
    """

    @staticmethod
    def create_noop_policy() -> Callable[[str], str]:
        """Terminates immediately without tool actions (EarlyTerminationFailure)."""
        def policy(dialog: str) -> str:
            return "Task completed immediately without calling any tools."
        return policy

    @staticmethod
    def create_spam_loop_policy(tool_name: str = "lookup_records") -> Callable[[str], str]:
        """Repeats identical call on every step (LoopingAndStallFailure)."""
        def policy(dialog: str) -> str:
            call_json = json.dumps({"name": tool_name, "arguments": {"table": "vendors"}})
            return f"<|START_ACTION|>{call_json}<|END_ACTION|>"
        return policy

    @staticmethod
    def create_stubborn_error_policy() -> Callable[[str], str]:
        """Never corrects invalid arguments despite error observations (ToolRecoveryFailure)."""
        def policy(dialog: str) -> str:
            call_json = json.dumps({"name": "lookup_records", "arguments": {"table": "inventory_items", "bad_arg": 123}})
            return f"<|START_ACTION|>{call_json}<|END_ACTION|>"
        return policy

    @staticmethod
    def create_hallucination_policy() -> Callable[[str], str]:
        """Invokes distinct non-existent tools across turns (ToolSelectionFailure)."""
        call_count = 0

        def policy(dialog: str) -> str:
            nonlocal call_count
            call_count += 1
            call_json = json.dumps({"name": f"unregistered_tool_{call_count}", "arguments": {}})
            return f"<|START_ACTION|>{call_json}<|END_ACTION|>"

        return policy


class DynamicFailureSuite:
    """Generates a complete, balanced suite of failure trajectories on demand from any task split."""

    def __init__(self, seed: int = 42) -> None:
        self.fuzzer = TrajectoryFuzzer(seed=seed)

    def generate_suite(
        self,
        base_trajectory: TrajectoryRecord,
        task: Optional[TaskSpec] = None,
    ) -> List[Tuple[str, TrajectoryRecord, FailureCategory]]:
        """Generate a balanced test suite covering all 6 categories from a single base trajectory."""
        suite: List[Tuple[str, TrajectoryRecord, FailureCategory]] = [
            (
                "Fuzzed: Unknown Tool Selection",
                self.fuzzer.mutate_tool_selection(base_trajectory, mode="unknown_tool"),
                FailureCategory.TOOL_SELECTION_FAILURE,
            ),
            (
                "Fuzzed: Disallowed Tool Selection",
                self.fuzzer.mutate_tool_selection(base_trajectory, mode="disallowed_tool"),
                FailureCategory.TOOL_SELECTION_FAILURE,
            ),
            (
                "Fuzzed: Argument Schema (Extra Forbidden)",
                self.fuzzer.mutate_argument_schema(base_trajectory, mode="extra_forbidden"),
                FailureCategory.ARGUMENT_SCHEMA_FAILURE,
            ),
            (
                "Fuzzed: Argument Schema (Semantic Bound)",
                self.fuzzer.mutate_argument_schema(base_trajectory, mode="semantic_bound"),
                FailureCategory.ARGUMENT_SCHEMA_FAILURE,
            ),
            (
                "Fuzzed: Dependency Order Inversion",
                self.fuzzer.mutate_dependency_order(base_trajectory),
                FailureCategory.DEPENDENCY_ORDER_FAILURE,
            ),
            (
                "Fuzzed: Looping (Identical 3x)",
                self.fuzzer.mutate_loop(base_trajectory, mode="identical_3x"),
                FailureCategory.LOOPING_AND_STALL_FAILURE,
            ),
            (
                "Fuzzed: Looping (Alternating Cycle)",
                self.fuzzer.mutate_loop(base_trajectory, mode="alternating_cycle"),
                FailureCategory.LOOPING_AND_STALL_FAILURE,
            ),
            (
                "Fuzzed: Early Termination (0 Steps)",
                self.fuzzer.mutate_early_termination(base_trajectory),
                FailureCategory.EARLY_TERMINATION_FAILURE,
            ),
            (
                "Fuzzed: Tool Recovery (Repeat Error)",
                self.fuzzer.mutate_recovery_failure(base_trajectory, mode="repeat_error"),
                FailureCategory.TOOL_RECOVERY_FAILURE,
            ),
            (
                "Fuzzed: Tool Recovery (Abort After Error)",
                self.fuzzer.mutate_recovery_failure(base_trajectory, mode="abort_after_error"),
                FailureCategory.TOOL_RECOVERY_FAILURE,
            ),
        ]
        return suite

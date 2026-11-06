"""Baseline Evaluation CLI Runner for Lexsi Track 3 Agent (M1.P5 / P2.2).

Executes untuned base-model evaluation across the held-out validation split,
capturing replayable redacted traces, calculating AuditKIT M1-M7 metrics with 95%
bootstrap confidence intervals, clustering failures with the 6-class taxonomy,
and archiving standardized artifacts under results/runs/<run_id>/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure writable matplotlib cache directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.redaction import redact_data
from src.data.schemas import TrajectoryRecord
from src.environment.task_generator import TaskSpec
from src.evaluation.metrics import evaluate_benchmark_run
from src.evaluation.taxonomy import FailureCategory, diagnose_failure
from src.runtime.gateway import ModelGateway
from src.runtime.orchestrator import EpisodeOrchestrator
from src.utils.compatibility import AuditKitRunResult, compute_run_fingerprint
from src.utils.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baseline_eval")


def load_tasks_from_manifest(manifest_path: Path) -> List[TaskSpec]:
    """Load and validate task instances from a split manifest file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks_raw = data.get("tasks", [])
    tasks = []
    for item in tasks_raw:
        task = TaskSpec(
            task_id=item["task_id"],
            split=item["split"],
            template_id=item["template_id"],
            user_query=item["user_query"],
            expected_tools=item["expected_tools"],
            oracle_answer=item.get("oracle_answer"),
            oracle_verdict=item.get("oracle_verdict"),
            target_entities=item.get("target_entities", {}),
        )
        tasks.append(task)
    return tasks


def create_base_model_policy(
    gateway: ModelGateway,
    model_id: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Callable[[str], str]:
    """Create a synchronous policy callable for EpisodeOrchestrator querying ModelGateway."""
    def policy(dialog_history: str) -> str:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            resp = loop.run_until_complete(
                gateway.generate(
                    messages=[{"role": "user", "content": dialog_history}],
                    model=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            return resp.content or ""
        except Exception as e:
            logger.warning("Gateway generation failed: %s; falling back to empty answer", e)
            return "Unable to complete action due to API error."
    return policy


def run_baseline_evaluation(
    split: str = "val",
    model_id: str = "cohere/command-r7b-12-2024",
    run_id: Optional[str] = None,
    smoke: bool = False,
    seed: int = 42,
    max_steps: int = 8,
    timeout_seconds: float = 30.0,
    allow_locked_test: bool = False,
    output_base_dir: Optional[Path] = None,
) -> AuditKitRunResult:
    """Execute complete baseline evaluation and persist standardized run artifacts."""
    # 1. Split safety guard: protect locked test set
    if split == "test" and not allow_locked_test:
        raise PermissionError(
            "Evaluation on the locked 'test' split is forbidden during baseline calibration (Gate G1). "
            "The test split is permanently sequestered for final Gate G4 candidate evaluation. "
            "Use --split val for baseline evaluation."
        )

    # 2. Setup run directories and identifier
    run_id = run_id or (f"smoke_eval_{split}_001" if smoke else f"baseline_{split}_001")
    output_dir = (output_base_dir or (ROOT / "results" / "runs")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "configs" / "manifests" / f"task_manifest_{split}.json"
    all_tasks = load_tasks_from_manifest(manifest_path)

    if smoke:
        tasks = all_tasks[:2]
        logger.info("[SMOKE RUN] Evaluating bounded 2-task subset from split '%s' (%s)", split, [t.task_id for t in tasks])
    else:
        tasks = all_tasks
        logger.info("[FULL BASELINE] Evaluating complete %d tasks from split '%s'", len(tasks), split)

    # 3. Initialize Orchestrator and Gateway
    orchestrator = EpisodeOrchestrator(
        seed=seed,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        use_database=True,
    )
    gateway = ModelGateway()
    policy = create_base_model_policy(gateway, model_id=model_id, temperature=0.0)

    # 4. Execute episodes and collect trajectories
    start_time = time.time()
    trajectories: List[TrajectoryRecord] = []
    failure_records: List[Dict[str, Any]] = []

    print("=" * 80)
    print(f"Starting Lexsi Track 3 Evaluation Run: {run_id}")
    print(f"Model ID: {model_id} | Split: {split} | Smoke: {smoke} | Tasks: {len(tasks)}")
    print("=" * 80)

    for idx, task in enumerate(tasks, start=1):
        task_start = time.perf_counter()
        logger.info("[%d/%d] Running Episode on Task: %s", idx, len(tasks), task.task_id)
        traj = orchestrator.run_episode(task, policy)
        traj_elapsed = (time.perf_counter() - task_start) * 1000.0

        # Redact any private or secret data in trajectory
        redacted_traj_dict = redact_data(traj.model_dump())
        redacted_traj = TrajectoryRecord.model_validate(redacted_traj_dict)
        trajectories.append(redacted_traj)

        # Diagnose failures
        if not redacted_traj.is_success:
            diagnosis = diagnose_failure(redacted_traj, task)
            failure_records.append({
                "task_id": task.task_id,
                "category": diagnosis.category.value if diagnosis.category else "Unclassified",
                "confidence": diagnosis.confidence,
                "evidence": diagnosis.evidence,
                "root_cause_step": diagnosis.root_cause_step,
            })
            cat_label = diagnosis.category.value if diagnosis.category else "None"
            logger.info("  -> Verdict: FAILED (%s, %.1fms) | Diagnosis: %s", "Outcome Mismatch", traj_elapsed, cat_label)
        else:
            logger.info("  -> Verdict: SUCCESS (%.1fms, %d steps)", traj_elapsed, redacted_traj.total_steps)

    total_duration = time.time() - start_time
    logger.info("All %d episodes completed in %.2fs", len(tasks), total_duration)

    # 5. Compute AuditKIT Metrics (M1-M7)
    run_result = evaluate_benchmark_run(
        trajectories=trajectories,
        tasks=tasks,
        model_id=model_id,
        run_id=run_id,
        split=split,
    )

    # 6. Save Artifacts to results/runs/<run_id>/
    artifact_paths = run_result.save(output_dir)

    # Save streaming traces.jsonl
    traces_path = output_dir / "traces.jsonl"
    with open(traces_path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t.model_dump(), default=str) + "\n")
    artifact_paths["traces"] = str(traces_path.resolve())

    # Save run_manifest.json with provenance fingerprint
    manifest_hash = "manual_manifest"
    if manifest_path.exists():
        import hashlib
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    run_manifest = {
        "run_id": run_id,
        "owner": "Lexsi Track 3 Engineering & Safety Team",
        "purpose": "Milestone 1 Base-Model Baseline (E0) Evaluation and Failure Diagnosis",
        "status": "COMPLETED",
        "model_id": model_id,
        "model_metadata": {
            "repository": model_id,
            "revision": "12-2024",
            "tokenizer_chat_template": "command-r",
            "adapter_loaded": None,
            "decoding_parameters": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 512,
                "seed": seed,
            },
        },
        "hardware_metadata": {
            "platform": "Darwin (Apple Silicon M5, 16GB unified RAM)",
            "python_version": "3.11.9",
            "evaluator_mode": "local-adapter-1.0.0",
        },
        "dataset_provenance": {
            "benchmark_dataset": "procurement_inventory_benchmark",
            "benchmark_type": "Deterministic synthetic closed-world relational task generator",
            "source_fixture": "src/environment/mock_data.py",
            "split_manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "license": "not separately asserted (synthetic closed-world benchmark)",
            "auxiliary_database_backend": "Turso Cloud libSQL (AWS ap-south-1) / local SQLite",
        },
        "runtime_guards_and_limits": {
            "max_steps_per_episode": max_steps,
            "per_call_tool_timeout_seconds": 5.0,
            "episode_deadline_seconds": timeout_seconds,
            "argument_validation": "Pydantic extra='forbid'",
            "outcome_verifier_version": "1.0.0 (environment-owned predicate)",
        },
        "split": split,
        "is_smoke": smoke,
        "task_count": len(tasks),
        "task_ids": [t.task_id for t in tasks],
        "fingerprint": run_result.sha256_fingerprint,
        "start_timestamp": start_time,
        "completion_timestamp": time.time(),
        "total_duration_seconds": round(total_duration, 2),
        "environment_version": "0.1.0",
        "failures_diagnosed": failure_records,
    }
    manifest_file = output_dir / "run_manifest.json"
    manifest_file.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    artifact_paths["manifest"] = str(manifest_file.resolve())

    # 7. Print rich Markdown Summary Table
    print("\n" + "=" * 80)
    print(run_result.to_markdown_table())
    print("=" * 80)

    print("\n[Archived Run Artifacts in %s]" % output_dir)
    for name, path in artifact_paths.items():
        print(f"  • {name.upper()}: {path}")

    return run_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Lexsi Track 3 Baseline Evaluation Runner")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "test"], help="Benchmark split")
    parser.add_argument("--model-id", type=str, default="cohere/command-r7b-12-2024", help="Model identifier")
    parser.add_argument("--run-id", type=str, default=None, help="Unique run identifier")
    parser.add_argument("--smoke", action="store_true", help="Run bounded 2-task smoke check first")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic evaluation seed")
    parser.add_argument("--max-steps", type=int, default=8, help="Max step budget per episode")
    parser.add_argument("--timeout", type=float, default=30.0, help="Max timeout seconds per episode")
    parser.add_argument("--allow-locked-test", action="store_true", help="Explicitly permit evaluation on test split")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    run_baseline_evaluation(
        split=args.split,
        model_id=args.model_id,
        run_id=args.run_id,
        smoke=args.smoke,
        seed=args.seed,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout,
        allow_locked_test=args.allow_locked_test,
        output_base_dir=out_dir,
    )


if __name__ == "__main__":
    main()


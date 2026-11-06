"""Machine-readable dataset manifests and deterministic provenance tracking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict

from src.environment.task_generator import TaskSpec, TaskSplitConfig, load_split_config


class TaskManifestEntry(BaseModel):
    """Manifest descriptor for a single benchmark task instance."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    split: str
    template_id: str
    target_entities: Dict[str, Any]
    expected_tools: List[str]
    provenance_hash: str
    is_locked: bool = False
    oracle_answer: float
    oracle_verdict: Optional[bool] = None
    user_query: str


class SplitManifest(BaseModel):
    """Manifest describing an isolated dataset split."""

    model_config = ConfigDict(extra="forbid")

    split_name: str
    count: int
    is_locked: bool
    status: Optional[str] = None
    split_content_hash: str
    tasks: List[TaskManifestEntry]


class DatasetManifest(BaseModel):
    """Canonical machine-readable dataset manifest spanning all benchmark splits."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: str = "1.0.0"
    dataset_name: str = "procurement_inventory_benchmark"
    domain: str
    generator_version: str = "0.1.0"
    seed: int
    config_hash: str
    license: str
    provenance: str
    total_tasks: int
    splits: Dict[str, SplitManifest]


def compute_config_hash(config_path: Optional[Path] = None) -> str:
    """Generate SHA-256 digest of task_splits.yaml for manifest provenance."""
    path = config_path or (Path(__file__).resolve().parent.parent.parent / "configs" / "task_splits.yaml")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_split_hash(tasks: List[TaskManifestEntry]) -> str:
    """Generate deterministic SHA-256 digest over sorted task provenance hashes."""
    concatenated = "".join(t.provenance_hash for t in sorted(tasks, key=lambda x: x.task_id))
    return hashlib.sha256(concatenated.encode("utf-8")).hexdigest()


def generate_task_manifests(
    splits: Dict[str, List[TaskSpec]],
    config: Optional[TaskSplitConfig] = None,
) -> DatasetManifest:
    """Produce machine-readable manifests with complete provenance for all splits."""
    cfg = config or load_split_config()
    cfg_hash = compute_config_hash()

    split_manifests: Dict[str, SplitManifest] = {}
    total_tasks = 0

    for split_name in ("train", "val", "test"):
        tasks = splits.get(split_name, [])
        is_locked = (split_name == "test")
        spec = getattr(cfg, split_name)

        entries = [
            TaskManifestEntry(
                task_id=t.task_id,
                split=t.split,
                template_id=t.template_id,
                target_entities=t.target_entities,
                expected_tools=t.expected_tools,
                provenance_hash=t.provenance_hash or "",
                is_locked=is_locked,
                oracle_answer=t.oracle_answer,
                oracle_verdict=t.oracle_verdict,
                user_query=t.user_query,
            )
            for t in tasks
        ]

        split_hash = compute_split_hash(entries)
        split_manifests[split_name] = SplitManifest(
            split_name=split_name,
            count=len(entries),
            is_locked=is_locked,
            status=spec.status,
            split_content_hash=split_hash,
            tasks=entries,
        )
        total_tasks += len(entries)

    return DatasetManifest(
        manifest_version="1.0.0",
        dataset_name="procurement_inventory_benchmark",
        domain=cfg.metadata.domain,
        generator_version="0.1.0",
        seed=cfg.metadata.seed,
        config_hash=cfg_hash,
        license=cfg.metadata.license,
        provenance=cfg.metadata.provenance,
        total_tasks=total_tasks,
        splits=split_manifests,
    )


def save_manifests(
    manifest: DatasetManifest,
    output_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """Persist machine-readable manifests to disk in canonical JSON format."""
    if output_dir is not None:
        target_dir = Path(output_dir)
    else:
        target_dir = Path(__file__).resolve().parent.parent.parent / "configs" / "manifests"
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[Path] = []

    # 1. Save unified dataset manifest
    unified_path = target_dir / "dataset_manifest.json"
    with open(unified_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
    saved_files.append(unified_path)

    # 2. Save individual split manifests
    for split_name, split_manifest in manifest.splits.items():
        split_path = target_dir / f"task_manifest_{split_name}.json"
        with open(split_path, "w", encoding="utf-8") as f:
            f.write(split_manifest.model_dump_json(indent=2))
        saved_files.append(split_path)

    return saved_files

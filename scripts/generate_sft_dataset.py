#!/usr/bin/env python3
"""CLI utility to generate, validate, and export the M2 SFT training dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sft_builder import build_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lexsi Track 3 M2 SFT Training Dataset")
    parser.add_argument("--train-manifest", type=str, default=str(ROOT / "configs" / "manifests" / "task_manifest_train.json"))
    parser.add_argument("--val-manifest", type=str, default=str(ROOT / "configs" / "manifests" / "task_manifest_val.json"))
    parser.add_argument("--test-manifest", type=str, default=str(ROOT / "configs" / "manifests" / "task_manifest_test.json"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "data" / "sft"))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_path = Path(args.train_manifest)
    val_path = Path(args.val_manifest)
    test_path = Path(args.test_manifest)
    out_dir = Path(args.output_dir)

    print("=" * 80)
    print("Generating M2 SFT Training Dataset from Train Manifest")
    print(f"Train Manifest: {train_path}")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)

    result = build_sft_dataset(
        train_manifest_path=train_path,
        val_manifest_path=val_path,
        test_manifest_path=test_path,
        output_dir=out_dir,
        seed=args.seed,
    )

    manifest = result["manifest"]
    stats = manifest["stats"]
    leakage = manifest["leakage_audit"]

    print("\n[Generation Complete]")
    print(f"  • Total Tasks: {stats['total_tasks_processed']}")
    print(f"  • Accepted Trajectories: {stats['accepted_trajectories_count']}")
    print(f"  • Unique Trajectories: {stats['unique_trajectories_count']}")
    print(f"  • Replicated Tasks: {stats['duplicate_count']}")
    print(f"  • Total Executed Calls: {stats['total_tool_calls']}")
    print(f"  • Dataset SHA-256 Digest: {manifest['dataset_sha256']}")
    print(f"  • Split Isolation Verdict: {'PASSED' if leakage['is_isolation_verified'] else 'FAILED'}")

    print("\n[Generated Artifacts]")
    for name, path in result["files"].items():
        print(f"  • {name.upper()}: {path}")


if __name__ == "__main__":
    main()


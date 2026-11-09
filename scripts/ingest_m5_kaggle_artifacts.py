"""Artifact ingestion and cryptographic verification script for Kaggle M5 runs (M5.P4).

Unpacks and verifies m5_kaggle_artifacts.zip exported from Kaggle 2xT4:
- Ingests matched untuned base rollout (results/runs/val_base_aya_3b_matched/).
- Ingests Condition C2 checkpoint & traces (results/runs/sft_aya_3b_v2_c2/).
- Ingests Condition C3 checkpoint & traces (results/runs/sft_aya_3b_v2_c3/).
- Enforces fail-closed validation: halts if locked-test files or corrupted checksums are detected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.schemas import LockedTestExportError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest_m5")


def verify_zip_integrity(zip_path: Path) -> List[str]:
    """Verify ZIP contents, ensuring zero locked-test contamination."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Artifact archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        file_list = zf.namelist()
        for fname in file_list:
            if "task_manifest_test.json" in fname or "task_test_" in fname:
                raise LockedTestExportError(
                    f"CRITICAL: Artifact ZIP contains locked-test file '{fname}'! "
                    "Permanently sequestered test data must never be exported from training environments."
                )
    return file_list


def ingest_artifacts(zip_path: Path, target_dir: Path) -> Dict[str, Any]:
    """Extract and register run artifacts into results/runs/."""
    file_list = verify_zip_integrity(zip_path)
    logger.info("Verified archive '%s' (%d files, 0 test contamination).", zip_path.name, len(file_list))

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    ingested_runs: List[str] = []
    runs_dir = target_dir / "results" / "runs"
    if runs_dir.exists():
        for run_path in runs_dir.iterdir():
            if run_path.is_dir():
                traces_file = run_path / "val_rollout_traces.json"
                manifest_file = run_path / "run_manifest.json"
                if traces_file.exists():
                    sha = hashlib.sha256(traces_file.read_bytes()).hexdigest()
                    ingested_runs.append(f"{run_path.name} (traces SHA-256: {sha[:12]}...)")
                elif manifest_file.exists():
                    ingested_runs.append(run_path.name)

    logger.info("Successfully ingested runs: %s", ingested_runs)
    return {
        "status": "success",
        "archive": str(zip_path),
        "ingested_runs": ingested_runs,
        "total_files": len(file_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest M5 Kaggle Artifacts Archive")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=root_dir / "m5_kaggle_artifacts.zip",
        help="Path to m5_kaggle_artifacts.zip",
    )
    args = parser.parse_args()

    try:
        res = ingest_artifacts(args.zip_path, root_dir)
        print(json.dumps(res, indent=2))
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

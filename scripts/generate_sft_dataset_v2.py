"""CLI script to build the expanded SFT training dataset procurement_sft_train_v2 (M5.P2).

Executes deterministic oracle trajectory generation, strict inclusion filtering, deduplication,
and split isolation audits without touching the sequestered locked test manifest.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.sft_builder_v2 import build_sft_dataset_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("generate_sft_v2")


def main() -> None:
    output_dir = root_dir / "data" / "sft"
    logger.info("Starting M5.P2 dataset generation for procurement_sft_train_v2...")

    res = build_sft_dataset_v2(output_dir=output_dir, target_count=220, seed=42)

    manifest = res["manifest"]
    logger.info("Successfully generated procurement_sft_train_v2:")
    logger.info("  Sample count: %d", manifest["sample_count"])
    logger.info("  Unique count: %d", manifest["deduped_sample_count"])
    logger.info("  Deduplication ratio: %.1f%%", manifest["deduplication_ratio"] * 100)
    logger.info("  Dataset SHA-256: %s", manifest["dataset_sha256"])
    logger.info("  Isolation verified: %s", manifest["isolation_verified"])
    logger.info("Outputs:")
    for k, v in res["files"].items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()

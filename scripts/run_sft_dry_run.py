#!/usr/bin/env python3
"""Run SFT Stage 1 Dry Run Smoke Step (M2.P4).

Validates:
1. Minimal 2-example subset from train split.
2. 2 forward + backward steps verifying finite loss and gradient flow on LoRA parameters.
3. Zero gradients on frozen base weights.
4. Persists run manifest and training loss CSV in results/runs/sft_dry_run_001/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import SFTExecutionHarness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sft_dry_run")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SFT Dry Run Smoke Step")
    parser.add_argument("--config", default="configs/training_sft.yaml", help="Path to SFT training YAML config")
    parser.add_argument("--output-dir", default="results/runs/sft_dry_run_001", help="Output directory for dry run artifacts")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Configuration file not found: {cfg_path}")
        return 1

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    harness = SFTExecutionHarness(config)

    try:
        manifest = harness.run_sft_dry_run(output_dir=args.output_dir, sample_count=2)
        logger.info(f"✓ SFT Dry Run SUCCESS: Verdict={manifest.get('verification_verdict')}")
        return 0
    except Exception as e:
        logger.error(f"✗ SFT Dry Run FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


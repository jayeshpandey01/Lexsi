#!/usr/bin/env python3
"""Run SFT Stage 2 Bounded Pilot (M2.P4).

Executes bounded Supervised Fine-Tuning across the verified 20-trajectory corpus:
1. Pinned seed 42, learning rate 2e-5, max_steps 30.
2. Token-level assistant loss masking with ignore_index=-100.
3. LoRA adapter weight updates with gradient clipping.
4. Generates qualitative sample actions and verifies action delimiters.
5. Saves immutable adapter checkpoint and run manifest in results/runs/sft_pilot_001/.
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
logger = logging.getLogger("sft_pilot")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SFT Bounded Pilot")
    parser.add_argument("--config", default="configs/training_sft.yaml", help="Path to SFT training YAML config")
    parser.add_argument("--output-dir", default="results/runs/sft_pilot_001", help="Output directory for pilot artifacts")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Configuration file not found: {cfg_path}")
        return 1

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    harness = SFTExecutionHarness(config)

    try:
        manifest = harness.run_bounded_pilot(output_dir=args.output_dir)
        logger.info(f"✓ SFT Bounded Pilot SUCCESS: Initial loss={manifest['initial_loss']}, Final loss={manifest['final_loss']}, Checkpoint={manifest['checkpoint_path']}")
        return 0
    except Exception as e:
        logger.error(f"✗ SFT Bounded Pilot FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


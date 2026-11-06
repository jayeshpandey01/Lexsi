"""CLI entrypoint to execute the offline RL plumbing smoke check."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path for standalone script execution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.rl_spike import RLPlumbingRunner, RLSpikeConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_rl_plumbing_smoke")


def main() -> None:
    config = RLSpikeConfig(
        run_id="rl_plumbing_smoke_001",
        seed=42,
        num_steps=2,
        learning_rate=1e-4,
        max_runtime_seconds=60.0,
        output_dir="results/runs/rl_plumbing_smoke_001",
    )
    logger.info("Initializing RL Plumbing Runner (Offline Smoke Check)...")
    runner = RLPlumbingRunner(config)
    res = runner.execute_spike()
    logger.info(f"RL Plumbing Smoke Check Completed: {res}")


if __name__ == "__main__":
    main()


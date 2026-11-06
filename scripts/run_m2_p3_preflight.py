#!/usr/bin/env python3
"""Run AgentTune SFT Compatibility and Training-Readiness Preflight (M2.P3).

Validates:
1. Pinned model ID and architecture in configs/model_config.yaml.
2. ShareGPT and Two-Tier EventLog format integrity in data/sft/.
3. Command-R action delimiters and Pydantic argument schemas.
4. Assistant-only loss masking and zero observation training leakage.
5. Saves authoritative audit report in data/sft/m2_p3_compatibility_report.md.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.training.agenttune_adapter import AgentTunePreflightChecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m2_p3_preflight")


def main() -> int:
    logger.info("Initializing M2.P3 AgentTune Compatibility Preflight...")

    # 1. Load model config
    cfg_path = Path("configs/model_config.yaml")
    if cfg_path.exists():
        raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        target_model_name = raw_cfg.get("model", {}).get("name", "CohereLabs/tiny-aya-global")
        target_family = raw_cfg.get("model", {}).get("family", "Aya")
        chat_template = raw_cfg.get("model", {}).get("chat_template", "command-r")
        context_window = raw_cfg.get("model", {}).get("context_window", 8192)
    else:
        target_model_name = "CohereLabs/tiny-aya-global"
        target_family = "Aya"
        chat_template = "command-r"
        context_window = 8192

    logger.info(f"Target Training Model: {target_model_name} (Architecture: {target_family})")
    logger.info(f"Chat Template: {chat_template} | Context: {context_window}")

    # 2. Check dataset files
    sft_dir = Path("data/sft")
    sharegpt_path = sft_dir / "sft_train_sharegpt.json"
    eventlog_path = sft_dir / "sft_train_eventlog.json"
    report_path = sft_dir / "m2_p3_compatibility_report.md"

    if not sharegpt_path.exists():
        logger.error(f"Missing ShareGPT dataset at {sharegpt_path}. Run scripts/generate_sft_dataset.py first.")
        return 1

    if not eventlog_path.exists():
        logger.error(f"Missing EventLog dataset at {eventlog_path}. Run scripts/generate_sft_dataset.py first.")
        return 1

    # 3. Execute Preflight Checks
    checker = AgentTunePreflightChecker(target_model_id=target_model_name)

    logger.info(f"Validating ShareGPT dataset: {sharegpt_path}")
    sharegpt_res = checker.validate_sharegpt_dataset(sharegpt_path)
    logger.info(f"  Samples: {sharegpt_res.sample_count}")
    logger.info(f"  Tool calls: {sharegpt_res.tool_call_stats.valid_tool_calls}/{sharegpt_res.tool_call_stats.total_tool_calls_detected} valid")
    logger.info(f"  Supervision ratio: {sharegpt_res.masking_stats.supervision_ratio:.1%} ({sharegpt_res.masking_stats.supervised_tokens_count}/{sharegpt_res.masking_stats.total_tokens_estimated} tokens)")
    logger.info(f"  Masking compliance: {sharegpt_res.masking_stats.is_masking_compliant}")

    logger.info(f"Validating EventLog dataset: {eventlog_path}")
    eventlog_res = checker.validate_eventlog_dataset(eventlog_path)
    logger.info(f"  Episodes: {eventlog_res.sample_count}")
    logger.info(f"  Tool actions: {eventlog_res.tool_call_stats.valid_tool_calls}/{eventlog_res.tool_call_stats.total_tool_calls_detected} valid")

    # 4. Generate Markdown Report
    logger.info(f"Compiling M2.P3 report to {report_path}")
    report_text = checker.generate_m2_p3_markdown_report(
        sharegpt_result=sharegpt_res,
        eventlog_result=eventlog_res,
        output_path=report_path,
    )

    if sharegpt_res.is_compatible and eventlog_res.is_compatible:
        logger.info("✓ M2.P3 PREFLIGHT PASSED: Dataset and templates are 100% compatible with AgentTune.")
        return 0
    else:
        logger.error(f"✗ M2.P3 PREFLIGHT FAILED: Violations encountered:\nShareGPT: {sharegpt_res.violations}\nEventLog: {eventlog_res.violations}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

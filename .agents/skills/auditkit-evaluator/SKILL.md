---
name: auditkit-evaluator
description: Runs AuditKIT benchmarks, metrics, and failure diagnosis for tool-calling agents. Use for baselines, post-training evaluation, before/after comparisons, and failure analysis in the Lexsi Track 3 project.
---

# AuditKIT Evaluator

## Goal
Produce trustworthy, reproducible measurements of tool-call accuracy, argument validity, order correctness, task success, steps per task, and redundant-call rate.

## Instructions
1. Read the AuditKIT README and https://auditkit.lexsi.ai. Record available benchmarks, metrics, and export formats in `references/auditkit_notes.md`.
2. Always evaluate the base model first and store it as the baseline.
3. Use identical settings, data, and seeds for baseline and trained models.
4. Run multiple seeds when budget allows; report mean and spread.
5. Categorise failures (wrong tool, bad arguments, wrong order, loops, redundant calls, premature stop) with 3-5 concrete examples each.
6. Write results to `results/` as CSV/JSON plus a markdown table.

## Examples
- Input: "Compare base vs SFT+RL." Output: table of metrics, deltas, failure-category shifts, caveats.

## Constraints
- Never report numbers that were not produced by a logged run.
- Never evaluate on training data.
- State sample sizes and any limitations next to every claim.


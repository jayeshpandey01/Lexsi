---
name: agenttune-trainer
description: Runs and configures AgentTune post-training (SFT and RL) for reliable, efficient tool-calling agents. Use when the user wants to train, fine-tune, or post-train an agent, or change training configs for the Lexsi Track 3 project.
---

# AgentTune Trainer

## Goal
Post-train a small open model with AgentTune to improve tool selection, argument validity, call ordering, and step efficiency.

## Instructions
1. Read the AgentTune README and https://agenttune.lexsi.ai. Record supported models, algorithms, dataset formats, and CLI/API entry points in `references/agenttune_notes.md`.
2. Validate the dataset format against the docs before training; fail early on schema errors.
3. Stage A: SFT for format and syntax. Stage B: RL with verifiable rewards (see `reward-design-guard`).
4. Save every run's config, seed, library versions, and git commit in `results/runs/<run_id>/`.
5. Log metrics with Trackio or wandb. Save checkpoints at fixed intervals.
6. After each run, hand off to `auditkit-evaluator`.

## Examples
- Input: "Train the SFT stage on dataset X." Output: validated config, run command, expected outputs, checkpoint path.

## Constraints
- Do not invent flags or APIs; cite the doc or source file used.
- Do not train on evaluation data.
- Do not delete checkpoints without asking.
- Start with small smoke-test runs before full runs.


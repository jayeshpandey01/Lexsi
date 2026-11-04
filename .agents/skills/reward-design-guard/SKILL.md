---
name: reward-design-guard
description: Designs and audits RL reward functions for tool-calling agents and tests them for reward hacking. Use whenever creating, changing, or debugging a reward, judge, or penalty.
---

# Reward Design Guard

## Goal
Raise real task success without letting the agent game the reward.

## Instructions
1. Decompose the reward: format, tool selection, arguments, order, verified outcome, efficiency penalty.
2. Apply the efficiency penalty only to runs that succeed, otherwise "do nothing" wins.
3. Prefer verifiable outcome checks over LLM judges; if a judge is used, calibrate it on labelled examples.
4. Before full training, run a hacking audit: try degenerate policies (empty call, repeated call, copy expected output, skip tools) and confirm each scores lower than a correct policy.
5. During training, sample and read outputs at fixed intervals; watch reward vs real success diverging.
6. Compare outcome-only vs process rewards when time allows; document findings.

## Examples
- Input: "Add a step penalty." Output: reward code, unit tests with degenerate policies, expected score ordering.

## Constraints
- Never ship a reward without unit tests.
- Keep held-out evaluation independent of the reward function.
- Report any discovered hacking honestly.


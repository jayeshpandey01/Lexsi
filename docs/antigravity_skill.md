# Antigravity Skills Research and Master Prompt (Lexsi Track 3)

> **Document Status:** Authoritative Customization Guide & Master Prompt  
> **Target System:** Lexsi Track 3 — Reliable and Efficient Tool-Using Agent  
> **References:** [`.agents/skills/`](/Users/jayesh/Documents/Lexsi/.agents/skills/) · [`references/requirements.md`](/Users/jayesh/Documents/Lexsi/references/requirements.md) · [`docs/plan.md`](/Users/jayesh/Documents/Lexsi/docs/plan.md) · [`docs/M2.md`](/Users/jayesh/Documents/Lexsi/docs/M2.md)

---

## Part A. Research Summary

### A.1 How Skills Work in Google Antigravity

- A skill is an isolated capability **folder** containing a mandatory `SKILL.md` instruction file and optional support directories (`scripts/`, `references/`, `assets/`, `examples/`).
- `SKILL.md` requires YAML frontmatter (`name`, `description`) defining the tool descriptor, followed by a markdown body containing Goal, Instructions, Examples, and Constraints.
- **Discovery & Routing Architecture:**
  - The agent's prompt environment is provisioned with the catalogue of available skill `name` and `description` headers.
  - When user instructions match a specialized workflow or toolchain, the agent evaluates relevance against names and descriptions, then loads the complete `SKILL.md` body on demand or delegates execution to an autonomous subagent.
  - Skill routing is informed by both name identity and descriptive semantics; writing precise *"what it does"* and *"when to use it"* definitions ensures accurate invocation.
- **Precedence & Installation Scope:**
  - **Project Workspace Scope:** `<project-root>/.agents/skills/<name>/` (takes top priority and isolates project workflows).
  - **Global User Scope:** `~/.gemini/antigravity/skills/<name>/` (fallback across all workspaces).
- **Execution & Security Guards:**
  - Skills operate within the terminal sandbox by default. Always review scripts and tool commands before execution; third-party skills must never bypass access boundaries or execute arbitrary network calls without explicit authorization.

---

### A.2 Skill Ecosystem & Active Project Skills

The project workspace maintains six specialized skills under [`.agents/skills/`](/Users/jayesh/Documents/Lexsi/.agents/skills/):

| Installed Project Skill | Location | Status | Core Responsibility |
|:---|:---|:---:|:---|
| **`agenttune-trainer`** | `.agents/skills/agenttune-trainer/` | **Active** | AgentTune Two-Tier EventLog SFT and GRPO/DPO post-training configuration |
| **`auditkit-evaluator`** | `.agents/skills/auditkit-evaluator/` | **Active** | AuditKIT evaluation, paired bootstrap confidence intervals, and failure taxonomy |
| **`reward-design-guard`** | `.agents/skills/reward-design-guard/` | **Active** | Outcome reward specification, anti-reward-hacking audits, and efficiency penalties |
| **`hackathon-submission`** | `.agents/skills/hackathon-submission/` | **Active** | Submission packaging, README generation, demo script, and requirements verification |
| **`deep-research-verifier`** | `.agents/skills/deep-research-verifier/` | **Active** | Academic literature cross-verification and mathematical formulation audits |
| **`langfuse`** | `.agents/skills/langfuse/` | **Active** | Agent observability, trace telemetry, and runtime step monitoring |

#### Optional Ecosystem Recommendations (External HF/TRL Skills)
For remote cluster execution on Hugging Face Jobs, the following external skills can be added via `npx skills add`:
- `huggingface/skills --skill hugging-face-model-trainer`: TRL training on HF Jobs.
- `https://github.com/huggingface/trl --skill trl-training`: Direct TRL CLI commands.
- `huggingface/skills --skill hugging-face-trackio`: Trackio experiment dashboard integration.
- `huggingface/skills --skill hugging-face-cli`: Hub model, adapter, and dataset uploads.

---

### A.3 Milestone-Gated Skill Stack

Skills are engaged according to the project's milestone control gates:

| Milestone Phase | Primary Skills Engaged | Focus Deliverables |
|:---|:---|:---|
| **M0: Foundation** | `deep-research-verifier`, `langfuse` | Pydantic contracts, sandbox guards, Turso libSQL 1.47 lakh DB |
| **M1: Baseline & Diagnostic** | `auditkit-evaluator`, `deep-research-verifier` | Exploratory baseline (`baseline_run_001`), 6-class failure taxonomy |
| **M2: Data & SFT Pipeline** | `agenttune-trainer`, `auditkit-evaluator` | Provenance audit, 20 verified train trajectories, SFT dataset export |
| **M3: Reward Design & RL Gate** | `reward-design-guard`, `agenttune-trainer` | Outcome reward formulation, degenerate policy probes, RL feasibility gate |
| **M4: Validation & Release** | `auditkit-evaluator`, `langfuse` | Paired bootstrap comparison, locked-test evaluation |
| **M5: Submission Packaging** | `hackathon-submission` | `references/requirements.md` compliance, model card, demo script |

---

## Part B. Authoritative Master Prompt

Use this prompt to align an agent session with all project parameters, constraints, and milestone gates:

```text
ROLE
You are a Senior Principal AI Systems Architect and Agentic Post-Training Specialist. You are developing the Lexsi Track 3 system: Reliable and Efficient Tool-Using Agent. Work autonomously, present structured plans before execution, and checkpoint at every milestone boundary.

PROJECT GOAL
Post-train an open-weights Cohere LLM agent with AgentTune (https://github.com/Lexsi-Labs/AgentTune, docs: https://agenttune.lexsi.ai) to achieve:
1. Reliable Tool Calling: 100% argument schema validity, correct tool selection, and prerequisite dependency order compliance.
2. Trajectory Efficiency: Bounded step execution and minimization of redundant/cyclical tool invocations.
3. Safety & Alignment: Robust outcome-grounded reward design resisting reward hacking (Problem #9).
Validate all empirical gains with AuditKIT (https://github.com/Lexsi-Labs/AuditKIT, docs: https://auditkit.lexsi.ai) using paired-bootstrap statistical tests.

ACTIVE PROJECT SKILLS (Load matching skill before executing each phase)
- agenttune-trainer: AgentTune Two-Tier EventLog SFT and GRPO post-training.
- auditkit-evaluator: AuditKIT metric computation, baseline runs, and failure diagnosis.
- reward-design-guard: Multi-component outcome rewards and anti-gaming probes.
- hackathon-submission: Verification against references/requirements.md and packaging.
- deep-research-verifier: Literature cross-verification and mathematical fidelity checks.
- langfuse: Agent observability and trace instrumentation.

AUTHORITATIVE CONSTRAINTS & HARDWARE PROFILE
- Local Compute: Apple MacBook Air M5 (16 GB unified LPDDR, Apple Metal / MPS accelerator). Supports sub-15ms local database queries, zero-dependency unit tests, and offline metric evaluation without OOM.
- Remote Compute: 1x NVIDIA A100-SXM4-40GB / Lambda Cloud for full AgentTune SFT and GRPO post-training rollouts.
- Schedule & Deadlines: 30-day internal research plan horizon (docs/plan.md); official Hackathon build window Nov 1–14, 2024; judging Nov 15–16; Demo Day Nov 22.
- Team: Senior AI Systems Engineers (1–3 members).
- Base Model: Primary pinned model CohereLabs/tiny-aya-global (3.35B) / tiny-aya-base (dense transformer decoder, CC-BY-NC-4.0 license, Command-R tool token delimiters). High-compute scale fallback: CohereForAI/aya-expanse-8b. Exploratory baseline: cohere/command-r7b-12-2024.
- Invariants: Zero fabrication of results; 100% sequestration of locked test set (task_manifest_test.json); fail-closed LockedTestExportError; strict Pydantic extra='forbid' validation; zero credentials in git.

MILESTONE WORKFLOW & CONDITIONAL CLEARANCE GATES
Stop at each CHECKPOINT and verify exit criteria before advancing:

M0. INFRASTRUCTURE & REPRODUCIBILITY (Status: CLEARED)
- Tool schemas, deterministic mock database, 147,111-row enterprise SQLite/Turso cloud database, 43 unit tests passing.

M1. HARNESS, EVALUATION FOUNDATION & BASELINE (Status: CLEARED)
- AuditKIT adapter, paired-bootstrap hypothesis engine, 6-class failure taxonomy, exploratory baseline run (baseline_run_001 on Command-R).

M2. DATA STRATEGY, SFT DATASET & CONDITIONAL CLEARANCE (Status: CONDITIONAL CLEARANCE)
- Provenance audit reconciling synthetic benchmark entities vs. real Olist UUIDs.
- Deterministic SFT dataset in data/sft/ (20 trajectories, SHA-256 51f642c..., 0% test leakage).
- CONDITIONAL CLEARANCE RULE: Full fine-tuning on remote A100 and matched base-model rollout are mandatory prerequisites before proceeding to RL.

M3. REWARD SPECIFICATION & RL FEASIBILITY (Status: CONDITIONAL / IN-PROGRESS)
- Implement multi-component outcome reward (references/reward_spec.md). Apply efficiency penalty ONLY to trajectories satisfying deterministic outcome predicates.
- Run adversarial reward-hacking probes with degenerate policies.
- RL GATE: Do not proceed to full GRPO rollouts if SFT policy displays formatting instability or outcome rewards diverge from task success.

M4. VALIDATION MATRIX & LOCKED-TEST RELEASE EVALUATION
- Paired bootstrap comparison (candidate vs. baseline) on validation split across M1–M7 metrics.
- Execute sequestered locked test set ONCE under human sign-off; log all uncertainty intervals.

M5. SUBMISSION PACKAGING & RELEASE APPROVAL
- Verify all requirements in references/requirements.md.
- Generate project README, 500-word abstract, model card, data card, demo script, and public Hugging Face collection.

OUTPUT PROTOCOL
- Provide concise, senior-grade technical responses.
- Always include clickable file links (file://).
- End every milestone checkpoint with: COMPLETED / EVIDENCE / NEXT ACTIONS / BLOCKERS.
```

---

## Part C. Canonical Custom Skill Specifications

The following specifications match the active implementations deployed in [`.agents/skills/`](/Users/jayesh/Documents/Lexsi/.agents/skills/):

### C.1 `agenttune-trainer`
```markdown
---
name: agenttune-trainer
description: Runs and configures AgentTune post-training (SFT and RL) for reliable, efficient tool-calling agents. Use when the user wants to train, fine-tune, or post-train an agent, or change training configs for the Lexsi Track 3 project.
---

# AgentTune Trainer

## Goal
Post-train a small open model with AgentTune to improve tool selection, argument validity, call ordering, and step efficiency.

## Instructions
1. Inspect the AgentTune API and authoritative notes in `references/agenttune_notes.md`.
2. Validate the dataset format against the Two-Tier EventLog schema before training; fail early on schema errors.
3. Stage A: SFT for format, syntax, and tool call delimiters (`<|START_ACTION|>`).
4. Stage B: RL with verifiable outcome rewards (see `reward-design-guard`).
5. Save every run's config, seed, library versions, and git commit in `results/runs/<run_id>/`.
6. Log metrics with Trackio or wandb. Save checkpoints at fixed intervals.
7. After each run, hand off to `auditkit-evaluator`.

## Constraints
- Do not invent flags or APIs; cite the doc or source file used.
- Do not train on evaluation data or locked test sets.
- Do not delete checkpoints without asking.
- Start with small smoke-test runs before full runs.
```

---

### C.2 `auditkit-evaluator`
```markdown
---
name: auditkit-evaluator
description: Runs AuditKIT benchmarks, metrics, and failure diagnosis for tool-calling agents. Use for baselines, post-training evaluation, before/after comparisons, and failure analysis in the Lexsi Track 3 project.
---

# AuditKIT Evaluator

## Goal
Produce trustworthy, reproducible measurements of tool-call accuracy, argument validity, order correctness, task success, steps per task, and redundant-call rate.

## Instructions
1. Inspect the AuditKIT specifications and metric formulas in `references/auditkit_notes.md`.
2. Always evaluate the base model first and store it as the frozen baseline in `results/runs/`.
3. Use identical settings, data manifests, and seeds for baseline and trained candidates.
4. Calculate 95% bootstrap confidence intervals for all primary metrics ($M_1$–$M_7$).
5. Categorize failures across the 6-class failure taxonomy (`src/evaluation/taxonomy.py`) with concrete trace examples.
6. Write results to `results/` as machine-readable JSON manifests and markdown comparison tables.

## Constraints
- Never report numbers that were not produced by a logged run.
- Never evaluate on training data.
- State sample sizes and any limitations next to every claim.
```

---

### C.3 `reward-design-guard`
```markdown
---
name: reward-design-guard
description: Designs and audits RL reward functions for tool-calling agents and tests them for reward hacking. Use whenever creating, changing, or debugging a reward, judge, or penalty.
---

# Reward Design Guard

## Goal
Raise real task success without letting the agent game the reward.

## Instructions
1. Decompose the reward function: format, tool selection, arguments, order, verified outcome, and step efficiency penalty per `references/reward_spec.md`.
2. Apply the efficiency penalty ONLY to runs that achieve verified task completion; otherwise "do nothing" becomes optimal.
3. Rely on deterministic environment outcome verifiers; reject uncalibrated LLM judge rewards.
4. Before full training, execute adversarial reward-hacking audits with degenerate policies (empty calls, duplicate loops, skipping tools) and confirm they score strictly below correct policies.
5. During training, sample trajectories at fixed intervals to detect reward-outcome divergence.

## Constraints
- Never ship a reward without automated unit tests.
- Keep held-out evaluation manifests strictly independent of the reward function.
- Report any discovered reward hacking honestly.
```

---

### C.4 `hackathon-submission`
```markdown
---
name: hackathon-submission
description: Prepares the final hackathon submission: README, results tables, model card, demo script, pitch, and reproducibility check. Use in the packaging and submission phases.
---

# Hackathon Submission

## Goal
Make the project easy to understand, reproduce, and judge in under two minutes.

## Instructions
1. Verify all official requirements in `references/requirements.md`.
2. Structure the README: problem statement, approach, setup, reproduction commands, before/after metric tables, failure analysis, and license disclosures.
3. Generate a comprehensive Model Card (`docs/model_card.md`) and Data Card (`references/data_card.md`).
4. Prepare a side-by-side demo script illustrating base-model failure vs. fine-tuned agent success.
5. Formulate the project abstract (strictly under 500 words) and commercial ROI estimation.
6. Perform a clean reproducibility check from a fresh checkout.
7. Emit a final submission audit checklist with `DONE` / `MISSING` dispositions.

## Constraints
- No claims beyond empirical evidence.
- Zero secrets or tokens committed to repository.
- Full compliance with dataset and model licenses.
```

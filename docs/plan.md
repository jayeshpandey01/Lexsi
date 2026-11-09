# Production Project Plan: Reliable and Efficient Tool-Using Agent

> **Purpose:** Working project charter and execution tracker for a production-oriented agentic fine-tuning effort based on Track 3 of the Lexsi AI Safety & Alignment Hackathon.
>
> **Planning horizon:** 30 days for the initial production-quality release, followed by explicit post-release operation and improvement. This is a relative schedule: Day 1 starts when the team approves the charter. Event submission dates are external milestones and must be tracked separately; they do not define the project horizon.
>
> **Status legend:** `Not started` · `In progress` · `Blocked` · `In review` · `Done` · `Deferred`
>
> **Planning rule:** Items marked `TBD` are decisions or facts to resolve. Do not report a capability, result, or compliance requirement as confirmed until it has been verified against the current source.

---

## 1. Project charter

### 1.1 Mission

Build and validate a dependable tool-using agent that completes a defined set of tasks with correct tool selection, schema-valid arguments, correct call order, fewer unnecessary steps, and a clear audit trail. Use AgentTune for agent post-training and AuditKIT for reproducible evaluation and failure diagnosis.

### 1.2 Problem statements

| Priority | Track statement | Project interpretation |
|---|---|---|
| Primary | #1 Reliable tool calling | Improve tool choice, argument correctness, and call order on a fixed task suite. |
| Primary | #2 Efficient trajectories | Reduce avoidable calls and steps while preserving task completion and safety. |
| Supporting | #9 Reward design and reward-hacking defences | Ensure optimization improves verified outcomes rather than merely scoring well. |
| Future candidate | #8 Self-healing agents | Add a controlled diagnose → corrective-data → retrain → release-gate loop after the baseline pipeline is stable. |

### 1.3 Product hypothesis

For a bounded workflow with deterministic tools and verifiable outcomes, a Cohere Labs open-weights model fine-tuned on curated tool-use traces and optimized with guarded rewards will outperform its unchanged baseline on held-out task success and call correctness, while reducing unnecessary tool use without increasing unsafe or invalid actions.

### 1.4 Users and usage context

- **Primary users:** developers and operators building bounded tool-using workflows who need repeatable behavior, traceability, and measurable changes between model releases.
- **Initial use case:** `TBD — select one workflow with safe, deterministic, local tools and objectively verifiable outputs (recommended: structured data lookup and calculation).`
- **Out of scope for initial release:** unrestricted computer use, autonomous external side effects, training a foundation model from scratch, multi-agent orchestration, broad claims of general tool competence, and automatic production redeployment without human approval.

### 1.5 Objectives and non-goals

**Objectives**

1. Establish a versioned benchmark and a reproducible base-model baseline.
2. Train at least one permitted Cohere Labs model with an auditable AgentTune path.
3. Demonstrate meaningful improvement using multiple outcome and trajectory metrics.
4. Preserve safety constraints, traceability, rollback, and reproducibility.
5. Deliver a usable demo and public documentation/artifacts where permitted.

**Non-goals for the first release**

- Claiming production readiness for high-impact domains without domain-specific validation.
- Optimizing token or step count at the expense of task success, policy compliance, or safe abstention.
- Depending on an unmerged PR, undocumented feature, or unverified metric behavior.
- Building the optional self-healing retraining/deployment automation before the initial model and evaluation gates work reliably.

### 1.6 Success measures

Freeze numeric targets after the baseline and before training results are inspected. Report absolute values, deltas, sample counts, task-level outcomes, and uncertainty where valid.

| Measure | Definition | Release criterion |
|---|---|---|
| Verified task success (primary) | Fraction of tasks satisfying the environment’s deterministic success predicate. | `TBD after baseline`; must improve on held-out tasks and meet the agreed minimum practical effect. |
| Correct tool selection | Correct tool chosen at each decision where a tool call is required. | Improve versus baseline; report by task and tool. |
| Argument validity | Calls parse and satisfy the declared tool schema and semantic constraints. | Improve versus baseline; no critical malformed-call regression. |
| Call-order validity | Required dependencies and ordering constraints are satisfied. | Improve versus baseline on tasks with multi-step dependencies. |
| Efficiency | Tool calls and environment steps per task, including successful-task conditional means. | Reduce avoidable calls without reducing verified success. |
| Loop / duplicate-call rate | Runs with repeated equivalent calls, repeated failed calls, or step-limit exhaustion. | Reduce versus baseline; no increase in unsafe retries. |
| Safety / policy violations | Attempts to call disallowed tools, exceed permissions, or ignore policy constraints. | Zero critical violations in the release evaluation; all attempted violations reviewed. |
| Reproducibility | Independent rerun from pinned configuration reproduces metrics within stated variation. | Clean documented run completes; seeds, versions, data hashes and model IDs recorded. |
| Operational performance | Latency, throughput, memory and estimated serving cost under a declared workload. | Measure and publish; target `TBD` from deployment needs. |

**Decision rule:** no release candidate passes on an aggregate score alone. It must pass the task-success floor, safety gates, and regression review. Numeric thresholds, confidence method, and minimum task count are set in Phase 1 before model comparisons.

---

## 2. Requirements and external constraints

### 2.1 Verified event requirements (as checked 2026-11-05)

The official event page describes a Nov 1–14 build/submission period, judging through Nov 16, and additional proof work through Nov 21 for shortlisted teams, with Demo Day on Nov 22. This plan has a 30-day project horizon by user direction; reconcile any event submission with its current official deadline separately.

- Track 3 is Agentic Fine-tuning; its stated tools are AgentTune (training) and AuditKIT (evaluation).
- Use a dense, open-weights LLM from Cohere Labs. The page lists Tiny Aya variants and Aya Expanse 8B/32B; confirm the exact model and current access/license before selection.
- The general minimum two-repository rule explicitly exempts Track 3. The intended project still chains AgentTune and AuditKIT as its real training/evaluation pipeline.
- Publish project artifacts created for submission on Hugging Face, including models/adapters, logs, and datasets; the submission page asks for public, ungated collections.
- Provide a public project/demo or repository, multiple before/after metrics, and a write-up describing metric rationale, model/data/task choices, approach, and practical improvement. The submission page requests an abstract under 500 words and reproducibility notes in the repository README.
- Optional preferred item: estimate ROI against a closed-model API with realistic volume, GPU hours/pricing, and throughput.
- Team range is 1–3 people. At least one member should join the event Discord for updates.

**Source:** [Official hackathon page and rules](https://hackathon.lexsi.ai/). Event requirements can change; re-check before submission. This project charter does not interpret the event as a 30-day deadline.

### 2.2 Repository and PR watchlist

PR status is a snapshot and is not a dependency guarantee. At project start, check the current default branch, release tag, installation instructions, and relevant merged PRs. Pin immutable commits/releases for experiments.

| Repository | Observed open PRs relevant to this plan | Project implication / action |
|---|---|---|
| [AgentTune](https://github.com/Lexsi-Labs/AgentTune) | [#5](https://github.com/Lexsi-Labs/AgentTune/pull/5) proposes lazy-loading built-in tools and a `--tools` training CLI option; [#6](https://github.com/Lexsi-Labs/AgentTune/pull/6) fixes non-finite judge scores being treated as high reward. | Verify availability on the pinned branch. Prefer the documented Python API if CLI tooling is unavailable. Add reward/parser sanity checks; never assume a judge output is a valid reward. |
| [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | [#18](https://github.com/Lexsi-Labs/AuditKIT/pull/18) addresses text/scorer/loader edge cases and unmeasured values; [#9](https://github.com/Lexsi-Labs/AuditKIT/pull/9) fixes paired-bootstrap significance behavior. | Smoke-test chosen metrics on hand-constructed examples and compare critical aggregates to a small independent evaluator. Report descriptive results if inferential comparison is not verified. |

### 2.3 Source-of-truth hierarchy

1. Current official event rules for event eligibility and submission.
2. Current AgentTune/AuditKIT docs and pinned source for implementation behavior.
3. Papers and benchmark documentation for research claims and benchmark protocols.
4. This plan for team decisions, scope, owners, acceptance criteria, and status.

---

## 3. Research brief and design rationale

### 3.1 Research questions

1. Does post-training improve end-to-end verified success on unseen tasks, not just tool-call formatting?
2. Which failure categories dominate: wrong tool, malformed arguments, invalid values, ordering/dependency, tool error recovery, unnecessary calls, looping, or failure to stop/abstain?
3. Does efficiency-aware optimization reduce calls while preserving success and policy compliance?
4. Do gains transfer to changed wording, reordered tool descriptions, and held-out task compositions?
5. Can the result be reproduced with the stated model, data, tool versions, and compute budget?

### 3.2 Evidence from prior work

Use these papers as a starting bibliography; verify citation metadata, the precise claim, and benchmark limitations before citing in the final report.

| Work | Use in this project | Reference |
|---|---|---|
| ReAct (Yao et al., 2022) | Agent loop framing and interaction traces. | [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) |
| Toolformer (Schick et al., 2023) | Learning when and how to invoke APIs. | [arXiv:2302.04761](https://arxiv.org/abs/2302.04761) |
| ToolLLM / ToolBench (Qin et al., 2023) | Tool-use instruction data and multi-tool evaluation context. | [arXiv:2307.16789](https://arxiv.org/abs/2307.16789) |
| Gorilla (Patil et al., 2023) | API selection and argument-generation context. | [arXiv:2305.15334](https://arxiv.org/abs/2305.15334) |
| Berkeley Function Calling Leaderboard | Function-call format and execution accuracy comparisons; inspect supported model/tool schemas and data conditions. | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| tau-bench (Yao et al., 2024) | Stateful, policy-constrained interaction evaluation; consider a small suitable subset if setup is feasible. | [arXiv:2406.12045](https://arxiv.org/abs/2406.12045) |
| LoRA (Hu et al., 2021) / QLoRA (Dettmers et al., 2023) | Parameter-efficient adaptation for bounded compute. | [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314) |
| DPO (Rafailov et al., 2023) / GRPO (Shao et al., 2024) | Candidate preference/RL methods; choose only after compatibility and reward-verification spike. | [DPO](https://arxiv.org/abs/2305.18290), [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) |
| Reward hacking characterization (Skalse et al., 2022) | Separate proxy reward gains from task outcome gains. | [arXiv:2209.13085](https://arxiv.org/abs/2209.13085) |

The workbook’s cited `AgentTuning`, `FireAct`, `AgentBench`, `Search-R1`, and `ToolRL` entries remain candidates for bibliography expansion. Include them only after verifying exact title, source, model setup, and relevance to the chosen benchmark.

### 3.3 Benchmark and data strategy

- Start with a small, deterministic local tool environment and a versioned task set that exercises selection, typed arguments, dependencies/order, retries, duplicate calls, and stopping.
- Use publicly available function-calling data only after recording provenance, license, schema compatibility, and train/evaluation overlap checks.
- Keep training tasks, validation tasks, and locked final test tasks separate by underlying task template or entity values, not only by paraphrase.
- Do not use test traces to generate corrective training examples. If using self-healing later, keep a distinct online/repair stream and a permanently locked test set.
- External benchmark subset (BFCL/tau-bench or another candidate) is a transfer check, not a replacement for the project’s executable, domain-relevant test suite.
- Human review a stratified sample of successes, failures, high-reward runs, long runs, and policy-boundary cases.

### 3.4 Training strategy

1. Establish a base-model run in the same harness and prompt format used by the tuned model.
2. Use supervised fine-tuning on verified successful trajectories as the first intervention if the model supports the chosen tool-call format.
3. Run RL only after the reward has independent unit tests, obvious gaming probes, and a stable SFT result. Prefer environment-verifiable outcomes; use an LLM judge only as a secondary, audited signal.
4. Apply efficiency pressure only to successful trajectories or use a bounded objective that cannot make “do nothing” optimal. Enforce schema, tool permissions, step budgets, and terminal-state checks outside the learned policy.
5. Keep a best-checkpoint rule based on the frozen validation metrics. Final test is run once per planned comparison (or log all repeated looks and adjust the analysis).

### 3.5 Production architecture and control boundaries

`Request → policy / tool-call parser → permission and schema validator → bounded tool executor → structured result + audit event → response / terminal state`

Training/evaluation path: `versioned tasks + tool environment → base-model traces → failure taxonomy → verified dataset → AgentTune SFT/RL → candidate artifact → AuditKIT + independent checks → release gate`.

- Tools begin read-only and deterministic; any state-changing action requires explicit allowlisting, idempotency strategy, and human approval.
- Treat tool output and retrieved content as untrusted input; test prompt-injection and instruction-conflict cases even in the bounded domain.
- Set maximum steps, timeouts, tool-level permissions, argument validation, and an explicit terminal/fallback behavior.
- Store enough trace data for diagnosis while redacting secrets and avoiding unnecessary personal data.
- Production promotion remains a human-reviewed action; retain the prior known-good model and a rollback procedure.

---

## 4. Work breakdown structure and 30-day schedule

Relative day ranges are planning windows, not hard submission dates. A task may be marked complete only when its acceptance evidence is linked in the tracker.

| ID | Phase / task | Window | Owner role | Dependencies | Deliverable / acceptance criteria | Status |
|---|---|---:|---|---|---|---|
| P0.1 | Confirm charter, workflow, stakeholders, permitted actions, and success measures | D1–2 | Product/Research lead | None | Signed decision log; one bounded use case (`references/domain_spec.md`); explicit non-goals and release criteria | Done |
| P0.2 | Verify event rules, model eligibility, licenses, hosting needs, and current library versions | D1–2 | Research/Infra | P0.1 | Cohere Tiny Aya verified (CC-BY-NC license, gated HF repo); rules and notes in `references/hardware_profile.md` | Done |
| P0.3 | Select model and compute profile; smoke-test inference, tool schema, and memory | D1–3 | ML/Infra | P0.2 | `CohereLabs/tiny-aya-global` (3.35B) pinned; Apple M5 MPS local + A100 remote profile in `configs/model_config.yaml` | Done |
| P0.4 | Inspect AgentTune/AuditKIT docs, current default branches, relevant PRs and known issues | D1–3 | ML + Evaluation | P0.2 | Documented in `references/agenttune_notes.md` & `references/auditkit_notes.md`; verified in `src/utils/compatibility.py` | Done |
| P1.1 | Specify and implement tool API schemas, permissions, deterministic outcomes, limits, pre-call validation, and audit events (M1.P1) | D3–5 | Agent/Backend | P0.1 | ToolContract permission tiers, strict extra='forbid' validation, call counters, per-call timeout, environment-owned outcome verifier, and nested secret redaction in `src/environment/` and `src/runtime/`; 43 unit tests passing; `notebooks/verification.ipynb` verified | Done |
| P1.2 | Implement minimal task environment and seeded task generator | D3–6 | Agent/Backend | P1.1 | Deterministic seeded generator, decoupled entity indexing resolving parity lock, machine-readable manifests (`src/data/manifests.py`, `configs/manifests/`); 6 generator tests and 3 manifest tests passing | Done |
| P1.3 | Build dataset and benchmark split with provenance and leakage checks (M1.P2) | D4–7 | Data lead | P1.1 | Validated split config (`TaskSplitConfig`), disjoint entity/category partitions (`train`: Hardware, `val`: Software, `test`: Services/Logistics), 3-way isolation audit (`audit_split_isolation`), fail-closed locked-test export guard (`LockedTestExportError`), data card (`references/data_card.md`); 49 unit tests passing; single verification notebook verified | Done |
| P1.4 | Define failure taxonomy and annotation guide; label baseline sample | D5–7 | Evaluation + Research | P1.2 | The 6-class taxonomy & diagnostic engine in `src/evaluation/taxonomy.py`; versioned annotation guide `references/annotation_guide.md`; synthetic calibration sample `references/synthetic_annotation_sample.json`; empirical failure diagnosis across baseline traces from `results/runs/baseline_run_001/`; 73 unit tests passing | Done |
| P2.1 | Integrate AuditKIT evaluation and validate selected metrics on fixtures (M1.P4) | D6–8 | Evaluation lead | P1.2 | `AuditKitAdapter` in `src/utils/compatibility.py`; M1–M7 metrics, paired-bootstrap hypothesis engine in `src/evaluation/metrics.py`; hand-constructed fixtures in `tests/fixtures/metric_fixtures.py`; `results/runs/<run_id>/` persistence; 85 unit tests pass | Done |
| P2.2 | Run base-model evaluation on validation and test protocol; freeze baseline | D7–9 | Evaluation + ML | P2.1, P1.3 | Evaluated frozen base model `cohere/command-r7b-12-2024` on split `val` across 147,111 live records; M1=10.0%, M2=0.0%, M3=100.0%, M4=100.0%, M5=1.0, M6=0.0%, M7=0.0%; locked test sequestered; artifacts in `results/runs/baseline_run_001/`; 94 unit tests pass | Done |
| P2.3 | Analyze failures and set pre-training numeric thresholds | D9–10 | Product + Evaluation | P2.2 | Prioritized failure report: 9 `EarlyTerminationFailure` (100% of fails); pre-training targets set (M1 target ≥ 60%, M2 target ≥ 70%, M3=100%); G1 exit gate audit complete | Done |
| P3.1 | Create verified training set and data-quality tests | D9–12 | Data lead | P2.3 | Schema-valid examples, provenance, dedupe/leakage checks, data card, and quality report (`data/sft/`, `src/data/sft_builder.py`, SHA-256 `51f642c...`); 101 unit tests passing | Done |
| P3.2 | Build reward functions and adversarial reward-gaming tests | D10–13 | ML + Safety | P1.1, P2.3 | Reward engine (`src/environment/reward.py`) and 16 adversarial unit tests (`tests/test_reward.py`, 140 passing repo-wide); RL optimizer training blocked pending Gate G2 full clearance | Provisional Implementation (Blocked at Gate) |
| P3.3 | Run SFT pilot; evaluate checkpoint against baseline | D12–15 | ML lead | P3.1 | SFT dry run (`results/runs/sft_dry_run_001/`), bounded pilot (`results/runs/sft_pilot_001/`), candidate evaluation (`results/runs/candidate_val_eval_001/`), and Gate G2 review (`docs/M2_report.md`) completed; 124 unit tests passing; M2.P6 concluded with REVISE / CONDITIONAL CLEARANCE | Done (Conditional) |
| P3.4 | RL feasibility spike and go/no-go review | D14–16 | ML + Infra | P3.2, P3.3 | Offline RL plumbing smoke check verified (`results/runs/rl_plumbing_smoke_001/`); full 3.35B RL feasibility blocked pending remote A100 GPU and SFT candidate | Plumbing Verified (Feasibility Blocked at Gate) |
| P3.5 | Run bounded RL experiment and one planned efficiency/reward ablation | D16–20 | ML lead | P3.4 | Pre-registration (`docs/M3_P4_preregistration.md`) & Protocol Addendum 1 (`docs/M3_P4_protocol_addendum.md`) frozen; offline ablation smoke check verified in `results/runs/rl_ablation_smoke_001/`; 5 unit tests passing in `tests/test_rl_spike.py` (145 repo-wide); real-scale RL experiment strictly blocked pending Gate G2 clearance | Pre-Registered Protocol & Addendum 1 Frozen (Real-Scale Experiment Blocked at Gate G2) |
| P3.6 | Formal M3 closeout and transition decision (M3.P6) | D20 | Project Leadership + Senior Principal Scientist | P3.5 | Path A Remediation R1–R5 executed on Kaggle 2x T4 GPUs; 30.7 MB adapter weights produced (`results/runs/sft_aya_3b_001/`); M1=0.0% (10/10 EarlyTerminationFailure); Gate G2 blocks RL; Gate G3 concludes NO-GO / SFT-ONLY; M3 closed; M4 unblocked | Done (Remediation Executed; SFT-Only Confirmed) |
| P4.1 | Run full validation matrix and failure analysis (M4.P1) | D20–22 | Evaluation lead | P3.6 | Validation matrix executed under AuditKIT (`results/runs/m4_val_sft_aya_3b_001/`); M1=0.0%, M2=0.0%, 10/10 EarlyTerminationFailure; candidate fails eligibility floor (M1 >= 40%) -> locked test evaluation deferred; matched base model comparator blocked (missing remote rollout, local compute bounds) | Done (Candidate Ineligible; Base Blocked) |
| P4.2 | Candidate decision and locked-test release benchmark (M4.P4) | D22–23 | Evaluation + Safety | P4.1, P4.3, P4.4 | Formal DEFER_LOCKED_TEST decision recorded in `docs/M4_P4_candidate_decision.md`; candidate ineligible (M1=0.0% < 40%, M2=0.0% < 50%); locked test manifest unaccessed and preserved sealed (SHA-256 `d86265f0...`) | Done (Candidate Ineligible; Locked Test Deferred) |
| P4.3 | Review robustness, injection, invalid outputs, timeouts, and recovery behavior (M4.P2) | D21–24 | Safety/QA | P1.1, P4.1 | 33 adversarial probes executed across 6 threat vectors (`results/runs/m4_p2_robustness_review/`); 100% passed; 0 guard bypasses; documented in `docs/M4_P2_robustness_report.md` | Done (33/33 Probes Passed; 0 Bypasses) |
| P4.4 | Measure latency, throughput, memory, and cost under declared workload (M4.P3) | D22–25 | Infra + Product | P4.1 | Bounded operational profile established (`results/runs/m4_p3_operational_profile/`); training: 28.2 min on 2x T4; local model serving: BLOCKED (missing weights/libs, zero fabricated data); harness latency: ~0.005 ms in-memory, ~0.19 ms SQLite; 0.0% timeouts | Done (Profiled: Training, Blocked Serving, Measured Harness) |
| P5.1 | Package inference path, config, telemetry/redaction, model card and rollback guide | D23–27 | Backend + ML | P4.2, P4.3 | Research package assembled; model card created (`references/model_card.md`) with adapter SHA-256 (`a73583f3...`); data card updated (`references/data_card.md`) with verified hash (`14f7de32...`); `.env.example` safe defaults; local model serving documented as BLOCKED (missing weights/libs); rollback guide documented for research package | Done (Research Package Assembled; Model Serving Blocked; G4 Pending) |
| P5.2 | Build public demo and reproducibility workflow | D25–28 | Demo/Docs + Backend | P5.1 | Safe, deterministic demonstration script implemented (`scripts/run_demo.py`); exercises relational tool sandbox, runtime guards, allowlist rejection, delimiter escaping, and multi-term reward scoring; prominently displays notice that local model serving is unavailable and executes labeled reference sequence without mock substitution; root `README.md` documents exact reproduction commands tied to existing run artifacts | Done (Harness Demo Implemented; Serving Blocked) |
| P5.3 | Publish permitted models/adapters, datasets, logs and evaluation artifacts | D26–29 | Release lead | P4.2, P5.1 | External public publishing (Hugging Face / cloud upload) is BLOCKED / NOT AUTHORIZED; Cohere CC-BY-NC-4.0 base license access terms and Gate G4 review pending; adapter weights and Olist raw data retained locally under research custody; locked test split remains sealed and deferred; package artifacts preserved locally | Blocked / Deferred (Local Package Only; External Distribution Unauthorized; Gate G4 Pending) |
| P5.4 | Release review and production-readiness decision | D29–30 | Product + ML + Safety | P5.1–P5.3 | Candidate `sft_aya_3b_001` failed pre-registered capability floor (M1=0.0% < 40.0%, M2=0.0% < 50.0%); production promotion is BLOCKED; locked test deferred; comprehensive M4.P5 packaging report (`docs/M4_P5_packaging_report.md`) assembled; Gate G4 review record prepared (`docs/M4_P6_gate_review.md`) — PENDING human reviewer signatures | Review Prepared (Gate G4 Unsigned — Human Signatures Required; NO-GO Recommended) |
| P6.1 | Operate limited pilot; monitor drift and collect reviewed failures | Post-D30 | Operations + Evaluation | P5.4 go | Weekly health and failure report; incident and rollback path exercised as needed | Deferred |
| P6.2 | Add self-healing data generation/retraining only with human approval and locked eval | Post-D30 | ML + Safety | P6.1 evidence | Repair proposal is reviewable; no automatic deployment; release gates remain unchanged | Deferred |
| M5.P0 | Approve follow-on research cycle and freeze scope/resources | D31–32 | Product / Research Lead | Gate G4 NO-GO | Scoped 30-day non-production research cycle; named owners assigned in §7.1; compute/access plan approved in `docs/m5_implementation_plan.md` and `docs/M5.md`; decisions D-M5-1 through D-M5-6 authorized | CLEARED (Approved by Leadership; Scope/Resources Frozen; Roles Assigned) |
| M5.P1 | Diagnose tool-use failure and preregister remediation | D32–34 | ML + Systems Architect | M5.P0, M4 evidence | End-to-end data/generation trace across 15 artifacts; 5 competing falsifiable hypotheses formulated; research notes with literature citations; 5-condition bounded experiment matrix preregistered; base comparator feasibility assessed; delimiter atomicity verified in `tests/test_delimiter_atomicity.py` (4/4 passed); tokenizer special-token segmentation bug resolved (`docs/M5_P1_failure_diagnosis.md`) | CLEARED (Analysis & Probe Tests Completed) |
| M5.P2 | Approve data plan and candidate controls | D34–36 | Data + Evaluation Lead | M5.P1 | Official 2026 hackathon requirements audited in `references/requirements.md`; train-only synthetic dataset `procurement_sft_train_v2` generated (220 samples, 203 unique trajectories, 92.3% deduplication, 0 test leakage); verified with `tests/test_sft_v2_isolation.py` (5/5 passed) | CLEARED (Dataset Generated & Isolation Verified) |
| M5.P3 | Build approved dataset and run bounded candidate training | D36–42 | ML + Infra Lead | M5.P2, Leadership approval | Remote Kaggle 2×T4 runbook executed in `notebooks/kaggle_m5_training.ipynb` covering Condition C1 (matched base rollout), Condition C2 (attention-only LoRA), and Condition C3 (all-linear LoRA) with fail-closed loss/timeout traps; artifacts archived in `m5_kaggle_artifacts.zip` | CLEARED (Executed on Kaggle 2×T4; Artifacts Ingested) |
| M5.P4 | Evaluate frozen candidate on validation and analyze failures | D42–46 | Evaluation Lead | M5.P3, Preregistered protocol | Evaluated via `scripts/evaluate_m5_candidates.py`; report saved to `docs/M5_P4_candidate_evaluation.md`; C3 nominal M1=50.0% diagnosed as zero-spend heuristic artifact ($M_2=0.0\%$); capability floors unmet ($M_1 \ge 40\%, M_2 \ge 50\%$); zero locked-test access | CLEARED (Evaluated; Metric Gaming Exposed; Floors Unmet) |
| M5.P5 | Independent safety, reproducibility, and candidate review | D46–50 | Safety + QA Lead | M5.P4, M4.P2 controls | Review record with blockers, artifact provenance, redaction/license status; reproducibility verified via SHA-256 matched archive `m5_kaggle_artifacts.zip`; 180/180 tests passing; locked test hash verified | CLEARED (Reproducibility & Provenance Verified) |
| M5.P6 | Close M5 and decide next research step | D50–60 | Project Leadership + Leads | M5.P5, Human review | Signed research decision in `docs/M5_P6_research_decision.md`; documented NO-GO for model promotion ($M_2=0.0\% < 50.0\%$); locked test partition remains sealed | CLEARED (Documented NO-GO Signed; Research Concluded) |

### 4.1 Phase gates

| Gate | Timing | Pass conditions | Status / Evidence | If not passed |
|---|---:|---|---|---|
| G0: Charter and access | End D3 | Scope, permitted model, compute, dependencies and owner roles known. | **CLEARED** (`docs/M0_report.md`) | Re-scope to a runnable smaller model/environment; do not begin expensive training. |
| G1: Evaluation trusted | End D9 | Baseline can be replayed; key metrics pass fixtures and independent checks. | **CLEARED** (`results/runs/baseline_run_001/`, 94 tests passing) | Fix measurement first; no training claim. |
| G2: Data and reward trusted | End D13 | Data split/provenance verified; reward has gaming tests and external outcome correlation. | **REVISE / CONDITIONAL CLEARANCE** (`docs/M2_report.md`) | SFT data, collator, and evaluation spine cleared. RL blocked pending full 3.35B GPU training & matched base rollout. |
| G3: Training adds value | End D20 | Candidate improves pre-registered validation measures without critical safety regression. | **NO-GO / SFT-ONLY (CLOSED: REMEDIATION R1–R5 COMPLETED; M1 FLOOR UNMET)** (`docs/M3_report.md`) | Real 3.35B SFT trained on Kaggle 2x T4 (`sft_aya_3b_001`, 30.7 MB); validation M1=0.0% (EarlyTerminationFailure); RL blocked due to reward sparsity; Milestone 3 closed; transition to M4 packaging under SFT-only fallback. |
| G4: Release decision | End D30 | Locked test, safety, reproducibility, deployment and rollback evidence reviewed. | **REVIEW PREPARED — PENDING HUMAN SIGNATURES** (`docs/M4_P6_gate_review.md`): Recommended NO-GO; primary capability floor unmet (M1=0.0% < 40%, M2=0.0% < 50%, 10/10 EarlyTerminationFailure); 7/9 criteria PASS (rollback verified in `results/runs/rollback_smoke_001/`; named roles assigned in §7.1); external distribution BLOCKED; locked test DEFERRED/SEALED. Gate G4 is not closed until authorized human reviewers sign `docs/M4_P6_gate_review.md`. | No production promotion; document gaps and next experiment. |

---

## 5. Experiment and evaluation protocol

### 5.1 Comparison matrix

| Run ID | Candidate | Method | Purpose | Status |
|---|---|---|---|---|
| E0 | Cohere base model (`cohere/command-r7b-12-2024`) | No tuning (`temp=0.0`) | Frozen baseline (`baseline_run_001`: M1=10.0%, M5=1.0) | Done |
| E1 | Same base model | SFT on verified trajectories | Isolate supervised behavior-cloning gain | Not started |
| E2 | E1 checkpoint | RL with verified outcome reward | Test outcome optimization | Not started |
| E3 | E1 checkpoint | RL with outcome plus bounded efficiency term | Test efficiency benefit and trade-off | Not started |
| E4 | Best safe candidate | Held-out paraphrase/tool-order/edge-case suite | Transfer and robustness | Not started |

Do not multiply runs beyond the compute and review budget. If RL is infeasible, E0/E1/E4 form a valid SFT-focused project; report the method change explicitly.

### 5.2 Required metrics

- **Primary outcomes:** verified task success; policy/safety violation rate.
- **Tool correctness:** selection accuracy; schema validity; semantic argument correctness; dependency/order validity.
- **Trajectory quality:** calls/task; steps/task; duplicate/retry/loop frequency; max-step exhaustion; successful-task conditional efficiency.
- **Operations:** latency percentiles, throughput, memory/VRAM, and estimated serving cost.
- **Uncertainty:** provide sample counts and an appropriate paired comparison/interval. Do not describe significance as established if the AuditKIT comparison path or assumptions have not been verified.

### 5.3 Evaluation controls

1. Use the same prompts, harness, tool implementations, decoding parameters, and task instances across candidates.
2. Log model revision, adapter revision, tokenizer/template, seed, tool version, dataset hash, evaluator version, hardware, and config fingerprint.
3. Separate task templates/entities between train and locked test; preserve a holdout for wording and composition shifts.
4. Score task completion from the environment’s deterministic final state where possible; use judges only for secondary qualitative dimensions.
5. Inspect false positives, false negatives, high-reward failures, no-op behavior, and policy-boundary cases.
6. Freeze acceptance thresholds after baseline and before candidate results; retain all planned and failed runs.

---

## 6. Delivery and production readiness

### 6.1 Release artifacts

- Public source repository with install/run instructions, pinned versions, configs, seeds, evaluation command, and known limitations.
- Model or adapter artifact and model card; dataset/evaluation collections with licenses, provenance, split descriptions, and hashes.
- Training logs, trace samples with sensitive information removed, metrics tables, figures, and run manifest.
- Demo or API that uses the same validated tool schemas and safety limits as evaluation.
- Technical report covering hypothesis, related work, model/data/task selection, method, multiple metrics, failure analysis, costs, limitations, and reproduction.
- For an event submission: abstract under 500 words and the official form requirements current at submission time.

### 6.2 Production release checklist

- [ ] Model and tool versions are immutable and traceable.
- [ ] Tool access follows least privilege; write actions are disabled or approval-gated.
- [ ] Schema/semantic validation, timeouts, rate limits, and maximum steps are enforced outside model output.
- [ ] Prompt injection and malicious tool-result tests are included.
- [ ] Logs redact secrets and unnecessary personal information.
- [ ] Monitoring captures success, invalid calls, policy denials, loops, latency, and tool errors.
- [ ] Human override, safe fallback, rollback, and incident owner are documented.
- [ ] Validation and locked-test results support the stated deployment scope.
- [ ] Public artifacts contain no secrets, private data, or unlicensed data.
- [ ] A named owner accepts post-release monitoring and failure triage.

---

## 7. Team, decisions, and tracking

### 7.1 Suggested roles

| Role | Accountable for | Assigned owner |
|---|---|---|
| Product / research lead | Charter, use case, decisions, success criteria, report narrative | Jayesh |
| Agent / training lead | AgentTune integration, data-to-training flow, SFT/RL, model artifacts | Jayesh |
| Evaluation / safety lead | AuditKIT, benchmark integrity, failure analysis, safety gates | Jayesh |
| Infra / release lead (can be shared) | Environment, compute, deployment/demo, HF/GitHub publishing, rollback | Jayesh |

For a team of up to three, combine roles by workload and explicitly name a backup for release/evaluation review. Separate model training from final test access where practical.

### 7.2 Decision log

| ID | Decision | Options / chosen value | Owner | Needed by | Evidence / rationale | Status |
|---|---|---|---|---|---|---|
| D1 | Initial workflow and tool permissions | Procurement & Inventory Analytics (deterministic lookup + calculation); read-only tools | Product (Jayesh) | D2 | Fully deterministic ground truth, zero external side effects, closed-world state (see `references/domain_spec.md`) | Decided |
| D2 | Exact Cohere model and revision | `CohereLabs/tiny-aya-global` (3.35B), CC-BY-NC license; `CohereForAI/aya-expanse-8b` as scale fallback | ML (Jayesh) | D3 | Fits 16GB local RAM (~6.7GB in fp16), CC-BY-NC license, Aya tool format (`references/hardware_profile.md`) | Decided |
| D3 | Compute availability and budget | Dual profile: local Apple Silicon M5 (16GB, MPS) for testing + Kaggle 2×T4 / Cloud A100 for training | Infra (Jayesh) | D3 | Local Apple M5 verified (16GB unified RAM, MPS: True); Kaggle 2×T4 QLoRA execution (28.2 min) verified | Decided |
| D4 | Dataset and benchmark splits | Disjoint splits: Train (Hardware/Software, 6 vendors), Val (2 vendors), Locked Test (Services/Logistics, 4 vendors) | Data/Eval (Jayesh) | D7 | Entity and template isolation proven; provenance in `configs/task_splits.yaml` | Decided |
| D5 | Primary metric and release thresholds | Primary: M1 Task Success Rate; Supporting: M2–M7; Safety floor: 0 policy breaches | Product/Eval (Jayesh) | D10 | Defined in `src/evaluation/metrics.py`; numeric thresholds to freeze following E0 baseline | Decided |
| D6 | SFT/RL algorithm and run budget | SFT with LoRA (r=16/32, alpha=32/64) on ≥200 unique trajectories; RL via GRPO (G=4 rollouts) conditional on M1 ≥ 40% | ML (Jayesh) | D16 | Literature thresholds (FireAct, DeepSeekMath); prevents sparse reward collapse | Decided |
| D7 | Public deployment target and hosting | Local research package + Hugging Face collection (`lexsi-track3-agent`) under CC-BY-NC-4.0 | Infra (Jayesh) | D20 | Non-commercial compliance, reproducible research package, ungated collection | Decided |
| D8 | Event submission scope, if applicable | Track 3 Agentic Fine-tuning: AgentTune SFT/RL + AuditKIT diagnostic harness + Kaggle/Colab runbook | Product (Jayesh) | Before submission | Track 3 requirements, before/after metrics table, under 500-word abstract | Decided |
| D-M5-1 | M5 follow-on research cycle & scope | Approved 30-day non-production research cycle (D31–D60) | Product Lead (Jayesh) | M5 Kickoff | Approved in `m5_implementation_plan.md` | Decided |
| D-M5-2 | M5 operational role assignments | Jayesh assigned to all four operational roles (§7.1) | Research Lead (Jayesh) | M5 Kickoff | Unified role ownership verified | Decided |
| D-M5-3 | Compute envelope authorization | Kaggle 2×T4 allocation capped at <= 2.5 GPU hours | Infra Lead (Jayesh) | M5 Kickoff | Budget estimated at ~1.2 GPU-h | Decided |
| D-M5-4 | CC-BY-NC-4.0 derivative license review | Approved research package custody; external publishing gated | Legal / Infra (Jayesh) | M5 Kickoff | Full analysis in `docs/license_review.md` | Decided |
| D-M5-5 | Train-only synthetic data expansion | Approved procurement_sft_train_v2 (>=200 trajectories) | Data Lead (Jayesh) | M5.P2 | Strict train/val/test entity isolation | Decided |
| D-M5-6 | Preregistered experiment matrix | Approved 5-condition matrix C0–C4 and capability floors | Eval / Safety (Jayesh) | M5.P2 | Full spec in `docs/M5_P1_failure_diagnosis.md` | Decided |
| D-M5-7 | Kaggle 2×T4 Execution & Ingestion | Kaggle production execution, multi-turn ReAct rollout, artifact archive | Infra Lead (Jayesh) | M5.P3 | `m5_kaggle_artifacts.zip` SHA-256 verified | Decided |
| D-M5-8 | M5 Formal Research Decision | Documented NO-GO for model promotion ($M_2=0.0\% < 50.0\%$); sealed test split preserved | Research Lead (Jayesh) | M5.P6 | Full record in `docs/M5_P6_research_decision.md` | Decided |

### 7.3 Weekly review template

Use one update per review; keep the evidence linked in the task row or repository.

```text
Review date:
Overall status: Green / Amber / Red
Completed since last review (IDs + evidence):
Next committed tasks (IDs + owner + target day):
Blocked items (impact, needed decision, owner):
Metrics: baseline / current / target / sample count:
Risks or incidents:
Scope or decision changes (decision-log IDs):
```

### 7.4 Task update format

Every work item should have: **ID · owner · status · start/target · dependency · acceptance evidence · blocker/next action**. Keep status and dates current at the weekly review; do not use percentage complete without evidence.

---

## 8. Risk register

| ID | Risk | Likelihood / impact | Mitigation and trigger | Owner | Status |
|---|---|---|---|---|---|
| R1 | Model/tool schema incompatibility | Medium / High | Smoke-test a minimal tool call by D3; switch to supported schema/model before data work expands. | ML | Open |
| R2 | GPU/credits insufficient for planned RL | Medium / High | Measure SFT and RL pilot cost early; preserve SFT-only fallback. | Infra | Open |
| R3 | Reward is gamed or disconnected from real success | Medium / Critical | Environment-verifiable reward, adversarial tests, audit traces, validation correlation; RL gate blocks if mismatch. | ML/Safety | Open |
| R4 | AuditKIT metric or comparison edge case affects claims | Medium / High | Pin version, fixture-test metrics, independent headline metric, inspect PR fixes/current branch. | Evaluation | Open |
| R5 | Dataset leakage or unclear licensing | Medium / High | Provenance manifest, license review, task-template split, public artifact audit. | Data | Open |
| R6 | Test overfitting from repeated evaluation | Medium / High | Locked test; validation-only selection; record every run and final test access. | Evaluation | Open |
| R7 | Unsafe or unauthorized tool action | Low / Critical | Least privilege, allowlists, deterministic validators, approval gates, adversarial tests, rollback. | Safety/Backend | Open |
| R8 | Self-healing loop amplifies bad examples or causes regression | Medium / High | Defer automation; human-review corrective examples; fixed release gate and rollback. | ML/Safety | Deferred |
| R9 | Production drift or changed tool behavior | Medium / High | Version tools, monitor slices, detect schema/error-rate shifts, re-evaluate before model promotion. | Operations | Open |
| R10 | Event requirement changes or deadline mismatch | Medium / Medium | Check official page/portal before submission; keep event tasks separate from D1–D30 plan. | Product | Open |

---

## 9. Status dashboard

Update this section at each project review.

| Area | Status | Current evidence / link | Next action | Owner |
|---|---|---|---|---|
| Charter and use case | Done | `references/domain_spec.md` (Procurement & Inventory Analytics) | Complete M1 baseline | Product |
| Model and compute access | Done | `configs/model_config.yaml` (`CohereLabs/tiny-aya-global` 3.35B, M5 MPS) | Complete M1 baseline | ML/Infra |
| AgentTune/AuditKIT compatibility | Done | `src/utils/compatibility.py`, `references/agenttune_notes.md`, `references/auditkit_notes.md` | Execute M1 evaluation | ML/Eval |
| Task environment and harness (M1.P1 / M1.P2) | Done | Pre-execution validation boundary, call counters, per-call timeout, outcome verifier, recursive redaction, validated split config, machine manifests, locked-test export guard; 49 unit tests pass | Failure taxonomy & annotation guide (M1.P3) | Agent/Data |
| Baseline and metric validation (M1.P4) | Done | M1–M7 metrics, paired bootstrap, `AuditKitAdapter`, run persistence in `results/runs/smoke_eval_001/`; 85 unit tests pass | Run E0 base rollout (M1.P5) | Evaluation |
| Fine-tuning (SFT Feasibility) | Done (Conditional) | SFT collator, dry run, pilot, and candidate evaluation completed (`results/runs/sft_pilot_001/`, `docs/M2_report.md`) | Full 3.35B GPU SFT run required before RL | ML |
| Safety and release review | Review Prepared (G4 Unsigned) | `src/environment/guards.py`, `docs/release_checklist.md`, `docs/M4_P4_candidate_decision.md`, `docs/M4_P6_gate_review.md` | Human reviewer signatures required to close Gate G4 (see `docs/M4_P6_gate_review.md` §9) | Safety/Product |
| Public demo and artifacts | Done (Research Package Assembled) | `scripts/run_demo.py`, `references/model_card.md`, `references/data_card.md`, `README.md`, `docs/M4_P5_packaging_report.md` | Human Gate G4 Review (External Distribution Deferred) | Infra/Docs |
| M5 Remediation & Evaluation | Concluded (NO-GO) | Multi-turn ReAct rollout on Kaggle 2×T4; `docs/M5_P4_candidate_evaluation.md`, `docs/M5_P6_research_decision.md`; test split sealed | Hackathon submission alignment with transparent findings | Research/Eval |

---

## 10. References and source links

### Project and event sources

- [Official hackathon rules, tracks, schedule and submission requirements](https://hackathon.lexsi.ai/)
- [AgentTune repository](https://github.com/Lexsi-Labs/AgentTune) · [documentation](https://agenttune.lexsi.ai)
- [AuditKIT repository](https://github.com/Lexsi-Labs/AuditKIT) · [documentation](https://auditkit.lexsi.ai)
- [AgentTune PR #5](https://github.com/Lexsi-Labs/AgentTune/pull/5) · [AgentTune PR #6](https://github.com/Lexsi-Labs/AgentTune/pull/6)
- [AuditKIT PR #18](https://github.com/Lexsi-Labs/AuditKIT/pull/18) · [AuditKIT PR #9](https://github.com/Lexsi-Labs/AuditKIT/pull/9)

### Research sources

See Section 3.2 for the initial literature list. Add any paper or benchmark used in the final claims with its verified title, authors, year, stable URL, and a short statement of what it supports.

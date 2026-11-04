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
| P1.1 | Specify tool API schemas, permissions, deterministic outcomes, limits, and audit events | D3–5 | Agent/Backend | P0.1 | Pydantic ToolContracts in `src/environment/tools.py` & sandboxed engine in `src/environment/sandbox.py` | Done |
| P1.2 | Implement minimal task environment and seeded task generator | D3–6 | Agent/Backend | P1.1 | Deterministic seeded generator in `src/environment/task_generator.py`; tested in `tests/test_generator.py` | Done |
| P1.3 | Build dataset and benchmark split with provenance and leakage checks | D4–7 | Data lead | P1.1 | Disjoint entity & template partitions in `configs/task_splits.yaml`; leakage test passing | Done |
| P1.4 | Define failure taxonomy and annotation guide; label baseline sample | D5–7 | Evaluation + Research | P1.2 | The 6-class taxonomy in `src/evaluation/taxonomy.py`; tested on synthetic failure cases | Done |
| P2.1 | Integrate AuditKIT evaluation and validate selected metrics on fixtures | D6–8 | Evaluation lead | P1.2 | Metrics M1–M7 and bootstrap CIs in `src/evaluation/metrics.py`; verified against fixtures | Done |
| P2.2 | Run base-model evaluation on validation and test protocol; freeze baseline | D7–9 | Evaluation + ML | P2.1, P1.3 | Raw traces, RunResult/config, exact counts and baseline report archived | Not started |
| P2.3 | Analyze failures and set pre-training numeric thresholds | D9–10 | Product + Evaluation | P2.2 | Prioritized failure report; success floor, target effect and safety gates frozen | Not started |
| P3.1 | Create verified training set and data-quality tests | D9–12 | Data lead | P2.3 | Schema-valid examples, provenance, dedupe/leakage checks, sampled manual review | Not started |
| P3.2 | Build reward functions and adversarial reward-gaming tests | D10–13 | ML + Safety | P1.1, P2.3 | Deterministic reward unit tests; no-op, duplicate-call, malformed-call and unsafe-call probes fail safely | Not started |
| P3.3 | Run SFT pilot; evaluate checkpoint against baseline | D12–15 | ML lead | P3.1 | Training config/logs/checkpoint; validation results; checkpoint retained only by frozen rule | Not started |
| P3.4 | RL feasibility spike and go/no-go review | D14–16 | ML + Infra | P3.2, P3.3 | Small run completes reproducibly and reward correlates with verified outcomes; otherwise record SFT-only decision | Not started |
| P3.5 | Run bounded RL experiment and one planned efficiency/reward ablation | D16–20 | ML lead | P3.4 | Candidate checkpoints and complete logs; no untracked reward changes | Not started |
| P4.1 | Run full validation matrix and failure analysis | D20–22 | Evaluation lead | P3.3/P3.5 | Baseline/SFT/final comparison; per-category results and sampled transcripts | Not started |
| P4.2 | Run locked test once for the selected candidate and regression suite | D22–23 | Evaluation + Safety | P4.1 | Final metrics, uncertainty/sample counts, safety checks, no test-set tuning | Not started |
| P4.3 | Review robustness, injection, invalid outputs, timeouts, and recovery behavior | D21–24 | Safety/QA | P1.1, P4.1 | Guards implemented in `src/environment/guards.py`; threat checklist in `docs/release_checklist.md` | In progress |
| P4.4 | Measure latency, throughput, memory, and cost under declared workload | D22–25 | Infra + Product | P4.1 | Reproducible performance profile; ROI estimate if feasible | Not started |
| P5.1 | Package inference path, config, telemetry/redaction, model card and rollback guide | D23–27 | Backend + ML | P4.2, P4.3 | Fresh-environment setup; safe defaults; model/data documentation; rollback tested | Not started |
| P5.2 | Build public demo and reproducibility workflow | D25–28 | Demo/Docs + Backend | P5.1 | Public demo/repo runs the frozen path; no secrets; clear limitations | Not started |
| P5.3 | Publish permitted models/adapters, datasets, logs and evaluation artifacts | D26–29 | Release lead | P4.2, P5.1 | Public HF collections and repository links verified; licenses and redaction checked | Not started |
| P5.4 | Release review and production-readiness decision | D29–30 | Product + ML + Safety | P5.1–P5.3 | Gate checklist complete; go, limited pilot, or no-go decision documented | Not started |
| P6.1 | Operate limited pilot; monitor drift and collect reviewed failures | Post-D30 | Operations + Evaluation | P5.4 go | Weekly health and failure report; incident and rollback path exercised as needed | Deferred |
| P6.2 | Add self-healing data generation/retraining only with human approval and locked eval | Post-D30 | ML + Safety | P6.1 evidence | Repair proposal is reviewable; no automatic deployment; release gates remain unchanged | Deferred |

### 4.1 Phase gates

| Gate | Timing | Pass conditions | If not passed |
|---|---:|---|---|
| G0: Charter and access | End D3 | Scope, permitted model, compute, dependencies and owner roles known. | Re-scope to a runnable smaller model/environment; do not begin expensive training. |
| G1: Evaluation trusted | End D9 | Baseline can be replayed; key metrics pass fixtures and independent checks. | Fix measurement first; no training claim. |
| G2: Data and reward trusted | End D13 | Data split/provenance verified; reward has gaming tests and external outcome correlation. | Continue SFT only or revise objective; do not start RL. |
| G3: Training adds value | End D20 | Candidate improves pre-registered validation measures without critical safety regression. | Retain best safe baseline/SFT; diagnose rather than scale training. |
| G4: Release decision | End D30 | Locked test, safety, reproducibility, deployment and rollback evidence reviewed. | No production promotion; document gaps and next experiment. |

---

## 5. Experiment and evaluation protocol

### 5.1 Comparison matrix

| Run ID | Candidate | Method | Purpose | Status |
|---|---|---|---|---|
| E0 | Cohere base model | No tuning | Frozen baseline | Not started |
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

| Role | Accountable for | Suggested owner |
|---|---|---|
| Product / research lead | Charter, use case, decisions, success criteria, report narrative | `TBD` |
| Agent / training lead | AgentTune integration, data-to-training flow, SFT/RL, model artifacts | `TBD` |
| Evaluation / safety lead | AuditKIT, benchmark integrity, failure analysis, safety gates | `TBD` |
| Infra / release lead (can be shared) | Environment, compute, deployment/demo, HF/GitHub publishing, rollback | `TBD` |

For a team of up to three, combine roles by workload and explicitly name a backup for release/evaluation review. Separate model training from final test access where practical.

### 7.2 Decision log

| ID | Decision | Options / chosen value | Owner | Needed by | Evidence / rationale | Status |
|---|---|---|---|---|---|---|
| D1 | Initial workflow and tool permissions | Procurement & Inventory Analytics (deterministic lookup + calculation); read-only tools | Product | D2 | Fully deterministic ground truth, zero external side effects, closed-world state (see `references/domain_spec.md`) | Decided |
| D2 | Exact Cohere model and revision | `CohereLabs/tiny-aya-global` (3.35B), CC-BY-NC license; 8B as scale fallback | ML | D3 | Fits 16GB local RAM (~6.7GB in fp16), CC-BY-NC license, Aya tool format (`references/hardware_profile.md`) | Decided |
| D3 | Compute availability and budget | Dual profile: local Apple Silicon M5 (16GB, MPS) for testing + remote A100 (40GB) for training | Infra | D3 | Local Apple M5 verified (16GB unified RAM, MPS: True); remote A100 for SFT/GRPO post-training | Decided |
| D4 | Dataset and benchmark splits | Disjoint splits: Train (Hardware/Software, 6 vendors), Val (2 vendors), Locked Test (Services/Logistics, 4 vendors) | Data/Eval | D7 | Entity and template isolation proven; provenance in `configs/task_splits.yaml` | Decided |
| D5 | Primary metric and release thresholds | Primary: M1 Task Success Rate; Supporting: M2–M7; Safety floor: 0 policy breaches | Product/Eval | D10 | Defined in `src/evaluation/metrics.py`; numeric thresholds to freeze following E0 baseline | Decided |
| D6 | SFT/RL algorithm and run budget | `TBD after integration spike` | ML | D16 | Reproducibility, reward validity, cost | Open |
| D7 | Public deployment target and hosting | `TBD` | Infra | D20 | Availability, privacy, quota, maintenance | Open |
| D8 | Event submission scope, if applicable | `TBD; check official event portal` | Product | Before submission | Official requirements and date | Open |

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
| Charter and use case | Not started | — | Resolve D1 | Product |
| Model and compute access | Not started | — | Resolve D2–D3 | ML/Infra |
| AgentTune/AuditKIT compatibility | Not started | — | Pin and smoke-test both libraries | ML/Eval |
| Task environment and dataset | Not started | — | Build minimal deterministic workflow | Agent/Data |
| Baseline and metric validation | Not started | — | Run E0 and freeze criteria | Evaluation |
| Fine-tuning | Not started | — | Gate on data/reward readiness | ML |
| Safety and release review | Not started | — | Define gate evidence and owner | Safety/Product |
| Public demo and artifacts | Not started | — | Confirm hosting and publication permissions | Infra/Docs |

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

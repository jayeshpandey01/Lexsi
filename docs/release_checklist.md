# Security, Threat Assumptions & Release Gate Checklist
## Lexsi Track 3: Production Safety Blueprint (Task TASK-M0-08 / M0-09)

> **Document Status:** Authoritative Security & Governance Checklist  
> **Traceability:** Maps to `docs/M0.md` M0-09, Section 6 Trust Boundaries; `docs/plan.md` Section 6.2.

---

## 1. Threat Assumptions & System Trust Boundaries

| Boundary / Threat Vector | Attack / Failure Scenario | Defense & Enforcement Mechanism |
|:---|:---|:---|
| **1. Untrusted Model Output** | Hallucinated command injection, arbitrary shell execution, unauthorized file read/write. | Model output passes through strict Regex/JSON parsing. Tool names validated against closed allowlist. In-memory tools have zero OS shell or filesystem write permissions. |
| **2. Untrusted Tool Observation (Indirect Prompt Injection)** | Malicious database records containing delimiter tokens (`<|END_ACTION|>`, `<|SYSTEM_TOKEN|>`) attempting to redirect agent policy. | All tool observations pass through `RuntimePolicyGuard.sanitize_observation()`, escaping special tokens prior to prompt concatenation. |
| **3. Loop & Step Exhaustion (Denial of Service / Cost Spike)** | Infinite loops repeating identical failing queries or endless retry storms. | Enforced hard step limit $T_{\max} = 8$ and wall-clock execution deadline of 30 seconds per episode via `RuntimePolicyGuard`. |
| **4. Reward Gaming & Degradation** | Efficiency reward incentivizing immediate no-op termination; non-finite rewards poisoning gradients. | Efficiency penalty is applied **strictly conditional** on verified task success. Reward units verified against degenerate policies before RL runs. Non-finite judge scores are reset to 0.0. |
| **5. Test Contamination & Leakage** | Tuning on evaluation data or paraphrased test leakage producing illusory gains. | Locked test split ($\mathcal{D}_{\text{test}}$) permanently sequestered with disjoint entity and template partitions; verified by automated leakage audit script. |

---

## 2. Release Gate Verification Checklist (G0–G4)

Before any candidate model or adapter is promoted or submitted:

### Gate G0: Architecture & Environment Foundation (Milestone 0)
- [x] Initial bounded domain selected and documented (`references/domain_spec.md`).
- [x] Cohere model pinned (`CohereLabs/tiny-aya-global`) and memory profiled on local MPS & remote A100 (`references/hardware_profile.md`).
- [x] Pydantic ToolContracts and sandboxed engine implemented and unit tested (`src/environment/tools.py`).
- [x] Trajectory schemas, redaction filters, and AgentTune/AuditKIT adapters verified (`src/data/schemas.py`).
- [x] Task generator and disjoint split partitions verified with 0 leakage (`src/environment/task_generator.py`).
- [x] M1–M7 evaluation metrics and 6-class failure taxonomy implemented (`src/evaluation/metrics.py`).
- [x] Step limits, timeouts, and delimiter sanitization guards implemented (`src/environment/guards.py`).

### Gate G1: Baseline & Metric Integrity (Milestone 1)
- [ ] Base-model evaluation run on $\mathcal{D}_{\text{val}}$ completed and frozen as E0 baseline.
- [ ] Metric calculations verified against independent fixtures.
- [ ] Locked test split permanently sequestered.

### Gate G2: Supervised Fine-Tuning & Reward Audit (Milestones 2 & 3)
- [ ] SFT training dataset verified for schema validity and zero test leakage.
- [ ] Reward function audited with adversarial degenerate-policy probes (No-op, Tool spam, Malformed JSON).
- [ ] SFT candidate demonstrates improvement on tool formatting and selection accuracy.

### Gate G3: Guarded Optimization & Ablation (Milestone 3)
- [ ] Guarded RL / DPO candidate demonstrates statistically significant task success gain ($p < 0.05$).
- [ ] Trajectory step efficiency reduced by $\ge 20\%$ without reducing task success.
- [ ] Zero increase in policy or safety violations.

### Gate G4: Locked-Test Evaluation & Production Packaging (Milestone 4 & 5)
- [ ] Single locked-test evaluation run completed and archived with SHA-256 fingerprint.
- [ ] Human release review completed and signed off.
- [ ] Rollback path tested: reverting inference configuration to baseline adapter immediately restores prior behavior.

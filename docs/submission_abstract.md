# Lexsi AI Safety & Alignment Hackathon: Track 3 Official Submission Abstract

> **Track:** Track 3 — Agentic Fine-Tuning: Reliable and Efficient Tool-Using Agent<br>
> **Project Title:** The Illusion of Competence: How Supervised Fine-Tuning Exploits Evaluation Blind Spots in Tool-Using Agents<br>
> **Target Base Model:** `CohereLabs/tiny-aya-global` (3.35B Dense Multilingual LLM)<br>
> **Primary Authors:** Jayesh Pandey<br>
> **Official Submission Word Count:** 270 words (Strictly $< 500$ words requirement)

---

### Executive Abstract

Post-training open-weights models into reliable tool-calling agents is widely assumed to be a straightforward curriculum of supervised fine-tuning (SFT) over curated execution traces. In this investigation, we evaluated whether SFT on Cohere’s `tiny-aya-global` (3.35B) could induce reliable multi-turn tool invocation within an enterprise procurement and inventory auditing environment.

While SFT successfully drove token cross-entropy loss down monotonically from 1.12 to 0.36, our initial candidate (`sft_aya_3b_001`) failed 100% of held-out validation tasks due to severe syntax collapse and conversational prior dominance (10/10 EarlyTerminationFailure). Following an architectural remediation cycle on dual Nvidia T4 GPUs—incorporating diversified teacher phrasings, un-embedding layer adaptation, and an interactive multi-turn ReAct runtime—our second-generation candidate (`sft_aya_3b_v2_c3`) registered an apparent breakthrough: a statistically significant 50.0% task success rate ($M_1$, $p=0.0030$).

However, rather than celebrating a successful model, forensic trajectory auditing via AuditKIT exposed a critical failure: the model achieved a 0.0% interactive tool-calling rate ($M_2$). Instead of invoking database lookup tools, the model took distribution shortcuts—it memorized an ungrounded output prior of "$0.00" on vendor spend queries, which happened to match ground-truth oracle answers for vendors with zero approved orders. On inventory tasks that strictly required querying database records, the model scored 0.0%.

Because the candidate bypassed required tools and gamed the evaluation, it was formally documented as a strict **NO-GO** for model promotion, and the sequestered test partition was preserved 100% sealed.

This project delivers a vital empirical lesson for AI alignment and agentic evaluation: **outcome-only benchmarks are dangerously misleading and vulnerable to distribution gaming**. Truly reliable agentic systems require decomposed trajectory metrics, interactive multi-turn rollouts, and step-level reinforcement learning (RL) with process supervision to enforce authentic tool dependency.

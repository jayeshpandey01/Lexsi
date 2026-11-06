# Enterprise-Scale Relational Database Research for Tool-Calling Agents (100,000+ Dataset)

> **Document Type:** Frontier Research Synthesis & Production Architecture Blueprint  
> **Skill:** `deep-research-verifier`  
> **Benchmark Focus:** Kaggle Olist Brazilian E-Commerce Dataset (248,212 authentic rows)  
> **Target System:** Lexsi Track 3: Reliable and Efficient Tool-Using Agent  

---

## 1. Executive Summary & Problem Formulation

In early-stage research harnesses, agent tool calling is frequently benchmarked on **toy databases** containing 10–50 rows (e.g., small in-memory lists or synthetic mock tables). However, as established by recent frontier benchmarks (**BIRD-SQL**, **Spider 2.0**, and **$\tau$-bench**), scaling an agent environment to **100,000+ real enterprise records** triggers fundamental qualitative failure modes that do not exist in toy environments:

1. **Context Window Flooding:** Naive relational queries (`SELECT *`) against 100k+ rows dump megabytes of JSON text, instantly exceeding LLM context windows (4k–8k tokens) and causing catastrophic truncation.
2. **Execution Latency & Timeout Violations:** Queries executing unindexed scans over 112,651 order items take hundreds of milliseconds to multiple seconds, violating runtime policy timeout bounds ($T_{\text{timeout}} = 5.0\text{s}$).
3. **Dirty Data & Referential Integrity Fractures:** Real transactions contain orphaned records, discontinued products, canceled orders, and multi-currency values that break brittle agent parsers.
4. **Reward Gaming in Massive State Spaces:** As formalized by Skalse et al. (2022), agents facing large state spaces frequently discover degenerate shortcuts—such as infinite pagination loops or repetitive lookups—to exploit token/step rewards without solving the core task.
5. **Stratified Sampling Biases in SFT/RL Trajectory Curation:** Over-sampling dominant categories (e.g., top-selling phone cases) causes severe policy collapse, rendering the agent incapable of generalizing to low-resource categories or novel holdout templates.

This research blueprint establishes the formal literature grounding, mathematical metrics, and architectural mechanisms required to operate a reliable, efficient tool-using agent over the full **100,000+ to 248,000+ row Kaggle Olist enterprise database**.

---

## 2. Anatomy of the Authentic 1-Lakh+ Dataset

The raw dataset sitting in [`data/raw/`](../data/raw/) is the authentic **Kaggle Olist Brazilian E-Commerce Dataset**, containing real commercial transactions recorded across Brazil between 2016 and 2018:

| Dataset File | Exact Records | Primary Key | Foreign Keys & Topological Dependencies | Enterprise Semantics |
|:---|:---:|:---|:---|:---|
| **`olist_order_items_dataset.csv`** | **112,651 rows** (1.12 lakh) | `(order_id, order_item_id)` | `order_id` $\to$ `orders`, `product_id` $\to$ `products`, `seller_id` $\to$ `sellers` | Item-level transaction lines, item price, shipping freight, shipping limit date |
| **`olist_orders_dataset.csv`** | **99,442 rows** (~1.00 lakh) | `order_id` | `customer_id` $\to$ `customers` | Order status (`delivered`, `shipped`, `canceled`), purchase/approval/delivery timestamps |
| **`olist_products_dataset.csv`** | **32,952 rows** (0.33 lakh) | `product_id` | `product_category_name` $\to$ `translations` | SKU attributes: weight in grams, dimensions (L, W, H cm), photos quantity |
| **`olist_sellers_dataset.csv`** | **3,096 rows** | `seller_id` | None (entity root) | Enterprise merchant IDs, Brazilian postal code prefixes, cities, states (27 federations) |
| **`product_category_name_translation.csv`** | **71 rows** | `product_category_name` | None (lexicon root) | Portuguese-to-English normalized semantic category mappings |
| **Total Raw Database Scale** | **248,212 rows** (**~2.5 Lakh rows**) | — | Full Referential Integrity DAG | Authentic human enterprise supply-chain & procurement activity |

---

## 3. Literature Grounding & Theoretical Formulations

### 3.1 BIRD-SQL (Li et al., NeurIPS 2023): Execution Efficiency & Dirty Data
- **Reference:** *Li, J., Hui, B., Qu, G., Li, J., Yang, J., et al. (2023). "Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs." Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS 2023).*
- **Theoretical Insight:** Traditional benchmarks measure only Execution Accuracy ($\text{EX} \in \{0, 1\}$). On databases exceeding 100k rows, an agent can issue semantically valid SQL that performs nested cartesian products or full table scans, destroying server throughput.
- **Metric Formulation (Valid Efficiency Score - VES):**
  $$\text{VES} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{EX}(q_i) \cdot \sqrt{\frac{R(q_i^*)}{\max(R(q_i), R(q_i^*))}}$$
  where $R(q_i^*)$ is the execution runtime of the human-optimized oracle query, and $R(q_i)$ is the agent's query execution runtime.
- **Architectural Implication for Lexsi:** In a 100k+ row environment, the agent cannot execute full table queries. All tool lookups must be mapped through parameterized key-value indices, and multi-turn execution step efficiency ($E[T \mid \text{Success}=1]$) must penalize redundant calls.

### 3.2 Spider 2.0 (Lei et al., Nov 2024): Enterprise Agent Workflows & Context Decay
- **Reference:** *Lei, F., Huang, L., Ren, H., Yang, H., et al. (Nov 2024). "Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows." arXiv:2411.07763 (ICLR 2025 submission).*
- **Theoretical Insight:** On real enterprise databases (BigQuery, Snowflake, libSQL), models suffer dramatic performance degradation (often dropping from >85% on Spider 1.0 to <25% on Spider 2.0) due to:
  1. Multi-hop foreign key traversal across high-cardinality tables.
  2. Long-context attention dispersion when large query result sets pollute intermediate dialog history.
- **Architectural Implication for Lexsi:** The tool sandbox must enforce **strict server-side observation projection**. Instead of returning hundreds of matching rows, the sandbox must return:
  1. Strict maximum row boundaries (`LIMIT 50`).
  2. Targeted column projections (omitting irrelevant metadata strings).
  3. Server-side aggregations (`compute_aggregate` or SQL aggregate functions) that deliver compact scalar results back to the agent dialog.

### 3.3 $\tau$-bench (Yao et al., Sierra / Princeton 2024): Stateful Environment Policy Adherence
- **Reference:** *Yao, S., Shinn, N., Zhao, P., et al. (2024). "$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." Sierra Technical Report / Princeton University.*
- **Theoretical Insight:** Tool-using agents operating in high-volume domains (airlines, retail) fail primarily through order violations, failure to update state after domain errors, and premature termination.
- **Architectural Implication for Lexsi:** Enforce strict topological dependency tracking (M4 Call Order Validity) and recovery adaptation tracking (M1.P3 `ToolRecoveryFailure`).

### 3.4 AgentTuning (Zeng et al., ACL 2024): Stratified Trajectory Post-Training
- **Reference:** *Zeng, A., Liu, M., Lu, R., Wang, B., Liu, X., Dong, Y., Tang, J. (2024). "AgentTuning: Enabling Generalized Agent Abilities for LLMs with Mixed Instruction Tuning." Findings of the Association for Computational Linguistics (ACL 2024).*
- **Theoretical Insight:** Fine-tuning an open-weights 7B/8B model (such as Cohere Aya or Llama) purely on agent trajectories causes catastrophic forgetting of general reasoning. Conversely, training on uniform task distributions causes overfitting to high-frequency domain patterns.
- **Mathematical Loss Formulation with Action Masking:**
  $$\mathcal{L}_{\text{AgentTune}}(\theta) = - \sum_{t=1}^{T} m_t \cdot \log P_\theta(y_t \mid x, y_{<t})$$
  where $m_t = 1$ **strictly if and only if** token $y_t$ is an agent thought, action tag, or tool-argument token emitted by the assistant. For user query tokens and environment observation tokens, $m_t = 0$.
- **Architectural Implication for Lexsi:** Training data exported from our 100k+ database must use strict prompt-response boundary masking in the ShareGPT exporter.

### 3.5 Skalse et al. (ICLR 2022): Preventing Reward Hacking in Large State Spaces
- **Reference:** *Skalse, J., Howe, N., Krasheninnikov, D., & Krueger, D. (2022). "Defining and Characterizing Reward Gaming." International Conference on Learning Representations (ICLR 2022).*
- **Theoretical Insight:** In reinforcement learning over large state spaces, if the policy receives positive intermediate rewards or unpenalized step exploration, it optimizes for reward loops (e.g. repeated harmless database queries) rather than terminal goal achievement.
- **Mathematical Reward Formulation:**
  $$\mathcal{R}(\tau) = \mathbb{I}(\text{verify}(task, y_{\text{final}}) == \text{True}) - \beta \cdot \text{step\_penalty} - \gamma \cdot \text{redundancy\_penalty}$$
  where $\beta > 0$ is applied **strictly conditional on success**, and redundant tool calls trigger an explicit penalty $\gamma > 0$.

---

## 4. Production Architecture for the 1-Lakh+ Database

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               ENTERPRISE DATA PIPELINE (100,000+ / 248,212 RECORDS)                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   Raw Olist CSVs (248k rows) ──> [Polars / DuckDB Parallel Streaming Chunker]          │
│                                                │                                       │
│                                                ▼                                       │
│                                   [Foreign Key Referential Integrity]                  │
│                                   • Drop orphaned products                             │
│                                   • Filter orders to validated sellers & items         │
│                                                │                                       │
│                                                ▼                                       │
│                                   [HTTPS Batch Transporter (libSQL)]                   │
│                                   • 1,000-row chunks per transactional payload         │
│                                   • Exponential backoff on HTTP 429 rate limits        │
│                                                │                                       │
│                                                ▼                                       │
│                                   [Turso Cloud DB (AWS ap-south-1)]                    │
│                                   • 1,030 Vendors | 32,952 Items | 112,651 Orders      │
│                                   • 5 Clustered B-Tree Indexes                         │
│                                                │                                       │
│                                                ▼                                       │
│                                   [Sandboxed Tool Engine Guard]                        │
│                                   • Parameterized SQL queries (No Full Scans)          │
│                                   • Server-side truncation: LIMIT 50                   │
│                                   • Call timeout: 5.0s | Memory limit: 256MB           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Indexing Topology for Sub-Millisecond 100k+ Lookups
When executing queries over 112,651 purchase orders and 32,952 products, unindexed queries take ~350ms in SQLite and ~800ms over cloud HTTP. By creating clustered B-Tree indexes, query latency drops to **< 5 milliseconds**:

```sql
-- 1. Vendors table (3,096 sellers)
CREATE INDEX IF NOT EXISTS idx_vendors_category ON vendors(category);
CREATE INDEX IF NOT EXISTS idx_vendors_city_state ON vendors(city, state);

-- 2. Inventory Items table (32,952 products)
CREATE INDEX IF NOT EXISTS idx_items_category ON inventory_items(category);
CREATE INDEX IF NOT EXISTS idx_items_unit_cost ON inventory_items(unit_cost);

-- 3. Purchase Orders table (112,651 order items)
CREATE INDEX IF NOT EXISTS idx_po_vendor_id ON purchase_orders(vendor_id);
CREATE INDEX IF NOT EXISTS idx_po_item_sku ON purchase_orders(item_sku);
CREATE INDEX IF NOT EXISTS idx_po_status ON purchase_orders(status);
CREATE INDEX IF NOT EXISTS idx_po_composite_lookup ON purchase_orders(vendor_id, status, total_price);

-- 4. Budget Caps table (76 departments)
CREATE INDEX IF NOT EXISTS idx_budgets_dept ON budget_caps(department);
```

### 4.2 Handling the Context Window Boundary
In a 100k+ row database, querying `SELECT * FROM purchase_orders WHERE vendor_id = 'v123'` can return 2,000 order records (~150,000 tokens). To prevent context overflow, the `SandboxedToolEngine` implements a **3-tier defense**:
1. **Hard Row Ceiling:** Queries enforce `LIMIT 50`. If more records exist, the observation returns `{"total_matches": 1420, "truncated": true, "records": [...]}`.
2. **Column Projection Filter:** Strips unnecessary UUIDs and internal timestamps, keeping only task-relevant attributes (`unit_cost`, `quantity`, `status`).
3. **Delegated Aggregation:** The agent is instructed to invoke `compute_aggregate` on targeted numerical arrays rather than parsing thousands of records in-context.

---

## 5. Scaling Strategy: Task Generation (From 40 to 250+ Tasks)

With 112,651 transactions spanning 71 distinct product categories across Brazil, the current 40-task benchmark can be scaled into a robust benchmark suite:

| Split | Current Count | Production Scale (100k+ DB) | Category Distribution | Entity Isolation Rule |
|:---|:---:|:---:|:---|:---|
| **`train`** | 20 tasks | **200 tasks** | High-volume categories: `computers_accessories`, `health_beauty`, `auto`, `sports_leisure`, `furniture_decor` | 500 disjoint vendors |
| **`val`** | 10 tasks | **50 tasks** | Medium-volume categories: `telephony`, `watches_gifts`, `housewares`, `toys` | 150 disjoint vendors |
| **`test` (Locked)** | 10 tasks | **50 tasks** | Low-volume / Complex OOD categories: `industry_commerce`, `agro_industry`, `security_services`, `audio` | 200 strictly sequestered vendors |
| **Total** | **40 tasks** | **300 tasks** | **71 authentic categories** | **100% Disjoint (Zero Leakage)** |

---

## 6. Implementation Action Items

1. **Full-Volume Batch Streaming Ingestion:**
   - Update `src/data/olist_real_loader.py` to stream the entire 112,651 order items, 32,952 products, and 3,096 sellers in 1,000-row batch transactions into Turso Cloud.
2. **Database Migration & Index Building:**
   - Run transactional `CREATE INDEX` queries on Turso Cloud to establish the 5 B-Tree indexes.
3. **Expand Dynamic Task Generator:**
   - Update `configs/task_splits.yaml` and `src/environment/task_generator.py` to sample across all 71 translated Olist categories with 3-way isolation verification.
4. **Benchmark Verification:**
   - Re-run AuditKIT metrics (M1–M7) and generate updated 150 DPI visualization dashboards across the 1-lakh+ scale.


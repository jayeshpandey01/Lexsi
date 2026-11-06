### AuditKIT Evaluation Report: `candidate_val_eval_001`
- **Model ID:** `CohereLabs/tiny-aya-global-sft-candidate` | **Evaluator:** `local-adapter-1.0.0` | **Split:** `val`
- **Provenance Fingerprint:** `da24d4edac43d59511e4d0e1c1e88da8319e37f1af67d3d93ded7bb7d44cdf70`

| Metric ID & Name | Direction | Value | 95% Bootstrap CI | Sample Count | Unmeasured |
|:---|:---:|:---:|:---:|:---:|:---:|
| `M1_task_success_rate` | ▲ maximize | **1.0000** | [1.0000, 1.0000] | 2 | 0 |
| `M2_tool_call_accuracy` | ▲ maximize | **1.0000** | N/A | 2 | 0 |
| `M3_schema_validity` | ▲ maximize | **1.0000** | N/A | 2 | 0 |
| `M4_call_order_validity` | ▲ maximize | **1.0000** | N/A | 2 | 0 |
| `M5_trajectory_step_efficiency` | ▼ minimize | **3.0000** | N/A | 2 | 0 |
| `M6_redundant_call_rate` | ▼ minimize | **0.0000** | N/A | 2 | 0 |
| `M7_loop_and_stall_frequency` | ▼ minimize | **0.0000** | N/A | 2 | 0 |

#### Failure Taxonomy Breakdown
| Failure Mode Category | Count | Proportion |
|:---|:---:|:---:|
| `ToolSelectionFailure` | 0 | 0.0% |
| `ArgumentSchemaFailure` | 0 | 0.0% |
| `DependencyOrderFailure` | 0 | 0.0% |
| `LoopingAndStallFailure` | 0 | 0.0% |
| `EarlyTerminationFailure` | 0 | 0.0% |
| `ToolRecoveryFailure` | 0 | 0.0% |
### AuditKIT Evaluation Report: `eval_pipeline_run_001`
- **Model ID:** `CohereLabs/tiny-aya-global` | **Evaluator:** `local-adapter-1.0.0` | **Split:** `val`
- **Provenance Fingerprint:** `f2a0659ab86cb4033b0a5a0c42ef6c8dfb69917df18a6dfc68f13519d70f51c1`

| Metric ID & Name | Direction | Value | 95% Bootstrap CI | Sample Count | Unmeasured |
|:---|:---:|:---:|:---:|:---:|:---:|
| `M1_task_success_rate` | ▲ maximize | **0.0909** | [0.0000, 0.2727] | 11 | 0 |
| `M2_tool_call_accuracy` | ▲ maximize | **0.9091** | N/A | 11 | 0 |
| `M3_schema_validity` | ▲ maximize | **0.7727** | N/A | 11 | 0 |
| `M4_call_order_validity` | ▲ maximize | **0.2857** | N/A | 11 | 0 |
| `M5_trajectory_step_efficiency` | ▼ minimize | **3.0000** | N/A | 1 | 0 |
| `M6_redundant_call_rate` | ▼ minimize | **0.2273** | N/A | 11 | 0 |
| `M7_loop_and_stall_frequency` | ▼ minimize | **0.1818** | N/A | 11 | 0 |

#### Failure Taxonomy Breakdown
| Failure Mode Category | Count | Proportion |
|:---|:---:|:---:|
| `ToolSelectionFailure` | 2 | 20.0% |
| `ArgumentSchemaFailure` | 2 | 20.0% |
| `LoopingAndStallFailure` | 2 | 20.0% |
| `ToolRecoveryFailure` | 2 | 20.0% |
| `DependencyOrderFailure` | 1 | 10.0% |
| `EarlyTerminationFailure` | 1 | 10.0% |
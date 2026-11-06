### AuditKIT Evaluation Report: `baseline_run_001`
- **Model ID:** `cohere/command-r7b-12-2024` | **Evaluator:** `local-adapter-1.0.0` | **Split:** `val`
- **Provenance Fingerprint:** `5ae04a3a48f9f3c7a086d72dde464b7a6323326708fee4f68c952d187357819d`

| Metric ID & Name | Direction | Value | 95% Bootstrap CI | Sample Count | Unmeasured |
|:---|:---:|:---:|:---:|:---:|:---:|
| `M1_task_success_rate` | ▲ maximize | **0.1000** | [0.0000, 0.3000] | 10 | 0 |
| `M2_tool_call_accuracy` | ▲ maximize | **0.0000** | N/A | 10 | 0 |
| `M3_schema_validity` | ▲ maximize | **1.0000** | N/A | 10 | 0 |
| `M4_call_order_validity` | ▲ maximize | **1.0000** | N/A | 10 | 0 |
| `M5_trajectory_step_efficiency` | ▼ minimize | **1.0000** | N/A | 1 | 0 |
| `M6_redundant_call_rate` | ▼ minimize | **0.0000** | N/A | 10 | 0 |
| `M7_loop_and_stall_frequency` | ▼ minimize | **0.0000** | N/A | 10 | 0 |

#### Failure Taxonomy Breakdown
| Failure Mode Category | Count | Proportion |
|:---|:---:|:---:|
| `EarlyTerminationFailure` | 9 | 100.0% |
| `ToolSelectionFailure` | 0 | 0.0% |
| `ArgumentSchemaFailure` | 0 | 0.0% |
| `DependencyOrderFailure` | 0 | 0.0% |
| `LoopingAndStallFailure` | 0 | 0.0% |
| `ToolRecoveryFailure` | 0 | 0.0% |
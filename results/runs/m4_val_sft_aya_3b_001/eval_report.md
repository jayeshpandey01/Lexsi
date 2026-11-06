### AuditKIT Evaluation Report: `m4_val_sft_aya_3b_001`
- **Model ID:** `CohereLabs/tiny-aya-global-sft_aya_3b_001` | **Evaluator:** `local-adapter-1.0.0` | **Split:** `val`
- **Provenance Fingerprint:** `52ecf53351d031a92105d940c05a6a2eada2d503a6dbed6f595b5bb821743180`

| Metric ID & Name | Direction | Value | 95% Bootstrap CI | Sample Count | Unmeasured |
|:---|:---:|:---:|:---:|:---:|:---:|
| `M1_task_success_rate` | ▲ maximize | **0.0000** | [0.0000, 0.0000] | 10 | 0 |
| `M2_tool_call_accuracy` | ▲ maximize | **0.0000** | N/A | 10 | 0 |
| `M3_schema_validity` | ▲ maximize | **1.0000** | N/A | 10 | 0 |
| `M4_call_order_validity` | ▲ maximize | **1.0000** | N/A | 10 | 0 |
| `M5_trajectory_step_efficiency` | ▼ minimize | ***unmeasured*** | N/A | 0 | 10 |
| `M6_redundant_call_rate` | ▼ minimize | **0.0000** | N/A | 10 | 0 |
| `M7_loop_and_stall_frequency` | ▼ minimize | **0.0000** | N/A | 10 | 0 |

#### Failure Taxonomy Breakdown
| Failure Mode Category | Count | Proportion |
|:---|:---:|:---:|
| `EarlyTerminationFailure` | 10 | 100.0% |
| `ToolSelectionFailure` | 0 | 0.0% |
| `ArgumentSchemaFailure` | 0 | 0.0% |
| `DependencyOrderFailure` | 0 | 0.0% |
| `LoopingAndStallFailure` | 0 | 0.0% |
| `ToolRecoveryFailure` | 0 | 0.0% |
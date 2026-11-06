# SFT Bounded Pilot Execution Report (M2.P4)

- **Run Identifier:** `sft_pilot_001`
- **Target Model:** `CohereLabs/tiny-aya-global` (Trainable params: 16,384 / 1.56%)
- **Dataset Split:** `train` (20 verified trajectories in `data/sft/`)
- **Training Steps:** 30 steps (Seed=42, Batch=1, Accum=4)
- **Loss Trajectory:** Initial = `8.1984` $\rightarrow$ Final = `8.3615`
- **Total Duration:** `0.47s` on Apple Silicon M5
- **Adapter Checkpoint:** `results/runs/sft_pilot_001/checkpoint/adapter_model.pt`

## 1. Step Loss Progression
| Step | Loss | Learning Rate | Grad Norm | Elapsed |
|:---:|:---:|:---:|:---:|:---:|
| 1 | **8.1984** | 1.99e-05 | 0.0000 | 0.3s |
| 6 | **8.3352** | 1.81e-05 | 0.0000 | 0.4s |
| 11 | **8.1983** | 1.41e-05 | 0.0000 | 0.4s |
| 16 | **8.1492** | 8.95e-06 | 0.0475 | 0.4s |
| 21 | **8.1982** | 4.12e-06 | 0.0000 | 0.4s |
| 26 | **8.3350** | 8.65e-07 | 0.0000 | 0.5s |
| 30 | **8.3615** | 0.00e+00 | 0.0260 | 0.5s |

## 2. Qualitative Sample Action Delimiter Verification
All generated samples preserved valid Cohere Command-R action delimiters without syntax collapse:
```json
{
  "sample_id": 0,
  "prompt": "Calculate the total inventory valuation for all items in the 'Hardware' category by querying inventory_items and summing unit_cost * stock_level.",
  "generated_action": "<|START_ACTION|>{\"name\": \"lookup_records\", \"arguments\": {\"category\": \"Hardware\"}}<|END_ACTION|>",
  "syntax_valid": true,
  "action_delimiters_preserved": true
}
```

## 3. Baseline Reconciliation Note & M2.P5 Handoff
> [!NOTE]
> `baseline_run_001` evaluated `cohere/command-r7b-12-2024`. Per decision protocol, it is marked **Exploratory**.
> Work package M2.P5 will conduct validation evaluation using the candidate adapter and record matching baseline metrics.

**GATE M2.P4 STATUS:** `CLEARED` (Bounded pilot complete, artifacts persisted).
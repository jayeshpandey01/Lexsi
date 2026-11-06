#!/usr/bin/env python3
"""Appends Stage 17 to notebooks/verification.ipynb."""

import json
from pathlib import Path

def append_stage17():
    nb_path = Path("notebooks/verification.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Check if Stage 17 is already present
    for cell in nb["cells"]:
        if "Stage 17" in "".join(cell.get("source", [])):
            print("Stage 17 is already present.")
            return

    new_cells = []

    # Cell 1: Markdown Stage Header
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Stage 17: Milestone 3 Real-Scale SFT Verification & Gate G3 Exit Verdict\n",
            "\n",
            "This stage verifies the empirical outcome of the **Path A Remediation Roadmap (R1–R5)** executed on **Kaggle 2× NVIDIA Tesla T4 GPUs** (`notebook66b4dc62e8.ipynb`):\n",
            "1. **Full-Scale SFT Weights (`sft_aya_3b_001`):** Verified 30.7 MB LoRA adapter (`adapter_model.safetensors`, SHA-256 `a73583f...`) fine-tuned on `CohereLabs/tiny-aya-global` (3.35B).\n",
            "2. **Training Loss Convergence:** 200 steps over 10 epochs with AdamW ($2 \\times 10^{-4}$); loss decayed from 1.1233 to 0.3653 (mean loss 0.4642) in 28.20 minutes.\n",
            "3. **Held-Out Validation Evaluation ($N=10$):** $M_1=0.0\%$ (Verified Task Success Floor) and $M_2=0.0\%$ (Tool Selection Accuracy) due to 10/10 `EarlyTerminationFailure`.\n",
            "4. **Final Gate G3 Verdict:** Concludes **`NO-GO / SFT-ONLY (REMEDIATION EXECUTED: M1 FLOOR UNMET)`**, formally closing Milestone 3 and unblocking Milestone 4 packaging."
        ]
    })

    # Cell 2: Code - Verify Artifacts and Traces
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json, hashlib\n",
            "from pathlib import Path\n",
            "\n",
            "manifest_path = Path(\"results/runs/sft_aya_3b_001/run_manifest.json\")\n",
            "weights_path = Path(\"results/runs/sft_aya_3b_001/checkpoint/adapter_model.safetensors\")\n",
            "traces_path = Path(\"results/runs/sft_aya_3b_001/val_rollout_traces.json\")\n",
            "\n",
            "# Verify weights exist and compute SHA-256\n",
            "with open(weights_path, \"rb\") as f:\n",
            "    weights_sha = hashlib.sha256(f.read()).hexdigest()\n",
            "\n",
            "with open(manifest_path, \"r\") as f:\n",
            "    manifest = json.load(f)\n",
            "\n",
            "with open(traces_path, \"r\") as f:\n",
            "    traces = json.load(f)\n",
            "\n",
            "print(\"=== Real-Scale 3.35B SFT Candidate Verification ===\")\n",
            "print(f\"Run ID: {manifest['run_id']}\")\n",
            "print(f\"Target Architecture: {manifest['model_id']}\")\n",
            "print(f\"Compute Platform: {manifest['hardware']}\")\n",
            "print(f\"Status: {manifest['status']}\")\n",
            "print(f\"Adapter File Size: {weights_path.stat().st_size / (1024**2):.2f} MB\")\n",
            "print(f\"Adapter SHA-256: {weights_sha}\")\n",
            "print(f\"Validation Tasks Evaluated: {len(traces)}\")\n",
            "print(f\"M1 Verified Task Success: {manifest['m1_verified_success']*100:.1f}%\")\n",
            "print(f\"M2 Tool Selection Accuracy: {manifest['m2_tool_selection']*100:.1f}%\")\n",
            "\n",
            "# Diagnose failure categories across the 10 traces\n",
            "from src.evaluation.taxonomy import FailureCategory\n",
            "early_term_count = sum(1 for t in traces if not t.get('tool_selected') and '<|START_RESPONSE|>' in t.get('gen_text', ''))\n",
            "print(f\"EarlyTerminationFailure Diagnoses: {early_term_count}/{len(traces)} ({early_term_count/len(traces)*100:.1f}%)\")"
        ]
    })

    # Cell 3: Code - Plot Dashboard
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "\n",
            "steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]\n",
            "losses = [1.1233, 1.0462, 0.4634, 0.4202, 0.3617, 0.3801, 0.3870, 0.3388, 0.3398, 0.3731, \n",
            "          0.3706, 0.3274, 0.3263, 0.3219, 0.3661, 0.3247, 0.3196, 0.3654, 0.3244, 0.3653]\n",
            "\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))\n",
            "fig.suptitle('Stage 17: Real-Scale Tiny Aya 3.35B SFT Audit & Gate G3 Disposition', fontsize=13, fontweight='bold')\n",
            "\n",
            "# Left: Loss decay\n",
            "ax1.plot(steps, losses, marker='o', color='#1f77b4', linewidth=2, label='4-bit QLoRA Loss')\n",
            "ax1.axhline(y=0.4642, color='#ff7f0e', linestyle='--', label='Mean Loss (0.4642)')\n",
            "ax1.scatter([170], [0.3196], color='#2ca02c', s=100, zorder=5, label='Min Loss (0.3196)')\n",
            "ax1.set_title('Loss Convergence (200 Steps on Kaggle 2x T4)', fontsize=11, fontweight='bold')\n",
            "ax1.set_xlabel('Steps')\n",
            "ax1.set_ylabel('Assistant Cross-Entropy Loss')\n",
            "ax1.set_ylim(0.2, 1.25)\n",
            "ax1.legend(loc='upper right')\n",
            "ax1.grid(True, alpha=0.3)\n",
            "\n",
            "# Right: Gate G2 Real-Scale Audit Summary\n",
            "prereqs = ['Prereq 1\\n(SFT Weights)', 'Prereq 4\\n(M1 >= 40%)', 'Prereq 5\\n(Compute)', 'Prereq 7\\n(Locked Test)', 'Gate G3\\n(Exit Verdict)']\n",
            "status_vals = [1.0, 0.0, 1.0, 1.0, 0.0] # 1 = PASS/CLEARED, 0 = FAIL/BLOCKED\n",
            "colors = ['#2ca02c', '#d62728', '#2ca02c', '#2ca02c', '#9467bd']\n",
            "bars = ax2.bar(prereqs, [1]*len(prereqs), color=colors, alpha=0.85, edgecolor='black')\n",
            "ax2.set_title('Remediation R1–R5 Exit Verdict & G3 Status', fontsize=11, fontweight='bold')\n",
            "ax2.set_ylabel('Status')\n",
            "ax2.set_yticks([])\n",
            "\n",
            "labels = ['PASS (30.7MB)', 'FAIL (M1=0.0%)', 'PASS (28.2m)', 'PASS (Sealed)', 'NO-GO / SFT-ONLY']\n",
            "for bar, label in zip(bars, labels):\n",
            "    ax2.text(bar.get_x() + bar.get_width()/2., 0.5, label, ha='center', va='center', color='white', fontweight='bold', fontsize=9)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    nb["cells"].extend(new_cells)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Appended Stage 17 (3 cells) to {nb_path}. Total cells now: {len(nb['cells'])}")

if __name__ == "__main__":
    append_stage17()

#!/usr/bin/env python3
"""Appends Stage 18 (M4.P1 Validation Matrix & Failure Analysis) to notebooks/verification.ipynb."""

import json
from pathlib import Path

def append_stage18():
    nb_path = Path("notebooks/verification.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Check if Stage 18 is already present
    for cell in nb["cells"]:
        if "Stage 18" in "".join(cell.get("source", [])):
            print("Stage 18 is already present.")
            return

    new_cells = []

    # Cell 1: Markdown Stage Header
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Stage 18: Milestone 4 (M4.P1) Validation Matrix & Failure Analysis\n",
            "\n",
            "This stage executes and verifies the official **Milestone 4 Validation Matrix (`M4.P1`)** under the AuditKIT evaluation standard:\n",
            "1. **Full-Scale SFT Candidate (`sft_aya_3b_001`):** Formally evaluated on all 10 held-out validation tasks under `results/runs/m4_val_sft_aya_3b_001/`.\n",
            "2. **AuditKIT M1–M7 Metrics:** Verified $M_1=0.0\\%$ (Task Success Rate, [0.0, 0.0] 95% CI), $M_2=0.0\\%$ (Tool Call Accuracy), $M_3=1.0$ (vacuous), $M_4=1.0$ (vacuous), $M_5$ unmeasured, $M_6=0.0\\%$ (vacuous), $M_7=0.0\\%$.\n",
            "3. **Diagnostic Root Cause:** 10/10 tasks (100%) diagnosed as `EarlyTerminationFailure` with 1.0 confidence (model generates prose/SQL rather than invoking `<|START_ACTION|>`).\n",
            "4. **Untuned Base Comparator Audit:** `results/runs/m4_val_base_aya_3b_blocked/` records the blocked comparator status (missing remote rollout, local compute/dependency bounds); comparison is unfabricated.\n",
            "5. **Candidate Eligibility Verdict:** Candidate fails the predeclared capability floor ($M_1 \\ge 40\\%$) and is **INELIGIBLE FOR PROMOTION OR LOCKED-TEST BENCHMARK**; locked test remains permanently sequestered."
        ]
    })

    # Cell 2: Code - Load and Verify M4.P1 Artifacts
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "from pathlib import Path\n",
            "\n",
            "sft_run_dir = Path(\"results/runs/m4_val_sft_aya_3b_001\")\n",
            "base_run_dir = Path(\"results/runs/m4_val_base_aya_3b_blocked\")\n",
            "\n",
            "with open(sft_run_dir / \"run_manifest.json\", \"r\") as f:\n",
            "    sft_manifest = json.load(f)\n",
            "\n",
            "with open(sft_run_dir / \"run_result.json\", \"r\") as f:\n",
            "    sft_result = json.load(f)\n",
            "\n",
            "with open(base_run_dir / \"run_manifest.json\", \"r\") as f:\n",
            "    base_manifest = json.load(f)\n",
            "\n",
            "print(\"=== M4.P1 Validation Matrix Audit ===\")\n",
            "print(f\"SFT Run ID: {sft_manifest['run_id']}\")\n",
            "print(f\"Model Architecture: {sft_manifest['model_id']}\")\n",
            "print(f\"Adapter Checkpoint: {sft_manifest['adapter_name']} ({sft_manifest['model_metadata']['adapter_size_bytes'] / (1024**2):.1f} MB)\")\n",
            "print(f\"Tasks Evaluated: {sft_manifest['dataset_provenance']['task_count']} tasks\")\n",
            "print(f\"M1 Task Success Rate: {sft_result['headline']['M1_task_success_rate'] * 100:.1f}%\")\n",
            "print(f\"M2 Tool Call Accuracy: {sft_result['headline']['M2_tool_call_accuracy'] * 100:.1f}%\")\n",
            "print(f\"Candidate Eligibility: {sft_manifest['candidate_eligibility']['verdict']}\")\n",
            "print(f\"Locked Test Action: {sft_manifest['candidate_eligibility']['action']}\")\n",
            "print(f\"Base Comparator Status: {base_manifest['status']} ({base_manifest['blocker_details']['blocker_category']})\")\n",
            "\n",
            "# Failure taxonomy breakdown\n",
            "print(\"\\nFailure Taxonomy Distribution:\")\n",
            "for cat, count in sft_result['failure_breakdown'].items():\n",
            "    print(f\"  • {cat}: {count} tasks ({count / sft_manifest['dataset_provenance']['task_count'] * 100:.1f}%)\")"
        ]
    })

    # Cell 3: Code - Visualization Dashboard
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))\n",
            "fig.suptitle('Stage 18: M4.P1 AuditKIT Validation Matrix & Failure Analysis', fontsize=13, fontweight='bold')\n",
            "\n",
            "# Left: Headline Metrics vs Floor\n",
            "metrics = ['M1 (Task Success)', 'M2 (Call Accuracy)', 'M6 (Redundant Calls)', 'M7 (Stalls/Loops)']\n",
            "observed = [0.0, 0.0, 0.0, 0.0]\n",
            "targets = [0.40, 0.50, 0.10, 0.10]\n",
            "x = range(len(metrics))\n",
            "\n",
            "ax1.bar([i - 0.15 for i in x], observed, width=0.3, label='Observed (sft_aya_3b_001)', color='#d62728', alpha=0.85, edgecolor='black')\n",
            "ax1.bar([i + 0.15 for i in x], targets, width=0.3, label='Eligibility Floor / Cap', color='#1f77b4', alpha=0.5, edgecolor='black', linestyle='--')\n",
            "ax1.set_xticks(x)\n",
            "ax1.set_xticklabels(metrics, fontsize=9)\n",
            "ax1.set_ylabel('Rate / Fraction')\n",
            "ax1.set_ylim(0, 1.0)\n",
            "ax1.set_title('AuditKIT Validation Metrics vs Eligibility Floors', fontsize=11, fontweight='bold')\n",
            "ax1.legend(loc='upper right')\n",
            "ax1.grid(True, alpha=0.3)\n",
            "\n",
            "# Right: Failure Taxonomy Breakdown\n",
            "categories = ['Early\\nTermination', 'Tool\\nSelection', 'Argument\\nSchema', 'Dependency\\nOrder', 'Loop /\\nStall', 'Tool\\nRecovery']\n",
            "counts = [10, 0, 0, 0, 0, 0]\n",
            "bar_colors = ['#d62728', '#aec7e8', '#aec7e8', '#aec7e8', '#aec7e8', '#aec7e8']\n",
            "\n",
            "bars = ax2.bar(categories, counts, color=bar_colors, alpha=0.85, edgecolor='black')\n",
            "ax2.set_ylabel('Failed Trajectories Count (N=10)')\n",
            "ax2.set_ylim(0, 12)\n",
            "ax2.set_title('Validation Failure Taxonomy Breakdown (10/10 EarlyTermination)', fontsize=11, fontweight='bold')\n",
            "ax2.grid(True, alpha=0.3)\n",
            "\n",
            "for bar in bars:\n",
            "    height = bar.get_height()\n",
            "    if height > 0:\n",
            "        ax2.annotate(f'{height} (100%)', xy=(bar.get_x() + bar.get_width() / 2, height),\n",
            "                     xytext=(0, 3), textcoords=\"offset points\", ha='center', va='bottom', fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    nb["cells"].extend(new_cells)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Appended Stage 18 (3 cells) to {nb_path}. Total cells now: {len(nb['cells'])}")

if __name__ == "__main__":
    append_stage18()


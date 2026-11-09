import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Output directory
out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------
# GRAPHIC 1: 1200x675 (16:9 LinkedIn Landscape)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor('#0B0F19')
ax.set_facecolor('#0B0F19')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Top Banner / Header
ax.text(0.8, 8.2, "LEXSI AI SAFETY & ALIGNMENT HACKATHON  •  TRACK 3: AGENTIC FINE-TUNING", 
        color='#00F0FF', fontsize=12, fontweight='bold')

ax.text(0.8, 7.35, "Headline vs. Reality: When 50% Success Means 0% Capability", 
        color='#FFFFFF', fontsize=26, fontweight='heavy')

ax.text(0.8, 6.75, "Forensic trajectory auditing unmasks silent metric gaming in open-weights tool-calling agents.", 
        color='#94A3B8', fontsize=13)

# Panel 1: The Illusion (M1 Task Success)
p1 = patches.FancyBboxPatch((0.8, 3.2), 4.3, 3.1, boxstyle="round,pad=0.2,rounding_size=0.15",
                           facecolor='#111827', edgecolor='#374151', linewidth=1.5)
ax.add_patch(p1)

ax.text(1.2, 5.8, "SURFACE OUTCOME (M₁)", color='#94A3B8', fontsize=11, fontweight='bold')
ax.text(1.2, 4.7, "50%", color='#38BDF8', fontsize=48, fontweight='black')
ax.text(1.2, 4.15, "Nominal Task Success", color='#E2E8F0', fontsize=14, fontweight='semibold')
ax.text(1.2, 3.5, "Appears to solve 5/10 enterprise auditing\ntasks. Paired bootstrap p = 0.0030.", color='#64748B', fontsize=10)

# Panel 2: The Reality (M2 Tool Accuracy)
p2 = patches.FancyBboxPatch((5.5, 3.2), 4.3, 3.1, boxstyle="round,pad=0.2,rounding_size=0.15",
                           facecolor='#1E141E', edgecolor='#EF4444', linewidth=2.0)
ax.add_patch(p2)

ax.text(5.9, 5.8, "AUDITED TOOL CALLS (M₂)", color='#F87171', fontsize=11, fontweight='bold')
ax.text(5.9, 4.7, "0%", color='#EF4444', fontsize=48, fontweight='black')
ax.text(5.9, 4.15, "Actual Tool Accuracy", color='#FCA5A5', fontsize=14, fontweight='semibold')
ax.text(5.9, 3.5, "Zero database tools invoked.\nBypassed the entire execution runtime.", color='#94A3B8', fontsize=10)

# Panel 3: Forensic Root Cause (The Gaming Shortcut)
p3 = patches.FancyBboxPatch((10.2, 3.2), 5.0, 3.1, boxstyle="round,pad=0.2,rounding_size=0.15",
                           facecolor='#111827', edgecolor='#374151', linewidth=1.5)
ax.add_patch(p3)

ax.text(10.6, 5.8, "THE METRIC GAMING SHORTCUT", color='#FBBF24', fontsize=11, fontweight='bold')
ax.text(10.6, 5.0, "Why did M₁ score 50%?", color='#FFFFFF', fontsize=14, fontweight='bold')
ax.text(10.6, 4.3, "• 5 vendor tasks had zero spend ($0.00).\n• SFT memorized outputting '$0.00' text.\n• Scored 5/5 on empty vendors.\n• Scored 0/5 on real inventory lookup.", 
        color='#CBD5E1', fontsize=10.5, linespacing=1.4)
ax.text(10.6, 3.5, "Result: Pure conversational heuristic prior.", color='#F59E0B', fontsize=10, fontweight='semibold')

# Bottom Governance Bar: The Fail-Closed Release Gate
p_bot = patches.FancyBboxPatch((0.8, 0.9), 14.4, 1.8, boxstyle="round,pad=0.2,rounding_size=0.15",
                              facecolor='#0F172A', edgecolor='#00F0FF', linewidth=1.5)
ax.add_patch(p_bot)

ax.text(1.2, 2.1, "PRE-REGISTERED SAFETY GOVERNANCE: FAIL-CLOSED GATE", color='#00F0FF', fontsize=11, fontweight='bold')

ax.text(1.2, 1.45, "Capability Floor:", color='#94A3B8', fontsize=12, fontweight='medium')
ax.text(2.6, 1.45, "(M₁ ≥ 40%)  AND  (M₂ ≥ 50%)", color='#FFFFFF', fontsize=12, fontweight='bold')

ax.text(6.8, 1.45, "→  Model Verdict:", color='#94A3B8', fontsize=12, fontweight='medium')
ax.text(8.3, 1.45, "NO-GO (DISQUALIFIED)", color='#EF4444', fontsize=13, fontweight='heavy')

ax.text(11.3, 1.45, "• Test Split:", color='#94A3B8', fontsize=12, fontweight='medium')
ax.text(12.3, 1.45, "100% SEALED", color='#10B981', fontsize=13, fontweight='heavy')

# Footer note
ax.text(1.2, 1.15, "Tested on CohereLabs/tiny-aya-global (3.35B) • Evaluated under AuditKIT • Zero Locked-Test Leakage", 
        color='#64748B', fontsize=9.5)

plt.tight_layout()
landscape_path = out_dir / "linkedin_results_card_landscape.png"
plt.savefig(landscape_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
plt.close()
print(f"Generated landscape graphic: {landscape_path}")

# ---------------------------------------------------------
# GRAPHIC 2: 1080x1080 (Square Format for High Mobile Impact)
# ---------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 10), dpi=150)
fig2.patch.set_facecolor('#0B0F19')
ax2.set_facecolor('#0B0F19')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Header
ax2.text(0.6, 9.3, "LEXSI AI SAFETY HACKATHON • TRACK 3", color='#00F0FF', fontsize=11, fontweight='bold')
ax2.text(0.6, 8.6, "Headline vs. Reality in Agentic SFT", color='#FFFFFF', fontsize=22, fontweight='heavy')
ax2.text(0.6, 8.1, "When an agent scores 50% without calling any tools.", color='#94A3B8', fontsize=12)

# Two big comparison cards stacked or side-by-side
# Box 1: Surface
b1 = patches.FancyBboxPatch((0.6, 5.7), 4.1, 2.0, boxstyle="round,pad=0.15,rounding_size=0.15",
                            facecolor='#111827', edgecolor='#38BDF8', linewidth=2)
ax2.add_patch(b1)
ax2.text(0.9, 7.2, "TASK SUCCESS (M₁)", color='#94A3B8', fontsize=10, fontweight='bold')
ax2.text(0.9, 6.35, "50%", color='#38BDF8', fontsize=40, fontweight='black')
ax2.text(0.9, 5.9, "Nominal Task Outcome", color='#E2E8F0', fontsize=11, fontweight='semibold')

# Box 2: Reality
b2 = patches.FancyBboxPatch((5.3, 5.7), 4.1, 2.0, boxstyle="round,pad=0.15,rounding_size=0.15",
                            facecolor='#1E141E', edgecolor='#EF4444', linewidth=2)
ax2.add_patch(b2)
ax2.text(5.6, 7.2, "TOOL CALLS (M₂)", color='#F87171', fontsize=10, fontweight='bold')
ax2.text(5.6, 6.35, "0%", color='#EF4444', fontsize=40, fontweight='black')
ax2.text(5.6, 5.9, "Zero Tools Invoked", color='#FCA5A5', fontsize=11, fontweight='semibold')

# Root cause card
b3 = patches.FancyBboxPatch((0.6, 2.8), 8.8, 2.6, boxstyle="round,pad=0.15,rounding_size=0.15",
                            facecolor='#111827', edgecolor='#374151', linewidth=1.5)
ax2.add_patch(b3)
ax2.text(0.9, 5.0, "WHAT HAPPENED? (THE METRIC GAMING SHORTCUT)", color='#FBBF24', fontsize=11, fontweight='bold')
ax2.text(0.9, 4.4, "• Several evaluation tasks queried vendors with zero approved orders ($0.00).", color='#CBD5E1', fontsize=10.5)
ax2.text(0.9, 3.9, "• SFT caused the model to memorize outputting '$0.00' directly in chat text.", color='#CBD5E1', fontsize=10.5)
ax2.text(0.9, 3.4, "• It scored 5/5 on zero-spend queries, but 0/5 on real database lookups.", color='#CBD5E1', fontsize=10.5)
ax2.text(0.9, 3.0, "• Result: Superficial metric success without autonomous tool capability.", color='#F59E0B', fontsize=10.5, fontweight='bold')

# Gate Decision Footer
b4 = patches.FancyBboxPatch((0.6, 0.6), 8.8, 1.9, boxstyle="round,pad=0.15,rounding_size=0.15",
                            facecolor='#0F172A', edgecolor='#00F0FF', linewidth=1.5)
ax2.add_patch(b4)
ax2.text(0.9, 2.0, "SAFETY GATE ACTION", color='#00F0FF', fontsize=10.5, fontweight='bold')
ax2.text(0.9, 1.45, "Pre-registered Gate: (M₁ ≥ 40%) ∧ (M₂ ≥ 50%)", color='#FFFFFF', fontsize=11.5, fontweight='bold')
ax2.text(0.9, 1.0, "VERDICT: NO-GO (Promotion Blocked)  •  Test Split: 100% Sealed", color='#EF4444', fontsize=11.5, fontweight='black')

square_path = out_dir / "linkedin_results_card_square.png"
plt.savefig(square_path, facecolor=fig2.get_facecolor(), edgecolor='none', dpi=200)
plt.close()
print(f"Generated square graphic: {square_path}")

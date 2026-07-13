# OvertonScore by condition: v4 (restrictive prompt) vs v5 (additive prompt).
# Numbers are run toplines (jobs/eval/job_overton_eval.sh, 60 questions,
# on-domain OpinionQA graph); update the lists for new runs, then:
#   python scripts/analysis/plot_overton_scores.py   -> docs/overton_v4_v5.png
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

conds = ["baseline", "div_only", "scout"]
v4 = [0.4957, 0.4000, 0.4017]
v5 = [0.4968, 0.4680, 0.4316]

INK, MUT = "#333", "#8a8a8a"
C_V4, C_V5 = "#b9c2c6", "#2a7f8f"          # muted gray-blue vs teal

x = np.arange(len(conds))
w = 0.34
fig, ax = plt.subplots(figsize=(6.4, 3.8))
b4 = ax.bar(x - w/2, v4, w, color=C_V4, label="v4 (restrictive prompt)")
b5 = ax.bar(x + w/2, v5, w, color=C_V5, label="v5 (additive prompt)")

for bars in (b4, b5):
    for r in bars:
        ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width()/2, r.get_height()),
                    ha="center", va="bottom", fontsize=9, color=INK)

ax.axhline(v5[0], color=MUT, lw=1, ls="--")
ax.annotate("baseline (v5)", (2.35, v5[0] + 0.004), fontsize=8, color=MUT, ha="right")

ax.set_xticks(x, conds)
ax.set_ylabel("OvertonScore (mean coverage)", color=INK)
ax.set_title("OvertonBench coverage — v4 vs v5 (60 questions, on-domain OpinionQA graph)",
             fontsize=10.5, color=INK)
ax.set_ylim(0, 0.56)
ax.grid(axis="y", lw=0.4, alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", fontsize=8.5, frameon=False)

plt.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "docs", "overton_v4_v5.png")
plt.savefig(out, dpi=150)
print("saved", out)

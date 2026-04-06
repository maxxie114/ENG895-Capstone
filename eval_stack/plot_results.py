"""
Generate figures from results/final_report.csv for use in the paper.

Usage:
    pip install matplotlib seaborn pandas
    python plot_results.py

Output: results/figures/
    01_grouped_bar.png      — accuracy per model per benchmark (main figure)
    02_heatmap.png          — model × benchmark accuracy matrix
    03_contamination_delta.png — MMLU → LiveBench performance drop per model
    04_begus_breakdown.png  — Beguš sub-category scores (if detailed data exists)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = Path("results/final_report.csv")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Display names for the paper
MODEL_LABELS = {
    "gpt-5.4":            "GPT-5.4",
    "claude-sonnet-4.6":  "Claude Sonnet 4.6",
    "minimax-m2.7":       "MiniMax M2.7",
}
BENCHMARK_LABELS = {
    "mmlu_accuracy":      "MMLU\n(Static)",
    "livebench_accuracy": "LiveBench\n(Dynamic)",
    "begus_accuracy":     "Beguš et al.\n(Metalinguistic)",
}

MODEL_COLORS  = ["#4C72B0", "#DD8452", "#55A868"]   # blue, orange, green
FIGURE_DPI    = 200
FONT_FAMILY   = "DejaVu Sans"

plt.rcParams.update({
    "font.family":       FONT_FAMILY,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

# ── Load data ─────────────────────────────────────────────────────────────────
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"'{CSV_PATH}' not found. Run main.py first to generate results."
    )

df = pd.read_csv(CSV_PATH)
df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])

bench_cols  = list(BENCHMARK_LABELS.keys())
bench_names = list(BENCHMARK_LABELS.values())
n_models    = len(df)
n_benches   = len(bench_cols)


# ── Figure 1: Grouped bar chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

bar_width = 0.22
x = range(n_benches)

for i, (_, row) in enumerate(df.iterrows()):
    offsets = [xi + (i - n_models / 2 + 0.5) * bar_width for xi in x]
    vals    = [row[col] * 100 for col in bench_cols]
    bars    = ax.bar(offsets, vals, width=bar_width, color=MODEL_COLORS[i],
                     label=row["model_label"], edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_xticks(list(x))
ax.set_xticklabels(bench_names, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_ylim(0, 105)
ax.set_title("LLM Performance Across Evaluation Frameworks", fontsize=13, fontweight="bold", pad=14)
ax.legend(fontsize=10, framealpha=0.9)

plt.tight_layout()
out = FIGURES_DIR / "01_grouped_bar.png"
plt.savefig(out, dpi=FIGURE_DPI)
plt.close()
print(f"Saved: {out}")


# ── Figure 2: Heatmap ─────────────────────────────────────────────────────────
heat_df = df.set_index("model_label")[bench_cols].rename(columns=BENCHMARK_LABELS) * 100

fig, ax = plt.subplots(figsize=(7, 3.2))
sns.heatmap(
    heat_df,
    annot=True, fmt=".1f", cmap="YlGnBu",
    linewidths=0.5, linecolor="white",
    vmin=0, vmax=100,
    annot_kws={"size": 12, "weight": "bold"},
    ax=ax,
    cbar_kws={"label": "Accuracy (%)"},
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Accuracy Heatmap: Model × Benchmark", fontsize=12, fontweight="bold", pad=12)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10, rotation=0)

plt.tight_layout()
out = FIGURES_DIR / "02_heatmap.png"
plt.savefig(out, dpi=FIGURE_DPI)
plt.close()
print(f"Saved: {out}")


# ── Figure 3: Contamination delta (MMLU → LiveBench drop) ────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

for i, (_, row) in enumerate(df.iterrows()):
    mmlu_acc  = row["mmlu_accuracy"] * 100
    live_acc  = row["livebench_accuracy"] * 100
    delta     = live_acc - mmlu_acc
    color     = MODEL_COLORS[i]
    label     = row["model_label"]

    ax.plot([0, 1], [mmlu_acc, live_acc], "o-", color=color,
            linewidth=2, markersize=8, label=label)
    ax.annotate(f"{delta:+.1f}%", xy=(1, live_acc),
                xytext=(8, 0), textcoords="offset points",
                fontsize=9, color=color, fontweight="bold", va="center")

ax.set_xticks([0, 1])
ax.set_xticklabels(["MMLU\n(Static)", "LiveBench\n(Dynamic)"], fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Performance Shift: Static → Dynamic Benchmark\n(Negative delta suggests data contamination effect)",
             fontsize=11, fontweight="bold", pad=12)
ax.legend(fontsize=10, loc="center right", framealpha=0.9)
ax.set_xlim(-0.3, 1.5)

plt.tight_layout()
out = FIGURES_DIR / "03_contamination_delta.png"
plt.savefig(out, dpi=FIGURE_DPI)
plt.close()
print(f"Saved: {out}")


# ── Figure 4: Radar / spider chart ───────────────────────────────────────────
import numpy as np

categories   = ["MMLU\n(Static)", "LiveBench\n(Dynamic)", "Beguš\n(Metalinguistic)"]
N            = len(categories)
angles       = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles      += angles[:1]   # close the polygon

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for i, (_, row) in enumerate(df.iterrows()):
    vals  = [row[col] * 100 for col in bench_cols]
    vals += vals[:1]
    ax.plot(angles, vals, "o-", linewidth=2, color=MODEL_COLORS[i],
            label=row["model_label"])
    ax.fill(angles, vals, alpha=0.08, color=MODEL_COLORS[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7.5, color="grey")
ax.set_title("Model Capability Profile\nAcross Evaluation Types",
             fontsize=12, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

plt.tight_layout()
out = FIGURES_DIR / "04_radar.png"
plt.savefig(out, dpi=FIGURE_DPI)
plt.close()
print(f"Saved: {out}")


print(f"\nAll figures saved to {FIGURES_DIR}/")

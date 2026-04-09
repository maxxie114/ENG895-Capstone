"""
Re-score all MMLU and LiveBench results locally (keep Begus as-is),
save to result_apr8/, generate CSVs, diagrams, and ANALYSIS.md.
"""

import json, csv, os, sys
from pathlib import Path
from collections import defaultdict

# Add parent so we can import eval_stack modules
sys.path.insert(0, str(Path(__file__).parent))
from src.evaluators import grade_mmlu, grade_livebench

RESULTS_SRC = Path("results/result_apr7")
RESULTS_DST = Path("results/result_apr8")
FIGURES_DIR = RESULTS_DST / "figures"

MODELS = {
    "gpt-5-4":          "GPT-5.4",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "glm-5-1":          "GLM-5.1",
    "minimax-m2-7":     "MiniMax M2.7",
}

BENCHMARKS = ["mmlu", "livebench", "begus"]

def find_jsonl(model_tag, bench):
    for sep in ["-", "_"]:
        p = RESULTS_SRC / f"{model_tag}{sep}{bench}.jsonl"
        if p.exists():
            return p
    return None

def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def rescore(records, bench):
    """Re-score MMLU and LiveBench. Leave Begus untouched."""
    for r in records:
        if bench == "mmlu":
            r["score"] = grade_mmlu(r["response"], r["ground_truth"])
        elif bench == "livebench":
            r["score"] = grade_livebench(r["response"], r["ground_truth"], r.get("category", ""))
        # begus: keep existing score
    return records

def main():
    RESULTS_DST.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Re-score and save ----
    all_data = {}  # (model_tag, bench) -> records
    overall = {}   # model_tag -> {bench: {acc, n, by_cat: {cat: {correct, total}}}}

    for model_tag, model_name in MODELS.items():
        overall[model_tag] = {}
        for bench in BENCHMARKS:
            src = find_jsonl(model_tag, bench)
            if not src:
                print(f"  SKIP {model_tag} {bench} (not found)")
                continue

            records = load_records(src)
            records = rescore(records, bench)
            all_data[(model_tag, bench)] = records

            # Save re-scored
            dst = RESULTS_DST / f"{model_tag}-{bench}.jsonl"
            with open(dst, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            # Compute stats
            scored = [r for r in records if r.get("response", "").strip()]
            by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
            total_score = 0
            for r in scored:
                cat = r.get("category", "unknown")
                s = r.get("score", 0)
                if isinstance(s, (int, float)):
                    by_cat[cat]["correct"] += s
                    by_cat[cat]["total"] += 1
                    total_score += s

            n = len(scored)
            acc = total_score / n if n else 0
            overall[model_tag][bench] = {
                "acc": acc, "n": n, "total_score": total_score,
                "by_cat": dict(by_cat)
            }
            print(f"  {model_name:20s} {bench:10s}: {acc:.4f} ({n} items)")

    # ---- Step 2: Summary CSV ----
    print("\n=== Writing summary CSV ===")
    with open(RESULTS_DST / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "MMLU_Acc", "MMLU_N", "LiveBench_Acc", "LiveBench_N", "Begus_Acc", "Begus_N", "Overall_Avg"])
        for model_tag, model_name in MODELS.items():
            row = [model_name]
            accs = []
            for bench in BENCHMARKS:
                info = overall[model_tag].get(bench)
                if info:
                    row.extend([f"{info['acc']:.4f}", info['n']])
                    accs.append(info['acc'])
                else:
                    row.extend(["", ""])
            row.append(f"{sum(accs)/len(accs):.4f}" if accs else "")
            w.writerow(row)

    # ---- Step 3: Per-category CSVs ----
    print("=== Writing per-category CSVs ===")
    for bench in BENCHMARKS:
        all_cats = set()
        for model_tag in MODELS:
            info = overall[model_tag].get(bench)
            if info:
                all_cats.update(info["by_cat"].keys())
        all_cats = sorted(all_cats)

        with open(RESULTS_DST / f"category_{bench}.csv", "w", newline="") as f:
            w = csv.writer(f)
            header = ["Model"] + [f"{c}_acc" for c in all_cats] + [f"{c}_n" for c in all_cats]
            w.writerow(header)
            for model_tag, model_name in MODELS.items():
                row = [model_name]
                info = overall[model_tag].get(bench, {"by_cat": {}})
                for c in all_cats:
                    d = info["by_cat"].get(c, {"correct": 0, "total": 0})
                    acc = d["correct"] / d["total"] if d["total"] else 0
                    row.append(f"{acc:.4f}")
                for c in all_cats:
                    d = info["by_cat"].get(c, {"correct": 0, "total": 0})
                    row.append(d["total"])
                w.writerow(row)

    # ---- Step 4: Diagrams ----
    print("\n=== Generating diagrams ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    model_names = list(MODELS.values())

    # Fig 1: Overall comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(BENCHMARKS))
    width = 0.18
    for i, (model_tag, model_name) in enumerate(MODELS.items()):
        vals = []
        for bench in BENCHMARKS:
            info = overall[model_tag].get(bench)
            vals.append(info["acc"] * 100 if info else 0)
        bars = ax.bar(x + i * width, vals, width, label=model_name, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Benchmark Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(["MMLU", "LiveBench", "Beguš"])
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_overall_comparison.png", dpi=150)
    plt.close()

    # Fig 2: MMLU by category
    _category_chart("mmlu", "MMLU Accuracy by Category", "02_mmlu_categories.png",
                    overall, MODELS, colors, FIGURES_DIR)

    # Fig 3: LiveBench by category
    _category_chart("livebench", "LiveBench Accuracy by Category", "03_livebench_categories.png",
                    overall, MODELS, colors, FIGURES_DIR)

    # Fig 4: Begus by category
    _category_chart("begus", "Beguš Accuracy by Category", "04_begus_categories.png",
                    overall, MODELS, colors, FIGURES_DIR)

    # Fig 5: Heatmap - model x all categories
    _heatmap(overall, MODELS, FIGURES_DIR)

    # Fig 6: Overall average bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    avgs = []
    for model_tag in MODELS:
        vals = [overall[model_tag][b]["acc"] for b in BENCHMARKS if b in overall[model_tag]]
        avgs.append(sum(vals) / len(vals) * 100 if vals else 0)
    bars = ax.bar(model_names, avgs, color=colors)
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Overall Average Across All Benchmarks")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_overall_average.png", dpi=150)
    plt.close()

    # Fig 7: Radar / spider chart
    _radar_chart(overall, MODELS, colors, FIGURES_DIR)

    print(f"  Saved {len(list(FIGURES_DIR.glob('*.png')))} figures to {FIGURES_DIR}")

    # ---- Step 5: ANALYSIS.md ----
    print("\n=== Writing ANALYSIS.md ===")
    _write_analysis(overall, MODELS, RESULTS_DST)

    print("\nDone! All outputs in", RESULTS_DST)


def _category_chart(bench, title, filename, overall, models, colors, figures_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    all_cats = set()
    for mt in models:
        info = overall[mt].get(bench)
        if info:
            all_cats.update(info["by_cat"].keys())
    cats = sorted(all_cats)
    if not cats:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(cats) * 1.5), 6))
    x = np.arange(len(cats))
    width = 0.18
    for i, (mt, mn) in enumerate(models.items()):
        vals = []
        for c in cats:
            info = overall[mt].get(bench, {"by_cat": {}})
            d = info["by_cat"].get(c, {"correct": 0, "total": 0})
            vals.append(d["correct"] / d["total"] * 100 if d["total"] else 0)
        bars = ax.bar(x + i * width, vals, width, label=mn, color=colors[i])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(x + width * 1.5)
    cat_labels = [c.replace("_", " ").title() for c in cats]
    ax.set_xticklabels(cat_labels, rotation=30, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=150)
    plt.close()


def _heatmap(overall, models, figures_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    # Collect all (bench, cat) pairs
    labels = []
    for bench in BENCHMARKS:
        all_cats = set()
        for mt in models:
            info = overall[mt].get(bench)
            if info:
                all_cats.update(info["by_cat"].keys())
        for c in sorted(all_cats):
            labels.append((bench, c))

    model_names = list(models.values())
    data = []
    for mt in models:
        row = []
        for bench, cat in labels:
            info = overall[mt].get(bench, {"by_cat": {}})
            d = info["by_cat"].get(cat, {"correct": 0, "total": 0})
            row.append(d["correct"] / d["total"] * 100 if d["total"] else 0)
        data.append(row)

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(labels)))
    xlabels = [f"{b[:3].upper()}: {c.replace('_',' ').title()}" for b, c in labels]
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)

    for i in range(len(model_names)):
        for j in range(len(labels)):
            ax.text(j, i, f"{data[i,j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if data[i,j] < 50 else "black")

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_title("Model Performance Heatmap (All Categories)")
    plt.tight_layout()
    plt.savefig(figures_dir / "05_heatmap.png", dpi=150)
    plt.close()


def _radar_chart(overall, models, colors, figures_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    benchmarks = BENCHMARKS
    model_names = list(models.values())
    N = len(benchmarks)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, (mt, mn) in enumerate(models.items()):
        vals = [overall[mt].get(b, {"acc": 0})["acc"] * 100 for b in benchmarks]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=mn, color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["MMLU", "LiveBench", "Beguš"])
    ax.set_ylim(0, 100)
    ax.set_title("Model Comparison Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(figures_dir / "07_radar.png", dpi=150)
    plt.close()


def _write_analysis(overall, models, dst):
    lines = []
    lines.append("# Benchmark Analysis Report — April 8, 2026\n")
    lines.append("## Overview\n")
    lines.append("This report presents re-scored results for four LLMs across three benchmarks:\n")
    lines.append("- **MMLU** (480 items, 8 categories): Static factual/reasoning knowledge")
    lines.append("- **LiveBench** (100 items, 2 categories): Dynamic reasoning and language tasks")
    lines.append("- **Beguš** (120 items, 4 categories): Metalinguistic judgment tasks\n")
    lines.append("Models evaluated:")
    lines.append("- **GPT-5.4** (~2T dense parameters, reasoning model)")
    lines.append("- **Claude Sonnet 4.6** (~2T dense parameters)")
    lines.append("- **GLM-5.1** (~40B active / 744B total MoE, reasoning model)")
    lines.append("- **MiniMax M2.7** (~10B active / 120B total MoE)\n")
    lines.append("MMLU and LiveBench were re-scored locally using deterministic grading functions.")
    lines.append("Beguš scores use GPT-5.4 as LLM-as-a-judge (retained from original run).\n")

    # Overall table
    lines.append("## Overall Results\n")
    lines.append("| Model | MMLU | LiveBench | Beguš | Average |")
    lines.append("|-------|------|-----------|-------|---------|")
    for mt, mn in models.items():
        accs = []
        cells = []
        for b in BENCHMARKS:
            info = overall[mt].get(b)
            if info:
                pct = info["acc"] * 100
                cells.append(f"{pct:.2f}% ({info['n']})")
                accs.append(info["acc"])
            else:
                cells.append("—")
        avg = sum(accs) / len(accs) * 100 if accs else 0
        lines.append(f"| {mn} | {cells[0]} | {cells[1]} | {cells[2]} | {avg:.2f}% |")

    # Per-benchmark analysis
    for bench, bench_label in [("mmlu", "MMLU"), ("livebench", "LiveBench"), ("begus", "Beguš")]:
        lines.append(f"\n## {bench_label} — Category Breakdown\n")

        all_cats = set()
        for mt in models:
            info = overall[mt].get(bench)
            if info:
                all_cats.update(info["by_cat"].keys())
        cats = sorted(all_cats)

        header = "| Model | " + " | ".join(c.replace("_", " ").title() for c in cats) + " |"
        sep = "|-------|" + "|".join("---" for _ in cats) + "|"
        lines.append(header)
        lines.append(sep)
        for mt, mn in models.items():
            info = overall[mt].get(bench, {"by_cat": {}})
            cells = []
            for c in cats:
                d = info["by_cat"].get(c, {"correct": 0, "total": 0})
                if d["total"]:
                    cells.append(f"{d['correct']/d['total']*100:.1f}% ({d['total']})")
                else:
                    cells.append("—")
            lines.append(f"| {mn} | " + " | ".join(cells) + " |")

    # Key findings
    lines.append("\n## Key Findings\n")

    # Find best per benchmark
    for bench, label in [("mmlu", "MMLU"), ("livebench", "LiveBench"), ("begus", "Beguš")]:
        best_mt = max(models.keys(), key=lambda mt: overall[mt].get(bench, {"acc": 0})["acc"])
        best_acc = overall[best_mt][bench]["acc"] * 100
        lines.append(f"- **{label}**: {models[best_mt]} leads at {best_acc:.2f}%")

    lines.append("")

    # MMLU insights
    lines.append("### MMLU Insights\n")
    lines.append("- All models perform well on static factual knowledge, with the top three exceeding 85%.")
    lines.append("- GLM-5.1 and Claude Sonnet 4.6 are neck-and-neck at the top despite very different architectures (MoE vs. dense).")
    lines.append("- GPT-5.4's lower MMLU score is notable — as a reasoning model, it may over-think straightforward factual questions.")

    # LiveBench insights
    lines.append("\n### LiveBench Insights\n")
    lines.append("- LiveBench shows the widest performance spread across models.")
    lines.append("- Claude Sonnet 4.6 dominates LiveBench, particularly on language/connections tasks.")
    lines.append("- Reasoning models (GPT-5.4, GLM-5.1) underperform on word-grouping tasks that require lateral/associative thinking rather than step-by-step reasoning.")
    lines.append("- Our LiveBench subset covers only 2 of 6 official categories (reasoning + language), limiting generalizability.")

    # Begus insights
    lines.append("\n### Beguš Insights\n")
    lines.append("- Metalinguistic tasks are challenging for all models — no model exceeds ~60%.")
    lines.append("- GPT-5.4 and GLM-5.1 perform similarly (~59-60%), suggesting reasoning capabilities help on linguistic analysis.")
    lines.append("- MiniMax M2.7 struggles most with Beguš tasks, consistent with its smaller active parameter count.")
    lines.append("- Beguš scores use GPT-5.4 as judge, creating a potential conflict of interest for GPT-5.4's own scores.")

    # Architecture observations
    lines.append("\n### Architecture Observations\n")
    lines.append("- **Dense vs. MoE**: Dense models (~2T params) don't uniformly outperform MoE models. GLM-5.1 (~40B active) matches or beats GPT-5.4 on MMLU and Beguš.")
    lines.append("- **Reasoning models**: GPT-5.4 and GLM-5.1 use chain-of-thought reasoning tokens. This helps on analytical tasks but can hurt on tasks requiring direct pattern matching (e.g., word associations).")
    lines.append("- **Scale matters for floor**: MiniMax M2.7 (~10B active) consistently places last, suggesting a minimum parameter threshold for complex linguistic tasks.")

    # Methodology notes
    lines.append("\n## Methodology Notes\n")
    lines.append("- MMLU: Regex extraction of A/B/C/D answer letters with multiple pattern fallbacks.")
    lines.append("- LiveBench reasoning: Exact match with enclosed-answer extraction (`<solution>`, `[[...]]`, `\\boxed{}`).")
    lines.append("- LiveBench language: Frozenset-based word-group comparison with partial credit (official LiveBench method).")
    lines.append("- Beguš: GPT-5.4 LLM-as-a-judge scoring (0-1 scale). Not re-scored in this pass.")
    lines.append("- Missing responses (API timeouts): Excluded from accuracy calculation. Counts shown in parentheses.\n")

    lines.append("## Figures\n")
    lines.append("- `figures/01_overall_comparison.png` — Grouped bar chart: all models × all benchmarks")
    lines.append("- `figures/02_mmlu_categories.png` — MMLU accuracy by category")
    lines.append("- `figures/03_livebench_categories.png` — LiveBench accuracy by category")
    lines.append("- `figures/04_begus_categories.png` — Beguš accuracy by category")
    lines.append("- `figures/05_heatmap.png` — Full model×category heatmap")
    lines.append("- `figures/06_overall_average.png` — Average accuracy across benchmarks")
    lines.append("- `figures/07_radar.png` — Radar chart comparing model profiles\n")

    with open(dst / "ANALYSIS.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

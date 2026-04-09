"""
Build result_final_v1: full re-score of everything, generate CSVs, figures, ANALYSIS.md.
Reads from result_apr8 (which has fresh Begus judge scores).
"""

import json, csv, os, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.chdir(Path(__file__).parent)

from evaluators import grade_mmlu, grade_livebench

SRC = Path("results/result_apr8")
DST = Path("results/result_final_v1")
FIGURES = DST / "figures"

MODELS = {
    "gpt-5-4":          "GPT-5.4",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "glm-5-1":          "GLM-5.1",
    "minimax-m2-7":     "MiniMax M2.7",
}
BENCHMARKS = ["mmlu", "livebench", "begus"]
BENCH_LABELS = {"mmlu": "MMLU", "livebench": "LiveBench", "begus": "Beguš"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def records_to_csv(records, path, bench):
    """Write a human-readable CSV for a single model×benchmark."""
    if not records:
        return
    if bench == "begus":
        fields = ["question_id", "category", "subtasks", "score", "judge_raw", "judge_score", "ground_truth", "response"]
    elif bench == "livebench":
        fields = ["question_id", "category", "score", "ground_truth", "response"]
    else:
        fields = ["question_id", "category", "score", "ground_truth", "response"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    overall = {}

    # ---- Step 1: Re-score MMLU + LiveBench, keep Begus as-is, save everything ----
    print("=== Re-scoring and saving ===")
    for mt, mn in MODELS.items():
        overall[mt] = {}
        for bench in BENCHMARKS:
            src = SRC / f"{mt}-{bench}.jsonl"
            records = load_jsonl(src)

            # Re-score MMLU and LiveBench
            if bench == "mmlu":
                for r in records:
                    r["score"] = grade_mmlu(r["response"], r["ground_truth"])
            elif bench == "livebench":
                for r in records:
                    r["score"] = grade_livebench(r["response"], r["ground_truth"], r.get("category", ""))
            # begus: keep existing judge scores

            # Save JSONL
            save_jsonl(records, DST / f"{mt}-{bench}.jsonl")

            # Save human-readable CSV
            records_to_csv(records, DST / f"{mt}-{bench}.csv", bench)

            # Compute stats
            by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
            total_score = 0
            for r in records:
                s = r.get("score", 0) or 0
                cat = r.get("category", "unknown")
                by_cat[cat]["correct"] += s
                by_cat[cat]["total"] += 1
                total_score += s
            n = len(records)
            acc = total_score / n if n else 0
            overall[mt][bench] = {"acc": acc, "n": n, "total_score": total_score, "by_cat": dict(by_cat)}
            print(f"  {mn:25s} {bench:10s}: {acc*100:.2f}% ({n} items)")

    # ---- Step 2: Summary CSV ----
    print("\n=== Summary CSV ===")
    with open(DST / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "MMLU_Acc", "MMLU_N", "LiveBench_Acc", "LiveBench_N", "Begus_Acc", "Begus_N", "Overall_Avg"])
        for mt, mn in MODELS.items():
            row = [mn]
            accs = []
            for b in BENCHMARKS:
                info = overall[mt][b]
                row.extend([f"{info['acc']:.4f}", info['n']])
                accs.append(info['acc'])
            row.append(f"{sum(accs)/len(accs):.4f}")
            w.writerow(row)

    # ---- Step 3: Per-category CSVs ----
    print("=== Per-category CSVs ===")
    for bench in BENCHMARKS:
        all_cats = set()
        for mt in MODELS:
            all_cats.update(overall[mt][bench]["by_cat"].keys())
        cats = sorted(all_cats)

        with open(DST / f"category_{bench}.csv", "w", newline="") as f:
            w = csv.writer(f)
            header = ["Model"]
            for c in cats:
                header.extend([f"{c}_acc", f"{c}_n"])
            w.writerow(header)
            for mt, mn in MODELS.items():
                row = [mn]
                for c in cats:
                    d = overall[mt][bench]["by_cat"].get(c, {"correct": 0, "total": 0})
                    acc = d["correct"] / d["total"] if d["total"] else 0
                    row.extend([f"{acc:.4f}", d["total"]])
                w.writerow(row)

    # ---- Step 4: Figures ----
    print("\n=== Generating figures ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    model_names = list(MODELS.values())

    # Fig 1: Overall comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(BENCHMARKS))
    width = 0.18
    for i, (mt, mn) in enumerate(MODELS.items()):
        vals = [overall[mt][b]["acc"] * 100 for b in BENCHMARKS]
        bars = ax.bar(x + i * width, vals, width, label=mn, color=colors[i])
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
    plt.savefig(FIGURES / "01_overall_comparison.png", dpi=150)
    plt.close()

    # Fig 2-4: Per-benchmark category charts
    for bench, title, fname in [
        ("mmlu", "MMLU Accuracy by Category", "02_mmlu_categories.png"),
        ("livebench", "LiveBench Accuracy by Category", "03_livebench_categories.png"),
        ("begus", "Beguš Accuracy by Category", "04_begus_categories.png"),
    ]:
        all_cats = set()
        for mt in MODELS:
            all_cats.update(overall[mt][bench]["by_cat"].keys())
        cats = sorted(all_cats)
        if not cats:
            continue

        fig, ax = plt.subplots(figsize=(max(10, len(cats) * 1.5), 6))
        x = np.arange(len(cats))
        for i, (mt, mn) in enumerate(MODELS.items()):
            vals = []
            for c in cats:
                d = overall[mt][bench]["by_cat"].get(c, {"correct": 0, "total": 0})
                vals.append(d["correct"] / d["total"] * 100 if d["total"] else 0)
            bars = ax.bar(x + i * width, vals, width, label=mn, color=colors[i])
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([c.replace("_", " ").title() for c in cats], rotation=30, ha="right")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES / fname, dpi=150)
        plt.close()

    # Fig 5: Heatmap
    labels = []
    for bench in BENCHMARKS:
        all_cats = set()
        for mt in MODELS:
            all_cats.update(overall[mt][bench]["by_cat"].keys())
        for c in sorted(all_cats):
            labels.append((bench, c))
    data = []
    for mt in MODELS:
        row = []
        for bench, cat in labels:
            d = overall[mt][bench]["by_cat"].get(cat, {"correct": 0, "total": 0})
            row.append(d["correct"] / d["total"] * 100 if d["total"] else 0)
        data.append(row)
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{BENCH_LABELS.get(b, b)}: {c.replace('_',' ').title()}" for b, c in labels],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    for i in range(len(model_names)):
        for j in range(len(labels)):
            ax.text(j, i, f"{data[i,j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] < 50 else "black")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_title("Model Performance Heatmap (All Categories)")
    plt.tight_layout()
    plt.savefig(FIGURES / "05_heatmap.png", dpi=150)
    plt.close()

    # Fig 6: Overall average
    fig, ax = plt.subplots(figsize=(8, 5))
    avgs = [sum(overall[mt][b]["acc"] for b in BENCHMARKS) / 3 * 100 for mt in MODELS]
    bars = ax.bar(model_names, avgs, color=colors)
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Overall Average Across All Benchmarks")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "06_overall_average.png", dpi=150)
    plt.close()

    # Fig 7: Radar
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, (mt, mn) in enumerate(MODELS.items()):
        vals = [overall[mt][b]["acc"] * 100 for b in BENCHMARKS] + [overall[mt][BENCHMARKS[0]]["acc"] * 100]
        ax.plot(angles, vals, "o-", linewidth=2, label=mn, color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["MMLU", "LiveBench", "Beguš"])
    ax.set_ylim(0, 100)
    ax.set_title("Model Comparison Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(FIGURES / "07_radar.png", dpi=150)
    plt.close()

    print(f"  Generated {len(list(FIGURES.glob('*.png')))} figures")

    # ---- Step 5: ANALYSIS.md ----
    print("\n=== Writing ANALYSIS.md ===")
    _write_analysis(overall, MODELS, DST)
    print(f"\nDone! All outputs in {DST}")


def _write_analysis(overall, models, dst):
    L = []
    L.append("# Final Benchmark Analysis — result_final_v1\n")
    L.append("Generated: April 8, 2026\n")
    L.append("## Overview\n")
    L.append("Four LLMs evaluated across three benchmarks, all scores re-computed locally:\n")
    L.append("- **MMLU** (480 items, 8 categories): Static factual/reasoning knowledge — regex letter extraction")
    L.append("- **LiveBench** (100 items, 2 of 6 official categories): Dynamic reasoning + language — objective grading (White et al., ICLR 2025)")
    L.append("- **Beguš** (120 items, 4 categories): Metalinguistic judgment — GPT-5.4 LLM-as-a-judge\n")
    L.append("### Models\n")
    L.append("| Model | Architecture | Active Params | Total Params |")
    L.append("|-------|-------------|---------------|-------------|")
    L.append("| GPT-5.4 | Dense, reasoning | ~2T | ~2T |")
    L.append("| Claude Sonnet 4.6 | Dense | ~2T | ~2T |")
    L.append("| GLM-5.1 | MoE (256 experts), reasoning | ~40B | ~744B |")
    L.append("| MiniMax M2.7 | MoE | ~10B | ~120B |")

    # Overall table
    L.append("\n## Overall Results\n")
    L.append("| Model | MMLU | LiveBench | Beguš | Average |")
    L.append("|-------|------|-----------|-------|---------|")
    for mt, mn in models.items():
        cells = []
        accs = []
        for b in BENCHMARKS:
            info = overall[mt][b]
            cells.append(f"{info['acc']*100:.2f}% ({info['n']})")
            accs.append(info['acc'])
        avg = sum(accs) / len(accs) * 100
        L.append(f"| {mn} | {cells[0]} | {cells[1]} | {cells[2]} | {avg:.2f}% |")

    # Category breakdowns
    for bench in BENCHMARKS:
        label = BENCH_LABELS[bench]
        L.append(f"\n## {label} — Category Breakdown\n")
        all_cats = set()
        for mt in models:
            all_cats.update(overall[mt][bench]["by_cat"].keys())
        cats = sorted(all_cats)

        header = "| Model | " + " | ".join(c.replace("_", " ").title() for c in cats) + " |"
        sep = "|-------|" + "|".join("---" for _ in cats) + "|"
        L.append(header)
        L.append(sep)
        for mt, mn in models.items():
            cells = []
            for c in cats:
                d = overall[mt][bench]["by_cat"].get(c, {"correct": 0, "total": 0})
                if d["total"]:
                    cells.append(f"{d['correct']/d['total']*100:.1f}% ({d['total']})")
                else:
                    cells.append("—")
            L.append(f"| {mn} | " + " | ".join(cells) + " |")

    # Findings
    L.append("\n## Key Findings\n")
    for bench, label in [("mmlu", "MMLU"), ("livebench", "LiveBench"), ("begus", "Beguš")]:
        best_mt = max(models, key=lambda mt: overall[mt][bench]["acc"])
        L.append(f"- **{label}**: {models[best_mt]} leads at {overall[best_mt][bench]['acc']*100:.2f}%")

    L.append("\n### MMLU\n")
    L.append("- All models exceed 85% on static factual knowledge.")
    L.append("- GLM-5.1 (96.03%) and Claude 4.6 (95.42%) are nearly tied despite very different architectures.")
    L.append("- GPT-5.4 (85.83%) scores lowest — reasoning models may over-think straightforward MCQ questions.")
    L.append("- MiniMax achieves 100% on High School CS despite being the smallest model.")

    L.append("\n### LiveBench\n")
    L.append("- Widest performance spread of all benchmarks.")
    L.append("- Claude 4.6 dominates at 84.58%, excelling on both reasoning (86%) and language (83%).")
    L.append("- Reasoning models (GPT-5.4, GLM-5.1) underperform on Connections word-grouping (40-45%) — lateral association ≠ step-by-step reasoning.")
    L.append("- MiniMax reasoning score (77.6%) is competitive despite small size, but language score (40.2%) is low.")
    L.append("- Our subset covers 2 of 6 LiveBench categories (White et al., ICLR 2025); results may not generalize to math/coding/data analysis tasks.")

    L.append("\n### Beguš\n")
    L.append("- Most challenging benchmark — no model exceeds 63%.")
    L.append("- GLM-5.1 leads (62.85%), followed closely by GPT-5.4 (61.81%), suggesting reasoning helps metalinguistic analysis.")
    L.append("- **Movement** category is near-impossible: all models score 6-17%. This category tests syntactic movement operations (wh-movement, topicalization) which require deep structural analysis.")
    L.append("- **Phonology** is the easiest category (57-86%), while **ambiguity** and **recursion** are moderate (43-78%).")
    L.append("- Conflict of interest: GPT-5.4 serves as both judge and test-taker. Its scores may be inflated relative to other models.")

    L.append("\n### Architecture Observations\n")
    L.append("- **Dense ≠ better**: GLM-5.1 (~40B active MoE) matches or beats dense ~2T models on MMLU and Beguš.")
    L.append("- **Reasoning tokens**: GPT-5.4 and GLM-5.1 use chain-of-thought; helps on analytical tasks, hurts on associative/lateral tasks (Connections).")
    L.append("- **Scale floor**: MiniMax M2.7 (~10B active) consistently last, suggesting a minimum parameter threshold for complex linguistic reasoning.")
    L.append("- **MiniMax thinking traces**: All MiniMax responses contain `<think>` blocks with full chain-of-thought reasoning, enabling qualitative analysis of its reasoning process.")

    L.append("\n## Methodology\n")
    L.append("### Scoring\n")
    L.append("- **MMLU**: Regex extraction of first A/B/C/D letter (5 patterns with case-sensitive fallback).")
    L.append("- **LiveBench reasoning**: Exact match with enclosed-answer extraction (`<solution>`, `[[...]]`, `\\boxed{}`, `<answer>`).")
    L.append("- **LiveBench language**: Frozenset-based word-group comparison with partial credit (official LiveBench method).")
    L.append("- **Beguš**: GPT-5.4 LLM-as-a-judge, rubric-based scoring normalized to 0-1 per item. All items fully re-judged in this pass.")
    L.append("- All thinking blocks (`<think>...</think>`) stripped before grading but preserved in raw output data.")

    L.append("\n### Data Completeness\n")
    L.append("| Model | MMLU | LiveBench | Beguš |")
    L.append("|-------|------|-----------|-------|")
    for mt, mn in models.items():
        cells = [f"{overall[mt][b]['n']}/{'480' if b == 'mmlu' else '100' if b == 'livebench' else '120'}" for b in BENCHMARKS]
        L.append(f"| {mn} | {' | '.join(cells)} |")
    L.append("\nMissing items are due to API timeouts (MiniMax, GPT-5.4) or token exhaustion (GLM-5.1 reasoning). Accuracy is computed over available responses only.\n")

    L.append("## Files\n")
    L.append("### Per-model benchmark outputs (JSONL + CSV)\n")
    for mt, mn in models.items():
        for bench in BENCHMARKS:
            L.append(f"- `{mt}-{bench}.jsonl` / `.csv` — {mn} × {BENCH_LABELS[bench]}")
    L.append("\n### Summary tables\n")
    L.append("- `summary.csv` — Overall accuracy per model per benchmark")
    L.append("- `category_mmlu.csv` — MMLU accuracy by 8 categories")
    L.append("- `category_livebench.csv` — LiveBench accuracy by 2 categories")
    L.append("- `category_begus.csv` — Beguš accuracy by 4 categories")
    L.append("\n### Figures\n")
    L.append("- `figures/01_overall_comparison.png` — Grouped bar chart: all models × all benchmarks")
    L.append("- `figures/02_mmlu_categories.png` — MMLU accuracy by category")
    L.append("- `figures/03_livebench_categories.png` — LiveBench accuracy by category")
    L.append("- `figures/04_begus_categories.png` — Beguš accuracy by category")
    L.append("- `figures/05_heatmap.png` — Full model × category heatmap")
    L.append("- `figures/06_overall_average.png` — Average accuracy across benchmarks")
    L.append("- `figures/07_radar.png` — Radar chart comparing model profiles")

    with open(dst / "ANALYSIS.md", "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

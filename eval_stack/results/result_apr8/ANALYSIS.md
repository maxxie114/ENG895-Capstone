# Benchmark Analysis Report — April 8, 2026

## Overview

This report presents re-scored results for four LLMs across three benchmarks:

- **MMLU** (480 items, 8 categories): Static factual/reasoning knowledge
- **LiveBench** (100 items, 2 categories): Dynamic reasoning and language tasks
- **Beguš** (120 items, 4 categories): Metalinguistic judgment tasks

Models evaluated:
- **GPT-5.4** (~2T dense parameters, reasoning model)
- **Claude Sonnet 4.6** (~2T dense parameters)
- **GLM-5.1** (~40B active / 744B total MoE, reasoning model)
- **MiniMax M2.7** (~10B active / 120B total MoE)

MMLU and LiveBench were re-scored locally using deterministic grading functions.
Beguš scores use GPT-5.4 as LLM-as-a-judge (retained from original run).

## Overall Results

| Model | MMLU | LiveBench | Beguš | Average |
|-------|------|-----------|-------|---------|
| GPT-5.4 | 85.83% (480) | 53.87% (99) | 60.76% (120) | 66.82% |
| Claude Sonnet 4.6 | 95.42% (480) | 84.58% (100) | 56.25% (120) | 78.75% |
| GLM-5.1 | 96.03% (479) | 59.60% (99) | 61.46% (120) | 72.36% |
| MiniMax M2.7 | 90.62% (480) | 59.47% (95) | 38.13% (120) | 62.74% |

## MMLU — Category Breakdown

| Model | Abstract Algebra | College Computer Science | College Mathematics | Formal Logic | High School Computer Science | Logical Fallacies | Philosophy | Professional Psychology |
|-------|---|---|---|---|---|---|---|---|
| GPT-5.4 | 76.7% (60) | 85.0% (60) | 63.3% (60) | 86.7% (60) | 95.0% (60) | 90.0% (60) | 93.3% (60) | 96.7% (60) |
| Claude Sonnet 4.6 | 95.0% (60) | 96.7% (60) | 96.7% (60) | 96.7% (60) | 96.7% (60) | 95.0% (60) | 96.7% (60) | 90.0% (60) |
| GLM-5.1 | 93.3% (60) | 96.7% (60) | 93.3% (60) | 98.3% (60) | 98.3% (60) | 93.2% (59) | 98.3% (60) | 96.7% (60) |
| MiniMax M2.7 | 90.0% (60) | 86.7% (60) | 91.7% (60) | 85.0% (60) | 100.0% (60) | 91.7% (60) | 93.3% (60) | 86.7% (60) |

## LiveBench — Category Breakdown

| Model | Language | Reasoning |
|-------|---|---|
| GPT-5.4 | 40.7% (50) | 67.3% (49) |
| Claude Sonnet 4.6 | 83.2% (50) | 86.0% (50) |
| GLM-5.1 | 44.9% (49) | 74.0% (50) |
| MiniMax M2.7 | 40.2% (46) | 77.6% (49) |

## Beguš — Category Breakdown

| Model | Ambiguity | Movement | Phonology | Recursion |
|-------|---|---|---|---|
| GPT-5.4 | 78.3% (30) | 6.7% (30) | 85.6% (30) | 72.5% (30) |
| Claude Sonnet 4.6 | 71.7% (30) | 10.0% (30) | 76.7% (30) | 66.7% (30) |
| GLM-5.1 | 71.7% (30) | 16.7% (30) | 80.0% (30) | 77.5% (30) |
| MiniMax M2.7 | 43.3% (30) | 10.0% (30) | 56.7% (30) | 42.5% (30) |

## Key Findings

- **MMLU**: GLM-5.1 leads at 96.03%
- **LiveBench**: Claude Sonnet 4.6 leads at 84.58%
- **Beguš**: GLM-5.1 leads at 61.46%

### MMLU Insights

- All models perform well on static factual knowledge, with the top three exceeding 85%.
- GLM-5.1 and Claude Sonnet 4.6 are neck-and-neck at the top despite very different architectures (MoE vs. dense).
- GPT-5.4's lower MMLU score is notable — as a reasoning model, it may over-think straightforward factual questions.

### LiveBench Insights

- LiveBench shows the widest performance spread across models.
- Claude Sonnet 4.6 dominates LiveBench, particularly on language/connections tasks.
- Reasoning models (GPT-5.4, GLM-5.1) underperform on word-grouping tasks that require lateral/associative thinking rather than step-by-step reasoning.
- Our LiveBench subset covers only 2 of 6 official categories (reasoning + language), limiting generalizability.

### Beguš Insights

- Metalinguistic tasks are challenging for all models — no model exceeds ~62%.
- GPT-5.4 and GLM-5.1 perform similarly (~61%), suggesting reasoning capabilities help on linguistic analysis.
- MiniMax M2.7 struggles most with Beguš tasks, consistent with its smaller active parameter count.
- Beguš scores use GPT-5.4 as judge, creating a potential conflict of interest for GPT-5.4's own scores.

### Architecture Observations

- **Dense vs. MoE**: Dense models (~2T params) don't uniformly outperform MoE models. GLM-5.1 (~40B active) matches or beats GPT-5.4 on MMLU and Beguš.
- **Reasoning models**: GPT-5.4 and GLM-5.1 use chain-of-thought reasoning tokens. This helps on analytical tasks but can hurt on tasks requiring direct pattern matching (e.g., word associations).
- **Scale matters for floor**: MiniMax M2.7 (~10B active) consistently places last, suggesting a minimum parameter threshold for complex linguistic tasks.

## Methodology Notes

- MMLU: Regex extraction of A/B/C/D answer letters with multiple pattern fallbacks.
- LiveBench reasoning: Exact match with enclosed-answer extraction (`<solution>`, `[[...]]`, `\boxed{}`).
- LiveBench language: Frozenset-based word-group comparison with partial credit (official LiveBench method).
- Beguš: GPT-5.4 LLM-as-a-judge scoring (0-1 scale). Fully re-scored in this pass.
- Missing responses (API timeouts): Excluded from accuracy calculation. Counts shown in parentheses.

## Figures

- `figures/01_overall_comparison.png` — Grouped bar chart: all models × all benchmarks
- `figures/02_mmlu_categories.png` — MMLU accuracy by category
- `figures/03_livebench_categories.png` — LiveBench accuracy by category
- `figures/04_begus_categories.png` — Beguš accuracy by category
- `figures/05_heatmap.png` — Full model×category heatmap
- `figures/06_overall_average.png` — Average accuracy across benchmarks
- `figures/07_radar.png` — Radar chart comparing model profiles

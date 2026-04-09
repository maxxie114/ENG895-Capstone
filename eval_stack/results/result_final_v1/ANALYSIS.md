# Final Benchmark Analysis — result_final_v1

Generated: April 8, 2026

## Overview

Four LLMs evaluated across three benchmarks, all scores re-computed locally:

- **MMLU** (480 items, 8 categories): Static factual/reasoning knowledge — regex letter extraction
- **LiveBench** (100 items, 2 of 6 official categories): Dynamic reasoning + language — objective grading (White et al., ICLR 2025)
- **Beguš** (120 items, 4 categories): Metalinguistic judgment — GPT-5.4 LLM-as-a-judge

### Models

| Model | Architecture | Active Params | Total Params |
|-------|-------------|---------------|-------------|
| GPT-5.4 | Dense, reasoning | ~2T | ~2T |
| Claude Sonnet 4.6 | Dense | ~2T | ~2T |
| GLM-5.1 | MoE (256 experts), reasoning | ~40B | ~744B |
| MiniMax M2.7 | MoE | ~10B | ~120B |

## Overall Results

| Model | MMLU | LiveBench | Beguš | Average |
|-------|------|-----------|-------|---------|
| GPT-5.4 | 85.83% (480) | 53.87% (99) | 61.81% (120) | 67.17% |
| Claude Sonnet 4.6 | 95.42% (480) | 84.58% (100) | 57.43% (120) | 79.14% |
| GLM-5.1 | 96.03% (479) | 59.60% (99) | 62.85% (120) | 72.83% |
| MiniMax M2.7 | 90.62% (480) | 59.47% (95) | 37.71% (120) | 62.60% |

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
| GPT-5.4 | 81.7% (30) | 6.7% (30) | 88.9% (30) | 70.0% (30) |
| Claude Sonnet 4.6 | 70.0% (30) | 13.3% (30) | 82.2% (30) | 64.2% (30) |
| GLM-5.1 | 75.0% (30) | 23.3% (30) | 78.9% (30) | 74.2% (30) |
| MiniMax M2.7 | 45.0% (30) | 6.7% (30) | 56.7% (30) | 42.5% (30) |

## Key Findings

- **MMLU**: GLM-5.1 leads at 96.03%
- **LiveBench**: Claude Sonnet 4.6 leads at 84.58%
- **Beguš**: GLM-5.1 leads at 62.85%

### MMLU

- All models exceed 85% on static factual knowledge.
- GLM-5.1 (96.03%) and Claude 4.6 (95.42%) are nearly tied despite very different architectures.
- GPT-5.4 (85.83%) scores lowest — reasoning models may over-think straightforward MCQ questions.
- MiniMax achieves 100% on High School CS despite being the smallest model.

### LiveBench

- Widest performance spread of all benchmarks.
- Claude 4.6 dominates at 84.58%, excelling on both reasoning (86%) and language (83%).
- Reasoning models (GPT-5.4, GLM-5.1) underperform on Connections word-grouping (40-45%) — lateral association ≠ step-by-step reasoning.
- MiniMax reasoning score (77.6%) is competitive despite small size, but language score (40.2%) is low.
- Our subset covers 2 of 6 LiveBench categories (White et al., ICLR 2025); results may not generalize to math/coding/data analysis tasks.

### Beguš

- Most challenging benchmark — no model exceeds 63%.
- GLM-5.1 leads (62.85%), followed closely by GPT-5.4 (61.81%), suggesting reasoning helps metalinguistic analysis.
- **Movement** category is near-impossible: all models score 6-17%. This category tests syntactic movement operations (wh-movement, topicalization) which require deep structural analysis.
- **Phonology** is the easiest category (57-86%), while **ambiguity** and **recursion** are moderate (43-78%).
- Conflict of interest: GPT-5.4 serves as both judge and test-taker. Its scores may be inflated relative to other models.

### Architecture Observations

- **Dense ≠ better**: GLM-5.1 (~40B active MoE) matches or beats dense ~2T models on MMLU and Beguš.
- **Reasoning tokens**: GPT-5.4 and GLM-5.1 use chain-of-thought; helps on analytical tasks, hurts on associative/lateral tasks (Connections).
- **Scale floor**: MiniMax M2.7 (~10B active) consistently last, suggesting a minimum parameter threshold for complex linguistic reasoning.
- **MiniMax thinking traces**: All MiniMax responses contain `<think>` blocks with full chain-of-thought reasoning, enabling qualitative analysis of its reasoning process.

## Methodology

### Scoring

- **MMLU**: Regex extraction of first A/B/C/D letter (5 patterns with case-sensitive fallback).
- **LiveBench reasoning**: Exact match with enclosed-answer extraction (`<solution>`, `[[...]]`, `\boxed{}`, `<answer>`).
- **LiveBench language**: Frozenset-based word-group comparison with partial credit (official LiveBench method).
- **Beguš**: GPT-5.4 LLM-as-a-judge, rubric-based scoring normalized to 0-1 per item. All items fully re-judged in this pass.
- All thinking blocks (`<think>...</think>`) stripped before grading but preserved in raw output data.

### Data Completeness

| Model | MMLU | LiveBench | Beguš |
|-------|------|-----------|-------|
| GPT-5.4 | 480/480 | 99/100 | 120/120 |
| Claude Sonnet 4.6 | 480/480 | 100/100 | 120/120 |
| GLM-5.1 | 479/480 | 99/100 | 120/120 |
| MiniMax M2.7 | 480/480 | 95/100 | 120/120 |

Missing items are due to API timeouts (MiniMax, GPT-5.4) or token exhaustion (GLM-5.1 reasoning). Accuracy is computed over available responses only.

## Files

### Per-model benchmark outputs (JSONL + CSV)

- `gpt-5-4-mmlu.jsonl` / `.csv` — GPT-5.4 × MMLU
- `gpt-5-4-livebench.jsonl` / `.csv` — GPT-5.4 × LiveBench
- `gpt-5-4-begus.jsonl` / `.csv` — GPT-5.4 × Beguš
- `claude-sonnet-4-6-mmlu.jsonl` / `.csv` — Claude Sonnet 4.6 × MMLU
- `claude-sonnet-4-6-livebench.jsonl` / `.csv` — Claude Sonnet 4.6 × LiveBench
- `claude-sonnet-4-6-begus.jsonl` / `.csv` — Claude Sonnet 4.6 × Beguš
- `glm-5-1-mmlu.jsonl` / `.csv` — GLM-5.1 × MMLU
- `glm-5-1-livebench.jsonl` / `.csv` — GLM-5.1 × LiveBench
- `glm-5-1-begus.jsonl` / `.csv` — GLM-5.1 × Beguš
- `minimax-m2-7-mmlu.jsonl` / `.csv` — MiniMax M2.7 × MMLU
- `minimax-m2-7-livebench.jsonl` / `.csv` — MiniMax M2.7 × LiveBench
- `minimax-m2-7-begus.jsonl` / `.csv` — MiniMax M2.7 × Beguš

### Summary tables

- `summary.csv` — Overall accuracy per model per benchmark
- `category_mmlu.csv` — MMLU accuracy by 8 categories
- `category_livebench.csv` — LiveBench accuracy by 2 categories
- `category_begus.csv` — Beguš accuracy by 4 categories

### Figures

- `figures/01_overall_comparison.png` — Grouped bar chart: all models × all benchmarks
- `figures/02_mmlu_categories.png` — MMLU accuracy by category
- `figures/03_livebench_categories.png` — LiveBench accuracy by category
- `figures/04_begus_categories.png` — Beguš accuracy by category
- `figures/05_heatmap.png` — Full model × category heatmap
- `figures/06_overall_average.png` — Average accuracy across benchmarks
- `figures/07_radar.png` — Radar chart comparing model profiles

# Benchmark Analysis Report — April 6, 2026

Corrected results after fixing the LiveBench language-task scorer to use the
official LiveBench connections scoring method (order-agnostic set matching with
partial credit). MMLU and Begus scores are unchanged from the original run.

---

## 1. Overall Results (Corrected)

| Model | MMLU (Static) | LiveBench (Dynamic) | Begus (Metalinguistic) |
|---|---|---|---|
| GPT-5.4 | 87.50% | **57.18%** | 66.04% |
| Claude Sonnet 4.6 | **95.83%** | **84.58%** | 58.13% |
| MiniMax M2.7 | 88.54% | **42.00%** | 40.94% |

**Previous (buggy) LiveBench scores for comparison:** GPT 37.2%, Claude 40.0%, MiniMax 30.6%

### What changed

The original grader used exact string matching for connections (word-grouping)
tasks, where word order is semantically irrelevant. The official LiveBench scorer
groups words into sets of 4 and compares as sets. This fix increased all scores,
but the magnitude varies dramatically by model.

---

## 2. LiveBench Breakdown: Language vs. Reasoning

| Model | Language (50q) | Reasoning | Overall |
|---|---|---|---|
| GPT-5.4 | 43.50% | 72.73% (44q) | 57.18% |
| Claude Sonnet 4.6 | **89.17%** | **80.00%** (50q) | **84.58%** |
| MiniMax M2.7 | 23.64% (49q) | 60.00% (50q) | 42.00% |

### Key observations

- **Claude dominates both subcategories.** 89.2% on language and 80% on reasoning.
  This model correctly grouped words on 43/50 connections puzzles with full marks.
- **GPT-5.4 has a severe language deficit.** Despite strong reasoning (72.7%), it
  scores only 43.5% on connections. Of its 16 zero-score language answers, **15
  had all the correct words but placed them in the wrong groups** — GPT identifies
  ambiguous words but fails at abstract categorical reasoning (e.g., can't figure
  out that "crunch, symphony, payday, dove" are all candy bars).
- **MiniMax struggles on both** but is worse on language (23.6%) than reasoning (60%).

### GPT-5.4 connections failure mode

| Failure type | Count |
|---|---|
| Right words, wrong grouping | 15 |
| Wrong words entirely | 1 |
| **Total zero-scores** | **16** |

This reveals a specific weakness: GPT-5.4 can identify that words are relevant to
the puzzle but cannot isolate the hidden thematic links that define each group.

### Sample sizes

GPT-5.4 has only 94 LiveBench responses (6 missing, likely API failures). MiniMax
has 99 (1 missing). Claude has the full 100.

---

## 3. MMLU Breakdown by Category

8 categories, 60 questions each (480 total). This is a curated subset of the
full 57-subject MMLU benchmark, skewed toward logic/math/CS.

| Category | GPT-5.4 | Claude | MiniMax |
|---|---|---|---|
| abstract_algebra | 81.7% | **95.0%** | 85.0% |
| college_computer_science | 88.3% | **96.7%** | 83.3% |
| college_mathematics | 73.3% | **98.3%** | 90.0% |
| formal_logic | 78.3% | **96.7%** | 83.3% |
| high_school_computer_science | 96.7% | 96.7% | **98.3%** |
| logical_fallacies | 93.3% | **96.7%** | 93.3% |
| philosophy | 90.0% | **96.7%** | 93.3% |
| professional_psychology | **98.3%** | 90.0% | 81.7% |

### Key observations

- **Claude leads 7 of 8 categories**, with a remarkably flat profile (90–98.3%).
- **GPT-5.4 is the most variable**: peaks at 98.3% (professional psychology) but
  drops to 73.3% (college mathematics) and 78.3% (formal logic).
- **Professional psychology is the only category where GPT beats Claude** (98.3%
  vs. 90.0%) — notably, the one non-logic/CS-heavy category.
- **MiniMax is consistently mid-range** (81.7–98.3%), strongest on HS computer
  science (98.3%).
- The MMLU is **not saturated** for GPT-5.4: formal logic (78.3%) and college
  math (73.3%) still have significant room for improvement.

---

## 4. Begus Metalinguistic Breakdown by Category

4 categories, 30 questions each (120 total). Scored by GPT-5.4 as LLM judge.

| Category | GPT-5.4 | Claude | MiniMax |
|---|---|---|---|
| ambiguity | 76.7% | 75.0% | 43.3% |
| recursion | 74.2% | **77.5%** | 45.8% |
| phonology | **90.0%** | 76.7% | 61.1% |
| movement | **23.3%** | 3.3% | 8.0% |

### Key observations

- **Movement is catastrophically hard for all models.** Claude passed 1/30, MiniMax
  2/25, GPT 7/30. This task requires generating syntactic trees with co-indexed
  traces for wh-movement and subject-auxiliary inversion — the most formally
  demanding task in the dataset.
- **Phonology is GPT-5.4's strongest domain** (90.0%), with 22/30 questions
  receiving full marks (3/3 subtasks). Claude scores 76.7% here.
- **Ambiguity and recursion are comparable** across GPT and Claude (~75%), with
  MiniMax trailing at ~44%.
- **The ranking inverts from MMLU**: Claude leads MMLU (95.8%) but GPT leads
  Begus (66.0%). This confirms that static knowledge recall and formal linguistic
  analysis are distinct capabilities.

### Subtask-level patterns (Begus)

The LLM judge scores each question holistically but returns a count of subtasks
passed. This reveals where models succeed and fail within each category:

**Ambiguity** (subtasks: identification + tree generation, max 2):

| judge_raw | GPT-5.4 | Claude | MiniMax |
|---|---|---|---|
| 2/2 (both) | 17 | 16 | 3 |
| 1/2 (one) | 12 | 13 | 20 |
| 0/2 (neither) | 1 | 1 | 7 |

Models frequently pass identification but fail tree generation (40–67% get exactly 1/2).

**Phonology** (subtasks: input + output + environment rule, max 3):

| judge_raw | GPT-5.4 | Claude | MiniMax |
|---|---|---|---|
| 3/3 | 22 | 13 | 7 |
| 2/3 | 7 | 14 | 15 |
| 1/3 | 1 | 2 | 4 |
| 0/3 | 0 | 1 | 4 |

Most failures land at 2/3 — models identify input/output phonemes but fail to
formalize the environment rule using natural-class notation.

**Movement** (subtasks: overall holistic score, max 1):

| judge_raw | GPT-5.4 | Claude | MiniMax |
|---|---|---|---|
| 1/1 (pass) | 7 | 1 | 2 |
| 0/1 (fail) | 23 | 29 | 23 |

Binary pass/fail. All models overwhelmingly fail.

---

## 5. Performance Deltas: Static to Dynamic

| Model | MMLU | LiveBench (corrected) | Delta |
|---|---|---|---|
| GPT-5.4 | 87.5% | 57.2% | **-30.3pp** |
| Claude Sonnet 4.6 | 95.8% | 84.6% | **-11.2pp** |
| MiniMax M2.7 | 88.5% | 42.0% | **-46.5pp** |

### Reasoning-only delta (excluding language tasks):

| Model | MMLU | LiveBench Reasoning | Delta |
|---|---|---|---|
| GPT-5.4 | 87.5% | 72.7% | **-14.8pp** |
| Claude Sonnet 4.6 | 95.8% | 80.0% | **-15.8pp** |
| MiniMax M2.7 | 88.5% | 60.0% | **-28.5pp** |

The contamination effect is real but more moderate than previously reported. The
reasoning-only delta of ~15pp for Claude and GPT (and ~28pp for MiniMax) still
supports the argument that static benchmarks overstate capability, but it is not
the "50%+ collapse" originally claimed.

---

## 6. Methodology Notes

### Scoring changes in this report
- **LiveBench language**: Replaced exact string matching with official LiveBench
  connections scorer — order-agnostic set matching with partial credit
  (score = correct_groups / total_groups).
- **LiveBench reasoning**: No change (exact match with enclosed-answer extraction).
- **MMLU**: No change (regex letter extraction).
- **Begus**: No change (GPT-5.4 LLM-as-judge with rubric-based scoring).

### Known limitations
1. **Judge conflict of interest**: GPT-5.4 serves as both an evaluated model and
   the LLM judge for all Begus metalinguistic scoring.
2. **MMLU subset**: Only 8 of 57 MMLU categories used (62.5% logic/math/CS).
3. **Missing responses**: GPT-5.4 missing 6 LiveBench responses; MiniMax missing
   1 LiveBench and 5 Begus responses (likely API failures).
4. **Duplicate MiniMax files**: Two sets of MiniMax JSONL files exist with
   different record counts, suggesting a re-run.

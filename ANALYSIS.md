# Paper Analysis: Issues & Findings

Analysis of *Dynamic Benchmarking of LLMs: Moving Beyond Static Evaluation* (Draft V1)
against source papers, code, and experimental data in this repo.

---

## Factual Errors in the Paper

### 1. Abstract: "near-tied scores of ~88%" on MMLU — WRONG
Actual scores: GPT-5.4: 87.5%, MiniMax: 88.5%, **Claude: 95.8%**.
Claude leads by ~7–8 percentage points. Only GPT and MiniMax are near-tied.
The abstract contradicts the results section (Section 4.1), which reports the correct numbers.

### 2. Abstract: MiniMax LiveBench score listed as 30.0% — should be 30.6%
The CSV (`final_report.csv`) shows 30.6%. The paper body uses 30.6% correctly; only the
abstract says 30.0%.

### 3. GPT-5.4 LiveBench n=94 not explained
GPT-5.4 answered only 94/100 LiveBench questions (6 failed API calls). MiniMax answered
98–99 depending on which run file is used. The paper does not mention missing responses
or unequal sample sizes across models.

---

## Major Unreported Findings

### 4. LiveBench language vs. reasoning split — the biggest story not being told

| Model | Language (50q) | Reasoning (50q) | Overall |
|---|---|---|---|
| Claude Sonnet 4.6 | 0/50 = **0%** | 40/50 = **80%** | 40% |
| GPT-5.4 | 3/50 = **6%** | 32/44 = **72.7%** | 37.2% |
| MiniMax M2.7 | 0/49 = **0%** | 30/50 = **60%** | 30.6% |

All three models effectively fail language tasks (word association/grouping) and succeed
on reasoning tasks (60–80%). The "severe performance drop" narrative hides this split.
The aggregate scores (37–40%) make reasoning look worse than it is.

**Possible grading artifact:** The language tasks require outputting exact comma-separated
word lists. Claude and MiniMax did not use `<solution>` tags in their responses, so the
grader extracted nothing and scored 0. GPT-5.4's 6% comes entirely from it formatting
answers with `<solution>` tags. This may not be a real capability difference — it may be
a formatting/grading issue. This needs to be investigated and disclosed.

### 5. Movement task: Claude scores 3.3% — catastrophic and not reported

| Model | Ambiguity | Recursion | Phonology | Movement |
|---|---|---|---|---|
| GPT-5.4 | 76.7% | 74.2% | 90.0% | **23.3%** |
| Claude Sonnet 4.6 | 75.0% | 77.5% | 76.7% | **3.3%** |
| MiniMax M2.7 | 43.3% | 45.8% | 61.1% | **8.0%** |

Claude's 3.3% on movement (wh-movement, subject-auxiliary inversion) is essentially
complete failure. The paper mentions models "frequently failed" on movement but does not
report the actual numbers. GPT-5.4 is ~7× better than Claude on movement specifically —
a ranking inversion within the metalinguistic evaluation that is not discussed.

### 6. MiniMax has 5 missing Beguš responses (115 vs. 120)
The paper states n=115 for MiniMax on Beguš but does not explain the 5 missing responses
(likely API failures). This reduces comparability slightly.

### 7. Duplicate MiniMax JSONL files suggest a re-run
Both `minimax-m2-7-livebench.jsonl` (98 records) and `minimax-m2-7_livebench.jsonl`
(99 records) exist in the results directory — same for Beguš (115 vs. 110 records).
This suggests MiniMax was re-run at some point. It is unclear which run produced the
numbers reported in the paper.

---

## Methodological Issues Not Disclosed

### 8. Judge conflict of interest
GPT-5.4 is used as the LLM judge (`eval_stack/src/llm_judge.py`) for scoring all models'
Beguš responses — including GPT-5.4's own responses. The paper cites the LLM-as-a-judge
paradigm (Zheng et al., 2024) but does not disclose that the judge and one of the evaluated
models are the same system. This is a limitation that should be stated explicitly.

### 9. MMLU is a curated 8-category subset, not the full benchmark
The dataset used is 480 questions across 8 categories: abstract algebra, formal logic,
college mathematics, philosophy, college/HS computer science, logical fallacies, and
professional psychology. The real MMLU spans 57 subjects including history, law, medicine,
and social sciences. The current subset is ~62.5% logic/math/CS, which likely inflates
Claude's score relative to a full MMLU run. The paper refers to it as "the MMLU dataset"
without disclosing the subsetting.

---

## What the Paper Gets Right

- The core contamination hypothesis holds. Even setting aside the language-task grading
  issue, reasoning scores still drop from 88–96% (MMLU) to 60–80% (LiveBench reasoning).
- The granular discussion in Section 5.3 (ident vs. tree/environ subtasks) is
  well-supported by the per-category Beguš data.
- The citation of Beguš et al. (2025) for the temperature-invariance claim and the
  metalinguistic framework is appropriate and correctly used.
- The overall conclusion — static benchmarks overstate capability — is supported by the data.

---

## Priority Fixes for the Final Draft

1. **Fix Abstract**: Change "near-tied scores of ~88%" to accurately reflect that Claude
   led at 95.8% while GPT (87.5%) and MiniMax (88.5%) were near-tied.
2. **Fix Abstract**: Change MiniMax LiveBench score from 30.0% to 30.6%.
3. **Add LiveBench language/reasoning breakdown** to Section 4.2 — this is the most
   interesting finding in the dataset and is currently hidden by aggregate scores.
4. **Investigate language task grading**: Determine whether Claude/MiniMax's 0% on
   language tasks reflects a real failure or a formatting/output-extraction issue.
5. **Report movement scores explicitly** in Section 4/5: Claude 3.3%, GPT 23.3%,
   MiniMax 8% — and discuss the within-metalinguistic ranking inversion.
6. **Add Methods disclosure**: MMLU is an 8-category, logic/math-skewed subset of the
   full 57-subject benchmark.
7. **Add Limitations disclosure**: GPT-5.4 served as both evaluated model and LLM judge
   for all Beguš metalinguistic scoring.
8. **Remove asterisk** at "compelling evidence*" in Section 5 opening paragraph.

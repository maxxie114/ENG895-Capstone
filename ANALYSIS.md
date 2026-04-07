# Comprehensive Paper Analysis

Analysis of *Dynamic Benchmarking of LLMs: Moving Beyond Static Evaluation* (Draft V1)
cross-referenced against all source papers, pipeline code (`eval_stack/`), raw JSONL
responses, and CSV result files in this repo.

---

## I. CRITICAL: LiveBench Scoring Bug Invalidates Core Claim

**Severity: Paper-breaking.** The entire "50% performance drop" narrative is almost
entirely an artifact of a grading bug.

### The Bug

`eval_stack/src/evaluators.py:grade_livebench()` uses exact string matching
(`_normalize()` → lowercase + collapse whitespace) for all LiveBench tasks. The
**language tasks** are word-grouping puzzles (like the NYT Connections game) where
models must output comma-separated word lists. Order within and between groups
is semantically irrelevant — but the grader treats it as significant.

### Impact — Claude Actually Got 49/50 Language Questions Right

Inspection of Claude's raw responses shows it used `<solution>` tags and produced
the correct word groupings on 49/50 language questions — but in a different word
order than the ground truth. The grader scored all 49 as 0.

| Model | Exact Match (grader) | Set Match (order-agnostic) |
|---|---|---|
| Claude Sonnet 4.6 | **0/50 (0%)** | **49/50 (98%)** |
| GPT-5.4 | **3/50 (6%)** | **47/50 (94%)** |
| MiniMax M2.7 | **0/49 (0%)** | **5/49 (10.2%)** |

GPT-5.4's 3 "correct" answers happened to produce words in the exact same order
as the ground truth — pure coincidence, not superior capability.

### Corrected LiveBench Scores

| Model | Reported | Corrected | MMLU | Real Delta |
|---|---|---|---|---|
| Claude Sonnet 4.6 | 40.0% | **89.0%** | 95.8% | **-6.8pp** (not -55.8pp) |
| GPT-5.4 | 37.2% | **84.0%** | 87.5% | **-3.5pp** (not -50.3pp) |
| MiniMax M2.7 | 30.6% | **35.4%** | 88.5% | **-53.1pp** (real) |

The paper claims: "all three models experienced a severe, immediate performance
degradation... a massive 55.8% drop" (Section 4.2). In reality, Claude and GPT barely
drop at all. Only MiniMax shows a genuine large drop, because it truly fails the language
tasks (5/49 correct even with set-matching) and only gets 60% on reasoning.

### What This Means for the Paper's Thesis

The contamination hypothesis **does not collapse entirely** — the reasoning-only
comparison still shows a meaningful delta:
- Claude: 95.8% → 80.0% = -15.8pp
- GPT: 87.5% → 72.7% = -14.8pp
- MiniMax: 88.5% → 60.0% = -28.5pp

A 15pp drop is still noteworthy and could support a more moderate contamination
argument. But it is categorically different from the "over 50% collapse" currently
claimed. The paper must either fix the grading pipeline and re-report, or restrict
the LiveBench analysis to reasoning tasks only and reframe accordingly.

### Fix

In `evaluators.py`, the language task grading needs a set-based comparator:

```python
def _set_match(response_list: str, gt_list: str) -> bool:
    resp_words = set(w.strip().lower() for w in response_list.split(','))
    gt_words = set(w.strip().lower() for w in gt_list.split(','))
    return resp_words == gt_words
```

---

## II. Factual Errors

### 1. Abstract: "near-tied scores of ~88% on the static MMLU"
**Wrong.** Actual MMLU: Claude 95.8%, MiniMax 88.5%, GPT 87.5%. Claude leads
by 7–8pp. Only GPT and MiniMax are near-tied at ~88%. The abstract contradicts
Section 4.1, which reports the correct numbers.

### 2. Abstract: "MiniMax M2.7 dropping drastically to 30.0%"
Should be **30.6%** per `final_report.csv`. The body text (Section 4.2) correctly
says 30.6%, so only the abstract has the error.

### 3. Section 5 opening: "compelling evidence*"
Stray asterisk — leftover placeholder or footnote marker with no corresponding note.

### 4. Citation year inconsistency: White et al.
Sections 1–3 cite "White et al., 2024". Sections 4–6 switch to "White et al., 2025".
The references page lists the ICLR 2025 publication. All in-text citations should be
**(White et al., 2025)** consistently.

### 5. Missing period after hypothesis
Page 7: "...genuine, generalized reasoning (Beguš et al., 2025; White et al., 2024)"
— missing final period before Section 3.

### 6. Pronoun inconsistency
Section 2 uses "I hypothesize" (singular), while Section 3 uses "we evaluate" (plural).
Pick one voice and use it throughout. For a single-author paper, "I" is typical in
linguistics; "we" is typical in CS. Either is fine, but not both.

### 7. Wang et al. (2018) in references but never cited in body
The GLUE paper appears in the references (page 21) but is not cited anywhere in the
text. Either cite it in the Background section or remove it from the references.

---

## III. Methodological Issues Not Disclosed

### 8. GPT-5.4 is both evaluated model and LLM judge

`eval_stack/src/llm_judge.py` uses GPT-5.4 (`openai_client`) as the judge for all
Beguš metalinguistic responses — including GPT-5.4's own answers. The paper
discusses LLM-as-a-judge (citing Zheng et al., 2024) but never discloses this conflict.

GPT-5.4 scored highest on Beguš (66.0%). While this may be genuine, the judge
overlap means self-preference bias cannot be ruled out without disclosure. At minimum,
add a Limitations paragraph acknowledging this.

### 9. MMLU is an 8-category subset, not the full 57-subject benchmark

The dataset (`eval_stack/data/mmlu_dataset.csv`) contains 480 questions across 8
hand-picked categories:

| Category | Count | Domain |
|---|---|---|
| abstract_algebra | 60 | Math |
| college_mathematics | 60 | Math |
| formal_logic | 60 | Logic |
| college_computer_science | 60 | CS |
| high_school_computer_science | 60 | CS |
| logical_fallacies | 60 | Logic |
| philosophy | 60 | Humanities |
| professional_psychology | 60 | Social Science |

This is **62.5% logic/math/CS** (5 of 8 categories). The full MMLU spans 57 subjects
including US history, law, medicine, biology, economics, etc. The paper says "the
Measuring Massive Multitask Language Understanding (MMLU) dataset" (Section 3.2)
without disclosing the subsetting. This must be stated in Methods.

The skew matters: Claude's dominance (95.8% vs. ~88%) may be category-dependent.
On professional_psychology, Claude scores only 90.0% while GPT scores 98.3% — the
one non-logic/CS category where GPT leads.

### 10. Unequal sample sizes not reported

| Benchmark | Claude | GPT-5.4 | MiniMax |
|---|---|---|---|
| MMLU | 480 | 480 | 480 |
| LiveBench | 100 | **94** | **98** |
| Beguš | 120 | 120 | **115** |

GPT-5.4 is missing 6 LiveBench responses. MiniMax is missing 2 LiveBench and 5 Beguš
responses. These are likely API failures. The paper reports aggregate percentages
without mentioning the denominator differences.

### 11. Duplicate MiniMax result files

Two sets of MiniMax JSONL files exist with different record counts:
- `minimax-m2-7-livebench.jsonl` (98) vs. `minimax-m2-7_livebench.jsonl` (99)
- `minimax-m2-7-begus.jsonl` (115) vs. `minimax-m2-7_begus.jsonl` (110)

This suggests MiniMax was re-run. The paper should state which run was used and why.

---

## IV. Unreported Findings in the Data

### 12. Movement is catastrophically harder than other Beguš categories

| Model | Ambiguity | Recursion | Phonology | **Movement** |
|---|---|---|---|---|
| GPT-5.4 | 76.7% | 74.2% | 90.0% | **23.3%** |
| Claude 4.6 | 75.0% | 77.5% | 76.7% | **3.3%** |
| MiniMax M2.7 | 43.3% | 45.8% | 61.1% | **8.0%** |

Claude passed **1 out of 30** movement questions. The paper (Section 5.3) says models
"frequently failed" on tree generation and movement but never reports these numbers.
The movement category should be discussed as a distinct finding: it shows that
syntactic movement (wh-traces, subject-aux inversion, co-indexation) is the single
hardest metalinguistic task by far, and that Claude — the highest MMLU scorer — is
the worst performer on it.

### 13. Phonology subtask distribution suggests environ is the bottleneck

For phonology (3 subtasks: input, output, environ), the judge_raw distribution is:

| judge_raw | GPT-5.4 | Claude | MiniMax |
|---|---|---|---|
| 3/3 | 22 | 13 | 7 |
| 2/3 | 7 | 14 | 15 |
| 1/3 | 1 | 2 | 4 |
| 0/3 | 0 | 1 | 4 |

Most failures land at 2/3 (pass input + output, fail environ), which supports the
paper's Section 5.3 claim about environ being the hard subtask. But the paper presents
this as a qualitative observation. The quantitative data above would strengthen it.

### 14. Ambiguity subtask distribution: ~40% get only 1 of 2 subtasks

For ambiguity (2 subtasks: ident, trees):

| judge_raw | GPT-5.4 | Claude | MiniMax |
|---|---|---|---|
| 2/2 | 17 | 16 | 3 |
| 1/2 | 12 | 13 | 20 |
| 0/2 | 1 | 1 | 7 |

A large proportion (40–67%) get exactly 1/2, consistent with passing identification but
failing tree generation. Again, quantitative data that would strengthen Section 5.3.

### 15. MMLU category-level variation

| Category | GPT-5.4 | Claude | MiniMax |
|---|---|---|---|
| abstract_algebra | 81.7% | 95.0% | 85.0% |
| college_computer_science | 88.3% | 96.7% | 83.3% |
| college_mathematics | 73.3% | 98.3% | 90.0% |
| formal_logic | 78.3% | 96.7% | 83.3% |
| high_school_computer_science | 96.7% | 96.7% | 98.3% |
| logical_fallacies | 93.3% | 96.7% | 93.3% |
| philosophy | 90.0% | 96.7% | 93.3% |
| professional_psychology | **98.3%** | 90.0% | 81.7% |

Claude dominates 7 of 8 categories. The exception is professional_psychology where
GPT leads (98.3% vs. 90.0%). This within-MMLU variation is not discussed and would
add nuance to the "MMLU is saturated" argument — it is not actually saturated for
GPT-5.4 on formal logic (78.3%) or college math (73.3%).

### 16. MiniMax response format: all MMLU answers are verbose

GPT-5.4 outputs single letters (all 480 responses < 10 chars). Claude is mixed (370
short, 100 long). MiniMax **always** outputs long chain-of-thought `<think>` blocks (all
480 responses > 100 chars). The MMLU grader must parse the answer letter from long
text for every MiniMax response. This worked (regex patterns in `grade_mmlu()` handle
it), but it's worth noting as a methodological detail: different models' scores depend
on different regex extraction patterns actually firing correctly.

---

## V. Proofreading & Style Issues

### Grammar and Mechanics

1. **Page 7, end of hypothesis**: Missing period after "(Beguš et al., 2025; White et al., 2024)"
2. **Page 13**: "compelling evidence**\***" — remove stray asterisk
3. **Page 16**: "input" and "output" appear with underline/highlight formatting that seems
   accidental (visible in PDF)

### Voice and Register

4. **Pronoun shift**: "I hypothesize" (p. 7) vs. "we evaluate" (p. 7), "In this study, we..."
   (p. 7). Choose one and be consistent. Single-author linguistics papers typically use "I".
5. **Section 5 title**: "Discussions" should be singular "Discussion" per APA convention.

### Citation Issues

6. **White et al. year**: Cited as (2024) in Sections 1–3 but (2025) in Sections 4–6.
   The reference list says 2025 (ICLR). All citations should read **(White et al., 2025)**.
7. **Wang et al. (2018)**: Listed in references (p. 21) but never cited in the body text.
   Either cite it or remove it.
8. **Jimenez et al. (2024)**: Cited once on page 5 but the SWE-bench paper is about
   code generation, not benchmark contamination. The citation supports the claim that
   "newer task-based benchmarks often contain underlying components that remain
   relatively static" — verify this is what the Jimenez paper actually argues.

### Structural Notes

9. **No Limitations section**: The paper has Introduction, Background, Methods, Results,
   Discussion, Conclusion — but no explicit Limitations section. APA-style empirical
   papers typically include one (often as 5.4 or within Discussion). The methodological
   issues above (judge conflict, MMLU subsetting, missing responses) should go there.
10. **No description of the LLM-as-judge implementation**: The Methods section (3.2)
    describes the three benchmarks but does not explain how Beguš responses were
    scored. The reader learns about LLM-as-a-judge from the Background (Section 2.1)
    but the Methods section never states "GPT-5.4 was used as the judge with the
    following rubric..." This is a gap.
11. **Figures not self-contained**: Figures 1–4 lack source annotations. Academic figures
    typically include a note like "Note. Data from author's benchmark runs, April 2026."

---

## VI. Summary of Priority Fixes

### Must-fix (paper integrity)

1. **Fix LiveBench grading bug** — re-run scoring with set-based matching for language
   tasks, or restrict analysis to reasoning tasks only and reframe the contamination
   argument around a ~15pp drop instead of ~55pp
2. **Fix Abstract** — "near-tied scores of ~88%" is factually wrong (Claude is 95.8%)
3. **Fix White et al. citation year** — use (2025) consistently throughout
4. **Disclose MMLU subsetting** — state it's 8/57 categories, logic/math-skewed
5. **Disclose judge = GPT-5.4** — add to Limitations
6. **Add Limitations section**

### Should-fix (scholarly rigor)

7. Add LiveBench language/reasoning subcategory breakdown
8. Report Beguš per-category scores (especially movement: 3.3%/23.3%/8%)
9. Fix pronoun inconsistency (I vs. we)
10. Fix "Discussions" → "Discussion"
11. Remove stray asterisk in Section 5
12. Remove uncited Wang et al. (2018) from references
13. Report and explain unequal sample sizes (n=94, n=98, n=115)

### Nice-to-have (strengthens the paper)

14. Add quantitative subtask distributions (judge_raw tables) to Section 5.3
15. Add MMLU per-category table showing within-benchmark variation
16. Add a methods paragraph describing the LLM judge rubric and implementation
17. Add figure source notes

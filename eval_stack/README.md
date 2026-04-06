# eval_stack

Async evaluation pipeline for the ENG 895 capstone paper.
Runs three frontier LLMs across three benchmarks and outputs accuracy scores as a CSV.

---

## Models

| Model | Provider | API |
|-------|----------|-----|
| GPT-5.4 | OpenAI | OpenAI API |
| Claude Sonnet 4.6 | Anthropic | OpenRouter |
| MiniMax M2.7 | MiniMax | MiniMax API |

## Benchmarks

| Benchmark | Type | Items |
|-----------|------|-------|
| MMLU | Static multiple-choice | ~600 (5-shot) |
| LiveBench | Dynamic reasoning/language | 100 |
| Beguš et al. | Metalinguistic analysis | 120 |

---

## Setup

### 1. Install dependencies

```bash
cd eval_stack
pip install -r requirements.txt
```

### 2. Add your API keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
OPENAI_API_KEY=sk-...           # from platform.openai.com
OPENROUTER_API_KEY=sk-or-...    # from openrouter.ai/keys
MINIMAX_API_KEY=...             # from platform.minimaxi.com
```

The `OPENROUTER_MODEL` and `BEGUS_DATASET_PATH` fields have working defaults — you don't need to change them unless the Claude model slug changes on OpenRouter or you move the dataset file.

### 3. Verify the Beguš dataset is in place

```bash
ls data/begus_dataset.csv
```

It should already be there. If not, run the build script from the project root or copy it from `osfstorage-archive/`.

---

## Running the evaluation

```bash
cd eval_stack
python main.py
```

That's it. The pipeline will:

1. Load all three datasets
2. Print how many total API calls it will make
3. Run all inference concurrently (8 calls at a time across all providers)
4. Save every response to `results/raw_responses.jsonl` as it comes in
5. Grade MMLU and LiveBench automatically via regex
6. Send all 120 Beguš responses through GPT-5.4 as the judge
7. Print the final results table and save it to `results/final_report.csv`

Expected runtime: **2–3 hours** depending on API latency.

---

## Output files

### `results/raw_responses.jsonl`

Every individual model response, written immediately after each API call (checkpoint file — if the run crashes, you won't lose what was already completed).

Each line is a JSON object:

```json
{
  "benchmark": "mmlu",
  "model": "gpt-5.4",
  "question_id": "mmlu_formal_logic_0",
  "category": "formal_logic",
  "subtasks": "",
  "response": "A",
  "ground_truth": "A",
  "score": 1
}
```

### `results/final_report.csv`

One row per model, one column per benchmark. This is the main result table for the paper.

```
model,mmlu_accuracy,mmlu_n,livebench_accuracy,livebench_n,begus_accuracy,begus_n
gpt-5.4,0.82,600,0.71,100,0.68,120
claude-sonnet-4.6,0.80,600,0.69,100,0.71,120
minimax-m2.7,0.77,600,0.64,100,0.59,120
```

- `*_accuracy` — proportion correct (0.0–1.0)
- `*_n` — number of items actually graded (should match expected counts)

**Beguš scoring note:** each item is scored 0–1 based on how many sub-tasks the model passes, normalized by the number of sub-tasks for that category (ambiguity: 2, recursion: 4, movement: 1, phonology: 3).

---

## Resuming a crashed run

If the run fails partway through, the JSONL checkpoint preserves everything completed so far. To avoid re-running already-completed items, filter them out by `question_id` before re-running — or just re-run from scratch if it failed early.

---

## File structure

```
eval_stack/
├── main.py                  # entry point
├── .env.example             # copy to .env and fill in keys
├── .env                     # your keys (never committed)
├── requirements.txt
├── data/
│   └── begus_dataset.csv    # 120-item Beguš et al. dataset
├── results/
│   ├── raw_responses.jsonl  # per-call checkpoint (generated at runtime)
│   └── final_report.csv     # final accuracy table (generated at runtime)
└── src/
    ├── api_clients.py        # OpenAI, OpenRouter, MiniMax wrappers
    ├── data_loaders.py       # dataset loaders for all three benchmarks
    ├── evaluators.py         # regex graders for MMLU + LiveBench
    └── llm_judge.py          # GPT-5.4 as judge for Beguš responses
```

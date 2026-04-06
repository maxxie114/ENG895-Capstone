"""
Dataset loaders for the three benchmarks.

- load_mmlu()      → ~600 items from cais/mmlu (5-shot formatted)
- load_livebench() → 100 items from livebench/livebench (reasoning + language)
- load_begus()     → 120 items from local CSV or JSON file
"""

import os
import json
import re
import pandas as pd
from datasets import load_dataset

# MMLU subjects chosen for relevance to the paper's themes
MMLU_SUBJECTS = [
    "formal_logic",
    "abstract_algebra",
    "college_mathematics",
    "philosophy",
    "college_computer_science",
    "high_school_computer_science",
    "linguistics",
    "professional_psychology",
    "logical_fallacies",
    "moral_reasoning",
]

ANSWER_LETTERS = ["A", "B", "C", "D"]

MMLU_SYSTEM = (
    "The following are multiple choice questions (with answers). "
    "Reply with only the letter of the correct answer (A, B, C, or D)."
)


def _format_mmlu_question(row: dict, include_answer: bool = False) -> str:
    choices = "\n".join(
        f"{ANSWER_LETTERS[i]}. {row['choices'][i]}" for i in range(4)
    )
    q = f"Question: {row['question']}\n{choices}\nAnswer:"
    if include_answer:
        q += f" {ANSWER_LETTERS[row['answer']]}"
    return q


def load_mmlu(samples_per_subject: int = 60) -> list[dict]:
    """
    Load MMLU with 5-shot prompting. Dev split provides the 5 examples;
    test split provides the evaluation items.
    Caps at samples_per_subject per subject (~600 total across 10 subjects).
    """
    items = []
    for subject in MMLU_SUBJECTS:
        try:
            dev = load_dataset("cais/mmlu", subject, split="dev")
            test = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"  [MMLU] Skipping {subject}: {e}")
            continue

        # Build 5-shot context from dev set (always exactly 5 examples)
        few_shot = "\n\n".join(
            _format_mmlu_question(dev[i], include_answer=True)
            for i in range(min(5, len(dev)))
        )

        count = 0
        for i, row in enumerate(test):
            if count >= samples_per_subject:
                break
            question_block = _format_mmlu_question(row, include_answer=False)
            prompt = f"{few_shot}\n\n{question_block}"
            items.append({
                "id": f"mmlu_{subject}_{i}",
                "benchmark": "mmlu",
                "category": subject,
                "prompt": prompt,
                "system": MMLU_SYSTEM,
                "ground_truth": ANSWER_LETTERS[row["answer"]],
            })
            count += 1

    print(f"[MMLU] Loaded {len(items)} items across {len(MMLU_SUBJECTS)} subjects.")
    return items


def load_livebench(n: int = 100) -> list[dict]:
    """
    Load LiveBench, filter to reasoning + language tasks, take n items.
    Uses the most recent available split.
    """
    try:
        ds = load_dataset("livebench/livebench", split="test")
    except Exception:
        # Some versions use 'train' as the only split
        ds = load_dataset("livebench/livebench", split="train")

    # Filter to reasoning and language categories
    target_cats = {"reasoning", "language"}
    filtered = [
        row for row in ds
        if str(row.get("category", "")).lower() in target_cats
        or str(row.get("task", "")).lower() in target_cats
    ]

    # Fallback: just take first n if filter returns nothing
    if not filtered:
        filtered = list(ds)

    filtered = filtered[:n]

    items = []
    for i, row in enumerate(filtered):
        question = row.get("question") or row.get("turns", [""])[0]
        ground_truth = str(row.get("answer", ""))
        category = row.get("category") or row.get("task", "livebench")
        system_prompt = row.get("system_prompt", None)

        items.append({
            "id": row.get("question_id", f"livebench_{i}"),
            "benchmark": "livebench",
            "category": category,
            "prompt": question,
            "system": system_prompt,
            "ground_truth": ground_truth,
        })

    print(f"[LiveBench] Loaded {len(items)} items.")
    return items


def load_begus() -> list[dict]:
    """
    Load the Beguš et al. 120-item metalinguistic dataset from a local file.
    Supports CSV (columns: id, category, prompt, ground_truth)
    or JSON (list of objects with the same keys).

    Set BEGUS_DATASET_PATH in .env to point to the file.
    """
    path = os.getenv("BEGUS_DATASET_PATH", "./data/begus_dataset.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Beguš dataset not found at '{path}'. "
            "Download the 120-item dataset from the paper's supplementary materials "
            "and set BEGUS_DATASET_PATH in your .env file."
        )

    if path.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        rows = raw if isinstance(raw, list) else raw.get("data", [])
    else:
        df = pd.read_csv(path)
        rows = df.to_dict(orient="records")

    items = []
    for i, row in enumerate(rows):
        items.append({
            "id": str(row.get("id", f"begus_{i}")),
            "benchmark": "begus",
            "category": str(row.get("category", "metalinguistics")),
            "prompt": str(row.get("prompt", row.get("question", ""))),
            "system": (
                "You are an expert computational linguist. "
                "Analyze the following linguistic data using formal linguistic notation "
                "where appropriate (e.g., phrase structure rules, feature matrices, "
                "phonological derivations)."
            ),
            "ground_truth": str(row.get("ground_truth", row.get("answer", ""))),
        })

    print(f"[Beguš] Loaded {len(items)} items.")
    return items

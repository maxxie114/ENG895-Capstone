"""
Dataset loaders for the three benchmarks.

All datasets are pre-downloaded as local CSV files so the pipeline works
inside Daytona sandboxes without needing HuggingFace access.

- load_mmlu()      → data/mmlu_dataset.csv      (~480 items, 5-shot formatted)
- load_livebench() → data/livebench_dataset.csv  (100 items, reasoning + language)
- load_begus()     → data/begus_dataset.csv       (120 items, metalinguistic)

To regenerate the pre-downloaded CSVs, run:
    python scripts/download_datasets.py
"""

import json
import os
from pathlib import Path

import pandas as pd


def _data_path(filename: str) -> Path:
    """Resolve data file relative to this file's location (works in sandbox too)."""
    return Path(__file__).parent.parent / "data" / filename


def load_mmlu() -> list[dict]:
    path = _data_path("mmlu_dataset.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"MMLU dataset not found at '{path}'. "
            "Run scripts/download_datasets.py to generate it."
        )
    df = pd.read_csv(path)
    items = df.to_dict(orient="records")
    # Fill missing system field
    for item in items:
        if not item.get("system"):
            item["system"] = (
                "The following are multiple choice questions (with answers). "
                "Reply with only the letter of the correct answer (A, B, C, or D)."
            )
    print(f"[MMLU] Loaded {len(items)} items.")
    return items


def load_livebench() -> list[dict]:
    path = _data_path("livebench_dataset.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"LiveBench dataset not found at '{path}'. "
            "Run scripts/download_datasets.py to generate it."
        )
    df = pd.read_csv(path)
    items = df.to_dict(orient="records")
    for item in items:
        item["system"] = item.get("system") or None
        item["subtasks"] = item.get("subtasks", "")
    print(f"[LiveBench] Loaded {len(items)} items.")
    return items


def load_begus() -> list[dict]:
    path_str = os.getenv("BEGUS_DATASET_PATH", str(_data_path("begus_dataset.csv")))
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(
            f"Beguš dataset not found at '{path}'. "
            "Set BEGUS_DATASET_PATH in .env or place begus_dataset.csv in data/."
        )

    if path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        rows = raw if isinstance(raw, list) else raw.get("data", [])
    else:
        rows = pd.read_csv(path).to_dict(orient="records")

    items = []
    for i, row in enumerate(rows):
        items.append({
            "id":       str(row.get("id", f"begus_{i}")),
            "benchmark": "begus",
            "category": str(row.get("category", "metalinguistics")),
            "prompt":   str(row.get("prompt", row.get("question", ""))),
            "subtasks": str(row.get("subtasks", "overall")),
            "system": (
                "You are an expert computational linguist. "
                "Analyze the following linguistic data using formal linguistic notation "
                "where appropriate (e.g., phrase structure rules, feature matrices, "
                "phonological derivations)."
            ),
            "ground_truth": str(row.get("judge_criteria", row.get("ground_truth", ""))),
        })

    print(f"[Beguš] Loaded {len(items)} items.")
    return items

"""
Single-model × single-benchmark runner.
Executed INSIDE a Daytona sandbox by run_daytona.py.

Usage:
    python3 src/runner.py --model gpt-5.4 --benchmark mmlu
    python3 src/runner.py --model claude-sonnet-4.6 --benchmark livebench
    python3 src/runner.py --model glm-5.1 --benchmark begus

Outputs:
    results/{model}_{benchmark}.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pathlib import Path as _Path
load_dotenv(_Path(__file__).parent.parent / ".env")

from api_clients import OpenAIClient, OpenRouterClient, MiniMaxClient
from data_loaders import load_mmlu, load_livebench, load_begus
from evaluators import grade_mmlu, grade_livebench
from llm_judge import judge_begus_batch

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SEMAPHORE_LIMIT = 5


def _build_client(model_name: str):
    """Instantiate the right API client for a given model name."""
    if model_name == "gpt-5.4":
        return OpenAIClient()
    elif model_name == "claude-sonnet-4.6":
        return OpenRouterClient(model="anthropic/claude-sonnet-4-6")
    elif model_name == "glm-5.1":
        return OpenRouterClient(model="z-ai/glm-5.1")
    elif model_name == "minimax-m2.7":
        return MiniMaxClient()
    else:
        raise ValueError(f"Unknown model: {model_name}")


SUPPORTED_MODELS = ["gpt-5.4", "claude-sonnet-4.6", "glm-5.1", "minimax-m2.7"]

LOADERS = {
    "mmlu":      load_mmlu,
    "livebench": load_livebench,
    "begus":     load_begus,
}


async def run_one(client, model_name: str, item: dict, semaphore: asyncio.Semaphore) -> dict | None:
    async with semaphore:
        try:
            response = await client.complete(item["prompt"], system=item.get("system"))
            return {
                "benchmark":   item["benchmark"],
                "model":       model_name,
                "question_id": item["id"],
                "category":    item.get("category", ""),
                "subtasks":    item.get("subtasks", ""),
                "response":    response,
                "ground_truth": item.get("ground_truth", ""),
            }
        except Exception as e:
            print(f"[ERROR] {model_name} | {item['id']}: {e}", file=sys.stderr)
            return None


async def main(model_name: str, benchmark: str):
    print(f"[{model_name} × {benchmark}] Starting...")

    client = _build_client(model_name)

    loader = LOADERS[benchmark]
    items = loader()
    print(f"[{model_name} × {benchmark}] Loaded {len(items)} items.")

    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    tasks = [run_one(client, model_name, item, semaphore) for item in items]

    out_path = RESULTS_DIR / f"{model_name.replace('.', '-')}_{benchmark}.jsonl"

    results = []
    with open(out_path, "w", buffering=1) as f:
        async def run_and_save(coro):
            r = await coro
            if r is not None:
                f.write(json.dumps(r) + "\n")
            return r

        raw = await asyncio.gather(*[run_and_save(t) for t in tasks])
        results = [r for r in raw if r is not None]

    print(f"[{model_name} × {benchmark}] Inference done: {len(results)} responses.")

    # Grade
    openai_client = OpenAIClient()
    for r in results:
        if benchmark == "mmlu":
            r["score"] = grade_mmlu(r["response"], r["ground_truth"])
        elif benchmark == "livebench":
            r["score"] = grade_livebench(r["response"], r["ground_truth"], r.get("category", ""))

    if benchmark == "begus":
        print(f"[{model_name} × {benchmark}] Running LLM judge...")
        results = await judge_begus_batch(results, openai_client, concurrency=5)
        for r in results:
            r["score"] = r.get("judge_score", 0)

    # Rewrite JSONL with scores
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    accuracy = sum(r.get("score", 0) for r in results) / len(results) if results else 0
    print(f"[{model_name} × {benchmark}] Accuracy: {accuracy:.4f}")
    print(f"[{model_name} × {benchmark}] Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--benchmark", required=True, choices=list(LOADERS.keys()))
    args = parser.parse_args()
    asyncio.run(main(args.model, args.benchmark))

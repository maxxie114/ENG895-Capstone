"""
Async orchestrator for the LLM evaluation pipeline.

Usage:
    cp .env.example .env          # fill in your keys
    pip install -r requirements.txt
    python main.py

Outputs:
    results/raw_responses.jsonl   # checkpoint after every call
    results/final_report.csv      # accuracy per model per benchmark
"""

import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

from src.api_clients import OpenAIClient, OpenRouterClient, MiniMaxClient
from src.data_loaders import load_mmlu, load_livebench, load_begus
from src.evaluators import grade_mmlu, grade_livebench
from src.llm_judge import judge_begus_batch

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RAW_JSONL = RESULTS_DIR / "raw_responses.jsonl"
FINAL_CSV = RESULTS_DIR / "final_report.csv"

# Max concurrent API calls across all providers combined
SEMAPHORE_LIMIT = 8


async def run_one(client, model_name: str, item: dict, semaphore: asyncio.Semaphore) -> dict | None:
    async with semaphore:
        try:
            response = await client.complete(item["prompt"], system=item.get("system"))
            return {
                "benchmark": item["benchmark"],
                "model": model_name,
                "question_id": item["id"],
                "category": item.get("category", ""),
                "subtasks": item.get("subtasks", ""),
                "response": response,
                "ground_truth": item.get("ground_truth", ""),
            }
        except Exception as e:
            print(f"\n[ERROR] {model_name} | {item['id']}: {e}")
            return None


async def main():
    # --- Init clients ---
    openai_client = OpenAIClient()
    claude_client = OpenRouterClient(model="anthropic/claude-sonnet-4-6")
    glm_client = OpenRouterClient(model="z-ai/glm-5.1")

    clients = {
        "gpt-5.4": openai_client,
        "claude-sonnet-4.6": claude_client,
        "glm-5.1": glm_client,
    }

    # Uncomment to include MiniMax (blocked from some cloud IPs):
    # clients["minimax-m2.7"] = MiniMaxClient()

    # --- Load datasets ---
    print("=" * 50)
    print("Loading datasets...")
    print("=" * 50)
    mmlu_items = load_mmlu()
    livebench_items = load_livebench()
    begus_items = load_begus()

    all_items = {
        "mmlu": mmlu_items,
        "livebench": livebench_items,
        "begus": begus_items,
    }

    total = sum(len(v) for v in all_items.values()) * len(clients)
    print(f"\nTotal API calls to make: {total}")
    print("=" * 50)

    # --- Build all coroutines ---
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    coroutines = []
    for benchmark, items in all_items.items():
        for model_name, client in clients.items():
            for item in items:
                coroutines.append(run_one(client, model_name, item, semaphore))

    # --- Run inference, writing checkpoint after each result ---
    results = []
    print("\nRunning inference...")
    with open(RAW_JSONL, "w", buffering=1) as f:
        async def run_and_checkpoint(coro):
            result = await coro
            if result is not None:
                f.write(json.dumps(result) + "\n")
            return result

        raw = await tqdm_asyncio.gather(
            *[run_and_checkpoint(c) for c in coroutines],
            desc="Inference"
        )
        results = [r for r in raw if r is not None]

    print(f"\nCompleted: {len(results)}/{total} calls succeeded.")

    # --- Grade MMLU and LiveBench ---
    print("\nGrading MMLU and LiveBench...")
    for r in results:
        if r["benchmark"] == "mmlu":
            r["score"] = grade_mmlu(r["response"], r["ground_truth"])
        elif r["benchmark"] == "livebench":
            r["score"] = grade_livebench(r["response"], r["ground_truth"], r.get("category", ""))

    # --- LLM-as-judge for Beguš ---
    begus_results = [r for r in results if r["benchmark"] == "begus"]
    if begus_results:
        print(f"\nRunning LLM judge on {len(begus_results)} Beguš responses...")
        begus_results = await judge_begus_batch(begus_results, openai_client)
        for r in begus_results:
            r["score"] = r.get("judge_score", 0)

    # --- Aggregate into final report ---
    print("\nAggregating results...")
    rows = []
    for model_name in clients:
        row = {"model": model_name}
        for bname in ["mmlu", "livebench", "begus"]:
            subset = [
                r for r in results
                if r["model"] == model_name
                and r["benchmark"] == bname
                and "score" in r
            ]
            if subset:
                row[f"{bname}_accuracy"] = round(sum(r["score"] for r in subset) / len(subset), 4)
                row[f"{bname}_n"] = len(subset)
            else:
                row[f"{bname}_accuracy"] = None
                row[f"{bname}_n"] = 0
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(FINAL_CSV, index=False)

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(df.to_string(index=False))
    print(f"\nSaved: {RAW_JSONL}")
    print(f"Saved: {FINAL_CSV}")


if __name__ == "__main__":
    asyncio.run(main())

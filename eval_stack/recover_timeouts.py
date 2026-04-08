"""
Recover timed-out benchmark questions by re-running only the missing ones.
Reads existing JSONL results, identifies gaps, re-queries the API, grades,
and appends to the existing files.

Usage:
    python3 recover_timeouts.py --model minimax-m2.7
    python3 recover_timeouts.py --model glm-5.1
    python3 recover_timeouts.py --model gpt-5.4
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from src.api_clients import OpenAIClient, OpenRouterClient, MiniMaxClient
from src.data_loaders import load_mmlu, load_livebench, load_begus
from src.evaluators import grade_mmlu, grade_livebench
from src.llm_judge import judge_begus_batch

RESULTS_DIR = Path("results")

LOADERS = {
    "mmlu": load_mmlu,
    "livebench": load_livebench,
    "begus": load_begus,
}


def build_client(model_name: str):
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


def find_missing(model_name: str):
    """Return {benchmark: [items]} for questions missing from results."""
    tag = model_name.replace(".", "-")
    missing = {}

    for bench, loader in LOADERS.items():
        jsonl_path = RESULTS_DIR / f"{tag}_{bench}.jsonl"
        if not jsonl_path.exists():
            print(f"  {bench}: no results file found, skipping")
            continue

        existing_ids = set()
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    existing_ids.add(json.loads(line)["question_id"])

        all_items = loader()
        gaps = [it for it in all_items if it["id"] not in existing_ids]

        if gaps:
            missing[bench] = gaps
            print(f"  {bench}: {len(gaps)} missing out of {len(all_items)}")
        else:
            print(f"  {bench}: complete ({len(existing_ids)}/{len(all_items)})")

    return missing


async def recover_one(client, model_name: str, item: dict, retries: int = 3) -> dict | None:
    """Try up to `retries` times to get a response."""
    for attempt in range(1, retries + 1):
        try:
            response = await client.complete(item["prompt"], system=item.get("system"))
            if not response:
                print(f"    [{item['id']}] Empty response on attempt {attempt}, retrying...")
                continue
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
            print(f"    [{item['id']}] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                await asyncio.sleep(5 * attempt)  # backoff
    return None


async def main(model_name: str):
    print(f"\n=== Recovering timeouts for {model_name} ===\n")

    missing = find_missing(model_name)
    if not missing:
        print("\nNothing to recover!")
        return

    total = sum(len(v) for v in missing.values())
    print(f"\nTotal to recover: {total} questions\n")

    client = build_client(model_name)
    openai_client = OpenAIClient()  # for LLM judge
    tag = model_name.replace(".", "-")

    for bench, items in missing.items():
        print(f"--- Recovering {bench}: {len(items)} items ---")

        # Run inference sequentially (one at a time to avoid more timeouts)
        results = []
        for item in items:
            print(f"  [{item['id']}] querying...", end=" ", flush=True)
            r = await recover_one(client, model_name, item)
            if r:
                print(f"OK ({len(r['response'])} chars)")
                results.append(r)
            else:
                print("FAILED (all retries exhausted)")

        print(f"  Got {len(results)}/{len(items)} responses")

        if not results:
            continue

        # Grade
        for r in results:
            if bench == "mmlu":
                r["score"] = grade_mmlu(r["response"], r["ground_truth"])
            elif bench == "livebench":
                r["score"] = grade_livebench(r["response"], r["ground_truth"], r.get("category", ""))

        if bench == "begus":
            print(f"  Running LLM judge on {len(results)} responses...")
            results = await judge_begus_batch(results, openai_client, concurrency=3)
            for r in results:
                r["score"] = r.get("judge_score", 0)

        # Append to existing JSONL
        jsonl_path = RESULTS_DIR / f"{tag}_{bench}.jsonl"
        with open(jsonl_path, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  Appended {len(results)} records to {jsonl_path}")

        # Show scores
        for r in results:
            print(f"    {r['question_id']}: score={r.get('score', '?')}")

    # Print updated totals
    print(f"\n=== Updated totals for {model_name} ===")
    for bench in LOADERS:
        jsonl_path = RESULTS_DIR / f"{tag}_{bench}.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                records = [json.loads(l) for l in f if l.strip()]
            scored = [r for r in records if "score" in r]
            acc = sum(r["score"] for r in scored) / len(scored) if scored else 0
            print(f"  {bench}: {len(records)} records, accuracy={acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.model))

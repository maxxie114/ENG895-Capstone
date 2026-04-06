"""Mini test runner — 5 MMLU questions, GPT-5.4 only. Used to verify the sandbox pipeline."""
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path("/eval/.env"))

sys.path.insert(0, "/eval/src")

from api_clients import OpenAIClient
from data_loaders import load_mmlu
from evaluators import grade_mmlu


async def main():
    client = OpenAIClient()
    items = load_mmlu()[:5]
    print(f"Loaded {len(items)} items")

    results = []
    for item in items:
        try:
            resp = await client.complete(item["prompt"], system=item.get("system"))
            score = grade_mmlu(resp, item["ground_truth"])
            results.append({
                "id":       item["id"],
                "response": resp[:80],
                "gt":       item["ground_truth"],
                "score":    score,
            })
            print(f"  {item['id']}: gt={item['ground_truth']} score={score}")
        except Exception as e:
            print(f"  ERROR on {item['id']}: {e}")

    Path("/eval/results").mkdir(exist_ok=True)
    Path("/eval/results/mini_test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results)
    )
    acc = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"\nAccuracy: {acc:.2%} ({sum(r['score'] for r in results)}/{len(results)})")


asyncio.run(main())

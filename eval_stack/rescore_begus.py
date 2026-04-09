"""Re-run LLM-as-a-judge scoring for ALL Begus results across all models."""
import asyncio, json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.chdir(Path(__file__).parent)

from dotenv import load_dotenv
load_dotenv()

from api_clients import OpenAIClient
from llm_judge import judge_begus_batch

RESULT_DIR = Path("results/result_apr8")

MODELS = [
    "gpt-5-4",
    "claude-sonnet-4-6",
    "glm-5-1",
    "minimax-m2-7",
]


async def main():
    client = OpenAIClient()

    for model in MODELS:
        fpath = RESULT_DIR / f"{model}-begus.jsonl"
        with open(fpath) as f:
            records = [json.loads(l) for l in f if l.strip()]
        print(f"\n{model}: {len(records)} records, judging ALL in parallel...")

        judged = await judge_begus_batch(records, client, concurrency=20)

        for r in judged:
            r["score"] = r.get("judge_score", 0)

        with open(fpath, "w") as f:
            for r in judged:
                f.write(json.dumps(r) + "\n")

        scored = [r for r in judged if r.get("score") is not None]
        avg = sum(r["score"] for r in scored) / len(scored) if scored else 0
        print(f"  {model}: {avg*100:.2f}% ({len(scored)} items)")

    print("\nDone!")

asyncio.run(main())

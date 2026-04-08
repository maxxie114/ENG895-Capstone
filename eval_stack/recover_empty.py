"""
Re-run benchmark items that got empty responses (reasoning model token exhaustion).
Fires ALL items across ALL benchmarks in parallel with a per-item timeout.

Usage:
    python3 recover_empty.py --model glm-5.1
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from src.api_clients import OpenAIClient, OpenRouterClient, MiniMaxClient
from src.data_loaders import load_mmlu, load_livebench, load_begus
from src.evaluators import grade_mmlu, grade_livebench
from src.llm_judge import judge_begus_batch

RESULTS_DIR = Path("results")
ITEM_TIMEOUT = 600  # 10 minutes per item

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


def find_empty_responses(model_name: str):
    """Return {benchmark: (jsonl_path, all_records, [(index, record, dataset_item)])}"""
    tag = model_name.replace(".", "-")
    empty = {}

    for bench, loader in LOADERS.items():
        jsonl_path = None
        for sep in ["_", "-"]:
            p = RESULTS_DIR / f"{tag}{sep}{bench}.jsonl"
            if p.exists():
                jsonl_path = p
                break
        if not jsonl_path:
            continue

        records = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        all_items = loader()
        item_map = {it["id"]: it for it in all_items}

        empties = []
        for i, r in enumerate(records):
            if not r.get("response", "").strip():
                qid = r["question_id"]
                if qid in item_map:
                    empties.append((i, r, item_map[qid]))

        if empties:
            empty[bench] = (jsonl_path, records, empties)
            print(f"  {bench}: {len(empties)} empty out of {len(records)}")
        else:
            print(f"  {bench}: all good ({len(records)} records)")

    return empty


async def recover_item(client, item, retries=2):
    """Query API with timeout. Returns response string or empty."""
    for attempt in range(1, retries + 1):
        try:
            resp = await asyncio.wait_for(
                client.complete(item["prompt"], system=item.get("system")),
                timeout=ITEM_TIMEOUT,
            )
            if resp and resp.strip():
                return resp
        except asyncio.TimeoutError:
            print(f"  [{item['id']}] TIMEOUT attempt {attempt}")
        except Exception as e:
            print(f"  [{item['id']}] ERROR attempt {attempt}: {e}")
    return ""


async def main(model_name: str):
    print(f"\n=== Recovering empty responses for {model_name} ===")
    print(f"=== Timeout: {ITEM_TIMEOUT}s per item, ALL parallel ===\n")

    empty = find_empty_responses(model_name)
    if not empty:
        print("\nNothing to recover!")
        return

    total = sum(len(v[2]) for v in empty.values())
    print(f"\nTotal: {total} items, firing ALL at once\n")

    client = build_client(model_name)

    # Build flat list of all tasks across all benchmarks
    all_tasks = []  # (bench, idx, record, item, future)
    for bench, (jsonl_path, all_records, empties) in empty.items():
        for idx, record, item in empties:
            coro = recover_item(client, item)
            all_tasks.append((bench, idx, record, item, coro))

    print(f"Launching {len(all_tasks)} parallel requests...")

    # Fire everything
    coros = [t[4] for t in all_tasks]
    responses = await asyncio.gather(*coros, return_exceptions=True)

    # Process results
    recovered_count = 0
    failed_count = 0
    begus_to_judge = []

    for (bench, idx, record, item, _), resp in zip(all_tasks, responses):
        if isinstance(resp, Exception):
            print(f"  [{item['id']}] EXCEPTION: {resp}")
            failed_count += 1
            continue

        if not resp or not resp.strip():
            print(f"  [{item['id']}] EMPTY after all retries")
            failed_count += 1
            continue

        record["response"] = resp
        recovered_count += 1

        # Grade non-begus inline
        if bench == "mmlu":
            record["score"] = grade_mmlu(record["response"], record["ground_truth"])
            print(f"  [{item['id']}] OK ({len(resp)} chars) score={record['score']}")
        elif bench == "livebench":
            record["score"] = grade_livebench(record["response"], record["ground_truth"], record.get("category", ""))
            print(f"  [{item['id']}] OK ({len(resp)} chars) score={record['score']}")
        elif bench == "begus":
            begus_to_judge.append(record)
            print(f"  [{item['id']}] OK ({len(resp)} chars) pending judge")

    print(f"\nRecovered: {recovered_count}/{total}, Failed: {failed_count}/{total}")

    # Judge begus batch
    if begus_to_judge:
        print(f"\nRunning LLM judge on {len(begus_to_judge)} begus responses...")
        openai_client = OpenAIClient()
        judged = await judge_begus_batch(begus_to_judge, openai_client, concurrency=5)
        for r, jr in zip(begus_to_judge, judged):
            r["score"] = jr.get("judge_score", 0)
            r["judge_raw"] = jr.get("judge_raw", "")
            print(f"  [{r['question_id'][:30]}...] score={r['score']}")

    # Save all benchmarks
    print("\nSaving...")
    for bench, (jsonl_path, all_records, empties) in empty.items():
        with open(jsonl_path, "w") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")
        remaining_empty = sum(1 for r in all_records if not r.get("response", "").strip())
        scored = [r for r in all_records if "score" in r]
        acc = sum(r["score"] for r in scored) / len(scored) if scored else 0
        print(f"  {bench}: {len(all_records)} records, acc={acc:.4f}, still_empty={remaining_empty}")
        print(f"  Saved to {jsonl_path}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.model))

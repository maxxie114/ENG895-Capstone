"""
LLM-as-a-Judge pipeline for the Beguš et al. metalinguistic dataset.

GPT-5.4 acts as the judge, scoring each model response 0 (Fail) or 1 (Pass)
against the ground-truth linguistic analysis.
"""

import asyncio
import re
from tqdm.asyncio import tqdm_asyncio

JUDGE_SYSTEM = (
    "You are an expert linguistics professor grading a student's formal linguistic analysis. "
    "Be strict but fair. Only award a pass if the core linguistic analysis is correct."
)

JUDGE_TEMPLATE = """\
Ground truth analysis:
{ground_truth}

Student's answer:
{response}

Does the student's answer correctly capture the formal linguistic analysis shown in the ground truth?
Reply with ONLY '1' for Pass or '0' for Fail."""


async def _judge_one(record: dict, openai_client, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        prompt = JUDGE_TEMPLATE.format(
            ground_truth=record["ground_truth"],
            response=record["response"],
        )
        try:
            verdict = await openai_client.complete(prompt, system=JUDGE_SYSTEM)
            match = re.search(r"[01]", verdict)
            record["judge_score"] = int(match.group()) if match else 0
        except Exception as e:
            print(f"Judge error for {record.get('question_id')}: {e}")
            record["judge_score"] = 0
        return record


async def judge_begus_batch(
    records: list[dict],
    openai_client,
    concurrency: int = 5,
) -> list[dict]:
    """
    Run GPT-5.4 judge over all Beguš records.
    Lower concurrency than inference (judge calls are expensive).
    """
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_judge_one(r, openai_client, semaphore) for r in records]
    return await tqdm_asyncio.gather(*tasks, desc="LLM judging (Beguš)")

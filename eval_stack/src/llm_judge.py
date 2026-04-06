"""
LLM-as-a-Judge pipeline for the Beguš et al. metalinguistic dataset.

GPT-5.4 acts as the judge, scoring each model response 0 (Fail) or 1 (Pass)
against the ground-truth linguistic analysis.
"""

import asyncio
import re
from tqdm.asyncio import tqdm_asyncio


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks before judging."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return stripped if stripped else text

JUDGE_SYSTEM = (
    "You are an expert linguistics professor grading a student's formal linguistic analysis. "
    "Be strict but fair. Apply the provided rubric exactly."
)

JUDGE_TEMPLATE = """\
TASK: {category} — {subtasks}

GRADING RUBRIC:
{judge_criteria}

STUDENT'S ANSWER:
{response}

Using the rubric above, count how many sub-tasks the student passed (each worth 1 point).
Reply with ONLY a single integer — the total number of sub-tasks passed (e.g. 0, 1, 2, ...)."""


def _count_subtasks(subtasks_str: str) -> int:
    if not subtasks_str:
        return 1
    return len([s for s in subtasks_str.split("|") if s.strip()])


async def _judge_one(record: dict, openai_client, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        n_subtasks = _count_subtasks(record.get("subtasks", ""))
        # For Beguš, the linguistic analysis is often inside <think> blocks.
        # Only strip if there is substantial content outside the think block;
        # otherwise judge the full response so the analysis isn't lost.
        raw = record["response"]
        stripped = _strip_thinking(raw)
        response_for_judge = stripped if len(stripped) >= 100 else raw

        prompt = JUDGE_TEMPLATE.format(
            category=record.get("category", "metalinguistics"),
            subtasks=record.get("subtasks", "overall"),
            judge_criteria=record.get("ground_truth", ""),  # judge_criteria stored in ground_truth field
            response=response_for_judge,
        )
        try:
            verdict = await openai_client.complete(prompt, system=JUDGE_SYSTEM)
            match = re.search(r"\d+", verdict)
            raw = int(match.group()) if match else 0
            # Clamp to valid range and normalise to 0–1
            record["judge_raw"] = min(raw, n_subtasks)
            record["judge_score"] = round(record["judge_raw"] / n_subtasks, 4) if n_subtasks else 0
        except Exception as e:
            print(f"Judge error for {record.get('question_id')}: {e}")
            record["judge_raw"] = 0
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

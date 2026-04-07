"""
Grading logic for MMLU and LiveBench.

- grade_mmlu        → regex exact-match on first A/B/C/D in response
- grade_livebench   → task-specific grading:
    - language/connections: order-agnostic set matching with partial credit,
      following the official LiveBench scorer (LiveBench/LiveBench on GitHub).
    - reasoning: objective regex parsers for logic-puzzle tasks.
"""

import re


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks before grading."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return stripped if stripped else text


def grade_mmlu(response: str, ground_truth: str) -> int:
    """
    Extract the first A/B/C/D letter from the model response and compare
    against ground_truth (also a single letter A-D).
    Returns 1 for correct, 0 for incorrect.
    """
    response = _strip_thinking(response)
    # Look for a standalone answer letter: "Answer: A", "(A)", "A.", "**A**", or just "A"
    # NOTE: patterns 1-4 use IGNORECASE; pattern 5 is case-sensitive to avoid matching
    # the English article "a" as answer choice A.
    patterns_icase = [
        r"(?:answer\s*(?:is|:)?\s*\**)\s*([A-D])\b",  # "answer is C" or "answer is **C**"
        r"\*\*([A-D])\*\*",                             # markdown bold **C**
        r"\(([A-D])\)",                                 # (C)
        r"^([A-D])[.\)]\s",                             # "C." or "C)" at line start
    ]
    for pattern in patterns_icase:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1).upper() == ground_truth.upper())

    # Case-sensitive fallback: only uppercase A/B/C/D to avoid matching article "a"
    match = re.search(r"\b([A-D])\b", response)
    if match:
        return int(match.group(1) == ground_truth.upper())
    return 0


def grade_livebench(response: str, ground_truth: str, category: str = "") -> float:
    """
    Objective grading for LiveBench tasks.

    Routes to task-specific scorers:
    - language (connections): order-agnostic set matching with partial credit,
      matching the official LiveBench scoring methodology.
    - reasoning and others: exact-match with enclosed-answer extraction.
    """
    response = _strip_thinking(response)
    gt = ground_truth.strip()

    if category == "language":
        return _grade_connections(response, gt)

    return _grade_reasoning(response, gt)


def _grade_connections(response: str, ground_truth: str) -> float:
    """
    Score connections (word-grouping) tasks using the official LiveBench method.

    Per the official LiveBench scorer (LiveBench/LiveBench, process_results/
    writing/connections/utils.py):
    - Words are grouped into sets of 4 (order within each group is irrelevant).
    - Each predicted group is checked against the ground-truth groups via set
      equality.
    - Score = correct_groups / total_groups (partial credit: e.g. 0.5 for 1 of
      2 groups correct).
    """
    extracted = _extract_enclosed(response)
    if not extracted:
        extracted = response

    gt_groups = _group_words(ground_truth)
    resp_groups = _group_words(extracted)

    if not gt_groups:
        return 0.0

    correct = sum(1 for rg in resp_groups if rg in gt_groups)
    return round(correct / len(gt_groups), 4)


def _group_words(text: str, group_size: int = 4) -> list[frozenset]:
    """
    Split a comma-separated word list into groups of ``group_size``, each
    represented as a frozenset for order-agnostic comparison.

    LiveBench connections ground truths list words sequentially by group:
    words 1-4 form group 1, words 5-8 form group 2, etc.
    """
    words = [w.strip().lower() for w in text.split(",") if w.strip()]
    groups = []
    for i in range(0, len(words), group_size):
        chunk = words[i:i + group_size]
        if len(chunk) == group_size:
            groups.append(frozenset(chunk))
    return groups


def _grade_reasoning(response: str, ground_truth: str) -> float:
    """
    Objective grading for reasoning / logic-puzzle LiveBench tasks.

    Strategies (applied in order):
    1. Exact string match after normalisation.
    2. Enclosed-answer patterns: [[answer]], <answer>, boxed{answer}.
    3. Last standalone number/token match (for math/reasoning tasks).
    """
    gt = ground_truth.strip()

    # 1. Normalised exact match
    if _normalize(response) == _normalize(gt):
        return 1.0

    # 2. Extract enclosed answer from response
    extracted = _extract_enclosed(response)
    if extracted and _normalize(extracted) == _normalize(gt):
        return 1.0

    # 3. For numeric ground truths, compare last number in response
    if re.fullmatch(r"-?\d+(\.\d+)?", gt):
        nums = re.findall(r"-?\d+(?:\.\d+)?", response)
        if nums and _normalize(nums[-1]) == _normalize(gt):
            return 1.0

    return 0.0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_enclosed(text: str) -> str | None:
    """Extract answer from common enclosure formats used in LiveBench."""
    patterns = [
        r"<solution>(.*?)</solution>",  # LiveBench requested format
        r"\[\[(.+?)\]\]",
        r"\\boxed\{(.+?)\}",
        r"<answer>(.*?)</answer>",
        r"answer:\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None

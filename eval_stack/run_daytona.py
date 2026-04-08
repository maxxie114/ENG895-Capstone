"""
Daytona-powered parallel benchmark runner.

Spins up sandboxes for each model × benchmark combination, runs each in
isolation, downloads results, and merges into a single final_report.csv.

Usage:
    pip install daytona
    python run_daytona.py

Requirements:
    - .env with OPENAI_API_KEY, OPENROUTER_API_KEY
    - DAYTONA_API_KEY set in environment (from app.daytona.io/dashboard/keys)

Note: MiniMax is excluded because its API is blocked from Daytona IPs.
      Run MiniMax separately via mini_runner.py if needed.
"""

import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, Resources

load_dotenv(Path(__file__).parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
MODELS     = ["gpt-5.4", "claude-sonnet-4.6", "glm-5.1"]
BENCHMARKS = ["mmlu", "livebench", "begus"]

# Skip combos that already completed successfully.
SKIP_COMBOS = {
    ("gpt-5.4", "mmlu"), ("gpt-5.4", "livebench"), ("gpt-5.4", "begus"),
    ("claude-sonnet-4.6", "mmlu"), ("claude-sonnet-4.6", "livebench"), ("claude-sonnet-4.6", "begus"),
    ("glm-5.1", "mmlu"),  # 474 records already saved
}

SANDBOX_CPU    = 1
SANDBOX_MEMORY = 1   # GB — keep within free tier (10 GiB total)
EXEC_TIMEOUT   = 10800  # 3 hours per sandbox
BATCH_SIZE     = 3   # sandboxes per batch (3 × 1 GiB = 3 GiB, safely within limits)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Expected item counts per benchmark (for progress reporting)
EXPECTED_COUNTS = {"mmlu": 480, "livebench": 100, "begus": 120}

# Files uploaded to every sandbox
BASE = Path(__file__).parent
UPLOAD_FILES = [
    "src/__init__.py",
    "src/api_clients.py",
    "src/data_loaders.py",
    "src/evaluators.py",
    "src/llm_judge.py",
    "src/runner.py",
    "data/mmlu_dataset.csv",
    "data/livebench_dataset.csv",
    "data/begus_dataset.csv",
    "requirements.txt",
    ".env",
]


# ── Sandbox runner ─────────────────────────────────────────────────────────────
async def run_sandbox(daytona, model: str, benchmark: str) -> list[dict]:
    tag  = f"{model.replace('.', '-')}-{benchmark}"
    name = f"eval-{tag}"
    jsonl_file = f"/eval/results/{model.replace('.', '-')}_{benchmark}.jsonl"
    print(f"[{tag}] Creating sandbox...")

    sandbox = await daytona.create(
        CreateSandboxFromImageParams(
            name=name,
            image=Image.debian_slim("3.12"),
            resources=Resources(cpu=SANDBOX_CPU, memory=SANDBOX_MEMORY),
        )
    )
    print(f"[{tag}] Sandbox ready: {sandbox.id}")

    try:
        # Directory structure
        await sandbox.process.exec("mkdir -p /eval/src /eval/data /eval/results")

        # Upload all files
        print(f"[{tag}] Uploading files...")
        for rel in UPLOAD_FILES:
            local = BASE / rel
            if not local.exists():
                print(f"[{tag}] WARNING: {rel} not found, skipping.")
                continue
            await sandbox.fs.upload_file(local.read_bytes(), f"/eval/{rel}")

        # Install dependencies
        print(f"[{tag}] Installing packages...")
        r = await sandbox.process.exec(
            "pip install -q openai datasets pandas python-dotenv tqdm aiohttp",
            cwd="/eval",
        )
        if r.exit_code != 0:
            raise RuntimeError(f"pip install failed:\n{r.result}")

        # Run directly with long timeout — Daytona exec doesn't support
        # backgrounding (&), so we run synchronously. Progress is monitored
        # via a concurrent polling task.
        expected = EXPECTED_COUNTS.get(benchmark, "?")
        print(f"[{tag}] Running benchmark (expecting ~{expected} items, timeout {EXEC_TIMEOUT}s)...")

        # Start a progress monitor that polls the JSONL line count
        stop_monitor = asyncio.Event()

        async def monitor_progress():
            elapsed = 0
            while not stop_monitor.is_set():
                await asyncio.sleep(30)
                elapsed += 30
                try:
                    r = await sandbox.process.exec(
                        f"wc -l < {jsonl_file} 2>/dev/null || echo 0",
                        timeout=10,
                    )
                    count = int(r.result.strip())
                    pct = f"{count}/{expected}" if isinstance(expected, int) else f"{count}"
                    print(f"[{tag}] {pct} items | {elapsed}s elapsed")
                except Exception:
                    pass  # sandbox might be busy, skip this poll

        monitor_task = asyncio.create_task(monitor_progress())

        try:
            r = await sandbox.process.exec(
                f"cd /eval && python3 -u src/runner.py --model {model} --benchmark {benchmark}",
                cwd="/eval",
                timeout=EXEC_TIMEOUT,
            )
            stop_monitor.set()
            await monitor_task

            if r.exit_code == 0:
                print(f"[{tag}] Benchmark finished successfully")
                # Show last few lines of output
                output_lines = r.result.strip().split("\n")
                for line in output_lines[-5:]:
                    print(f"[{tag}]   {line}")
            else:
                print(f"[{tag}] Benchmark FAILED (exit {r.exit_code})")
                output_lines = r.result.strip().split("\n")
                for line in output_lines[-20:]:
                    print(f"[{tag}]   {line}")
        except Exception as e:
            stop_monitor.set()
            await monitor_task
            print(f"[{tag}] Exec error: {e}")

        # Download results (even partial)
        print(f"[{tag}] Downloading results from {jsonl_file}...")
        try:
            content = await sandbox.fs.download_file(jsonl_file)
            records = []
            for line in content.decode("utf-8").strip().splitlines():
                if line:
                    records.append(json.loads(line))
            print(f"[{tag}] Got {len(records)} records.")

            # Save locally as checkpoint
            local_out = RESULTS_DIR / f"{tag}.jsonl"
            local_out.write_text(content.decode("utf-8"))
            return records
        except Exception as e:
            print(f"[{tag}] Could not download results: {e}")
            return []

    except Exception as e:
        print(f"[{tag}] ERROR: {e}")
        return []

    finally:
        try:
            await daytona.delete(sandbox)
            print(f"[{tag}] Sandbox deleted.")
        except Exception as e:
            print(f"[{tag}] Warning: could not delete sandbox: {e}")


# ── Aggregator ────────────────────────────────────────────────────────────────
def build_report(all_records: list[dict]) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        row = {"model": model}
        for benchmark in BENCHMARKS:
            subset = [
                r for r in all_records
                if r["model"] == model
                and r["benchmark"] == benchmark
                and "score" in r
            ]
            if subset:
                row[f"{benchmark}_accuracy"] = round(
                    sum(r["score"] for r in subset) / len(subset), 4
                )
                row[f"{benchmark}_n"] = len(subset)
            else:
                row[f"{benchmark}_accuracy"] = None
                row[f"{benchmark}_n"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


# ── Cleanup ───────────────────────────────────────────────────────────────────
async def cleanup_sandboxes(daytona):
    """Delete any leftover eval sandboxes so we start with a clean slate."""
    try:
        result = await daytona.list()
        items = result.items
        stale = [s for s in items if getattr(s, 'name', '').startswith('eval-')]
        if stale:
            print(f"Cleaning up {len(stale)} stale sandbox(es)...")
            for s in stale:
                try:
                    await daytona.delete(s)
                    print(f"  Deleted: {s.name} ({s.memory}GiB)")
                except Exception as e:
                    print(f"  Warning: could not delete {s.name}: {e}")
        else:
            print("No stale sandboxes found.")
    except Exception as e:
        print(f"Warning: sandbox cleanup failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    if not os.getenv("DAYTONA_API_KEY"):
        raise EnvironmentError(
            "DAYTONA_API_KEY not set. Get it from https://app.daytona.io/dashboard/keys"
        )

    combos = [(m, b) for m in MODELS for b in BENCHMARKS if (m, b) not in SKIP_COMBOS]
    print(f"Total: {len(combos)} sandboxes in {-(-len(combos) // BATCH_SIZE)} batches of {BATCH_SIZE}")
    print("  " + "\n  ".join(f"{m} × {b}" for m, b in combos))
    print()

    all_combo_results = []
    async with AsyncDaytona() as daytona:
        await cleanup_sandboxes(daytona)
        print()
        for i in range(0, len(combos), BATCH_SIZE):
            batch = combos[i:i + BATCH_SIZE]
            print(f"\n{'='*60}")
            print(f"  Batch {i // BATCH_SIZE + 1}/{-(-len(combos)//BATCH_SIZE)}")
            print(f"  {', '.join(f'{m}×{b}' for m, b in batch)}")
            print(f"{'='*60}")
            batch_results = await asyncio.gather(
                *[run_sandbox(daytona, m, b) for m, b in batch],
                return_exceptions=True,
            )
            all_combo_results.extend(zip(batch, batch_results))

    results_nested = [r for _, r in all_combo_results]
    combos         = [c for c, _ in all_combo_results]

    # Load already-saved checkpoint files for skipped combos
    all_records = []
    for (m, b) in SKIP_COMBOS:
        tag = f"{m.replace('.', '-')}-{b}"
        ckpt = RESULTS_DIR / f"{tag}.jsonl"
        if ckpt.exists():
            with open(ckpt) as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line))
            print(f"[{m} × {b}] Loaded {sum(1 for r in all_records if r.get('model')==m and r.get('benchmark')==b)} records from checkpoint.")

    # Flatten new results
    for (m, b), result in zip(combos, results_nested):
        if isinstance(result, Exception):
            print(f"[{m} × {b}] Failed with exception: {result}")
        else:
            all_records.extend(result)

    print(f"\nTotal records collected: {len(all_records)}")

    # Save merged JSONL
    merged_path = RESULTS_DIR / "raw_responses.jsonl"
    with open(merged_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved: {merged_path}")

    # Build and save final report
    df = build_report(all_records)
    csv_path = RESULTS_DIR / "final_report.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())

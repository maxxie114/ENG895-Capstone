"""
Daytona-powered parallel benchmark runner.

Spins up 9 sandboxes simultaneously (3 models × 3 benchmarks),
runs each combination in isolation, downloads results, and merges
into a single final_report.csv.

Usage:
    pip install daytona
    python run_daytona.py

Requirements:
    - .env with OPENAI_API_KEY, OPENROUTER_API_KEY, MINIMAX_API_KEY
    - DAYTONA_API_KEY set in environment (from app.daytona.io/dashboard/keys)
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
MODELS     = ["gpt-5.4", "claude-sonnet-4.6", "minimax-m2.7"]
BENCHMARKS = ["mmlu", "livebench", "begus"]

SANDBOX_CPU    = 2
SANDBOX_MEMORY = 4   # GB
EXEC_TIMEOUT   = 10800  # 3 hours per sandbox

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

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

        # Run the benchmark
        print(f"[{tag}] Running benchmark (this will take a while)...")
        r = await sandbox.process.exec(
            f"python3 src/runner.py --model {model} --benchmark {benchmark}",
            cwd="/eval",
            timeout=EXEC_TIMEOUT,
        )
        print(f"[{tag}] Exit code: {r.exit_code}")
        if r.exit_code != 0:
            print(f"[{tag}] STDERR:\n{r.result}")
            return []

        # Download results
        remote_path = f"/eval/results/{model.replace('.', '-')}_{benchmark}.jsonl"
        print(f"[{tag}] Downloading results from {remote_path}...")
        content = await sandbox.fs.download_file(remote_path)

        records = []
        for line in content.decode("utf-8").strip().splitlines():
            if line:
                records.append(json.loads(line))

        print(f"[{tag}] Got {len(records)} records.")

        # Also save locally as a checkpoint
        local_out = RESULTS_DIR / f"{tag}.jsonl"
        local_out.write_text(content.decode("utf-8"))

        return records

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


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    if not os.getenv("DAYTONA_API_KEY"):
        raise EnvironmentError(
            "DAYTONA_API_KEY not set. Get it from https://app.daytona.io/dashboard/keys"
        )

    combos = [(m, b) for m in MODELS for b in BENCHMARKS]
    print(f"Launching {len(combos)} sandboxes in parallel...")
    print("  " + "\n  ".join(f"{m} × {b}" for m, b in combos))
    print()

    async with AsyncDaytona() as daytona:
        tasks = [run_sandbox(daytona, m, b) for m, b in combos]
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten
    all_records = []
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

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())

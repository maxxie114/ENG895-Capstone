# DESIGN.md: LLM Evaluation Pipeline (Static, Dynamic, Metalinguistic)

## 1. Project Overview
Build a lightweight, highly concurrent Python evaluation pipeline to benchmark three frontier LLMs across three distinct datasets. The system must execute all API calls, handle rate limits, auto-grade the responses, and generate a final CSV report within a 5-hour window. 

**Target Models & API Providers:**
1. **GPT-5.4** via **OpenAI Official API** (Fallback: OpenRouter).
2. **Claude Sonnet 4.6** via **AWS Bedrock** (using `boto3` to invoke the Bedrock runtime). 
3. **MiniMax M2.7** via **MiniMax Official API** (using the OpenAI-compatible endpoint format).

**Target Benchmarks:**
1. **MMLU (Subset)**: Static multiple-choice reasoning.
2. **LiveBench**: Dynamic, time-stamped task solving.
3. **Beguš et al. Metalinguistics**: 120-item dataset requiring formal linguistic analysis.

---

## 2. Architecture & Tech Stack
*   **Language**: Python 3.10+
*   **Core Libraries**: `openai`, `boto3`, `datasets` (HuggingFace), `pandas`, `asyncio`, `aiohttp`, `python-dotenv`, `tqdm`.
*   **Concurrency**: Use `asyncio` with semaphores to respect rate limits while maximizing throughput to meet the 5-hour deadline.
*   **State Management**: Save intermediate outputs to a local JSONL file after every successful API call to prevent data loss in case of timeouts.

---

## 3. Environment Variables (`.env`)
The agent must generate a `.env.example` file requiring the following keys:
```text
OPENAI_API_KEY=your_openai_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1  # or applicable region
MINIMAX_API_KEY=your_minimax_key
```

---

## 4. API Wrapper Implementations

### A. OpenAI Wrapper (GPT-5.4)
*   **Client**: `AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))`
*   **Model String**: `"gpt-5.4"`

### B. AWS Bedrock Wrapper (Claude Sonnet 4.6)
*   **Client**: `boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION"))`
*   **Endpoint Details**: Use the Bedrock Converse API or `invoke_model`. Note that Bedrock model IDs change; use the standard identifier for Claude 3.5/4.6 Sonnet available in the user's AWS region.

### C. MiniMax Wrapper (M2.7)
*   **Client**: Use the `AsyncOpenAI` client (MiniMax provides an OpenAI-compatible API).
*   **Base URL**: `https://api.minimax.io/v1`
*   **API Key**: `os.getenv("MINIMAX_API_KEY")`
*   **Model String**: `"MiniMax-M2.7"`

**Global Inference Settings:**
*   `temperature=0.0` (for maximum determinism and reproducibility).
*   `max_tokens=4096`.

---

## 5. Dataset Loaders & Execution Logic

### Benchmark 1: MMLU Subset (Static)
*   **Source**: Load ~600 samples from HuggingFace `datasets` (`cais/mmlu`, subjects: logic, formal_logic, abstract_algebra, etc.).
*   **Prompting**: 5-shot multiple choice. 
*   **Grading**: Regex exact match. Extract the first letter (A, B, C, or D) from the output and compare it to the ground truth.

### Benchmark 2: LiveBench (Dynamic)
*   **Source**: Download the latest release from HuggingFace (`livebench/livebench`). Filter to a 100-question subset of the `reasoning` and `language` tasks to save time.
*   **Prompting**: Use the built-in system prompts provided by the LiveBench dataset.
*   **Grading**: Implement the LiveBench objective regex parsers (e.g., searching for specific enclosed answers or JSON formats). 

### Benchmark 3: Beguš et al. Dataset (Metalinguistics)
*   **Source**: Load the 120-item local CSV/JSON dataset (spanning ambiguity, syntax, phonology).
*   **Prompting**: Zero-shot. 
    *   *System Prompt*: "You are an expert computational linguist. Analyze the following linguistic data..."
*   **Grading (LLM-as-a-Judge)**: 
    *   Since human grading takes too long, configure a secondary pipeline where **GPT-5.4** acts as the judge.
    *   *Judge Prompt*: "You are an expert linguistics professor grading a student's analysis. Ground truth: {ground_truth}. Student answer: {model_response}. Does the student answer correctly capture the formal linguistic analysis? Reply with ONLY '1' for Pass or '0' for Fail."

---

## 6. Required Deliverables from Coding Agent
When the coding agent processes this document, it must output the following directory structure:

```text
/eval_stack
  ├── main.py                 # Async orchestrator and entry point
  ├── .env.example            # Environment variables template
  ├── requirements.txt        # Python dependencies
  ├── /src
  │   ├── api_clients.py      # OpenAI, Bedrock, MiniMax wrappers
  │   ├── data_loaders.py     # HF Dataset pulling and formatting
  │   ├── llm_judge.py        # GPT-5.4 automated grading logic
  │   └── evaluators.py       # Exact match and LiveBench regex logic
  └── /results                # Directory for JSONL checkpoints and final CSV
```

## 7. Execution Flow
1. **Initialize**: Load `.env` and initialize all 3 API clients.
2. **Fetch Data**: Pull MMLU, LiveBench, and load the Linguistics dataset.
3. **Run Inference**: Use `asyncio.gather` with a semaphore (e.g., limit to 5-10 concurrent requests to avoid API rate limits). Save outputs to `results/raw_responses.jsonl` continuously.
4. **Auto-Grade**: 
   - Apply regex to MMLU and LiveBench. 
   - Pass the 480 Linguistics responses (120 items × 4 models) back through the GPT-5.4 Judge pipeline.
5. **Aggregate**: Output a final `results/final_report.csv` containing overall accuracy percentages for each model across the three benchmarks.
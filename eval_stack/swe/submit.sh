#!/usr/bin/env bash
# Submit a completed run's preds.json to sb-cli for SWE-bench Verified grading.
# Run from local machine after sync_loop has pulled the run down.
#
# Usage:
#   ./submit.sh results/glm51_2026-05-13_2230 [run_id]

set -euo pipefail

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  echo "Usage: $0 <run_dir> [run_id]"
  echo "  e.g.  $0 results/glm51_2026-05-13_2230 glm51-v1"
  exit 1
fi

RUN_ID="${2:-$(basename "$RUN_DIR")}"
PREDS="$RUN_DIR/preds.json"

[ -f "$PREDS" ] || { echo "Missing $PREDS"; exit 1; }

# Load SB key from gitignored .env
ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env"
SB_KEY=$(grep '^SWEBENCH_API_KEY=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")
[ -n "$SB_KEY" ] || { echo "SWEBENCH_API_KEY missing from $ENV_FILE"; exit 1; }
export SWEBENCH_API_KEY="$SB_KEY"

REPORT_DIR="$RUN_DIR/sb-report"
mkdir -p "$REPORT_DIR"

echo "Submitting:"
echo "  preds  = $PREDS"
echo "  run_id = $RUN_ID"
echo "  report = $REPORT_DIR"
echo

sb-cli submit swe-bench_verified test \
  --predictions_path "$PREDS" \
  --run_id "$RUN_ID" \
  --output_dir "$REPORT_DIR"

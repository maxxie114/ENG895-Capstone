#!/usr/bin/env bash
# Run mini-SWE-agent against SWE-bench Verified using GLM Coding Plan.
# Lives on the droplet at ~/swebench/run_inference.sh.
#
# Usage (on droplet, under nohup):
#   RUN_ID=glm51-pilot SLICE=0:3 WORKERS=4 nohup ./run_inference.sh > /tmp/run.out 2>&1 &
#   RUN_ID=glm51-full WORKERS=24 nohup ./run_inference.sh > /tmp/run.out 2>&1 &

set -euo pipefail

# 1. Source secrets (set -a auto-exports so litellm subprocess inherits them)
set -a; source ~/swebench/.env; set +a

# 2. Provider routing — switch GLM Coding Plan vs OpenRouter via PROVIDER env
PROVIDER="${PROVIDER:-glm}"
case "$PROVIDER" in
  glm)
    export OPENAI_API_KEY="$GLM_API_KEY"
    export OPENAI_BASE_URL="$GLM_BASE_URL"
    MODEL_OVERRIDE=""
    if [[ "$OPENAI_BASE_URL" != *"/coding/"* ]]; then
      echo "REFUSING TO RUN: PROVIDER=glm but OPENAI_BASE_URL is not the coding plan endpoint" >&2
      echo "  got: $OPENAI_BASE_URL" >&2
      exit 1
    fi
    ;;
  openrouter)
    export OPENAI_API_KEY="$OPENROUTER_API_KEY"
    export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
    MODEL_OVERRIDE="-c model.model_name=openai/z-ai/glm-5.1"
    if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
      echo "REFUSING TO RUN: PROVIDER=openrouter but OPENROUTER_API_KEY is unset" >&2
      exit 1
    fi
    ;;
  *)
    echo "PROVIDER must be 'glm' or 'openrouter' (got: $PROVIDER)" >&2
    exit 1
    ;;
esac

# 3. Args (env-overridable)
RUN_ID="${RUN_ID:-glm51_$(date +%Y-%m-%d_%H%M)}"
WORKERS="${WORKERS:-24}"             # inference workers (mini-swe-agent threads)
EVAL_WORKERS="${EVAL_WORKERS:-8}"    # eval workers (swebench harness; <= 75% of vCPU)
SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
SLICE="${SLICE:-}"   # e.g. "0:3" for pilot
DATASET="${DATASET:-princeton-nlp/SWE-Bench_Verified}"

OUTPUT_DIR=~/swebench/runs/$RUN_ID
mkdir -p "$OUTPUT_DIR"

# 4. Activate venv, find default config, layer ours on top
# Note: importing minisweagent prints a banner to stdout, so we silence it
# during import to keep DEFAULT_CFG clean.
source ~/swebench-venv/bin/activate
DEFAULT_CFG=$(python3 -c "
import os, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import minisweagent
print(os.path.join(os.path.dirname(minisweagent.__file__), 'config/benchmarks/swebench.yaml'))
")
OVERRIDE_CFG=~/swebench/config.yaml

ARGS=(
  --subset "$SUBSET"
  --split  "$SPLIT"
  --workers "$WORKERS"
  -o "$OUTPUT_DIR"
  -c "$DEFAULT_CFG"
  -c "$OVERRIDE_CFG"
)
# Provider-specific model name override (OpenRouter wraps "z-ai/glm-5.1" prefix)
[ -n "$MODEL_OVERRIDE" ] && ARGS+=($MODEL_OVERRIDE)
[ -n "$SLICE" ] && ARGS+=(--slice "$SLICE")

# 5. Banner + run, tee everything to run.log
# Write our PID so external watchers can poll with `kill -0 $(cat run.pid)`.
echo $$ > "$OUTPUT_DIR/run.pid"
trap "rm -f '$OUTPUT_DIR/run.pid'" EXIT

{
  echo "================================================================"
  echo "[$(date -Is)] starting (pid=$$)"
  echo "  run_id    = $RUN_ID"
  echo "  provider  = $PROVIDER"
  echo "  subset    = $SUBSET / $SPLIT"
  echo "  slice     = ${SLICE:-<all 500>}"
  echo "  workers   = $WORKERS"
  echo "  output    = $OUTPUT_DIR"
  echo "  endpoint  = $OPENAI_BASE_URL"
  echo "  override  = ${MODEL_OVERRIDE:-(none)}"
  echo "================================================================"
} | tee -a "$OUTPUT_DIR/run.log"

mini-extra swebench "${ARGS[@]}" 2>&1 | tee -a "$OUTPUT_DIR/run.log"
MINI_RC=${PIPESTATUS[0]}

# 6. Auto-grade with local SWE-bench harness (sb-cli backend is broken upstream).
# Same harness logic, same verdicts. Runs Docker per instance, prunes per-instance
# images via --cache_level env to keep disk under control.
PREDS="$OUTPUT_DIR/preds.json"
{
  echo
  echo "================================================================"
  echo "[$(date -Is)] inference exit=$MINI_RC; starting local harness eval"
  echo "  dataset      = $DATASET"
  echo "  eval_workers = $EVAL_WORKERS"
  echo "  cache_level  = env"
  echo "================================================================"
} | tee -a "$OUTPUT_DIR/run.log"

if [ ! -s "$PREDS" ]; then
  echo "[$(date -Is)] SKIPPING eval: $PREDS missing or empty" | tee -a "$OUTPUT_DIR/run.log"
  exit "$MINI_RC"
fi

cd "$OUTPUT_DIR"
python3 -m swebench.harness.run_evaluation \
  --dataset_name "$DATASET" \
  --split "$SPLIT" \
  --predictions_path "$PREDS" \
  --max_workers "$EVAL_WORKERS" \
  --cache_level env \
  --run_id "$RUN_ID" \
  --report_dir "$OUTPUT_DIR" 2>&1 | tee -a "$OUTPUT_DIR/run.log"
EVAL_RC=${PIPESTATUS[0]}

echo "[$(date -Is)] pipeline finished: inference=$MINI_RC, eval=$EVAL_RC" | tee -a "$OUTPUT_DIR/run.log"

# Surface the report path
REPORT=$(ls "$OUTPUT_DIR"/*."$RUN_ID".json 2>/dev/null | head -1)
[ -n "$REPORT" ] && echo "[$(date -Is)] grading report: $REPORT" | tee -a "$OUTPUT_DIR/run.log"

exit "$EVAL_RC"

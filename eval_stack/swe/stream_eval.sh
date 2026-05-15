#!/usr/bin/env bash
# Streaming eval: runs alongside an active mini-extra inference run.
# Polls preds.json every cycle and feeds new instances into
# swebench.harness.run_evaluation. The harness skips already-evaluated
# instances internally (checks report.json existence per-instance), so calling
# it repeatedly is safe.
#
# Usage (on droplet, under nohup):
#   RUN_ID=glm51-full-1 EVAL_WORKERS=4 INTERVAL=120 \
#     nohup ./stream_eval.sh > /tmp/stream-glm51-full-1.out 2>&1 &
#
# Exits when:
#   1. The inference PID (in run.pid) is gone
#   2. AND one final harness pass has run

set -uo pipefail

RUN_ID="${RUN_ID:?RUN_ID required}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
INTERVAL="${INTERVAL:-120}"
DATASET="${DATASET:-princeton-nlp/SWE-Bench_Verified}"
SPLIT="${SPLIT:-test}"

RUN_DIR=~/swebench/runs/$RUN_ID
PREDS=$RUN_DIR/preds.json
PID_FILE=$RUN_DIR/run.pid
LOG_FILE=$RUN_DIR/stream_eval.log

source ~/swebench-venv/bin/activate

INFERENCE_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
echo "[$(date -Is)] streaming eval started; watching inference pid=${INFERENCE_PID:-unknown}; eval_workers=$EVAL_WORKERS" | tee -a "$LOG_FILE"

while true; do
  RUNNING=true
  if [ -n "$INFERENCE_PID" ] && ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
    RUNNING=false
  fi

  # Pull all instance_ids currently in preds.json
  IDS=$(python3 -c "
import json
try:
    print(' '.join(json.load(open('$PREDS')).keys()))
except Exception:
    pass
" 2>/dev/null)
  NUM=0
  [ -n "$IDS" ] && NUM=$(echo "$IDS" | wc -w)

  if [ "$NUM" -gt 0 ]; then
    echo "[$(date -Is)] $NUM instances in preds.json; harness pass (skips already-done)" | tee -a "$LOG_FILE"
    python3 -m swebench.harness.run_evaluation \
      --dataset_name "$DATASET" \
      --split "$SPLIT" \
      --predictions_path "$PREDS" \
      --instance_ids $IDS \
      --max_workers "$EVAL_WORKERS" \
      --cache_level env \
      --run_id "$RUN_ID" \
      --report_dir "$RUN_DIR" 2>&1 | tee -a "$LOG_FILE"
    HC=${PIPESTATUS[0]}
    echo "[$(date -Is)] harness pass exit=$HC" | tee -a "$LOG_FILE"
  else
    echo "[$(date -Is)] no instances in preds.json yet" | tee -a "$LOG_FILE"
  fi

  if [ "$RUNNING" = "false" ]; then
    echo "[$(date -Is)] inference is done and final harness pass complete; streamer exiting" | tee -a "$LOG_FILE"
    break
  fi

  sleep "$INTERVAL"
done

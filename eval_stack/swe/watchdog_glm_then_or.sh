#!/usr/bin/env bash
# Watchdog with auto-fallback to OpenRouter.
#
# Flow:
#   1. Wait for GLM API to return 200 (poll every 5 min).
#   2. Launch GLM run (and streaming eval).
#   3. Monitor preds.json growth. If preds count stalls > 10 min AND GLM API
#      returns 429, kill GLM run and switch to OpenRouter.
#   4. Run on OpenRouter until preds count == 500.
#   5. Exit.
#
# Usage:
#   nohup ./watchdog_glm_then_or.sh > /tmp/watchdog.out 2>&1 &

set -uo pipefail

RUN_ID="${RUN_ID:-glm51-full-1}"
RUN_DIR=/root/swebench/runs/$RUN_ID
PREDS=$RUN_DIR/preds.json
LOG=$RUN_DIR/watchdog.log
TARGET=500
STALL_SEC=600                     # 10 min of no preds growth
GLM_WORKERS="${GLM_WORKERS:-24}"
OR_WORKERS="${OR_WORKERS:-12}"    # lower on OpenRouter (it costs $)
EVAL_WORKERS="${EVAL_WORKERS:-4}"

mkdir -p "$RUN_DIR"
set -a; source ~/swebench/.env; set +a

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

test_glm() {
  curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$GLM_BASE_URL/chat/completions" \
    -H "Authorization: Bearer $GLM_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-5.1","messages":[{"role":"user","content":"ok"}],"max_tokens":3}' \
    --max-time 15
}

count_preds() {
  python3 -c "import json; print(len(json.load(open('$PREDS'))))" 2>/dev/null || echo 0
}

wait_for_pid_gone() {
  local pid=$1
  while kill -0 "$pid" 2>/dev/null; do sleep 10; done
}

launch_inference() {
  local provider=$1 workers=$2
  cd /root/swebench
  PROVIDER=$provider RUN_ID=$RUN_ID WORKERS=$workers EVAL_WORKERS=$EVAL_WORKERS \
    nohup ./run_inference.sh > /tmp/${provider}-watchdog.out 2>&1 &
  disown
  sleep 5
  cat "$RUN_DIR/run.pid" 2>/dev/null
}

launch_streamer() {
  cd /root/swebench
  RUN_ID=$RUN_ID EVAL_WORKERS=$EVAL_WORKERS INTERVAL=120 \
    nohup ./stream_eval.sh > /tmp/stream-watchdog.out 2>&1 &
  disown
  sleep 2
  pgrep -f "stream_eval.sh" | head -1
}

log "watchdog starting; target=$TARGET; current preds=$(count_preds)"

# ── Phase 1: Wait for GLM quota to come back ─────────────────────────────────
log "phase 1: waiting for GLM API to return 200"
while true; do
  RC=$(test_glm)
  if [ "$RC" = "200" ]; then
    log "GLM API back (HTTP 200)"
    break
  fi
  log "GLM still HTTP $RC; sleep 5min"
  sleep 300
done

# ── Phase 2: Launch GLM run + streamer ───────────────────────────────────────
log "phase 2: launching GLM inference"
GLM_PID=$(launch_inference glm $GLM_WORKERS)
log "GLM inference pid=$GLM_PID"

STREAM_PID=$(launch_streamer)
log "streamer pid=$STREAM_PID"

# ── Phase 3: Monitor for stall on GLM ────────────────────────────────────────
LAST=$(count_preds)
LAST_GROWTH=$(date +%s)
log "phase 3: monitoring; starting preds=$LAST"

while true; do
  sleep 60
  CURR=$(count_preds)
  if [ "$CURR" -gt "$LAST" ]; then
    LAST=$CURR
    LAST_GROWTH=$(date +%s)
  fi

  # done?
  if [ "$CURR" -ge "$TARGET" ]; then
    log "complete on GLM: preds=$CURR/$TARGET"
    exit 0
  fi

  # GLM run died? (unexpected; could be normal completion of remaining work)
  if ! kill -0 "$GLM_PID" 2>/dev/null; then
    log "GLM PID $GLM_PID gone; preds=$CURR/$TARGET"
    if [ "$CURR" -ge "$TARGET" ]; then
      log "all done, exiting"
      exit 0
    fi
    log "incomplete — falling through to OpenRouter"
    break
  fi

  STALL=$(( $(date +%s) - LAST_GROWTH ))
  echo "[$(date -Is)] preds=$CURR/$TARGET; stall=${STALL}s" >> "$LOG"

  if [ "$STALL" -gt "$STALL_SEC" ]; then
    RC=$(test_glm)
    if [ "$RC" = "429" ]; then
      log "STALL: ${STALL}s no growth + GLM=$RC — switching to OpenRouter"
      kill -TERM "$GLM_PID" 2>/dev/null || true
      sleep 5
      pkill -TERM -f "swebench-venv/bin/mini-extra" 2>/dev/null || true
      sleep 5
      break
    else
      log "stall ${STALL}s but GLM=$RC (not 429) — keep watching"
      LAST_GROWTH=$(date +%s)  # reset stall timer
    fi
  fi
done

# ── Phase 4: Switch to OpenRouter ────────────────────────────────────────────
CURR=$(count_preds)
log "phase 4: launching OpenRouter for remaining $((TARGET-CURR)) instances"
OR_PID=$(launch_inference openrouter $OR_WORKERS)
log "OR inference pid=$OR_PID"

# Wait for OR run to complete (no rate limit on OR; just runs until done)
while kill -0 "$OR_PID" 2>/dev/null; do
  CURR=$(count_preds)
  echo "[$(date -Is)] OR running; preds=$CURR/$TARGET" >> "$LOG"
  if [ "$CURR" -ge "$TARGET" ]; then
    log "OR reached target"
    break
  fi
  sleep 120
done

CURR=$(count_preds)
log "watchdog done; final preds=$CURR/$TARGET"

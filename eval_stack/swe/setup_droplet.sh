#!/usr/bin/env bash
# Idempotently push config + run script to droplet. Re-run any time scripts change.
set -euo pipefail

DROPLET="${DROPLET:-root@64.23.189.5}"
HERE="$(cd "$(dirname "$0")" && pwd)"

ssh "$DROPLET" "mkdir -p ~/swebench/runs"
scp "$HERE/config.yaml"                  "$DROPLET:/root/swebench/config.yaml"
scp "$HERE/run_inference.sh"             "$DROPLET:/root/swebench/run_inference.sh"
scp "$HERE/stream_eval.sh"               "$DROPLET:/root/swebench/stream_eval.sh"
scp "$HERE/watchdog_glm_then_or.sh"      "$DROPLET:/root/swebench/watchdog_glm_then_or.sh"
ssh "$DROPLET" "chmod +x ~/swebench/*.sh && ls -la ~/swebench/"

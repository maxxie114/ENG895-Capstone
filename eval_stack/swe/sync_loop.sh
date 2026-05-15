#!/usr/bin/env bash
# Continuously rsync droplet's run dir back to local.
# Run from local machine in its own terminal/tmux while inference is running.
#
# Usage:
#   ./sync_loop.sh
#   INTERVAL_SEC=120 ./sync_loop.sh

set -euo pipefail

DROPLET="${DROPLET:-root@64.23.189.5}"
REMOTE_DIR="${REMOTE_DIR:-/root/swebench/runs/}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")" && pwd)/results/}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"

mkdir -p "$LOCAL_DIR"
echo "Syncing $DROPLET:$REMOTE_DIR -> $LOCAL_DIR every ${INTERVAL_SEC}s (Ctrl-C to stop)"

while true; do
  echo "[$(date -Is)] rsync starting..."
  if rsync -av --append-verify "$DROPLET:$REMOTE_DIR" "$LOCAL_DIR"; then
    echo "[$(date -Is)] rsync done"
  else
    echo "[$(date -Is)] rsync failed (exit $?); will retry next cycle" >&2
  fi
  sleep "$INTERVAL_SEC"
done

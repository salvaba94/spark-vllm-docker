#!/bin/bash
#
# watchdog-vllm.sh - Check vLLM health and restart if unresponsive
#
# Intended to be run via cron every 2 minutes.
# Checks the /health endpoint; restarts the container if it fails
# multiple consecutive times (to avoid restarting during model load).
#

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="${SPARK_VLLM_RECIPE:-recipes/qwen3.5-122b-int4-autoround.yaml}"
PORT="${SPARK_VLLM_PORT:-8000}"
LOG="/var/log/spark-vllm-watchdog.log"
FAIL_COUNT_FILE="/tmp/spark-vllm-watchdog-failures"
MAX_FAILURES=3  # Restart after this many consecutive failures (~6 min)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Read current failure count
FAILURES=0
if [ -f "$FAIL_COUNT_FILE" ]; then
    FAILURES=$(cat "$FAIL_COUNT_FILE" 2>/dev/null || echo 0)
fi

# Check if any vllm container is running
RUNNING=$(docker ps --filter "ancestor=vllm-node-tf5" --format "{{.Names}}" 2>/dev/null | head -1)
if [ -z "$RUNNING" ]; then
    log "No vLLM container running — skipping health check"
    echo 0 > "$FAIL_COUNT_FILE"
    exit 0
fi

# Health check
if curl -sf --connect-timeout 5 --max-time 10 "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    # Healthy — reset counter
    if [ "$FAILURES" -gt 0 ]; then
        log "Health check passed (recovered after $FAILURES failure(s))"
    fi
    echo 0 > "$FAIL_COUNT_FILE"
    exit 0
fi

# Health check failed
FAILURES=$((FAILURES + 1))
echo "$FAILURES" > "$FAIL_COUNT_FILE"
log "Health check failed ($FAILURES/$MAX_FAILURES) for container $RUNNING"

if [ "$FAILURES" -lt "$MAX_FAILURES" ]; then
    exit 0
fi

# Too many consecutive failures — restart
log "Max failures reached, restarting..."
echo 0 > "$FAIL_COUNT_FILE"

cd "$SCRIPT_DIR"
docker stop "$RUNNING" >> "$LOG" 2>&1 || true
docker rm "$RUNNING" >> "$LOG" 2>&1 || true

log "Starting model with recipe: $RECIPE"
./run-recipe.sh "$RECIPE" >> "$LOG" 2>&1 &

log "Restart initiated"

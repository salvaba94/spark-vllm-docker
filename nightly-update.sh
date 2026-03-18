#!/bin/bash
#
# nightly-update.sh - Pull latest wheels, rebuild image, and restart the model
#
# Intended to be run via cron. Logs to /var/log/spark-vllm-nightly.log
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="${SPARK_VLLM_RECIPE:-recipes/qwen3.5-122b-int4-autoround.yaml}"
LOG="/var/log/spark-vllm-nightly.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Nightly update started ==="
cd "$SCRIPT_DIR"

# Pull latest repo changes
log "Pulling latest changes..."
git pull origin main >> "$LOG" 2>&1

# Rebuild image (downloads latest wheels automatically)
log "Rebuilding image..."
if ! ./build-and-copy.sh --tf5 -t vllm-node-tf5 >> "$LOG" 2>&1; then
    log "ERROR: Image build failed, aborting restart"
    exit 1
fi

# Find and stop any running sparkrun container
RUNNING=$(docker ps --filter "ancestor=vllm-node-tf5" --format "{{.Names}}" 2>/dev/null || true)
if [ -n "$RUNNING" ]; then
    log "Stopping running container(s): $RUNNING"
    echo "$RUNNING" | xargs -r docker stop >> "$LOG" 2>&1
    echo "$RUNNING" | xargs -r docker rm >> "$LOG" 2>&1
fi

# Restart with recipe
log "Starting model with recipe: $RECIPE"
./run-recipe.sh "$RECIPE" >> "$LOG" 2>&1 &

log "=== Nightly update complete ==="

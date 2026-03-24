#!/bin/bash
# Nightly update: rebase on eugr's latest, rebuild if needed, restart vLLM
#
# Runs via cron at 6am daily:
#   0 6 * * * /home/rob/spark-vllm-docker/nightly-update.sh
#
# What it does:
#   1. Fetch eugr's latest main
#   2. Rebase current branch on top
#   3. If Dockerfile or build scripts changed, rebuild the container image
#   4. Restart the active recipe (docker --restart unless-stopped handles recovery)
#
# Logs to /home/rob/spark-vllm-docker/nightly-update.log

set -euo pipefail

REPO_DIR="/home/rob/spark-vllm-docker"
LOG="$REPO_DIR/nightly-update.log"
RECIPE="recipes/mistral-small-4-119b-nvfp4.yaml"
UPSTREAM_REMOTE="origin"
UPSTREAM_BRANCH="main"

exec >> "$LOG" 2>&1
echo ""
echo "============================================"
echo "  Nightly update: $(date)"
echo "============================================"

cd "$REPO_DIR"

# Record state before
BEFORE=$(git rev-parse HEAD)
BRANCH=$(git branch --show-current)
echo "Branch: $BRANCH at $BEFORE"

# Fetch upstream
echo "Fetching $UPSTREAM_REMOTE/$UPSTREAM_BRANCH..."
git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"

UPSTREAM_HEAD=$(git rev-parse "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH")
echo "Upstream head: $UPSTREAM_HEAD"

# Check if rebase needed
if git merge-base --is-ancestor "$UPSTREAM_HEAD" HEAD; then
    echo "Already up to date with upstream."
else
    echo "Rebasing $BRANCH onto $UPSTREAM_REMOTE/$UPSTREAM_BRANCH..."
    if git rebase "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"; then
        echo "Rebase successful."
    else
        echo "ERROR: Rebase failed. Aborting rebase and skipping update."
        git rebase --abort
        exit 1
    fi
fi

AFTER=$(git rev-parse HEAD)

# Check if rebuild is needed (Dockerfile, build scripts, or mods changed)
REBUILD_NEEDED=false
if [ "$BEFORE" != "$AFTER" ]; then
    CHANGED_FILES=$(git diff --name-only "$BEFORE" "$AFTER")
    echo "Changed files since last run:"
    echo "$CHANGED_FILES" | head -20

    if echo "$CHANGED_FILES" | grep -qE "^(Dockerfile|build-and-copy\.sh|requirements)"; then
        REBUILD_NEEDED=true
        echo "Build files changed — rebuild needed."
    fi
fi

# Rebuild if needed
if [ "$REBUILD_NEEDED" = true ]; then
    echo "Rebuilding container image..."
    python3 "$REPO_DIR/run-recipe.py" "$RECIPE" --build-only -j 20
    echo "Rebuild complete."
fi

# Restart the recipe
echo "Restarting $RECIPE..."
docker stop vllm_node 2>/dev/null || true
docker rm vllm_node 2>/dev/null || true
# Small delay to let GPU memory free
sleep 5
python3 "$REPO_DIR/run-recipe.py" "$RECIPE" -d
echo "Restart initiated."

# Wait for health check
echo "Waiting for health check..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "Server healthy after ${i}0s"
        break
    fi
    sleep 10
done

echo "Nightly update complete at $(date)"
echo "  Before: $BEFORE"
echo "  After:  $AFTER"
echo "  Rebuild: $REBUILD_NEEDED"

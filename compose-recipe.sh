#!/bin/bash
#
# compose-recipe.sh - Launch a recipe with Docker Compose (vLLM + Open WebUI)
#
# Usage: ./compose-recipe.sh <recipe-name> [options] [docker compose up flags...]
#
# Options:
#   -e VAR=VALUE      Override environment variable for docker compose (can repeat)
#   --no-build        Skip auto-build even if image is missing
#   --build-only      Build the image and exit without starting compose
#
# Examples:
#   ./compose-recipe.sh qwen3-coder-next-int4-autoround
#   ./compose-recipe.sh qwen3-coder-next-int4-autoround -d
#   ./compose-recipe.sh qwen3-coder-next-nvfp4 -e VLLM_IMAGE=vllm-node-mxfp4 -d
#   ./compose-recipe.sh qwen3.5-122b-fp8 -e HF_TOKEN=hf_xxx -e VLLM_PORT=9000 -d
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="${1:?Usage: ./compose-recipe.sh <recipe-name> [options] [docker compose up flags...]}"
shift

# Parse options
ENV_OVERRIDES=()
NO_BUILD=false
BUILD_ONLY=false
COMPOSE_ARGS=()

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -e)
            ENV_OVERRIDES+=("-e" "$2")
            shift 2
            ;;
        --no-build)
            NO_BUILD=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        *)
            COMPOSE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate recipe exists
RECIPE_FILE=""
for candidate in "$SCRIPT_DIR/recipes/${RECIPE}.yaml" "$SCRIPT_DIR/recipes/${RECIPE}.yml" "$RECIPE"; do
    [[ -f "$candidate" ]] && RECIPE_FILE="$candidate" && break
done
[[ -z "$RECIPE_FILE" ]] && { echo "Error: Recipe not found: $RECIPE"; exit 1; }

echo "Recipe: $RECIPE"

# Determine the image name for this recipe (from env override or .env file or default)
VLLM_IMAGE=""
for override in "${ENV_OVERRIDES[@]}"; do
    if [[ "$override" == VLLM_IMAGE=* ]]; then
        VLLM_IMAGE="${override#VLLM_IMAGE=}"
    fi
done
if [[ -z "$VLLM_IMAGE" && -f "$SCRIPT_DIR/.env" ]]; then
    VLLM_IMAGE=$(grep -E '^VLLM_IMAGE=' "$SCRIPT_DIR/.env" | cut -d= -f2-)
fi
VLLM_IMAGE="${VLLM_IMAGE:-vllm-node}"

# Auto-build if image is missing
if [[ "$NO_BUILD" == false ]]; then
    if ! docker image inspect "$VLLM_IMAGE" &>/dev/null; then
        echo "Image '$VLLM_IMAGE' not found locally. Building..."
        "$SCRIPT_DIR/run-recipe.py" "$RECIPE" --build-only || { echo "Build failed."; exit 1; }
    else
        echo "Image '$VLLM_IMAGE' found."
    fi
fi

[[ "$BUILD_ONLY" == true ]] && { echo "Build complete."; exit 0; }

exec docker compose \
    -f "$SCRIPT_DIR/docker-compose.yaml" \
    --env-file /dev/null \
    -e "RECIPE=$RECIPE" \
    -e "VLLM_IMAGE=$VLLM_IMAGE" \
    "${ENV_OVERRIDES[@]}" \
    up "${COMPOSE_ARGS[@]}"

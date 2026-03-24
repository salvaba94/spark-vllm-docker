#!/bin/bash
# LLM Quality Bake-off: Qwen3.5-122B vs Qwen3.5-35B vs Nemotron-3-Super
# Uses lm-evaluation-harness with gsm8k, ifeval, gpqa
#
# Usage: ./bakeoff.sh <model-name> <recipe>
#   e.g.: ./bakeoff.sh qwen122b recipes/qwen3.5-122b-a10b-nvfp4.yaml

set -e

MODEL_LABEL="${1:?Usage: $0 <model-label> <recipe>}"
RECIPE="${2:?Usage: $0 <model-label> <recipe>}"
RESULTS_DIR="./bakeoff-results/${MODEL_LABEL}"
PORT=8000

echo "=== Bake-off: ${MODEL_LABEL} ==="
echo "Recipe: ${RECIPE}"
echo "Results: ${RESULTS_DIR}"
echo ""

# Get model name from running server
echo "Detecting model name from server..."
MODEL_NAME=$(curl -s http://localhost:${PORT}/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Model: ${MODEL_NAME}"
echo ""

mkdir -p "${RESULTS_DIR}"

# Run lm-eval benchmarks
echo "=== Running lm-eval benchmarks ==="
echo "Tasks: gsm8k_cot_llama, ifeval"
echo ""

LIMIT="${3:-100}"

lm_eval --model local-chat-completions \
  --tasks gsm8k_cot_llama,ifeval \
  --limit "${LIMIT}" \
  --model_args "model=${MODEL_NAME},base_url=http://localhost:${PORT}/v1/chat/completions,num_concurrent=32,max_retries=5,tokenized_requests=False" \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path "${RESULTS_DIR}" \
  --log_samples \
  2>&1 | tee "${RESULTS_DIR}/benchmark.log"

echo ""
echo "=== Results saved to ${RESULTS_DIR} ==="

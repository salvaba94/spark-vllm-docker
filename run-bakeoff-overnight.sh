#!/bin/bash
# Quality eval bakeoff — runs all 3 models sequentially, 4 concurrent, 4096 max tokens
# Resilient: if a server crashes mid-eval, captures what we got and moves on

wait_for_server() {
    echo "Waiting for server to come up (up to 20 min)..."
    for i in $(seq 1 240); do
        if curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null; then
            echo "Server is ready!"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Server didn't come up after 20 minutes"
    return 1
}

run_eval() {
    local label="$1"
    local recipe="$2"

    echo "========================================="
    echo "EVAL: $label"
    echo "Recipe: $recipe"
    echo "========================================="

    docker rm -f vllm_node 2>/dev/null || true
    python3 run-recipe.py "$recipe" --solo -d

    if ! wait_for_server; then
        echo "Server failed to start for $label"
        mkdir -p "bakeoff-results/$label"
        docker logs vllm_node 2>&1 | tail -100 > "bakeoff-results/$label/crash.log"
        return 1
    fi

    # Run eval — don't let failures kill the script
    python3 eval-quality.py "$label" || echo "WARNING: eval-quality.py exited with error for $label"

    echo "$label done!"
    echo ""
}

# Run all 3 models
run_eval "nemotron3-super-nvfp4" "recipes/nemotron-3-super-nvfp4.yaml"
run_eval "qwen122b-nvfp4" "recipes/qwen3.5-122b-a10b-nvfp4.yaml"
run_eval "qwen122b-autoround" "recipes/qwen3.5-122b-a10b-int4-autoround.yaml"

echo "========================================="
echo "ALL EVALS COMPLETE"
echo "========================================="
echo "Results in:"
ls -la bakeoff-results/*/eval-quality.json 2>/dev/null || echo "  (some results may be missing)"

#!/usr/bin/env python3
"""Benchmark MTP speculative decoding throughput.

Methodology:
- Uses diverse prompts to avoid caching artifacts
- Discards first 2 runs as warmup (Triton autotuning, JIT)
- Reports mean, std, min, max over N measured runs
- Measures wall-clock time via API (end-to-end, includes scheduling)
- Reports both tok/s and TPOT for comparison
- Uses temperature=0.7 for realistic decode (not greedy)
"""
import requests
import time
import sys
import json
import statistics

MODEL = None  # auto-detect from /v1/models
BASE_URL = "http://localhost:8000"
WARMUP_RUNS = 2
MEASURED_RUNS = 8
TARGET_TOKENS = 500

# Diverse prompts to avoid cache effects
PROMPTS = [
    "Write a detailed story about a scientist discovering a new element.",
    "Explain the history of the Roman Empire from founding to fall.",
    "Describe the process of building a compiler from scratch.",
    "Write about the evolution of music from classical to modern genres.",
    "Explain how neural networks learn through backpropagation in detail.",
    "Write a story about an astronaut stranded on Mars.",
    "Describe the economics of renewable energy transition worldwide.",
    "Explain the physics behind black holes and event horizons.",
    "Write about the development of the internet from ARPANET to today.",
    "Describe the biology of how vaccines train the immune system.",
]


def get_model():
    resp = requests.get(f"{BASE_URL}/v1/models")
    models = resp.json()["data"]
    return models[0]["id"]


def run_completion(model, prompt, max_tokens):
    start = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
    )
    elapsed = time.perf_counter() - start
    data = resp.json()
    tokens = data["usage"]["completion_tokens"]
    return tokens, elapsed


def main():
    model = get_model()
    print(f"Model: {model}")
    print(f"Target tokens: {TARGET_TOKENS}")
    print(f"Warmup runs: {WARMUP_RUNS}, Measured runs: {MEASURED_RUNS}")
    print()

    # Warmup
    print("Warming up...", end=" ", flush=True)
    for i in range(WARMUP_RUNS):
        run_completion(model, PROMPTS[i], TARGET_TOKENS)
        print(f"{i+1}", end=" ", flush=True)
    print("done")
    print()

    # Measured runs
    results = []
    for i in range(MEASURED_RUNS):
        prompt = PROMPTS[(i + WARMUP_RUNS) % len(PROMPTS)]
        tokens, elapsed = run_completion(model, prompt, TARGET_TOKENS)
        tok_s = tokens / elapsed
        tpot_ms = elapsed / tokens * 1000
        results.append({"tokens": tokens, "elapsed": elapsed, "tok_s": tok_s, "tpot_ms": tpot_ms})
        print(f"  Run {i+1}: {tokens} tok in {elapsed:.2f}s = {tok_s:.1f} tok/s (TPOT {tpot_ms:.1f}ms)")

    # Statistics
    tok_s_values = [r["tok_s"] for r in results]
    tpot_values = [r["tpot_ms"] for r in results]

    print()
    print(f"=== Results ({MEASURED_RUNS} runs) ===")
    print(f"Throughput: {statistics.mean(tok_s_values):.1f} ± {statistics.stdev(tok_s_values):.1f} tok/s")
    print(f"  min={min(tok_s_values):.1f}, max={max(tok_s_values):.1f}")
    print(f"TPOT: {statistics.mean(tpot_values):.1f} ± {statistics.stdev(tpot_values):.1f} ms")
    print(f"  min={min(tpot_values):.1f}, max={max(tpot_values):.1f}")

    # Also grab MTP metrics
    try:
        metrics = requests.get(f"{BASE_URL}/metrics").text
        for line in metrics.split("\n"):
            if "spec_decode_num_accepted_tokens_per_pos_total" in line and "position" in line and "created" not in line:
                print(f"  {line.strip()}")
            elif "spec_decode_num_drafts_total{" in line and "created" not in line:
                print(f"  {line.strip()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

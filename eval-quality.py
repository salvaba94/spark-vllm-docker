#!/usr/bin/env python3
"""
Quality evaluation: send identical prompts to vLLM, collect responses.
Usage: python3 eval-quality.py <label> [--port 8000]
Results saved to bakeoff-results/<label>/eval-quality.json
"""
import argparse
import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPTS = [
    # --- Coding ---
    {
        "id": "code-1",
        "category": "coding",
        "prompt": "Write a Python function that finds the longest palindromic substring in a given string. Include type hints and handle edge cases."
    },
    {
        "id": "code-2",
        "category": "coding",
        "prompt": "Write a Python class that implements an LRU cache with O(1) get and put operations. Include a usage example."
    },
    {
        "id": "code-3",
        "category": "coding",
        "prompt": "Write a Python function that takes a nested dictionary of arbitrary depth and flattens it into a single-level dictionary with dot-separated keys. For example, {'a': {'b': 1, 'c': {'d': 2}}} becomes {'a.b': 1, 'a.c.d': 2}."
    },
    {
        "id": "code-4",
        "category": "coding",
        "prompt": "Write a Python async function that fetches multiple URLs concurrently using aiohttp, with a configurable concurrency limit (semaphore), retry logic for failed requests, and returns results as a dict mapping URL to response text or error."
    },
    {
        "id": "code-5",
        "category": "coding",
        "prompt": "Write a Python function that parses a cron expression (minute, hour, day-of-month, month, day-of-week) and returns the next N datetime objects when the cron job would fire, given a start time."
    },
    # --- Debugging ---
    {
        "id": "debug-1",
        "category": "debugging",
        "prompt": "This Python code has a bug. Find and fix it, then explain what was wrong:\n\n```python\ndef merge_sorted_lists(list1, list2):\n    result = []\n    i = j = 0\n    while i < len(list1) and j < len(list2):\n        if list1[i] <= list2[j]:\n            result.append(list1[i])\n            i += 1\n        else:\n            result.append(list2[j])\n            j += 1\n    return result\n```"
    },
    {
        "id": "debug-2",
        "category": "debugging",
        "prompt": "This Python code has a subtle concurrency bug. Find it and explain the fix:\n\n```python\nimport threading\n\nclass Counter:\n    def __init__(self):\n        self.count = 0\n    \n    def increment(self):\n        current = self.count\n        self.count = current + 1\n\ncounter = Counter()\nthreads = [threading.Thread(target=counter.increment) for _ in range(1000)]\nfor t in threads: t.start()\nfor t in threads: t.join()\nprint(counter.count)  # Expected: 1000\n```"
    },
    # --- Reasoning ---
    {
        "id": "reason-1",
        "category": "reasoning",
        "prompt": "A farmer has 100 meters of fencing and wants to enclose a rectangular area against an existing wall (so only 3 sides need fencing). What dimensions maximize the enclosed area? Show your work step by step."
    },
    {
        "id": "reason-2",
        "category": "reasoning",
        "prompt": "You have 8 identical-looking balls. One is slightly heavier than the others. Using a balance scale, what is the minimum number of weighings needed to find the heavy ball? Explain your strategy."
    },
    {
        "id": "reason-3",
        "category": "reasoning",
        "prompt": "Three people check into a hotel room that costs $30. They each pay $10. Later, the manager realizes the room is only $25, so he gives the bellboy $5 to return. The bellboy keeps $2 and gives each person $1 back. Now each person paid $9 (total $27), the bellboy has $2. That's $29. Where did the missing dollar go? Explain clearly."
    },
    # --- Instruction following ---
    {
        "id": "instruct-1",
        "category": "instruction-following",
        "prompt": "Write a haiku about recursion. Then explain the concept of recursion in exactly 3 sentences. Then provide a Python example of recursion in no more than 5 lines of code."
    },
    {
        "id": "instruct-2",
        "category": "instruction-following",
        "prompt": "List exactly 5 advantages and exactly 5 disadvantages of microservice architecture vs monolithic architecture. Format as two numbered lists with headers."
    },
    # --- Agentic / tool-use style ---
    {
        "id": "agent-1",
        "category": "agentic",
        "prompt": "You are a coding assistant. The user says: 'My Python script processes a 10GB CSV file and keeps running out of memory. It currently reads the whole file with pandas.read_csv(). How should I fix this?' Provide a concrete, actionable solution with code."
    },
    {
        "id": "agent-2",
        "category": "agentic",
        "prompt": "You are a coding assistant. The user shows you this error:\n\n```\nTraceback (most recent call last):\n  File \"app.py\", line 45, in handle_request\n    data = json.loads(request.body)\n  File \"/usr/lib/python3.12/json/__init__.py\", line 346, in loads\n    return _default_decoder.decode(s)\n  File \"/usr/lib/python3.12/json/decoder.py\", line 337, in decode\n    obj, end = self.raw_decode(s, idx=_w(s, 0).end())\n  File \"/usr/lib/python3.12/json/decoder.py\", line 355, in raw_decode\n    raise JSONDecodeError(\"Expecting value\", s, err.value)\njson.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n```\n\nDiagnose the likely causes and provide a robust fix."
    },
    {
        "id": "agent-3",
        "category": "agentic",
        "prompt": "You are a senior engineer reviewing a pull request. The PR adds a new REST API endpoint. Review this code and provide constructive feedback:\n\n```python\n@app.route('/api/users', methods=['POST'])\ndef create_user():\n    data = request.json\n    username = data['username']\n    email = data['email']\n    password = data['password']\n    \n    conn = sqlite3.connect('users.db')\n    cursor = conn.cursor()\n    cursor.execute(f\"INSERT INTO users (username, email, password) VALUES ('{username}', '{email}', '{password}')\")\n    conn.commit()\n    conn.close()\n    \n    return jsonify({'status': 'created', 'username': username}), 201\n```"
    },
    # --- Math / quantitative ---
    {
        "id": "math-1",
        "category": "math",
        "prompt": "What is the probability that in a group of 30 people, at least two share the same birthday? Show the calculation step by step."
    },
    {
        "id": "math-2",
        "category": "math",
        "prompt": "Solve: A train leaves station A at 60 mph. Two hours later, another train leaves station A on a parallel track at 90 mph. How far from station A will the second train catch the first? Show your work."
    },
    # --- System design ---
    {
        "id": "design-1",
        "category": "system-design",
        "prompt": "Design a simple rate limiter for an API. Describe the algorithm (token bucket or sliding window), then implement it as a Python class that can check if a request should be allowed. Keep it under 30 lines."
    },
    {
        "id": "design-2",
        "category": "system-design",
        "prompt": "You need to build a simple job queue that processes tasks asynchronously. Workers should pick up jobs, process them, and mark them complete. Describe the architecture briefly, then implement a minimal working version in Python using only the standard library (threading + queue)."
    },
]


def query_model(prompt, port=8000, timeout=300):
    """Send a prompt to the vLLM server and return the response."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "",  # will be auto-detected
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }

    # Auto-detect model name
    try:
        models = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()
        payload["model"] = models["data"][0]["id"]
    except Exception as e:
        return {"error": f"Failed to detect model: {e}", "content": None, "reasoning_content": None}

    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.time() - start
        data = resp.json()

        if "error" in data:
            return {"error": data["error"], "content": None, "reasoning_content": None, "elapsed": elapsed}

        choice = data["choices"][0]["message"]
        return {
            "content": choice.get("content"),
            "reasoning_content": choice.get("reasoning_content"),
            "usage": data.get("usage"),
            "elapsed": elapsed,
            "error": None,
        }
    except requests.exceptions.Timeout:
        return {"error": "timeout", "content": None, "reasoning_content": None, "elapsed": timeout}
    except Exception as e:
        return {"error": str(e), "content": None, "reasoning_content": None, "elapsed": time.time() - start}


def run_eval(label, port=8000):
    """Run all prompts against the model and save results."""
    results_dir = f"./bakeoff-results/{label}"
    os.makedirs(results_dir, exist_ok=True)

    # Detect model
    try:
        models = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()
        model_name = models["data"][0]["id"]
    except Exception as e:
        print(f"ERROR: Cannot reach server: {e}")
        return

    print(f"Model: {model_name}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Concurrency: 4")
    print()

    results = {
        "label": label,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "responses": [None] * len(PROMPTS),
    }

    completed = [0]

    def run_one(idx, prompt_data):
        pid = prompt_data["id"]
        cat = prompt_data["category"]
        prompt = prompt_data["prompt"]
        resp = query_model(prompt, port=port)

        answer = resp.get("content") or resp.get("reasoning_content") or ""
        has_thinking = bool(resp.get("reasoning_content"))
        elapsed = resp.get("elapsed", 0)
        error = resp.get("error")

        completed[0] += 1
        if error:
            print(f"[{completed[0]}/{len(PROMPTS)}] {pid} ({cat}): ERROR: {error}")
        else:
            tokens = resp.get("usage", {}).get("completion_tokens", 0)
            tok_s = tokens / elapsed if elapsed > 0 else 0
            content_len = len(answer) if answer else 0
            print(f"[{completed[0]}/{len(PROMPTS)}] {pid} ({cat}): {elapsed:.1f}s, {tokens} tok ({tok_s:.1f} tok/s), {content_len} chars" +
                  (" [thinking]" if has_thinking else ""))

        return idx, {
            "id": pid,
            "category": cat,
            "prompt": prompt,
            "content": resp.get("content"),
            "reasoning_content": resp.get("reasoning_content"),
            "usage": resp.get("usage"),
            "elapsed": elapsed,
            "error": error,
        }

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_one, i, p) for i, p in enumerate(PROMPTS)]
        for f in as_completed(futures):
            idx, result = f.result()
            results["responses"][idx] = result

    total_time = time.time() - start_all
    print(f"\nAll {len(PROMPTS)} prompts completed in {total_time:.1f}s")

    # Save results
    outfile = os.path.join(results_dir, "eval-quality.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {outfile}")

    # Print summary
    total = len(results["responses"])
    errors = sum(1 for r in results["responses"] if r["error"])
    nulls = sum(1 for r in results["responses"] if not r["content"] and not r["error"])
    with_thinking = sum(1 for r in results["responses"] if r["reasoning_content"])
    avg_elapsed = sum(r["elapsed"] for r in results["responses"]) / total if total else 0

    print(f"\n=== Summary: {label} ({model_name}) ===")
    print(f"Total prompts: {total}")
    print(f"Errors: {errors}")
    print(f"Null content (no answer): {nulls}")
    print(f"Responses with thinking: {with_thinking}")
    print(f"Avg response time: {avg_elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="Label for this eval run")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_eval(args.label, port=args.port)

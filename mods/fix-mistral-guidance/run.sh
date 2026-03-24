#!/bin/bash
# Runtime mod: Cherry-pick vLLM PR #37081 — Mistral Guidance
#
# Applies the full source changes from https://github.com/vllm-project/vllm/pull/37081
# which adds Lark grammar-based constrained decoding for Mistral models:
#
# - Forces [THINK]...[/THINK] generation via grammar when reasoning_effort is set
# - Fixes streaming tool call parsing for post-v15 Mistral tokenizers
# - Adds MistralGrammarFactory with BASE/OPTIONAL_THINK/THINK grammar variants
# - Adds MistralLLGTokenizer wrapper for llguidance engine
# - Wires Lark grammar through structured output backends
#
# Files modified:
#   vllm/entrypoints/openai/chat_completion/serving.py
#   vllm/entrypoints/openai/engine/serving.py
#   vllm/entrypoints/serve/render/serving.py
#   vllm/sampling_params.py
#   vllm/tokenizers/mistral.py
#   vllm/tool_parsers/mistral_tool_parser.py
#   vllm/v1/structured_output/backend_guidance.py
#   vllm/v1/structured_output/backend_types.py
#   vllm/v1/structured_output/backend_xgrammar.py
#   vllm/v1/structured_output/request.py

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/pr37081-src-only.patch"
SITE_PACKAGES=$(python3 -c "import vllm; import os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")

echo "Applying PR #37081 (Mistral Guidance) to $SITE_PACKAGES"

if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: Patch file not found: $PATCH_FILE"
    exit 1
fi

cd "$SITE_PACKAGES"

# Check if already applied by looking for a unique string from the PR
if grep -q "MistralGrammarFactory" "$SITE_PACKAGES/vllm/tool_parsers/mistral_tool_parser.py" 2>/dev/null; then
    echo "  Already applied — skipping"
    exit 0
fi

# Apply the patch
if patch -p1 --forward --batch < "$PATCH_FILE"; then
    echo "  Successfully applied PR #37081"
else
    echo "  WARNING: Patch had errors (some hunks may have already been applied)"
    echo "  Attempting with --force..."
    patch -p1 --forward --batch --force < "$PATCH_FILE" || true
fi

# Verify key change landed
if grep -q "MistralGrammarFactory" "$SITE_PACKAGES/vllm/tool_parsers/mistral_tool_parser.py" 2>/dev/null; then
    echo "  Verified: MistralGrammarFactory present in mistral_tool_parser.py"
else
    echo "  ERROR: Verification failed — MistralGrammarFactory not found"
    exit 1
fi

echo "Done. Mistral guidance (Lark grammar constrained decoding) enabled."

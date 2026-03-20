#!/bin/bash
set -e
# Fix RoPE validation type error (list | set fails, need set | set)
# Merged upstream in transformers PR #43830 — this patch is a no-op if already fixed
ROPE_FILE="/usr/local/lib/python3.12/dist-packages/transformers/modeling_rope_utils.py"
if [ -f "$ROPE_FILE" ] && grep -q 'ignore_keys_at_rope_validation | {"partial_rotary_factor"}' "$ROPE_FILE"; then
    patch -p1 -d /usr/local/lib/python3.12/dist-packages < "$(dirname "$0")/transformers.patch"
else
    echo "RoPE fix already applied or not needed (transformers PR #43830 merged)"
fi

#!/bin/bash
# fix-fla-sm121: Optimize FLA (Flash Linear Attention) Triton kernels for SM121
#
# SM121 (GB10 DGX Spark) was misclassified as Hopper (SM90) because the
# capability check used major >= 9 instead of major == 9. This caused:
# - NUM_WARPS restricted to [2, 4] instead of [2, 4, 8]
# - BKV_LIST restricted to [32, 64] instead of [32, 64, 128]
# - is_tma_supported = True (SM121 has no TMA)
#
# These fixes give ~1.8x faster decode on GDN (Gated Delta Net) layers.
#
# Measured on Qwen3.5-35B: TPOT 64ms → 35ms (1.83x improvement)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Applying FLA SM121 optimizations..."
python3 "$SCRIPT_DIR/patch_fla_sm121.py"

# Clear Triton cache to force recompilation with new configs
if [ -d "$HOME/.triton/cache" ]; then
    rm -rf "$HOME/.triton/cache"/*
    echo "Cleared Triton cache"
fi
echo "Done."

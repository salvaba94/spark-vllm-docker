#!/bin/bash
# Runtime mod: Fix FlashInfer CUTLASS E2M1 for SM121 (GB10)
#
# SM121 has mma.e2m1 tensor cores but lacks the cvt.rn.satfinite.e2m1x2.f32
# PTX instruction. This mod patches the installed FlashInfer package's CUTLASS
# headers to exclude SM121 from CUDA_PTX_FP4FP6_CVT_ENABLED, forcing CUTLASS
# to use its software fallback for FP4 conversions during JIT compilation.
#
# Reference: https://github.com/Avarok-Cybersecurity/dgx-vllm

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =========================================================================
# FIX 0: Prevent FlashInfer FP4 quantization JIT from redirecting SM121→120f
#
# FlashInfer's fp4_quantization.py redirects SM121 to the "120f" family
# backend when CUDA >= 12.9. This compiles with compute_120f, setting
# __CUDA_ARCH__=1200 instead of 1210. Our SM121 software E2M1 guards
# use __CUDA_ARCH__==1210, so the redirect causes them to miss — and the
# cvt.rn.satfinite.e2m1x2.f32 PTX (which SM121 lacks) gets emitted → NaN.
# =========================================================================
FP4_QUANT_PY=$(python3 -c "
import flashinfer, os
print(os.path.join(os.path.dirname(flashinfer.__file__), 'fp4_quantization.py'))
" 2>/dev/null)

if [ -n "$FP4_QUANT_PY" ] && [ -f "$FP4_QUANT_PY" ]; then
    echo "Patching: $FP4_QUANT_PY"
    if grep -q '"120", "121"' "$FP4_QUANT_PY"; then
        sed -i 's/("120", "121")/("120",)/' "$FP4_QUANT_PY"
        echo "  Removed SM121 from 120f redirect — will compile as 121a"
    elif grep -q '"121", "120"' "$FP4_QUANT_PY"; then
        sed -i 's/("121", "120")/("120",)/' "$FP4_QUANT_PY"
        echo "  Removed SM121 from 120f redirect (alt order) — will compile as 121a"
    else
        echo "  SM121 redirect already removed or pattern not found"
    fi
fi

# =========================================================================
# FIX 0b: SKIPPED — fp4_common.py and float_subbyte.h patches
#
# These CUTLASS header patches were needed when building with 12.0a,
# but now that the image is built with FLASHINFER_CUDA_ARCH_LIST=12.1a,
# SM121 is handled correctly at build time. Applying these runtime patches
# breaks GDN (Gated Delta Net) warmup on Qwen3.5-122B because they alter
# how FlashInfer's JIT compiler handles FP4 conversion in headers shared
# with non-FP4 kernels (GDN attention uses the same CUTLASS includes).
# =========================================================================
echo "Skipping fp4_common.py patch (handled at build time with FLASHINFER_CUDA_ARCH_LIST=12.1a)"

echo "Skipping float_subbyte.h patch (handled at build time with FLASHINFER_CUDA_ARCH_LIST=12.1a)"
echo "quantization_utils.cuh: patched at build time (selective SM121 guards)"

# Clear only the FP4-specific FlashInfer JIT caches.
# DO NOT clear fused_moe — it contains pre-built SM120 CUTLASS kernels that work.
# Clearing fused_moe forces fresh JIT compilation which triggers a GDN warmup
# crash on SM121 (Triton codegen issue with some kernel configs).
CACHE_DIR="$HOME/.cache/flashinfer"
if [ -d "$CACHE_DIR" ]; then
    find "$CACHE_DIR" -type d \( -name "fp4_quantization*" -o -name "rmsnorm*fp4*" \) -exec rm -rf {} + 2>/dev/null || true
    echo "Cleared FlashInfer JIT cache (fp4_quantization, rmsnorm_fp4 only)"
else
    echo "No FlashInfer JIT cache found (first run)"
fi

# Patch TVM FFI to use RTLD_LAZY instead of RTLD_NOW for .so loading.
# FP4×FP4 128×256×128 kernels can't be compiled (SMEM overflow) but the dispatch
# header references them. RTLD_LAZY defers resolution until the symbol is called
# (which it never is — the autotuner skips unsupported shapes at runtime).
TVM_FFI_SO=$(python3 -c "import tvm_ffi; import os; print(os.path.join(os.path.dirname(tvm_ffi.__file__), 'lib', 'libtvm_ffi.so'))" 2>/dev/null)
if [ -n "$TVM_FFI_SO" ] && [ -f "$TVM_FFI_SO" ]; then
    # Replace RTLD_NOW (0x2) with RTLD_LAZY (0x1) in the dlopen call
    python3 -c "
import struct
with open('$TVM_FFI_SO', 'rb') as f:
    data = bytearray(f.read())
# Find 'Failed to load dynamic shared library' and nearby dlopen
marker = b'Failed to load dynamic shared library'
idx = data.find(marker)
if idx > 0:
    print(f'  Found error string at offset {idx}')
    # The dlopen flag RTLD_NOW=2 is typically passed as an immediate
    # We need to find and patch it in the .text section
    # For now, use LD_PRELOAD approach instead
    print('  Will use LD_PRELOAD for lazy loading')
"
    echo "  Creating RTLD_LAZY wrapper..."
    cat > /tmp/lazy_dlopen.c << 'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
void* dlopen(const char* filename, int flags) {
    static void* (*real_dlopen)(const char*, int) = NULL;
    if (!real_dlopen) real_dlopen = dlsym(RTLD_NEXT, "dlopen");
    if (filename && strstr(filename, "fused_moe_120")) {
        flags = (flags & ~RTLD_NOW) | RTLD_LAZY;
    }
    return real_dlopen(filename, flags);
}
CEOF
    gcc -shared -fPIC -o /tmp/lazy_dlopen.so /tmp/lazy_dlopen.c -ldl
    cp /tmp/lazy_dlopen.so /usr/local/lib/lazy_dlopen.so
    # Add to LD_PRELOAD via environment
    echo '/usr/local/lib/lazy_dlopen.so' >> /etc/ld.so.preload
    echo "  Installed RTLD_LAZY wrapper for fused_moe_120.so"
fi

# =========================================================================
# FIX 5: Mamba SSM kernel — recognize SM121 as Blackwell-class
#
# mamba_mixer2.py uses is_device_capability_family(100) to set is_blackwell.
# SM121 is family 120, not 100, so it falls through to a generic code path
# with BLOCK_SIZE_M=4. With prefix caching (Mamba cache 'all' mode) and
# dstate > 64, this causes illegal memory access in selective_state_update.
# =========================================================================
MAMBA_MIXER2=$(python3 -c "
import vllm, os
print(os.path.join(os.path.dirname(vllm.__file__), 'model_executor', 'layers', 'mamba', 'mamba_mixer2.py'))
" 2>/dev/null)

if [ -n "$MAMBA_MIXER2" ] && [ -f "$MAMBA_MIXER2" ]; then
    if grep -q 'is_device_capability_family(100)' "$MAMBA_MIXER2"; then
        sed -i 's/is_device_capability_family(100)/is_device_capability_family(100) or current_platform.is_device_capability_family(120)/' "$MAMBA_MIXER2"
        echo "Patched mamba_mixer2.py: SM121 recognized as Blackwell for SSM kernel"
    else
        echo "mamba_mixer2.py: already patched or pattern not found"
    fi
fi

echo "Done. FlashInfer will use software E2M1 conversion on SM121."

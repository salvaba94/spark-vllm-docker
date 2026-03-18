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
# FIX 0b: Patch CuTe-DSL fp4_common.py to guard cvt_e2m1x8_f32 PTX
#
# The cute_dsl cvt_e2m1x8_f32 emits cvt.rn.satfinite.e2m1x2.f32 PTX
# with NO architecture guard. Add #if __CUDA_ARCH__ != 1210 around the
# PTX path and provide a software fallback for SM121.
# =========================================================================
FP4_COMMON_PY=$(python3 -c "
import flashinfer, os
print(os.path.join(os.path.dirname(flashinfer.__file__), 'cute_dsl', 'fp4_common.py'))
" 2>/dev/null)

if [ -n "$FP4_COMMON_PY" ] && [ -f "$FP4_COMMON_PY" ]; then
    echo "Patching: $FP4_COMMON_PY"
    if grep -q 'cvt.rn.satfinite.e2m1x2.f32' "$FP4_COMMON_PY" && ! grep -q 'SM121' "$FP4_COMMON_PY"; then
        python3 "$SCRIPT_DIR/patch_fp4_common.py" "$FP4_COMMON_PY"
    else
        echo "  Already patched or PTX pattern not found"
    fi
fi

# =========================================================================
# FIX 1: Patch CUTLASS float_subbyte.h — remove SM121 from PTX FP4 CVT
# =========================================================================

# Find the installed FlashInfer float_subbyte.h
FLOAT_SUBBYTE=$(python3 -c "
import flashinfer, os
base = os.path.dirname(flashinfer.__file__)
path = os.path.join(base, 'data', 'cutlass', 'include', 'cutlass', 'float_subbyte.h')
if os.path.exists(path):
    print(path)
else:
    # Try alternate layout
    import glob
    matches = glob.glob(os.path.join(base, '**', 'float_subbyte.h'), recursive=True)
    print(matches[0] if matches else '')
" 2>/dev/null)

if [ -z "$FLOAT_SUBBYTE" ] || [ ! -f "$FLOAT_SUBBYTE" ]; then
    echo "WARNING: float_subbyte.h not found in FlashInfer package, skipping"
    exit 0
fi

echo "Patching: $FLOAT_SUBBYTE"

# Remove SM121A from CUDA_PTX_FP4FP6_CVT_ENABLED
if grep -q 'SM121A_ENABLED' "$FLOAT_SUBBYTE"; then
    sed -i 's/ || defined(CUTLASS_ARCH_MMA_SM121A_ENABLED)//' "$FLOAT_SUBBYTE"
    echo "  Removed SM121A from CUDA_PTX_FP4FP6_CVT_ENABLED"
else
    echo "  SM121A already removed or not present"
fi

# Remove SM121F from CUDA_PTX_FP4FP6_CVT_ENABLED
if grep -q 'SM121F_ENABLED' "$FLOAT_SUBBYTE"; then
    sed -i 's/ || defined(CUTLASS_ARCH_MMA_SM121F_ENABLED)//' "$FLOAT_SUBBYTE"
    echo "  Removed SM121F from CUDA_PTX_FP4FP6_CVT_ENABLED"
else
    echo "  SM121F already removed or not present"
fi

# Note: quantization_utils.cuh patching is now done ONLY at build time by
# fix_quantization_utils_sm121.py which SELECTIVELY guards the fp32_vec_to_e2m1
# functions. We do NOT globally exclude SM121 from all >= 1000 guards here
# because the wrapper functions (cvt_warp_fp16_to_fp4, etc.) need to run on
# SM121 — they do generic float math and call fp32_vec_to_e2m1 which has the
# software fallback. The old global sed caused wrapper functions to return 0
# with uninitialized scale factors → NaN.
echo "quantization_utils.cuh: patched at build time (selective SM121 guards)"

# Clear ALL FlashInfer JIT cache to force recompilation with correct arch flags.
# Both fused_moe and fp4_quantization modules may have been compiled with the
# wrong arch (120f instead of 121a), producing binaries with broken PTX.
CACHE_DIR="$HOME/.cache/flashinfer"
if [ -d "$CACHE_DIR" ]; then
    find "$CACHE_DIR" -type d \( -name "fused_moe*" -o -name "fp4_quantization*" -o -name "rmsnorm*fp4*" \) -exec rm -rf {} + 2>/dev/null || true
    echo "Cleared FlashInfer JIT cache (fused_moe, fp4_quantization, rmsnorm_fp4)"
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

echo "Done. FlashInfer will use software E2M1 conversion on SM121."

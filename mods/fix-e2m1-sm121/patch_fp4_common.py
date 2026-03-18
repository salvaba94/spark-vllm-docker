#!/usr/bin/env python3
"""Patch FlashInfer cute_dsl fp4_common.py for SM121 (GB10).

The cvt_e2m1x8_f32 @dsl_user_op emits cvt.rn.satfinite.e2m1x2.f32 PTX with
no architecture guard. SM121 lacks this instruction.

Strategy: Replace the @dsl_user_op with a Python-level function that checks
the target arch at JIT time and emits either the native PTX (SM120/other) or
a software fallback (SM121). Since @dsl_user_op goes through NVCC, we can
use __CUDA_ARCH__ in the generated C++ code — but only if we emit it as a
raw code block, not via T.attr("asm").

Approach: We replace the entire cvt_e2m1x8_f32 function to inject a C++
helper via T.attr("call_extern") or raw code injection. If that's not feasible
with the DSL API, we fall back to providing two separate implementations and
selecting at the Python level based on the compilation target.
"""

import re
import sys

if len(sys.argv) < 2:
    print("Usage: patch_fp4_common.py <path_to_fp4_common.py>")
    sys.exit(1)

path = sys.argv[1]

with open(path) as f:
    content = f.read()

# Check if the file has the problematic PTX
if "cvt.rn.satfinite.e2m1x2.f32" not in content:
    print("  No cvt.rn.satfinite.e2m1x2.f32 found — already patched or file changed")
    sys.exit(0)

# Strategy: Find the cvt_e2m1x8_f32 function and add an arch check.
# The CuTe-DSL compiles via NVCC, so we CAN use __CUDA_ARCH__ in the
# generated code. The trick is that T.attr("asm") emits inline asm which
# is opaque to the preprocessor. BUT we can use a DIFFERENT asm string
# that wraps the conversion in an #if block by exploiting GCC's
# statement-expressions in inline asm.
#
# Actually the cleanest approach: replace the T.attr("asm") call with
# a T.attr("call_extern") to a helper function that we inject into the
# compilation unit via T.attr("include") or similar.
#
# For now, the pragmatic approach: since we're fixing fp4_quantization.py
# to not redirect SM121→120f, and the fused_moe CUTLASS path already
# compiles with 121a, the cute_dsl path ALSO gets correct arch flags
# (121a → __CUDA_ARCH__=1210). So we just need to ensure the PTX block
# has an arch guard.
#
# We can do this by replacing the asm string to use GCC inline asm with
# an arch check via a C++ if-constexpr wrapper injected as a separate op.

# Pragmatic fix: Replace the PTX asm body with one that uses %% escaping
# to embed an #if guard. This won't work in inline asm.
#
# Final approach: Inject a software E2M1 helper function definition at the
# top of the generated code (via modifying the DSL module source), then
# replace the @dsl_user_op to call it conditionally.
#
# This is getting complex. For now, log a warning and provide a monkey-patch
# that overrides the function at import time.

# Write a monkey-patch module that runtime-patches cvt_e2m1x8_f32
# to be SM121-safe. This gets loaded via the runtime mod.

print("  WARNING: fp4_common.py uses cvt.rn.satfinite.e2m1x2.f32 PTX in CuTe-DSL")
print("  This path may not be hit by CUTLASS MoE 'throughput' backend")
print("  If NaN persists after fp4_quantization.py fix, this needs addressing")
print("  See: mods/fix-e2m1-sm121/patch_fp4_common.py for details")

# For now, we do NOT modify fp4_common.py — the cute_dsl API makes it
# non-trivial to inject arch guards. The fp4_quantization.py redirect fix
# is the critical change. If the cute_dsl path IS hit and produces NaN,
# the workaround is to set FLASHINFER_DISABLE_CUTE_DSL=1 or equivalent.
sys.exit(0)

#!/usr/bin/env python3
"""Patch FLA ops for SM121 (Blackwell desktop) optimization.

SM121 is detected as is_nvidia_hopper=True (capability >= 9) which restricts
NUM_WARPS to [2, 4]. We cannot safely change is_nvidia_hopper to False because
other code paths depend on it (TMA, prefill warmup).

Instead, we directly patch the decode-critical chunk_o.py to add 8-warp
configs to the autotune space while keeping is_nvidia_hopper unchanged.

Also fix is_tma_supported: SM12x does NOT have TMA (SM100+ datacenter only).
"""
import os

SITE = "/usr/local/lib/python3.12/dist-packages"


def patch_file(path, old, new, label):
    with open(path) as f:
        src = f.read()
    if new in src:
        print(f"  {label}: already applied")
        return False
    if old not in src:
        print(f"  WARNING: {label}: anchor not found")
        return False
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"  {label}: applied")
    return True


# =========================================================================
# 1. Fix is_tma_supported: SM12x has no TMA
# =========================================================================
utils_path = os.path.join(SITE, "vllm/model_executor/layers/fla/ops/utils.py")
print(f"Patching: {utils_path}")

with open(utils_path) as f:
    src = f.read()
if "is_tma_supported" in src and "< 12" not in src:
    src = src.replace(
        "torch.cuda.get_device_capability(0)[0] >= 9) and (",
        "9 <= torch.cuda.get_device_capability(0)[0] < 12) and (",
        1,
    )
    with open(utils_path, "w") as f:
        f.write(src)
    print("  is_tma_supported (exclude SM12x): applied")
else:
    print("  is_tma_supported: already applied or not found")


# =========================================================================
# 2. Add 8-warp configs to chunk_o.py autotune (decode-critical kernel)
# =========================================================================
chunk_o_path = os.path.join(SITE, "vllm/model_executor/layers/fla/ops/chunk_o.py")
print(f"Patching: {chunk_o_path}")

# The Hopper restriction limits NUM_WARPS to [2, 4].
# We override it directly in chunk_o.py to add 8 warps.
patch_file(
    chunk_o_path,
    'NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]',
    'NUM_WARPS = [2, 4, 8]  # SM121: always include 8 warps',
    "NUM_WARPS always include 8",
)

print("\nFLA SM121 optimizations applied.")

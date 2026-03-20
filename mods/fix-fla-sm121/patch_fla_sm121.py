#!/usr/bin/env python3
"""Patch FLA ops for SM121 (Blackwell desktop / GB10) optimization.

SM121 has capability (12, 1) which trips all the >= 9 checks designed for
Hopper (SM90). This causes three misclassifications:

1. is_nvidia_hopper = True  →  restricts NUM_WARPS to [2, 4] in chunk_o.py
   (the decode-critical GDN kernel). 8 warps gives 1.83x on 35B eager mode.
2. is_tma_supported = True  →  SM12x desktop has NO TMA (datacenter SM100+ only).
   solve_tril.py passes USE_TMA=True which wastes cycles or hits bugs.

Additionally, GB10 has 101,376 bytes shared mem — just 1KB under the DEFAULT
threshold (102,400) used by check_shared_mem(). This restricts BKV_LIST to
[32, 64] when [64, 128] would mostly fit. We override BKV_LIST in chunk_o.py
to include 128, letting the Triton autotuner skip configs that exceed SMEM.

All fixes applied here. is_nvidia_hopper=False gives NUM_WARPS=[2,4,8]
automatically through the existing conditional in chunk_o.py.
"""
import os

SITE = "/usr/local/lib/python3.12/dist-packages"


def patch_replace(path, old, new, label):
    """Replace old with new in file. Returns True if applied."""
    with open(path) as f:
        src = f.read()
    if new in src:
        print(f"  {label}: already applied")
        return False
    if old not in src:
        print(f"  WARNING: {label}: anchor not found in {path}")
        return False
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"  {label}: applied")
    return True


utils_path = os.path.join(SITE, "vllm/model_executor/layers/fla/ops/utils.py")
print(f"Patching: {utils_path}")

# =========================================================================
# 1. Fix is_nvidia_hopper: SM12x is NOT Hopper (SM90)
#    capability[0] >= 9 catches SM12x. Use range check for SM90-SM99.
# =========================================================================
patch_replace(
    utils_path,
    "torch.cuda.get_device_capability()[0] >= 9",
    "9 <= torch.cuda.get_device_capability()[0] < 12",
    "is_nvidia_hopper (exclude SM12x)",
)

# =========================================================================
# 2. Fix is_tma_supported: SM12x desktop has no TMA
#    Same >= 9 check. TMA is SM90+ datacenter (H100/B200), not GB10.
# =========================================================================
patch_replace(
    utils_path,
    "torch.cuda.get_device_capability(0)[0] >= 9) and (",
    "9 <= torch.cuda.get_device_capability(0)[0] < 12) and (",
    "is_tma_supported (exclude SM12x)",
)

# =========================================================================
# 3. Override BKV_LIST in chunk_o.py to include 128
#    GB10 has 101,376 bytes SMEM — just under the 102,400 DEFAULT threshold.
#    check_shared_mem() returns False, giving BKV_LIST=[32,64].
#    Most BK/BV=128 combos fit at stages=2 (96KB). Triton autotuner
#    automatically skips configs that exceed SMEM, so this is safe.
# =========================================================================
chunk_o_path = os.path.join(SITE, "vllm/model_executor/layers/fla/ops/chunk_o.py")
print(f"Patching: {chunk_o_path}")

patch_replace(
    chunk_o_path,
    "BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]",
    "BKV_LIST = [32, 64, 128]  # SM121: include 128, autotuner skips SMEM-overflow configs",
    "BKV_LIST (include 128)",
)

# =========================================================================
# Summary: with is_nvidia_hopper=False, chunk_o.py automatically gets
# NUM_WARPS=[2,4,8] from the existing conditional.
# With BKV_LIST=[32,64,128], the autotuner has 81 configs to search.
# =========================================================================
print("\nFLA SM121 optimizations applied.")
print("  - is_nvidia_hopper: False (SM12x is not Hopper)")
print("  - is_tma_supported: False (SM12x has no TMA)")
print("  - NUM_WARPS: [2, 4, 8] (via existing chunk_o.py conditional)")
print("  - BKV_LIST: [32, 64, 128] (override check_shared_mem threshold)")

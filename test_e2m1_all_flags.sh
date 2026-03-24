#!/bin/bash
# Test E2M1 PTX instruction across all relevant architecture flags on SM121.
# Determines which compile flags correctly enable cvt.rn.satfinite.e2m1x2.f32.

set -u

ARCHS=(
    "sm_100a"
    "sm_100f"
    "sm_120a"
    "sm_120f"
    "sm_121a"
    "compute_100"
    "compute_120"
)

SRC="/workspace/test_e2m1_flags.cu"
if [ ! -f "$SRC" ]; then
    SRC="test_e2m1_flags.cu"
fi

echo "CUDA version: $(nvcc --version | tail -1)"
echo "Source: $SRC"
echo ""

for arch in "${ARCHS[@]}"; do
    echo "============================================="
    echo "Testing -arch=$arch"
    echo "============================================="

    outfile="/tmp/test_e2m1_${arch}"
    if nvcc -arch="$arch" "$SRC" -o "$outfile" 2>&1; then
        echo "[COMPILE] OK"
        if "$outfile" 2>&1; then
            echo "[RUN] OK"
        else
            echo "[RUN] FAILED (exit code $?)"
        fi
    else
        echo "[COMPILE] FAILED"
    fi
    echo ""
done

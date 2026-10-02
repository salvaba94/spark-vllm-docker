#!/usr/bin/env bash
set -euo pipefail

python_root=${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}
mod_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
patch_file="$mod_dir/vllm-57508.patch"
cd "$python_root"
if git apply --reverse --check --exclude='tests/*' "$patch_file" 2>/dev/null; then
  echo '[mimo-v2.6-fp8-qkv] Upstream vLLM fix already present.'
elif git apply --check --exclude='tests/*' "$patch_file"; then
  git apply --exclude='tests/*' "$patch_file"
  echo '[mimo-v2.6-fp8-qkv] Applied upstream vLLM PR #57508.'
else
  echo '[mimo-v2.6-fp8-qkv] Image is incompatible with the pinned upstream patch.' >&2
  exit 1
fi

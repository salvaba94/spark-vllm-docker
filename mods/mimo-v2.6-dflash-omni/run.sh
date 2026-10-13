#!/usr/bin/env bash
set -euo pipefail

python_root=${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}
mod_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
model_file="$python_root/vllm/model_executor/models/mimo_v2_omni.py"
cd "$python_root"

if grep -A2 '^class MiMoV2OmniForCausalLM' "$model_file" | grep -q 'SupportsEagle3'; then
  echo '[mimo-v2.6-dflash-omni] DFlash capability already present.'
elif git apply --check "$mod_dir/vllm-eagle3.patch"; then
  git apply "$mod_dir/vllm-eagle3.patch"
  echo '[mimo-v2.6-dflash-omni] Applied MiMo omni DFlash capability patch.'
else
  echo '[mimo-v2.6-dflash-omni] Image is incompatible with the pinned patch.' >&2
  exit 1
fi

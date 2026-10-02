# MiMo-V2 FP8 QKV sharding

This mod vendors the model-loader patch from [vLLM PR #57508](https://github.com/vllm-project/vllm/pull/57508), merged 2026-09-19. It corrects MiMo's fused FP8 QKV sharding at tensor parallel size 2. The mod skips itself when the installed vLLM already includes the exact fix and fails if the image's source differs. It makes no network requests at launch.

The patch includes upstream's test file; `run.sh` excludes that file when applying to the installed package. Source: vLLM (Apache-2.0).

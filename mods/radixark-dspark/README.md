# RadixArk Qwen DSpark compatibility

This experimental runtime mod enables Qwen DSpark checkpoints such as
`RadixArk/Qwen3.8-27B-DSpark` on vLLM builds that already contain the Qwen3
DSpark model implementation. The publisher's documented path is SGLang; use
this vLLM adapter only after validating it on the installed vLLM revision.

Those checkpoints declare `architectures: ["DSparkDraftModel"]` and
`model_type: "qwen3"`. Older vLLM code treats every generic
`DSparkDraftModel` as a DeepSeek-V4 draft and rewrites the config to the wrong
loader. The patch detects this Qwen combination and normalizes its architecture
to `Qwen3DSparkModel` before model resolution.

Without that normalization, the DeepSeek fallback can also copy the NVFP4
target's quantization setting onto the unquantized BF16 draft. Routing the
checkpoint before that fallback keeps the draft unquantized.

The mod is idempotent and becomes a no-op when installed vLLM already has
equivalent Qwen draft normalization. It refuses to modify an unknown source
layout.

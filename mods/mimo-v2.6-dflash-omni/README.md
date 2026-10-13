# MiMo V2.6 DFlash capability

The checkpoint resolves to vLLM's multimodal `MiMoV2OmniForCausalLM` class.
Its inner language model already supports the hidden-state hooks used by
DFlash, but this wrapper lacks the `SupportsEagle3` marker. This small patch
adds the marker through eugr's existing mod mechanism. It fails if the image
source no longer matches, and skips when upstream vLLM already includes it.

Source: [MiMo two-Spark recipe patch 02](https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe/blob/main/patches/02-mimo_v2_omni-supports-eagle3.patch)
(MIT license). The patch is applied only to the local container at launch.

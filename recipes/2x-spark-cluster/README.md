# Two-Spark cluster recipes

These recipes require two DGX Spark nodes. Invoke them by relative path, for
example:

```bash
./run-recipe.sh 2x-spark-cluster/inkling-small-nvfp4 --ray
```

Bare recipe names remain supported when the filename is unique. Recipes marked
`solo_only: true` stay in the top-level directory. A checkpoint name containing
“DSpark” denotes a speculative-decoding method, not a two-node deployment.

`mimo-v2.6-flash-rl` uses Xiaomi's official checkpoint with TP2. The recipe
starts with BF16 KV and no speculative decoding so its text/tool path can be
validated independently of the newer DFlash and FP8 KV backends. Its runtime
mod applies the merged upstream vLLM QKV loader fix only when the image lacks
it. After discovery, prepare with `./run-recipe.sh
2x-spark-cluster/mimo-v2.6-flash-rl --setup`; a local checkpoint already in
cache is still copied to the worker. Keep enough free disk for a copy on each
Spark (about 166 GiB each).

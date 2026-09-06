# Two-Spark cluster recipes

These recipes require two DGX Spark nodes. Invoke them by relative path, for
example:

```bash
./run-recipe.sh 2x-spark-cluster/inkling-small-nvfp4-mtp --ray
```

Bare recipe names remain supported when the filename is unique. Recipes marked
`solo_only: true` stay in the top-level directory. A checkpoint name containing
“DSpark” denotes a speculative-decoding method, not a two-node deployment.

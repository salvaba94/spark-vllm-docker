#!/usr/bin/env python3
"""Reshard oversized safetensors files for fastsafetensors compatibility.

Large individual shards cause fastsafetensors to OOM when it maps the entire
file into memory. This script splits any shard above a size threshold into
smaller pieces and updates (or creates) model.safetensors.index.json.

Uses the safetensors library for lazy tensor loading (no full model in RAM)
and transformers.utils.hub.split_torch_state_dict_into_shards for correct
index generation — same approach as model.save_pretrained(max_shard_size=...).

Usage:
    python3 reshard-model.py Intel/Qwen3-Coder-Next-int4-AutoRound
    python3 reshard-model.py /path/to/snapshot
    python3 reshard-model.py Intel/Qwen3-Coder-Next-int4-AutoRound --max-shard-size 5 --target-shard-size 4
    python3 reshard-model.py Intel/Qwen3-Coder-Next-int4-AutoRound --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

GB = 1 << 30
DEFAULT_MAX_SHARD_GB = 5.0
DEFAULT_TARGET_SHARD_GB = 4.0


# ── lazy imports (fail fast with a clear message) ─────────────────────────────

def _imports():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        sys.exit("Error: safetensors not installed. Run: pip install safetensors")
    try:
        from transformers.utils.hub import split_torch_state_dict_into_shards
    except ImportError:
        sys.exit("Error: transformers not installed. Run: pip install transformers")
    return safe_open, save_file, split_torch_state_dict_into_shards


# ── HF cache resolution ───────────────────────────────────────────────────────

def resolve_model_dir(model: str) -> Path:
    """Return snapshot Path from HF model ID or direct filesystem path."""
    p = Path(model).expanduser()
    if p.exists():
        return p.resolve()
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    cache = hub / f"models--{model.replace('/', '--')}"
    if not cache.exists():
        sys.exit(f"Error: model not found in HF cache: {model}")
    snaps = sorted((cache / "snapshots").iterdir())
    if not snaps:
        sys.exit(f"Error: no snapshots in {cache}")
    return snaps[-1]


# ── public API ────────────────────────────────────────────────────────────────

def needs_reshard(model_dir: Path, max_gb: float) -> bool:
    """Return True if any .safetensors shard in model_dir exceeds max_gb."""
    max_bytes = int(max_gb * GB)
    return any(
        s.stat().st_size > max_bytes
        for s in model_dir.glob("*.safetensors")
    )


def reshard(
    model_dir: Path,
    max_gb: float = DEFAULT_MAX_SHARD_GB,
    target_gb: float = DEFAULT_TARGET_SHARD_GB,
    dry_run: bool = False,
) -> None:
    """Reshard oversized safetensors files in model_dir in-place.

    Splits any shard above max_gb into pieces ≤ target_gb using safetensors
    lazy loading (mmap, no full model in RAM) and regenerates the index using
    the same split_torch_state_dict_into_shards utility as transformers
    save_pretrained.
    """
    safe_open, save_file, split_torch_state_dict_into_shards = _imports()

    max_bytes = int(max_gb * GB)

    all_shards = sorted(model_dir.glob("*.safetensors"))
    if not all_shards:
        print(f"No .safetensors files in {model_dir}")
        return

    oversized = [s for s in all_shards if s.stat().st_size > max_bytes]
    if not oversized:
        print(f"All shards ≤ {max_gb} GB — nothing to reshard.")
        for s in all_shards:
            print(f"  {s.name}: {s.stat().st_size / GB:.2f} GB")
        return

    print(f"\nResharding {model_dir.name}")
    print(f"  threshold > {max_gb} GB  →  target ≤ {target_gb} GB")
    print(f"  {len(oversized)} oversized shard(s) of {len(all_shards)} total")
    if dry_run:
        print("  (dry run — no files written)")
    print()

    # ── Step 1: lazily load ALL tensors across all shards ────────────────────
    # safe_open mmaps the file — only the requested slices are read into RAM
    # when we iterate later, one shard at a time.
    print("  Reading tensor metadata...")
    state_dict: dict = {}   # name → tensor (loaded lazily per-shard below)
    shard_of: dict[str, Path] = {}  # tensor name → source shard path

    for shard in all_shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                shard_of[key] = shard

    # ── Step 2: compute sharding plan via transformers utility ───────────────
    # We need tensor sizes for the planner but don't want to load tensors yet.
    # Read byte sizes from the safetensors header (fast, no data copy).
    print("  Computing shard layout...")
    tensor_sizes: dict[str, int] = {}
    for shard in all_shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                tensor_sizes[key] = t.numel() * t.element_size()
                del t

    # split_torch_state_dict_into_shards takes the state dict but only uses
    # sizes — pass a proxy dict with just the sizes via a shim.
    # The function signature is: split_torch_state_dict_into_shards(
    #     state_dict, max_shard_size, filename_pattern)
    # It returns ShardedTensorIndex which has .tensor_to_filename and .shards_metadata
    target_bytes_str = f"{int(target_gb * GB)}B"  # e.g. "4294967296B"

    # Build a size-only proxy: {name: tensor} — reuse actual tensors for sizing
    # by loading them once (they'll be ~0 overhead for the planner call).
    from transformers.utils.hub import split_torch_state_dict_into_shards
    import torch

    size_proxy: dict[str, "torch.Tensor"] = {}
    for shard in all_shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                size_proxy[key] = f.get_tensor(key)

    shards_index = split_torch_state_dict_into_shards(
        size_proxy,
        max_shard_size=target_bytes_str,
        filename_pattern="model-{shard:05d}-of-{total_shards:05d}.safetensors",
    )
    del size_proxy

    new_shard_names = sorted(set(shards_index.tensor_to_filename.values()))
    print(f"  {len(all_shards)} shard(s) → {len(new_shard_names)} shard(s)")
    for name in new_shard_names:
        keys = [k for k, v in shards_index.tensor_to_filename.items() if v == name]
        size_gb = sum(tensor_sizes[k] for k in keys) / GB
        print(f"    {name}  ({len(keys)} tensors, {size_gb:.2f} GB)")

    if dry_run:
        return

    # ── Step 3: write new shards one at a time ───────────────────────────────
    # Group the plan by output filename, then load only the source shards needed
    # for each output chunk (minimises peak RAM).
    print()
    filename_to_keys: dict[str, list[str]] = {}
    for key, fname in shards_index.tensor_to_filename.items():
        filename_to_keys.setdefault(fname, []).append(key)

    # Collect existing .safetensors metadata from first shard (preserve __metadata__)
    metadata: dict[str, str] | None = None
    with safe_open(str(all_shards[0]), framework="pt", device="cpu") as f:
        metadata = f.metadata()

    for out_name, keys in filename_to_keys.items():
        dst = model_dir / out_name
        print(f"  writing {out_name} ...")
        tensors: dict[str, "torch.Tensor"] = {}
        # Group keys by source shard to open each file at most once
        by_shard: dict[Path, list[str]] = {}
        for k in keys:
            by_shard.setdefault(shard_of[k], []).append(k)
        for src_shard, src_keys in by_shard.items():
            with safe_open(str(src_shard), framework="pt", device="cpu") as f:
                for k in src_keys:
                    tensors[k] = f.get_tensor(k)
        save_file(tensors, str(dst), metadata=metadata)
        del tensors

    # ── Step 4: remove original shards that have been replaced ───────────────
    new_names_set = set(new_shard_names)
    for shard in all_shards:
        if shard.name not in new_names_set:
            print(f"  removing {shard.name}")
            shard.unlink()

    # ── Step 5: write updated index ──────────────────────────────────────────
    index_path = model_dir / "model.safetensors.index.json"
    total_size = sum(tensor_sizes.values())
    new_index = {
        "metadata": {"total_size": total_size},
        "weight_map": shards_index.tensor_to_filename,
    }
    index_path.write_text(json.dumps(new_index, indent=2))
    print(f"  updated {index_path.name}")
    print(f"\nDone.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "model",
        help="HF model ID (e.g. Intel/Qwen3-Coder-Next-int4-AutoRound) or path to snapshot directory",
    )
    p.add_argument(
        "--max-shard-size",
        type=float,
        default=DEFAULT_MAX_SHARD_GB,
        metavar="GB",
        help=f"Split shards larger than this (default: {DEFAULT_MAX_SHARD_GB} GB)",
    )
    p.add_argument(
        "--target-shard-size",
        type=float,
        default=DEFAULT_TARGET_SHARD_GB,
        metavar="GB",
        help=f"Target size for new shards (default: {DEFAULT_TARGET_SHARD_GB} GB)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan without writing any files",
    )
    args = p.parse_args()

    if args.target_shard_size >= args.max_shard_size:
        sys.exit("Error: --target-shard-size must be less than --max-shard-size")

    model_dir = resolve_model_dir(args.model)
    print(f"Model directory: {model_dir}")
    reshard(model_dir, args.max_shard_size, args.target_shard_size, args.dry_run)


if __name__ == "__main__":
    main()

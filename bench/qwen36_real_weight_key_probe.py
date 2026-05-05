"""Probe real Qwen3.6 HF config and layer-local safetensor key names."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fastkernels.models import Qwen36A3BSpec


def _require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for the Qwen3.6 key probe") from exc
    return hf_hub_download


def _download_json(repo_id: str, revision: str, cache_dir: str | None, filename: str) -> dict:
    hf_hub_download = _require_huggingface_hub()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=(0, 3))
    parser.add_argument("--show-moe", action="store_true")
    parser.add_argument("--load-shapes", action="store_true")
    args = parser.parse_args()

    config = _download_json(args.repo_id, args.revision, args.cache_dir, "config.json")
    index = _download_json(args.repo_id, args.revision, args.cache_dir, "model.safetensors.index.json")
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json does not contain a valid weight_map")

    spec = Qwen36A3BSpec.from_hf_config(config, name="Qwen3.6-35B-A3B")
    layer_kinds = spec.layer_kinds()
    print(f"model: {spec.name}")
    print(f"repo_id: {args.repo_id}")
    print(f"revision: {args.revision}")
    print(f"hidden_size: {spec.hidden_size}")
    print(f"layer_counts: {spec.layer_counts()}")
    print(f"hf_total_size_bytes: {index.get('metadata', {}).get('total_size', 'unknown')}")

    prefixes = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "transformer.layers",
        "layers",
    )
    for layer in args.layers:
        print("---")
        print(f"layer: {layer}")
        if 0 <= layer < len(layer_kinds):
            print(f"layer_kind: {layer_kinds[layer]}")
        layer_prefixes = tuple(f"{prefix}.{layer}." for prefix in prefixes)
        keys = sorted(key for key in weight_map if key.startswith(layer_prefixes))
        if not args.show_moe:
            keys = [
                key
                for key in keys
                if ".mlp.experts." not in key
                and ".mlp.shared_expert" not in key
                and ".mlp.gate." not in key
            ]
        print(f"layer_key_count: {len(keys)}")
        tensors = {}
        if args.load_shapes and keys:
            from safetensors import safe_open

            hf_hub_download = _require_huggingface_hub()
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            shard_paths = {}
            for shard_name in sorted({weight_map[key] for key in keys}):
                shard_paths[shard_name] = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=shard_name,
                    revision=args.revision,
                    cache_dir=args.cache_dir,
                    token=token,
                )
            for shard_name, shard_path in shard_paths.items():
                with safe_open(shard_path, framework="pt", device="cpu") as handle:
                    for key in keys:
                        if weight_map[key] == shard_name:
                            tensors[key] = tuple(handle.get_tensor(key).shape)
        for key in keys:
            shape = f" shape={tensors[key]}" if key in tensors else ""
            print(f"{key} -> {weight_map[key]}{shape}")


if __name__ == "__main__":
    main()

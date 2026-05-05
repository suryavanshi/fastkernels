"""Resolve the complete real Qwen3.6 40-layer safetensors weight plan."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from fastkernels.models import Qwen36A3BSpec, resolve_qwen36_full_weight_plan


def _require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for the Qwen3.6 full weight plan") from exc
    return hf_hub_download


def _download_json(repo_id: str, revision: str, cache_dir: str | None, filename: str) -> tuple[Path, dict]:
    hf_hub_download = _require_huggingface_hub()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
    )
    with path.open("r", encoding="utf-8") as handle:
        return path, json.load(handle)


def _download_shards(repo_id: str, revision: str, cache_dir: str | None, shards: tuple[str, ...]) -> float:
    hf_hub_download = _require_huggingface_hub()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    start = time.perf_counter()
    for shard in shards:
        hf_hub_download(
            repo_id=repo_id,
            filename=shard,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--show-layers", action="store_true")
    parser.add_argument("--show-shards", action="store_true")
    parser.add_argument("--download-shards", action="store_true")
    args = parser.parse_args()

    _, config = _download_json(args.repo_id, args.revision, args.cache_dir, "config.json")
    index_path, index = _download_json(args.repo_id, args.revision, args.cache_dir, "model.safetensors.index.json")
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json does not contain a valid weight_map")

    spec = Qwen36A3BSpec.from_hf_config(config, name="Qwen3.6-35B-A3B")
    key_index = {key: None for key in weight_map}
    plan = resolve_qwen36_full_weight_plan(key_index, spec=spec)
    required_keys = plan.keys()
    missing = sorted(key for key in required_keys if key not in weight_map)
    if missing:
        raise KeyError(f"resolved keys missing from weight_map: {missing[:8]}")
    shards = plan.required_shards(weight_map)
    total_size = index.get("metadata", {}).get("total_size", "unknown")

    print(f"model: {spec.name}")
    print(f"repo_id: {args.repo_id}")
    print(f"revision: {args.revision}")
    print(f"index_path: {index_path}")
    print(f"hidden_size: {spec.hidden_size}")
    print(f"layers: {spec.num_layers}")
    print(f"layer_counts: {plan.layer_counts()}")
    print(f"required_tensor_count: {len(required_keys)}")
    print(f"required_shard_count: {len(shards)}")
    print(f"hf_total_size_bytes: {total_size}")
    print(f"root_embedding_key: {plan.roots.embedding_weight}")
    print(f"root_output_norm_key: {plan.roots.output_norm_weight}")
    print(f"root_lm_head_key: {plan.roots.lm_head_weight}")
    print("fastkernels_full_reference_serving_available: True")
    print("fastkernels_full_triton_serving_ready: False")
    print("fastkernels_full_triton_serving_blocker: streaming reference path has not been replaced by fused kernels")
    print("vllm_full_serving_probe_available: True")
    print("vllm_full_serving_ready: unknown_until_benchmark")
    print("recommended_vllm_tensor_parallel_size: 8")

    if args.show_layers:
        for layer in plan.layers:
            print("---")
            print(f"layer: {layer.layer_idx}")
            print(f"layer_kind: {layer.layer_kind}")
            print(f"layer_tensor_count: {len(layer.keys())}")
            print(f"layer_shards: {','.join(sorted({weight_map[key] for key in layer.keys()}))}")
    if args.show_shards:
        print("required_shards: " + ",".join(shards))
    if args.download_shards:
        elapsed = _download_shards(args.repo_id, args.revision, args.cache_dir, shards)
        print(f"downloaded_shards: {len(shards)}")
        print(f"download_seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()

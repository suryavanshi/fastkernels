"""Smoke-test the Qwen3.6 MoE wrapper with real HF safetensor shards."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from fastkernels.kernels.triton import triton_qwen36_batched_moe_decode, triton_qwen36_batched_moe_layer_decode
from fastkernels.models import (
    load_qwen36_moe_weights_from_safetensors,
    qwen36_35b_a3b_spec,
    resolve_qwen36_moe_weight_keys,
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen3.6 real-weight smoke test") from exc
    return torch


def _require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for the Qwen3.6 real-weight smoke test") from exc
    return hf_hub_download


def _flatten_keys(keys) -> tuple[str, ...]:
    names = [keys.norm_weight, keys.router_weight, keys.shared_down_weight]
    if isinstance(keys.expert_gate_up_weight, str):
        names.append(keys.expert_gate_up_weight)
    else:
        for item in keys.expert_gate_up_weight:
            names.extend((item,) if isinstance(item, str) else item)
    if isinstance(keys.expert_down_weight, str):
        names.append(keys.expert_down_weight)
    else:
        names.extend(keys.expert_down_weight)
    if isinstance(keys.shared_gate_up_weight, str):
        names.append(keys.shared_gate_up_weight)
    else:
        names.extend(keys.shared_gate_up_weight)
    if keys.shared_expert_gate_weight is not None:
        names.append(keys.shared_expert_gate_weight)
    return tuple(names)


def _reference_expert(torch, hidden, gate_up_weight, down_weight):
    gate_up = torch.matmul(gate_up_weight.float(), hidden.float())
    gate, up = torch.chunk(gate_up, 2, dim=0)
    activation = torch.nn.functional.silu(gate) * up
    return torch.matmul(down_weight.float(), activation)


def _download_selected_shards(repo_id: str, revision: str, layer: int, cache_dir: str | None):
    hf_hub_download = _require_huggingface_hub()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    index_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors.index.json",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    with Path(index_path).open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json does not contain a valid weight_map")

    spec = qwen36_35b_a3b_spec()
    keys = resolve_qwen36_moe_weight_keys({key: None for key in weight_map}, layer, spec=spec)
    needed_keys = _flatten_keys(keys)
    shard_names = sorted({weight_map[key] for key in needed_keys})
    for shard_name in shard_names:
        hf_hub_download(
            repo_id=repo_id,
            filename=shard_name,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
    return Path(index_path).parent, index, keys, needed_keys, shard_names


def _run_layer_smoke(torch, args, spec, dtype, layer: int) -> None:
    weights_dir, index, resolved_keys, needed_keys, shard_names = _download_selected_shards(
        args.repo_id,
        args.revision,
        layer,
        args.cache_dir,
    )
    weights = load_qwen36_moe_weights_from_safetensors(weights_dir, layer, spec=spec, device=args.device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + layer)
    hidden = (
        args.hidden_scale
        * torch.randn(args.tokens, spec.hidden_size, generator=generator, dtype=dtype)
    ).to(args.device)

    hidden_f = hidden.float()
    normed = (
        hidden_f
        * torch.rsqrt(torch.mean(hidden_f * hidden_f, dim=1, keepdim=True) + 1e-6)
        * (1.0 + weights.norm_weight.float())
    )
    reference_logits = torch.matmul(normed, weights.router_weight.float().t())
    reference_probs = torch.softmax(reference_logits, dim=-1)
    reference_values, reference_ids = torch.topk(reference_probs, k=spec.num_routed_experts, dim=1)
    reference_weights = reference_values / torch.sum(reference_values, dim=1, keepdim=True)
    routed_reference = torch.zeros(args.tokens, spec.hidden_size, device=args.device, dtype=torch.float32)
    shared_reference = torch.zeros_like(routed_reference)
    for token in range(args.tokens):
        for idx in range(spec.num_routed_experts):
            expert_id = int(reference_ids[token, idx].item())
            routed_reference[token] = routed_reference[token] + reference_weights[token, idx] * _reference_expert(
                torch,
                hidden[token],
                weights.expert_gate_up_weight[expert_id],
                weights.expert_down_weight[expert_id],
            )
        shared_reference[token] = _reference_expert(
            torch,
            hidden[token],
            weights.shared_gate_up_weight,
            weights.shared_down_weight,
        )
    reference = routed_reference + shared_reference
    shared_gate_ungated_gap = None
    if weights.shared_expert_gate_weight is not None:
        gate_weight = weights.shared_expert_gate_weight.float().reshape(-1, spec.hidden_size)
        shared_gate = torch.sigmoid(torch.matmul(hidden_f, gate_weight.t()))
        ungated_reference = reference
        reference = routed_reference + shared_gate[:, :1] * shared_reference
        shared_gate_ungated_gap = torch.max(torch.abs(ungated_reference.float() - reference.float())).item()

    for _ in range(args.warmup):
        triton_qwen36_batched_moe_decode(
            hidden,
            weights.norm_weight,
            weights.router_weight,
            weights.expert_gate_up_weight,
            weights.expert_down_weight,
            weights.shared_gate_up_weight,
            weights.shared_down_weight,
            weights.shared_expert_gate_weight,
            top_k=spec.num_routed_experts,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    candidate = logits = topk_ids = topk_weights = None
    for _ in range(args.iters):
        candidate, logits, topk_ids, topk_weights = triton_qwen36_batched_moe_decode(
            hidden,
            weights.norm_weight,
            weights.router_weight,
            weights.expert_gate_up_weight,
            weights.expert_down_weight,
            weights.shared_gate_up_weight,
            weights.shared_down_weight,
            weights.shared_expert_gate_weight,
            top_k=spec.num_routed_experts,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    batch_ms = elapsed * 1000 / args.iters
    output_diff = torch.max(torch.abs(candidate.float() - reference.float())).item()
    logits_diff = torch.max(torch.abs(logits.float() - reference_logits.float())).item()
    ids_match = bool(torch.equal(topk_ids.cpu(), reference_ids.cpu()))
    weights_diff = torch.max(torch.abs(topk_weights.float() - reference_weights.float())).item()
    total_size = index.get("metadata", {}).get("total_size", "unknown")

    print(f"model: {spec.name}")
    print(f"repo_id: {args.repo_id}")
    print(f"revision: {args.revision}")
    print(f"device: {args.device}")
    print(f"dtype: {dtype}")
    print(f"tokens: {args.tokens}")
    print(f"layer: {layer}")
    print(f"hidden_size: {spec.hidden_size}")
    print(f"experts: {spec.num_experts}")
    print(f"top_k: {spec.num_routed_experts}")
    print(f"intermediate: {spec.expert_intermediate_size}")
    print(f"hf_total_size_bytes: {total_size}")
    print(f"downloaded_shards: {','.join(shard_names)}")
    print(f"downloaded_shard_count: {len(shard_names)}")
    print(f"needed_tensor_count: {len(needed_keys)}")
    print(f"resolved_norm_key: {weights.keys.norm_weight}")
    print(f"resolved_router_key: {weights.keys.router_weight}")
    print(f"resolved_expert_gate_up_key: {resolved_keys.expert_gate_up_weight}")
    print(f"resolved_expert_down_key: {resolved_keys.expert_down_weight}")
    print(f"shared_expert_gate_present: {weights.shared_expert_gate_weight is not None}")
    print(f"shared_expert_gate_applied: {weights.shared_expert_gate_weight is not None}")
    if shared_gate_ungated_gap is not None:
        print(f"shared_expert_gate_ungated_gap_max_abs: {shared_gate_ungated_gap:.6f}")
    print(f"iters: {args.iters}")
    print(f"output_max_abs_diff: {output_diff:.6f}")
    print(f"logits_max_abs_diff: {logits_diff:.6f}")
    print(f"topk_ids_match: {ids_match}")
    print(f"topk_weights_max_abs_diff: {weights_diff:.6f}")
    print(f"real_weight_batched_moe_ms: {batch_ms:.4f}")
    print(f"real_weight_batched_moe_ms_per_token: {batch_ms / args.tokens:.4f}")

    if args.run_layer_harness:
        reference_layer = hidden_f + reference.float()
        for _ in range(args.warmup):
            triton_qwen36_batched_moe_layer_decode(
                hidden,
                weights.norm_weight,
                weights.router_weight,
                weights.expert_gate_up_weight,
                weights.expert_down_weight,
                weights.shared_gate_up_weight,
                weights.shared_down_weight,
                weights.shared_expert_gate_weight,
                top_k=spec.num_routed_experts,
            )
        if args.device == "cuda":
            torch.cuda.synchronize()

        layer_start = time.perf_counter()
        layer_hidden = layer_update = layer_logits = layer_topk_ids = layer_topk_weights = None
        for _ in range(args.iters):
            layer_hidden, layer_update, layer_logits, layer_topk_ids, layer_topk_weights = (
                triton_qwen36_batched_moe_layer_decode(
                    hidden,
                    weights.norm_weight,
                    weights.router_weight,
                    weights.expert_gate_up_weight,
                    weights.expert_down_weight,
                    weights.shared_gate_up_weight,
                    weights.shared_down_weight,
                    weights.shared_expert_gate_weight,
                    top_k=spec.num_routed_experts,
                )
            )
        if args.device == "cuda":
            torch.cuda.synchronize()
        layer_elapsed = time.perf_counter() - layer_start
        layer_batch_ms = layer_elapsed * 1000 / args.iters

        layer_output_diff = torch.max(torch.abs(layer_hidden.float() - reference_layer.float())).item()
        layer_update_diff = torch.max(torch.abs(layer_update.float() - reference.float())).item()
        layer_logits_diff = torch.max(torch.abs(layer_logits.float() - reference_logits.float())).item()
        layer_ids_match = bool(torch.equal(layer_topk_ids.cpu(), reference_ids.cpu()))
        layer_weights_diff = torch.max(torch.abs(layer_topk_weights.float() - reference_weights.float())).item()

        print(f"real_weight_layer_harness: True")
        print(f"layer_output_max_abs_diff: {layer_output_diff:.6f}")
        print(f"layer_update_max_abs_diff: {layer_update_diff:.6f}")
        print(f"layer_logits_max_abs_diff: {layer_logits_diff:.6f}")
        print(f"layer_topk_ids_match: {layer_ids_match}")
        print(f"layer_topk_weights_max_abs_diff: {layer_weights_diff:.6f}")
        print(f"real_weight_batched_moe_layer_ms: {layer_batch_ms:.4f}")
        print(f"real_weight_batched_moe_layer_ms_per_token: {layer_batch_ms / args.tokens:.4f}")

    del weights, hidden, candidate, logits, topk_ids, topk_weights, reference
    if args.device == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--run-layer-harness", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-scale", type=float, default=0.02)
    args = parser.parse_args()

    torch = _require_torch()
    dtype = getattr(torch, args.dtype)
    spec = qwen36_35b_a3b_spec()
    layers = tuple(args.layers) if args.layers is not None else (args.layer,)
    for idx, layer in enumerate(layers):
        if idx:
            print("---")
        _run_layer_smoke(torch, args, spec, dtype, layer)


if __name__ == "__main__":
    main()

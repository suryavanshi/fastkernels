"""Compare real-shape Qwen3.6 routed MoE against vLLM fused_moe when present."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastkernels.kernels.triton import triton_qwen36_batched_routed_experts_decode
from fastkernels.models import (
    load_qwen36_moe_weights_from_safetensors,
    qwen36_35b_a3b_spec,
    resolve_qwen36_moe_weight_keys,
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen3.6 vLLM comparison") from exc
    return torch


def _load_vllm_fused_moe(require_vllm: bool) -> tuple[Callable[..., Any] | None, str | None]:
    try:
        import vllm
        from vllm.model_executor.layers.fused_moe import fused_moe
    except Exception as exc:  # pragma: no cover - depends on optional package
        if require_vllm:
            raise RuntimeError("vLLM is required for this comparison but could not be imported") from exc
        return None, None
    return fused_moe, getattr(vllm, "__version__", "unknown")


def _require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for the real-weight vLLM comparison") from exc
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


def _reference_routed(torch, hidden, topk_ids, topk_weights, expert_gate_up_weight, expert_down_weight):
    hidden_f = hidden.float()
    if hidden_f.ndim == 1:
        hidden_f = hidden_f.unsqueeze(0)
        topk_ids = topk_ids.unsqueeze(0)
        topk_weights = topk_weights.unsqueeze(0)
    output = torch.zeros_like(hidden_f)
    for token in range(hidden_f.shape[0]):
        for idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token, idx].item())
            gate_up = torch.matmul(expert_gate_up_weight[expert_id].float(), hidden_f[token])
            gate, up = torch.chunk(gate_up, 2, dim=0)
            activation = torch.nn.functional.silu(gate) * up
            output[token] = output[token] + topk_weights[token, idx] * torch.matmul(
                expert_down_weight[expert_id].float(),
                activation,
            )
    return output


def _topk_from_logits(torch, router_logits, top_k: int):
    probs = torch.softmax(router_logits.float(), dim=-1)
    topk_values, topk_ids = torch.topk(probs, k=top_k, dim=-1)
    topk_weights = topk_values / torch.sum(topk_values, dim=-1, keepdim=True)
    return topk_ids.to(torch.int64), topk_weights


def _call_vllm_fused_moe(
    fused_moe: Callable[..., Any],
    hidden,
    expert_gate_up_weight,
    expert_down_weight,
    router_logits,
    top_k: int,
):
    squeeze = hidden.ndim == 1
    hidden_batch = hidden.unsqueeze(0).contiguous() if squeeze else hidden.contiguous()
    gating_output = router_logits.unsqueeze(0).contiguous() if router_logits.ndim == 1 else router_logits.contiguous()
    w1 = expert_gate_up_weight.contiguous()
    w2 = expert_down_weight.contiguous()

    try:
        parameters = inspect.signature(fused_moe).parameters
    except (TypeError, ValueError):
        parameters = {}

    kwargs: dict[str, Any] = {
        "hidden_states": hidden_batch,
        "w1": w1,
        "w2": w2,
        "gating_output": gating_output,
        "topk": top_k,
    }
    optional_defaults = {
        "renormalize": True,
        "inplace": False,
        "activation": "silu",
        "global_num_experts": -1,
        "expert_map": None,
    }
    if parameters:
        for key, value in optional_defaults.items():
            if key in parameters:
                kwargs[key] = value
        try:
            output = fused_moe(**kwargs)
        except TypeError:
            output = fused_moe(hidden_batch, w1, w2, gating_output, top_k, True)
    else:
        try:
            output = fused_moe(**{**kwargs, "renormalize": True, "inplace": False})
        except TypeError:
            output = fused_moe(hidden_batch, w1, w2, gating_output, top_k, True)

    if isinstance(output, tuple):
        output = output[0]
    return output.squeeze(0) if squeeze else output


def _make_random_case(torch, args, spec, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    scale = args.hidden_scale
    hidden = (scale * torch.randn(args.tokens, spec.hidden_size, generator=generator, dtype=dtype)).to(args.device)
    router_logits = torch.randn(args.tokens, spec.num_experts, generator=generator, dtype=torch.float32).to(args.device)
    expert_gate_up_weight = (
        scale
        * torch.randn(
            spec.num_experts,
            2 * spec.expert_intermediate_size,
            spec.hidden_size,
            generator=generator,
            dtype=dtype,
        )
    ).to(args.device)
    expert_down_weight = (
        scale
        * torch.randn(
            spec.num_experts,
            spec.hidden_size,
            spec.expert_intermediate_size,
            generator=generator,
            dtype=dtype,
        )
    ).to(args.device)
    return hidden, router_logits, expert_gate_up_weight, expert_down_weight, None


def _make_real_weight_case(torch, args, spec, dtype):
    weights_dir, index, resolved_keys, needed_keys, shard_names = _download_selected_shards(
        args.repo_id,
        args.revision,
        args.layer,
        args.cache_dir,
    )
    weights = load_qwen36_moe_weights_from_safetensors(weights_dir, args.layer, spec=spec, device=args.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + args.layer)
    hidden = (
        args.hidden_scale
        * torch.randn(args.tokens, spec.hidden_size, generator=generator, dtype=dtype)
    ).to(args.device)

    hidden_f = hidden.float()
    normed = (
        hidden_f
        * torch.rsqrt(torch.mean(hidden_f * hidden_f, dim=1, keepdim=True) + 1e-6)
        * weights.norm_weight.float()
    )
    router_logits = torch.matmul(normed, weights.router_weight.float().t())
    metadata = {
        "hf_total_size_bytes": index.get("metadata", {}).get("total_size", "unknown"),
        "downloaded_shards": ",".join(shard_names),
        "downloaded_shard_count": len(shard_names),
        "needed_tensor_count": len(needed_keys),
        "resolved_norm_key": weights.keys.norm_weight,
        "resolved_router_key": weights.keys.router_weight,
        "resolved_expert_gate_up_key": resolved_keys.expert_gate_up_weight,
        "resolved_expert_down_key": resolved_keys.expert_down_weight,
    }
    return hidden, router_logits, weights.expert_gate_up_weight, weights.expert_down_weight, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-scale", type=float, default=0.02)
    parser.add_argument("--require-vllm", action="store_true")
    parser.add_argument("--real-weights", action="store_true")
    parser.add_argument("--repo-id", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    torch = _require_torch()
    dtype = getattr(torch, args.dtype)
    fused_moe, vllm_version = _load_vllm_fused_moe(args.require_vllm)
    spec = qwen36_35b_a3b_spec()
    if args.tokens < 1:
        raise ValueError("--tokens must be >= 1")

    if args.real_weights:
        hidden, router_logits, expert_gate_up_weight, expert_down_weight, metadata = _make_real_weight_case(
            torch,
            args,
            spec,
            dtype,
        )
    else:
        hidden, router_logits, expert_gate_up_weight, expert_down_weight, metadata = _make_random_case(
            torch,
            args,
            spec,
            dtype,
        )
    topk_ids, topk_weights = _topk_from_logits(torch, router_logits, spec.num_routed_experts)
    reference = _reference_routed(
        torch,
        hidden,
        topk_ids,
        topk_weights,
        expert_gate_up_weight,
        expert_down_weight,
    )

    for _ in range(args.warmup):
        triton_qwen36_batched_routed_experts_decode(
            hidden,
            topk_ids,
            topk_weights,
            expert_gate_up_weight,
            expert_down_weight,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    fastkernels_output = None
    for _ in range(args.iters):
        fastkernels_output = triton_qwen36_batched_routed_experts_decode(
            hidden,
            topk_ids,
            topk_weights,
            expert_gate_up_weight,
            expert_down_weight,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    fastkernels_elapsed = time.perf_counter() - start

    fastkernels_diff = torch.max(torch.abs(fastkernels_output.float() - reference.float())).item()
    vllm_ms = None
    vllm_diff = None

    if fused_moe is not None:
        for _ in range(args.warmup):
            _call_vllm_fused_moe(
                fused_moe,
                hidden,
                expert_gate_up_weight,
                expert_down_weight,
                router_logits,
                spec.num_routed_experts,
            )
        if args.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        vllm_output = None
        for _ in range(args.iters):
            vllm_output = _call_vllm_fused_moe(
                fused_moe,
                hidden,
                expert_gate_up_weight,
                expert_down_weight,
                router_logits,
                spec.num_routed_experts,
            )
        if args.device == "cuda":
            torch.cuda.synchronize()
        vllm_elapsed = time.perf_counter() - start
        vllm_ms = vllm_elapsed * 1000 / args.iters
        vllm_diff = torch.max(torch.abs(vllm_output.float() - reference.float())).item()

    fastkernels_ms = fastkernels_elapsed * 1000 / args.iters
    print(f"model: {spec.name}")
    print(f"weights: {'real' if args.real_weights else 'random'}")
    if args.real_weights:
        print(f"repo_id: {args.repo_id}")
        print(f"revision: {args.revision}")
        print(f"layer: {args.layer}")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    print(f"device: {args.device}")
    print(f"dtype: {dtype}")
    print(f"tokens: {args.tokens}")
    print(f"hidden_size: {spec.hidden_size}")
    print(f"experts: {spec.num_experts}")
    print(f"top_k: {spec.num_routed_experts}")
    print(f"intermediate: {spec.expert_intermediate_size}")
    print("comparison_scope: routed_experts_only")
    print(f"iters: {args.iters}")
    print(f"fastkernels_routed_max_abs_diff: {fastkernels_diff:.6f}")
    print(f"fastkernels_routed_moe_ms: {fastkernels_ms:.4f}")
    print(f"fastkernels_routed_moe_ms_per_token: {fastkernels_ms / args.tokens:.4f}")
    print(f"fastkernels_routed_moe_tokens_per_second: {args.tokens * 1000 / fastkernels_ms:.2f}")
    if fused_moe is None:
        print("vllm_available: False")
    else:
        ratio = fastkernels_ms / vllm_ms if vllm_ms else float("nan")
        print("vllm_available: True")
        print(f"vllm_version: {vllm_version}")
        print(f"vllm_routed_max_abs_diff: {vllm_diff:.6f}")
        print(f"vllm_routed_moe_ms: {vllm_ms:.4f}")
        print(f"vllm_routed_moe_ms_per_token: {vllm_ms / args.tokens:.4f}")
        print(f"vllm_routed_moe_tokens_per_second: {args.tokens * 1000 / vllm_ms:.2f}")
        print(f"fastkernels_to_vllm_ms_ratio: {ratio:.4f}")


if __name__ == "__main__":
    main()

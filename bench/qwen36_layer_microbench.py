"""Synthetic Qwen3.6 layer-boundary benchmark."""

from __future__ import annotations

import argparse
import time

from fastkernels.models import synthetic_qwen36_spec
from fastkernels.reference import (
    make_synthetic_qwen36_decode_weights,
    reference_qwen36_attention_decode,
    reference_qwen36_deltanet_decode,
    reference_qwen36_moe_decode,
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen3.6 layer microbench") from exc
    return torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-kind", choices=["deltanet_moe", "attention_moe"], default="deltanet_moe")
    parser.add_argument("--impl", choices=["reference", "triton"], default="triton")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch = _require_torch()
    dtype = getattr(torch, args.dtype)
    spec = synthetic_qwen36_spec()
    weights = make_synthetic_qwen36_decode_weights(spec, device=args.device, dtype=dtype, seed=args.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1)

    hidden = torch.randn(spec.hidden_size, generator=generator, dtype=dtype).to(args.device)
    deltanet_state = torch.randn(spec.deltanet_state_shape(), generator=generator, dtype=dtype).to(args.device)
    key_cache = torch.randn(spec.attention_cache_shape(max_positions=8), generator=generator, dtype=dtype).to(args.device)
    value_cache = torch.randn_like(key_cache)
    position = 2

    if args.layer_kind == "deltanet_moe":
        layer = weights["layers"][0]

        def reference_run():
            layer_hidden, next_state = reference_qwen36_deltanet_decode(hidden, deltanet_state, layer, spec)
            return reference_qwen36_moe_decode(layer_hidden, layer, spec), next_state

        if args.impl == "triton":
            from fastkernels.kernels.triton import triton_synthetic_qwen36_deltanet_moe_decode

            def candidate_run():
                return triton_synthetic_qwen36_deltanet_moe_decode(
                    hidden,
                    deltanet_state,
                    layer,
                    qk_heads=spec.deltanet_qk_heads,
                    head_dim=spec.deltanet_head_dim,
                    value_dim_per_head=spec.deltanet_value_dim_per_qk_head,
                    top_k=spec.num_routed_experts,
                )
        else:
            candidate_run = reference_run
    else:
        layer = weights["layers"][3]

        def reference_run():
            layer_hidden, next_key_cache, next_value_cache = reference_qwen36_attention_decode(
                hidden,
                key_cache,
                value_cache,
                layer,
                spec,
                position,
            )
            return reference_qwen36_moe_decode(layer_hidden, layer, spec), next_key_cache, next_value_cache

        if args.impl == "triton":
            from fastkernels.kernels.triton import triton_synthetic_qwen36_attention_moe_decode

            def candidate_run():
                return triton_synthetic_qwen36_attention_moe_decode(
                    hidden,
                    key_cache,
                    value_cache,
                    layer,
                    position=position,
                    attention_heads=spec.attention_heads,
                    kv_heads=spec.attention_kv_heads,
                    head_dim=spec.attention_head_dim,
                    rope_dim=spec.rope_dim,
                    top_k=spec.num_routed_experts,
                    copy_cache=False,
                )
        else:
            candidate_run = reference_run

    reference_outputs = reference_run()
    for _ in range(args.warmup):
        candidate_run()
    if args.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    outputs = None
    for _ in range(args.iters):
        outputs = candidate_run()
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    hidden_diff = torch.max(torch.abs(outputs[0].float() - reference_outputs[0].float())).item()
    state_diff = torch.max(torch.abs(outputs[1].float() - reference_outputs[1].float())).item()
    print(f"model: {spec.name}")
    print(f"layer_kind: {args.layer_kind}")
    print(f"impl: {args.impl}")
    print(f"device: {args.device}")
    print(f"dtype: {dtype}")
    print(f"iters: {args.iters}")
    print(f"hidden_max_abs_diff: {hidden_diff:.6f}")
    print(f"state_max_abs_diff: {state_diff:.6f}")
    if args.layer_kind == "attention_moe":
        value_diff = torch.max(torch.abs(outputs[2].float() - reference_outputs[2].float())).item()
        print(f"value_cache_max_abs_diff: {value_diff:.6f}")
    print(f"layer_ms: {elapsed * 1000 / args.iters:.4f}")


if __name__ == "__main__":
    main()

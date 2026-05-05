"""Synthetic Qwen3.6 MoE top-k/expert-count benchmark."""

from __future__ import annotations

import argparse
import time

from fastkernels.kernels.triton import triton_synthetic_qwen36_moe_decode
from fastkernels.reference import reference_qwen36_moe_decode


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen3.6 MoE top-k microbench") from exc
    return torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--intermediate", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch = _require_torch()
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    def randn(*shape):
        return (0.02 * torch.randn(*shape, generator=generator, dtype=dtype)).to(args.device)

    hidden = randn(args.hidden)
    layer = {
        "norm_weight": torch.ones(args.hidden, device=args.device, dtype=dtype),
        "router_weight": randn(args.experts, args.hidden),
        "expert_gate_up_weight": randn(args.experts, 2 * args.intermediate, args.hidden),
        "expert_down_weight": randn(args.experts, args.hidden, args.intermediate),
        "shared_gate_up_weight": randn(2 * args.intermediate, args.hidden),
        "shared_down_weight": randn(args.hidden, args.intermediate),
    }
    spec = type(
        "SyntheticMoESpec",
        (),
        {
            "num_routed_experts": args.top_k,
        },
    )()

    reference = reference_qwen36_moe_decode(hidden, layer, spec)
    for _ in range(args.warmup):
        triton_synthetic_qwen36_moe_decode(
            hidden,
            layer["norm_weight"],
            layer["router_weight"],
            layer["expert_gate_up_weight"],
            layer["expert_down_weight"],
            layer["shared_gate_up_weight"],
            layer["shared_down_weight"],
            top_k=args.top_k,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    candidate = None
    for _ in range(args.iters):
        candidate = triton_synthetic_qwen36_moe_decode(
            hidden,
            layer["norm_weight"],
            layer["router_weight"],
            layer["expert_gate_up_weight"],
            layer["expert_down_weight"],
            layer["shared_gate_up_weight"],
            layer["shared_down_weight"],
            top_k=args.top_k,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    max_abs_diff = torch.max(torch.abs(candidate.float() - reference.float())).item()
    print(f"device: {args.device}")
    print(f"dtype: {dtype}")
    print(f"hidden: {args.hidden}")
    print(f"intermediate: {args.intermediate}")
    print(f"experts: {args.experts}")
    print(f"top_k: {args.top_k}")
    print(f"iters: {args.iters}")
    print(f"max_abs_diff: {max_abs_diff:.6f}")
    print(f"moe_ms: {elapsed * 1000 / args.iters:.4f}")


if __name__ == "__main__":
    main()

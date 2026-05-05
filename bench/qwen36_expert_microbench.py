"""Real-shape Qwen3.6 single-expert MLP microbenchmark."""

from __future__ import annotations

import argparse
import time

from fastkernels.kernels.triton import triton_qwen36_single_expert_mlp_decode
from fastkernels.models import qwen36_35b_a3b_spec


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen3.6 expert microbench") from exc
    return torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch = _require_torch()
    dtype = getattr(torch, args.dtype)
    spec = qwen36_35b_a3b_spec()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    scale = 0.02

    hidden = (scale * torch.randn(spec.hidden_size, generator=generator, dtype=dtype)).to(args.device)
    expert_gate_up_weight = (
        scale * torch.randn(2 * spec.expert_intermediate_size, spec.hidden_size, generator=generator, dtype=dtype)
    ).to(args.device)
    expert_down_weight = (
        scale * torch.randn(spec.hidden_size, spec.expert_intermediate_size, generator=generator, dtype=dtype)
    ).to(args.device)

    gate_up = torch.matmul(expert_gate_up_weight.float(), hidden.float())
    gate, up = torch.chunk(gate_up, 2, dim=0)
    activation = torch.nn.functional.silu(gate) * up
    reference = torch.matmul(expert_down_weight.float(), activation)

    for _ in range(args.warmup):
        triton_qwen36_single_expert_mlp_decode(
            hidden,
            expert_gate_up_weight,
            expert_down_weight,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    candidate = None
    for _ in range(args.iters):
        candidate = triton_qwen36_single_expert_mlp_decode(
            hidden,
            expert_gate_up_weight,
            expert_down_weight,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    max_abs_diff = torch.max(torch.abs(candidate.float() - reference.float())).item()
    print(f"model: {spec.name}")
    print(f"device: {args.device}")
    print(f"dtype: {dtype}")
    print(f"hidden_size: {spec.hidden_size}")
    print(f"intermediate: {spec.expert_intermediate_size}")
    print(f"iters: {args.iters}")
    print(f"max_abs_diff: {max_abs_diff:.6f}")
    print(f"expert_mlp_ms: {elapsed * 1000 / args.iters:.4f}")


if __name__ == "__main__":
    main()

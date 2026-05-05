"""Real-shape Qwen3.6 batched routed-plus-shared MoE microbenchmark."""

from __future__ import annotations

import argparse
import time

from fastkernels.kernels.triton import triton_qwen36_batched_routed_shared_experts_decode
from fastkernels.models import qwen36_35b_a3b_spec


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen3.6 batched routed MoE microbench") from exc
    return torch


def _reference_expert(torch, hidden, gate_up_weight, down_weight):
    gate_up = torch.matmul(gate_up_weight.float(), hidden.float())
    gate, up = torch.chunk(gate_up, 2, dim=0)
    activation = torch.nn.functional.silu(gate) * up
    return torch.matmul(down_weight.float(), activation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch = _require_torch()
    dtype = getattr(torch, args.dtype)
    spec = qwen36_35b_a3b_spec()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    scale = 0.02

    hidden = (scale * torch.randn(args.tokens, spec.hidden_size, generator=generator, dtype=dtype)).to(args.device)
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
    shared_gate_up_weight = (
        scale * torch.randn(2 * spec.expert_intermediate_size, spec.hidden_size, generator=generator, dtype=dtype)
    ).to(args.device)
    shared_down_weight = (
        scale * torch.randn(spec.hidden_size, spec.expert_intermediate_size, generator=generator, dtype=dtype)
    ).to(args.device)
    topk_ids = torch.empty((args.tokens, spec.num_routed_experts), device=args.device, dtype=torch.int64)
    base_ids = torch.tensor([3, 17, 42, 89, 127, 199, 231, 255], device=args.device, dtype=torch.int64)
    for token in range(args.tokens):
        topk_ids[token] = (base_ids + token) % spec.num_experts
    topk_weights = torch.softmax(
        torch.randn(args.tokens, spec.num_routed_experts, generator=generator),
        dim=1,
    ).to(args.device)

    reference = torch.zeros(args.tokens, spec.hidden_size, device=args.device, dtype=torch.float32)
    for token in range(args.tokens):
        for idx in range(spec.num_routed_experts):
            expert_id = int(topk_ids[token, idx].item())
            reference[token] = reference[token] + topk_weights[token, idx] * _reference_expert(
                torch,
                hidden[token],
                expert_gate_up_weight[expert_id],
                expert_down_weight[expert_id],
            )
        reference[token] = reference[token] + _reference_expert(
            torch,
            hidden[token],
            shared_gate_up_weight,
            shared_down_weight,
        )

    for _ in range(args.warmup):
        triton_qwen36_batched_routed_shared_experts_decode(
            hidden,
            topk_ids,
            topk_weights,
            expert_gate_up_weight,
            expert_down_weight,
            shared_gate_up_weight,
            shared_down_weight,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    candidate = None
    for _ in range(args.iters):
        candidate = triton_qwen36_batched_routed_shared_experts_decode(
            hidden,
            topk_ids,
            topk_weights,
            expert_gate_up_weight,
            expert_down_weight,
            shared_gate_up_weight,
            shared_down_weight,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    batch_ms = elapsed * 1000 / args.iters
    max_abs_diff = torch.max(torch.abs(candidate.float() - reference.float())).item()
    print(f"model: {spec.name}")
    print(f"device: {args.device}")
    print(f"dtype: {dtype}")
    print(f"tokens: {args.tokens}")
    print(f"hidden_size: {spec.hidden_size}")
    print(f"experts: {spec.num_experts}")
    print(f"top_k: {spec.num_routed_experts}")
    print(f"intermediate: {spec.expert_intermediate_size}")
    print(f"iters: {args.iters}")
    print(f"max_abs_diff: {max_abs_diff:.6f}")
    print(f"batched_routed_shared_moe_ms: {batch_ms:.4f}")
    print(f"batched_routed_shared_moe_ms_per_token: {batch_ms / args.tokens:.4f}")


if __name__ == "__main__":
    main()

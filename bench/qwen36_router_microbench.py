"""Real-shape Qwen3.6 MoE router microbenchmark."""

from __future__ import annotations

import argparse
import time

from fastkernels.kernels.triton import triton_qwen36_moe_router_decode
from fastkernels.models import qwen36_35b_a3b_spec


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen3.6 router microbench") from exc
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

    hidden = torch.randn(spec.hidden_size, generator=generator, dtype=dtype).to(args.device)
    norm_weight = torch.randn(spec.hidden_size, generator=generator, dtype=dtype).to(args.device)
    router_weight = torch.randn(spec.num_experts, spec.hidden_size, generator=generator, dtype=dtype).to(args.device)

    hidden_f = hidden.float()
    normed = hidden_f * torch.rsqrt(torch.mean(hidden_f * hidden_f) + 1e-6) * norm_weight.float()
    reference_logits = torch.matmul(router_weight.float(), normed)
    reference_probs = torch.softmax(reference_logits, dim=-1)
    reference_values, reference_ids = torch.topk(reference_probs, k=spec.num_routed_experts)
    reference_weights = reference_values / torch.sum(reference_values)

    for _ in range(args.warmup):
        triton_qwen36_moe_router_decode(
            hidden,
            norm_weight,
            router_weight,
            top_k=spec.num_routed_experts,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    logits = topk_ids = topk_weights = None
    for _ in range(args.iters):
        logits, topk_ids, topk_weights = triton_qwen36_moe_router_decode(
            hidden,
            norm_weight,
            router_weight,
            top_k=spec.num_routed_experts,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    logits_diff = torch.max(torch.abs(logits.float() - reference_logits.float())).item()
    ids_match = bool(torch.equal(topk_ids.cpu(), reference_ids.cpu()))
    weights_diff = torch.max(torch.abs(topk_weights.float() - reference_weights.float())).item()

    print(f"model: {spec.name}")
    print(f"device: {args.device}")
    print(f"dtype: {dtype}")
    print(f"hidden_size: {spec.hidden_size}")
    print(f"experts: {spec.num_experts}")
    print(f"top_k: {spec.num_routed_experts}")
    print(f"iters: {args.iters}")
    print(f"logits_max_abs_diff: {logits_diff:.6f}")
    print(f"topk_ids_match: {ids_match}")
    print(f"topk_weights_max_abs_diff: {weights_diff:.6f}")
    print(f"router_ms: {elapsed * 1000 / args.iters:.4f}")


if __name__ == "__main__":
    main()

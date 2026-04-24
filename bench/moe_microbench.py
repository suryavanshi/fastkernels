"""Microbenchmarks and correctness checks for MoE building blocks."""

from __future__ import annotations

import argparse
import time

from fastkernels.reference import reference_expert_histogram, reference_fused_swiglu, reference_routed_moe
from fastkernels.testing import default_tolerance


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the MoE microbench") from exc
    return torch


def _maybe_triton_fused_swiglu(gate_up):
    try:
        from fastkernels.kernels.triton import triton_fused_swiglu
    except Exception as exc:
        return None, exc
    try:
        return triton_fused_swiglu(gate_up), None
    except Exception as exc:
        return None, exc


def _maybe_triton_expert_histogram(topk_ids, num_experts):
    try:
        from fastkernels.kernels.triton import triton_expert_histogram
    except Exception as exc:
        return None, exc
    try:
        return triton_expert_histogram(topk_ids, num_experts), None
    except Exception as exc:
        return None, exc


def bench_swiglu(args: argparse.Namespace) -> None:
    torch = _require_torch()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    gate_up = torch.randn(args.rows, 2 * args.intermediate, device=device, dtype=dtype)

    ref = reference_fused_swiglu(gate_up)
    candidate, error = _maybe_triton_fused_swiglu(gate_up)
    if candidate is not None:
        atol, rtol = default_tolerance(dtype)
        torch.testing.assert_close(candidate, ref, atol=atol, rtol=rtol)
        print("triton_fused_swiglu: correctness ok")
    else:
        print(f"triton_fused_swiglu: skipped ({error})")

    for _ in range(args.warmup):
        reference_fused_swiglu(gate_up)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        reference_fused_swiglu(gate_up)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"torch_swiglu_ms: {elapsed * 1000 / args.iters:.4f}")

    if candidate is not None:
        for _ in range(args.warmup):
            _maybe_triton_fused_swiglu(gate_up)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            _maybe_triton_fused_swiglu(gate_up)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"triton_swiglu_ms: {elapsed * 1000 / args.iters:.4f}")


def check_expert_histogram(args: argparse.Namespace) -> None:
    torch = _require_torch()
    device = torch.device(args.device)
    topk_ids = torch.randint(0, args.experts, (args.tokens, args.top_k), device=device, dtype=torch.int32)
    ref = reference_expert_histogram(topk_ids, args.experts)
    candidate, error = _maybe_triton_expert_histogram(topk_ids, args.experts)
    if candidate is None:
        print(f"triton_expert_histogram: skipped ({error})")
        return

    torch.testing.assert_close(candidate.cpu(), ref.cpu(), atol=0, rtol=0)
    print("triton_expert_histogram: correctness ok")


def check_routed_moe(args: argparse.Namespace) -> None:
    torch = _require_torch()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    hidden = args.hidden
    intermediate = args.intermediate
    experts = args.experts
    tokens = args.tokens

    hidden_states = torch.randn(tokens, hidden, device=device, dtype=dtype)
    router_logits = torch.randn(tokens, experts, device=device, dtype=torch.float32)
    gate_up_weight = torch.randn(experts, 2 * intermediate, hidden, device=device, dtype=dtype)
    down_weight = torch.randn(experts, hidden, intermediate, device=device, dtype=dtype)

    out = reference_routed_moe(
        hidden_states,
        router_logits,
        gate_up_weight,
        down_weight,
        top_k=args.top_k,
    )
    print(f"reference_routed_moe: shape={tuple(out.shape)} dtype={out.dtype} device={out.device}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--skip-routed-moe", action="store_true")
    args = parser.parse_args()

    bench_swiglu(args)
    check_expert_histogram(args)
    if not args.skip_routed_moe:
        check_routed_moe(args)


if __name__ == "__main__":
    main()

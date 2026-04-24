"""Triton MoE kernel building blocks."""

from __future__ import annotations

from typing import Any


def _load_triton() -> tuple[Any, Any]:
    try:
        import torch
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - depends on optional packages
        raise RuntimeError("Triton and PyTorch are required for this kernel") from exc
    return torch, triton, tl


def _compile_fused_swiglu_kernel(triton: Any, tl: Any) -> Any:
    @triton.jit
    def _kernel(gate_up, out, total_elements: tl.constexpr, intermediate: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < total_elements
        row = offsets // intermediate
        col = offsets - row * intermediate

        gate = tl.load(gate_up + row * (2 * intermediate) + col, mask=mask, other=0.0)
        up = tl.load(gate_up + row * (2 * intermediate) + intermediate + col, mask=mask, other=0.0)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        value = gate * sigmoid * up
        tl.store(out + offsets, value, mask=mask)

    return _kernel


def _compile_expert_histogram_kernel(triton: Any, tl: Any) -> Any:
    @triton.jit
    def _kernel(topk_ids, counts, total_ids: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < total_ids
        expert_ids = tl.load(topk_ids + offsets, mask=mask, other=0)
        tl.atomic_add(counts + expert_ids, 1, sem="relaxed", mask=mask)

    return _kernel


def triton_fused_swiglu(gate_up: Any, *, block_size: int = 256) -> Any:
    """Compute fused SwiGLU for `[N, 2 * intermediate]` with Triton."""

    torch, triton, tl = _load_triton()
    if gate_up.ndim != 2:
        raise ValueError("gate_up must have shape [N, 2 * intermediate]")
    if gate_up.shape[1] % 2:
        raise ValueError("last dimension must be even")
    if not gate_up.is_cuda:
        raise ValueError("gate_up must be a CUDA tensor")

    intermediate = gate_up.shape[1] // 2
    out = torch.empty((gate_up.shape[0], intermediate), device=gate_up.device, dtype=gate_up.dtype)
    total_elements = out.numel()
    grid = (triton.cdiv(total_elements, block_size),)
    kernel = _compile_fused_swiglu_kernel(triton, tl)
    kernel[grid](gate_up, out, total_elements, intermediate, block=block_size)
    return out


def triton_expert_histogram(topk_ids: Any, num_experts: int, *, block_size: int = 256) -> Any:
    """Count routed tokens per expert with a Triton atomic histogram kernel."""

    torch, triton, tl = _load_triton()
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must have shape [tokens, top_k]")
    if not topk_ids.is_cuda:
        raise ValueError("topk_ids must be a CUDA tensor")

    ids = topk_ids.contiguous()
    counts = torch.zeros((num_experts,), device=ids.device, dtype=torch.int32)
    total_ids = ids.numel()
    grid = (triton.cdiv(total_ids, block_size),)
    kernel = _compile_expert_histogram_kernel(triton, tl)
    kernel[grid](ids, counts, total_ids, block=block_size)
    return counts

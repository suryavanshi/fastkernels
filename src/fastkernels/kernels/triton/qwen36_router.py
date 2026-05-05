"""Real-shape Triton routing kernels for Qwen3.6 MoE decode."""

from __future__ import annotations

from typing import Any

torch = None
triton = None
tl = None
_router_logits_kernel = None
_router_topk_kernel = None
_batched_router_logits_kernel = None
_batched_router_topk_kernel = None


def _load_triton() -> tuple[Any, Any, Any]:
    global torch, triton, tl
    try:
        import torch as torch_module
        import triton as triton_module
        import triton.language as tl_module
    except ImportError as exc:  # pragma: no cover - depends on optional packages
        raise RuntimeError("Triton and PyTorch are required for this kernel") from exc
    torch = torch_module
    triton = triton_module
    tl = tl_module
    return torch, triton, tl


def _compile_router_logits_kernel(triton: Any, tl: Any) -> Any:
    global _router_logits_kernel
    if _router_logits_kernel is not None:
        return _router_logits_kernel

    @triton.jit
    def _kernel(
        hidden,
        norm_weight,
        router_weight,
        logits,
        eps: tl.constexpr,
        norm_offset: tl.constexpr,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        expert = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        mask = offsets < hidden_size

        hidden_values = tl.load(hidden + offsets, mask=mask, other=0.0).to(tl.float32)
        norm_values = tl.load(norm_weight + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(hidden_values * hidden_values, axis=0) / hidden_size
        normalized = hidden_values * tl.rsqrt(mean_square + eps) * (norm_offset + norm_values)
        router_values = tl.load(
            router_weight + expert * hidden_size + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(logits + expert, tl.sum(router_values * normalized, axis=0))

    _router_logits_kernel = _kernel
    return _router_logits_kernel


def _compile_router_topk_kernel(triton: Any, tl: Any) -> Any:
    global _router_topk_kernel
    if _router_topk_kernel is not None:
        return _router_topk_kernel

    @triton.jit
    def _kernel(
        logits,
        topk_ids,
        topk_weights,
        num_experts: tl.constexpr,
        top_k: tl.constexpr,
        block_e: tl.constexpr,
    ):
        expert_offsets = tl.arange(0, block_e)
        expert_mask = expert_offsets < num_experts
        raw_logits = tl.load(logits + expert_offsets, mask=expert_mask, other=-float("inf")).to(tl.float32)
        max_logit = tl.max(raw_logits, axis=0)
        exp_logits = tl.exp(raw_logits - max_logit)
        probs = exp_logits / tl.sum(exp_logits, axis=0)
        probs = tl.where(expert_mask, probs, -1.0)

        remaining = probs
        denom = tl.full((), 0.0, tl.float32)
        selected_values = tl.full((8,), 0.0, tl.float32)
        selected_ids = tl.full((8,), 0, tl.int64)
        top_offsets = tl.arange(0, 8)

        for idx in tl.static_range(0, top_k):
            top_value = tl.max(remaining, axis=0)
            top_id = tl.max(tl.where(remaining == top_value, expert_offsets, 0), axis=0)
            selected_values = tl.where(top_offsets == idx, top_value, selected_values)
            selected_ids = tl.where(top_offsets == idx, top_id, selected_ids)
            denom += top_value
            remaining = tl.where(expert_offsets == top_id, -1.0, remaining)

        tl.store(topk_ids + top_offsets, selected_ids, mask=top_offsets < top_k)
        tl.store(topk_weights + top_offsets, selected_values / denom, mask=top_offsets < top_k)

    _router_topk_kernel = _kernel
    return _router_topk_kernel


def _compile_batched_router_logits_kernel(triton: Any, tl: Any) -> Any:
    global _batched_router_logits_kernel
    if _batched_router_logits_kernel is not None:
        return _batched_router_logits_kernel

    @triton.jit
    def _kernel(
        hidden,
        norm_weight,
        router_weight,
        logits,
        eps: tl.constexpr,
        norm_offset: tl.constexpr,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        block_h: tl.constexpr,
    ):
        token = tl.program_id(0)
        expert = tl.program_id(1)
        offsets = tl.arange(0, block_h)
        mask = offsets < hidden_size

        hidden_values = tl.load(hidden + token * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        norm_values = tl.load(norm_weight + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(hidden_values * hidden_values, axis=0) / hidden_size
        normalized = hidden_values * tl.rsqrt(mean_square + eps) * (norm_offset + norm_values)
        router_values = tl.load(
            router_weight + expert * hidden_size + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(logits + token * num_experts + expert, tl.sum(router_values * normalized, axis=0))

    _batched_router_logits_kernel = _kernel
    return _batched_router_logits_kernel


def _compile_batched_router_topk_kernel(triton: Any, tl: Any) -> Any:
    global _batched_router_topk_kernel
    if _batched_router_topk_kernel is not None:
        return _batched_router_topk_kernel

    @triton.jit
    def _kernel(
        logits,
        topk_ids,
        topk_weights,
        num_experts: tl.constexpr,
        top_k: tl.constexpr,
        block_e: tl.constexpr,
    ):
        token = tl.program_id(0)
        expert_offsets = tl.arange(0, block_e)
        expert_mask = expert_offsets < num_experts
        raw_logits = tl.load(
            logits + token * num_experts + expert_offsets,
            mask=expert_mask,
            other=-float("inf"),
        ).to(tl.float32)
        max_logit = tl.max(raw_logits, axis=0)
        exp_logits = tl.exp(raw_logits - max_logit)
        probs = exp_logits / tl.sum(exp_logits, axis=0)
        probs = tl.where(expert_mask, probs, -1.0)

        remaining = probs
        denom = tl.full((), 0.0, tl.float32)
        selected_values = tl.full((8,), 0.0, tl.float32)
        selected_ids = tl.full((8,), 0, tl.int64)
        top_offsets = tl.arange(0, 8)

        for idx in tl.static_range(0, top_k):
            top_value = tl.max(remaining, axis=0)
            top_id = tl.max(tl.where(remaining == top_value, expert_offsets, 0), axis=0)
            selected_values = tl.where(top_offsets == idx, top_value, selected_values)
            selected_ids = tl.where(top_offsets == idx, top_id, selected_ids)
            denom += top_value
            remaining = tl.where(expert_offsets == top_id, -1.0, remaining)

        tl.store(topk_ids + token * top_k + top_offsets, selected_ids, mask=top_offsets < top_k)
        tl.store(topk_weights + token * top_k + top_offsets, selected_values / denom, mask=top_offsets < top_k)

    _batched_router_topk_kernel = _kernel
    return _batched_router_topk_kernel


def triton_qwen36_moe_router_decode(
    hidden: Any,
    norm_weight: Any,
    router_weight: Any,
    *,
    top_k: int = 8,
    eps: float = 1e-6,
    norm_offset: float = 1.0,
) -> tuple[Any, Any, Any]:
    """Compute real-shape one-token Qwen3.6 MoE router logits and top-k weights.

    This is the first real-model-sized Qwen3.6 kernel in the repo: it supports
    the 35B-A3B router shape `[256, 2048]` and returns the softmax-renormalized
    top-k routing ids/weights used by the routed expert path. Expert MLP compute
    is still separate work.
    """

    torch, triton, tl = _load_triton()
    if hidden.ndim != 1:
        raise ValueError("hidden must have shape [hidden_size]")
    if norm_weight.shape != hidden.shape:
        raise ValueError("norm_weight must have the same shape as hidden")
    if router_weight.ndim != 2 or router_weight.shape[1] != hidden.shape[0]:
        raise ValueError("router_weight must have shape [num_experts, hidden_size]")
    if not hidden.is_cuda or not norm_weight.is_cuda or not router_weight.is_cuda:
        raise ValueError("hidden, norm_weight, and router_weight must be CUDA tensors")
    if top_k < 1 or top_k > 8:
        raise ValueError("router kernel supports 1 <= top_k <= 8")

    hidden_size = hidden.shape[0]
    num_experts = router_weight.shape[0]
    if hidden_size > 4096 or num_experts > 1024:
        raise ValueError("router kernel currently supports hidden_size<=4096 and num_experts<=1024")
    if top_k > num_experts:
        raise ValueError("top_k must be <= number of experts")

    logits = torch.empty((num_experts,), device=hidden.device, dtype=torch.float32)
    topk_ids = torch.empty((top_k,), device=hidden.device, dtype=torch.int64)
    topk_weights = torch.empty((top_k,), device=hidden.device, dtype=torch.float32)

    block_h = triton.next_power_of_2(hidden_size)
    block_e = triton.next_power_of_2(num_experts)

    logits_kernel = _compile_router_logits_kernel(triton, tl)
    logits_kernel[(num_experts,)](
        hidden,
        norm_weight,
        router_weight,
        logits,
        eps,
        norm_offset,
        hidden_size,
        block_h,
    )

    topk_kernel = _compile_router_topk_kernel(triton, tl)
    topk_kernel[(1,)](
        logits,
        topk_ids,
        topk_weights,
        num_experts,
        top_k,
        block_e,
    )
    return logits, topk_ids, topk_weights


def triton_qwen36_batched_moe_router_decode(
    hidden: Any,
    norm_weight: Any,
    router_weight: Any,
    *,
    top_k: int = 8,
    eps: float = 1e-6,
    norm_offset: float = 1.0,
) -> tuple[Any, Any, Any]:
    """Compute real-shape Qwen3.6 router logits/top-k for batched decode tokens."""

    torch, triton, tl = _load_triton()
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden_size]")
    if norm_weight.ndim != 1 or norm_weight.shape[0] != hidden.shape[1]:
        raise ValueError("norm_weight must have shape [hidden_size]")
    if router_weight.ndim != 2 or router_weight.shape[1] != hidden.shape[1]:
        raise ValueError("router_weight must have shape [num_experts, hidden_size]")
    if not hidden.is_cuda or not norm_weight.is_cuda or not router_weight.is_cuda:
        raise ValueError("hidden, norm_weight, and router_weight must be CUDA tensors")
    if top_k < 1 or top_k > 8:
        raise ValueError("router kernel supports 1 <= top_k <= 8")

    tokens, hidden_size = hidden.shape
    num_experts = router_weight.shape[0]
    if hidden_size > 4096 or num_experts > 1024:
        raise ValueError("router kernel currently supports hidden_size<=4096 and num_experts<=1024")
    if top_k > num_experts:
        raise ValueError("top_k must be <= number of experts")

    logits = torch.empty((tokens, num_experts), device=hidden.device, dtype=torch.float32)
    topk_ids = torch.empty((tokens, top_k), device=hidden.device, dtype=torch.int64)
    topk_weights = torch.empty((tokens, top_k), device=hidden.device, dtype=torch.float32)

    block_h = triton.next_power_of_2(hidden_size)
    block_e = triton.next_power_of_2(num_experts)

    logits_kernel = _compile_batched_router_logits_kernel(triton, tl)
    logits_kernel[(tokens, num_experts)](
        hidden,
        norm_weight,
        router_weight,
        logits,
        eps,
        norm_offset,
        hidden_size,
        num_experts,
        block_h,
    )

    topk_kernel = _compile_batched_router_topk_kernel(triton, tl)
    topk_kernel[(tokens,)](
        logits,
        topk_ids,
        topk_weights,
        num_experts,
        top_k,
        block_e,
    )
    return logits, topk_ids, topk_weights

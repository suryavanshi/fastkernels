"""Prototype Triton decode kernels for synthetic Qwen3.6 MoE blocks."""

from __future__ import annotations

from typing import Any

torch = None
triton = None
tl = None
_synthetic_moe_decode_kernel = None
_synthetic_moe_decode_top2_kernel = None


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


def _compile_synthetic_moe_decode_kernel(triton: Any, tl: Any) -> Any:
    global _synthetic_moe_decode_kernel
    if _synthetic_moe_decode_kernel is not None:
        return _synthetic_moe_decode_kernel

    @triton.jit
    def _kernel(
        hidden,
        norm_weight,
        router_weight,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        out,
        eps: tl.constexpr,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
        block_e: tl.constexpr,
        block_i: tl.constexpr,
    ):
        hidden_offsets = tl.arange(0, block_h)
        expert_offsets = tl.arange(0, block_e)
        intermediate_offsets = tl.arange(0, block_i)
        hidden_mask = hidden_offsets < hidden_size
        expert_mask = expert_offsets < num_experts
        intermediate_mask = intermediate_offsets < intermediate

        hidden_values = tl.load(hidden + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        norm_values = tl.load(norm_weight + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(hidden_values * hidden_values, axis=0) / hidden_size
        normalized = hidden_values * tl.rsqrt(mean_square + eps) * norm_values

        router = tl.load(
            router_weight + expert_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=expert_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(router * normalized[None, :], axis=1)
        logits = tl.where(expert_mask, logits, -float("inf"))
        max_logit = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - max_logit)
        probs = exp_logits / tl.sum(exp_logits, axis=0)

        gate_up_stride = 2 * intermediate * hidden_size
        down_stride = hidden_size * intermediate

        if top_k == 2:
            top1_value = tl.max(probs, axis=0)
            top1_id = tl.max(tl.where(probs == top1_value, expert_offsets, 0), axis=0)
            masked_probs = tl.where(expert_offsets == top1_id, -1.0, probs)
            top2_value = tl.max(masked_probs, axis=0)
            top2_id = tl.max(tl.where(masked_probs == top2_value, expert_offsets, 0), axis=0)
            top_denom = top1_value + top2_value
            top1_weight = top1_value / top_denom
            top2_weight = top2_value / top_denom

            top1_gate = tl.load(
                expert_gate_up_weight
                + top1_id * gate_up_stride
                + intermediate_offsets[:, None] * hidden_size
                + hidden_offsets[None, :],
                mask=intermediate_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            top1_up = tl.load(
                expert_gate_up_weight
                + top1_id * gate_up_stride
                + (intermediate + intermediate_offsets[:, None]) * hidden_size
                + hidden_offsets[None, :],
                mask=intermediate_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            top1_gate_values = tl.sum(top1_gate * hidden_values[None, :], axis=1)
            top1_up_values = tl.sum(top1_up * hidden_values[None, :], axis=1)
            top1_activation = top1_gate_values * (1.0 / (1.0 + tl.exp(-top1_gate_values))) * top1_up_values
            top1_down = tl.load(
                expert_down_weight
                + top1_id * down_stride
                + hidden_offsets[:, None] * intermediate
                + intermediate_offsets[None, :],
                mask=hidden_mask[:, None] & intermediate_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            top1_out = tl.sum(top1_down * top1_activation[None, :], axis=1)

            top2_gate = tl.load(
                expert_gate_up_weight
                + top2_id * gate_up_stride
                + intermediate_offsets[:, None] * hidden_size
                + hidden_offsets[None, :],
                mask=intermediate_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            top2_up = tl.load(
                expert_gate_up_weight
                + top2_id * gate_up_stride
                + (intermediate + intermediate_offsets[:, None]) * hidden_size
                + hidden_offsets[None, :],
                mask=intermediate_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            top2_gate_values = tl.sum(top2_gate * hidden_values[None, :], axis=1)
            top2_up_values = tl.sum(top2_up * hidden_values[None, :], axis=1)
            top2_activation = top2_gate_values * (1.0 / (1.0 + tl.exp(-top2_gate_values))) * top2_up_values
            top2_down = tl.load(
                expert_down_weight
                + top2_id * down_stride
                + hidden_offsets[:, None] * intermediate
                + intermediate_offsets[None, :],
                mask=hidden_mask[:, None] & intermediate_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            top2_out = tl.sum(top2_down * top2_activation[None, :], axis=1)
            routed_out = top1_weight * top1_out + top2_weight * top2_out
        else:
            remaining_probs = probs
            routed_weighted = tl.full((block_h,), 0.0, tl.float32)
            top_denom = tl.full((), 0.0, tl.float32)

            for _ in tl.static_range(0, top_k):
                top_value = tl.max(remaining_probs, axis=0)
                top_id = tl.max(tl.where(remaining_probs == top_value, expert_offsets, 0), axis=0)
                top_denom += top_value

                top_gate = tl.load(
                    expert_gate_up_weight
                    + top_id * gate_up_stride
                    + intermediate_offsets[:, None] * hidden_size
                    + hidden_offsets[None, :],
                    mask=intermediate_mask[:, None] & hidden_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                top_up = tl.load(
                    expert_gate_up_weight
                    + top_id * gate_up_stride
                    + (intermediate + intermediate_offsets[:, None]) * hidden_size
                    + hidden_offsets[None, :],
                    mask=intermediate_mask[:, None] & hidden_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                top_gate_values = tl.sum(top_gate * hidden_values[None, :], axis=1)
                top_up_values = tl.sum(top_up * hidden_values[None, :], axis=1)
                top_activation = top_gate_values * (1.0 / (1.0 + tl.exp(-top_gate_values))) * top_up_values
                top_down = tl.load(
                    expert_down_weight
                    + top_id * down_stride
                    + hidden_offsets[:, None] * intermediate
                    + intermediate_offsets[None, :],
                    mask=hidden_mask[:, None] & intermediate_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                top_out = tl.sum(top_down * top_activation[None, :], axis=1)
                routed_weighted += top_value * top_out
                remaining_probs = tl.where(expert_offsets == top_id, -1.0, remaining_probs)
            routed_out = routed_weighted / top_denom

        shared_gate = tl.load(
            shared_gate_up_weight + intermediate_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        shared_up = tl.load(
            shared_gate_up_weight
            + (intermediate + intermediate_offsets[:, None]) * hidden_size
            + hidden_offsets[None, :],
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        shared_gate_values = tl.sum(shared_gate * hidden_values[None, :], axis=1)
        shared_up_values = tl.sum(shared_up * hidden_values[None, :], axis=1)
        shared_activation = shared_gate_values * (1.0 / (1.0 + tl.exp(-shared_gate_values))) * shared_up_values
        shared_down = tl.load(
            shared_down_weight + hidden_offsets[:, None] * intermediate + intermediate_offsets[None, :],
            mask=hidden_mask[:, None] & intermediate_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        shared_out = tl.sum(shared_down * shared_activation[None, :], axis=1)

        output = hidden_values + routed_out + shared_out
        tl.store(out + hidden_offsets, output, mask=hidden_mask)

    _synthetic_moe_decode_kernel = _kernel
    return _synthetic_moe_decode_kernel


def _compile_synthetic_moe_decode_top2_kernel(triton: Any, tl: Any) -> Any:
    global _synthetic_moe_decode_top2_kernel
    if _synthetic_moe_decode_top2_kernel is not None:
        return _synthetic_moe_decode_top2_kernel

    @triton.jit
    def _kernel(
        hidden,
        norm_weight,
        router_weight,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        out,
        eps: tl.constexpr,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        block_h: tl.constexpr,
        block_e: tl.constexpr,
        block_i: tl.constexpr,
    ):
        hidden_offsets = tl.arange(0, block_h)
        expert_offsets = tl.arange(0, block_e)
        intermediate_offsets = tl.arange(0, block_i)
        hidden_mask = hidden_offsets < hidden_size
        expert_mask = expert_offsets < num_experts
        intermediate_mask = intermediate_offsets < intermediate

        hidden_values = tl.load(hidden + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        norm_values = tl.load(norm_weight + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(hidden_values * hidden_values, axis=0) / hidden_size
        normalized = hidden_values * tl.rsqrt(mean_square + eps) * norm_values

        router = tl.load(
            router_weight + expert_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=expert_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(router * normalized[None, :], axis=1)
        logits = tl.where(expert_mask, logits, -float("inf"))
        max_logit = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - max_logit)
        probs = exp_logits / tl.sum(exp_logits, axis=0)

        top1_value = tl.max(probs, axis=0)
        top1_id = tl.max(tl.where(probs == top1_value, expert_offsets, 0), axis=0)
        masked_probs = tl.where(expert_offsets == top1_id, -1.0, probs)
        top2_value = tl.max(masked_probs, axis=0)
        top2_id = tl.max(tl.where(masked_probs == top2_value, expert_offsets, 0), axis=0)
        top_denom = top1_value + top2_value
        top1_weight = top1_value / top_denom
        top2_weight = top2_value / top_denom

        gate_up_stride = 2 * intermediate * hidden_size
        down_stride = hidden_size * intermediate

        top1_gate = tl.load(
            expert_gate_up_weight
            + top1_id * gate_up_stride
            + intermediate_offsets[:, None] * hidden_size
            + hidden_offsets[None, :],
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        top1_up = tl.load(
            expert_gate_up_weight
            + top1_id * gate_up_stride
            + (intermediate + intermediate_offsets[:, None]) * hidden_size
            + hidden_offsets[None, :],
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        top1_gate_values = tl.sum(top1_gate * hidden_values[None, :], axis=1)
        top1_up_values = tl.sum(top1_up * hidden_values[None, :], axis=1)
        top1_activation = top1_gate_values * (1.0 / (1.0 + tl.exp(-top1_gate_values))) * top1_up_values
        top1_down = tl.load(
            expert_down_weight
            + top1_id * down_stride
            + hidden_offsets[:, None] * intermediate
            + intermediate_offsets[None, :],
            mask=hidden_mask[:, None] & intermediate_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        top1_out = tl.sum(top1_down * top1_activation[None, :], axis=1)

        top2_gate = tl.load(
            expert_gate_up_weight
            + top2_id * gate_up_stride
            + intermediate_offsets[:, None] * hidden_size
            + hidden_offsets[None, :],
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        top2_up = tl.load(
            expert_gate_up_weight
            + top2_id * gate_up_stride
            + (intermediate + intermediate_offsets[:, None]) * hidden_size
            + hidden_offsets[None, :],
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        top2_gate_values = tl.sum(top2_gate * hidden_values[None, :], axis=1)
        top2_up_values = tl.sum(top2_up * hidden_values[None, :], axis=1)
        top2_activation = top2_gate_values * (1.0 / (1.0 + tl.exp(-top2_gate_values))) * top2_up_values
        top2_down = tl.load(
            expert_down_weight
            + top2_id * down_stride
            + hidden_offsets[:, None] * intermediate
            + intermediate_offsets[None, :],
            mask=hidden_mask[:, None] & intermediate_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        top2_out = tl.sum(top2_down * top2_activation[None, :], axis=1)

        shared_gate = tl.load(
            shared_gate_up_weight + intermediate_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        shared_up = tl.load(
            shared_gate_up_weight
            + (intermediate + intermediate_offsets[:, None]) * hidden_size
            + hidden_offsets[None, :],
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        shared_gate_values = tl.sum(shared_gate * hidden_values[None, :], axis=1)
        shared_up_values = tl.sum(shared_up * hidden_values[None, :], axis=1)
        shared_activation = shared_gate_values * (1.0 / (1.0 + tl.exp(-shared_gate_values))) * shared_up_values
        shared_down = tl.load(
            shared_down_weight + hidden_offsets[:, None] * intermediate + intermediate_offsets[None, :],
            mask=hidden_mask[:, None] & intermediate_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        shared_out = tl.sum(shared_down * shared_activation[None, :], axis=1)

        output = hidden_values + top1_weight * top1_out + top2_weight * top2_out + shared_out
        tl.store(out + hidden_offsets, output, mask=hidden_mask)

    _synthetic_moe_decode_top2_kernel = _kernel
    return _synthetic_moe_decode_top2_kernel


def triton_synthetic_qwen36_moe_decode(
    hidden: Any,
    norm_weight: Any,
    router_weight: Any,
    expert_gate_up_weight: Any,
    expert_down_weight: Any,
    shared_gate_up_weight: Any,
    shared_down_weight: Any,
    *,
    top_k: int,
    eps: float = 1e-6,
) -> Any:
    """Run a prototype one-token Qwen3.6 synthetic MoE decode block.

    This intentionally targets the tiny synthetic spec used for megakernel
    dataflow tests. It fuses RMSNorm for routing, router softmax/top-2,
    routed expert MLPs, the shared expert MLP, and the residual add.
    """

    torch, triton, tl = _load_triton()
    if top_k < 1 or top_k > 8:
        raise ValueError("prototype synthetic Qwen3.6 MoE kernel supports 1 <= top_k <= 8")
    if hidden.ndim != 1:
        raise ValueError("hidden must have shape [hidden_size]")
    if not hidden.is_cuda:
        raise ValueError("hidden must be a CUDA tensor")
    if expert_gate_up_weight.ndim != 3:
        raise ValueError("expert_gate_up_weight must have shape [experts, 2 * intermediate, hidden]")
    if expert_down_weight.ndim != 3:
        raise ValueError("expert_down_weight must have shape [experts, hidden, intermediate]")

    hidden_size = hidden.shape[0]
    num_experts = router_weight.shape[0]
    intermediate = expert_down_weight.shape[2]
    if hidden_size > 128 or intermediate > 128 or num_experts > 256:
        raise ValueError("prototype synthetic Qwen3.6 MoE kernel supports hidden/intermediate<=128 and experts<=256")
    if top_k > num_experts:
        raise ValueError("top_k must be <= number of experts")

    out = torch.empty_like(hidden)
    block_h = triton.next_power_of_2(hidden_size)
    block_i = triton.next_power_of_2(intermediate)
    block_e = triton.next_power_of_2(num_experts)
    if top_k == 2:
        kernel = _compile_synthetic_moe_decode_top2_kernel(triton, tl)
        kernel[(1,)](
            hidden,
            norm_weight,
            router_weight,
            expert_gate_up_weight,
            expert_down_weight,
            shared_gate_up_weight,
            shared_down_weight,
            out,
            eps,
            hidden_size,
            num_experts,
            intermediate,
            block_h,
            block_e,
            block_i,
        )
    else:
        kernel = _compile_synthetic_moe_decode_kernel(triton, tl)
        kernel[(1,)](
            hidden,
            norm_weight,
            router_weight,
            expert_gate_up_weight,
            expert_down_weight,
            shared_gate_up_weight,
            shared_down_weight,
            out,
            eps,
            hidden_size,
            num_experts,
            intermediate,
            top_k,
            block_h,
            block_e,
            block_i,
        )
    return out

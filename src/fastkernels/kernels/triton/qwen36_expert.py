"""Real-shape Triton expert MLP kernels for Qwen3.6 MoE decode."""

from __future__ import annotations

from typing import Any

torch = None
triton = None
tl = None
_expert_activation_kernel = None
_expert_down_kernel = None
_routed_activation_kernel = None
_routed_down_kernel = None
_routed_shared_activation_kernel = None
_routed_shared_down_kernel = None
_shared_expert_gate_kernel = None
_routed_shared_gated_down_kernel = None
_batched_routed_activation_kernel = None
_batched_routed_down_kernel = None
_batched_routed_shared_activation_kernel = None
_batched_routed_shared_down_kernel = None
_batched_shared_expert_gate_kernel = None
_batched_routed_shared_gated_down_kernel = None


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


def _compile_expert_activation_kernel(triton: Any, tl: Any) -> Any:
    global _expert_activation_kernel
    if _expert_activation_kernel is not None:
        return _expert_activation_kernel

    @triton.jit
    def _kernel(
        hidden,
        expert_gate_up_weight,
        activation,
        hidden_size: tl.constexpr,
        intermediate: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        block = tl.program_id(0)
        intermediate_offsets = block * block_i + tl.arange(0, block_i)
        hidden_offsets = tl.arange(0, block_h)
        intermediate_mask = intermediate_offsets < intermediate
        hidden_mask = hidden_offsets < hidden_size

        hidden_values = tl.load(hidden + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        gate_weights = tl.load(
            expert_gate_up_weight + intermediate_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        up_weights = tl.load(
            expert_gate_up_weight
            + (intermediate + intermediate_offsets[:, None]) * hidden_size
            + hidden_offsets[None, :],
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        gate_values = tl.sum(gate_weights * hidden_values[None, :], axis=1)
        up_values = tl.sum(up_weights * hidden_values[None, :], axis=1)
        silu = gate_values * (1.0 / (1.0 + tl.exp(-gate_values)))
        tl.store(activation + intermediate_offsets, silu * up_values, mask=intermediate_mask)

    _expert_activation_kernel = _kernel
    return _expert_activation_kernel


def _compile_expert_down_kernel(triton: Any, tl: Any) -> Any:
    global _expert_down_kernel
    if _expert_down_kernel is not None:
        return _expert_down_kernel

    @triton.jit
    def _kernel(
        activation,
        expert_down_weight,
        out,
        hidden_size: tl.constexpr,
        intermediate: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        block = tl.program_id(0)
        hidden_offsets = block * block_h + tl.arange(0, block_h)
        intermediate_offsets = tl.arange(0, block_i)
        hidden_mask = hidden_offsets < hidden_size
        intermediate_mask = intermediate_offsets < intermediate

        activation_values = tl.load(activation + intermediate_offsets, mask=intermediate_mask, other=0.0).to(
            tl.float32
        )
        down_weights = tl.load(
            expert_down_weight + hidden_offsets[:, None] * intermediate + intermediate_offsets[None, :],
            mask=hidden_mask[:, None] & intermediate_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        values = tl.sum(down_weights * activation_values[None, :], axis=1)
        tl.store(out + hidden_offsets, values, mask=hidden_mask)

    _expert_down_kernel = _kernel
    return _expert_down_kernel


def _compile_routed_activation_kernel(triton: Any, tl: Any) -> Any:
    global _routed_activation_kernel
    if _routed_activation_kernel is not None:
        return _routed_activation_kernel

    @triton.jit
    def _kernel(
        hidden,
        topk_ids,
        expert_gate_up_weight,
        activation,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block = tl.program_id(1)
        intermediate_offsets = block * block_i + tl.arange(0, block_i)
        hidden_offsets = tl.arange(0, block_h)
        intermediate_mask = intermediate_offsets < intermediate
        hidden_mask = hidden_offsets < hidden_size

        expert_id = tl.load(topk_ids + slot, mask=slot < top_k, other=0)
        expert_id = tl.minimum(tl.maximum(expert_id, 0), num_experts - 1)
        expert_stride = 2 * intermediate * hidden_size
        gate_offsets = (
            expert_id * expert_stride
            + intermediate_offsets[:, None] * hidden_size
            + hidden_offsets[None, :]
        )
        up_offsets = (
            expert_id * expert_stride
            + (intermediate + intermediate_offsets[:, None]) * hidden_size
            + hidden_offsets[None, :]
        )

        hidden_values = tl.load(hidden + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        gate_weights = tl.load(
            expert_gate_up_weight + gate_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        up_weights = tl.load(
            expert_gate_up_weight + up_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_values = tl.sum(gate_weights * hidden_values[None, :], axis=1)
        up_values = tl.sum(up_weights * hidden_values[None, :], axis=1)
        silu = gate_values * (1.0 / (1.0 + tl.exp(-gate_values)))
        tl.store(activation + slot * intermediate + intermediate_offsets, silu * up_values, mask=intermediate_mask)

    _routed_activation_kernel = _kernel
    return _routed_activation_kernel


def _compile_routed_down_kernel(triton: Any, tl: Any) -> Any:
    global _routed_down_kernel
    if _routed_down_kernel is not None:
        return _routed_down_kernel

    @triton.jit
    def _kernel(
        activation,
        topk_ids,
        topk_weights,
        expert_down_weight,
        out,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        block = tl.program_id(0)
        hidden_offsets = block * block_h + tl.arange(0, block_h)
        intermediate_offsets = tl.arange(0, block_i)
        hidden_mask = hidden_offsets < hidden_size
        intermediate_mask = intermediate_offsets < intermediate
        values = tl.full((block_h,), 0.0, tl.float32)

        for slot in tl.static_range(0, top_k):
            expert_id = tl.load(topk_ids + slot)
            expert_id = tl.minimum(tl.maximum(expert_id, 0), num_experts - 1)
            weight = tl.load(topk_weights + slot).to(tl.float32)
            activation_values = tl.load(
                activation + slot * intermediate + intermediate_offsets,
                mask=intermediate_mask,
                other=0.0,
            ).to(tl.float32)
            down_weights = tl.load(
                expert_down_weight
                + expert_id * hidden_size * intermediate
                + hidden_offsets[:, None] * intermediate
                + intermediate_offsets[None, :],
                mask=hidden_mask[:, None] & intermediate_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            values += weight * tl.sum(down_weights * activation_values[None, :], axis=1)

        tl.store(out + hidden_offsets, values, mask=hidden_mask)

    _routed_down_kernel = _kernel
    return _routed_down_kernel


def _compile_routed_shared_activation_kernel(triton: Any, tl: Any) -> Any:
    global _routed_shared_activation_kernel
    if _routed_shared_activation_kernel is not None:
        return _routed_shared_activation_kernel

    @triton.jit
    def _kernel(
        hidden,
        topk_ids,
        expert_gate_up_weight,
        shared_gate_up_weight,
        activation,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block = tl.program_id(1)
        intermediate_offsets = block * block_i + tl.arange(0, block_i)
        hidden_offsets = tl.arange(0, block_h)
        intermediate_mask = intermediate_offsets < intermediate
        hidden_mask = hidden_offsets < hidden_size
        is_shared = slot == top_k

        expert_id = tl.load(topk_ids + slot, mask=slot < top_k, other=0)
        expert_id = tl.minimum(tl.maximum(expert_id, 0), num_experts - 1)
        expert_stride = 2 * intermediate * hidden_size
        routed_gate_offsets = (
            expert_id * expert_stride
            + intermediate_offsets[:, None] * hidden_size
            + hidden_offsets[None, :]
        )
        routed_up_offsets = (
            expert_id * expert_stride
            + (intermediate + intermediate_offsets[:, None]) * hidden_size
            + hidden_offsets[None, :]
        )
        shared_gate_offsets = intermediate_offsets[:, None] * hidden_size + hidden_offsets[None, :]
        shared_up_offsets = (intermediate + intermediate_offsets[:, None]) * hidden_size + hidden_offsets[None, :]

        hidden_values = tl.load(hidden + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        routed_gate = tl.load(
            expert_gate_up_weight + routed_gate_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        routed_up = tl.load(
            expert_gate_up_weight + routed_up_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        shared_gate = tl.load(
            shared_gate_up_weight + shared_gate_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        shared_up = tl.load(
            shared_gate_up_weight + shared_up_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_weights = tl.where(is_shared, shared_gate, routed_gate)
        up_weights = tl.where(is_shared, shared_up, routed_up)
        gate_values = tl.sum(gate_weights * hidden_values[None, :], axis=1)
        up_values = tl.sum(up_weights * hidden_values[None, :], axis=1)
        silu = gate_values * (1.0 / (1.0 + tl.exp(-gate_values)))
        tl.store(activation + slot * intermediate + intermediate_offsets, silu * up_values, mask=intermediate_mask)

    _routed_shared_activation_kernel = _kernel
    return _routed_shared_activation_kernel


def _compile_routed_shared_down_kernel(triton: Any, tl: Any) -> Any:
    global _routed_shared_down_kernel
    if _routed_shared_down_kernel is not None:
        return _routed_shared_down_kernel

    @triton.jit
    def _kernel(
        activation,
        topk_ids,
        topk_weights,
        expert_down_weight,
        shared_down_weight,
        out,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        block = tl.program_id(0)
        hidden_offsets = block * block_h + tl.arange(0, block_h)
        intermediate_offsets = tl.arange(0, block_i)
        hidden_mask = hidden_offsets < hidden_size
        intermediate_mask = intermediate_offsets < intermediate
        values = tl.full((block_h,), 0.0, tl.float32)

        for slot in tl.static_range(0, top_k):
            expert_id = tl.load(topk_ids + slot)
            expert_id = tl.minimum(tl.maximum(expert_id, 0), num_experts - 1)
            weight = tl.load(topk_weights + slot).to(tl.float32)
            activation_values = tl.load(
                activation + slot * intermediate + intermediate_offsets,
                mask=intermediate_mask,
                other=0.0,
            ).to(tl.float32)
            down_weights = tl.load(
                expert_down_weight
                + expert_id * hidden_size * intermediate
                + hidden_offsets[:, None] * intermediate
                + intermediate_offsets[None, :],
                mask=hidden_mask[:, None] & intermediate_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            values += weight * tl.sum(down_weights * activation_values[None, :], axis=1)

        shared_activation = tl.load(
            activation + top_k * intermediate + intermediate_offsets,
            mask=intermediate_mask,
            other=0.0,
        ).to(tl.float32)
        shared_down = tl.load(
            shared_down_weight + hidden_offsets[:, None] * intermediate + intermediate_offsets[None, :],
            mask=hidden_mask[:, None] & intermediate_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        values += tl.sum(shared_down * shared_activation[None, :], axis=1)
        tl.store(out + hidden_offsets, values, mask=hidden_mask)

    _routed_shared_down_kernel = _kernel
    return _routed_shared_down_kernel


def _compile_shared_expert_gate_kernel(triton: Any, tl: Any) -> Any:
    global _shared_expert_gate_kernel
    if _shared_expert_gate_kernel is not None:
        return _shared_expert_gate_kernel

    @triton.jit
    def _kernel(
        hidden,
        shared_expert_gate_weight,
        shared_gate,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        offsets = tl.arange(0, block_h)
        mask = offsets < hidden_size
        hidden_values = tl.load(hidden + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_values = tl.load(shared_expert_gate_weight + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_logit = tl.sum(hidden_values * gate_values, axis=0)
        tl.store(shared_gate, 1.0 / (1.0 + tl.exp(-gate_logit)))

    _shared_expert_gate_kernel = _kernel
    return _shared_expert_gate_kernel


def _compile_routed_shared_gated_down_kernel(triton: Any, tl: Any) -> Any:
    global _routed_shared_gated_down_kernel
    if _routed_shared_gated_down_kernel is not None:
        return _routed_shared_gated_down_kernel

    @triton.jit
    def _kernel(
        activation,
        topk_ids,
        topk_weights,
        expert_down_weight,
        shared_down_weight,
        shared_gate,
        out,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        block = tl.program_id(0)
        hidden_offsets = block * block_h + tl.arange(0, block_h)
        intermediate_offsets = tl.arange(0, block_i)
        hidden_mask = hidden_offsets < hidden_size
        intermediate_mask = intermediate_offsets < intermediate
        values = tl.full((block_h,), 0.0, tl.float32)

        for slot in tl.static_range(0, top_k):
            expert_id = tl.load(topk_ids + slot)
            expert_id = tl.minimum(tl.maximum(expert_id, 0), num_experts - 1)
            weight = tl.load(topk_weights + slot).to(tl.float32)
            activation_values = tl.load(
                activation + slot * intermediate + intermediate_offsets,
                mask=intermediate_mask,
                other=0.0,
            ).to(tl.float32)
            down_weights = tl.load(
                expert_down_weight
                + expert_id * hidden_size * intermediate
                + hidden_offsets[:, None] * intermediate
                + intermediate_offsets[None, :],
                mask=hidden_mask[:, None] & intermediate_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            values += weight * tl.sum(down_weights * activation_values[None, :], axis=1)

        shared_activation = tl.load(
            activation + top_k * intermediate + intermediate_offsets,
            mask=intermediate_mask,
            other=0.0,
        ).to(tl.float32)
        shared_down = tl.load(
            shared_down_weight + hidden_offsets[:, None] * intermediate + intermediate_offsets[None, :],
            mask=hidden_mask[:, None] & intermediate_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        values += tl.load(shared_gate).to(tl.float32) * tl.sum(shared_down * shared_activation[None, :], axis=1)
        tl.store(out + hidden_offsets, values, mask=hidden_mask)

    _routed_shared_gated_down_kernel = _kernel
    return _routed_shared_gated_down_kernel


def _compile_batched_routed_shared_activation_kernel(triton: Any, tl: Any) -> Any:
    global _batched_routed_shared_activation_kernel
    if _batched_routed_shared_activation_kernel is not None:
        return _batched_routed_shared_activation_kernel

    @triton.jit
    def _kernel(
        hidden,
        topk_ids,
        expert_gate_up_weight,
        shared_gate_up_weight,
        activation,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        token = tl.program_id(0)
        slot = tl.program_id(1)
        block = tl.program_id(2)
        intermediate_offsets = block * block_i + tl.arange(0, block_i)
        hidden_offsets = tl.arange(0, block_h)
        intermediate_mask = intermediate_offsets < intermediate
        hidden_mask = hidden_offsets < hidden_size
        is_shared = slot == top_k

        expert_id = tl.load(topk_ids + token * top_k + slot, mask=slot < top_k, other=0)
        expert_id = tl.minimum(tl.maximum(expert_id, 0), num_experts - 1)
        expert_stride = 2 * intermediate * hidden_size
        routed_gate_offsets = (
            expert_id * expert_stride
            + intermediate_offsets[:, None] * hidden_size
            + hidden_offsets[None, :]
        )
        routed_up_offsets = (
            expert_id * expert_stride
            + (intermediate + intermediate_offsets[:, None]) * hidden_size
            + hidden_offsets[None, :]
        )
        shared_gate_offsets = intermediate_offsets[:, None] * hidden_size + hidden_offsets[None, :]
        shared_up_offsets = (intermediate + intermediate_offsets[:, None]) * hidden_size + hidden_offsets[None, :]

        hidden_values = tl.load(
            hidden + token * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        routed_gate = tl.load(
            expert_gate_up_weight + routed_gate_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        routed_up = tl.load(
            expert_gate_up_weight + routed_up_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        shared_gate = tl.load(
            shared_gate_up_weight + shared_gate_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        shared_up = tl.load(
            shared_gate_up_weight + shared_up_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_weights = tl.where(is_shared, shared_gate, routed_gate)
        up_weights = tl.where(is_shared, shared_up, routed_up)
        gate_values = tl.sum(gate_weights * hidden_values[None, :], axis=1)
        up_values = tl.sum(up_weights * hidden_values[None, :], axis=1)
        silu = gate_values * (1.0 / (1.0 + tl.exp(-gate_values)))
        tl.store(
            activation + (token * (top_k + 1) + slot) * intermediate + intermediate_offsets,
            silu * up_values,
            mask=intermediate_mask,
        )

    _batched_routed_shared_activation_kernel = _kernel
    return _batched_routed_shared_activation_kernel


def _compile_batched_routed_activation_kernel(triton: Any, tl: Any) -> Any:
    global _batched_routed_activation_kernel
    if _batched_routed_activation_kernel is not None:
        return _batched_routed_activation_kernel

    @triton.jit
    def _kernel(
        hidden,
        topk_ids,
        expert_gate_up_weight,
        activation,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        token = tl.program_id(0)
        slot = tl.program_id(1)
        block = tl.program_id(2)
        intermediate_offsets = block * block_i + tl.arange(0, block_i)
        hidden_offsets = tl.arange(0, block_h)
        intermediate_mask = intermediate_offsets < intermediate
        hidden_mask = hidden_offsets < hidden_size

        expert_id = tl.load(topk_ids + token * top_k + slot)
        expert_id = tl.minimum(tl.maximum(expert_id, 0), num_experts - 1)
        expert_stride = 2 * intermediate * hidden_size
        gate_offsets = (
            expert_id * expert_stride
            + intermediate_offsets[:, None] * hidden_size
            + hidden_offsets[None, :]
        )
        up_offsets = (
            expert_id * expert_stride
            + (intermediate + intermediate_offsets[:, None]) * hidden_size
            + hidden_offsets[None, :]
        )

        hidden_values = tl.load(
            hidden + token * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        gate_weights = tl.load(
            expert_gate_up_weight + gate_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        up_weights = tl.load(
            expert_gate_up_weight + up_offsets,
            mask=intermediate_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_values = tl.sum(gate_weights * hidden_values[None, :], axis=1)
        up_values = tl.sum(up_weights * hidden_values[None, :], axis=1)
        silu = gate_values * (1.0 / (1.0 + tl.exp(-gate_values)))
        tl.store(
            activation + (token * top_k + slot) * intermediate + intermediate_offsets,
            silu * up_values,
            mask=intermediate_mask,
        )

    _batched_routed_activation_kernel = _kernel
    return _batched_routed_activation_kernel


def _compile_batched_routed_down_kernel(triton: Any, tl: Any) -> Any:
    global _batched_routed_down_kernel
    if _batched_routed_down_kernel is not None:
        return _batched_routed_down_kernel

    @triton.jit
    def _kernel(
        activation,
        topk_ids,
        topk_weights,
        expert_down_weight,
        out,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        hidden_offsets = block * block_h + tl.arange(0, block_h)
        intermediate_offsets = tl.arange(0, block_i)
        hidden_mask = hidden_offsets < hidden_size
        intermediate_mask = intermediate_offsets < intermediate
        values = tl.full((block_h,), 0.0, tl.float32)

        for slot in tl.static_range(0, top_k):
            expert_id = tl.load(topk_ids + token * top_k + slot)
            expert_id = tl.minimum(tl.maximum(expert_id, 0), num_experts - 1)
            weight = tl.load(topk_weights + token * top_k + slot).to(tl.float32)
            activation_values = tl.load(
                activation + (token * top_k + slot) * intermediate + intermediate_offsets,
                mask=intermediate_mask,
                other=0.0,
            ).to(tl.float32)
            down_weights = tl.load(
                expert_down_weight
                + expert_id * hidden_size * intermediate
                + hidden_offsets[:, None] * intermediate
                + intermediate_offsets[None, :],
                mask=hidden_mask[:, None] & intermediate_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            values += weight * tl.sum(down_weights * activation_values[None, :], axis=1)

        tl.store(out + token * hidden_size + hidden_offsets, values, mask=hidden_mask)

    _batched_routed_down_kernel = _kernel
    return _batched_routed_down_kernel


def _compile_batched_routed_shared_down_kernel(triton: Any, tl: Any) -> Any:
    global _batched_routed_shared_down_kernel
    if _batched_routed_shared_down_kernel is not None:
        return _batched_routed_shared_down_kernel

    @triton.jit
    def _kernel(
        activation,
        topk_ids,
        topk_weights,
        expert_down_weight,
        shared_down_weight,
        out,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        hidden_offsets = block * block_h + tl.arange(0, block_h)
        intermediate_offsets = tl.arange(0, block_i)
        hidden_mask = hidden_offsets < hidden_size
        intermediate_mask = intermediate_offsets < intermediate
        values = tl.full((block_h,), 0.0, tl.float32)

        for slot in tl.static_range(0, top_k):
            expert_id = tl.load(topk_ids + token * top_k + slot)
            expert_id = tl.minimum(tl.maximum(expert_id, 0), num_experts - 1)
            weight = tl.load(topk_weights + token * top_k + slot).to(tl.float32)
            activation_values = tl.load(
                activation + (token * (top_k + 1) + slot) * intermediate + intermediate_offsets,
                mask=intermediate_mask,
                other=0.0,
            ).to(tl.float32)
            down_weights = tl.load(
                expert_down_weight
                + expert_id * hidden_size * intermediate
                + hidden_offsets[:, None] * intermediate
                + intermediate_offsets[None, :],
                mask=hidden_mask[:, None] & intermediate_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            values += weight * tl.sum(down_weights * activation_values[None, :], axis=1)

        shared_activation = tl.load(
            activation + (token * (top_k + 1) + top_k) * intermediate + intermediate_offsets,
            mask=intermediate_mask,
            other=0.0,
        ).to(tl.float32)
        shared_down = tl.load(
            shared_down_weight + hidden_offsets[:, None] * intermediate + intermediate_offsets[None, :],
            mask=hidden_mask[:, None] & intermediate_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        values += tl.sum(shared_down * shared_activation[None, :], axis=1)
        tl.store(out + token * hidden_size + hidden_offsets, values, mask=hidden_mask)

    _batched_routed_shared_down_kernel = _kernel
    return _batched_routed_shared_down_kernel


def _compile_batched_shared_expert_gate_kernel(triton: Any, tl: Any) -> Any:
    global _batched_shared_expert_gate_kernel
    if _batched_shared_expert_gate_kernel is not None:
        return _batched_shared_expert_gate_kernel

    @triton.jit
    def _kernel(
        hidden,
        shared_expert_gate_weight,
        shared_gate,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        mask = offsets < hidden_size
        hidden_values = tl.load(hidden + token * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_values = tl.load(shared_expert_gate_weight + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_logit = tl.sum(hidden_values * gate_values, axis=0)
        tl.store(shared_gate + token, 1.0 / (1.0 + tl.exp(-gate_logit)))

    _batched_shared_expert_gate_kernel = _kernel
    return _batched_shared_expert_gate_kernel


def _compile_batched_routed_shared_gated_down_kernel(triton: Any, tl: Any) -> Any:
    global _batched_routed_shared_gated_down_kernel
    if _batched_routed_shared_gated_down_kernel is not None:
        return _batched_routed_shared_gated_down_kernel

    @triton.jit
    def _kernel(
        activation,
        topk_ids,
        topk_weights,
        expert_down_weight,
        shared_down_weight,
        shared_gate,
        out,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        intermediate: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        hidden_offsets = block * block_h + tl.arange(0, block_h)
        intermediate_offsets = tl.arange(0, block_i)
        hidden_mask = hidden_offsets < hidden_size
        intermediate_mask = intermediate_offsets < intermediate
        values = tl.full((block_h,), 0.0, tl.float32)

        for slot in tl.static_range(0, top_k):
            expert_id = tl.load(topk_ids + token * top_k + slot)
            expert_id = tl.minimum(tl.maximum(expert_id, 0), num_experts - 1)
            weight = tl.load(topk_weights + token * top_k + slot).to(tl.float32)
            activation_values = tl.load(
                activation + (token * (top_k + 1) + slot) * intermediate + intermediate_offsets,
                mask=intermediate_mask,
                other=0.0,
            ).to(tl.float32)
            down_weights = tl.load(
                expert_down_weight
                + expert_id * hidden_size * intermediate
                + hidden_offsets[:, None] * intermediate
                + intermediate_offsets[None, :],
                mask=hidden_mask[:, None] & intermediate_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            values += weight * tl.sum(down_weights * activation_values[None, :], axis=1)

        shared_activation = tl.load(
            activation + (token * (top_k + 1) + top_k) * intermediate + intermediate_offsets,
            mask=intermediate_mask,
            other=0.0,
        ).to(tl.float32)
        shared_down = tl.load(
            shared_down_weight + hidden_offsets[:, None] * intermediate + intermediate_offsets[None, :],
            mask=hidden_mask[:, None] & intermediate_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        values += tl.load(shared_gate + token).to(tl.float32) * tl.sum(
            shared_down * shared_activation[None, :],
            axis=1,
        )
        tl.store(out + token * hidden_size + hidden_offsets, values, mask=hidden_mask)

    _batched_routed_shared_gated_down_kernel = _kernel
    return _batched_routed_shared_gated_down_kernel


def triton_qwen36_single_expert_mlp_decode(
    hidden: Any,
    expert_gate_up_weight: Any,
    expert_down_weight: Any,
    *,
    block_intermediate: int = 16,
    block_hidden: int = 16,
) -> Any:
    """Compute one real-shape Qwen3.6 expert MLP for one decode token.

    Supports the 35B-A3B expert shape `hidden=2048`, `intermediate=512`.
    The output is float32 so callers can accumulate routed and shared expert
    contributions before casting back to the model dtype.
    """

    torch, triton, tl = _load_triton()
    if hidden.ndim != 1:
        raise ValueError("hidden must have shape [hidden_size]")
    if expert_gate_up_weight.ndim != 2:
        raise ValueError("expert_gate_up_weight must have shape [2 * intermediate, hidden_size]")
    if expert_down_weight.ndim != 2:
        raise ValueError("expert_down_weight must have shape [hidden_size, intermediate]")
    if not hidden.is_cuda or not expert_gate_up_weight.is_cuda or not expert_down_weight.is_cuda:
        raise ValueError("hidden and expert weights must be CUDA tensors")

    hidden_size = hidden.shape[0]
    if expert_gate_up_weight.shape[1] != hidden_size or expert_down_weight.shape[0] != hidden_size:
        raise ValueError("expert weight hidden dimensions must match hidden")
    if expert_gate_up_weight.shape[0] % 2:
        raise ValueError("expert_gate_up_weight first dimension must be even")
    intermediate = expert_gate_up_weight.shape[0] // 2
    if expert_down_weight.shape[1] != intermediate:
        raise ValueError("expert_down_weight intermediate dimension must match gate/up weight")
    if hidden_size > 4096 or intermediate > 1024:
        raise ValueError("expert MLP kernel currently supports hidden_size<=4096 and intermediate<=1024")

    activation = torch.empty((intermediate,), device=hidden.device, dtype=torch.float32)
    out = torch.empty((hidden_size,), device=hidden.device, dtype=torch.float32)

    block_h = triton.next_power_of_2(hidden_size)
    block_i = triton.next_power_of_2(block_intermediate)
    down_block_h = triton.next_power_of_2(block_hidden)
    down_block_i = triton.next_power_of_2(intermediate)

    activation_kernel = _compile_expert_activation_kernel(triton, tl)
    activation_grid = (triton.cdiv(intermediate, block_intermediate),)
    activation_kernel[activation_grid](
        hidden,
        expert_gate_up_weight,
        activation,
        hidden_size,
        intermediate,
        block_h,
        block_i,
        num_warps=8,
    )

    down_kernel = _compile_expert_down_kernel(triton, tl)
    down_grid = (triton.cdiv(hidden_size, block_hidden),)
    down_kernel[down_grid](
        activation,
        expert_down_weight,
        out,
        hidden_size,
        intermediate,
        down_block_h,
        down_block_i,
        num_warps=8,
    )
    return out


def triton_qwen36_routed_experts_decode(
    hidden: Any,
    topk_ids: Any,
    topk_weights: Any,
    expert_gate_up_weight: Any,
    expert_down_weight: Any,
) -> Any:
    """Accumulate real-shape top-k routed experts without the shared expert."""

    torch, triton, tl = _load_triton()
    if hidden.ndim != 1:
        raise ValueError("hidden must have shape [hidden_size]")
    if topk_ids.ndim != 1 or topk_weights.ndim != 1:
        raise ValueError("topk_ids and topk_weights must have shape [top_k]")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have the same shape")
    if expert_gate_up_weight.ndim != 3:
        raise ValueError("expert_gate_up_weight must have shape [experts, 2 * intermediate, hidden_size]")
    if expert_down_weight.ndim != 3:
        raise ValueError("expert_down_weight must have shape [experts, hidden_size, intermediate]")
    if not hidden.is_cuda or not topk_ids.is_cuda or not topk_weights.is_cuda:
        raise ValueError("hidden, topk_ids, and topk_weights must be CUDA tensors")
    if not expert_gate_up_weight.is_cuda or not expert_down_weight.is_cuda:
        raise ValueError("expert weights must be CUDA tensors")

    hidden_size = hidden.shape[0]
    num_experts = expert_gate_up_weight.shape[0]
    if expert_down_weight.shape[0] != num_experts:
        raise ValueError("expert gate/up and down weights must have the same number of experts")
    if expert_gate_up_weight.shape[2] != hidden_size or expert_down_weight.shape[1] != hidden_size:
        raise ValueError("expert weight hidden dimensions must match hidden")
    if expert_gate_up_weight.shape[1] % 2:
        raise ValueError("expert_gate_up_weight second dimension must be even")
    intermediate = expert_gate_up_weight.shape[1] // 2
    if expert_down_weight.shape[2] != intermediate:
        raise ValueError("expert_down_weight intermediate dimension must match gate/up weight")
    if topk_ids.numel() > 8:
        raise ValueError("routed expert path currently supports top_k <= 8")

    top_k = topk_ids.numel()
    activation = torch.empty((top_k, intermediate), device=hidden.device, dtype=torch.float32)
    out = torch.empty((hidden_size,), device=hidden.device, dtype=torch.float32)
    block_h = triton.next_power_of_2(hidden_size)
    block_i = triton.next_power_of_2(16)
    down_block_h = triton.next_power_of_2(16)
    down_block_i = triton.next_power_of_2(intermediate)

    activation_kernel = _compile_routed_activation_kernel(triton, tl)
    activation_kernel[(top_k, triton.cdiv(intermediate, 16))](
        hidden,
        topk_ids,
        expert_gate_up_weight,
        activation,
        hidden_size,
        num_experts,
        intermediate,
        top_k,
        block_h,
        block_i,
        num_warps=8,
    )

    down_kernel = _compile_routed_down_kernel(triton, tl)
    down_kernel[(triton.cdiv(hidden_size, 16),)](
        activation,
        topk_ids,
        topk_weights,
        expert_down_weight,
        out,
        hidden_size,
        num_experts,
        intermediate,
        top_k,
        down_block_h,
        down_block_i,
        num_warps=8,
    )
    return out


def triton_qwen36_routed_shared_experts_decode(
    hidden: Any,
    topk_ids: Any,
    topk_weights: Any,
    expert_gate_up_weight: Any,
    expert_down_weight: Any,
    shared_gate_up_weight: Any | None = None,
    shared_down_weight: Any | None = None,
    shared_expert_gate_weight: Any | None = None,
) -> Any:
    """Accumulate real-shape top-k routed experts and the shared expert."""

    torch, triton, tl = _load_triton()
    if hidden.ndim != 1:
        raise ValueError("hidden must have shape [hidden_size]")
    if topk_ids.ndim != 1 or topk_weights.ndim != 1:
        raise ValueError("topk_ids and topk_weights must have shape [top_k]")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have the same shape")
    if expert_gate_up_weight.ndim != 3:
        raise ValueError("expert_gate_up_weight must have shape [experts, 2 * intermediate, hidden_size]")
    if expert_down_weight.ndim != 3:
        raise ValueError("expert_down_weight must have shape [experts, hidden_size, intermediate]")
    if not hidden.is_cuda or not topk_ids.is_cuda or not topk_weights.is_cuda:
        raise ValueError("hidden, topk_ids, and topk_weights must be CUDA tensors")
    if not expert_gate_up_weight.is_cuda or not expert_down_weight.is_cuda:
        raise ValueError("expert weights must be CUDA tensors")

    hidden_size = hidden.shape[0]
    num_experts = expert_gate_up_weight.shape[0]
    if expert_down_weight.shape[0] != num_experts:
        raise ValueError("expert gate/up and down weights must have the same number of experts")
    if expert_gate_up_weight.shape[2] != hidden_size or expert_down_weight.shape[1] != hidden_size:
        raise ValueError("expert weight hidden dimensions must match hidden")
    if expert_gate_up_weight.shape[1] % 2:
        raise ValueError("expert_gate_up_weight second dimension must be even")
    intermediate = expert_gate_up_weight.shape[1] // 2
    if expert_down_weight.shape[2] != intermediate:
        raise ValueError("expert_down_weight intermediate dimension must match gate/up weight")
    if topk_ids.numel() > 8:
        raise ValueError("routed/shared expert path currently supports top_k <= 8")
    if shared_gate_up_weight is None or shared_down_weight is None:
        raise ValueError("shared_gate_up_weight and shared_down_weight are required")
    if not shared_gate_up_weight.is_cuda or not shared_down_weight.is_cuda:
        raise ValueError("shared expert weights must be CUDA tensors")
    if shared_gate_up_weight.shape != expert_gate_up_weight.shape[1:]:
        raise ValueError("shared_gate_up_weight must have shape [2 * intermediate, hidden_size]")
    if shared_down_weight.shape != expert_down_weight.shape[1:]:
        raise ValueError("shared_down_weight must have shape [hidden_size, intermediate]")
    if shared_expert_gate_weight is not None:
        if not shared_expert_gate_weight.is_cuda:
            raise ValueError("shared_expert_gate_weight must be a CUDA tensor")
        if shared_expert_gate_weight.numel() != hidden_size:
            raise ValueError("shared_expert_gate_weight must have hidden_size elements")

    top_k = topk_ids.numel()
    activation = torch.empty((top_k + 1, intermediate), device=hidden.device, dtype=torch.float32)
    out = torch.empty((hidden_size,), device=hidden.device, dtype=torch.float32)
    block_h = triton.next_power_of_2(hidden_size)
    block_i = triton.next_power_of_2(16)
    down_block_h = triton.next_power_of_2(16)
    down_block_i = triton.next_power_of_2(intermediate)

    activation_kernel = _compile_routed_shared_activation_kernel(triton, tl)
    activation_kernel[(top_k + 1, triton.cdiv(intermediate, 16))](
        hidden,
        topk_ids,
        expert_gate_up_weight,
        shared_gate_up_weight,
        activation,
        hidden_size,
        num_experts,
        intermediate,
        top_k,
        block_h,
        block_i,
        num_warps=8,
    )

    if shared_expert_gate_weight is None:
        down_kernel = _compile_routed_shared_down_kernel(triton, tl)
        down_kernel[(triton.cdiv(hidden_size, 16),)](
            activation,
            topk_ids,
            topk_weights,
            expert_down_weight,
            shared_down_weight,
            out,
            hidden_size,
            num_experts,
            intermediate,
            top_k,
            down_block_h,
            down_block_i,
            num_warps=8,
        )
    else:
        shared_gate = torch.empty((1,), device=hidden.device, dtype=torch.float32)
        gate_kernel = _compile_shared_expert_gate_kernel(triton, tl)
        gate_kernel[(1,)](
            hidden,
            shared_expert_gate_weight,
            shared_gate,
            hidden_size,
            block_h,
            num_warps=8,
        )
        down_kernel = _compile_routed_shared_gated_down_kernel(triton, tl)
        down_kernel[(triton.cdiv(hidden_size, 16),)](
            activation,
            topk_ids,
            topk_weights,
            expert_down_weight,
            shared_down_weight,
            shared_gate,
            out,
            hidden_size,
            num_experts,
            intermediate,
            top_k,
            down_block_h,
            down_block_i,
            num_warps=8,
        )
    return out


def triton_qwen36_batched_routed_experts_decode(
    hidden: Any,
    topk_ids: Any,
    topk_weights: Any,
    expert_gate_up_weight: Any,
    expert_down_weight: Any,
) -> Any:
    """Accumulate real-shape routed experts for a batch of decode tokens.

    This is the matched operator boundary for vLLM's `fused_moe`: no shared
    expert path, no router, and no residual add.
    """

    torch, triton, tl = _load_triton()
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden_size]")
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("topk_ids and topk_weights must have shape [tokens, top_k]")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have the same shape")
    if topk_ids.shape[0] != hidden.shape[0]:
        raise ValueError("topk_ids/topk_weights token dimension must match hidden")
    if expert_gate_up_weight.ndim != 3:
        raise ValueError("expert_gate_up_weight must have shape [experts, 2 * intermediate, hidden_size]")
    if expert_down_weight.ndim != 3:
        raise ValueError("expert_down_weight must have shape [experts, hidden_size, intermediate]")
    tensors = (hidden, topk_ids, topk_weights, expert_gate_up_weight, expert_down_weight)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("hidden, routing tensors, and expert weights must be CUDA tensors")

    tokens, hidden_size = hidden.shape
    num_experts = expert_gate_up_weight.shape[0]
    if expert_down_weight.shape[0] != num_experts:
        raise ValueError("expert gate/up and down weights must have the same number of experts")
    if expert_gate_up_weight.shape[2] != hidden_size or expert_down_weight.shape[1] != hidden_size:
        raise ValueError("expert weight hidden dimensions must match hidden")
    if expert_gate_up_weight.shape[1] % 2:
        raise ValueError("expert_gate_up_weight second dimension must be even")
    intermediate = expert_gate_up_weight.shape[1] // 2
    if expert_down_weight.shape[2] != intermediate:
        raise ValueError("expert_down_weight intermediate dimension must match gate/up weight")
    if topk_ids.shape[1] > 8:
        raise ValueError("batched routed expert path currently supports top_k <= 8")

    top_k = topk_ids.shape[1]
    activation = torch.empty((tokens, top_k, intermediate), device=hidden.device, dtype=torch.float32)
    out = torch.empty((tokens, hidden_size), device=hidden.device, dtype=torch.float32)
    block_h = triton.next_power_of_2(hidden_size)
    block_i = triton.next_power_of_2(16)
    down_block_h = triton.next_power_of_2(16)
    down_block_i = triton.next_power_of_2(intermediate)

    activation_kernel = _compile_batched_routed_activation_kernel(triton, tl)
    activation_kernel[(tokens, top_k, triton.cdiv(intermediate, 16))](
        hidden,
        topk_ids,
        expert_gate_up_weight,
        activation,
        hidden_size,
        num_experts,
        intermediate,
        top_k,
        block_h,
        block_i,
        num_warps=8,
    )

    down_kernel = _compile_batched_routed_down_kernel(triton, tl)
    down_kernel[(tokens, triton.cdiv(hidden_size, 16))](
        activation,
        topk_ids,
        topk_weights,
        expert_down_weight,
        out,
        hidden_size,
        num_experts,
        intermediate,
        top_k,
        down_block_h,
        down_block_i,
        num_warps=8,
    )
    return out


def triton_qwen36_batched_routed_shared_experts_decode(
    hidden: Any,
    topk_ids: Any,
    topk_weights: Any,
    expert_gate_up_weight: Any,
    expert_down_weight: Any,
    shared_gate_up_weight: Any,
    shared_down_weight: Any,
    shared_expert_gate_weight: Any | None = None,
) -> Any:
    """Accumulate real-shape routed/shared experts for a batch of decode tokens."""

    torch, triton, tl = _load_triton()
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden_size]")
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("topk_ids and topk_weights must have shape [tokens, top_k]")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have the same shape")
    if topk_ids.shape[0] != hidden.shape[0]:
        raise ValueError("topk_ids/topk_weights token dimension must match hidden")
    if expert_gate_up_weight.ndim != 3:
        raise ValueError("expert_gate_up_weight must have shape [experts, 2 * intermediate, hidden_size]")
    if expert_down_weight.ndim != 3:
        raise ValueError("expert_down_weight must have shape [experts, hidden_size, intermediate]")
    tensors = (
        hidden,
        topk_ids,
        topk_weights,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("hidden, routing tensors, and expert weights must be CUDA tensors")
    if shared_expert_gate_weight is not None and not shared_expert_gate_weight.is_cuda:
        raise ValueError("shared_expert_gate_weight must be a CUDA tensor")

    tokens, hidden_size = hidden.shape
    num_experts = expert_gate_up_weight.shape[0]
    if expert_down_weight.shape[0] != num_experts:
        raise ValueError("expert gate/up and down weights must have the same number of experts")
    if expert_gate_up_weight.shape[2] != hidden_size or expert_down_weight.shape[1] != hidden_size:
        raise ValueError("expert weight hidden dimensions must match hidden")
    if expert_gate_up_weight.shape[1] % 2:
        raise ValueError("expert_gate_up_weight second dimension must be even")
    intermediate = expert_gate_up_weight.shape[1] // 2
    if expert_down_weight.shape[2] != intermediate:
        raise ValueError("expert_down_weight intermediate dimension must match gate/up weight")
    if topk_ids.shape[1] > 8:
        raise ValueError("batched routed/shared expert path currently supports top_k <= 8")
    if shared_gate_up_weight.shape != expert_gate_up_weight.shape[1:]:
        raise ValueError("shared_gate_up_weight must have shape [2 * intermediate, hidden_size]")
    if shared_down_weight.shape != expert_down_weight.shape[1:]:
        raise ValueError("shared_down_weight must have shape [hidden_size, intermediate]")
    if shared_expert_gate_weight is not None and shared_expert_gate_weight.numel() != hidden_size:
        raise ValueError("shared_expert_gate_weight must have hidden_size elements")

    top_k = topk_ids.shape[1]
    activation = torch.empty((tokens, top_k + 1, intermediate), device=hidden.device, dtype=torch.float32)
    out = torch.empty((tokens, hidden_size), device=hidden.device, dtype=torch.float32)
    block_h = triton.next_power_of_2(hidden_size)
    block_i = triton.next_power_of_2(16)
    down_block_h = triton.next_power_of_2(16)
    down_block_i = triton.next_power_of_2(intermediate)

    activation_kernel = _compile_batched_routed_shared_activation_kernel(triton, tl)
    activation_kernel[(tokens, top_k + 1, triton.cdiv(intermediate, 16))](
        hidden,
        topk_ids,
        expert_gate_up_weight,
        shared_gate_up_weight,
        activation,
        hidden_size,
        num_experts,
        intermediate,
        top_k,
        block_h,
        block_i,
        num_warps=8,
    )

    if shared_expert_gate_weight is None:
        down_kernel = _compile_batched_routed_shared_down_kernel(triton, tl)
        down_kernel[(tokens, triton.cdiv(hidden_size, 16))](
            activation,
            topk_ids,
            topk_weights,
            expert_down_weight,
            shared_down_weight,
            out,
            hidden_size,
            num_experts,
            intermediate,
            top_k,
            down_block_h,
            down_block_i,
            num_warps=8,
        )
    else:
        shared_gate = torch.empty((tokens,), device=hidden.device, dtype=torch.float32)
        gate_kernel = _compile_batched_shared_expert_gate_kernel(triton, tl)
        gate_kernel[(tokens,)](
            hidden,
            shared_expert_gate_weight,
            shared_gate,
            hidden_size,
            block_h,
            num_warps=8,
        )
        down_kernel = _compile_batched_routed_shared_gated_down_kernel(triton, tl)
        down_kernel[(tokens, triton.cdiv(hidden_size, 16))](
            activation,
            topk_ids,
            topk_weights,
            expert_down_weight,
            shared_down_weight,
            shared_gate,
            out,
            hidden_size,
            num_experts,
            intermediate,
            top_k,
            down_block_h,
            down_block_i,
            num_warps=8,
        )
    return out


def triton_qwen36_moe_decode(
    hidden: Any,
    norm_weight: Any,
    router_weight: Any,
    expert_gate_up_weight: Any,
    expert_down_weight: Any,
    shared_gate_up_weight: Any,
    shared_down_weight: Any,
    shared_expert_gate_weight: Any | None = None,
    *,
    top_k: int = 8,
    eps: float = 1e-6,
) -> tuple[Any, Any, Any, Any]:
    """Run the current real-shape Qwen3.6 MoE decode path.

    Returns `(output, router_logits, topk_ids, topk_weights)` so callers can
    check both hidden output parity and routing parity.
    """

    from .qwen36_router import triton_qwen36_moe_router_decode

    logits, topk_ids, topk_weights = triton_qwen36_moe_router_decode(
        hidden,
        norm_weight,
        router_weight,
        top_k=top_k,
        eps=eps,
    )
    output = triton_qwen36_routed_shared_experts_decode(
        hidden,
        topk_ids,
        topk_weights,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        shared_expert_gate_weight,
    )
    return output, logits, topk_ids, topk_weights


def triton_qwen36_batched_moe_decode(
    hidden: Any,
    norm_weight: Any,
    router_weight: Any,
    expert_gate_up_weight: Any,
    expert_down_weight: Any,
    shared_gate_up_weight: Any,
    shared_down_weight: Any,
    shared_expert_gate_weight: Any | None = None,
    *,
    top_k: int = 8,
    eps: float = 1e-6,
) -> tuple[Any, Any, Any, Any]:
    """Run the current real-shape Qwen3.6 MoE decode path for batched tokens."""

    from .qwen36_router import triton_qwen36_batched_moe_router_decode

    logits, topk_ids, topk_weights = triton_qwen36_batched_moe_router_decode(
        hidden,
        norm_weight,
        router_weight,
        top_k=top_k,
        eps=eps,
    )
    output = triton_qwen36_batched_routed_shared_experts_decode(
        hidden,
        topk_ids,
        topk_weights,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        shared_expert_gate_weight,
    )
    return output, logits, topk_ids, topk_weights

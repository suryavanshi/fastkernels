"""Layer-boundary Triton prototypes for Qwen3.6 decode."""

from __future__ import annotations

from typing import Any

from .qwen36_attention import triton_qwen36_batched_attention_decode, triton_synthetic_qwen36_attention_decode
from .qwen36_deltanet import (
    triton_qwen36_batched_deltanet_conv,
    triton_qwen36_batched_deltanet_project,
    triton_qwen36_batched_deltanet_recurrent_output,
    triton_synthetic_qwen36_deltanet_decode,
)
from .qwen36_expert import triton_qwen36_batched_moe_decode
from .qwen36_moe import triton_synthetic_qwen36_moe_decode

torch = None
triton = None
tl = None
_residual_add_kernel = None


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


def _compile_residual_add_kernel(triton: Any, tl: Any) -> Any:
    global _residual_add_kernel
    if _residual_add_kernel is not None:
        return _residual_add_kernel

    @triton.jit
    def _kernel(hidden, update, out, numel: tl.constexpr, block: tl.constexpr):
        program = tl.program_id(0)
        offsets = program * block + tl.arange(0, block)
        mask = offsets < numel
        hidden_values = tl.load(hidden + offsets, mask=mask, other=0.0).to(tl.float32)
        update_values = tl.load(update + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + offsets, hidden_values + update_values, mask=mask)

    _residual_add_kernel = _kernel
    return _residual_add_kernel


def triton_qwen36_batched_moe_layer_decode(
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
) -> tuple[Any, Any, Any, Any, Any]:
    """Run the real-shape Qwen3.6 MoE layer boundary for batched tokens.

    This wraps the current real-weight MoE block with the Transformer residual
    add, returning `(layer_hidden, moe_update, router_logits, topk_ids,
    topk_weights)`. It is intentionally still a staged path; DeltaNet/attention
    real weights are not part of this wrapper yet.
    """

    torch, triton, tl = _load_triton()
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden_size]")
    if not hidden.is_cuda:
        raise ValueError("hidden must be a CUDA tensor")

    moe_update, logits, topk_ids, topk_weights = triton_qwen36_batched_moe_decode(
        hidden,
        norm_weight,
        router_weight,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        shared_expert_gate_weight,
        top_k=top_k,
        eps=eps,
    )
    layer_hidden = torch.empty_like(moe_update)
    numel = hidden.numel()
    block = 1024
    kernel = _compile_residual_add_kernel(triton, tl)
    kernel[(triton.cdiv(numel, block),)](
        hidden,
        moe_update,
        layer_hidden,
        numel,
        block,
        num_warps=8,
    )
    return layer_hidden, moe_update, logits, topk_ids, topk_weights


def triton_qwen36_batched_attention_moe_layer_decode(
    hidden: Any,
    key_cache: Any,
    value_cache: Any,
    attention_norm_weight: Any,
    q_proj_weight: Any,
    k_proj_weight: Any,
    v_proj_weight: Any,
    out_proj_weight: Any,
    q_norm_weight: Any,
    k_norm_weight: Any,
    moe_norm_weight: Any,
    router_weight: Any,
    expert_gate_up_weight: Any,
    expert_down_weight: Any,
    shared_gate_up_weight: Any,
    shared_down_weight: Any,
    shared_expert_gate_weight: Any | None = None,
    *,
    start_position: int = 0,
    attention_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 256,
    rope_dim: int = 64,
    rope_theta: float = 10000.0,
    top_k: int = 8,
    eps: float = 1e-6,
    qk_norm_eps: float = 1e-6,
    copy_cache: bool = True,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Run a staged real-weight `Attention -> MoE` layer boundary.

    Returns `(layer_hidden, attention_hidden, attention_update, moe_update,
    key_cache, value_cache, router_logits, topk_ids, topk_weights)`.
    """

    torch, triton, tl = _load_triton()
    attention_update, next_key_cache, next_value_cache = triton_qwen36_batched_attention_decode(
        hidden,
        key_cache,
        value_cache,
        attention_norm_weight,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        out_proj_weight,
        q_norm_weight,
        k_norm_weight,
        start_position=start_position,
        attention_heads=attention_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        rope_theta=rope_theta,
        eps=eps,
        qk_norm_eps=qk_norm_eps,
        copy_cache=copy_cache,
    )
    attention_hidden = torch.empty_like(attention_update)
    numel = hidden.numel()
    block = 1024
    residual_kernel = _compile_residual_add_kernel(triton, tl)
    residual_kernel[(triton.cdiv(numel, block),)](
        hidden,
        attention_update,
        attention_hidden,
        numel,
        block,
        num_warps=8,
    )
    layer_hidden, moe_update, logits, topk_ids, topk_weights = triton_qwen36_batched_moe_layer_decode(
        attention_hidden,
        moe_norm_weight,
        router_weight,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        shared_expert_gate_weight,
        top_k=top_k,
        eps=eps,
    )
    return (
        layer_hidden,
        attention_hidden,
        attention_update,
        moe_update,
        next_key_cache,
        next_value_cache,
        logits,
        topk_ids,
        topk_weights,
    )


def triton_qwen36_batched_deltanet_moe_layer_decode(
    hidden: Any,
    conv_state: Any,
    recurrent_state: Any,
    deltanet_norm_weight: Any,
    in_proj_qkv_weight: Any,
    in_proj_z_weight: Any,
    in_proj_a_weight: Any,
    in_proj_b_weight: Any,
    conv1d_weight: Any,
    out_proj_weight: Any,
    linear_norm_weight: Any,
    a_log: Any,
    dt_bias: Any,
    moe_norm_weight: Any,
    router_weight: Any,
    expert_gate_up_weight: Any,
    expert_down_weight: Any,
    shared_gate_up_weight: Any,
    shared_down_weight: Any,
    shared_expert_gate_weight: Any | None = None,
    *,
    qk_heads: int = 16,
    value_heads: int = 32,
    head_dim: int = 128,
    top_k: int = 8,
    eps: float = 1e-6,
    block_value: int = 1024,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Run a staged real-weight `DeltaNet -> MoE` layer boundary.

    Returns `(layer_hidden, deltanet_hidden, deltanet_update, moe_update,
    conv_state, recurrent_state, router_logits, topk_ids, topk_weights)`.
    """

    torch, triton, tl = _load_triton()
    mixed_qkv, z, a_logits, b_logits = triton_qwen36_batched_deltanet_project(
        hidden,
        deltanet_norm_weight,
        in_proj_qkv_weight,
        in_proj_z_weight,
        in_proj_a_weight,
        in_proj_b_weight,
        eps=eps,
    )
    mixed_qkv, next_conv_state = triton_qwen36_batched_deltanet_conv(
        mixed_qkv,
        conv_state,
        conv1d_weight,
    )
    deltanet_update, next_recurrent_state = triton_qwen36_batched_deltanet_recurrent_output(
        mixed_qkv,
        z,
        a_logits,
        b_logits,
        recurrent_state,
        out_proj_weight,
        linear_norm_weight,
        a_log,
        dt_bias,
        qk_heads=qk_heads,
        value_heads=value_heads,
        head_dim=head_dim,
        eps=eps,
        block_value=block_value,
    )

    deltanet_hidden = torch.empty_like(deltanet_update)
    numel = hidden.numel()
    block = 1024
    residual_kernel = _compile_residual_add_kernel(triton, tl)
    residual_kernel[(triton.cdiv(numel, block),)](
        hidden,
        deltanet_update,
        deltanet_hidden,
        numel,
        block,
        num_warps=8,
    )
    layer_hidden, moe_update, logits, topk_ids, topk_weights = triton_qwen36_batched_moe_layer_decode(
        deltanet_hidden,
        moe_norm_weight,
        router_weight,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        shared_expert_gate_weight,
        top_k=top_k,
        eps=eps,
    )
    return (
        layer_hidden,
        deltanet_hidden,
        deltanet_update,
        moe_update,
        next_conv_state,
        next_recurrent_state,
        logits,
        topk_ids,
        topk_weights,
    )


def triton_synthetic_qwen36_deltanet_moe_decode(
    hidden: Any,
    state: Any,
    layer: dict[str, Any],
    *,
    qk_heads: int,
    head_dim: int,
    value_dim_per_head: int,
    top_k: int,
) -> tuple[Any, Any]:
    """Run a staged Triton synthetic `DeltaNet -> MoE` decode layer."""

    hidden, next_state = triton_synthetic_qwen36_deltanet_decode(
        hidden,
        state,
        layer["norm_weight"],
        layer["q_weight"],
        layer["k_weight"],
        layer["v_weight"],
        layer["gate_weight"],
        layer["out_weight"],
        qk_heads=qk_heads,
        head_dim=head_dim,
        value_dim_per_head=value_dim_per_head,
    )
    hidden = triton_synthetic_qwen36_moe_decode(
        hidden,
        layer["norm_weight"],
        layer["router_weight"],
        layer["expert_gate_up_weight"],
        layer["expert_down_weight"],
        layer["shared_gate_up_weight"],
        layer["shared_down_weight"],
        top_k=top_k,
    )
    return hidden, next_state


def triton_synthetic_qwen36_attention_moe_decode(
    hidden: Any,
    key_cache: Any,
    value_cache: Any,
    layer: dict[str, Any],
    *,
    position: int,
    attention_heads: int,
    kv_heads: int,
    head_dim: int,
    rope_dim: int,
    top_k: int,
    copy_cache: bool = True,
) -> tuple[Any, Any, Any]:
    """Run a staged Triton synthetic `Attention -> MoE` decode layer."""

    hidden, next_key_cache, next_value_cache = triton_synthetic_qwen36_attention_decode(
        hidden,
        key_cache,
        value_cache,
        layer["norm_weight"],
        layer["q_weight"],
        layer["k_weight"],
        layer["v_weight"],
        layer["out_weight"],
        position=position,
        attention_heads=attention_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        copy_cache=copy_cache,
    )
    hidden = triton_synthetic_qwen36_moe_decode(
        hidden,
        layer["norm_weight"],
        layer["router_weight"],
        layer["expert_gate_up_weight"],
        layer["expert_down_weight"],
        layer["shared_gate_up_weight"],
        layer["shared_down_weight"],
        top_k=top_k,
    )
    return hidden, next_key_cache, next_value_cache

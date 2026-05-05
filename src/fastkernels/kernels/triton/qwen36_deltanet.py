"""Prototype Triton decode kernels for synthetic Qwen3.6 DeltaNet blocks."""

from __future__ import annotations

from typing import Any

torch = None
triton = None
tl = None
_project_kernel = None
_state_kernel = None
_out_kernel = None
_fused_kernel = None
_real_batched_project_kernel = None
_real_batched_conv_kernel = None
_real_conv_state_kernel = None
_real_recurrent_kernel = None
_real_gated_norm_kernel = None
_real_out_partial_kernel = None
_real_out_reduce_kernel = None


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


def _compile_project_kernel(triton: Any, tl: Any) -> Any:
    global _project_kernel
    if _project_kernel is not None:
        return _project_kernel

    @triton.jit
    def _kernel(
        hidden,
        norm_weight,
        q_weight,
        k_weight,
        v_weight,
        gate_weight,
        q_out,
        k_out,
        v_out,
        gate_out,
        eps: tl.constexpr,
        hidden_size: tl.constexpr,
        q_width: tl.constexpr,
        value_width: tl.constexpr,
        block_h: tl.constexpr,
        block_q: tl.constexpr,
        block_v: tl.constexpr,
    ):
        hidden_offsets = tl.arange(0, block_h)
        q_offsets = tl.arange(0, block_q)
        value_offsets = tl.arange(0, block_v)
        hidden_mask = hidden_offsets < hidden_size
        q_mask = q_offsets < q_width
        value_mask = value_offsets < value_width

        hidden_values = tl.load(hidden + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        norm_values = tl.load(norm_weight + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(hidden_values * hidden_values, axis=0) / hidden_size
        x = hidden_values * tl.rsqrt(mean_square + eps) * norm_values

        q_weights = tl.load(
            q_weight + q_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=q_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        k_weights = tl.load(
            k_weight + q_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=q_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_values = tl.sum(q_weights * x[None, :], axis=1)
        k_values = tl.sum(k_weights * x[None, :], axis=1)
        tl.store(q_out + q_offsets, q_values, mask=q_mask)
        tl.store(k_out + q_offsets, k_values, mask=q_mask)

        v_weights = tl.load(
            v_weight + value_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=value_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_weights = tl.load(
            gate_weight + value_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=value_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_values = tl.sum(v_weights * x[None, :], axis=1)
        gate_logits = tl.sum(gate_weights * x[None, :], axis=1)
        gate_values = 1.0 / (1.0 + tl.exp(-gate_logits))
        tl.store(v_out + value_offsets, v_values, mask=value_mask)
        tl.store(gate_out + value_offsets, gate_values, mask=value_mask)

    _project_kernel = _kernel
    return _project_kernel


def _compile_state_kernel(triton: Any, tl: Any) -> Any:
    global _state_kernel
    if _state_kernel is not None:
        return _state_kernel

    @triton.jit
    def _kernel(
        state,
        q_values,
        k_values,
        v_values,
        gate_values,
        new_state,
        recurrent,
        qk_heads: tl.constexpr,
        head_dim: tl.constexpr,
        value_dim_per_head: tl.constexpr,
        block_s: tl.constexpr,
        block_v: tl.constexpr,
        block_d: tl.constexpr,
    ):
        state_offsets = tl.arange(0, block_s)
        total_state = qk_heads * head_dim * value_dim_per_head
        state_mask = state_offsets < total_state
        state_head = state_offsets // (head_dim * value_dim_per_head)
        state_rem = state_offsets - state_head * head_dim * value_dim_per_head
        state_dim = state_rem // value_dim_per_head
        state_value = state_rem - state_dim * value_dim_per_head

        k = tl.load(k_values + state_head * head_dim + state_dim, mask=state_mask, other=0.0).to(tl.float32)
        v = tl.load(v_values + state_head * value_dim_per_head + state_value, mask=state_mask, other=0.0).to(tl.float32)
        old = tl.load(state + state_offsets, mask=state_mask, other=0.0).to(tl.float32)
        updated = 0.95 * old + k * v
        tl.store(new_state + state_offsets, updated, mask=state_mask)

        value_offsets = tl.arange(0, block_v)
        dim_offsets = tl.arange(0, block_d)
        total_values = qk_heads * value_dim_per_head
        value_mask = value_offsets < total_values
        dim_mask = dim_offsets < head_dim
        value_head = value_offsets // value_dim_per_head
        value_col = value_offsets - value_head * value_dim_per_head

        matrix_state_offsets = (
            value_head[:, None] * head_dim * value_dim_per_head
            + dim_offsets[None, :] * value_dim_per_head
            + value_col[:, None]
        )
        q = tl.load(
            q_values + value_head[:, None] * head_dim + dim_offsets[None, :],
            mask=value_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        new = tl.load(
            new_state + matrix_state_offsets,
            mask=value_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(gate_values + value_offsets, mask=value_mask, other=0.0).to(tl.float32)
        recurrent_values = tl.sum(q * new, axis=1) * gate
        tl.store(recurrent + value_offsets, recurrent_values, mask=value_mask)

    _state_kernel = _kernel
    return _state_kernel


def _compile_out_kernel(triton: Any, tl: Any) -> Any:
    global _out_kernel
    if _out_kernel is not None:
        return _out_kernel

    @triton.jit
    def _kernel(
        hidden,
        recurrent,
        out_weight,
        out,
        hidden_size: tl.constexpr,
        value_width: tl.constexpr,
        block_h: tl.constexpr,
        block_v: tl.constexpr,
    ):
        hidden_offsets = tl.arange(0, block_h)
        value_offsets = tl.arange(0, block_v)
        hidden_mask = hidden_offsets < hidden_size
        value_mask = value_offsets < value_width

        recurrent_values = tl.load(recurrent + value_offsets, mask=value_mask, other=0.0).to(tl.float32)
        weights = tl.load(
            out_weight + hidden_offsets[:, None] * value_width + value_offsets[None, :],
            mask=hidden_mask[:, None] & value_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        layer_out = tl.sum(weights * recurrent_values[None, :], axis=1)
        residual = tl.load(hidden + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        tl.store(out + hidden_offsets, residual + layer_out, mask=hidden_mask)

    _out_kernel = _kernel
    return _out_kernel


def _compile_fused_kernel(triton: Any, tl: Any) -> Any:
    global _fused_kernel
    if _fused_kernel is not None:
        return _fused_kernel

    @triton.jit
    def _kernel(
        hidden,
        state,
        norm_weight,
        q_weight,
        k_weight,
        v_weight,
        gate_weight,
        out_weight,
        out,
        new_state,
        eps: tl.constexpr,
        hidden_size: tl.constexpr,
        qk_heads: tl.constexpr,
        head_dim: tl.constexpr,
        value_dim_per_head: tl.constexpr,
        block_h: tl.constexpr,
        block_s: tl.constexpr,
        block_v: tl.constexpr,
        block_d: tl.constexpr,
    ):
        hidden_offsets = tl.arange(0, block_h)
        state_offsets = tl.arange(0, block_s)
        value_offsets = tl.arange(0, block_v)
        dim_offsets = tl.arange(0, block_d)

        value_width = qk_heads * value_dim_per_head
        state_width = qk_heads * head_dim * value_dim_per_head
        hidden_mask = hidden_offsets < hidden_size
        state_mask = state_offsets < state_width
        value_mask = value_offsets < value_width
        dim_mask = dim_offsets < head_dim

        hidden_values = tl.load(hidden + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        norm_values = tl.load(norm_weight + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(hidden_values * hidden_values, axis=0) / hidden_size
        x = hidden_values * tl.rsqrt(mean_square + eps) * norm_values

        state_head = state_offsets // (head_dim * value_dim_per_head)
        state_rem = state_offsets - state_head * head_dim * value_dim_per_head
        state_dim = state_rem // value_dim_per_head
        state_value = state_rem - state_dim * value_dim_per_head

        store_k_weights = tl.load(
            k_weight + (state_head[:, None] * head_dim + state_dim[:, None]) * hidden_size + hidden_offsets[None, :],
            mask=state_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        store_v_weights = tl.load(
            v_weight + (state_head[:, None] * value_dim_per_head + state_value[:, None]) * hidden_size
            + hidden_offsets[None, :],
            mask=state_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        store_k = tl.sum(store_k_weights * x[None, :], axis=1)
        store_v = tl.sum(store_v_weights * x[None, :], axis=1)
        old_state = tl.load(state + state_offsets, mask=state_mask, other=0.0).to(tl.float32)
        updated_state = 0.95 * old_state + store_k * store_v
        tl.store(new_state + state_offsets, updated_state, mask=state_mask)

        value_head = value_offsets // value_dim_per_head
        value_col = value_offsets - value_head * value_dim_per_head

        v_weights = tl.load(
            v_weight + value_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=value_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_weights = tl.load(
            gate_weight + value_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=value_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_values = tl.sum(v_weights * x[None, :], axis=1)
        gate_logits = tl.sum(gate_weights * x[None, :], axis=1)
        gate_values = 1.0 / (1.0 + tl.exp(-gate_logits))

        recurrent_acc = tl.full((block_v,), 0.0, tl.float32)
        for dim in tl.static_range(0, head_dim):
            qk_row = value_head * head_dim + dim
            q_weights = tl.load(
                q_weight + qk_row[:, None] * hidden_size + hidden_offsets[None, :],
                mask=value_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            k_weights = tl.load(
                k_weight + qk_row[:, None] * hidden_size + hidden_offsets[None, :],
                mask=value_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            q_values = tl.sum(q_weights * x[None, :], axis=1)
            k_values = tl.sum(k_weights * x[None, :], axis=1)
            old_matrix = tl.load(
                state + value_head * head_dim * value_dim_per_head + dim * value_dim_per_head + value_col,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            updated_matrix = 0.95 * old_matrix + k_values * v_values
            recurrent_acc += q_values * updated_matrix
        recurrent = recurrent_acc * gate_values

        out_weights = tl.load(
            out_weight + hidden_offsets[:, None] * value_width + value_offsets[None, :],
            mask=hidden_mask[:, None] & value_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        layer_out = tl.sum(out_weights * recurrent[None, :], axis=1)
        tl.store(out + hidden_offsets, hidden_values + layer_out, mask=hidden_mask)

    _fused_kernel = _kernel
    return _fused_kernel


def _compile_real_batched_project_kernel(triton: Any, tl: Any) -> Any:
    global _real_batched_project_kernel
    if _real_batched_project_kernel is not None:
        return _real_batched_project_kernel

    @triton.jit
    def _kernel(
        hidden,
        norm_weight,
        weight,
        out,
        eps: tl.constexpr,
        norm_offset: tl.constexpr,
        hidden_size: tl.constexpr,
        out_width: tl.constexpr,
        block_h: tl.constexpr,
    ):
        token = tl.program_id(0)
        row = tl.program_id(1)
        offsets = tl.arange(0, block_h)
        mask = offsets < hidden_size

        hidden_values = tl.load(hidden + token * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        norm_values = tl.load(norm_weight + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(hidden_values * hidden_values, axis=0) / hidden_size
        x = hidden_values * tl.rsqrt(mean_square + eps) * (norm_offset + norm_values)
        weights = tl.load(weight + row * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + token * out_width + row, tl.sum(weights * x, axis=0))

    _real_batched_project_kernel = _kernel
    return _real_batched_project_kernel


def _triton_qwen36_batched_deltanet_linear(
    hidden: Any,
    norm_weight: Any,
    weight: Any,
    *,
    eps: float,
    norm_offset: float,
    block_hidden: int,
) -> Any:
    torch, triton, tl = _load_triton()
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden_size]")
    if norm_weight.ndim != 1 or norm_weight.shape[0] != hidden.shape[1]:
        raise ValueError("norm_weight must have shape [hidden_size]")
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[1]:
        raise ValueError("weight must have shape [out_width, hidden_size]")
    if not hidden.is_cuda or not norm_weight.is_cuda or not weight.is_cuda:
        raise ValueError("hidden, norm_weight, and weight must be CUDA tensors")

    tokens, hidden_size = hidden.shape
    out_width = weight.shape[0]
    out = torch.empty((tokens, out_width), device=hidden.device, dtype=torch.float32)
    block_h = triton.next_power_of_2(block_hidden)
    if block_h < hidden_size:
        raise ValueError("block_hidden must be at least hidden_size")
    kernel = _compile_real_batched_project_kernel(triton, tl)
    kernel[(tokens, out_width)](
        hidden,
        norm_weight,
        weight,
        out,
        eps,
        norm_offset,
        hidden_size,
        out_width,
        block_h,
        num_warps=8,
    )
    return out


def triton_qwen36_batched_deltanet_project(
    hidden: Any,
    norm_weight: Any,
    in_proj_qkv_weight: Any,
    in_proj_z_weight: Any,
    in_proj_a_weight: Any,
    in_proj_b_weight: Any,
    *,
    eps: float = 1e-6,
    norm_offset: float = 1.0,
    block_hidden: int = 2048,
) -> tuple[Any, Any, Any, Any]:
    """Project real Qwen3.6 DeltaNet tensors for batched decode rows.

    This is a correctness-first real-weight staging boundary. It covers the
    input RMSNorm plus `in_proj_qkv`, `in_proj_z`, `in_proj_a`, and `in_proj_b`.
    Pair it with `triton_qwen36_batched_deltanet_conv` for the causal
    depthwise convolution stage and
    `triton_qwen36_batched_deltanet_recurrent_output` for the recurrent/output
    stage.
    """

    mixed_qkv = _triton_qwen36_batched_deltanet_linear(
        hidden,
        norm_weight,
        in_proj_qkv_weight,
        eps=eps,
        norm_offset=norm_offset,
        block_hidden=block_hidden,
    )
    z = _triton_qwen36_batched_deltanet_linear(
        hidden,
        norm_weight,
        in_proj_z_weight,
        eps=eps,
        norm_offset=norm_offset,
        block_hidden=block_hidden,
    )
    a_logits = _triton_qwen36_batched_deltanet_linear(
        hidden,
        norm_weight,
        in_proj_a_weight,
        eps=eps,
        norm_offset=norm_offset,
        block_hidden=block_hidden,
    )
    b_logits = _triton_qwen36_batched_deltanet_linear(
        hidden,
        norm_weight,
        in_proj_b_weight,
        eps=eps,
        norm_offset=norm_offset,
        block_hidden=block_hidden,
    )
    return mixed_qkv, z, a_logits, b_logits


def _compile_real_batched_conv_kernel(triton: Any, tl: Any) -> Any:
    global _real_batched_conv_kernel
    if _real_batched_conv_kernel is not None:
        return _real_batched_conv_kernel

    @triton.jit
    def _kernel(
        mixed,
        conv_state,
        conv_weight,
        conv_out,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        kernel_size: tl.constexpr,
    ):
        token = tl.program_id(0)
        channel = tl.program_id(1)
        acc = tl.full((), 0.0, tl.float32)
        start = token + 1
        for idx in tl.static_range(0, kernel_size):
            position = start + idx
            value = tl.load(
                conv_state + channel * kernel_size + position,
                mask=position < kernel_size,
                other=0.0,
            ).to(tl.float32)
            mixed_index = position - kernel_size
            mixed_value = tl.load(
                mixed + mixed_index * channels + channel,
                mask=position >= kernel_size,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(conv_weight + channel * kernel_size + idx).to(tl.float32)
            acc += tl.where(position < kernel_size, value, mixed_value) * weight
        tl.store(conv_out + token * channels + channel, acc / (1.0 + tl.exp(-acc)))

    _real_batched_conv_kernel = _kernel
    return _real_batched_conv_kernel


def _compile_real_conv_state_kernel(triton: Any, tl: Any) -> Any:
    global _real_conv_state_kernel
    if _real_conv_state_kernel is not None:
        return _real_conv_state_kernel

    @triton.jit
    def _kernel(
        mixed,
        conv_state,
        next_state,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        kernel_size: tl.constexpr,
    ):
        channel = tl.program_id(0)
        slot = tl.program_id(1)
        position = tokens + slot
        value = tl.load(
            conv_state + channel * kernel_size + position,
            mask=position < kernel_size,
            other=0.0,
        ).to(tl.float32)
        mixed_index = position - kernel_size
        mixed_value = tl.load(
            mixed + mixed_index * channels + channel,
            mask=position >= kernel_size,
            other=0.0,
        ).to(tl.float32)
        tl.store(next_state + channel * kernel_size + slot, tl.where(position < kernel_size, value, mixed_value))

    _real_conv_state_kernel = _kernel
    return _real_conv_state_kernel


def triton_qwen36_batched_deltanet_conv(
    mixed_qkv: Any,
    conv_state: Any,
    conv_weight: Any,
) -> tuple[Any, Any]:
    """Apply real Qwen3.6 DeltaNet causal depthwise conv for batched rows."""

    torch, triton, tl = _load_triton()
    if mixed_qkv.ndim != 2:
        raise ValueError("mixed_qkv must have shape [tokens, channels]")
    if conv_state.ndim != 2:
        raise ValueError("conv_state must have shape [channels, kernel_size]")
    if conv_weight.ndim != 3:
        raise ValueError("conv_weight must have shape [channels, 1, kernel_size]")
    tokens, channels = mixed_qkv.shape
    if conv_state.shape[0] != channels:
        raise ValueError("mixed_qkv and conv_state disagree on channel count")
    if conv_weight.shape[0] != channels or conv_weight.shape[1] != 1 or conv_weight.shape[2] != conv_state.shape[1]:
        raise ValueError("conv_weight must have shape [channels, 1, kernel_size]")
    if not mixed_qkv.is_cuda or not conv_state.is_cuda or not conv_weight.is_cuda:
        raise ValueError("mixed_qkv, conv_state, and conv_weight must be CUDA tensors")
    if tokens < 1:
        raise ValueError("tokens must be positive")

    kernel_size = conv_state.shape[1]
    conv_out = torch.empty_like(mixed_qkv, dtype=torch.float32)
    next_state = torch.empty_like(conv_state)

    conv_kernel = _compile_real_batched_conv_kernel(triton, tl)
    conv_kernel[(tokens, channels)](
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_out,
        tokens,
        channels,
        kernel_size,
        num_warps=1,
    )

    state_kernel = _compile_real_conv_state_kernel(triton, tl)
    state_kernel[(channels, kernel_size)](
        mixed_qkv,
        conv_state,
        next_state,
        tokens,
        channels,
        kernel_size,
        num_warps=1,
    )
    return conv_out, next_state


def _compile_real_recurrent_kernel(triton: Any, tl: Any) -> Any:
    global _real_recurrent_kernel
    if _real_recurrent_kernel is not None:
        return _real_recurrent_kernel

    @triton.jit
    def _kernel(
        mixed_qkv,
        a_logits,
        b_logits,
        recurrent_state,
        a_log,
        dt_bias,
        core,
        next_state,
        tokens: tl.constexpr,
        qk_heads: tl.constexpr,
        value_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_cols: tl.constexpr,
        eps: tl.constexpr,
    ):
        head = tl.program_id(0)
        col_block = tl.program_id(1)
        rows = tl.arange(0, head_dim)
        cols = col_block * block_cols + tl.arange(0, block_cols)
        col_mask = cols < head_dim

        qk_head = head // (value_heads // qk_heads)
        qk_width = qk_heads * head_dim
        value_width = value_heads * head_dim
        mixed_width = 2 * qk_width + value_width
        state_offsets = head * head_dim * head_dim + rows[:, None] * head_dim + cols[None, :]
        state_mask = (rows[:, None] < head_dim) & col_mask[None, :]
        state_values = tl.load(
            recurrent_state + state_offsets,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)

        for token in tl.static_range(0, tokens):
            q_offsets = token * mixed_width + qk_head * head_dim + rows
            k_offsets = token * mixed_width + qk_width + qk_head * head_dim + rows
            q_raw = tl.load(mixed_qkv + q_offsets).to(tl.float32)
            k_raw = tl.load(mixed_qkv + k_offsets).to(tl.float32)
            q_scale = tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) + eps) * (head_dim**-0.5)
            k_scale = tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) + eps)
            q = q_raw * q_scale
            k = k_raw * k_scale

            gate_input = tl.load(a_logits + token * value_heads + head).to(tl.float32) + tl.load(dt_bias + head).to(tl.float32)
            softplus = tl.maximum(gate_input, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(gate_input)))
            decay = tl.exp(-tl.exp(tl.load(a_log + head).to(tl.float32)) * softplus)
            beta_input = tl.load(b_logits + token * value_heads + head).to(tl.float32)
            beta = 1.0 / (1.0 + tl.exp(-beta_input))

            state_values = state_values * decay
            projected = tl.sum(state_values * k[:, None], axis=0)
            value = tl.load(
                mixed_qkv + token * mixed_width + 2 * qk_width + head * head_dim + cols,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            delta = (value - projected) * beta
            state_values = state_values + k[:, None] * delta[None, :]
            out = tl.sum(state_values * q[:, None], axis=0)
            tl.store(core + token * value_width + head * head_dim + cols, out, mask=col_mask)

        tl.store(next_state + state_offsets, state_values, mask=state_mask)

    _real_recurrent_kernel = _kernel
    return _real_recurrent_kernel


def _compile_real_gated_norm_kernel(triton: Any, tl: Any) -> Any:
    global _real_gated_norm_kernel
    if _real_gated_norm_kernel is not None:
        return _real_gated_norm_kernel

    @triton.jit
    def _kernel(
        core,
        z,
        linear_norm_weight,
        normed,
        value_heads: tl.constexpr,
        head_dim: tl.constexpr,
        eps: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, head_dim)
        value_width = value_heads * head_dim
        base = token * value_width + head * head_dim + offsets
        values = tl.load(core + base).to(tl.float32)
        gates = tl.load(z + base).to(tl.float32)
        weights = tl.load(linear_norm_weight + offsets).to(tl.float32)
        rms = tl.rsqrt(tl.sum(values * values, axis=0) / head_dim + eps)
        silu_gates = gates / (1.0 + tl.exp(-gates))
        tl.store(normed + base, values * rms * weights * silu_gates)

    _real_gated_norm_kernel = _kernel
    return _real_gated_norm_kernel


def _compile_real_out_partial_kernel(triton: Any, tl: Any) -> Any:
    global _real_out_partial_kernel
    if _real_out_partial_kernel is not None:
        return _real_out_partial_kernel

    @triton.jit
    def _kernel(
        normed,
        out_proj_weight,
        partials,
        hidden_size: tl.constexpr,
        value_width: tl.constexpr,
        chunks: tl.constexpr,
        block_value: tl.constexpr,
    ):
        token = tl.program_id(0)
        row = tl.program_id(1)
        chunk = tl.program_id(2)
        offsets = chunk * block_value + tl.arange(0, block_value)
        mask = offsets < value_width
        values = tl.load(normed + token * value_width + offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(out_proj_weight + row * value_width + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(partials + (token * hidden_size + row) * chunks + chunk, tl.sum(values * weights, axis=0))

    _real_out_partial_kernel = _kernel
    return _real_out_partial_kernel


def _compile_real_out_reduce_kernel(triton: Any, tl: Any) -> Any:
    global _real_out_reduce_kernel
    if _real_out_reduce_kernel is not None:
        return _real_out_reduce_kernel

    @triton.jit
    def _kernel(
        partials,
        out,
        hidden_size: tl.constexpr,
        chunks: tl.constexpr,
        block_chunks: tl.constexpr,
    ):
        token = tl.program_id(0)
        row = tl.program_id(1)
        offsets = tl.arange(0, block_chunks)
        mask = offsets < chunks
        values = tl.load(partials + (token * hidden_size + row) * chunks + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + token * hidden_size + row, tl.sum(values, axis=0))

    _real_out_reduce_kernel = _kernel
    return _real_out_reduce_kernel


def triton_qwen36_batched_deltanet_recurrent_output(
    mixed_qkv: Any,
    z: Any,
    a_logits: Any,
    b_logits: Any,
    recurrent_state: Any,
    out_proj_weight: Any,
    linear_norm_weight: Any,
    a_log: Any,
    dt_bias: Any,
    *,
    qk_heads: int = 16,
    value_heads: int = 32,
    head_dim: int = 128,
    eps: float = 1e-6,
    block_cols: int = 8,
    block_value: int = 1024,
) -> tuple[Any, Any]:
    """Run real Qwen3.6 DeltaNet recurrent rule, gated norm, and output projection."""

    torch, triton, tl = _load_triton()
    if mixed_qkv.ndim != 2:
        raise ValueError("mixed_qkv must have shape [tokens, 2*qk_width + value_width]")
    if z.ndim != 2 or a_logits.ndim != 2 or b_logits.ndim != 2:
        raise ValueError("z, a_logits, and b_logits must be batched rank-2 tensors")
    tokens, mixed_width = mixed_qkv.shape
    qk_width = qk_heads * head_dim
    value_width = value_heads * head_dim
    if mixed_width != 2 * qk_width + value_width:
        raise ValueError("mixed_qkv shape does not match qk/value head dimensions")
    if z.shape != (tokens, value_width):
        raise ValueError("z must have shape [tokens, value_heads * head_dim]")
    if a_logits.shape != (tokens, value_heads) or b_logits.shape != (tokens, value_heads):
        raise ValueError("a_logits and b_logits must have shape [tokens, value_heads]")
    if recurrent_state.shape != (value_heads, head_dim, head_dim):
        raise ValueError("recurrent_state must have shape [value_heads, head_dim, head_dim]")
    if out_proj_weight.ndim != 2 or out_proj_weight.shape[1] != value_width:
        raise ValueError("out_proj_weight must have shape [hidden_size, value_heads * head_dim]")
    if linear_norm_weight.shape != (head_dim,):
        raise ValueError("linear_norm_weight must have shape [head_dim]")
    if a_log.shape != (value_heads,) or dt_bias.shape != (value_heads,):
        raise ValueError("a_log and dt_bias must have shape [value_heads]")
    tensors = (mixed_qkv, z, a_logits, b_logits, recurrent_state, out_proj_weight, linear_norm_weight, a_log, dt_bias)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("all DeltaNet recurrent/output inputs must be CUDA tensors")
    if tokens < 1:
        raise ValueError("tokens must be positive")
    if value_heads % qk_heads:
        raise ValueError("value_heads must be divisible by qk_heads")

    block_cols = triton.next_power_of_2(block_cols)
    if block_cols > head_dim:
        block_cols = triton.next_power_of_2(head_dim)
    if head_dim % block_cols:
        raise ValueError("block_cols must divide head_dim")
    block_value = triton.next_power_of_2(block_value)
    if value_width % block_value:
        raise ValueError("block_value must divide value width")

    hidden_size = out_proj_weight.shape[0]
    chunks = triton.cdiv(value_width, block_value)
    core = torch.empty((tokens, value_width), device=mixed_qkv.device, dtype=torch.float32)
    normed = torch.empty_like(core)
    partials = torch.empty((tokens, hidden_size, chunks), device=mixed_qkv.device, dtype=torch.float32)
    out = torch.empty((tokens, hidden_size), device=mixed_qkv.device, dtype=torch.float32)
    next_state = torch.empty_like(recurrent_state)

    recurrent_kernel = _compile_real_recurrent_kernel(triton, tl)
    recurrent_kernel[(value_heads, triton.cdiv(head_dim, block_cols))](
        mixed_qkv,
        a_logits,
        b_logits,
        recurrent_state,
        a_log,
        dt_bias,
        core,
        next_state,
        tokens,
        qk_heads,
        value_heads,
        head_dim,
        block_cols,
        eps,
        num_warps=4,
    )

    norm_kernel = _compile_real_gated_norm_kernel(triton, tl)
    norm_kernel[(tokens, value_heads)](
        core,
        z,
        linear_norm_weight,
        normed,
        value_heads,
        head_dim,
        eps,
        num_warps=4,
    )

    partial_kernel = _compile_real_out_partial_kernel(triton, tl)
    partial_kernel[(tokens, hidden_size, chunks)](
        normed,
        out_proj_weight,
        partials,
        hidden_size,
        value_width,
        chunks,
        block_value,
        num_warps=4,
    )

    reduce_kernel = _compile_real_out_reduce_kernel(triton, tl)
    reduce_kernel[(tokens, hidden_size)](
        partials,
        out,
        hidden_size,
        chunks,
        triton.next_power_of_2(chunks),
        num_warps=1,
    )
    return out, next_state


def _triton_synthetic_qwen36_deltanet_decode_staged(
    hidden: Any,
    state: Any,
    norm_weight: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    gate_weight: Any,
    out_weight: Any,
    *,
    qk_heads: int,
    head_dim: int,
    value_dim_per_head: int,
    eps: float = 1e-6,
) -> tuple[Any, Any]:
    """Run a prototype one-token synthetic Qwen3.6 DeltaNet decode block."""

    torch, triton, tl = _load_triton()
    if hidden.ndim != 1:
        raise ValueError("hidden must have shape [hidden_size]")
    if state.ndim != 3:
        raise ValueError("state must have shape [qk_heads, head_dim, value_dim_per_head]")
    if not hidden.is_cuda or not state.is_cuda:
        raise ValueError("hidden and state must be CUDA tensors")
    if state.shape != (qk_heads, head_dim, value_dim_per_head):
        raise ValueError("state shape does not match qk_heads/head_dim/value_dim_per_head")

    hidden_size = hidden.shape[0]
    q_width = qk_heads * head_dim
    value_width = qk_heads * value_dim_per_head
    state_width = qk_heads * head_dim * value_dim_per_head
    if hidden_size > 128 or q_width > 128 or value_width > 128 or state_width > 256:
        raise ValueError("prototype synthetic Qwen3.6 DeltaNet kernel only supports tiny synthetic shapes")

    q_values = torch.empty((q_width,), device=hidden.device, dtype=hidden.dtype)
    k_values = torch.empty_like(q_values)
    v_values = torch.empty((value_width,), device=hidden.device, dtype=hidden.dtype)
    gate_values = torch.empty_like(v_values)
    new_state = torch.empty_like(state)
    recurrent = torch.empty_like(v_values)
    out = torch.empty_like(hidden)

    block_h = triton.next_power_of_2(hidden_size)
    block_q = triton.next_power_of_2(q_width)
    block_v = triton.next_power_of_2(value_width)
    block_s = triton.next_power_of_2(state_width)
    block_d = triton.next_power_of_2(head_dim)

    project_kernel = _compile_project_kernel(triton, tl)
    project_kernel[(1,)](
        hidden,
        norm_weight,
        q_weight,
        k_weight,
        v_weight,
        gate_weight,
        q_values,
        k_values,
        v_values,
        gate_values,
        eps,
        hidden_size,
        q_width,
        value_width,
        block_h,
        block_q,
        block_v,
    )

    state_kernel = _compile_state_kernel(triton, tl)
    state_kernel[(1,)](
        state,
        q_values,
        k_values,
        v_values,
        gate_values,
        new_state,
        recurrent,
        qk_heads,
        head_dim,
        value_dim_per_head,
        block_s,
        block_v,
        block_d,
    )

    out_kernel = _compile_out_kernel(triton, tl)
    out_kernel[(1,)](
        hidden,
        recurrent,
        out_weight,
        out,
        hidden_size,
        value_width,
        block_h,
        block_v,
    )
    return out, new_state


def triton_synthetic_qwen36_deltanet_decode(
    hidden: Any,
    state: Any,
    norm_weight: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    gate_weight: Any,
    out_weight: Any,
    *,
    qk_heads: int,
    head_dim: int,
    value_dim_per_head: int,
    eps: float = 1e-6,
) -> tuple[Any, Any]:
    """Run a fused one-token synthetic Qwen3.6 DeltaNet decode block."""

    torch, triton, tl = _load_triton()
    if hidden.ndim != 1:
        raise ValueError("hidden must have shape [hidden_size]")
    if state.ndim != 3:
        raise ValueError("state must have shape [qk_heads, head_dim, value_dim_per_head]")
    if not hidden.is_cuda or not state.is_cuda:
        raise ValueError("hidden and state must be CUDA tensors")
    if state.shape != (qk_heads, head_dim, value_dim_per_head):
        raise ValueError("state shape does not match qk_heads/head_dim/value_dim_per_head")

    hidden_size = hidden.shape[0]
    q_width = qk_heads * head_dim
    value_width = qk_heads * value_dim_per_head
    state_width = qk_heads * head_dim * value_dim_per_head
    if hidden_size > 32 or q_width > 32 or value_width > 32 or state_width > 128:
        return _triton_synthetic_qwen36_deltanet_decode_staged(
            hidden,
            state,
            norm_weight,
            q_weight,
            k_weight,
            v_weight,
            gate_weight,
            out_weight,
            qk_heads=qk_heads,
            head_dim=head_dim,
            value_dim_per_head=value_dim_per_head,
            eps=eps,
        )

    out = torch.empty_like(hidden)
    new_state = torch.empty_like(state)
    block_h = triton.next_power_of_2(hidden_size)
    block_s = triton.next_power_of_2(state_width)
    block_v = triton.next_power_of_2(value_width)
    block_d = triton.next_power_of_2(head_dim)
    kernel = _compile_fused_kernel(triton, tl)
    kernel[(1,)](
        hidden,
        state,
        norm_weight,
        q_weight,
        k_weight,
        v_weight,
        gate_weight,
        out_weight,
        out,
        new_state,
        eps,
        hidden_size,
        qk_heads,
        head_dim,
        value_dim_per_head,
        block_h,
        block_s,
        block_v,
        block_d,
    )
    return out, new_state

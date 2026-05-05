"""Prototype Triton decode kernels for synthetic Qwen3.6 attention blocks."""

from __future__ import annotations

from typing import Any

torch = None
triton = None
tl = None
_project_kernel = None
_project_rope_cache_kernel = None
_rope_cache_kernel = None
_attention_kernel = None
_out_kernel = None
_real_batched_project_kernel = None
_real_head_rmsnorm_kernel = None
_real_batched_rope_cache_kernel = None
_real_batched_attention_kernel = None
_real_batched_out_kernel = None


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
        q_raw,
        k_raw,
        v_raw,
        eps: tl.constexpr,
        hidden_size: tl.constexpr,
        q_width: tl.constexpr,
        kv_width: tl.constexpr,
        block_h: tl.constexpr,
        block_q: tl.constexpr,
        block_kv: tl.constexpr,
    ):
        hidden_offsets = tl.arange(0, block_h)
        q_offsets = tl.arange(0, block_q)
        kv_offsets = tl.arange(0, block_kv)
        hidden_mask = hidden_offsets < hidden_size
        q_mask = q_offsets < q_width
        kv_mask = kv_offsets < kv_width

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
            k_weight + kv_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=kv_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_weights = tl.load(
            v_weight + kv_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=kv_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        tl.store(q_raw + q_offsets, tl.sum(q_weights * x[None, :], axis=1), mask=q_mask)
        tl.store(k_raw + kv_offsets, tl.sum(k_weights * x[None, :], axis=1), mask=kv_mask)
        tl.store(v_raw + kv_offsets, tl.sum(v_weights * x[None, :], axis=1), mask=kv_mask)

    _project_kernel = _kernel
    return _project_kernel


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


def _triton_qwen36_batched_attention_linear(
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


def triton_qwen36_batched_attention_project(
    hidden: Any,
    norm_weight: Any,
    q_proj_weight: Any,
    k_proj_weight: Any,
    v_proj_weight: Any,
    *,
    eps: float = 1e-6,
    norm_offset: float = 1.0,
    block_hidden: int = 2048,
) -> tuple[Any, Any, Any]:
    """Project real Qwen3.6 attention Q/K/V tensors for batched decode rows.

    This is a real-weight staging kernel boundary: it intentionally covers the
    RMSNorm + Q/K/V linear projections only. The real Qwen3.6 attention contract
    has a wider Q projection than the synthetic attention prototype, so RoPE,
    Q/K norm, cache update, and attention accumulation remain separate work.
    """

    q = _triton_qwen36_batched_attention_linear(
        hidden,
        norm_weight,
        q_proj_weight,
        eps=eps,
        norm_offset=norm_offset,
        block_hidden=block_hidden,
    )
    k = _triton_qwen36_batched_attention_linear(
        hidden,
        norm_weight,
        k_proj_weight,
        eps=eps,
        norm_offset=norm_offset,
        block_hidden=block_hidden,
    )
    v = _triton_qwen36_batched_attention_linear(
        hidden,
        norm_weight,
        v_proj_weight,
        eps=eps,
        norm_offset=norm_offset,
        block_hidden=block_hidden,
    )
    return q, k, v


def _compile_real_head_rmsnorm_kernel(triton: Any, tl: Any) -> Any:
    global _real_head_rmsnorm_kernel
    if _real_head_rmsnorm_kernel is not None:
        return _real_head_rmsnorm_kernel

    @triton.jit
    def _kernel(
        values,
        norm_weight,
        out,
        eps: tl.constexpr,
        norm_offset: tl.constexpr,
        width: tl.constexpr,
        head_dim: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, block_d)
        mask = offsets < head_dim
        base = token * width + head * head_dim
        x = tl.load(values + base + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(norm_weight + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / head_dim
        tl.store(out + base + offsets, x * tl.rsqrt(mean_square + eps) * (norm_offset + weight), mask=mask)

    _real_head_rmsnorm_kernel = _kernel
    return _real_head_rmsnorm_kernel


def _compile_real_batched_rope_cache_kernel(triton: Any, tl: Any) -> Any:
    global _real_batched_rope_cache_kernel
    if _real_batched_rope_cache_kernel is not None:
        return _real_batched_rope_cache_kernel

    @triton.jit
    def _kernel(
        q_normed,
        k_normed,
        v_raw,
        q_values,
        key_cache,
        value_cache,
        start_position: tl.constexpr,
        q_width: tl.constexpr,
        kv_width: tl.constexpr,
        attention_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rope_dim: tl.constexpr,
        rope_theta: tl.constexpr,
        max_positions: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, block_d)
        mask = offsets < head_dim
        rotate_dim = tl.minimum(rope_dim, head_dim)
        rotate_dim = rotate_dim - rotate_dim % 2
        position = start_position + token

        pair_dim = tl.where(offsets % 2 == 0, offsets + 1, offsets - 1)
        angle_dim = (offsets // 2) * 2
        angle = position / tl.exp(tl.log(rope_theta) * (angle_dim / rotate_dim))
        cos = tl.cos(angle)
        sin = tl.sin(angle)

        q_base = token * q_width + head * head_dim
        q_raw = tl.load(q_normed + q_base + offsets, mask=mask, other=0.0).to(tl.float32)
        q_pair = tl.load(q_normed + q_base + pair_dim, mask=mask & (offsets < rotate_dim), other=0.0).to(
            tl.float32
        )
        q_rotated = tl.where(
            offsets % 2 == 0,
            q_raw * cos - q_pair * sin,
            q_pair * sin + q_raw * cos,
        )
        q_out = tl.where(offsets < rotate_dim, q_rotated, q_raw)
        tl.store(q_values + q_base + offsets, q_out, mask=mask)

        kv_head_mask = head < kv_heads
        kv_base = token * kv_width + head * head_dim
        k_raw = tl.load(k_normed + kv_base + offsets, mask=mask & kv_head_mask, other=0.0).to(tl.float32)
        k_pair = tl.load(
            k_normed + kv_base + pair_dim,
            mask=mask & kv_head_mask & (offsets < rotate_dim),
            other=0.0,
        ).to(tl.float32)
        k_rotated = tl.where(
            offsets % 2 == 0,
            k_raw * cos - k_pair * sin,
            k_pair * sin + k_raw * cos,
        )
        k_out = tl.where(offsets < rotate_dim, k_rotated, k_raw)
        cache_base = position * kv_width + head * head_dim
        tl.store(key_cache + cache_base + offsets, k_out, mask=mask & kv_head_mask & (position < max_positions))
        tl.store(
            value_cache + cache_base + offsets,
            tl.load(v_raw + kv_base + offsets, mask=mask & kv_head_mask, other=0.0).to(tl.float32),
            mask=mask & kv_head_mask & (position < max_positions),
        )

    _real_batched_rope_cache_kernel = _kernel
    return _real_batched_rope_cache_kernel


def _compile_real_batched_attention_kernel(triton: Any, tl: Any) -> Any:
    global _real_batched_attention_kernel
    if _real_batched_attention_kernel is not None:
        return _real_batched_attention_kernel

    @triton.jit
    def _kernel(
        q_values,
        key_cache,
        value_cache,
        attended,
        start_position: tl.constexpr,
        q_width: tl.constexpr,
        kv_width: tl.constexpr,
        attention_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_positions: tl.constexpr,
        block_t: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offsets_t = tl.arange(0, block_t)
        offsets_d = tl.arange(0, block_d)
        position = start_position + token
        mask_t = offsets_t <= position
        mask_d = offsets_d < head_dim
        heads_per_kv = attention_heads // kv_heads
        kv_head = head // heads_per_kv

        q = tl.load(
            q_values + token * q_width + head * head_dim + offsets_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        key_offsets = offsets_t[:, None] * kv_width + kv_head * head_dim + offsets_d[None, :]
        keys = tl.load(
            key_cache + key_offsets,
            mask=mask_t[:, None] & mask_d[None, :] & (offsets_t[:, None] < max_positions),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * q[None, :], axis=1) * tl.rsqrt(head_dim + 0.0)
        scores = tl.where(mask_t, scores, -float("inf"))
        scores = scores - tl.max(scores, axis=0)
        weights = tl.exp(scores)
        weights = weights / tl.sum(weights, axis=0)

        value_offsets = offsets_t[:, None] * kv_width + kv_head * head_dim + offsets_d[None, :]
        values = tl.load(
            value_cache + value_offsets,
            mask=mask_t[:, None] & mask_d[None, :] & (offsets_t[:, None] < max_positions),
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(values * weights[:, None], axis=0)
        tl.store(attended + token * q_width + head * head_dim + offsets_d, out, mask=mask_d)

    _real_batched_attention_kernel = _kernel
    return _real_batched_attention_kernel


def _compile_real_batched_out_kernel(triton: Any, tl: Any) -> Any:
    global _real_batched_out_kernel
    if _real_batched_out_kernel is not None:
        return _real_batched_out_kernel

    @triton.jit
    def _kernel(
        attended,
        gate,
        out_weight,
        out,
        q_width: tl.constexpr,
        hidden_size: tl.constexpr,
        block_q: tl.constexpr,
    ):
        token = tl.program_id(0)
        row = tl.program_id(1)
        offsets = tl.arange(0, block_q)
        mask = offsets < q_width
        values = tl.load(attended + token * q_width + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_values = tl.load(gate + token * q_width + offsets, mask=mask, other=0.0).to(tl.float32)
        values = values * (1.0 / (1.0 + tl.exp(-gate_values)))
        weights = tl.load(out_weight + row * q_width + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + token * hidden_size + row, tl.sum(values * weights, axis=0))

    _real_batched_out_kernel = _kernel
    return _real_batched_out_kernel


def _triton_real_head_rmsnorm(
    values: Any,
    norm_weight: Any,
    *,
    head_dim: int,
    eps: float,
    norm_offset: float = 1.0,
) -> Any:
    torch, triton, tl = _load_triton()
    if values.ndim != 2:
        raise ValueError("values must have shape [tokens, width]")
    if norm_weight.ndim != 1 or norm_weight.shape[0] != head_dim:
        raise ValueError("norm_weight must have shape [head_dim]")
    if values.shape[1] % head_dim:
        raise ValueError("values width must be divisible by head_dim")
    if not values.is_cuda or not norm_weight.is_cuda:
        raise ValueError("values and norm_weight must be CUDA tensors")

    tokens, width = values.shape
    heads = width // head_dim
    out = torch.empty_like(values, dtype=torch.float32)
    block_d = triton.next_power_of_2(head_dim)
    kernel = _compile_real_head_rmsnorm_kernel(triton, tl)
    kernel[(tokens, heads)](
        values,
        norm_weight,
        out,
        eps,
        norm_offset,
        width,
        head_dim,
        block_d,
        num_warps=8,
    )
    return out


def triton_qwen36_batched_attention_decode(
    hidden: Any,
    key_cache: Any,
    value_cache: Any,
    norm_weight: Any,
    q_proj_weight: Any,
    k_proj_weight: Any,
    v_proj_weight: Any,
    out_proj_weight: Any,
    q_norm_weight: Any,
    k_norm_weight: Any,
    *,
    start_position: int = 0,
    attention_heads: int = 16,
    kv_heads: int = 2,
    head_dim: int = 256,
    rope_dim: int = 64,
    rope_theta: float = 10000.0,
    eps: float = 1e-6,
    qk_norm_eps: float = 1e-6,
    block_hidden: int = 2048,
    copy_cache: bool = True,
) -> tuple[Any, Any, Any]:
    """Run a staged real-shape Qwen3.6 full-attention decode boundary.

    The real Qwen3.6 `q_proj` tensor stores per-head query and output gate
    rows interleaved as `[query, gate]`. This staged path applies the gate
    before the output projection, uses model-configurable RoPE theta, and
    updates the KV cache before causal attention.
    """

    torch, triton, tl = _load_triton()
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden_size]")
    if key_cache.ndim != 3 or value_cache.ndim != 3:
        raise ValueError("key/value caches must have shape [positions, kv_heads, head_dim]")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key and value caches must have the same shape")
    if key_cache.shape[1:] != (kv_heads, head_dim):
        raise ValueError("cache shape does not match kv_heads/head_dim")
    if not all(tensor.is_cuda for tensor in (hidden, key_cache, value_cache)):
        raise ValueError("hidden and caches must be CUDA tensors")
    if start_position < 0 or start_position + hidden.shape[0] > key_cache.shape[0]:
        raise ValueError("decode positions exceed cache length")

    q_width = attention_heads * head_dim
    kv_width = kv_heads * head_dim
    if out_proj_weight.ndim != 2 or out_proj_weight.shape[1] != q_width:
        raise ValueError("out_proj_weight must have shape [hidden_size, attention_heads * head_dim]")
    if q_proj_weight.ndim != 2 or q_proj_weight.shape[0] < 2 * q_width or q_proj_weight.shape[1] != hidden.shape[1]:
        raise ValueError("q_proj_weight must have at least 2 * attention_heads * head_dim rows")
    if k_proj_weight.shape != (kv_width, hidden.shape[1]) or v_proj_weight.shape != (kv_width, hidden.shape[1]):
        raise ValueError("k/v projection weights must have shape [kv_heads * head_dim, hidden_size]")

    q_raw_full, k_raw, v_raw = triton_qwen36_batched_attention_project(
        hidden,
        norm_weight,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        eps=eps,
        block_hidden=block_hidden,
    )
    q_pairs = q_raw_full[:, : 2 * q_width].reshape(hidden.shape[0], attention_heads, 2 * head_dim)
    q_raw = q_pairs[:, :, :head_dim].contiguous().reshape(hidden.shape[0], q_width)
    gate = q_pairs[:, :, head_dim:].contiguous().reshape(hidden.shape[0], q_width)
    q_normed = _triton_real_head_rmsnorm(q_raw, q_norm_weight, head_dim=head_dim, eps=qk_norm_eps)
    k_normed = _triton_real_head_rmsnorm(k_raw, k_norm_weight, head_dim=head_dim, eps=qk_norm_eps)

    next_key_cache = key_cache.clone() if copy_cache else key_cache
    next_value_cache = value_cache.clone() if copy_cache else value_cache
    q_values = torch.empty_like(q_normed)
    attended = torch.empty_like(q_normed)
    out = torch.empty((hidden.shape[0], hidden.shape[1]), device=hidden.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(head_dim)
    block_t = triton.next_power_of_2(start_position + hidden.shape[0])
    block_q = triton.next_power_of_2(q_width)
    max_positions = key_cache.shape[0]

    rope_cache_kernel = _compile_real_batched_rope_cache_kernel(triton, tl)
    rope_cache_kernel[(hidden.shape[0], attention_heads)](
        q_normed,
        k_normed,
        v_raw,
        q_values,
        next_key_cache,
        next_value_cache,
        start_position,
        q_width,
        kv_width,
        attention_heads,
        kv_heads,
        head_dim,
        rope_dim,
        rope_theta,
        max_positions,
        block_d,
        num_warps=8,
    )

    attention_kernel = _compile_real_batched_attention_kernel(triton, tl)
    attention_kernel[(hidden.shape[0], attention_heads)](
        q_values,
        next_key_cache,
        next_value_cache,
        attended,
        start_position,
        q_width,
        kv_width,
        attention_heads,
        kv_heads,
        head_dim,
        max_positions,
        block_t,
        block_d,
        num_warps=8,
    )

    out_kernel = _compile_real_batched_out_kernel(triton, tl)
    out_kernel[(hidden.shape[0], hidden.shape[1])](
        attended,
        gate,
        out_proj_weight,
        out,
        q_width,
        hidden.shape[1],
        block_q,
        num_warps=8,
    )
    return out, next_key_cache, next_value_cache


def _compile_rope_cache_kernel(triton: Any, tl: Any) -> Any:
    global _rope_cache_kernel
    if _rope_cache_kernel is not None:
        return _rope_cache_kernel

    @triton.jit
    def _kernel(
        q_raw,
        k_raw,
        v_raw,
        q_values,
        key_cache,
        value_cache,
        position: tl.constexpr,
        attention_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rope_dim: tl.constexpr,
        max_positions: tl.constexpr,
        block_q: tl.constexpr,
        block_kv: tl.constexpr,
    ):
        q_offsets = tl.arange(0, block_q)
        kv_offsets = tl.arange(0, block_kv)
        q_width = attention_heads * head_dim
        kv_width = kv_heads * head_dim
        q_mask = q_offsets < q_width
        kv_mask = kv_offsets < kv_width
        rotate_dim = tl.minimum(rope_dim, head_dim)
        rotate_dim = rotate_dim - rotate_dim % 2

        q_dim = q_offsets % head_dim
        q_pair_dim = tl.where(q_dim % 2 == 0, q_dim + 1, q_dim - 1)
        q_pair_offsets = q_offsets - q_dim + q_pair_dim
        q_raw_values = tl.load(q_raw + q_offsets, mask=q_mask, other=0.0).to(tl.float32)
        q_pair_values = tl.load(q_raw + q_pair_offsets, mask=q_mask & (q_dim < rotate_dim), other=0.0).to(tl.float32)
        q_angle_dim = (q_dim // 2) * 2
        q_angle = position / tl.exp(tl.log(10000.0) * (q_angle_dim / rotate_dim))
        q_cos = tl.cos(q_angle)
        q_sin = tl.sin(q_angle)
        q_rotated = tl.where(
            q_dim % 2 == 0,
            q_raw_values * q_cos - q_pair_values * q_sin,
            q_pair_values * q_sin + q_raw_values * q_cos,
        )
        q_out = tl.where(q_dim < rotate_dim, q_rotated, q_raw_values)
        tl.store(q_values + q_offsets, q_out, mask=q_mask)

        kv_dim = kv_offsets % head_dim
        kv_pair_dim = tl.where(kv_dim % 2 == 0, kv_dim + 1, kv_dim - 1)
        kv_pair_offsets = kv_offsets - kv_dim + kv_pair_dim
        k_raw_values = tl.load(k_raw + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)
        k_pair_values = tl.load(k_raw + kv_pair_offsets, mask=kv_mask & (kv_dim < rotate_dim), other=0.0).to(
            tl.float32
        )
        kv_angle_dim = (kv_dim // 2) * 2
        kv_angle = position / tl.exp(tl.log(10000.0) * (kv_angle_dim / rotate_dim))
        kv_cos = tl.cos(kv_angle)
        kv_sin = tl.sin(kv_angle)
        k_rotated = tl.where(
            kv_dim % 2 == 0,
            k_raw_values * kv_cos - k_pair_values * kv_sin,
            k_pair_values * kv_sin + k_raw_values * kv_cos,
        )
        k_out = tl.where(kv_dim < rotate_dim, k_rotated, k_raw_values)

        cache_offset = position * kv_width + kv_offsets
        tl.store(key_cache + cache_offset, k_out, mask=kv_mask & (position < max_positions))
        tl.store(value_cache + cache_offset, tl.load(v_raw + kv_offsets, mask=kv_mask, other=0.0), mask=kv_mask)

    _rope_cache_kernel = _kernel
    return _rope_cache_kernel


def _compile_project_rope_cache_kernel(triton: Any, tl: Any) -> Any:
    global _project_rope_cache_kernel
    if _project_rope_cache_kernel is not None:
        return _project_rope_cache_kernel

    @triton.jit
    def _kernel(
        hidden,
        norm_weight,
        q_weight,
        k_weight,
        v_weight,
        q_values,
        key_cache,
        value_cache,
        eps: tl.constexpr,
        position: tl.constexpr,
        hidden_size: tl.constexpr,
        attention_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rope_dim: tl.constexpr,
        max_positions: tl.constexpr,
        block_h: tl.constexpr,
        block_q: tl.constexpr,
        block_kv: tl.constexpr,
    ):
        hidden_offsets = tl.arange(0, block_h)
        q_offsets = tl.arange(0, block_q)
        kv_offsets = tl.arange(0, block_kv)
        q_width = attention_heads * head_dim
        kv_width = kv_heads * head_dim
        hidden_mask = hidden_offsets < hidden_size
        q_mask = q_offsets < q_width
        kv_mask = kv_offsets < kv_width
        rotate_dim = tl.minimum(rope_dim, head_dim)
        rotate_dim = rotate_dim - rotate_dim % 2

        hidden_values = tl.load(hidden + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        norm_values = tl.load(norm_weight + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(hidden_values * hidden_values, axis=0) / hidden_size
        x = hidden_values * tl.rsqrt(mean_square + eps) * norm_values

        q_dim = q_offsets % head_dim
        q_pair_dim = tl.where(q_dim % 2 == 0, q_dim + 1, q_dim - 1)
        q_pair_offsets = q_offsets - q_dim + q_pair_dim
        q_weights = tl.load(
            q_weight + q_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=q_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_pair_weights = tl.load(
            q_weight + q_pair_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=q_mask[:, None] & (q_dim[:, None] < rotate_dim) & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_raw = tl.sum(q_weights * x[None, :], axis=1)
        q_pair_raw = tl.sum(q_pair_weights * x[None, :], axis=1)
        q_angle_dim = (q_dim // 2) * 2
        q_angle = position / tl.exp(tl.log(10000.0) * (q_angle_dim / rotate_dim))
        q_cos = tl.cos(q_angle)
        q_sin = tl.sin(q_angle)
        q_rotated = tl.where(
            q_dim % 2 == 0,
            q_raw * q_cos - q_pair_raw * q_sin,
            q_pair_raw * q_sin + q_raw * q_cos,
        )
        q_out = tl.where(q_dim < rotate_dim, q_rotated, q_raw)
        tl.store(q_values + q_offsets, q_out, mask=q_mask)

        kv_dim = kv_offsets % head_dim
        kv_pair_dim = tl.where(kv_dim % 2 == 0, kv_dim + 1, kv_dim - 1)
        kv_pair_offsets = kv_offsets - kv_dim + kv_pair_dim
        k_weights = tl.load(
            k_weight + kv_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=kv_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        k_pair_weights = tl.load(
            k_weight + kv_pair_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=kv_mask[:, None] & (kv_dim[:, None] < rotate_dim) & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_weights = tl.load(
            v_weight + kv_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=kv_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        k_raw = tl.sum(k_weights * x[None, :], axis=1)
        k_pair_raw = tl.sum(k_pair_weights * x[None, :], axis=1)
        v_raw = tl.sum(v_weights * x[None, :], axis=1)
        kv_angle_dim = (kv_dim // 2) * 2
        kv_angle = position / tl.exp(tl.log(10000.0) * (kv_angle_dim / rotate_dim))
        kv_cos = tl.cos(kv_angle)
        kv_sin = tl.sin(kv_angle)
        k_rotated = tl.where(
            kv_dim % 2 == 0,
            k_raw * kv_cos - k_pair_raw * kv_sin,
            k_pair_raw * kv_sin + k_raw * kv_cos,
        )
        k_out = tl.where(kv_dim < rotate_dim, k_rotated, k_raw)
        cache_offset = position * kv_width + kv_offsets
        tl.store(key_cache + cache_offset, k_out, mask=kv_mask & (position < max_positions))
        tl.store(value_cache + cache_offset, v_raw, mask=kv_mask & (position < max_positions))

    _project_rope_cache_kernel = _kernel
    return _project_rope_cache_kernel


def _compile_attention_kernel(triton: Any, tl: Any) -> Any:
    global _attention_kernel
    if _attention_kernel is not None:
        return _attention_kernel

    @triton.jit
    def _kernel(
        q_values,
        key_cache,
        value_cache,
        attended,
        position: tl.constexpr,
        attention_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_positions: tl.constexpr,
        block_t: tl.constexpr,
        block_d: tl.constexpr,
    ):
        head = tl.program_id(0)
        offsets_t = tl.arange(0, block_t)
        offsets_d = tl.arange(0, block_d)
        mask_t = offsets_t <= position
        mask_d = offsets_d < head_dim
        heads_per_kv = attention_heads // kv_heads
        kv_head = head // heads_per_kv
        kv_width = kv_heads * head_dim

        q = tl.load(q_values + head * head_dim + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        key_offsets = offsets_t[:, None] * kv_width + kv_head * head_dim + offsets_d[None, :]
        keys = tl.load(
            key_cache + key_offsets,
            mask=mask_t[:, None] & mask_d[None, :] & (offsets_t[:, None] < max_positions),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * q[None, :], axis=1) * tl.rsqrt(head_dim + 0.0)
        scores = tl.where(mask_t, scores, -float("inf"))
        scores = scores - tl.max(scores, axis=0)
        weights = tl.exp(scores)
        weights = weights / tl.sum(weights, axis=0)

        value_offsets = offsets_t[:, None] * kv_width + kv_head * head_dim + offsets_d[None, :]
        values = tl.load(
            value_cache + value_offsets,
            mask=mask_t[:, None] & mask_d[None, :] & (offsets_t[:, None] < max_positions),
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(values * weights[:, None], axis=0)
        tl.store(attended + head * head_dim + offsets_d, out, mask=mask_d)

    _attention_kernel = _kernel
    return _attention_kernel


def _compile_out_kernel(triton: Any, tl: Any) -> Any:
    global _out_kernel
    if _out_kernel is not None:
        return _out_kernel

    @triton.jit
    def _kernel(
        hidden,
        attended,
        out_weight,
        out,
        hidden_size: tl.constexpr,
        q_width: tl.constexpr,
        block_h: tl.constexpr,
        block_q: tl.constexpr,
    ):
        hidden_offsets = tl.arange(0, block_h)
        q_offsets = tl.arange(0, block_q)
        hidden_mask = hidden_offsets < hidden_size
        q_mask = q_offsets < q_width

        attended_values = tl.load(attended + q_offsets, mask=q_mask, other=0.0).to(tl.float32)
        weights = tl.load(
            out_weight + hidden_offsets[:, None] * q_width + q_offsets[None, :],
            mask=hidden_mask[:, None] & q_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        layer_out = tl.sum(weights * attended_values[None, :], axis=1)
        residual = tl.load(hidden + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        tl.store(out + hidden_offsets, residual + layer_out, mask=hidden_mask)

    _out_kernel = _kernel
    return _out_kernel


def triton_synthetic_qwen36_attention_decode(
    hidden: Any,
    key_cache: Any,
    value_cache: Any,
    norm_weight: Any,
    q_weight: Any,
    k_weight: Any,
    v_weight: Any,
    out_weight: Any,
    *,
    position: int,
    attention_heads: int,
    kv_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float = 1e-6,
    fuse_project_rope: bool = False,
    copy_cache: bool = True,
) -> tuple[Any, Any, Any]:
    """Run a prototype one-token synthetic Qwen3.6 attention decode block."""

    torch, triton, tl = _load_triton()
    if hidden.ndim != 1:
        raise ValueError("hidden must have shape [hidden_size]")
    if key_cache.ndim != 3 or value_cache.ndim != 3:
        raise ValueError("key/value caches must have shape [positions, kv_heads, head_dim]")
    if not hidden.is_cuda or not key_cache.is_cuda or not value_cache.is_cuda:
        raise ValueError("hidden and caches must be CUDA tensors")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key and value caches must have the same shape")
    if key_cache.shape[1:] != (kv_heads, head_dim):
        raise ValueError("cache shape does not match kv_heads/head_dim")
    if position >= key_cache.shape[0]:
        raise ValueError("position exceeds cache length")

    hidden_size = hidden.shape[0]
    q_width = attention_heads * head_dim
    kv_width = kv_heads * head_dim
    max_positions = key_cache.shape[0]
    if hidden_size > 128 or q_width > 128 or kv_width > 128 or max_positions > 128:
        raise ValueError("prototype synthetic Qwen3.6 attention kernel only supports tiny synthetic shapes")

    q_values = torch.empty((q_width,), device=hidden.device, dtype=hidden.dtype)
    if copy_cache:
        new_key_cache = key_cache.clone()
        new_value_cache = value_cache.clone()
    else:
        new_key_cache = key_cache
        new_value_cache = value_cache
    attended = torch.empty_like(q_values)
    out = torch.empty_like(hidden)

    block_h = triton.next_power_of_2(hidden_size)
    block_q = triton.next_power_of_2(q_width)
    block_kv = triton.next_power_of_2(kv_width)
    block_t = triton.next_power_of_2(position + 1)
    block_d = triton.next_power_of_2(head_dim)

    if fuse_project_rope:
        project_rope_cache_kernel = _compile_project_rope_cache_kernel(triton, tl)
        project_rope_cache_kernel[(1,)](
            hidden,
            norm_weight,
            q_weight,
            k_weight,
            v_weight,
            q_values,
            new_key_cache,
            new_value_cache,
            eps,
            position,
            hidden_size,
            attention_heads,
            kv_heads,
            head_dim,
            rope_dim,
            max_positions,
            block_h,
            block_q,
            block_kv,
        )
    else:
        q_raw = torch.empty((q_width,), device=hidden.device, dtype=hidden.dtype)
        k_raw = torch.empty((kv_width,), device=hidden.device, dtype=hidden.dtype)
        v_raw = torch.empty_like(k_raw)
        project_kernel = _compile_project_kernel(triton, tl)
        project_kernel[(1,)](
            hidden,
            norm_weight,
            q_weight,
            k_weight,
            v_weight,
            q_raw,
            k_raw,
            v_raw,
            eps,
            hidden_size,
            q_width,
            kv_width,
            block_h,
            block_q,
            block_kv,
        )

        rope_cache_kernel = _compile_rope_cache_kernel(triton, tl)
        rope_cache_kernel[(1,)](
            q_raw,
            k_raw,
            v_raw,
            q_values,
            new_key_cache,
            new_value_cache,
            position,
            attention_heads,
            kv_heads,
            head_dim,
            rope_dim,
            max_positions,
            block_q,
            block_kv,
        )

    attention_kernel = _compile_attention_kernel(triton, tl)
    attention_kernel[(attention_heads,)](
        q_values,
        new_key_cache,
        new_value_cache,
        attended,
        position,
        attention_heads,
        kv_heads,
        head_dim,
        max_positions,
        block_t,
        block_d,
    )

    out_kernel = _compile_out_kernel(triton, tl)
    out_kernel[(1,)](
        hidden,
        attended,
        out_weight,
        out,
        hidden_size,
        q_width,
        block_h,
        block_q,
    )
    return out, new_key_cache, new_value_cache

"""Real-weight PyTorch decode pieces for Qwen3.6 serving bring-up."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastkernels.models import Qwen36A3BSpec, Qwen36AttentionWeights, Qwen36LinearAttentionWeights, Qwen36MoEWeights
from fastkernels.reference.moe import _require_torch


@dataclass(frozen=True)
class Qwen36RealDecodeState:
    """Decode state for the real-weight Qwen3.6 reference runtime."""

    deltanet_conv_states: tuple[Any, ...]
    deltanet_recurrent_states: tuple[Any, ...]
    attention_key_cache: tuple[Any, ...]
    attention_value_cache: tuple[Any, ...]
    position: int = 0


def qwen36_real_rms_norm(hidden: Any, weight: Any, eps: float = 1e-6) -> Any:
    """Qwen3.5/3.6 RMSNorm: normalized input times `(1 + weight)`."""

    hidden_f = hidden.float()
    normed = hidden_f * _require_torch().rsqrt((hidden_f * hidden_f).mean(dim=-1, keepdim=True) + eps)
    return normed * (1.0 + weight.float())


def qwen36_l2_norm(hidden: Any, eps: float = 1e-6) -> Any:
    """L2 norm used by the gated DeltaNet recurrent rule."""

    torch = _require_torch()
    return hidden * torch.rsqrt((hidden * hidden).sum(dim=-1, keepdim=True) + eps)


def initial_qwen36_real_decode_state(
    spec: Qwen36A3BSpec,
    *,
    max_positions: int | None = None,
    conv_kernel_size: int = 4,
    device: Any = "cpu",
    dtype: Any = None,
) -> Qwen36RealDecodeState:
    """Create zero recurrent, convolution, and KV-cache state for one sequence."""

    torch = _require_torch()
    dtype = dtype or torch.float32
    max_positions = max_positions or spec.context_length
    counts = spec.layer_counts()
    conv_dim = 2 * spec.deltanet_qk_heads * spec.deltanet_head_dim + spec.deltanet_value_heads * spec.deltanet_head_dim

    return Qwen36RealDecodeState(
        deltanet_conv_states=tuple(
            torch.zeros(conv_dim, conv_kernel_size, device=device, dtype=dtype)
            for _ in range(counts["deltanet_moe"])
        ),
        deltanet_recurrent_states=tuple(
            torch.zeros(
                spec.deltanet_value_heads,
                spec.deltanet_head_dim,
                spec.deltanet_head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(counts["deltanet_moe"])
        ),
        attention_key_cache=tuple(
            torch.zeros(spec.attention_cache_shape(max_positions), device=device, dtype=dtype)
            for _ in range(counts["attention_moe"])
        ),
        attention_value_cache=tuple(
            torch.zeros(spec.attention_cache_shape(max_positions), device=device, dtype=dtype)
            for _ in range(counts["attention_moe"])
        ),
        position=0,
    )


def _linear(hidden: Any, weight: Any) -> Any:
    return hidden.float() @ weight.float().transpose(-1, -2)


def _gated_rms_norm(hidden: Any, gate: Any, weight: Any, eps: float = 1e-6) -> Any:
    torch = _require_torch()
    hidden_f = hidden.float()
    normed = hidden_f * torch.rsqrt((hidden_f * hidden_f).mean(dim=-1, keepdim=True) + eps)
    return normed * weight.float() * torch.nn.functional.silu(gate.float())


def _causal_depthwise_conv_update(mixed: Any, conv_state: Any, conv_weight: Any) -> tuple[Any, Any]:
    torch = _require_torch()
    if mixed.ndim != 2:
        raise ValueError("mixed must have shape [tokens, channels]")
    if conv_state.ndim != 2:
        raise ValueError("conv_state must have shape [channels, kernel_size]")
    channels, kernel_size = conv_state.shape
    if mixed.shape[1] != channels:
        raise ValueError("mixed and conv_state disagree on channel count")
    if conv_weight.ndim != 3 or conv_weight.shape[0] != channels:
        raise ValueError("conv_weight must have shape [channels, 1, kernel_size]")

    x = mixed.transpose(0, 1).unsqueeze(0)
    combined = torch.cat((conv_state.unsqueeze(0).float(), x.float()), dim=-1)
    next_state = combined[:, :, -kernel_size:].squeeze(0).to(conv_state.dtype)
    conv = torch.nn.functional.conv1d(combined, conv_weight.float(), padding=0, groups=channels)
    return torch.nn.functional.silu(conv[:, :, -mixed.shape[0] :]).squeeze(0).transpose(0, 1), next_state


def qwen36_real_deltanet_update(
    hidden: Any,
    conv_state: Any,
    recurrent_state: Any,
    weights: Qwen36LinearAttentionWeights,
    spec: Qwen36A3BSpec,
    *,
    eps: float = 1e-6,
) -> tuple[Any, Any, Any]:
    """Run the real Qwen3.6 Gated DeltaNet mixer update."""

    torch = _require_torch()
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden_size]")

    x = qwen36_real_rms_norm(hidden, weights.input_norm_weight, eps=eps)
    mixed_qkv = _linear(x, weights.in_proj_qkv_weight)
    z = _linear(x, weights.in_proj_z_weight)
    a_logits = _linear(x, weights.in_proj_a_weight)
    b_logits = _linear(x, weights.in_proj_b_weight)
    return qwen36_real_deltanet_update_from_projections(
        hidden,
        conv_state,
        recurrent_state,
        weights,
        spec,
        mixed_qkv,
        z,
        a_logits,
        b_logits,
        eps=eps,
    )


def qwen36_real_deltanet_update_from_projections(
    hidden: Any,
    conv_state: Any,
    recurrent_state: Any,
    weights: Qwen36LinearAttentionWeights,
    spec: Qwen36A3BSpec,
    mixed_qkv: Any,
    z: Any,
    a_logits: Any,
    b_logits: Any,
    *,
    eps: float = 1e-6,
) -> tuple[Any, Any, Any]:
    """Run real Qwen3.6 DeltaNet after input-norm projection staging."""

    torch = _require_torch()
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden_size]")
    tokens = hidden.shape[0]
    key_dim = spec.deltanet_qk_heads * spec.deltanet_head_dim
    value_dim = spec.deltanet_value_heads * spec.deltanet_head_dim
    if mixed_qkv.shape != (tokens, 2 * key_dim + value_dim):
        raise ValueError("mixed_qkv shape does not match Qwen3.6 DeltaNet dimensions")
    if z.shape != (tokens, value_dim):
        raise ValueError("z shape does not match Qwen3.6 DeltaNet value dimensions")
    if a_logits.shape != (tokens, spec.deltanet_value_heads):
        raise ValueError("a_logits shape does not match Qwen3.6 DeltaNet value heads")
    if b_logits.shape != (tokens, spec.deltanet_value_heads):
        raise ValueError("b_logits shape does not match Qwen3.6 DeltaNet value heads")

    mixed_qkv, next_conv_state = _causal_depthwise_conv_update(mixed_qkv, conv_state, weights.conv1d_weight)
    return qwen36_real_deltanet_update_from_convolved_projections(
        hidden,
        recurrent_state,
        weights,
        spec,
        mixed_qkv,
        z,
        a_logits,
        b_logits,
        next_conv_state,
        eps=eps,
    )


def qwen36_real_deltanet_update_from_convolved_projections(
    hidden: Any,
    recurrent_state: Any,
    weights: Qwen36LinearAttentionWeights,
    spec: Qwen36A3BSpec,
    mixed_qkv: Any,
    z: Any,
    a_logits: Any,
    b_logits: Any,
    next_conv_state: Any,
    *,
    eps: float = 1e-6,
) -> tuple[Any, Any, Any]:
    """Run real Qwen3.6 DeltaNet after projection and conv staging."""

    torch = _require_torch()
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden_size]")
    tokens = hidden.shape[0]
    key_dim = spec.deltanet_qk_heads * spec.deltanet_head_dim
    value_dim = spec.deltanet_value_heads * spec.deltanet_head_dim
    if mixed_qkv.shape != (tokens, 2 * key_dim + value_dim):
        raise ValueError("mixed_qkv shape does not match Qwen3.6 DeltaNet dimensions")
    if z.shape != (tokens, value_dim):
        raise ValueError("z shape does not match Qwen3.6 DeltaNet value dimensions")
    if a_logits.shape != (tokens, spec.deltanet_value_heads):
        raise ValueError("a_logits shape does not match Qwen3.6 DeltaNet value heads")
    if b_logits.shape != (tokens, spec.deltanet_value_heads):
        raise ValueError("b_logits shape does not match Qwen3.6 DeltaNet value heads")

    query, key, value = torch.split(mixed_qkv, (key_dim, key_dim, value_dim), dim=-1)
    query = query.reshape(tokens, spec.deltanet_qk_heads, spec.deltanet_head_dim)
    key = key.reshape(tokens, spec.deltanet_qk_heads, spec.deltanet_head_dim)
    value = value.reshape(tokens, spec.deltanet_value_heads, spec.deltanet_head_dim)

    z = z.reshape(tokens, spec.deltanet_value_heads, spec.deltanet_head_dim)
    beta = torch.sigmoid(b_logits)
    g = -weights.a_log.float().exp() * torch.nn.functional.softplus(
        a_logits + weights.dt_bias.float()
    )

    repeat = spec.deltanet_value_heads // spec.deltanet_qk_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=1)
        key = key.repeat_interleave(repeat, dim=1)

    query = qwen36_l2_norm(query, eps=eps) * (spec.deltanet_head_dim**-0.5)
    key = qwen36_l2_norm(key, eps=eps)
    state = recurrent_state.float()
    outputs = []
    for token_idx in range(tokens):
        q_t = query[token_idx].float()
        k_t = key[token_idx].float()
        v_t = value[token_idx].float()
        state = state * g[token_idx].float().exp().reshape(spec.deltanet_value_heads, 1, 1)
        delta = (v_t - (state * k_t.unsqueeze(-1)).sum(dim=-2)) * beta[token_idx].float().reshape(
            spec.deltanet_value_heads,
            1,
        )
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))

    core = torch.stack(outputs, dim=0)
    core = _gated_rms_norm(
        core.reshape(tokens * spec.deltanet_value_heads, spec.deltanet_head_dim),
        z.reshape(tokens * spec.deltanet_value_heads, spec.deltanet_head_dim),
        weights.linear_norm_weight,
        eps=eps,
    ).reshape(tokens, value_dim)
    update = _linear(core, weights.out_proj_weight)
    return update, next_conv_state, state.to(recurrent_state.dtype)


def _rotate_half(hidden: Any) -> Any:
    torch = _require_torch()
    first, second = hidden[..., : hidden.shape[-1] // 2], hidden[..., hidden.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rope(values: Any, positions: Any, rope_dim: int, rope_theta: float = 10000.0) -> Any:
    torch = _require_torch()
    rotate_dim = min(rope_dim, values.shape[-1])
    rotate_dim -= rotate_dim % 2
    if rotate_dim <= 0:
        return values
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotate_dim, 2, device=values.device, dtype=torch.float32) / rotate_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    rotary, passed = values[..., :rotate_dim], values[..., rotate_dim:]
    rotated = rotary * emb.cos()[:, None, :] + _rotate_half(rotary) * emb.sin()[:, None, :]
    return torch.cat((rotated, passed), dim=-1)


def qwen36_real_attention_update(
    hidden: Any,
    key_cache: Any,
    value_cache: Any,
    weights: Qwen36AttentionWeights,
    spec: Qwen36A3BSpec,
    *,
    start_position: int = 0,
    eps: float = 1e-6,
    rope_theta: float | None = None,
) -> tuple[Any, Any, Any]:
    """Run the real Qwen3.6 gated full-attention mixer update in PyTorch."""

    torch = _require_torch()
    tokens = hidden.shape[0]
    x = qwen36_real_rms_norm(hidden, weights.input_norm_weight, eps=eps)
    q_pairs = _linear(x, weights.q_proj_weight).reshape(tokens, spec.attention_heads, 2 * spec.attention_head_dim)
    query, gate = torch.chunk(q_pairs, 2, dim=-1)
    key = _linear(x, weights.k_proj_weight).reshape(tokens, spec.attention_kv_heads, spec.attention_head_dim)
    value = _linear(x, weights.v_proj_weight).reshape(tokens, spec.attention_kv_heads, spec.attention_head_dim)
    query = qwen36_real_rms_norm(query, weights.q_norm_weight, eps=eps)
    key = qwen36_real_rms_norm(key, weights.k_norm_weight, eps=eps)
    positions = torch.arange(start_position, start_position + tokens, device=hidden.device)
    theta = spec.rope_theta if rope_theta is None else rope_theta
    query = _apply_rope(query, positions, spec.rope_dim, rope_theta=theta)
    key = _apply_rope(key, positions, spec.rope_dim, rope_theta=theta)

    next_key_cache = key_cache.clone()
    next_value_cache = value_cache.clone()
    next_key_cache[start_position : start_position + tokens] = key
    next_value_cache[start_position : start_position + tokens] = value

    attended = torch.empty_like(query)
    scale = spec.attention_head_dim**-0.5
    for token_idx in range(tokens):
        position = start_position + token_idx
        for head_idx in range(spec.attention_heads):
            kv_head = head_idx // spec.attention_heads_per_kv_head
            scores = (next_key_cache[: position + 1, kv_head] * query[token_idx, head_idx]).sum(dim=-1) * scale
            probs = torch.softmax(scores, dim=-1)
            attended[token_idx, head_idx] = (next_value_cache[: position + 1, kv_head] * probs[:, None]).sum(dim=0)

    attended = attended.reshape(tokens, spec.attention_heads * spec.attention_head_dim)
    return _linear(attended * torch.sigmoid(gate.reshape(tokens, -1).float()), weights.o_proj_weight), next_key_cache, next_value_cache


def qwen36_real_moe_update(hidden: Any, weights: Qwen36MoEWeights, spec: Qwen36A3BSpec, *, eps: float = 1e-6) -> Any:
    """Run the real Qwen3.6 sparse MoE block in PyTorch."""

    torch = _require_torch()
    hidden_f = hidden.float()
    logits = _linear(qwen36_real_rms_norm(hidden_f, weights.norm_weight, eps=eps), weights.router_weight)
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, k=spec.num_routed_experts, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    output = torch.zeros_like(hidden_f)
    for token_idx in range(hidden.shape[0]):
        token = hidden_f[token_idx]
        for route_idx in range(spec.num_routed_experts):
            expert_id = int(topk_ids[token_idx, route_idx].item())
            gate_up = weights.expert_gate_up_weight[expert_id].float() @ token
            gate, up = torch.chunk(gate_up, 2, dim=0)
            output[token_idx] += topk_weights[token_idx, route_idx].float() * (
                weights.expert_down_weight[expert_id].float() @ (torch.nn.functional.silu(gate) * up)
            )
        shared_gate_up = weights.shared_gate_up_weight.float() @ token
        shared_gate, shared_up = torch.chunk(shared_gate_up, 2, dim=0)
        shared = weights.shared_down_weight.float() @ (torch.nn.functional.silu(shared_gate) * shared_up)
        if weights.shared_expert_gate_weight is not None:
            shared = shared * torch.sigmoid((weights.shared_expert_gate_weight.float().reshape(-1, spec.hidden_size) @ token)[0])
        output[token_idx] += shared
    return output


def qwen36_real_deltanet_moe_layer(
    hidden: Any,
    conv_state: Any,
    recurrent_state: Any,
    linear_weights: Qwen36LinearAttentionWeights,
    moe_weights: Qwen36MoEWeights,
    spec: Qwen36A3BSpec,
) -> tuple[Any, Any, Any]:
    """Run one real `Gated DeltaNet -> MoE` layer."""

    update, next_conv_state, next_recurrent_state = qwen36_real_deltanet_update(
        hidden,
        conv_state,
        recurrent_state,
        linear_weights,
        spec,
    )
    hidden = hidden.float() + update.float()
    return hidden + qwen36_real_moe_update(hidden, moe_weights, spec), next_conv_state, next_recurrent_state


def qwen36_real_attention_moe_layer(
    hidden: Any,
    key_cache: Any,
    value_cache: Any,
    attention_weights: Qwen36AttentionWeights,
    moe_weights: Qwen36MoEWeights,
    spec: Qwen36A3BSpec,
    *,
    start_position: int,
) -> tuple[Any, Any, Any]:
    """Run one real `Gated Attention -> MoE` layer."""

    update, next_key_cache, next_value_cache = qwen36_real_attention_update(
        hidden,
        key_cache,
        value_cache,
        attention_weights,
        spec,
        start_position=start_position,
    )
    hidden = hidden.float() + update.float()
    return hidden + qwen36_real_moe_update(hidden, moe_weights, spec), next_key_cache, next_value_cache

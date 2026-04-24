"""Synthetic PyTorch decode reference for Qwen3.6 megakernel work."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from fastkernels.models import Qwen36A3BSpec
from fastkernels.reference.moe import _require_torch, reference_fused_swiglu, reference_routed_moe


@dataclass(frozen=True)
class Qwen36DecodeState:
    """Decode state for synthetic Qwen3.6 layer-fusion experiments."""

    deltanet_states: tuple[Any, ...]
    attention_key_cache: tuple[Any, ...]
    attention_value_cache: tuple[Any, ...]
    position: int = 0


def _linear(x: Any, weight: Any) -> Any:
    return x @ weight.transpose(-1, -2)


def _rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    torch = _require_torch()
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def _expert_path(hidden: Any, gate_up_weight: Any, down_weight: Any) -> Any:
    activation = reference_fused_swiglu(_linear(hidden, gate_up_weight))
    return _linear(activation, down_weight)


def _apply_rope(x: Any, position: int, rope_dim: int) -> Any:
    torch = _require_torch()
    if rope_dim <= 0:
        return x
    rotate_dim = min(rope_dim, x.shape[-1])
    rotate_dim -= rotate_dim % 2
    if rotate_dim == 0:
        return x

    angles = torch.arange(0, rotate_dim, 2, device=x.device, dtype=torch.float32)
    angles = position / (10000.0 ** (angles / rotate_dim))
    cos = torch.cos(angles).to(x.dtype)
    sin = torch.sin(angles).to(x.dtype)
    rotated = x.clone()
    even = x[..., :rotate_dim:2]
    odd = x[..., 1:rotate_dim:2]
    rotated[..., :rotate_dim:2] = even * cos - odd * sin
    rotated[..., 1:rotate_dim:2] = even * sin + odd * cos
    return rotated


def initial_qwen36_decode_state(
    spec: Qwen36A3BSpec,
    *,
    max_positions: int | None = None,
    device: Any = "cpu",
    dtype: Any = None,
) -> Qwen36DecodeState:
    """Create a zero decode state for synthetic Qwen3.6 reference runs."""

    torch = _require_torch()
    dtype = dtype or torch.float32
    max_positions = max_positions or spec.context_length
    counts = spec.layer_counts()

    deltanet_states = tuple(
        torch.zeros(spec.deltanet_state_shape(), device=device, dtype=dtype) for _ in range(counts["deltanet_moe"])
    )
    attention_key_cache = tuple(
        torch.zeros(spec.attention_cache_shape(max_positions), device=device, dtype=dtype)
        for _ in range(counts["attention_moe"])
    )
    attention_value_cache = tuple(
        torch.zeros(spec.attention_cache_shape(max_positions), device=device, dtype=dtype)
        for _ in range(counts["attention_moe"])
    )
    return Qwen36DecodeState(
        deltanet_states=deltanet_states,
        attention_key_cache=attention_key_cache,
        attention_value_cache=attention_value_cache,
        position=0,
    )


def make_synthetic_qwen36_decode_weights(
    spec: Qwen36A3BSpec,
    *,
    device: Any = "cpu",
    dtype: Any = None,
    seed: int = 0,
    scale: float = 0.02,
) -> dict[str, Any]:
    """Create deterministic tiny weights for the synthetic decode reference."""

    torch = _require_torch()
    dtype = dtype or torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def randn(*shape: int) -> Any:
        return (scale * torch.randn(*shape, generator=generator, dtype=dtype)).to(device)

    value_width = spec.deltanet_qk_heads * spec.deltanet_value_dim_per_qk_head
    attn_q_width = spec.attention_heads * spec.attention_head_dim
    attn_kv_width = spec.attention_kv_heads * spec.attention_head_dim

    layers = []
    for kind in spec.layer_kinds():
        layer: dict[str, Any] = {
            "kind": kind,
            "norm_weight": torch.ones(spec.hidden_size, device=device, dtype=dtype),
            "router_weight": randn(spec.num_experts, spec.hidden_size),
            "expert_gate_up_weight": randn(spec.num_experts, spec.gated_up_features, spec.hidden_size),
            "expert_down_weight": randn(spec.num_experts, spec.hidden_size, spec.expert_intermediate_size),
            "shared_gate_up_weight": randn(spec.gated_up_features, spec.hidden_size),
            "shared_down_weight": randn(spec.hidden_size, spec.expert_intermediate_size),
        }
        if kind == "deltanet_moe":
            layer.update(
                {
                    "q_weight": randn(spec.deltanet_qk_heads * spec.deltanet_head_dim, spec.hidden_size),
                    "k_weight": randn(spec.deltanet_qk_heads * spec.deltanet_head_dim, spec.hidden_size),
                    "v_weight": randn(value_width, spec.hidden_size),
                    "gate_weight": randn(value_width, spec.hidden_size),
                    "out_weight": randn(spec.hidden_size, value_width),
                }
            )
        elif kind == "attention_moe":
            layer.update(
                {
                    "q_weight": randn(attn_q_width, spec.hidden_size),
                    "k_weight": randn(attn_kv_width, spec.hidden_size),
                    "v_weight": randn(attn_kv_width, spec.hidden_size),
                    "out_weight": randn(spec.hidden_size, attn_q_width),
                }
            )
        else:  # pragma: no cover - protected by spec layer pattern
            raise ValueError(f"unknown layer kind: {kind}")
        layers.append(layer)

    return {
        "embedding": randn(spec.vocab_size, spec.hidden_size),
        "output_norm_weight": torch.ones(spec.hidden_size, device=device, dtype=dtype),
        "lm_head_weight": randn(spec.vocab_size, spec.hidden_size),
        "layers": tuple(layers),
    }


def reference_qwen36_deltanet_decode(hidden: Any, state: Any, layer: dict[str, Any], spec: Qwen36A3BSpec) -> tuple[Any, Any]:
    """Run one synthetic Gated DeltaNet decode layer."""

    torch = _require_torch()
    x = _rms_norm(hidden, layer["norm_weight"])
    q = _linear(x, layer["q_weight"]).reshape(spec.deltanet_qk_heads, spec.deltanet_head_dim)
    k = _linear(x, layer["k_weight"]).reshape(spec.deltanet_qk_heads, spec.deltanet_head_dim)
    v = _linear(x, layer["v_weight"]).reshape(spec.deltanet_qk_heads, spec.deltanet_value_dim_per_qk_head)
    gate = torch.sigmoid(_linear(x, layer["gate_weight"])).reshape(
        spec.deltanet_qk_heads,
        spec.deltanet_value_dim_per_qk_head,
    )

    new_state = 0.95 * state + torch.einsum("hd,hv->hdv", k, v)
    recurrent = torch.einsum("hd,hdv->hv", q, new_state) * gate
    layer_out = _linear(recurrent.reshape(-1), layer["out_weight"])
    return hidden + layer_out, new_state


def reference_qwen36_attention_decode(
    hidden: Any,
    key_cache: Any,
    value_cache: Any,
    layer: dict[str, Any],
    spec: Qwen36A3BSpec,
    position: int,
) -> tuple[Any, Any, Any]:
    """Run one synthetic grouped-query attention decode layer."""

    torch = _require_torch()
    x = _rms_norm(hidden, layer["norm_weight"])
    q = _linear(x, layer["q_weight"]).reshape(spec.attention_heads, spec.attention_head_dim)
    k = _linear(x, layer["k_weight"]).reshape(spec.attention_kv_heads, spec.attention_head_dim)
    v = _linear(x, layer["v_weight"]).reshape(spec.attention_kv_heads, spec.attention_head_dim)

    q = _apply_rope(q, position, spec.rope_dim)
    k = _apply_rope(k, position, spec.rope_dim)
    new_key_cache = key_cache.clone()
    new_value_cache = value_cache.clone()
    new_key_cache[position] = k
    new_value_cache[position] = v

    keys = new_key_cache[: position + 1]
    values = new_value_cache[: position + 1]
    head_outputs = []
    scale = 1.0 / math.sqrt(spec.attention_head_dim)
    for head_idx in range(spec.attention_heads):
        kv_idx = head_idx // spec.attention_heads_per_kv_head
        scores = torch.einsum("d,td->t", q[head_idx], keys[:, kv_idx]) * scale
        weights = torch.softmax(scores, dim=-1)
        head_outputs.append(torch.einsum("t,td->d", weights, values[:, kv_idx]))

    attended = torch.stack(head_outputs, dim=0).reshape(-1)
    layer_out = _linear(attended, layer["out_weight"])
    return hidden + layer_out, new_key_cache, new_value_cache


def reference_qwen36_moe_decode(hidden: Any, layer: dict[str, Any], spec: Qwen36A3BSpec) -> Any:
    """Run one synthetic Qwen3.6 routed + shared MoE decode block."""

    router_logits = _linear(_rms_norm(hidden, layer["norm_weight"]), layer["router_weight"])
    routed = reference_routed_moe(
        hidden.unsqueeze(0),
        router_logits.unsqueeze(0),
        layer["expert_gate_up_weight"],
        layer["expert_down_weight"],
        top_k=spec.num_routed_experts,
    ).squeeze(0)
    shared = _expert_path(hidden, layer["shared_gate_up_weight"], layer["shared_down_weight"])
    return hidden + routed + shared


def reference_qwen36_decode_step(
    token_id: int,
    state: Qwen36DecodeState,
    weights: dict[str, Any],
    spec: Qwen36A3BSpec,
) -> tuple[Any, Qwen36DecodeState]:
    """Run one synthetic Qwen3.6 decode step and return logits plus new state."""

    if state.position >= spec.context_length:
        raise ValueError("decode state position exceeds spec.context_length")

    hidden = weights["embedding"][token_id]
    next_deltanet_states = []
    next_attention_keys = []
    next_attention_values = []
    deltanet_idx = 0
    attention_idx = 0

    for layer in weights["layers"]:
        if layer["kind"] == "deltanet_moe":
            hidden, new_delta_state = reference_qwen36_deltanet_decode(
                hidden,
                state.deltanet_states[deltanet_idx],
                layer,
                spec,
            )
            next_deltanet_states.append(new_delta_state)
            deltanet_idx += 1
        elif layer["kind"] == "attention_moe":
            hidden, new_key_cache, new_value_cache = reference_qwen36_attention_decode(
                hidden,
                state.attention_key_cache[attention_idx],
                state.attention_value_cache[attention_idx],
                layer,
                spec,
                state.position,
            )
            next_attention_keys.append(new_key_cache)
            next_attention_values.append(new_value_cache)
            attention_idx += 1
        else:  # pragma: no cover - protected by generated weights
            raise ValueError(f"unknown layer kind: {layer['kind']}")
        hidden = reference_qwen36_moe_decode(hidden, layer, spec)

    hidden = _rms_norm(hidden, weights["output_norm_weight"])
    logits = _linear(hidden, weights["lm_head_weight"])
    next_state = Qwen36DecodeState(
        deltanet_states=tuple(next_deltanet_states),
        attention_key_cache=tuple(next_attention_keys),
        attention_value_cache=tuple(next_attention_values),
        position=state.position + 1,
    )
    return logits, next_state

"""Qwen3.6 MoE weight layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .qwen36 import Qwen36A3BSpec, qwen36_35b_a3b_spec


@dataclass(frozen=True)
class Qwen36MoEWeightKeys:
    """Resolved source tensor names for one Qwen3.6 MoE layer."""

    norm_weight: str
    router_weight: str
    expert_gate_up_weight: tuple[tuple[str, str] | str, ...] | str
    expert_down_weight: tuple[str, ...] | str
    shared_gate_up_weight: tuple[str, str] | str
    shared_down_weight: str
    shared_expert_gate_weight: str | None = None


@dataclass(frozen=True)
class Qwen36MoEWeights:
    """Kernel-facing tensors for one Qwen3.6 MoE layer."""

    norm_weight: Any
    router_weight: Any
    expert_gate_up_weight: Any
    expert_down_weight: Any
    shared_gate_up_weight: Any
    shared_down_weight: Any
    keys: Qwen36MoEWeightKeys
    shared_expert_gate_weight: Any | None = None


@dataclass(frozen=True)
class Qwen36LinearAttentionWeightKeys:
    """Resolved source tensor names for one Qwen3.6 linear-attention layer."""

    input_norm_weight: str
    in_proj_qkv_weight: str
    in_proj_z_weight: str
    out_proj_weight: str
    linear_norm_weight: str
    a_log: str
    dt_bias: str
    in_proj_a_weight: str
    in_proj_b_weight: str
    conv1d_weight: str
    post_attention_norm_weight: str


@dataclass(frozen=True)
class Qwen36LinearAttentionWeights:
    """Kernel-facing tensors for one real Qwen3.6 linear-attention layer."""

    input_norm_weight: Any
    in_proj_qkv_weight: Any
    in_proj_z_weight: Any
    out_proj_weight: Any
    linear_norm_weight: Any
    a_log: Any
    dt_bias: Any
    in_proj_a_weight: Any
    in_proj_b_weight: Any
    conv1d_weight: Any
    post_attention_norm_weight: Any
    keys: Qwen36LinearAttentionWeightKeys


@dataclass(frozen=True)
class Qwen36AttentionWeightKeys:
    """Resolved source tensor names for one Qwen3.6 full-attention layer."""

    input_norm_weight: str
    q_proj_weight: str
    k_proj_weight: str
    v_proj_weight: str
    o_proj_weight: str
    q_norm_weight: str
    k_norm_weight: str
    post_attention_norm_weight: str


@dataclass(frozen=True)
class Qwen36AttentionWeights:
    """Kernel-facing tensors for one real Qwen3.6 full-attention layer."""

    input_norm_weight: Any
    q_proj_weight: Any
    k_proj_weight: Any
    v_proj_weight: Any
    o_proj_weight: Any
    q_norm_weight: Any
    k_norm_weight: Any
    post_attention_norm_weight: Any
    keys: Qwen36AttentionWeightKeys


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required to pack Qwen3.6 MoE weights") from exc
    return torch


def _layer_bases(layer_idx: int, layer_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{prefix}.{layer_idx}" if prefix else str(layer_idx) for prefix in layer_prefixes)


def _first_existing(tensors: Mapping[str, Any], candidates: tuple[str, ...], label: str) -> str:
    for key in candidates:
        if key in tensors:
            return key
    joined = "\n  ".join(candidates[:16])
    raise KeyError(f"missing {label}; tried:\n  {joined}")


def _shape_tuple(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _check_shape(name: str, tensor: Any, expected: tuple[int, ...]) -> None:
    shape = _shape_tuple(tensor)
    if shape != expected:
        raise ValueError(f"{name} has shape {shape}, expected {expected}")


def _flatten_weight_keys(keys: Qwen36MoEWeightKeys) -> tuple[str, ...]:
    names = [keys.norm_weight, keys.router_weight, keys.shared_down_weight]
    if isinstance(keys.expert_down_weight, str):
        names.append(keys.expert_down_weight)
    else:
        names.extend(keys.expert_down_weight)
    if isinstance(keys.expert_gate_up_weight, str):
        names.append(keys.expert_gate_up_weight)
    else:
        for item in keys.expert_gate_up_weight:
            if isinstance(item, str):
                names.append(item)
            else:
                names.extend(item)
    if isinstance(keys.shared_gate_up_weight, str):
        names.append(keys.shared_gate_up_weight)
    else:
        names.extend(keys.shared_gate_up_weight)
    if keys.shared_expert_gate_weight is not None:
        names.append(keys.shared_expert_gate_weight)
    return tuple(names)


def _flatten_dataclass_keys(keys: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in keys.__dict__.values())


def resolve_qwen36_moe_weight_keys(
    tensors: Mapping[str, Any],
    layer_idx: int,
    *,
    spec: Qwen36A3BSpec | None = None,
    layer_prefixes: tuple[str, ...] = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "transformer.layers",
        "layers",
    ),
) -> Qwen36MoEWeightKeys:
    """Resolve likely HF-style tensor names for one Qwen3.6 MoE layer."""

    spec = spec or qwen36_35b_a3b_spec()
    bases = _layer_bases(layer_idx, layer_prefixes)
    norm_key = _first_existing(
        tensors,
        tuple(
            f"{base}.{suffix}"
            for base in bases
            for suffix in (
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "mlp_norm.weight",
                "input_layernorm.weight",
            )
        ),
        "MoE input norm weight",
    )
    router_key = _first_existing(
        tensors,
        tuple(
            f"{base}.mlp.{suffix}"
            for base in bases
            for suffix in (
                "gate.weight",
                "router.weight",
                "router.linear.weight",
            )
        ),
        "router weight",
    )

    grouped_gate_up_candidates = tuple(
        f"{base}.mlp.experts.{suffix}"
        for base in bases
        for suffix in (
            "gate_up_proj",
            "gate_up_proj.weight",
            "gate_up",
            "gate_up.weight",
            "w1",
            "w1.weight",
        )
    )
    grouped_down_candidates = tuple(
        f"{base}.mlp.experts.{suffix}"
        for base in bases
        for suffix in (
            "down_proj",
            "down_proj.weight",
            "down",
            "down.weight",
            "w2",
            "w2.weight",
        )
    )
    grouped_gate_up = next((key for key in grouped_gate_up_candidates if key in tensors), None)
    grouped_down = next((key for key in grouped_down_candidates if key in tensors), None)
    if grouped_gate_up is not None and grouped_down is not None:
        expert_gate_up: tuple[tuple[str, str] | str, ...] | str = grouped_gate_up
        expert_down: tuple[str, ...] | str = grouped_down
    else:
        expert_gate_up_keys: list[tuple[str, str] | str] = []
        expert_down_keys: list[str] = []
        for expert_idx in range(spec.num_experts):
            packed_candidates = tuple(
                f"{base}.mlp.experts.{expert_idx}.{suffix}"
                for base in bases
                for suffix in (
                    "gate_up_proj.weight",
                    "gate_up.weight",
                    "w1.weight",
                )
            )
            packed_key = next((key for key in packed_candidates if key in tensors), None)
            if packed_key is not None:
                expert_gate_up_keys.append(packed_key)
            else:
                gate_key = _first_existing(
                    tensors,
                    tuple(
                        f"{base}.mlp.experts.{expert_idx}.{suffix}"
                        for base in bases
                        for suffix in ("gate_proj.weight", "gate.weight", "w1_gate.weight")
                    ),
                    f"expert {expert_idx} gate projection",
                )
                up_key = _first_existing(
                    tensors,
                    tuple(
                        f"{base}.mlp.experts.{expert_idx}.{suffix}"
                        for base in bases
                        for suffix in ("up_proj.weight", "up.weight", "w1_up.weight")
                    ),
                    f"expert {expert_idx} up projection",
                )
                expert_gate_up_keys.append((gate_key, up_key))

            down_key = _first_existing(
                tensors,
                tuple(
                    f"{base}.mlp.experts.{expert_idx}.{suffix}"
                    for base in bases
                    for suffix in ("down_proj.weight", "down.weight", "w2.weight")
                ),
                f"expert {expert_idx} down projection",
            )
            expert_down_keys.append(down_key)
        expert_gate_up = tuple(expert_gate_up_keys)
        expert_down = tuple(expert_down_keys)

    shared_packed_candidates = tuple(
        f"{base}.mlp.{shared}.{suffix}"
        for base in bases
        for shared in ("shared_expert", "shared_experts.0")
        for suffix in ("gate_up_proj.weight", "gate_up.weight", "w1.weight")
    )
    shared_packed = next((key for key in shared_packed_candidates if key in tensors), None)
    if shared_packed is not None:
        shared_gate_up: tuple[str, str] | str = shared_packed
    else:
        shared_gate = _first_existing(
            tensors,
            tuple(
                f"{base}.mlp.{shared}.{suffix}"
                for base in bases
                for shared in ("shared_expert", "shared_experts.0")
                for suffix in ("gate_proj.weight", "gate.weight", "w1_gate.weight")
            ),
            "shared expert gate projection",
        )
        shared_up = _first_existing(
            tensors,
            tuple(
                f"{base}.mlp.{shared}.{suffix}"
                for base in bases
                for shared in ("shared_expert", "shared_experts.0")
                for suffix in ("up_proj.weight", "up.weight", "w1_up.weight")
            ),
            "shared expert up projection",
        )
        shared_gate_up = (shared_gate, shared_up)

    shared_down = _first_existing(
        tensors,
        tuple(
            f"{base}.mlp.{shared}.{suffix}"
            for base in bases
            for shared in ("shared_expert", "shared_experts.0")
            for suffix in ("down_proj.weight", "down.weight", "w2.weight")
        ),
        "shared expert down projection",
    )
    shared_expert_gate = next(
        (
            key
            for key in tuple(
                f"{base}.mlp.shared_expert_gate.{suffix}"
                for base in bases
                for suffix in ("weight",)
            )
            if key in tensors
        ),
        None,
    )

    return Qwen36MoEWeightKeys(
        norm_weight=norm_key,
        router_weight=router_key,
        expert_gate_up_weight=expert_gate_up,
        expert_down_weight=expert_down,
        shared_gate_up_weight=shared_gate_up,
        shared_down_weight=shared_down,
        shared_expert_gate_weight=shared_expert_gate,
    )


def resolve_qwen36_linear_attention_weight_keys(
    tensors: Mapping[str, Any],
    layer_idx: int,
    *,
    layer_prefixes: tuple[str, ...] = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "transformer.layers",
        "layers",
    ),
) -> Qwen36LinearAttentionWeightKeys:
    """Resolve HF-style tensor names for one real Qwen3.6 linear-attention layer."""

    bases = _layer_bases(layer_idx, layer_prefixes)

    def first(suffixes: tuple[str, ...], label: str) -> str:
        return _first_existing(tensors, tuple(f"{base}.{suffix}" for base in bases for suffix in suffixes), label)

    return Qwen36LinearAttentionWeightKeys(
        input_norm_weight=first(("input_layernorm.weight",), "linear-attention input norm weight"),
        in_proj_qkv_weight=first(("linear_attn.in_proj_qkv.weight",), "linear-attention packed QKV projection"),
        in_proj_z_weight=first(("linear_attn.in_proj_z.weight",), "linear-attention Z projection"),
        out_proj_weight=first(("linear_attn.out_proj.weight",), "linear-attention output projection"),
        linear_norm_weight=first(("linear_attn.norm.weight",), "linear-attention inner norm weight"),
        a_log=first(("linear_attn.A_log",), "linear-attention A_log"),
        dt_bias=first(("linear_attn.dt_bias",), "linear-attention dt_bias"),
        in_proj_a_weight=first(("linear_attn.in_proj_a.weight",), "linear-attention A projection"),
        in_proj_b_weight=first(("linear_attn.in_proj_b.weight",), "linear-attention B projection"),
        conv1d_weight=first(("linear_attn.conv1d.weight",), "linear-attention conv1d weight"),
        post_attention_norm_weight=first(("post_attention_layernorm.weight",), "post-attention norm weight"),
    )


def resolve_qwen36_attention_weight_keys(
    tensors: Mapping[str, Any],
    layer_idx: int,
    *,
    layer_prefixes: tuple[str, ...] = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "transformer.layers",
        "layers",
    ),
) -> Qwen36AttentionWeightKeys:
    """Resolve HF-style tensor names for one real Qwen3.6 full-attention layer."""

    bases = _layer_bases(layer_idx, layer_prefixes)

    def first(suffixes: tuple[str, ...], label: str) -> str:
        return _first_existing(tensors, tuple(f"{base}.{suffix}" for base in bases for suffix in suffixes), label)

    return Qwen36AttentionWeightKeys(
        input_norm_weight=first(("input_layernorm.weight",), "attention input norm weight"),
        q_proj_weight=first(("self_attn.q_proj.weight",), "attention Q projection"),
        k_proj_weight=first(("self_attn.k_proj.weight",), "attention K projection"),
        v_proj_weight=first(("self_attn.v_proj.weight",), "attention V projection"),
        o_proj_weight=first(("self_attn.o_proj.weight",), "attention output projection"),
        q_norm_weight=first(("self_attn.q_norm.weight",), "attention Q norm"),
        k_norm_weight=first(("self_attn.k_norm.weight",), "attention K norm"),
        post_attention_norm_weight=first(("post_attention_layernorm.weight",), "post-attention norm weight"),
    )


def pack_qwen36_moe_weights_from_state_dict(
    tensors: Mapping[str, Any],
    layer_idx: int,
    *,
    spec: Qwen36A3BSpec | None = None,
    layer_prefixes: tuple[str, ...] = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "transformer.layers",
        "layers",
    ),
    device: str | None = None,
) -> Qwen36MoEWeights:
    """Pack one Qwen3.6 MoE layer into the kernel-facing tensor layout."""

    torch = _require_torch()
    spec = spec or qwen36_35b_a3b_spec()
    keys = resolve_qwen36_moe_weight_keys(tensors, layer_idx, spec=spec, layer_prefixes=layer_prefixes)

    def tensor(key: str):
        value = tensors[key]
        return value.to(device) if device is not None else value

    norm_weight = tensor(keys.norm_weight)
    router_weight = tensor(keys.router_weight)
    _check_shape(keys.norm_weight, norm_weight, (spec.hidden_size,))
    _check_shape(keys.router_weight, router_weight, (spec.num_experts, spec.hidden_size))

    if isinstance(keys.expert_gate_up_weight, str):
        expert_gate_up_weight = tensor(keys.expert_gate_up_weight)
        gate_up_shape = _shape_tuple(expert_gate_up_weight)
        canonical_gate_up_shape = (spec.num_experts, spec.gated_up_features, spec.hidden_size)
        transposed_gate_up_shape = (spec.num_experts, spec.hidden_size, spec.gated_up_features)
        if gate_up_shape != canonical_gate_up_shape and gate_up_shape == transposed_gate_up_shape:
            expert_gate_up_weight = expert_gate_up_weight.transpose(1, 2)
        _check_shape(
            keys.expert_gate_up_weight,
            expert_gate_up_weight,
            canonical_gate_up_shape,
        )
        expert_gate_up_weight = expert_gate_up_weight.contiguous()
    else:
        expert_gate_up_parts = []
        for item in keys.expert_gate_up_weight:
            if isinstance(item, str):
                packed = tensor(item)
                _check_shape(item, packed, (spec.gated_up_features, spec.hidden_size))
                expert_gate_up_parts.append(packed)
            else:
                gate_key, up_key = item
                gate = tensor(gate_key)
                up = tensor(up_key)
                _check_shape(gate_key, gate, (spec.expert_intermediate_size, spec.hidden_size))
                _check_shape(up_key, up, (spec.expert_intermediate_size, spec.hidden_size))
                expert_gate_up_parts.append(torch.cat((gate, up), dim=0))
        expert_gate_up_weight = torch.stack(expert_gate_up_parts, dim=0).contiguous()

    if isinstance(keys.expert_down_weight, str):
        expert_down_weight = tensor(keys.expert_down_weight)
        down_shape = _shape_tuple(expert_down_weight)
        canonical_down_shape = (spec.num_experts, spec.hidden_size, spec.expert_intermediate_size)
        transposed_down_shape = (spec.num_experts, spec.expert_intermediate_size, spec.hidden_size)
        if down_shape != canonical_down_shape and down_shape == transposed_down_shape:
            expert_down_weight = expert_down_weight.transpose(1, 2)
        _check_shape(
            keys.expert_down_weight,
            expert_down_weight,
            canonical_down_shape,
        )
        expert_down_weight = expert_down_weight.contiguous()
    else:
        expert_down_parts = []
        for down_key in keys.expert_down_weight:
            down = tensor(down_key)
            _check_shape(down_key, down, (spec.hidden_size, spec.expert_intermediate_size))
            expert_down_parts.append(down)
        expert_down_weight = torch.stack(expert_down_parts, dim=0).contiguous()

    if isinstance(keys.shared_gate_up_weight, str):
        shared_gate_up_weight = tensor(keys.shared_gate_up_weight)
        _check_shape(keys.shared_gate_up_weight, shared_gate_up_weight, (spec.gated_up_features, spec.hidden_size))
    else:
        shared_gate_key, shared_up_key = keys.shared_gate_up_weight
        shared_gate = tensor(shared_gate_key)
        shared_up = tensor(shared_up_key)
        _check_shape(shared_gate_key, shared_gate, (spec.expert_intermediate_size, spec.hidden_size))
        _check_shape(shared_up_key, shared_up, (spec.expert_intermediate_size, spec.hidden_size))
        shared_gate_up_weight = torch.cat((shared_gate, shared_up), dim=0).contiguous()

    shared_down_weight = tensor(keys.shared_down_weight)
    _check_shape(keys.shared_down_weight, shared_down_weight, (spec.hidden_size, spec.expert_intermediate_size))
    shared_expert_gate_weight = None
    if keys.shared_expert_gate_weight is not None:
        shared_expert_gate_weight = tensor(keys.shared_expert_gate_weight).contiguous()
        shape = _shape_tuple(shared_expert_gate_weight)
        if shape not in ((spec.hidden_size,), (1, spec.hidden_size)):
            raise ValueError(
                f"{keys.shared_expert_gate_weight} has shape {shape}, expected "
                f"({spec.hidden_size},) or (1, {spec.hidden_size})"
            )

    return Qwen36MoEWeights(
        norm_weight=norm_weight.contiguous(),
        router_weight=router_weight.contiguous(),
        expert_gate_up_weight=expert_gate_up_weight,
        expert_down_weight=expert_down_weight,
        shared_gate_up_weight=shared_gate_up_weight.contiguous(),
        shared_down_weight=shared_down_weight.contiguous(),
        keys=keys,
        shared_expert_gate_weight=shared_expert_gate_weight,
    )


def pack_qwen36_linear_attention_weights_from_state_dict(
    tensors: Mapping[str, Any],
    layer_idx: int,
    *,
    spec: Qwen36A3BSpec | None = None,
    layer_prefixes: tuple[str, ...] = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "transformer.layers",
        "layers",
    ),
    device: str | None = None,
) -> Qwen36LinearAttentionWeights:
    """Pack one real Qwen3.6 linear-attention layer into a validated layout."""

    spec = spec or qwen36_35b_a3b_spec()
    keys = resolve_qwen36_linear_attention_weight_keys(tensors, layer_idx, layer_prefixes=layer_prefixes)

    def tensor(key: str):
        value = tensors[key]
        return value.to(device) if device is not None else value

    input_norm_weight = tensor(keys.input_norm_weight).contiguous()
    in_proj_qkv_weight = tensor(keys.in_proj_qkv_weight).contiguous()
    in_proj_z_weight = tensor(keys.in_proj_z_weight).contiguous()
    out_proj_weight = tensor(keys.out_proj_weight).contiguous()
    linear_norm_weight = tensor(keys.linear_norm_weight).contiguous()
    a_log = tensor(keys.a_log).contiguous()
    dt_bias = tensor(keys.dt_bias).contiguous()
    in_proj_a_weight = tensor(keys.in_proj_a_weight).contiguous()
    in_proj_b_weight = tensor(keys.in_proj_b_weight).contiguous()
    conv1d_weight = tensor(keys.conv1d_weight).contiguous()
    post_attention_norm_weight = tensor(keys.post_attention_norm_weight).contiguous()

    value_width = spec.deltanet_value_heads * spec.deltanet_head_dim
    qk_width = spec.deltanet_qk_heads * spec.deltanet_head_dim
    _check_shape(keys.input_norm_weight, input_norm_weight, (spec.hidden_size,))
    _check_shape(keys.in_proj_qkv_weight, in_proj_qkv_weight, (2 * qk_width + value_width, spec.hidden_size))
    _check_shape(keys.in_proj_z_weight, in_proj_z_weight, (value_width, spec.hidden_size))
    _check_shape(keys.out_proj_weight, out_proj_weight, (spec.hidden_size, value_width))
    _check_shape(keys.linear_norm_weight, linear_norm_weight, (spec.deltanet_head_dim,))
    _check_shape(keys.a_log, a_log, (spec.deltanet_value_heads,))
    _check_shape(keys.dt_bias, dt_bias, (spec.deltanet_value_heads,))
    _check_shape(keys.in_proj_a_weight, in_proj_a_weight, (spec.deltanet_value_heads, spec.hidden_size))
    _check_shape(keys.in_proj_b_weight, in_proj_b_weight, (spec.deltanet_value_heads, spec.hidden_size))
    if conv1d_weight.ndim != 3 or conv1d_weight.shape[0] != 2 * qk_width + value_width:
        raise ValueError(
            f"{keys.conv1d_weight} has shape {_shape_tuple(conv1d_weight)}, expected "
            f"({2 * qk_width + value_width}, *, *)"
        )
    _check_shape(keys.post_attention_norm_weight, post_attention_norm_weight, (spec.hidden_size,))

    return Qwen36LinearAttentionWeights(
        input_norm_weight=input_norm_weight,
        in_proj_qkv_weight=in_proj_qkv_weight,
        in_proj_z_weight=in_proj_z_weight,
        out_proj_weight=out_proj_weight,
        linear_norm_weight=linear_norm_weight,
        a_log=a_log,
        dt_bias=dt_bias,
        in_proj_a_weight=in_proj_a_weight,
        in_proj_b_weight=in_proj_b_weight,
        conv1d_weight=conv1d_weight,
        post_attention_norm_weight=post_attention_norm_weight,
        keys=keys,
    )


def pack_qwen36_attention_weights_from_state_dict(
    tensors: Mapping[str, Any],
    layer_idx: int,
    *,
    spec: Qwen36A3BSpec | None = None,
    layer_prefixes: tuple[str, ...] = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "transformer.layers",
        "layers",
    ),
    device: str | None = None,
) -> Qwen36AttentionWeights:
    """Pack one real Qwen3.6 full-attention layer into a validated layout."""

    spec = spec or qwen36_35b_a3b_spec()
    keys = resolve_qwen36_attention_weight_keys(tensors, layer_idx, layer_prefixes=layer_prefixes)

    def tensor(key: str):
        value = tensors[key]
        return value.to(device) if device is not None else value

    input_norm_weight = tensor(keys.input_norm_weight).contiguous()
    q_proj_weight = tensor(keys.q_proj_weight).contiguous()
    k_proj_weight = tensor(keys.k_proj_weight).contiguous()
    v_proj_weight = tensor(keys.v_proj_weight).contiguous()
    o_proj_weight = tensor(keys.o_proj_weight).contiguous()
    q_norm_weight = tensor(keys.q_norm_weight).contiguous()
    k_norm_weight = tensor(keys.k_norm_weight).contiguous()
    post_attention_norm_weight = tensor(keys.post_attention_norm_weight).contiguous()

    kv_width = spec.attention_kv_heads * spec.attention_head_dim
    _check_shape(keys.input_norm_weight, input_norm_weight, (spec.hidden_size,))
    if q_proj_weight.ndim != 2 or q_proj_weight.shape[1] != spec.hidden_size:
        raise ValueError(f"{keys.q_proj_weight} has shape {_shape_tuple(q_proj_weight)}, expected (*, {spec.hidden_size})")
    _check_shape(keys.k_proj_weight, k_proj_weight, (kv_width, spec.hidden_size))
    _check_shape(keys.v_proj_weight, v_proj_weight, (kv_width, spec.hidden_size))
    if o_proj_weight.ndim != 2 or o_proj_weight.shape[0] != spec.hidden_size:
        raise ValueError(f"{keys.o_proj_weight} has shape {_shape_tuple(o_proj_weight)}, expected ({spec.hidden_size}, *)")
    _check_shape(keys.q_norm_weight, q_norm_weight, (spec.attention_head_dim,))
    _check_shape(keys.k_norm_weight, k_norm_weight, (spec.attention_head_dim,))
    _check_shape(keys.post_attention_norm_weight, post_attention_norm_weight, (spec.hidden_size,))

    return Qwen36AttentionWeights(
        input_norm_weight=input_norm_weight,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        post_attention_norm_weight=post_attention_norm_weight,
        keys=keys,
    )


def load_qwen36_moe_weights_from_safetensors(
    path: str | Path,
    layer_idx: int,
    *,
    spec: Qwen36A3BSpec | None = None,
    layer_prefixes: tuple[str, ...] = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "transformer.layers",
        "layers",
    ),
    device: str | None = None,
) -> Qwen36MoEWeights:
    """Load and pack one Qwen3.6 MoE layer from a safetensors file or directory."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("safetensors is required to load Qwen3.6 MoE weights") from exc

    source = Path(path)
    if source.is_dir():
        index_files = sorted(source.glob("*.safetensors.index.json"))
        if index_files:
            with index_files[0].open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map", {})
            if not isinstance(weight_map, dict):
                raise ValueError(f"{index_files[0]} does not contain a valid weight_map")
            key_index = {key: None for key in weight_map}
            spec = spec or qwen36_35b_a3b_spec()
            keys = resolve_qwen36_moe_weight_keys(
                key_index,
                layer_idx,
                spec=spec,
                layer_prefixes=layer_prefixes,
            )
            needed_keys = set(_flatten_weight_keys(keys))
            missing_keys = sorted(key for key in needed_keys if key not in weight_map)
            if missing_keys:
                raise KeyError(f"safetensors index is missing resolved keys: {missing_keys[:8]}")
            files = sorted({source / str(weight_map[key]) for key in needed_keys})
        else:
            files = sorted(source.glob("*.safetensors"))
    else:
        files = [source]
    if not files:
        raise FileNotFoundError(f"no safetensors files found at {source}")

    tensors: dict[str, Any] = {}
    for file_path in files:
        tensors.update(load_file(str(file_path), device="cpu"))

    return pack_qwen36_moe_weights_from_state_dict(
        tensors,
        layer_idx,
        spec=spec,
        layer_prefixes=layer_prefixes,
        device=device,
    )


def _load_selected_safetensors(
    path: str | Path,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("safetensors is required to load Qwen3.6 weights") from exc

    source = Path(path)
    if source.is_dir():
        index_files = sorted(source.glob("*.safetensors.index.json"))
        if index_files:
            with index_files[0].open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map", {})
            if not isinstance(weight_map, dict):
                raise ValueError(f"{index_files[0]} does not contain a valid weight_map")
            files = sorted({source / str(weight_map[key]) for key in keys})
        else:
            files = sorted(source.glob("*.safetensors"))
    else:
        files = [source]
    if not files:
        raise FileNotFoundError(f"no safetensors files found at {source}")

    selected = set(keys)
    tensors: dict[str, Any] = {}
    for file_path in files:
        loaded = load_file(str(file_path), device="cpu")
        if selected:
            tensors.update({key: value for key, value in loaded.items() if key in selected})
        else:
            tensors.update(loaded)
    missing = sorted(selected - set(tensors))
    if missing:
        raise KeyError(f"missing tensors after safetensors load: {missing[:8]}")
    return tensors


def load_qwen36_linear_attention_weights_from_safetensors(
    path: str | Path,
    layer_idx: int,
    *,
    spec: Qwen36A3BSpec | None = None,
    layer_prefixes: tuple[str, ...] = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "transformer.layers",
        "layers",
    ),
    device: str | None = None,
) -> Qwen36LinearAttentionWeights:
    """Load and pack one real Qwen3.6 linear-attention layer from safetensors."""

    source = Path(path)
    if source.is_dir():
        index_files = sorted(source.glob("*.safetensors.index.json"))
        if index_files:
            with index_files[0].open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map", {})
            if not isinstance(weight_map, dict):
                raise ValueError(f"{index_files[0]} does not contain a valid weight_map")
            key_index = {key: None for key in weight_map}
            keys = resolve_qwen36_linear_attention_weight_keys(
                key_index,
                layer_idx,
                layer_prefixes=layer_prefixes,
            )
            tensors = _load_selected_safetensors(source, _flatten_dataclass_keys(keys))
            return pack_qwen36_linear_attention_weights_from_state_dict(
                tensors,
                layer_idx,
                spec=spec,
                layer_prefixes=layer_prefixes,
                device=device,
            )
    tensors = _load_selected_safetensors(source, tuple())
    return pack_qwen36_linear_attention_weights_from_state_dict(
        tensors,
        layer_idx,
        spec=spec,
        layer_prefixes=layer_prefixes,
        device=device,
    )


def load_qwen36_attention_weights_from_safetensors(
    path: str | Path,
    layer_idx: int,
    *,
    spec: Qwen36A3BSpec | None = None,
    layer_prefixes: tuple[str, ...] = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "transformer.layers",
        "layers",
    ),
    device: str | None = None,
) -> Qwen36AttentionWeights:
    """Load and pack one real Qwen3.6 full-attention layer from safetensors."""

    source = Path(path)
    if source.is_dir():
        index_files = sorted(source.glob("*.safetensors.index.json"))
        if index_files:
            with index_files[0].open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map", {})
            if not isinstance(weight_map, dict):
                raise ValueError(f"{index_files[0]} does not contain a valid weight_map")
            key_index = {key: None for key in weight_map}
            keys = resolve_qwen36_attention_weight_keys(key_index, layer_idx, layer_prefixes=layer_prefixes)
            tensors = _load_selected_safetensors(source, _flatten_dataclass_keys(keys))
            return pack_qwen36_attention_weights_from_state_dict(
                tensors,
                layer_idx,
                spec=spec,
                layer_prefixes=layer_prefixes,
                device=device,
            )
    tensors = _load_selected_safetensors(source, tuple())
    return pack_qwen36_attention_weights_from_state_dict(
        tensors,
        layer_idx,
        spec=spec,
        layer_prefixes=layer_prefixes,
        device=device,
    )

"""Full-model planning helpers for real Qwen3.6 weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .qwen36 import Qwen36A3BSpec, qwen36_35b_a3b_spec
from .qwen36_weights import (
    Qwen36AttentionWeightKeys,
    Qwen36LinearAttentionWeightKeys,
    Qwen36MoEWeightKeys,
    resolve_qwen36_attention_weight_keys,
    resolve_qwen36_linear_attention_weight_keys,
    resolve_qwen36_moe_weight_keys,
)


@dataclass(frozen=True)
class Qwen36RootWeightKeys:
    """Resolved source tensor names outside the per-layer stack."""

    embedding_weight: str
    output_norm_weight: str
    lm_head_weight: str


@dataclass(frozen=True)
class Qwen36LayerWeightPlan:
    """Resolved source tensor names for one full Qwen3.6 layer."""

    layer_idx: int
    layer_kind: str
    moe: Qwen36MoEWeightKeys
    linear_attention: Qwen36LinearAttentionWeightKeys | None = None
    attention: Qwen36AttentionWeightKeys | None = None

    def keys(self) -> tuple[str, ...]:
        names = []
        if self.linear_attention is not None:
            names.extend(_flatten_dataclass_keys(self.linear_attention))
        if self.attention is not None:
            names.extend(_flatten_dataclass_keys(self.attention))
        names.extend(flatten_qwen36_moe_weight_keys(self.moe))
        return tuple(dict.fromkeys(names))


@dataclass(frozen=True)
class Qwen36FullWeightPlan:
    """Resolved tensor plan for a complete Qwen3.6 decode model."""

    spec: Qwen36A3BSpec
    roots: Qwen36RootWeightKeys
    layers: tuple[Qwen36LayerWeightPlan, ...]

    def keys(self) -> tuple[str, ...]:
        names = [
            self.roots.embedding_weight,
            self.roots.output_norm_weight,
            self.roots.lm_head_weight,
        ]
        for layer in self.layers:
            names.extend(layer.keys())
        return tuple(dict.fromkeys(names))

    def required_shards(self, weight_map: Mapping[str, str]) -> tuple[str, ...]:
        return tuple(sorted({str(weight_map[key]) for key in self.keys()}))

    def layer_counts(self) -> dict[str, int]:
        return {
            "deltanet_moe": sum(layer.layer_kind == "deltanet_moe" for layer in self.layers),
            "attention_moe": sum(layer.layer_kind == "attention_moe" for layer in self.layers),
        }


def _first_existing(tensors: Mapping[str, object], candidates: tuple[str, ...], label: str) -> str:
    for key in candidates:
        if key in tensors:
            return key
    joined = "\n  ".join(candidates)
    raise KeyError(f"missing {label}; tried:\n  {joined}")


def _flatten_dataclass_keys(keys: object) -> tuple[str, ...]:
    return tuple(str(value) for value in keys.__dict__.values())


def flatten_qwen36_moe_weight_keys(keys: Qwen36MoEWeightKeys) -> tuple[str, ...]:
    """Return all tensor names referenced by resolved MoE keys."""

    names = [keys.norm_weight, keys.router_weight, keys.shared_down_weight]
    if isinstance(keys.expert_gate_up_weight, str):
        names.append(keys.expert_gate_up_weight)
    else:
        for item in keys.expert_gate_up_weight:
            if isinstance(item, str):
                names.append(item)
            else:
                names.extend(item)
    if isinstance(keys.expert_down_weight, str):
        names.append(keys.expert_down_weight)
    else:
        names.extend(keys.expert_down_weight)
    if isinstance(keys.shared_gate_up_weight, str):
        names.append(keys.shared_gate_up_weight)
    else:
        names.extend(keys.shared_gate_up_weight)
    if keys.shared_expert_gate_weight is not None:
        names.append(keys.shared_expert_gate_weight)
    return tuple(names)


def resolve_qwen36_root_weight_keys(
    tensors: Mapping[str, object],
    *,
    prefixes: tuple[str, ...] = (
        "model.language_model",
        "model",
        "language_model.model",
        "transformer",
        "",
    ),
) -> Qwen36RootWeightKeys:
    """Resolve embedding, final norm, and LM head names for Qwen3.6 layouts."""

    def candidates(suffix: str) -> tuple[str, ...]:
        return tuple(f"{prefix}.{suffix}" if prefix else suffix for prefix in prefixes)

    embedding = _first_existing(
        tensors,
        candidates("embed_tokens.weight") + candidates("wte.weight"),
        "token embedding weight",
    )
    output_norm = _first_existing(
        tensors,
        candidates("norm.weight") + candidates("final_layernorm.weight"),
        "final norm weight",
    )
    lm_head = _first_existing(
        tensors,
        (
            "lm_head.weight",
            "model.lm_head.weight",
            "model.language_model.lm_head.weight",
            "language_model.lm_head.weight",
        )
        + (embedding,),
        "LM head weight",
    )
    return Qwen36RootWeightKeys(
        embedding_weight=embedding,
        output_norm_weight=output_norm,
        lm_head_weight=lm_head,
    )


def resolve_qwen36_full_weight_plan(
    tensors: Mapping[str, object],
    *,
    spec: Qwen36A3BSpec | None = None,
) -> Qwen36FullWeightPlan:
    """Resolve all tensor names needed by a full 40-layer Qwen3.6 decode pass."""

    spec = spec or qwen36_35b_a3b_spec()
    roots = resolve_qwen36_root_weight_keys(tensors)
    layers = []
    for layer_idx, layer_kind in enumerate(spec.layer_kinds()):
        moe = resolve_qwen36_moe_weight_keys(tensors, layer_idx, spec=spec)
        if layer_kind == "deltanet_moe":
            layers.append(
                Qwen36LayerWeightPlan(
                    layer_idx=layer_idx,
                    layer_kind=layer_kind,
                    linear_attention=resolve_qwen36_linear_attention_weight_keys(tensors, layer_idx),
                    moe=moe,
                )
            )
        elif layer_kind == "attention_moe":
            layers.append(
                Qwen36LayerWeightPlan(
                    layer_idx=layer_idx,
                    layer_kind=layer_kind,
                    attention=resolve_qwen36_attention_weight_keys(tensors, layer_idx),
                    moe=moe,
                )
            )
        else:
            raise ValueError(f"unknown Qwen3.6 layer kind at layer {layer_idx}: {layer_kind}")
    return Qwen36FullWeightPlan(spec=spec, roots=roots, layers=tuple(layers))

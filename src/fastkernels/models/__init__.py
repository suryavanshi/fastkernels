"""Model shape specifications used by fastkernels."""

from .qwen35 import Qwen35MoESpec, qwen35_35b_a3b_spec
from .qwen36 import Qwen36A3BSpec, qwen36_35b_a3b_spec, synthetic_qwen36_spec
from .qwen36_full import (
    Qwen36FullWeightPlan,
    Qwen36LayerWeightPlan,
    Qwen36RootWeightKeys,
    flatten_qwen36_moe_weight_keys,
    resolve_qwen36_full_weight_plan,
    resolve_qwen36_root_weight_keys,
)
from .qwen36_weights import (
    Qwen36AttentionWeightKeys,
    Qwen36AttentionWeights,
    Qwen36LinearAttentionWeightKeys,
    Qwen36LinearAttentionWeights,
    Qwen36MoEWeightKeys,
    Qwen36MoEWeights,
    load_qwen36_attention_weights_from_safetensors,
    load_qwen36_linear_attention_weights_from_safetensors,
    load_qwen36_moe_weights_from_safetensors,
    pack_qwen36_attention_weights_from_state_dict,
    pack_qwen36_linear_attention_weights_from_state_dict,
    pack_qwen36_moe_weights_from_state_dict,
    resolve_qwen36_attention_weight_keys,
    resolve_qwen36_linear_attention_weight_keys,
    resolve_qwen36_moe_weight_keys,
)

__all__ = [
    "Qwen35MoESpec",
    "Qwen36A3BSpec",
    "Qwen36AttentionWeightKeys",
    "Qwen36AttentionWeights",
    "Qwen36FullWeightPlan",
    "Qwen36LayerWeightPlan",
    "Qwen36LinearAttentionWeightKeys",
    "Qwen36LinearAttentionWeights",
    "Qwen36MoEWeightKeys",
    "Qwen36MoEWeights",
    "Qwen36RootWeightKeys",
    "flatten_qwen36_moe_weight_keys",
    "load_qwen36_attention_weights_from_safetensors",
    "load_qwen36_linear_attention_weights_from_safetensors",
    "load_qwen36_moe_weights_from_safetensors",
    "pack_qwen36_attention_weights_from_state_dict",
    "pack_qwen36_linear_attention_weights_from_state_dict",
    "pack_qwen36_moe_weights_from_state_dict",
    "qwen35_35b_a3b_spec",
    "qwen36_35b_a3b_spec",
    "resolve_qwen36_attention_weight_keys",
    "resolve_qwen36_full_weight_plan",
    "resolve_qwen36_linear_attention_weight_keys",
    "resolve_qwen36_moe_weight_keys",
    "resolve_qwen36_root_weight_keys",
    "synthetic_qwen36_spec",
]

"""Reference implementations used for custom-kernel correctness checks."""

from .moe import reference_expert_histogram, reference_fused_swiglu, reference_routed_moe
from .qwen36_decode import (
    Qwen36DecodeState,
    initial_qwen36_decode_state,
    make_synthetic_qwen36_decode_weights,
    reference_qwen36_attention_decode,
    reference_qwen36_decode_step,
    reference_qwen36_deltanet_decode,
    reference_qwen36_moe_decode,
)
from .qwen36_real import (
    Qwen36RealDecodeState,
    initial_qwen36_real_decode_state,
    qwen36_real_attention_moe_layer,
    qwen36_real_attention_update,
    qwen36_real_deltanet_moe_layer,
    qwen36_real_deltanet_update,
    qwen36_real_deltanet_update_from_convolved_projections,
    qwen36_real_deltanet_update_from_projections,
    qwen36_real_moe_update,
    qwen36_real_rms_norm,
)

__all__ = [
    "Qwen36DecodeState",
    "initial_qwen36_decode_state",
    "make_synthetic_qwen36_decode_weights",
    "reference_expert_histogram",
    "reference_fused_swiglu",
    "reference_qwen36_attention_decode",
    "reference_qwen36_decode_step",
    "reference_qwen36_deltanet_decode",
    "reference_qwen36_moe_decode",
    "reference_routed_moe",
    "Qwen36RealDecodeState",
    "initial_qwen36_real_decode_state",
    "qwen36_real_attention_moe_layer",
    "qwen36_real_attention_update",
    "qwen36_real_deltanet_moe_layer",
    "qwen36_real_deltanet_update",
    "qwen36_real_deltanet_update_from_convolved_projections",
    "qwen36_real_deltanet_update_from_projections",
    "qwen36_real_moe_update",
    "qwen36_real_rms_norm",
]

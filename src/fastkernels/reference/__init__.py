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
]

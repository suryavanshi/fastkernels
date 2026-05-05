"""Optional Triton kernels."""

from .qwen36_attention import (
    triton_qwen36_batched_attention_decode,
    triton_qwen36_batched_attention_project,
    triton_synthetic_qwen36_attention_decode,
)
from .qwen36_deltanet import (
    triton_qwen36_batched_deltanet_conv,
    triton_qwen36_batched_deltanet_project,
    triton_qwen36_batched_deltanet_recurrent_output,
    triton_synthetic_qwen36_deltanet_decode,
)
from .qwen36_layer import (
    triton_qwen36_batched_attention_moe_layer_decode,
    triton_qwen36_batched_deltanet_moe_layer_decode,
    triton_qwen36_batched_moe_layer_decode,
    triton_synthetic_qwen36_attention_moe_decode,
    triton_synthetic_qwen36_deltanet_moe_decode,
)
from .moe import triton_expert_histogram, triton_fused_swiglu
from .qwen36_expert import (
    triton_qwen36_batched_moe_decode,
    triton_qwen36_batched_routed_experts_decode,
    triton_qwen36_batched_routed_shared_experts_decode,
    triton_qwen36_moe_decode,
    triton_qwen36_routed_experts_decode,
    triton_qwen36_routed_shared_experts_decode,
    triton_qwen36_single_expert_mlp_decode,
)
from .qwen36_moe import triton_synthetic_qwen36_moe_decode
from .qwen36_router import triton_qwen36_batched_moe_router_decode, triton_qwen36_moe_router_decode

__all__ = [
    "triton_expert_histogram",
    "triton_fused_swiglu",
    "triton_qwen36_batched_moe_decode",
    "triton_qwen36_batched_attention_decode",
    "triton_qwen36_batched_attention_moe_layer_decode",
    "triton_qwen36_batched_attention_project",
    "triton_qwen36_batched_deltanet_moe_layer_decode",
    "triton_qwen36_batched_deltanet_conv",
    "triton_qwen36_batched_deltanet_project",
    "triton_qwen36_batched_deltanet_recurrent_output",
    "triton_qwen36_batched_moe_layer_decode",
    "triton_qwen36_batched_moe_router_decode",
    "triton_qwen36_batched_routed_experts_decode",
    "triton_qwen36_batched_routed_shared_experts_decode",
    "triton_qwen36_moe_decode",
    "triton_qwen36_moe_router_decode",
    "triton_qwen36_routed_experts_decode",
    "triton_qwen36_routed_shared_experts_decode",
    "triton_qwen36_single_expert_mlp_decode",
    "triton_synthetic_qwen36_attention_decode",
    "triton_synthetic_qwen36_attention_moe_decode",
    "triton_synthetic_qwen36_deltanet_decode",
    "triton_synthetic_qwen36_deltanet_moe_decode",
    "triton_synthetic_qwen36_moe_decode",
]

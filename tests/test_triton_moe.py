from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from fastkernels.kernels.triton import (
    triton_expert_histogram,
    triton_fused_swiglu,
    triton_qwen36_batched_attention_decode,
    triton_qwen36_batched_attention_project,
    triton_qwen36_batched_deltanet_conv,
    triton_qwen36_batched_deltanet_moe_layer_decode,
    triton_qwen36_batched_deltanet_project,
    triton_qwen36_batched_deltanet_recurrent_output,
    triton_qwen36_batched_moe_layer_decode,
    triton_qwen36_batched_moe_decode,
    triton_qwen36_batched_moe_router_decode,
    triton_qwen36_batched_routed_experts_decode,
    triton_qwen36_batched_routed_shared_experts_decode,
    triton_qwen36_moe_decode,
    triton_qwen36_moe_router_decode,
    triton_qwen36_routed_experts_decode,
    triton_qwen36_routed_shared_experts_decode,
    triton_qwen36_single_expert_mlp_decode,
    triton_synthetic_qwen36_attention_decode,
    triton_synthetic_qwen36_attention_moe_decode,
    triton_synthetic_qwen36_deltanet_decode,
    triton_synthetic_qwen36_deltanet_moe_decode,
    triton_synthetic_qwen36_moe_decode,
)
from fastkernels.models import synthetic_qwen36_spec
from fastkernels.reference import (
    make_synthetic_qwen36_decode_weights,
    reference_expert_histogram,
    reference_fused_swiglu,
    reference_qwen36_attention_decode,
    reference_qwen36_deltanet_decode,
    reference_qwen36_moe_decode,
)
from fastkernels.testing import default_tolerance


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _zero_centered_rms_norm(values, weight):
    values_f = values.float()
    return (
        values_f
        * torch.rsqrt(torch.mean(values_f * values_f, dim=-1, keepdim=True) + 1e-6)
        * (1.0 + weight.float())
    )


def test_triton_fused_swiglu_matches_reference():
    torch.manual_seed(0)
    gate_up = torch.randn(128, 1024, device="cuda", dtype=torch.bfloat16)

    candidate = triton_fused_swiglu(gate_up)
    reference = reference_fused_swiglu(gate_up)

    atol, rtol = default_tolerance(gate_up.dtype)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


def test_triton_expert_histogram_matches_reference():
    torch.manual_seed(0)
    topk_ids = torch.randint(0, 256, (64, 8), device="cuda", dtype=torch.int32)

    candidate = triton_expert_histogram(topk_ids, 256)
    reference = reference_expert_histogram(topk_ids, 256)

    torch.testing.assert_close(candidate.cpu(), reference.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_qwen36_moe_router_decode_matches_real_shape_reference(dtype):
    torch.manual_seed(0)
    hidden_size = 2048
    num_experts = 256
    top_k = 8
    scale = 0.02
    hidden = torch.randn(hidden_size, device="cuda", dtype=dtype)
    norm_weight = scale * torch.randn(hidden_size, device="cuda", dtype=dtype)
    router_weight = scale * torch.randn(num_experts, hidden_size, device="cuda", dtype=dtype)

    logits, topk_ids, topk_weights = triton_qwen36_moe_router_decode(
        hidden,
        norm_weight,
        router_weight,
        top_k=top_k,
    )

    hidden_f = hidden.float()
    normed = _zero_centered_rms_norm(hidden_f, norm_weight)
    reference_logits = torch.matmul(router_weight.float(), normed)
    reference_probs = torch.softmax(reference_logits, dim=-1)
    reference_values, reference_ids = torch.topk(reference_probs, k=top_k)
    reference_weights = reference_values / torch.sum(reference_values)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(logits, reference_logits, atol=atol, rtol=rtol)
    torch.testing.assert_close(topk_ids.cpu(), reference_ids.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(topk_weights, reference_weights, atol=atol, rtol=rtol)


def test_triton_qwen36_batched_moe_router_decode_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 4
    hidden_size = 2048
    num_experts = 256
    top_k = 8
    scale = 0.02
    hidden = scale * torch.randn(tokens, hidden_size, device="cuda", dtype=dtype)
    norm_weight = scale * torch.randn(hidden_size, device="cuda", dtype=dtype)
    router_weight = scale * torch.randn(num_experts, hidden_size, device="cuda", dtype=dtype)

    logits, topk_ids, topk_weights = triton_qwen36_batched_moe_router_decode(
        hidden,
        norm_weight,
        router_weight,
        top_k=top_k,
    )

    hidden_f = hidden.float()
    normed = _zero_centered_rms_norm(hidden_f, norm_weight)
    reference_logits = torch.matmul(normed, router_weight.float().t())
    reference_probs = torch.softmax(logits, dim=-1)
    reference_values, reference_ids = torch.topk(reference_probs, k=top_k, dim=1)
    reference_weights = reference_values / torch.sum(reference_values, dim=1, keepdim=True)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(logits, reference_logits, atol=atol, rtol=rtol)
    torch.testing.assert_close(topk_ids.cpu(), reference_ids.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(topk_weights, reference_weights, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_qwen36_single_expert_mlp_decode_matches_real_shape_reference(dtype):
    torch.manual_seed(0)
    hidden_size = 2048
    intermediate = 512
    scale = 0.02
    hidden = scale * torch.randn(hidden_size, device="cuda", dtype=dtype)
    expert_gate_up_weight = scale * torch.randn(2 * intermediate, hidden_size, device="cuda", dtype=dtype)
    expert_down_weight = scale * torch.randn(hidden_size, intermediate, device="cuda", dtype=dtype)

    candidate = triton_qwen36_single_expert_mlp_decode(
        hidden,
        expert_gate_up_weight,
        expert_down_weight,
    )

    hidden_f = hidden.float()
    gate_up = torch.matmul(expert_gate_up_weight.float(), hidden_f)
    gate, up = torch.chunk(gate_up, 2, dim=0)
    activation = torch.nn.functional.silu(gate) * up
    reference = torch.matmul(expert_down_weight.float(), activation)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


def test_triton_qwen36_routed_shared_experts_decode_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    hidden_size = 2048
    intermediate = 512
    num_experts = 256
    top_k = 8
    scale = 0.02
    hidden = scale * torch.randn(hidden_size, device="cuda", dtype=dtype)
    expert_gate_up_weight = scale * torch.randn(
        num_experts,
        2 * intermediate,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    expert_down_weight = scale * torch.randn(
        num_experts,
        hidden_size,
        intermediate,
        device="cuda",
        dtype=dtype,
    )
    shared_gate_up_weight = scale * torch.randn(2 * intermediate, hidden_size, device="cuda", dtype=dtype)
    shared_down_weight = scale * torch.randn(hidden_size, intermediate, device="cuda", dtype=dtype)
    topk_ids = torch.tensor([3, 17, 42, 89, 127, 199, 231, 255], device="cuda", dtype=torch.int64)
    topk_weights = torch.softmax(torch.randn(top_k, device="cuda", dtype=torch.float32), dim=0)

    candidate = triton_qwen36_routed_shared_experts_decode(
        hidden,
        topk_ids,
        topk_weights,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
    )

    hidden_f = hidden.float()
    reference = torch.zeros(hidden_size, device="cuda", dtype=torch.float32)
    for idx in range(top_k):
        expert_id = int(topk_ids[idx].item())
        gate_up = torch.matmul(expert_gate_up_weight[expert_id].float(), hidden_f)
        gate, up = torch.chunk(gate_up, 2, dim=0)
        activation = torch.nn.functional.silu(gate) * up
        reference = reference + topk_weights[idx] * torch.matmul(expert_down_weight[expert_id].float(), activation)
    shared_gate_up = torch.matmul(shared_gate_up_weight.float(), hidden_f)
    shared_gate, shared_up = torch.chunk(shared_gate_up, 2, dim=0)
    shared_activation = torch.nn.functional.silu(shared_gate) * shared_up
    reference = reference + torch.matmul(shared_down_weight.float(), shared_activation)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


def test_triton_qwen36_routed_experts_decode_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    hidden_size = 2048
    intermediate = 512
    num_experts = 256
    top_k = 8
    scale = 0.02
    hidden = scale * torch.randn(hidden_size, device="cuda", dtype=dtype)
    expert_gate_up_weight = scale * torch.randn(
        num_experts,
        2 * intermediate,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    expert_down_weight = scale * torch.randn(
        num_experts,
        hidden_size,
        intermediate,
        device="cuda",
        dtype=dtype,
    )
    topk_ids = torch.tensor([3, 17, 42, 89, 127, 199, 231, 255], device="cuda", dtype=torch.int64)
    topk_weights = torch.softmax(torch.randn(top_k, device="cuda", dtype=torch.float32), dim=0)

    candidate = triton_qwen36_routed_experts_decode(
        hidden,
        topk_ids,
        topk_weights,
        expert_gate_up_weight,
        expert_down_weight,
    )

    hidden_f = hidden.float()
    reference = torch.zeros(hidden_size, device="cuda", dtype=torch.float32)
    for idx in range(top_k):
        expert_id = int(topk_ids[idx].item())
        gate_up = torch.matmul(expert_gate_up_weight[expert_id].float(), hidden_f)
        gate, up = torch.chunk(gate_up, 2, dim=0)
        activation = torch.nn.functional.silu(gate) * up
        reference = reference + topk_weights[idx] * torch.matmul(expert_down_weight[expert_id].float(), activation)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


def test_triton_qwen36_batched_routed_shared_experts_decode_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 4
    hidden_size = 2048
    intermediate = 512
    num_experts = 256
    top_k = 8
    scale = 0.02
    hidden = scale * torch.randn(tokens, hidden_size, device="cuda", dtype=dtype)
    expert_gate_up_weight = scale * torch.randn(
        num_experts,
        2 * intermediate,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    expert_down_weight = scale * torch.randn(
        num_experts,
        hidden_size,
        intermediate,
        device="cuda",
        dtype=dtype,
    )
    shared_gate_up_weight = scale * torch.randn(2 * intermediate, hidden_size, device="cuda", dtype=dtype)
    shared_down_weight = scale * torch.randn(hidden_size, intermediate, device="cuda", dtype=dtype)
    topk_ids = torch.tensor(
        [
            [3, 17, 42, 89, 127, 199, 231, 255],
            [4, 18, 43, 90, 128, 200, 232, 254],
            [5, 19, 44, 91, 129, 201, 233, 253],
            [6, 20, 45, 92, 130, 202, 234, 252],
        ],
        device="cuda",
        dtype=torch.int64,
    )
    topk_weights = torch.softmax(torch.randn(tokens, top_k, device="cuda", dtype=torch.float32), dim=1)

    candidate = triton_qwen36_batched_routed_shared_experts_decode(
        hidden,
        topk_ids,
        topk_weights,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
    )

    reference = torch.zeros(tokens, hidden_size, device="cuda", dtype=torch.float32)
    hidden_f = hidden.float()
    for token in range(tokens):
        for idx in range(top_k):
            expert_id = int(topk_ids[token, idx].item())
            gate_up = torch.matmul(expert_gate_up_weight[expert_id].float(), hidden_f[token])
            gate, up = torch.chunk(gate_up, 2, dim=0)
            activation = torch.nn.functional.silu(gate) * up
            reference[token] = reference[token] + topk_weights[token, idx] * torch.matmul(
                expert_down_weight[expert_id].float(),
                activation,
            )
        shared_gate_up = torch.matmul(shared_gate_up_weight.float(), hidden_f[token])
        shared_gate, shared_up = torch.chunk(shared_gate_up, 2, dim=0)
        shared_activation = torch.nn.functional.silu(shared_gate) * shared_up
        reference[token] = reference[token] + torch.matmul(shared_down_weight.float(), shared_activation)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


def test_triton_qwen36_batched_routed_experts_decode_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 4
    hidden_size = 2048
    intermediate = 512
    num_experts = 256
    top_k = 8
    scale = 0.02
    hidden = scale * torch.randn(tokens, hidden_size, device="cuda", dtype=dtype)
    expert_gate_up_weight = scale * torch.randn(
        num_experts,
        2 * intermediate,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    expert_down_weight = scale * torch.randn(
        num_experts,
        hidden_size,
        intermediate,
        device="cuda",
        dtype=dtype,
    )
    topk_ids = torch.tensor(
        [
            [3, 17, 42, 89, 127, 199, 231, 255],
            [4, 18, 43, 90, 128, 200, 232, 254],
            [5, 19, 44, 91, 129, 201, 233, 253],
            [6, 20, 45, 92, 130, 202, 234, 252],
        ],
        device="cuda",
        dtype=torch.int64,
    )
    topk_weights = torch.softmax(torch.randn(tokens, top_k, device="cuda", dtype=torch.float32), dim=1)

    candidate = triton_qwen36_batched_routed_experts_decode(
        hidden,
        topk_ids,
        topk_weights,
        expert_gate_up_weight,
        expert_down_weight,
    )

    reference = torch.zeros(tokens, hidden_size, device="cuda", dtype=torch.float32)
    hidden_f = hidden.float()
    for token in range(tokens):
        for idx in range(top_k):
            expert_id = int(topk_ids[token, idx].item())
            gate_up = torch.matmul(expert_gate_up_weight[expert_id].float(), hidden_f[token])
            gate, up = torch.chunk(gate_up, 2, dim=0)
            activation = torch.nn.functional.silu(gate) * up
            reference[token] = reference[token] + topk_weights[token, idx] * torch.matmul(
                expert_down_weight[expert_id].float(),
                activation,
            )

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


def test_triton_qwen36_batched_routed_shared_experts_decode_applies_shared_expert_gate():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 2
    hidden_size = 2048
    intermediate = 512
    num_experts = 256
    top_k = 8
    scale = 0.02
    hidden = scale * torch.randn(tokens, hidden_size, device="cuda", dtype=dtype)
    expert_gate_up_weight = scale * torch.randn(
        num_experts,
        2 * intermediate,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    expert_down_weight = scale * torch.randn(
        num_experts,
        hidden_size,
        intermediate,
        device="cuda",
        dtype=dtype,
    )
    shared_gate_up_weight = scale * torch.randn(2 * intermediate, hidden_size, device="cuda", dtype=dtype)
    shared_down_weight = scale * torch.randn(hidden_size, intermediate, device="cuda", dtype=dtype)
    shared_expert_gate_weight = torch.randn(1, hidden_size, device="cuda", dtype=dtype)
    topk_ids = torch.tensor(
        [
            [3, 17, 42, 89, 127, 199, 231, 255],
            [4, 18, 43, 90, 128, 200, 232, 254],
        ],
        device="cuda",
        dtype=torch.int64,
    )
    topk_weights = torch.softmax(torch.randn(tokens, top_k, device="cuda", dtype=torch.float32), dim=1)

    candidate = triton_qwen36_batched_routed_shared_experts_decode(
        hidden,
        topk_ids,
        topk_weights,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        shared_expert_gate_weight,
    )

    reference = torch.zeros(tokens, hidden_size, device="cuda", dtype=torch.float32)
    hidden_f = hidden.float()
    shared_scales = torch.sigmoid(torch.matmul(hidden_f, shared_expert_gate_weight.float().reshape(1, hidden_size).t()))
    for token in range(tokens):
        for idx in range(top_k):
            expert_id = int(topk_ids[token, idx].item())
            gate_up = torch.matmul(expert_gate_up_weight[expert_id].float(), hidden_f[token])
            gate, up = torch.chunk(gate_up, 2, dim=0)
            activation = torch.nn.functional.silu(gate) * up
            reference[token] = reference[token] + topk_weights[token, idx] * torch.matmul(
                expert_down_weight[expert_id].float(),
                activation,
            )
        shared_gate_up = torch.matmul(shared_gate_up_weight.float(), hidden_f[token])
        shared_gate, shared_up = torch.chunk(shared_gate_up, 2, dim=0)
        shared_activation = torch.nn.functional.silu(shared_gate) * shared_up
        reference[token] = reference[token] + shared_scales[token, 0] * torch.matmul(
            shared_down_weight.float(),
            shared_activation,
        )

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


def test_triton_qwen36_moe_decode_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    hidden_size = 2048
    intermediate = 512
    num_experts = 256
    top_k = 8
    scale = 0.02
    hidden = scale * torch.randn(hidden_size, device="cuda", dtype=dtype)
    norm_weight = scale * torch.randn(hidden_size, device="cuda", dtype=dtype)
    router_weight = scale * torch.randn(num_experts, hidden_size, device="cuda", dtype=dtype)
    expert_gate_up_weight = scale * torch.randn(
        num_experts,
        2 * intermediate,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    expert_down_weight = scale * torch.randn(
        num_experts,
        hidden_size,
        intermediate,
        device="cuda",
        dtype=dtype,
    )
    shared_gate_up_weight = scale * torch.randn(2 * intermediate, hidden_size, device="cuda", dtype=dtype)
    shared_down_weight = scale * torch.randn(hidden_size, intermediate, device="cuda", dtype=dtype)

    candidate, logits, topk_ids, topk_weights = triton_qwen36_moe_decode(
        hidden,
        norm_weight,
        router_weight,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        top_k=top_k,
    )

    hidden_f = hidden.float()
    normed = _zero_centered_rms_norm(hidden_f, norm_weight)
    reference_logits = torch.matmul(router_weight.float(), normed)
    reference_probs = torch.softmax(reference_logits, dim=-1)
    reference_values, reference_ids = torch.topk(reference_probs, k=top_k)
    reference_weights = reference_values / torch.sum(reference_values)
    reference = torch.zeros(hidden_size, device="cuda", dtype=torch.float32)
    for idx in range(top_k):
        expert_id = int(reference_ids[idx].item())
        gate_up = torch.matmul(expert_gate_up_weight[expert_id].float(), hidden_f)
        gate, up = torch.chunk(gate_up, 2, dim=0)
        activation = torch.nn.functional.silu(gate) * up
        reference = reference + reference_weights[idx] * torch.matmul(
            expert_down_weight[expert_id].float(),
            activation,
        )
    shared_gate_up = torch.matmul(shared_gate_up_weight.float(), hidden_f)
    shared_gate, shared_up = torch.chunk(shared_gate_up, 2, dim=0)
    shared_activation = torch.nn.functional.silu(shared_gate) * shared_up
    reference = reference + torch.matmul(shared_down_weight.float(), shared_activation)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(logits, reference_logits, atol=atol, rtol=rtol)
    torch.testing.assert_close(topk_ids.cpu(), reference_ids.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(topk_weights, reference_weights, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


def test_triton_qwen36_batched_moe_decode_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 4
    hidden_size = 2048
    intermediate = 512
    num_experts = 256
    top_k = 8
    scale = 0.02
    hidden = scale * torch.randn(tokens, hidden_size, device="cuda", dtype=dtype)
    norm_weight = scale * torch.randn(hidden_size, device="cuda", dtype=dtype)
    router_weight = scale * torch.randn(num_experts, hidden_size, device="cuda", dtype=dtype)
    expert_gate_up_weight = scale * torch.randn(
        num_experts,
        2 * intermediate,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    expert_down_weight = scale * torch.randn(
        num_experts,
        hidden_size,
        intermediate,
        device="cuda",
        dtype=dtype,
    )
    shared_gate_up_weight = scale * torch.randn(2 * intermediate, hidden_size, device="cuda", dtype=dtype)
    shared_down_weight = scale * torch.randn(hidden_size, intermediate, device="cuda", dtype=dtype)

    candidate, logits, topk_ids, topk_weights = triton_qwen36_batched_moe_decode(
        hidden,
        norm_weight,
        router_weight,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        top_k=top_k,
    )

    hidden_f = hidden.float()
    normed = _zero_centered_rms_norm(hidden_f, norm_weight)
    reference_logits = torch.matmul(normed, router_weight.float().t())
    reference_probs = torch.softmax(logits, dim=-1)
    reference_values, reference_ids = torch.topk(reference_probs, k=top_k, dim=1)
    reference_weights = reference_values / torch.sum(reference_values, dim=1, keepdim=True)
    reference = torch.zeros(tokens, hidden_size, device="cuda", dtype=torch.float32)
    for token in range(tokens):
        for idx in range(top_k):
            expert_id = int(reference_ids[token, idx].item())
            gate_up = torch.matmul(expert_gate_up_weight[expert_id].float(), hidden_f[token])
            gate, up = torch.chunk(gate_up, 2, dim=0)
            activation = torch.nn.functional.silu(gate) * up
            reference[token] = reference[token] + reference_weights[token, idx] * torch.matmul(
                expert_down_weight[expert_id].float(),
                activation,
            )
        shared_gate_up = torch.matmul(shared_gate_up_weight.float(), hidden_f[token])
        shared_gate, shared_up = torch.chunk(shared_gate_up, 2, dim=0)
        shared_activation = torch.nn.functional.silu(shared_gate) * shared_up
        reference[token] = reference[token] + torch.matmul(shared_down_weight.float(), shared_activation)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(logits, reference_logits, atol=atol, rtol=rtol)
    torch.testing.assert_close(topk_ids.cpu(), reference_ids.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(topk_weights, reference_weights, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


def test_triton_qwen36_batched_moe_layer_decode_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 2
    hidden_size = 2048
    intermediate = 512
    num_experts = 256
    top_k = 8
    scale = 0.02
    hidden = scale * torch.randn(tokens, hidden_size, device="cuda", dtype=dtype)
    norm_weight = scale * torch.randn(hidden_size, device="cuda", dtype=dtype)
    router_weight = scale * torch.randn(num_experts, hidden_size, device="cuda", dtype=dtype)
    expert_gate_up_weight = scale * torch.randn(
        num_experts,
        2 * intermediate,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    expert_down_weight = scale * torch.randn(
        num_experts,
        hidden_size,
        intermediate,
        device="cuda",
        dtype=dtype,
    )
    shared_gate_up_weight = scale * torch.randn(2 * intermediate, hidden_size, device="cuda", dtype=dtype)
    shared_down_weight = scale * torch.randn(hidden_size, intermediate, device="cuda", dtype=dtype)
    shared_expert_gate_weight = torch.randn(1, hidden_size, device="cuda", dtype=dtype)

    layer_hidden, candidate, logits, topk_ids, topk_weights = triton_qwen36_batched_moe_layer_decode(
        hidden,
        norm_weight,
        router_weight,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        shared_expert_gate_weight,
        top_k=top_k,
    )

    hidden_f = hidden.float()
    normed = _zero_centered_rms_norm(hidden_f, norm_weight)
    reference_logits = torch.matmul(normed, router_weight.float().t())
    reference_probs = torch.softmax(logits, dim=-1)
    reference_values, reference_ids = torch.topk(reference_probs, k=top_k, dim=1)
    reference_weights = reference_values / torch.sum(reference_values, dim=1, keepdim=True)
    reference = torch.zeros(tokens, hidden_size, device="cuda", dtype=torch.float32)
    shared_scales = torch.sigmoid(torch.matmul(hidden_f, shared_expert_gate_weight.float().reshape(1, hidden_size).t()))
    for token in range(tokens):
        for idx in range(top_k):
            expert_id = int(reference_ids[token, idx].item())
            gate_up = torch.matmul(expert_gate_up_weight[expert_id].float(), hidden_f[token])
            gate, up = torch.chunk(gate_up, 2, dim=0)
            activation = torch.nn.functional.silu(gate) * up
            reference[token] = reference[token] + reference_weights[token, idx] * torch.matmul(
                expert_down_weight[expert_id].float(),
                activation,
            )
        shared_gate_up = torch.matmul(shared_gate_up_weight.float(), hidden_f[token])
        shared_gate, shared_up = torch.chunk(shared_gate_up, 2, dim=0)
        shared_activation = torch.nn.functional.silu(shared_gate) * shared_up
        reference[token] = reference[token] + shared_scales[token, 0] * torch.matmul(
            shared_down_weight.float(),
            shared_activation,
        )

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(logits, reference_logits, atol=atol, rtol=rtol)
    torch.testing.assert_close(topk_ids.cpu(), reference_ids.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(topk_weights, reference_weights, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)
    torch.testing.assert_close(layer_hidden, hidden_f + reference, atol=atol, rtol=rtol)


def test_triton_qwen36_batched_attention_project_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 2
    hidden_size = 2048
    q_width = 8192
    kv_width = 512
    scale = 0.02
    hidden = scale * torch.randn(tokens, hidden_size, device="cuda", dtype=dtype)
    norm_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    q_proj_weight = scale * torch.randn(q_width, hidden_size, device="cuda", dtype=dtype)
    k_proj_weight = scale * torch.randn(kv_width, hidden_size, device="cuda", dtype=dtype)
    v_proj_weight = scale * torch.randn(kv_width, hidden_size, device="cuda", dtype=dtype)

    q, k, v = triton_qwen36_batched_attention_project(
        hidden,
        norm_weight,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
    )

    hidden_f = hidden.float()
    normed = _zero_centered_rms_norm(hidden_f, norm_weight)
    reference_q = torch.matmul(normed, q_proj_weight.float().t())
    reference_k = torch.matmul(normed, k_proj_weight.float().t())
    reference_v = torch.matmul(normed, v_proj_weight.float().t())

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(q, reference_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(k, reference_k, atol=atol, rtol=rtol)
    torch.testing.assert_close(v, reference_v, atol=atol, rtol=rtol)


def test_triton_qwen36_batched_deltanet_project_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 2
    hidden_size = 2048
    qkv_width = 8192
    z_width = 4096
    gate_width = 32
    scale = 0.02
    hidden = scale * torch.randn(tokens, hidden_size, device="cuda", dtype=dtype)
    norm_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    in_proj_qkv_weight = scale * torch.randn(qkv_width, hidden_size, device="cuda", dtype=dtype)
    in_proj_z_weight = scale * torch.randn(z_width, hidden_size, device="cuda", dtype=dtype)
    in_proj_a_weight = scale * torch.randn(gate_width, hidden_size, device="cuda", dtype=dtype)
    in_proj_b_weight = scale * torch.randn(gate_width, hidden_size, device="cuda", dtype=dtype)

    mixed_qkv, z, a_logits, b_logits = triton_qwen36_batched_deltanet_project(
        hidden,
        norm_weight,
        in_proj_qkv_weight,
        in_proj_z_weight,
        in_proj_a_weight,
        in_proj_b_weight,
    )

    hidden_f = hidden.float()
    normed = (
        hidden_f
        * torch.rsqrt(torch.mean(hidden_f * hidden_f, dim=1, keepdim=True) + 1e-6)
        * (1.0 + norm_weight.float())
    )
    reference_qkv = torch.matmul(normed, in_proj_qkv_weight.float().t())
    reference_z = torch.matmul(normed, in_proj_z_weight.float().t())
    reference_a = torch.matmul(normed, in_proj_a_weight.float().t())
    reference_b = torch.matmul(normed, in_proj_b_weight.float().t())

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(mixed_qkv, reference_qkv, atol=atol, rtol=rtol)
    torch.testing.assert_close(z, reference_z, atol=atol, rtol=rtol)
    torch.testing.assert_close(a_logits, reference_a, atol=atol, rtol=rtol)
    torch.testing.assert_close(b_logits, reference_b, atol=atol, rtol=rtol)


def test_triton_qwen36_batched_deltanet_conv_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 2
    channels = 8192
    kernel_size = 4
    scale = 0.02
    mixed_qkv = scale * torch.randn(tokens, channels, device="cuda", dtype=dtype)
    conv_state = scale * torch.randn(channels, kernel_size, device="cuda", dtype=dtype)
    conv_weight = scale * torch.randn(channels, 1, kernel_size, device="cuda", dtype=dtype)

    candidate, candidate_state = triton_qwen36_batched_deltanet_conv(
        mixed_qkv,
        conv_state,
        conv_weight,
    )

    combined = torch.cat((conv_state.unsqueeze(0).float(), mixed_qkv.t().unsqueeze(0).float()), dim=-1)
    reference = torch.nn.functional.conv1d(combined, conv_weight.float(), padding=0, groups=channels)
    reference = torch.nn.functional.silu(reference[:, :, -tokens:]).squeeze(0).t()
    reference_state = combined[:, :, -kernel_size:].squeeze(0).to(conv_state.dtype)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate_state, reference_state, atol=atol, rtol=rtol)


def test_triton_qwen36_batched_deltanet_recurrent_output_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 2
    qk_heads = 16
    value_heads = 32
    head_dim = 128
    hidden_size = 2048
    qk_width = qk_heads * head_dim
    value_width = value_heads * head_dim
    scale = 0.02
    mixed_qkv = scale * torch.randn(tokens, 2 * qk_width + value_width, device="cuda", dtype=dtype)
    z = scale * torch.randn(tokens, value_width, device="cuda", dtype=dtype)
    a_logits = scale * torch.randn(tokens, value_heads, device="cuda", dtype=dtype)
    b_logits = scale * torch.randn(tokens, value_heads, device="cuda", dtype=dtype)
    recurrent_state = scale * torch.randn(value_heads, head_dim, head_dim, device="cuda", dtype=dtype)
    out_proj_weight = scale * torch.randn(hidden_size, value_width, device="cuda", dtype=dtype)
    linear_norm_weight = torch.randn(head_dim, device="cuda", dtype=dtype)
    a_log = scale * torch.randn(value_heads, device="cuda", dtype=dtype)
    dt_bias = scale * torch.randn(value_heads, device="cuda", dtype=dtype)

    candidate, candidate_state = triton_qwen36_batched_deltanet_recurrent_output(
        mixed_qkv,
        z,
        a_logits,
        b_logits,
        recurrent_state,
        out_proj_weight,
        linear_norm_weight,
        a_log,
        dt_bias,
        qk_heads=qk_heads,
        value_heads=value_heads,
        head_dim=head_dim,
    )

    query, key, value = torch.split(mixed_qkv, (qk_width, qk_width, value_width), dim=-1)
    query = query.reshape(tokens, qk_heads, head_dim)
    key = key.reshape(tokens, qk_heads, head_dim)
    value = value.reshape(tokens, value_heads, head_dim)
    repeat = value_heads // qk_heads
    query = query.repeat_interleave(repeat, dim=1)
    key = key.repeat_interleave(repeat, dim=1)
    query = query.float() * torch.rsqrt(torch.sum(query.float() * query.float(), dim=-1, keepdim=True) + 1e-6) * (
        head_dim**-0.5
    )
    key = key.float() * torch.rsqrt(torch.sum(key.float() * key.float(), dim=-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(b_logits.float())
    g = -a_log.float().exp() * torch.nn.functional.softplus(a_logits.float() + dt_bias.float())

    state = recurrent_state.float()
    outputs = []
    for token in range(tokens):
        state = state * g[token].exp().reshape(value_heads, 1, 1)
        delta = (value[token].float() - (state * key[token].unsqueeze(-1)).sum(dim=-2)) * beta[token].reshape(
            value_heads,
            1,
        )
        state = state + key[token].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * query[token].unsqueeze(-1)).sum(dim=-2))
    core = torch.stack(outputs, dim=0)
    z_heads = z.reshape(tokens, value_heads, head_dim)
    normed = core * torch.rsqrt(torch.mean(core * core, dim=-1, keepdim=True) + 1e-6)
    normed = normed * linear_norm_weight.float().reshape(1, 1, head_dim) * torch.nn.functional.silu(z_heads.float())
    reference = torch.matmul(normed.reshape(tokens, value_width), out_proj_weight.float().t())
    reference_state = state.to(recurrent_state.dtype)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate_state, reference_state, atol=atol, rtol=rtol)


def test_triton_qwen36_batched_deltanet_moe_layer_decode_matches_reference_composition():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 2
    hidden_size = 64
    qk_heads = 2
    value_heads = 4
    head_dim = 8
    num_experts = 16
    top_k = 4
    intermediate = 16
    kernel_size = 4
    qk_width = qk_heads * head_dim
    value_width = value_heads * head_dim
    qkv_width = 2 * qk_width + value_width
    scale = 0.02

    hidden = scale * torch.randn(tokens, hidden_size, device="cuda", dtype=dtype)
    conv_state = scale * torch.randn(qkv_width, kernel_size, device="cuda", dtype=dtype)
    recurrent_state = scale * torch.randn(value_heads, head_dim, head_dim, device="cuda", dtype=dtype)
    deltanet_norm_weight = scale * torch.randn(hidden_size, device="cuda", dtype=dtype)
    in_proj_qkv_weight = scale * torch.randn(qkv_width, hidden_size, device="cuda", dtype=dtype)
    in_proj_z_weight = scale * torch.randn(value_width, hidden_size, device="cuda", dtype=dtype)
    in_proj_a_weight = scale * torch.randn(value_heads, hidden_size, device="cuda", dtype=dtype)
    in_proj_b_weight = scale * torch.randn(value_heads, hidden_size, device="cuda", dtype=dtype)
    conv1d_weight = scale * torch.randn(qkv_width, 1, kernel_size, device="cuda", dtype=dtype)
    out_proj_weight = scale * torch.randn(hidden_size, value_width, device="cuda", dtype=dtype)
    linear_norm_weight = scale * torch.randn(head_dim, device="cuda", dtype=dtype)
    a_log = scale * torch.randn(value_heads, device="cuda", dtype=dtype)
    dt_bias = scale * torch.randn(value_heads, device="cuda", dtype=dtype)
    moe_norm_weight = scale * torch.randn(hidden_size, device="cuda", dtype=dtype)
    router_weight = scale * torch.randn(num_experts, hidden_size, device="cuda", dtype=dtype)
    expert_gate_up_weight = scale * torch.randn(num_experts, 2 * intermediate, hidden_size, device="cuda", dtype=dtype)
    expert_down_weight = scale * torch.randn(num_experts, hidden_size, intermediate, device="cuda", dtype=dtype)
    shared_gate_up_weight = scale * torch.randn(2 * intermediate, hidden_size, device="cuda", dtype=dtype)
    shared_down_weight = scale * torch.randn(hidden_size, intermediate, device="cuda", dtype=dtype)
    shared_expert_gate_weight = scale * torch.randn(1, hidden_size, device="cuda", dtype=dtype)

    candidate = triton_qwen36_batched_deltanet_moe_layer_decode(
        hidden,
        conv_state,
        recurrent_state,
        deltanet_norm_weight,
        in_proj_qkv_weight,
        in_proj_z_weight,
        in_proj_a_weight,
        in_proj_b_weight,
        conv1d_weight,
        out_proj_weight,
        linear_norm_weight,
        a_log,
        dt_bias,
        moe_norm_weight,
        router_weight,
        expert_gate_up_weight,
        expert_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        shared_expert_gate_weight,
        qk_heads=qk_heads,
        value_heads=value_heads,
        head_dim=head_dim,
        top_k=top_k,
        block_value=value_width,
    )

    deltanet_x = _zero_centered_rms_norm(hidden, deltanet_norm_weight)
    mixed_qkv = torch.matmul(deltanet_x, in_proj_qkv_weight.float().t())
    z = torch.matmul(deltanet_x, in_proj_z_weight.float().t())
    a_logits = torch.matmul(deltanet_x, in_proj_a_weight.float().t())
    b_logits = torch.matmul(deltanet_x, in_proj_b_weight.float().t())
    combined = torch.cat((conv_state.unsqueeze(0).float(), mixed_qkv.t().unsqueeze(0).float()), dim=-1)
    mixed_qkv = torch.nn.functional.conv1d(combined, conv1d_weight.float(), padding=0, groups=qkv_width)
    mixed_qkv = torch.nn.functional.silu(mixed_qkv[:, :, -tokens:]).squeeze(0).t()
    reference_conv_state = combined[:, :, -kernel_size:].squeeze(0).to(dtype)

    query, key, value = torch.split(mixed_qkv, (qk_width, qk_width, value_width), dim=-1)
    query = query.reshape(tokens, qk_heads, head_dim).repeat_interleave(value_heads // qk_heads, dim=1)
    key = key.reshape(tokens, qk_heads, head_dim).repeat_interleave(value_heads // qk_heads, dim=1)
    value = value.reshape(tokens, value_heads, head_dim)
    query = query.float() * torch.rsqrt(torch.sum(query.float() * query.float(), dim=-1, keepdim=True) + 1e-6) * (
        head_dim**-0.5
    )
    key = key.float() * torch.rsqrt(torch.sum(key.float() * key.float(), dim=-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(b_logits.float())
    g = -a_log.float().exp() * torch.nn.functional.softplus(a_logits.float() + dt_bias.float())
    state = recurrent_state.float()
    outputs = []
    for token in range(tokens):
        state = state * g[token].exp().reshape(value_heads, 1, 1)
        delta = (value[token].float() - (state * key[token].unsqueeze(-1)).sum(dim=-2)) * beta[token].reshape(
            value_heads,
            1,
        )
        state = state + key[token].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * query[token].unsqueeze(-1)).sum(dim=-2))
    core = torch.stack(outputs, dim=0)
    z_heads = z.reshape(tokens, value_heads, head_dim)
    normed = core * torch.rsqrt(torch.mean(core * core, dim=-1, keepdim=True) + 1e-6)
    normed = normed * linear_norm_weight.float().reshape(1, 1, head_dim) * torch.nn.functional.silu(z_heads.float())
    reference_update = torch.matmul(normed.reshape(tokens, value_width), out_proj_weight.float().t())
    reference_recurrent_state = state.to(dtype)
    reference_deltanet_hidden = hidden.float() + reference_update

    moe_x = _zero_centered_rms_norm(reference_deltanet_hidden, moe_norm_weight)
    reference_logits = torch.matmul(moe_x, router_weight.float().t())
    probs = torch.softmax(candidate[-3], dim=-1)
    reference_values, reference_ids = torch.topk(probs, k=top_k, dim=1)
    reference_weights = reference_values / torch.sum(reference_values, dim=1, keepdim=True)
    reference_moe_update = torch.zeros_like(reference_deltanet_hidden)
    for token in range(tokens):
        for slot in range(top_k):
            expert_id = int(reference_ids[token, slot].item())
            gate_up = torch.matmul(expert_gate_up_weight[expert_id].float(), reference_deltanet_hidden[token])
            gate, up = torch.chunk(gate_up, 2, dim=0)
            activation = torch.nn.functional.silu(gate) * up
            reference_moe_update[token] += reference_weights[token, slot] * torch.matmul(
                expert_down_weight[expert_id].float(),
                activation,
            )
        shared_gate_up = torch.matmul(shared_gate_up_weight.float(), reference_deltanet_hidden[token])
        shared_gate, shared_up = torch.chunk(shared_gate_up, 2, dim=0)
        shared_activation = torch.nn.functional.silu(shared_gate) * shared_up
        shared = torch.matmul(shared_down_weight.float(), shared_activation)
        shared_scale = torch.sigmoid(torch.matmul(reference_deltanet_hidden[token], shared_expert_gate_weight.float().t()))
        reference_moe_update[token] += shared_scale[0] * shared
    reference_layer_hidden = reference_deltanet_hidden + reference_moe_update

    (
        layer_hidden,
        deltanet_hidden,
        deltanet_update,
        moe_update,
        next_conv_state,
        next_recurrent_state,
        logits,
        topk_ids,
        topk_weights,
    ) = candidate
    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(deltanet_update, reference_update, atol=atol, rtol=rtol)
    torch.testing.assert_close(deltanet_hidden, reference_deltanet_hidden, atol=atol, rtol=rtol)
    torch.testing.assert_close(moe_update, reference_moe_update, atol=atol, rtol=rtol)
    torch.testing.assert_close(layer_hidden, reference_layer_hidden, atol=atol, rtol=rtol)
    torch.testing.assert_close(next_conv_state, reference_conv_state, atol=atol, rtol=rtol)
    torch.testing.assert_close(next_recurrent_state, reference_recurrent_state, atol=atol, rtol=rtol)
    torch.testing.assert_close(logits, reference_logits, atol=atol, rtol=rtol)
    torch.testing.assert_close(topk_ids.cpu(), reference_ids.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(topk_weights, reference_weights, atol=atol, rtol=rtol)


def _reference_rope(torch, values, positions, rope_dim, rope_theta=10000.0):
    tokens, heads, head_dim = values.shape
    out = values.clone()
    rotate_dim = min(rope_dim, head_dim)
    rotate_dim = rotate_dim - rotate_dim % 2
    dims = torch.arange(0, rotate_dim, 2, device=values.device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (dims / rotate_dim))
    angles = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cos(angles)[:, None, :]
    sin = torch.sin(angles)[:, None, :]
    even = values[:, :, :rotate_dim:2]
    odd = values[:, :, 1:rotate_dim:2]
    out[:, :, :rotate_dim:2] = even * cos - odd * sin
    out[:, :, 1:rotate_dim:2] = even * sin + odd * cos
    return out


def test_triton_qwen36_batched_attention_decode_matches_real_shape_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tokens = 2
    hidden_size = 2048
    attention_heads = 16
    kv_heads = 2
    head_dim = 256
    rope_dim = 64
    rope_theta = 10000000.0
    q_width = attention_heads * head_dim
    q_proj_rows = 2 * q_width
    kv_width = kv_heads * head_dim
    max_positions = 4
    start_position = 1
    scale = 0.02
    hidden = scale * torch.randn(tokens, hidden_size, device="cuda", dtype=dtype)
    key_cache = scale * torch.randn(max_positions, kv_heads, head_dim, device="cuda", dtype=torch.float32)
    value_cache = scale * torch.randn_like(key_cache)
    norm_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    q_proj_weight = scale * torch.randn(q_proj_rows, hidden_size, device="cuda", dtype=dtype)
    k_proj_weight = scale * torch.randn(kv_width, hidden_size, device="cuda", dtype=dtype)
    v_proj_weight = scale * torch.randn(kv_width, hidden_size, device="cuda", dtype=dtype)
    out_proj_weight = scale * torch.randn(hidden_size, q_width, device="cuda", dtype=dtype)
    q_norm_weight = torch.randn(head_dim, device="cuda", dtype=dtype)
    k_norm_weight = torch.randn(head_dim, device="cuda", dtype=dtype)

    candidate, candidate_key_cache, candidate_value_cache = triton_qwen36_batched_attention_decode(
        hidden,
        key_cache,
        value_cache,
        norm_weight,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        out_proj_weight,
        q_norm_weight,
        k_norm_weight,
        start_position=start_position,
        attention_heads=attention_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        rope_theta=rope_theta,
    )

    hidden_f = hidden.float()
    normed = _zero_centered_rms_norm(hidden_f, norm_weight)
    q_pairs = torch.matmul(normed, q_proj_weight.float().t()).reshape(tokens, attention_heads, 2 * head_dim)
    q_raw = q_pairs[:, :, :head_dim].reshape(tokens, q_width)
    gate = q_pairs[:, :, head_dim:].reshape(tokens, q_width)
    k_raw = torch.matmul(normed, k_proj_weight.float().t())
    v_raw = torch.matmul(normed, v_proj_weight.float().t())
    q = q_raw.reshape(tokens, attention_heads, head_dim)
    k = k_raw.reshape(tokens, kv_heads, head_dim)
    q = _zero_centered_rms_norm(q, q_norm_weight)
    k = _zero_centered_rms_norm(k, k_norm_weight)
    positions = torch.arange(start_position, start_position + tokens, device="cuda")
    q = _reference_rope(torch, q, positions, rope_dim, rope_theta=rope_theta)
    k = _reference_rope(torch, k, positions, rope_dim, rope_theta=rope_theta)

    reference_key_cache = key_cache.clone()
    reference_value_cache = value_cache.clone()
    reference_key_cache[start_position : start_position + tokens] = k
    reference_value_cache[start_position : start_position + tokens] = v_raw.reshape(tokens, kv_heads, head_dim)
    attended = torch.empty(tokens, attention_heads, head_dim, device="cuda", dtype=torch.float32)
    heads_per_kv = attention_heads // kv_heads
    for token in range(tokens):
        position = start_position + token
        for head in range(attention_heads):
            kv_head = head // heads_per_kv
            scores = torch.sum(reference_key_cache[: position + 1, kv_head] * q[token, head], dim=-1) / (
                head_dim**0.5
            )
            weights = torch.softmax(scores, dim=0)
            attended[token, head] = torch.sum(reference_value_cache[: position + 1, kv_head] * weights[:, None], dim=0)
    reference = torch.matmul(attended.reshape(tokens, q_width) * torch.sigmoid(gate.float()), out_proj_weight.float().t())

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate_key_cache, reference_key_cache, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate_value_cache, reference_value_cache, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_k", [1, 2, 4, 8])
def test_triton_synthetic_qwen36_moe_decode_matches_reference(dtype, top_k):
    spec = synthetic_qwen36_spec()
    spec = replace(spec, num_experts=max(spec.num_experts, top_k), num_routed_experts=top_k)
    weights = make_synthetic_qwen36_decode_weights(spec, device="cuda", dtype=dtype, seed=123)
    hidden = torch.randn(spec.hidden_size, device="cuda", dtype=dtype)
    layer = weights["layers"][0]

    candidate = triton_synthetic_qwen36_moe_decode(
        hidden,
        layer["norm_weight"],
        layer["router_weight"],
        layer["expert_gate_up_weight"],
        layer["expert_down_weight"],
        layer["shared_gate_up_weight"],
        layer["shared_down_weight"],
        top_k=spec.num_routed_experts,
    )
    reference = reference_qwen36_moe_decode(hidden, layer, spec)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


def test_triton_synthetic_qwen36_moe_decode_matches_reference_with_256_experts():
    spec = synthetic_qwen36_spec()
    spec = replace(spec, num_experts=256, num_routed_experts=8)
    weights = make_synthetic_qwen36_decode_weights(spec, device="cuda", dtype=torch.bfloat16, seed=123)
    hidden = torch.randn(spec.hidden_size, device="cuda", dtype=torch.bfloat16)
    layer = weights["layers"][0]

    candidate = triton_synthetic_qwen36_moe_decode(
        hidden,
        layer["norm_weight"],
        layer["router_weight"],
        layer["expert_gate_up_weight"],
        layer["expert_down_weight"],
        layer["shared_gate_up_weight"],
        layer["shared_down_weight"],
        top_k=spec.num_routed_experts,
    )
    reference = reference_qwen36_moe_decode(hidden, layer, spec)

    atol, rtol = default_tolerance(torch.bfloat16)
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_synthetic_qwen36_deltanet_decode_matches_reference(dtype):
    spec = synthetic_qwen36_spec()
    weights = make_synthetic_qwen36_decode_weights(spec, device="cuda", dtype=dtype, seed=123)
    hidden = torch.randn(spec.hidden_size, device="cuda", dtype=dtype)
    state = torch.randn(spec.deltanet_state_shape(), device="cuda", dtype=dtype)
    layer = weights["layers"][0]

    candidate_hidden, candidate_state = triton_synthetic_qwen36_deltanet_decode(
        hidden,
        state,
        layer["norm_weight"],
        layer["q_weight"],
        layer["k_weight"],
        layer["v_weight"],
        layer["gate_weight"],
        layer["out_weight"],
        qk_heads=spec.deltanet_qk_heads,
        head_dim=spec.deltanet_head_dim,
        value_dim_per_head=spec.deltanet_value_dim_per_qk_head,
    )
    reference_hidden, reference_state = reference_qwen36_deltanet_decode(hidden, state, layer, spec)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate_hidden, reference_hidden, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate_state, reference_state, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_synthetic_qwen36_attention_decode_matches_reference(dtype):
    spec = synthetic_qwen36_spec()
    weights = make_synthetic_qwen36_decode_weights(spec, device="cuda", dtype=dtype, seed=123)
    hidden = torch.randn(spec.hidden_size, device="cuda", dtype=dtype)
    key_cache = torch.randn(spec.attention_cache_shape(max_positions=8), device="cuda", dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    position = 2
    layer = weights["layers"][3]

    candidate_hidden, candidate_key_cache, candidate_value_cache = triton_synthetic_qwen36_attention_decode(
        hidden,
        key_cache,
        value_cache,
        layer["norm_weight"],
        layer["q_weight"],
        layer["k_weight"],
        layer["v_weight"],
        layer["out_weight"],
        position=position,
        attention_heads=spec.attention_heads,
        kv_heads=spec.attention_kv_heads,
        head_dim=spec.attention_head_dim,
        rope_dim=spec.rope_dim,
    )
    reference_hidden, reference_key_cache, reference_value_cache = reference_qwen36_attention_decode(
        hidden,
        key_cache,
        value_cache,
        layer,
        spec,
        position,
    )

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate_hidden, reference_hidden, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate_key_cache, reference_key_cache, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate_value_cache, reference_value_cache, atol=atol, rtol=rtol)


def test_triton_synthetic_qwen36_attention_decode_in_place_cache_matches_reference():
    dtype = torch.bfloat16
    spec = synthetic_qwen36_spec()
    weights = make_synthetic_qwen36_decode_weights(spec, device="cuda", dtype=dtype, seed=123)
    hidden = torch.randn(spec.hidden_size, device="cuda", dtype=dtype)
    key_cache = torch.randn(spec.attention_cache_shape(max_positions=8), device="cuda", dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    candidate_key_cache = key_cache.clone()
    candidate_value_cache = value_cache.clone()
    position = 2
    layer = weights["layers"][3]

    candidate_hidden, updated_key_cache, updated_value_cache = triton_synthetic_qwen36_attention_decode(
        hidden,
        candidate_key_cache,
        candidate_value_cache,
        layer["norm_weight"],
        layer["q_weight"],
        layer["k_weight"],
        layer["v_weight"],
        layer["out_weight"],
        position=position,
        attention_heads=spec.attention_heads,
        kv_heads=spec.attention_kv_heads,
        head_dim=spec.attention_head_dim,
        rope_dim=spec.rope_dim,
        copy_cache=False,
    )
    reference_hidden, reference_key_cache, reference_value_cache = reference_qwen36_attention_decode(
        hidden,
        key_cache,
        value_cache,
        layer,
        spec,
        position,
    )

    assert updated_key_cache.data_ptr() == candidate_key_cache.data_ptr()
    assert updated_value_cache.data_ptr() == candidate_value_cache.data_ptr()
    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate_hidden, reference_hidden, atol=atol, rtol=rtol)
    torch.testing.assert_close(updated_key_cache, reference_key_cache, atol=atol, rtol=rtol)
    torch.testing.assert_close(updated_value_cache, reference_value_cache, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_synthetic_qwen36_deltanet_moe_layer_matches_reference(dtype):
    spec = synthetic_qwen36_spec()
    weights = make_synthetic_qwen36_decode_weights(spec, device="cuda", dtype=dtype, seed=123)
    hidden = torch.randn(spec.hidden_size, device="cuda", dtype=dtype)
    state = torch.randn(spec.deltanet_state_shape(), device="cuda", dtype=dtype)
    layer = weights["layers"][0]

    candidate_hidden, candidate_state = triton_synthetic_qwen36_deltanet_moe_decode(
        hidden,
        state,
        layer,
        qk_heads=spec.deltanet_qk_heads,
        head_dim=spec.deltanet_head_dim,
        value_dim_per_head=spec.deltanet_value_dim_per_qk_head,
        top_k=spec.num_routed_experts,
    )
    reference_hidden, reference_state = reference_qwen36_deltanet_decode(hidden, state, layer, spec)
    reference_hidden = reference_qwen36_moe_decode(reference_hidden, layer, spec)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate_hidden, reference_hidden, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate_state, reference_state, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_synthetic_qwen36_attention_moe_layer_matches_reference(dtype):
    spec = synthetic_qwen36_spec()
    weights = make_synthetic_qwen36_decode_weights(spec, device="cuda", dtype=dtype, seed=123)
    hidden = torch.randn(spec.hidden_size, device="cuda", dtype=dtype)
    key_cache = torch.randn(spec.attention_cache_shape(max_positions=8), device="cuda", dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    position = 2
    layer = weights["layers"][3]

    candidate_hidden, candidate_key_cache, candidate_value_cache = triton_synthetic_qwen36_attention_moe_decode(
        hidden,
        key_cache,
        value_cache,
        layer,
        position=position,
        attention_heads=spec.attention_heads,
        kv_heads=spec.attention_kv_heads,
        head_dim=spec.attention_head_dim,
        rope_dim=spec.rope_dim,
        top_k=spec.num_routed_experts,
    )
    reference_hidden, reference_key_cache, reference_value_cache = reference_qwen36_attention_decode(
        hidden,
        key_cache,
        value_cache,
        layer,
        spec,
        position,
    )
    reference_hidden = reference_qwen36_moe_decode(reference_hidden, layer, spec)

    atol, rtol = default_tolerance(dtype)
    torch.testing.assert_close(candidate_hidden, reference_hidden, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate_key_cache, reference_key_cache, atol=atol, rtol=rtol)
    torch.testing.assert_close(candidate_value_cache, reference_value_cache, atol=atol, rtol=rtol)

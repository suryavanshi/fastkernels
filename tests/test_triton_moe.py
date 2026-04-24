import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from fastkernels.kernels.triton import triton_expert_histogram, triton_fused_swiglu
from fastkernels.reference import reference_expert_histogram, reference_fused_swiglu
from fastkernels.testing import default_tolerance


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


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

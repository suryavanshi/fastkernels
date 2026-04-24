from fastkernels.testing import default_tolerance


def test_default_tolerance_names():
    assert default_tolerance("torch.float32") == (1e-4, 1e-4)
    assert default_tolerance("torch.bfloat16") == (3e-2, 3e-2)

from fastkernels.models import Qwen35MoESpec, qwen35_35b_a3b_spec


def test_qwen35_default_moe_shapes():
    spec = qwen35_35b_a3b_spec()

    assert spec.hidden_size == 2048
    assert spec.num_layers == 40
    assert spec.num_experts == 256
    assert spec.active_experts_per_token == 9
    assert spec.layer_kinds()[:4] == (
        "deltanet_moe",
        "deltanet_moe",
        "deltanet_moe",
        "attention_moe",
    )

    shapes = spec.moe_shapes(tokens=16)
    assert shapes["hidden_states"] == (16, 2048)
    assert shapes["routed_hidden_states"] == (128, 2048)
    assert shapes["expert_gate_up"] == (128, 1024)
    assert shapes["expert_activation"] == (128, 512)


def test_qwen35_from_hf_config_aliases():
    spec = Qwen35MoESpec.from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "moe_intermediate_size": 512,
        },
        name="fixture",
    )

    assert spec.name == "fixture"
    assert spec.num_experts == 256
    assert spec.num_routed_experts == 8
    assert spec.gated_up_features == 1024

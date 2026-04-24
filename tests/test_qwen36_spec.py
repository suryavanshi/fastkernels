from fastkernels.models import Qwen36A3BSpec, qwen36_35b_a3b_spec, synthetic_qwen36_spec


def test_qwen36_default_shapes():
    spec = qwen36_35b_a3b_spec()

    assert spec.hidden_size == 2048
    assert spec.num_layers == 40
    assert spec.num_experts == 256
    assert spec.num_routed_experts == 8
    assert spec.num_shared_experts == 1
    assert spec.layer_counts() == {"deltanet_moe": 30, "attention_moe": 10}
    assert spec.layer_kinds()[:4] == (
        "deltanet_moe",
        "deltanet_moe",
        "deltanet_moe",
        "attention_moe",
    )
    assert spec.deltanet_state_shape() == (16, 128, 256)
    assert spec.attention_cache_shape(max_positions=128) == (128, 2, 256)

    shapes = spec.decode_shapes(max_positions=128)
    assert shapes["deltanet_states"] == (30, 16, 128, 256)
    assert shapes["attention_key_cache"] == (10, 128, 2, 256)
    assert shapes["expert_gate_up_weight"] == (256, 1024, 2048)


def test_qwen36_synthetic_spec_is_small_but_structural():
    spec = synthetic_qwen36_spec()

    assert spec.num_layers == 4
    assert spec.layer_counts() == {"deltanet_moe": 3, "attention_moe": 1}
    assert spec.deltanet_state_shape() == (2, 4, 8)
    assert spec.attention_heads_per_kv_head == 2
    assert spec.decode_shapes(max_positions=8)["attention_value_cache"] == (1, 8, 2, 4)


def test_qwen36_from_hf_config_aliases():
    spec = Qwen36A3BSpec.from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "moe_intermediate_size": 512,
            "num_linear_attention_value_heads": 32,
            "num_linear_attention_qk_heads": 16,
            "linear_attention_head_dim": 128,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rope_dim": 64,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
        },
        name="fixture",
    )

    assert spec.name == "fixture"
    assert spec.layer_counts()["attention_moe"] == 10
    assert spec.deltanet_value_dim_per_qk_head == 256
    assert spec.attention_heads_per_kv_head == 8

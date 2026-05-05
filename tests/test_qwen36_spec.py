import pytest

from fastkernels.models import (
    Qwen36A3BSpec,
    flatten_qwen36_moe_weight_keys,
    pack_qwen36_attention_weights_from_state_dict,
    pack_qwen36_linear_attention_weights_from_state_dict,
    load_qwen36_moe_weights_from_safetensors,
    pack_qwen36_moe_weights_from_state_dict,
    qwen36_35b_a3b_spec,
    resolve_qwen36_full_weight_plan,
    resolve_qwen36_attention_weight_keys,
    resolve_qwen36_linear_attention_weight_keys,
    resolve_qwen36_moe_weight_keys,
    resolve_qwen36_root_weight_keys,
    synthetic_qwen36_spec,
)
from fastkernels.reference import (
    initial_qwen36_real_decode_state,
    qwen36_real_attention_update,
    qwen36_real_deltanet_update,
    qwen36_real_rms_norm,
)


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
    assert spec.rope_theta == 10000000.0

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


def test_qwen36_from_real_hf_text_config_layout():
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10
    spec = Qwen36A3BSpec.from_hf_config(
        {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_size": 2048,
                "layer_types": layer_types,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_value_head_dim": 128,
                "max_position_embeddings": 262144,
                "moe_intermediate_size": 512,
                "num_attention_heads": 16,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 40,
                "num_key_value_heads": 2,
                "partial_rotary_factor": 0.25,
                "rope_parameters": {
                    "mrope_interleaved": True,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default",
                },
                "shared_expert_intermediate_size": 512,
                "vocab_size": 248320,
            },
        },
        name="Qwen/Qwen3.6-35B-A3B",
    )

    assert spec.hidden_size == 2048
    assert spec.num_experts == 256
    assert spec.num_routed_experts == 8
    assert spec.expert_intermediate_size == 512
    assert spec.deltanet_value_heads == 32
    assert spec.deltanet_qk_heads == 16
    assert spec.deltanet_head_dim == 128
    assert spec.attention_head_dim == 256
    assert spec.rope_dim == 64
    assert spec.rope_theta == 10000000.0
    assert spec.context_length == 262144
    assert spec.vocab_size == 248320
    assert spec.layer_kinds() == (
        "deltanet_moe",
        "deltanet_moe",
        "deltanet_moe",
        "attention_moe",
    ) * 10


def test_qwen36_moe_weight_key_resolution_for_split_hf_layout():
    torch = pytest.importorskip("torch")
    spec = synthetic_qwen36_spec()
    layer = 2
    tensors = {
        f"model.layers.{layer}.post_attention_layernorm.weight": torch.empty(spec.hidden_size),
        f"model.layers.{layer}.mlp.gate.weight": torch.empty(spec.num_experts, spec.hidden_size),
        f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight": torch.empty(
            spec.expert_intermediate_size,
            spec.hidden_size,
        ),
        f"model.layers.{layer}.mlp.shared_expert.up_proj.weight": torch.empty(
            spec.expert_intermediate_size,
            spec.hidden_size,
        ),
        f"model.layers.{layer}.mlp.shared_expert.down_proj.weight": torch.empty(
            spec.hidden_size,
            spec.expert_intermediate_size,
        ),
    }
    for expert_idx in range(spec.num_experts):
        tensors[f"model.layers.{layer}.mlp.experts.{expert_idx}.gate_proj.weight"] = torch.empty(
            spec.expert_intermediate_size,
            spec.hidden_size,
        )
        tensors[f"model.layers.{layer}.mlp.experts.{expert_idx}.up_proj.weight"] = torch.empty(
            spec.expert_intermediate_size,
            spec.hidden_size,
        )
        tensors[f"model.layers.{layer}.mlp.experts.{expert_idx}.down_proj.weight"] = torch.empty(
            spec.hidden_size,
            spec.expert_intermediate_size,
        )

    keys = resolve_qwen36_moe_weight_keys(tensors, layer, spec=spec)

    assert keys.norm_weight == f"model.layers.{layer}.post_attention_layernorm.weight"
    assert keys.router_weight == f"model.layers.{layer}.mlp.gate.weight"
    assert keys.expert_gate_up_weight[0] == (
        f"model.layers.{layer}.mlp.experts.0.gate_proj.weight",
        f"model.layers.{layer}.mlp.experts.0.up_proj.weight",
    )
    assert keys.expert_down_weight[-1] == f"model.layers.{layer}.mlp.experts.{spec.num_experts - 1}.down_proj.weight"
    assert keys.shared_gate_up_weight == (
        f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
        f"model.layers.{layer}.mlp.shared_expert.up_proj.weight",
    )


def test_qwen36_moe_weight_packing_for_split_hf_layout():
    torch = pytest.importorskip("torch")
    spec = synthetic_qwen36_spec()
    layer = 0
    tensors = {
        f"model.layers.{layer}.post_attention_layernorm.weight": torch.arange(spec.hidden_size, dtype=torch.float32),
        f"model.layers.{layer}.mlp.gate.weight": torch.arange(
            spec.num_experts * spec.hidden_size,
            dtype=torch.float32,
        ).reshape(spec.num_experts, spec.hidden_size),
        f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight": torch.full(
            (spec.expert_intermediate_size, spec.hidden_size),
            3.0,
        ),
        f"model.layers.{layer}.mlp.shared_expert.up_proj.weight": torch.full(
            (spec.expert_intermediate_size, spec.hidden_size),
            4.0,
        ),
        f"model.layers.{layer}.mlp.shared_expert.down_proj.weight": torch.full(
            (spec.hidden_size, spec.expert_intermediate_size),
            5.0,
        ),
    }
    for expert_idx in range(spec.num_experts):
        tensors[f"model.layers.{layer}.mlp.experts.{expert_idx}.gate_proj.weight"] = torch.full(
            (spec.expert_intermediate_size, spec.hidden_size),
            float(expert_idx + 1),
        )
        tensors[f"model.layers.{layer}.mlp.experts.{expert_idx}.up_proj.weight"] = torch.full(
            (spec.expert_intermediate_size, spec.hidden_size),
            float(expert_idx + 11),
        )
        tensors[f"model.layers.{layer}.mlp.experts.{expert_idx}.down_proj.weight"] = torch.full(
            (spec.hidden_size, spec.expert_intermediate_size),
            float(expert_idx + 21),
        )

    packed = pack_qwen36_moe_weights_from_state_dict(tensors, layer, spec=spec)

    assert packed.norm_weight.shape == (spec.hidden_size,)
    assert packed.router_weight.shape == (spec.num_experts, spec.hidden_size)
    assert packed.expert_gate_up_weight.shape == (spec.num_experts, spec.gated_up_features, spec.hidden_size)
    assert packed.expert_down_weight.shape == (spec.num_experts, spec.hidden_size, spec.expert_intermediate_size)
    torch.testing.assert_close(
        packed.expert_gate_up_weight[0, : spec.expert_intermediate_size],
        tensors[f"model.layers.{layer}.mlp.experts.0.gate_proj.weight"],
    )
    torch.testing.assert_close(
        packed.expert_gate_up_weight[0, spec.expert_intermediate_size :],
        tensors[f"model.layers.{layer}.mlp.experts.0.up_proj.weight"],
    )
    torch.testing.assert_close(
        packed.shared_gate_up_weight[: spec.expert_intermediate_size],
        tensors[f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight"],
    )
    torch.testing.assert_close(
        packed.shared_gate_up_weight[spec.expert_intermediate_size :],
        tensors[f"model.layers.{layer}.mlp.shared_expert.up_proj.weight"],
    )


def test_qwen36_moe_weight_packing_for_real_grouped_hf_layout():
    torch = pytest.importorskip("torch")
    spec = synthetic_qwen36_spec()
    layer = 0
    tensors = {
        f"model.language_model.layers.{layer}.post_attention_layernorm.weight": torch.arange(
            spec.hidden_size,
            dtype=torch.float32,
        ),
        f"model.language_model.layers.{layer}.mlp.gate.weight": torch.arange(
            spec.num_experts * spec.hidden_size,
            dtype=torch.float32,
        ).reshape(spec.num_experts, spec.hidden_size),
        f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj": torch.arange(
            spec.num_experts * spec.gated_up_features * spec.hidden_size,
            dtype=torch.float32,
        ).reshape(spec.num_experts, spec.gated_up_features, spec.hidden_size),
        f"model.language_model.layers.{layer}.mlp.experts.down_proj": torch.arange(
            spec.num_experts * spec.hidden_size * spec.expert_intermediate_size,
            dtype=torch.float32,
        ).reshape(spec.num_experts, spec.hidden_size, spec.expert_intermediate_size),
        f"model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight": torch.full(
            (spec.expert_intermediate_size, spec.hidden_size),
            3.0,
        ),
        f"model.language_model.layers.{layer}.mlp.shared_expert.up_proj.weight": torch.full(
            (spec.expert_intermediate_size, spec.hidden_size),
            4.0,
        ),
        f"model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight": torch.full(
            (spec.hidden_size, spec.expert_intermediate_size),
            5.0,
        ),
        f"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight": torch.full(
            (1, spec.hidden_size),
            0.5,
        ),
    }

    keys = resolve_qwen36_moe_weight_keys(tensors, layer, spec=spec)
    packed = pack_qwen36_moe_weights_from_state_dict(tensors, layer, spec=spec)

    assert keys.expert_gate_up_weight == f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
    assert keys.expert_down_weight == f"model.language_model.layers.{layer}.mlp.experts.down_proj"
    assert keys.shared_expert_gate_weight == f"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight"
    assert packed.expert_gate_up_weight.shape == (spec.num_experts, spec.gated_up_features, spec.hidden_size)
    assert packed.expert_down_weight.shape == (spec.num_experts, spec.hidden_size, spec.expert_intermediate_size)
    assert packed.shared_expert_gate_weight.shape == (1, spec.hidden_size)
    torch.testing.assert_close(
        packed.expert_gate_up_weight,
        tensors[f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"],
    )


def test_qwen36_linear_attention_weight_packing_for_real_hf_layout():
    torch = pytest.importorskip("torch")
    spec = synthetic_qwen36_spec()
    layer = 0
    value_width = spec.deltanet_value_heads * spec.deltanet_head_dim
    qk_width = spec.deltanet_qk_heads * spec.deltanet_head_dim
    qkv_width = 2 * qk_width + value_width
    prefix = f"model.language_model.layers.{layer}"
    tensors = {
        f"{prefix}.input_layernorm.weight": torch.ones(spec.hidden_size),
        f"{prefix}.linear_attn.in_proj_qkv.weight": torch.ones(qkv_width, spec.hidden_size),
        f"{prefix}.linear_attn.in_proj_z.weight": torch.ones(value_width, spec.hidden_size),
        f"{prefix}.linear_attn.out_proj.weight": torch.ones(spec.hidden_size, value_width),
        f"{prefix}.linear_attn.norm.weight": torch.ones(spec.deltanet_head_dim),
        f"{prefix}.linear_attn.A_log": torch.ones(spec.deltanet_value_heads),
        f"{prefix}.linear_attn.dt_bias": torch.ones(spec.deltanet_value_heads),
        f"{prefix}.linear_attn.in_proj_a.weight": torch.ones(spec.deltanet_value_heads, spec.hidden_size),
        f"{prefix}.linear_attn.in_proj_b.weight": torch.ones(spec.deltanet_value_heads, spec.hidden_size),
        f"{prefix}.linear_attn.conv1d.weight": torch.ones(qkv_width, 1, 4),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(spec.hidden_size),
    }

    keys = resolve_qwen36_linear_attention_weight_keys(tensors, layer)
    packed = pack_qwen36_linear_attention_weights_from_state_dict(tensors, layer, spec=spec)

    assert keys.in_proj_qkv_weight == f"{prefix}.linear_attn.in_proj_qkv.weight"
    assert packed.in_proj_qkv_weight.shape == (qkv_width, spec.hidden_size)
    assert packed.in_proj_z_weight.shape == (value_width, spec.hidden_size)
    assert packed.out_proj_weight.shape == (spec.hidden_size, value_width)


def test_qwen36_attention_weight_packing_for_real_hf_layout():
    torch = pytest.importorskip("torch")
    spec = synthetic_qwen36_spec()
    layer = 3
    q_width = spec.attention_heads * spec.attention_head_dim
    kv_width = spec.attention_kv_heads * spec.attention_head_dim
    prefix = f"model.language_model.layers.{layer}"
    tensors = {
        f"{prefix}.input_layernorm.weight": torch.ones(spec.hidden_size),
        f"{prefix}.self_attn.q_proj.weight": torch.ones(q_width, spec.hidden_size),
        f"{prefix}.self_attn.k_proj.weight": torch.ones(kv_width, spec.hidden_size),
        f"{prefix}.self_attn.v_proj.weight": torch.ones(kv_width, spec.hidden_size),
        f"{prefix}.self_attn.o_proj.weight": torch.ones(spec.hidden_size, q_width),
        f"{prefix}.self_attn.q_norm.weight": torch.ones(spec.attention_head_dim),
        f"{prefix}.self_attn.k_norm.weight": torch.ones(spec.attention_head_dim),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(spec.hidden_size),
    }

    keys = resolve_qwen36_attention_weight_keys(tensors, layer)
    packed = pack_qwen36_attention_weights_from_state_dict(tensors, layer, spec=spec)

    assert keys.q_proj_weight == f"{prefix}.self_attn.q_proj.weight"
    assert packed.q_proj_weight.shape == (q_width, spec.hidden_size)
    assert packed.k_proj_weight.shape == (kv_width, spec.hidden_size)
    assert packed.o_proj_weight.shape == (spec.hidden_size, q_width)


def test_qwen36_real_rms_norm_uses_qwen35_offset_weight():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[3.0, 4.0]])
    weight = torch.zeros(2)

    normalized = qwen36_real_rms_norm(hidden, weight)

    expected = hidden * torch.rsqrt(torch.mean(hidden * hidden, dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(normalized, expected)
    torch.testing.assert_close(qwen36_real_rms_norm(hidden, torch.ones(2)), 2.0 * expected)


def test_qwen36_real_deltanet_update_runs_synthetic_shapes():
    torch = pytest.importorskip("torch")
    spec = synthetic_qwen36_spec()
    layer = 0
    value_width = spec.deltanet_value_heads * spec.deltanet_head_dim
    qk_width = spec.deltanet_qk_heads * spec.deltanet_head_dim
    qkv_width = 2 * qk_width + value_width
    prefix = f"model.language_model.layers.{layer}"
    tensors = {
        f"{prefix}.input_layernorm.weight": torch.zeros(spec.hidden_size),
        f"{prefix}.linear_attn.in_proj_qkv.weight": torch.full((qkv_width, spec.hidden_size), 0.01),
        f"{prefix}.linear_attn.in_proj_z.weight": torch.full((value_width, spec.hidden_size), 0.01),
        f"{prefix}.linear_attn.out_proj.weight": torch.full((spec.hidden_size, value_width), 0.01),
        f"{prefix}.linear_attn.norm.weight": torch.ones(spec.deltanet_head_dim),
        f"{prefix}.linear_attn.A_log": torch.zeros(spec.deltanet_value_heads),
        f"{prefix}.linear_attn.dt_bias": torch.ones(spec.deltanet_value_heads),
        f"{prefix}.linear_attn.in_proj_a.weight": torch.full((spec.deltanet_value_heads, spec.hidden_size), 0.01),
        f"{prefix}.linear_attn.in_proj_b.weight": torch.full((spec.deltanet_value_heads, spec.hidden_size), 0.01),
        f"{prefix}.linear_attn.conv1d.weight": torch.ones(qkv_width, 1, 4),
        f"{prefix}.post_attention_layernorm.weight": torch.zeros(spec.hidden_size),
    }
    weights = pack_qwen36_linear_attention_weights_from_state_dict(tensors, layer, spec=spec)
    state = initial_qwen36_real_decode_state(spec, max_positions=4)
    hidden = torch.full((2, spec.hidden_size), 0.01)

    update, next_conv, next_recurrent = qwen36_real_deltanet_update(
        hidden,
        state.deltanet_conv_states[0],
        state.deltanet_recurrent_states[0],
        weights,
        spec,
    )

    assert update.shape == hidden.shape
    assert next_conv.shape == state.deltanet_conv_states[0].shape
    assert next_recurrent.shape == state.deltanet_recurrent_states[0].shape


def test_qwen36_real_attention_update_uses_gated_q_projection_layout():
    torch = pytest.importorskip("torch")
    spec = synthetic_qwen36_spec()
    layer = 3
    q_width = spec.attention_heads * spec.attention_head_dim
    kv_width = spec.attention_kv_heads * spec.attention_head_dim
    prefix = f"model.language_model.layers.{layer}"
    tensors = {
        f"{prefix}.input_layernorm.weight": torch.zeros(spec.hidden_size),
        f"{prefix}.self_attn.q_proj.weight": torch.full((2 * q_width, spec.hidden_size), 0.01),
        f"{prefix}.self_attn.k_proj.weight": torch.full((kv_width, spec.hidden_size), 0.01),
        f"{prefix}.self_attn.v_proj.weight": torch.full((kv_width, spec.hidden_size), 0.01),
        f"{prefix}.self_attn.o_proj.weight": torch.full((spec.hidden_size, q_width), 0.01),
        f"{prefix}.self_attn.q_norm.weight": torch.zeros(spec.attention_head_dim),
        f"{prefix}.self_attn.k_norm.weight": torch.zeros(spec.attention_head_dim),
        f"{prefix}.post_attention_layernorm.weight": torch.zeros(spec.hidden_size),
    }
    weights = pack_qwen36_attention_weights_from_state_dict(tensors, layer, spec=spec)
    state = initial_qwen36_real_decode_state(spec, max_positions=4)
    hidden = torch.full((2, spec.hidden_size), 0.01)

    update, key_cache, value_cache = qwen36_real_attention_update(
        hidden,
        state.attention_key_cache[0],
        state.attention_value_cache[0],
        weights,
        spec,
    )

    assert update.shape == hidden.shape
    assert key_cache.shape == state.attention_key_cache[0].shape
    assert value_cache.shape == state.attention_value_cache[0].shape


def _add_synthetic_moe_keys(tensors, spec, layer, prefix):
    tensors[f"{prefix}.post_attention_layernorm.weight"] = object()
    tensors[f"{prefix}.mlp.gate.weight"] = object()
    tensors[f"{prefix}.mlp.experts.gate_up_proj"] = object()
    tensors[f"{prefix}.mlp.experts.down_proj"] = object()
    tensors[f"{prefix}.mlp.shared_expert.gate_proj.weight"] = object()
    tensors[f"{prefix}.mlp.shared_expert.up_proj.weight"] = object()
    tensors[f"{prefix}.mlp.shared_expert.down_proj.weight"] = object()
    tensors[f"{prefix}.mlp.shared_expert_gate.weight"] = object()


def _add_synthetic_linear_attention_keys(tensors, spec, layer, prefix):
    tensors[f"{prefix}.input_layernorm.weight"] = object()
    tensors[f"{prefix}.linear_attn.in_proj_qkv.weight"] = object()
    tensors[f"{prefix}.linear_attn.in_proj_z.weight"] = object()
    tensors[f"{prefix}.linear_attn.out_proj.weight"] = object()
    tensors[f"{prefix}.linear_attn.norm.weight"] = object()
    tensors[f"{prefix}.linear_attn.A_log"] = object()
    tensors[f"{prefix}.linear_attn.dt_bias"] = object()
    tensors[f"{prefix}.linear_attn.in_proj_a.weight"] = object()
    tensors[f"{prefix}.linear_attn.in_proj_b.weight"] = object()
    tensors[f"{prefix}.linear_attn.conv1d.weight"] = object()


def _add_synthetic_attention_keys(tensors, spec, layer, prefix):
    tensors[f"{prefix}.input_layernorm.weight"] = object()
    tensors[f"{prefix}.self_attn.q_proj.weight"] = object()
    tensors[f"{prefix}.self_attn.k_proj.weight"] = object()
    tensors[f"{prefix}.self_attn.v_proj.weight"] = object()
    tensors[f"{prefix}.self_attn.o_proj.weight"] = object()
    tensors[f"{prefix}.self_attn.q_norm.weight"] = object()
    tensors[f"{prefix}.self_attn.k_norm.weight"] = object()


def test_qwen36_root_weight_key_resolution_for_real_hf_layout():
    tensors = {
        "model.language_model.embed_tokens.weight": object(),
        "model.language_model.norm.weight": object(),
        "lm_head.weight": object(),
    }

    keys = resolve_qwen36_root_weight_keys(tensors)

    assert keys.embedding_weight == "model.language_model.embed_tokens.weight"
    assert keys.output_norm_weight == "model.language_model.norm.weight"
    assert keys.lm_head_weight == "lm_head.weight"


def test_qwen36_full_weight_plan_resolves_all_layers_for_grouped_real_layout():
    spec = synthetic_qwen36_spec()
    tensors = {
        "model.language_model.embed_tokens.weight": object(),
        "model.language_model.norm.weight": object(),
        "lm_head.weight": object(),
    }
    for layer, kind in enumerate(spec.layer_kinds()):
        prefix = f"model.language_model.layers.{layer}"
        if kind == "deltanet_moe":
            _add_synthetic_linear_attention_keys(tensors, spec, layer, prefix)
        else:
            _add_synthetic_attention_keys(tensors, spec, layer, prefix)
        _add_synthetic_moe_keys(tensors, spec, layer, prefix)

    plan = resolve_qwen36_full_weight_plan(tensors, spec=spec)

    assert plan.layer_counts() == {"deltanet_moe": 3, "attention_moe": 1}
    assert len(plan.layers) == spec.num_layers
    assert plan.layers[0].linear_attention is not None
    assert plan.layers[3].attention is not None
    assert plan.roots.lm_head_weight == "lm_head.weight"
    assert len(plan.keys()) == len(set(plan.keys()))
    assert plan.required_shards({key: f"shard-{idx % 2}.safetensors" for idx, key in enumerate(plan.keys())}) == (
        "shard-0.safetensors",
        "shard-1.safetensors",
    )


def test_qwen36_moe_key_flattening_includes_shared_expert_gate():
    spec = synthetic_qwen36_spec()
    layer = 0
    prefix = f"model.language_model.layers.{layer}"
    tensors = {}
    _add_synthetic_moe_keys(tensors, spec, layer, prefix)

    keys = resolve_qwen36_moe_weight_keys(tensors, layer, spec=spec)
    flattened = flatten_qwen36_moe_weight_keys(keys)

    assert f"{prefix}.mlp.experts.gate_up_proj" in flattened
    assert f"{prefix}.mlp.experts.down_proj" in flattened
    assert f"{prefix}.mlp.shared_expert_gate.weight" in flattened


def test_qwen36_moe_weight_loader_uses_safetensors_index(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")

    spec = synthetic_qwen36_spec()
    layer = 0
    shard_a = {}
    shard_b = {
        f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight": torch.full(
            (spec.expert_intermediate_size, spec.hidden_size),
            3.0,
        ),
        f"model.layers.{layer}.mlp.shared_expert.up_proj.weight": torch.full(
            (spec.expert_intermediate_size, spec.hidden_size),
            4.0,
        ),
        f"model.layers.{layer}.mlp.shared_expert.down_proj.weight": torch.full(
            (spec.hidden_size, spec.expert_intermediate_size),
            5.0,
        ),
    }
    shard_a[f"model.layers.{layer}.post_attention_layernorm.weight"] = torch.arange(
        spec.hidden_size,
        dtype=torch.float32,
    )
    shard_a[f"model.layers.{layer}.mlp.gate.weight"] = torch.arange(
        spec.num_experts * spec.hidden_size,
        dtype=torch.float32,
    ).reshape(spec.num_experts, spec.hidden_size)
    for expert_idx in range(spec.num_experts):
        shard_b[f"model.layers.{layer}.mlp.experts.{expert_idx}.gate_proj.weight"] = torch.full(
            (spec.expert_intermediate_size, spec.hidden_size),
            float(expert_idx + 1),
        )
        shard_b[f"model.layers.{layer}.mlp.experts.{expert_idx}.up_proj.weight"] = torch.full(
            (spec.expert_intermediate_size, spec.hidden_size),
            float(expert_idx + 11),
        )
        shard_b[f"model.layers.{layer}.mlp.experts.{expert_idx}.down_proj.weight"] = torch.full(
            (spec.hidden_size, spec.expert_intermediate_size),
            float(expert_idx + 21),
        )

    safetensors.save_file(shard_a, str(tmp_path / "model-00001-of-00002.safetensors"))
    safetensors.save_file(shard_b, str(tmp_path / "model-00002-of-00002.safetensors"))
    weight_map = {key: "model-00001-of-00002.safetensors" for key in shard_a}
    weight_map.update({key: "model-00002-of-00002.safetensors" for key in shard_b})
    (tmp_path / "model.safetensors.index.json").write_text(
        __import__("json").dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )

    packed = load_qwen36_moe_weights_from_safetensors(tmp_path, layer, spec=spec)

    assert packed.expert_gate_up_weight.shape == (spec.num_experts, spec.gated_up_features, spec.hidden_size)
    torch.testing.assert_close(
        packed.expert_gate_up_weight[1, : spec.expert_intermediate_size],
        shard_b[f"model.layers.{layer}.mlp.experts.1.gate_proj.weight"],
    )
    torch.testing.assert_close(
        packed.shared_gate_up_weight[spec.expert_intermediate_size :],
        shard_b[f"model.layers.{layer}.mlp.shared_expert.up_proj.weight"],
    )

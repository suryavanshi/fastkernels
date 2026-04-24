import pytest


torch = pytest.importorskip("torch")

from fastkernels.models import synthetic_qwen36_spec
from fastkernels.reference import (
    initial_qwen36_decode_state,
    make_synthetic_qwen36_decode_weights,
    reference_qwen36_decode_step,
)


def test_synthetic_qwen36_decode_state_shapes():
    spec = synthetic_qwen36_spec()
    state = initial_qwen36_decode_state(spec, max_positions=8)

    assert len(state.deltanet_states) == 3
    assert len(state.attention_key_cache) == 1
    assert len(state.attention_value_cache) == 1
    assert state.deltanet_states[0].shape == (2, 4, 8)
    assert state.attention_key_cache[0].shape == (8, 2, 4)
    assert state.position == 0


def test_synthetic_qwen36_decode_step_is_deterministic():
    spec = synthetic_qwen36_spec()
    weights_a = make_synthetic_qwen36_decode_weights(spec, seed=123)
    weights_b = make_synthetic_qwen36_decode_weights(spec, seed=123)
    state_a = initial_qwen36_decode_state(spec, max_positions=8)
    state_b = initial_qwen36_decode_state(spec, max_positions=8)

    logits_a, next_state_a = reference_qwen36_decode_step(7, state_a, weights_a, spec)
    logits_b, next_state_b = reference_qwen36_decode_step(7, state_b, weights_b, spec)

    torch.testing.assert_close(logits_a, logits_b, atol=0, rtol=0)
    torch.testing.assert_close(next_state_a.deltanet_states[0], next_state_b.deltanet_states[0], atol=0, rtol=0)
    assert logits_a.shape == (spec.vocab_size,)
    assert next_state_a.position == 1


def test_synthetic_qwen36_decode_updates_attention_cache_position():
    spec = synthetic_qwen36_spec()
    weights = make_synthetic_qwen36_decode_weights(spec, seed=0)
    state = initial_qwen36_decode_state(spec, max_positions=8)

    _, state = reference_qwen36_decode_step(1, state, weights, spec)
    _, state = reference_qwen36_decode_step(2, state, weights, spec)

    assert state.position == 2
    assert torch.count_nonzero(state.attention_key_cache[0][0]) > 0
    assert torch.count_nonzero(state.attention_key_cache[0][1]) > 0
    assert torch.count_nonzero(state.attention_key_cache[0][2:]) == 0

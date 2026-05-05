"""Synthetic Qwen3.6 decode reference benchmark."""

from __future__ import annotations

import argparse
import time

from fastkernels.models import synthetic_qwen36_spec
from fastkernels.reference import (
    Qwen36DecodeState,
    initial_qwen36_decode_state,
    make_synthetic_qwen36_decode_weights,
    reference_qwen36_attention_decode,
    reference_qwen36_decode_step,
    reference_qwen36_deltanet_decode,
    reference_qwen36_moe_decode,
)
from fastkernels.reference.qwen36_decode import _linear, _rms_norm


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen3.6 decode reference benchmark") from exc
    return torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-impl", choices=["reference", "triton"], default="reference")
    parser.add_argument("--deltanet-impl", choices=["reference", "triton"], default="reference")
    parser.add_argument("--moe-impl", choices=["reference", "triton"], default="reference")
    args = parser.parse_args()

    torch = _require_torch()
    dtype = getattr(torch, args.dtype)
    spec = synthetic_qwen36_spec()
    if args.steps > spec.context_length:
        raise ValueError(f"steps must be <= synthetic context_length ({spec.context_length})")

    weights = make_synthetic_qwen36_decode_weights(spec, device=args.device, dtype=dtype, seed=args.seed)
    token_ids = [(args.seed + idx) % spec.vocab_size for idx in range(args.steps)]
    triton_attention = None
    triton_moe = None
    triton_deltanet = None
    if args.attention_impl == "triton":
        from fastkernels.kernels.triton import triton_synthetic_qwen36_attention_decode

        triton_attention = triton_synthetic_qwen36_attention_decode
    if args.deltanet_impl == "triton":
        from fastkernels.kernels.triton import triton_synthetic_qwen36_deltanet_decode

        triton_deltanet = triton_synthetic_qwen36_deltanet_decode
    if args.moe_impl == "triton":
        from fastkernels.kernels.triton import triton_synthetic_qwen36_moe_decode

        triton_moe = triton_synthetic_qwen36_moe_decode

    def decode_step(token_id, state):
        if args.attention_impl == "reference" and args.deltanet_impl == "reference" and args.moe_impl == "reference":
            return reference_qwen36_decode_step(token_id, state, weights, spec)

        hidden = weights["embedding"][token_id]
        next_deltanet_states = []
        next_attention_keys = []
        next_attention_values = []
        deltanet_idx = 0
        attention_idx = 0

        for layer in weights["layers"]:
            if layer["kind"] == "deltanet_moe":
                if args.deltanet_impl == "triton":
                    hidden, new_delta_state = triton_deltanet(
                        hidden,
                        state.deltanet_states[deltanet_idx],
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
                else:
                    hidden, new_delta_state = reference_qwen36_deltanet_decode(
                        hidden,
                        state.deltanet_states[deltanet_idx],
                        layer,
                        spec,
                    )
                next_deltanet_states.append(new_delta_state)
                deltanet_idx += 1
            else:
                if args.attention_impl == "triton":
                    hidden, new_key_cache, new_value_cache = triton_attention(
                        hidden,
                        state.attention_key_cache[attention_idx],
                        state.attention_value_cache[attention_idx],
                        layer["norm_weight"],
                        layer["q_weight"],
                        layer["k_weight"],
                        layer["v_weight"],
                        layer["out_weight"],
                        position=state.position,
                        attention_heads=spec.attention_heads,
                        kv_heads=spec.attention_kv_heads,
                        head_dim=spec.attention_head_dim,
                        rope_dim=spec.rope_dim,
                        copy_cache=False,
                    )
                else:
                    hidden, new_key_cache, new_value_cache = reference_qwen36_attention_decode(
                        hidden,
                        state.attention_key_cache[attention_idx],
                        state.attention_value_cache[attention_idx],
                        layer,
                        spec,
                        state.position,
                    )
                next_attention_keys.append(new_key_cache)
                next_attention_values.append(new_value_cache)
                attention_idx += 1

            if args.moe_impl == "triton":
                hidden = triton_moe(
                    hidden,
                    layer["norm_weight"],
                    layer["router_weight"],
                    layer["expert_gate_up_weight"],
                    layer["expert_down_weight"],
                    layer["shared_gate_up_weight"],
                    layer["shared_down_weight"],
                    top_k=spec.num_routed_experts,
                )
            else:
                hidden = reference_qwen36_moe_decode(hidden, layer, spec)

        hidden = _rms_norm(hidden, weights["output_norm_weight"])
        logits = _linear(hidden, weights["lm_head_weight"])
        next_state = Qwen36DecodeState(
            deltanet_states=tuple(next_deltanet_states),
            attention_key_cache=tuple(next_attention_keys),
            attention_value_cache=tuple(next_attention_values),
            position=state.position + 1,
        )
        return logits, next_state

    def run_decode():
        state = initial_qwen36_decode_state(spec, max_positions=spec.context_length, device=args.device, dtype=dtype)
        logits = None
        for token_id in token_ids:
            logits, state = decode_step(token_id, state)
        return logits, state

    reference_logits = None
    if args.attention_impl == "triton" or args.deltanet_impl == "triton" or args.moe_impl == "triton":
        reference_state = initial_qwen36_decode_state(
            spec,
            max_positions=spec.context_length,
            device=args.device,
            dtype=dtype,
        )
        for token_id in token_ids:
            reference_logits, reference_state = reference_qwen36_decode_step(token_id, reference_state, weights, spec)

    for _ in range(args.warmup):
        run_decode()
    if args.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    logits, state = run_decode()
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"model: {spec.name}")
    print(f"device: {args.device}")
    print(f"dtype: {dtype}")
    print(f"attention_impl: {args.attention_impl}")
    print(f"deltanet_impl: {args.deltanet_impl}")
    print(f"moe_impl: {args.moe_impl}")
    print(f"steps: {args.steps}")
    print(f"final_position: {state.position}")
    print(f"logits_shape: {tuple(logits.shape)}")
    if reference_logits is not None:
        max_abs_diff = torch.max(torch.abs(logits.float() - reference_logits.float())).item()
        print(f"reference_max_abs_diff: {max_abs_diff:.6f}")
    print(f"decode_ms_per_step: {elapsed * 1000 / args.steps:.4f}")


if __name__ == "__main__":
    main()

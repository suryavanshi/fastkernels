"""Smoke-test Qwen3.6 MoE safetensors layout loading with real-shaped tensors."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from fastkernels.kernels.triton import triton_qwen36_batched_moe_decode
from fastkernels.models import load_qwen36_moe_weights_from_safetensors, qwen36_35b_a3b_spec


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen3.6 safetensor smoke test") from exc
    return torch


def _require_save_file():
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for the Qwen3.6 safetensor smoke test") from exc
    return save_file


def _reference_expert(torch, hidden, gate_up_weight, down_weight):
    gate_up = torch.matmul(gate_up_weight.float(), hidden.float())
    gate, up = torch.chunk(gate_up, 2, dim=0)
    activation = torch.nn.functional.silu(gate) * up
    return torch.matmul(down_weight.float(), activation)


def _make_fake_layer_state(torch, spec, layer: int, dtype, generator, scale: float):
    state = {
        f"model.layers.{layer}.post_attention_layernorm.weight": torch.randn(
            spec.hidden_size,
            generator=generator,
            dtype=dtype,
        ),
        f"model.layers.{layer}.mlp.gate.weight": torch.randn(
            spec.num_experts,
            spec.hidden_size,
            generator=generator,
            dtype=dtype,
        ),
        f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight": scale
        * torch.randn(
            spec.expert_intermediate_size,
            spec.hidden_size,
            generator=generator,
            dtype=dtype,
        ),
        f"model.layers.{layer}.mlp.shared_expert.up_proj.weight": scale
        * torch.randn(
            spec.expert_intermediate_size,
            spec.hidden_size,
            generator=generator,
            dtype=dtype,
        ),
        f"model.layers.{layer}.mlp.shared_expert.down_proj.weight": scale
        * torch.randn(
            spec.hidden_size,
            spec.expert_intermediate_size,
            generator=generator,
            dtype=dtype,
        ),
    }
    for expert_idx in range(spec.num_experts):
        state[f"model.layers.{layer}.mlp.experts.{expert_idx}.gate_proj.weight"] = scale * torch.randn(
            spec.expert_intermediate_size,
            spec.hidden_size,
            generator=generator,
            dtype=dtype,
        )
        state[f"model.layers.{layer}.mlp.experts.{expert_idx}.up_proj.weight"] = scale * torch.randn(
            spec.expert_intermediate_size,
            spec.hidden_size,
            generator=generator,
            dtype=dtype,
        )
        state[f"model.layers.{layer}.mlp.experts.{expert_idx}.down_proj.weight"] = scale * torch.randn(
            spec.hidden_size,
            spec.expert_intermediate_size,
            generator=generator,
            dtype=dtype,
        )
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch = _require_torch()
    save_file = _require_save_file()
    dtype = getattr(torch, args.dtype)
    spec = qwen36_35b_a3b_spec()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    scale = 0.02

    hidden = (scale * torch.randn(args.tokens, spec.hidden_size, generator=generator, dtype=dtype)).to(args.device)
    state = _make_fake_layer_state(torch, spec, args.layer, dtype, generator, scale)

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_dir = Path(tmpdir)
        shard_a = {
            key: value
            for key, value in state.items()
            if key.endswith("post_attention_layernorm.weight") or key.endswith("mlp.gate.weight")
        }
        shard_b = {key: value for key, value in state.items() if key not in shard_a}
        shard_a_name = "model-00001-of-00002.safetensors"
        shard_b_name = "model-00002-of-00002.safetensors"
        save_file(shard_a, str(weights_dir / shard_a_name))
        save_file(shard_b, str(weights_dir / shard_b_name))
        weight_map = {key: shard_a_name for key in shard_a}
        weight_map.update({key: shard_b_name for key in shard_b})
        (weights_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"format": "pt"}, "weight_map": weight_map}),
            encoding="utf-8",
        )
        weights = load_qwen36_moe_weights_from_safetensors(weights_dir, args.layer, spec=spec, device=args.device)

    hidden_f = hidden.float()
    normed = (
        hidden_f
        * torch.rsqrt(torch.mean(hidden_f * hidden_f, dim=1, keepdim=True) + 1e-6)
        * weights.norm_weight.float()
    )
    reference_logits = torch.matmul(normed, weights.router_weight.float().t())
    reference_probs = torch.softmax(reference_logits, dim=-1)
    reference_values, reference_ids = torch.topk(reference_probs, k=spec.num_routed_experts, dim=1)
    reference_weights = reference_values / torch.sum(reference_values, dim=1, keepdim=True)
    reference = torch.zeros(args.tokens, spec.hidden_size, device=args.device, dtype=torch.float32)
    for token in range(args.tokens):
        for idx in range(spec.num_routed_experts):
            expert_id = int(reference_ids[token, idx].item())
            reference[token] = reference[token] + reference_weights[token, idx] * _reference_expert(
                torch,
                hidden[token],
                weights.expert_gate_up_weight[expert_id],
                weights.expert_down_weight[expert_id],
            )
        reference[token] = reference[token] + _reference_expert(
            torch,
            hidden[token],
            weights.shared_gate_up_weight,
            weights.shared_down_weight,
        )

    for _ in range(args.warmup):
        triton_qwen36_batched_moe_decode(
            hidden,
            weights.norm_weight,
            weights.router_weight,
            weights.expert_gate_up_weight,
            weights.expert_down_weight,
            weights.shared_gate_up_weight,
            weights.shared_down_weight,
            top_k=spec.num_routed_experts,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    candidate = logits = topk_ids = topk_weights = None
    for _ in range(args.iters):
        candidate, logits, topk_ids, topk_weights = triton_qwen36_batched_moe_decode(
            hidden,
            weights.norm_weight,
            weights.router_weight,
            weights.expert_gate_up_weight,
            weights.expert_down_weight,
            weights.shared_gate_up_weight,
            weights.shared_down_weight,
            top_k=spec.num_routed_experts,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    batch_ms = elapsed * 1000 / args.iters
    output_diff = torch.max(torch.abs(candidate.float() - reference.float())).item()
    logits_diff = torch.max(torch.abs(logits.float() - reference_logits.float())).item()
    ids_match = bool(torch.equal(topk_ids.cpu(), reference_ids.cpu()))
    weights_diff = torch.max(torch.abs(topk_weights.float() - reference_weights.float())).item()

    print(f"model: {spec.name}")
    print(f"device: {args.device}")
    print(f"dtype: {dtype}")
    print(f"tokens: {args.tokens}")
    print(f"layer: {args.layer}")
    print(f"hidden_size: {spec.hidden_size}")
    print(f"experts: {spec.num_experts}")
    print(f"top_k: {spec.num_routed_experts}")
    print(f"intermediate: {spec.expert_intermediate_size}")
    print(f"resolved_norm_key: {weights.keys.norm_weight}")
    print(f"resolved_router_key: {weights.keys.router_weight}")
    print("safetensor_indexed: True")
    print(f"iters: {args.iters}")
    print(f"output_max_abs_diff: {output_diff:.6f}")
    print(f"logits_max_abs_diff: {logits_diff:.6f}")
    print(f"topk_ids_match: {ids_match}")
    print(f"topk_weights_max_abs_diff: {weights_diff:.6f}")
    print(f"safetensor_batched_real_moe_ms: {batch_ms:.4f}")
    print(f"safetensor_batched_real_moe_ms_per_token: {batch_ms / args.tokens:.4f}")


if __name__ == "__main__":
    main()

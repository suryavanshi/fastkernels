"""Smoke-test real Qwen3.6 non-MoE layer block weight loading."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from fastkernels.kernels.triton import (
    triton_qwen36_batched_attention_moe_layer_decode,
    triton_qwen36_batched_attention_decode,
    triton_qwen36_batched_attention_project,
    triton_qwen36_batched_deltanet_conv,
    triton_qwen36_batched_deltanet_moe_layer_decode,
    triton_qwen36_batched_deltanet_project,
    triton_qwen36_batched_deltanet_recurrent_output,
)
from fastkernels.models import (
    load_qwen36_moe_weights_from_safetensors,
    load_qwen36_attention_weights_from_safetensors,
    load_qwen36_linear_attention_weights_from_safetensors,
    qwen36_35b_a3b_spec,
    resolve_qwen36_moe_weight_keys,
    resolve_qwen36_attention_weight_keys,
    resolve_qwen36_linear_attention_weight_keys,
)
from fastkernels.reference import qwen36_real_deltanet_update_from_convolved_projections


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen3.6 block smoke test") from exc
    return torch


def _require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for the Qwen3.6 block smoke test") from exc
    return hf_hub_download


def _download_index(repo_id: str, revision: str, cache_dir: str | None) -> tuple[Path, dict[str, Any]]:
    hf_hub_download = _require_huggingface_hub()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    index_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors.index.json",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    with Path(index_path).open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    return Path(index_path).parent, index


def _download_keys(repo_id: str, revision: str, cache_dir: str | None, weight_map: dict[str, str], keys: tuple[str, ...]):
    hf_hub_download = _require_huggingface_hub()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    shard_names = sorted({weight_map[key] for key in keys})
    for shard_name in shard_names:
        hf_hub_download(
            repo_id=repo_id,
            filename=shard_name,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
    return shard_names


def _rms_norm(torch, hidden, norm_weight):
    hidden_f = hidden.float()
    return (
        hidden_f
        * torch.rsqrt(torch.mean(hidden_f * hidden_f, dim=-1, keepdim=True) + 1e-6)
        * (1.0 + norm_weight.float())
    )


def _key_tuple(keys) -> tuple[str, ...]:
    return tuple(str(value) for value in keys.__dict__.values())


def _flatten_moe_keys(keys) -> tuple[str, ...]:
    names = [keys.norm_weight, keys.router_weight, keys.shared_down_weight]
    if isinstance(keys.expert_gate_up_weight, str):
        names.append(keys.expert_gate_up_weight)
    else:
        for item in keys.expert_gate_up_weight:
            names.extend((item,) if isinstance(item, str) else item)
    if isinstance(keys.expert_down_weight, str):
        names.append(keys.expert_down_weight)
    else:
        names.extend(keys.expert_down_weight)
    if isinstance(keys.shared_gate_up_weight, str):
        names.append(keys.shared_gate_up_weight)
    else:
        names.extend(keys.shared_gate_up_weight)
    if keys.shared_expert_gate_weight is not None:
        names.append(keys.shared_expert_gate_weight)
    return tuple(names)


def _shape_tuple(tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _reference_rope(torch, values, positions, rope_dim, rope_theta):
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


def _reference_attention_decode(torch, hidden, key_cache, value_cache, weights, spec, start_position):
    q_width = weights.o_proj_weight.shape[1]
    kv_width = spec.attention_kv_heads * spec.attention_head_dim
    hidden_f = hidden.float()
    normed = _rms_norm(torch, hidden, weights.input_norm_weight)
    q_pairs = torch.matmul(normed, weights.q_proj_weight.float().t()).reshape(
        hidden.shape[0],
        spec.attention_heads,
        2 * spec.attention_head_dim,
    )
    q_raw = q_pairs[:, :, : spec.attention_head_dim].reshape(hidden.shape[0], q_width)
    gate = q_pairs[:, :, spec.attention_head_dim :].reshape(hidden.shape[0], q_width)
    k_raw = torch.matmul(normed, weights.k_proj_weight.float().t())
    v_raw = torch.matmul(normed, weights.v_proj_weight.float().t())
    q = q_raw.reshape(hidden.shape[0], spec.attention_heads, spec.attention_head_dim)
    k = k_raw.reshape(hidden.shape[0], spec.attention_kv_heads, spec.attention_head_dim)
    q = _rms_norm(torch, q, weights.q_norm_weight)
    k = _rms_norm(torch, k, weights.k_norm_weight)
    positions = torch.arange(start_position, start_position + hidden.shape[0], device=hidden.device)
    q = _reference_rope(torch, q, positions, spec.rope_dim, spec.rope_theta)
    k = _reference_rope(torch, k, positions, spec.rope_dim, spec.rope_theta)

    next_key_cache = key_cache.clone()
    next_value_cache = value_cache.clone()
    next_key_cache[start_position : start_position + hidden.shape[0]] = k
    next_value_cache[start_position : start_position + hidden.shape[0]] = v_raw.reshape(
        hidden.shape[0],
        spec.attention_kv_heads,
        spec.attention_head_dim,
    )
    attended = torch.empty(
        hidden.shape[0],
        spec.attention_heads,
        spec.attention_head_dim,
        device=hidden.device,
        dtype=torch.float32,
    )
    heads_per_kv = spec.attention_heads_per_kv_head
    for token in range(hidden.shape[0]):
        position = start_position + token
        for head in range(spec.attention_heads):
            kv_head = head // heads_per_kv
            scores = torch.sum(next_key_cache[: position + 1, kv_head] * q[token, head], dim=-1) / (
                spec.attention_head_dim**0.5
            )
            attn_weights = torch.softmax(scores, dim=0)
            attended[token, head] = torch.sum(
                next_value_cache[: position + 1, kv_head] * attn_weights[:, None],
                dim=0,
            )
    return (
        torch.matmul(attended.reshape(hidden.shape[0], q_width) * torch.sigmoid(gate.float()), weights.o_proj_weight.float().t()),
        next_key_cache,
        next_value_cache,
        kv_width,
    )


def _reference_expert(torch, hidden, gate_up_weight, down_weight):
    gate_up = torch.matmul(gate_up_weight.float(), hidden.float())
    gate, up = torch.chunk(gate_up, 2, dim=0)
    activation = torch.nn.functional.silu(gate) * up
    return torch.matmul(down_weight.float(), activation)


def _reference_moe_update(torch, hidden, weights, spec):
    hidden_f = hidden.float()
    normed = _rms_norm(torch, hidden_f, weights.norm_weight)
    logits = torch.matmul(normed, weights.router_weight.float().t())
    probs = torch.softmax(logits, dim=-1)
    topk_values, topk_ids = torch.topk(probs, k=spec.num_routed_experts, dim=1)
    topk_weights = topk_values / torch.sum(topk_values, dim=1, keepdim=True)
    routed = torch.zeros_like(hidden_f)
    shared = torch.zeros_like(hidden_f)
    for token in range(hidden.shape[0]):
        for idx in range(spec.num_routed_experts):
            expert_id = int(topk_ids[token, idx].item())
            routed[token] = routed[token] + topk_weights[token, idx] * _reference_expert(
                torch,
                hidden[token],
                weights.expert_gate_up_weight[expert_id],
                weights.expert_down_weight[expert_id],
            )
        shared[token] = _reference_expert(torch, hidden[token], weights.shared_gate_up_weight, weights.shared_down_weight)
    if weights.shared_expert_gate_weight is not None:
        gate_weight = weights.shared_expert_gate_weight.float().reshape(-1, spec.hidden_size)
        shared_scale = torch.sigmoid(torch.matmul(hidden_f, gate_weight.t()))
        shared = shared_scale[:, :1] * shared
    return routed + shared, logits, topk_ids, topk_weights


def _run_linear_layer(torch, args, spec, weights_dir: Path, weight_map: dict[str, str], layer: int) -> None:
    key_index = {key: None for key in weight_map}
    keys = resolve_qwen36_linear_attention_weight_keys(key_index, layer)
    needed_keys = _key_tuple(keys)
    shard_names = _download_keys(args.repo_id, args.revision, args.cache_dir, weight_map, needed_keys)
    weights = load_qwen36_linear_attention_weights_from_safetensors(weights_dir, layer, spec=spec, device=args.device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + layer)
    dtype = getattr(torch, args.dtype)
    hidden = (args.hidden_scale * torch.randn(args.tokens, spec.hidden_size, generator=generator, dtype=dtype)).to(
        args.device
    )
    normed = _rms_norm(torch, hidden, weights.input_norm_weight)
    qkv = torch.matmul(normed, weights.in_proj_qkv_weight.float().t())
    z = torch.matmul(normed, weights.in_proj_z_weight.float().t())
    a = torch.matmul(normed, weights.in_proj_a_weight.float().t())
    b = torch.matmul(normed, weights.in_proj_b_weight.float().t())
    conv_state = torch.zeros(qkv.shape[1], weights.conv1d_weight.shape[-1], device=args.device, dtype=dtype)
    combined = torch.cat((conv_state.unsqueeze(0).float(), qkv.t().unsqueeze(0).float()), dim=-1)
    reference_conv = torch.nn.functional.conv1d(
        combined,
        weights.conv1d_weight.float(),
        padding=0,
        groups=qkv.shape[1],
    )
    reference_conv = torch.nn.functional.silu(reference_conv[:, :, -args.tokens:]).squeeze(0).t()
    reference_conv_state = combined[:, :, -conv_state.shape[1] :].squeeze(0).to(dtype)
    recurrent_state = torch.zeros(
        spec.deltanet_value_heads,
        spec.deltanet_head_dim,
        spec.deltanet_head_dim,
        device=args.device,
        dtype=dtype,
    )
    reference_update, _, reference_recurrent_state = qwen36_real_deltanet_update_from_convolved_projections(
        hidden,
        recurrent_state,
        weights,
        spec,
        reference_conv,
        z,
        a,
        b,
        reference_conv_state,
    )
    triton_qkv = triton_z = triton_a = triton_b = None
    triton_conv = triton_conv_state = None
    triton_update = triton_recurrent_state = None
    for _ in range(args.warmup):
        triton_qkv, triton_z, triton_a, triton_b = triton_qwen36_batched_deltanet_project(
            hidden,
            weights.input_norm_weight,
            weights.in_proj_qkv_weight,
            weights.in_proj_z_weight,
            weights.in_proj_a_weight,
            weights.in_proj_b_weight,
        )
        triton_conv, triton_conv_state = triton_qwen36_batched_deltanet_conv(
            triton_qkv,
            conv_state,
            weights.conv1d_weight,
        )
        triton_update, triton_recurrent_state = triton_qwen36_batched_deltanet_recurrent_output(
            triton_conv,
            triton_z,
            triton_a,
            triton_b,
            recurrent_state,
            weights.out_proj_weight,
            weights.linear_norm_weight,
            weights.a_log,
            weights.dt_bias,
            qk_heads=spec.deltanet_qk_heads,
            value_heads=spec.deltanet_value_heads,
            head_dim=spec.deltanet_head_dim,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        triton_qkv, triton_z, triton_a, triton_b = triton_qwen36_batched_deltanet_project(
            hidden,
            weights.input_norm_weight,
            weights.in_proj_qkv_weight,
            weights.in_proj_z_weight,
            weights.in_proj_a_weight,
            weights.in_proj_b_weight,
        )
        triton_conv, triton_conv_state = triton_qwen36_batched_deltanet_conv(
            triton_qkv,
            conv_state,
            weights.conv1d_weight,
        )
        triton_update, triton_recurrent_state = triton_qwen36_batched_deltanet_recurrent_output(
            triton_conv,
            triton_z,
            triton_a,
            triton_b,
            recurrent_state,
            weights.out_proj_weight,
            weights.linear_norm_weight,
            weights.a_log,
            weights.dt_bias,
            qk_heads=spec.deltanet_qk_heads,
            value_heads=spec.deltanet_value_heads,
            head_dim=spec.deltanet_head_dim,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    project_conv_ms = (time.perf_counter() - start) * 1000 / args.iters

    moe_keys = resolve_qwen36_moe_weight_keys(key_index, layer, spec=spec)
    moe_needed_keys = _flatten_moe_keys(moe_keys)
    moe_shards = _download_keys(args.repo_id, args.revision, args.cache_dir, weight_map, moe_needed_keys)
    moe_weights = load_qwen36_moe_weights_from_safetensors(weights_dir, layer, spec=spec, device=args.device)
    reference_deltanet_hidden = hidden.float() + reference_update.float()
    reference_moe_update, reference_logits, reference_topk_ids, reference_topk_weights = _reference_moe_update(
        torch,
        reference_deltanet_hidden,
        moe_weights,
        spec,
    )
    reference_layer_hidden = reference_deltanet_hidden + reference_moe_update
    triton_layer = None
    for _ in range(args.warmup):
        triton_layer = triton_qwen36_batched_deltanet_moe_layer_decode(
            hidden,
            conv_state,
            recurrent_state,
            weights.input_norm_weight,
            weights.in_proj_qkv_weight,
            weights.in_proj_z_weight,
            weights.in_proj_a_weight,
            weights.in_proj_b_weight,
            weights.conv1d_weight,
            weights.out_proj_weight,
            weights.linear_norm_weight,
            weights.a_log,
            weights.dt_bias,
            moe_weights.norm_weight,
            moe_weights.router_weight,
            moe_weights.expert_gate_up_weight,
            moe_weights.expert_down_weight,
            moe_weights.shared_gate_up_weight,
            moe_weights.shared_down_weight,
            moe_weights.shared_expert_gate_weight,
            qk_heads=spec.deltanet_qk_heads,
            value_heads=spec.deltanet_value_heads,
            head_dim=spec.deltanet_head_dim,
            top_k=spec.num_routed_experts,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        triton_layer = triton_qwen36_batched_deltanet_moe_layer_decode(
            hidden,
            conv_state,
            recurrent_state,
            weights.input_norm_weight,
            weights.in_proj_qkv_weight,
            weights.in_proj_z_weight,
            weights.in_proj_a_weight,
            weights.in_proj_b_weight,
            weights.conv1d_weight,
            weights.out_proj_weight,
            weights.linear_norm_weight,
            weights.a_log,
            weights.dt_bias,
            moe_weights.norm_weight,
            moe_weights.router_weight,
            moe_weights.expert_gate_up_weight,
            moe_weights.expert_down_weight,
            moe_weights.shared_gate_up_weight,
            moe_weights.shared_down_weight,
            moe_weights.shared_expert_gate_weight,
            qk_heads=spec.deltanet_qk_heads,
            value_heads=spec.deltanet_value_heads,
            head_dim=spec.deltanet_head_dim,
            top_k=spec.num_routed_experts,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    layer_ms = (time.perf_counter() - start) * 1000 / args.iters
    (
        triton_layer_hidden,
        triton_deltanet_hidden,
        triton_layer_update,
        triton_moe_update,
        triton_layer_conv_state,
        triton_layer_recurrent_state,
        triton_logits,
        triton_topk_ids,
        triton_topk_weights,
    ) = triton_layer

    print(f"layer: {layer}")
    print("layer_kind: deltanet_moe")
    print(f"downloaded_shards: {','.join(shard_names)}")
    print(f"needed_tensor_count: {len(needed_keys)}")
    print(f"resolved_input_norm_key: {keys.input_norm_weight}")
    print(f"resolved_in_proj_qkv_key: {keys.in_proj_qkv_weight}")
    print(f"resolved_in_proj_z_key: {keys.in_proj_z_weight}")
    print(f"resolved_out_proj_key: {keys.out_proj_weight}")
    print(f"linear_in_proj_qkv_shape: {_shape_tuple(weights.in_proj_qkv_weight)}")
    print(f"linear_in_proj_z_shape: {_shape_tuple(weights.in_proj_z_weight)}")
    print(f"linear_out_proj_shape: {_shape_tuple(weights.out_proj_weight)}")
    print(f"linear_conv1d_shape: {_shape_tuple(weights.conv1d_weight)}")
    print(f"linear_qkv_projection_shape: {_shape_tuple(qkv)}")
    print(f"linear_z_projection_shape: {_shape_tuple(z)}")
    print(f"linear_a_projection_shape: {_shape_tuple(a)}")
    print(f"linear_b_projection_shape: {_shape_tuple(b)}")
    print(f"linear_triton_qkv_max_abs_diff: {torch.max(torch.abs(triton_qkv.float() - qkv.float())).item():.6f}")
    print(f"linear_triton_z_max_abs_diff: {torch.max(torch.abs(triton_z.float() - z.float())).item():.6f}")
    print(f"linear_triton_a_max_abs_diff: {torch.max(torch.abs(triton_a.float() - a.float())).item():.6f}")
    print(f"linear_triton_b_max_abs_diff: {torch.max(torch.abs(triton_b.float() - b.float())).item():.6f}")
    print(
        f"linear_triton_conv_max_abs_diff: "
        f"{torch.max(torch.abs(triton_conv.float() - reference_conv.float())).item():.6f}"
    )
    print(
        f"linear_triton_conv_state_max_abs_diff: "
        f"{torch.max(torch.abs(triton_conv_state.float() - reference_conv_state.float())).item():.6f}"
    )
    print(
        f"linear_triton_update_max_abs_diff: "
        f"{torch.max(torch.abs(triton_update.float() - reference_update.float())).item():.6f}"
    )
    print(
        f"linear_triton_recurrent_state_max_abs_diff: "
        f"{torch.max(torch.abs(triton_recurrent_state.float() - reference_recurrent_state.float())).item():.6f}"
    )
    print(f"linear_triton_full_update_ms: {project_conv_ms:.4f}")
    print(f"linear_triton_full_update_ms_per_token: {project_conv_ms / args.tokens:.4f}")
    print(f"linear_moe_downloaded_shards: {','.join(moe_shards)}")
    print(f"linear_moe_needed_tensor_count: {len(moe_needed_keys)}")
    print(
        f"linear_moe_deltanet_hidden_max_abs_diff: "
        f"{torch.max(torch.abs(triton_deltanet_hidden.float() - reference_deltanet_hidden.float())).item():.6f}"
    )
    print(
        f"linear_moe_deltanet_update_max_abs_diff: "
        f"{torch.max(torch.abs(triton_layer_update.float() - reference_update.float())).item():.6f}"
    )
    print(
        f"linear_moe_update_max_abs_diff: "
        f"{torch.max(torch.abs(triton_moe_update.float() - reference_moe_update.float())).item():.6f}"
    )
    print(
        f"linear_moe_layer_hidden_max_abs_diff: "
        f"{torch.max(torch.abs(triton_layer_hidden.float() - reference_layer_hidden.float())).item():.6f}"
    )
    print(
        f"linear_moe_conv_state_max_abs_diff: "
        f"{torch.max(torch.abs(triton_layer_conv_state.float() - reference_conv_state.float())).item():.6f}"
    )
    print(
        f"linear_moe_recurrent_state_max_abs_diff: "
        f"{torch.max(torch.abs(triton_layer_recurrent_state.float() - reference_recurrent_state.float())).item():.6f}"
    )
    print(f"linear_moe_logits_max_abs_diff: {torch.max(torch.abs(triton_logits.float() - reference_logits.float())).item():.6f}")
    print(f"linear_moe_topk_ids_match: {bool(torch.equal(triton_topk_ids.cpu(), reference_topk_ids.cpu()))}")
    print(
        f"linear_moe_topk_weights_max_abs_diff: "
        f"{torch.max(torch.abs(triton_topk_weights.float() - reference_topk_weights.float())).item():.6f}"
    )
    print(f"linear_moe_layer_ms: {layer_ms:.4f}")
    print(f"linear_moe_layer_ms_per_token: {layer_ms / args.tokens:.4f}")


def _run_attention_layer(torch, args, spec, weights_dir: Path, weight_map: dict[str, str], layer: int) -> None:
    key_index = {key: None for key in weight_map}
    keys = resolve_qwen36_attention_weight_keys(key_index, layer)
    needed_keys = _key_tuple(keys)
    shard_names = _download_keys(args.repo_id, args.revision, args.cache_dir, weight_map, needed_keys)
    weights = load_qwen36_attention_weights_from_safetensors(weights_dir, layer, spec=spec, device=args.device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + layer)
    dtype = getattr(torch, args.dtype)
    hidden = (args.hidden_scale * torch.randn(args.tokens, spec.hidden_size, generator=generator, dtype=dtype)).to(
        args.device
    )
    normed = _rms_norm(torch, hidden, weights.input_norm_weight)
    q = torch.matmul(normed, weights.q_proj_weight.float().t())
    k = torch.matmul(normed, weights.k_proj_weight.float().t())
    v = torch.matmul(normed, weights.v_proj_weight.float().t())
    triton_q = triton_k = triton_v = None
    for _ in range(args.warmup):
        triton_q, triton_k, triton_v = triton_qwen36_batched_attention_project(
            hidden,
            weights.input_norm_weight,
            weights.q_proj_weight,
            weights.k_proj_weight,
            weights.v_proj_weight,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        triton_q, triton_k, triton_v = triton_qwen36_batched_attention_project(
            hidden,
            weights.input_norm_weight,
            weights.q_proj_weight,
            weights.k_proj_weight,
            weights.v_proj_weight,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    project_ms = elapsed * 1000 / args.iters

    max_positions = max(args.attention_cache_positions, args.attention_start_position + args.tokens)
    key_cache = (
        args.hidden_scale
        * torch.randn(
            max_positions,
            spec.attention_kv_heads,
            spec.attention_head_dim,
            generator=generator,
            dtype=torch.float32,
        )
    ).to(args.device)
    value_cache = (
        args.hidden_scale
        * torch.randn(
            max_positions,
            spec.attention_kv_heads,
            spec.attention_head_dim,
            generator=generator,
            dtype=torch.float32,
        )
    ).to(args.device)
    reference_update, reference_key_cache, reference_value_cache, kv_width = _reference_attention_decode(
        torch,
        hidden,
        key_cache,
        value_cache,
        weights,
        spec,
        args.attention_start_position,
    )
    triton_update = triton_key_cache = triton_value_cache = None
    for _ in range(args.warmup):
        triton_update, triton_key_cache, triton_value_cache = triton_qwen36_batched_attention_decode(
            hidden,
            key_cache,
            value_cache,
            weights.input_norm_weight,
            weights.q_proj_weight,
            weights.k_proj_weight,
            weights.v_proj_weight,
            weights.o_proj_weight,
            weights.q_norm_weight,
            weights.k_norm_weight,
            start_position=args.attention_start_position,
            attention_heads=spec.attention_heads,
            kv_heads=spec.attention_kv_heads,
            head_dim=spec.attention_head_dim,
            rope_dim=spec.rope_dim,
            rope_theta=spec.rope_theta,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        triton_update, triton_key_cache, triton_value_cache = triton_qwen36_batched_attention_decode(
            hidden,
            key_cache,
            value_cache,
            weights.input_norm_weight,
            weights.q_proj_weight,
            weights.k_proj_weight,
            weights.v_proj_weight,
            weights.o_proj_weight,
            weights.q_norm_weight,
            weights.k_norm_weight,
            start_position=args.attention_start_position,
            attention_heads=spec.attention_heads,
            kv_heads=spec.attention_kv_heads,
            head_dim=spec.attention_head_dim,
            rope_dim=spec.rope_dim,
            rope_theta=spec.rope_theta,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    decode_elapsed = time.perf_counter() - start
    decode_ms = decode_elapsed * 1000 / args.iters

    print(f"layer: {layer}")
    print("layer_kind: attention_moe")
    print(f"downloaded_shards: {','.join(shard_names)}")
    print(f"needed_tensor_count: {len(needed_keys)}")
    print(f"resolved_input_norm_key: {keys.input_norm_weight}")
    print(f"resolved_q_proj_key: {keys.q_proj_weight}")
    print(f"resolved_k_proj_key: {keys.k_proj_weight}")
    print(f"resolved_v_proj_key: {keys.v_proj_weight}")
    print(f"resolved_o_proj_key: {keys.o_proj_weight}")
    print(f"attention_q_proj_shape: {_shape_tuple(weights.q_proj_weight)}")
    print(f"attention_k_proj_shape: {_shape_tuple(weights.k_proj_weight)}")
    print(f"attention_v_proj_shape: {_shape_tuple(weights.v_proj_weight)}")
    print(f"attention_o_proj_shape: {_shape_tuple(weights.o_proj_weight)}")
    print(f"attention_q_projection_shape: {_shape_tuple(q)}")
    print(f"attention_k_projection_shape: {_shape_tuple(k)}")
    print(f"attention_v_projection_shape: {_shape_tuple(v)}")
    print(f"attention_q_runtime_width: {weights.o_proj_weight.shape[1]}")
    print(f"attention_kv_runtime_width: {kv_width}")
    print(f"attention_triton_q_max_abs_diff: {torch.max(torch.abs(triton_q.float() - q.float())).item():.6f}")
    print(f"attention_triton_k_max_abs_diff: {torch.max(torch.abs(triton_k.float() - k.float())).item():.6f}")
    print(f"attention_triton_v_max_abs_diff: {torch.max(torch.abs(triton_v.float() - v.float())).item():.6f}")
    print(f"attention_triton_project_ms: {project_ms:.4f}")
    print(f"attention_triton_project_ms_per_token: {project_ms / args.tokens:.4f}")
    print(f"attention_decode_start_position: {args.attention_start_position}")
    print(f"attention_decode_cache_positions: {max_positions}")
    print(
        f"attention_decode_update_max_abs_diff: "
        f"{torch.max(torch.abs(triton_update.float() - reference_update.float())).item():.6f}"
    )
    print(
        f"attention_decode_key_cache_max_abs_diff: "
        f"{torch.max(torch.abs(triton_key_cache.float() - reference_key_cache.float())).item():.6f}"
    )
    print(
        f"attention_decode_value_cache_max_abs_diff: "
        f"{torch.max(torch.abs(triton_value_cache.float() - reference_value_cache.float())).item():.6f}"
    )
    print(f"attention_triton_decode_ms: {decode_ms:.4f}")
    print(f"attention_triton_decode_ms_per_token: {decode_ms / args.tokens:.4f}")

    moe_keys = resolve_qwen36_moe_weight_keys(key_index, layer, spec=spec)
    moe_needed_keys = _flatten_moe_keys(moe_keys)
    moe_shards = _download_keys(args.repo_id, args.revision, args.cache_dir, weight_map, moe_needed_keys)
    moe_weights = load_qwen36_moe_weights_from_safetensors(weights_dir, layer, spec=spec, device=args.device)
    reference_attention_hidden = hidden.float() + reference_update.float()
    reference_moe_update, reference_logits, reference_topk_ids, reference_topk_weights = _reference_moe_update(
        torch,
        reference_attention_hidden,
        moe_weights,
        spec,
    )
    reference_layer_hidden = reference_attention_hidden + reference_moe_update
    for _ in range(args.warmup):
        triton_layer = triton_qwen36_batched_attention_moe_layer_decode(
            hidden,
            key_cache,
            value_cache,
            weights.input_norm_weight,
            weights.q_proj_weight,
            weights.k_proj_weight,
            weights.v_proj_weight,
            weights.o_proj_weight,
            weights.q_norm_weight,
            weights.k_norm_weight,
            moe_weights.norm_weight,
            moe_weights.router_weight,
            moe_weights.expert_gate_up_weight,
            moe_weights.expert_down_weight,
            moe_weights.shared_gate_up_weight,
            moe_weights.shared_down_weight,
            moe_weights.shared_expert_gate_weight,
            start_position=args.attention_start_position,
            attention_heads=spec.attention_heads,
            kv_heads=spec.attention_kv_heads,
            head_dim=spec.attention_head_dim,
            rope_dim=spec.rope_dim,
            rope_theta=spec.rope_theta,
            top_k=spec.num_routed_experts,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    triton_layer = None
    for _ in range(args.iters):
        triton_layer = triton_qwen36_batched_attention_moe_layer_decode(
            hidden,
            key_cache,
            value_cache,
            weights.input_norm_weight,
            weights.q_proj_weight,
            weights.k_proj_weight,
            weights.v_proj_weight,
            weights.o_proj_weight,
            weights.q_norm_weight,
            weights.k_norm_weight,
            moe_weights.norm_weight,
            moe_weights.router_weight,
            moe_weights.expert_gate_up_weight,
            moe_weights.expert_down_weight,
            moe_weights.shared_gate_up_weight,
            moe_weights.shared_down_weight,
            moe_weights.shared_expert_gate_weight,
            start_position=args.attention_start_position,
            attention_heads=spec.attention_heads,
            kv_heads=spec.attention_kv_heads,
            head_dim=spec.attention_head_dim,
            rope_dim=spec.rope_dim,
            rope_theta=spec.rope_theta,
            top_k=spec.num_routed_experts,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    layer_elapsed = time.perf_counter() - start
    layer_ms = layer_elapsed * 1000 / args.iters
    (
        triton_layer_hidden,
        triton_attention_hidden,
        triton_attention_update,
        triton_moe_update,
        triton_layer_key_cache,
        triton_layer_value_cache,
        triton_logits,
        triton_topk_ids,
        triton_topk_weights,
    ) = triton_layer

    print(f"attention_moe_downloaded_shards: {','.join(moe_shards)}")
    print(f"attention_moe_needed_tensor_count: {len(moe_needed_keys)}")
    print(f"attention_moe_resolved_norm_key: {moe_weights.keys.norm_weight}")
    print(f"attention_moe_resolved_router_key: {moe_weights.keys.router_weight}")
    print(
        f"attention_moe_attention_hidden_max_abs_diff: "
        f"{torch.max(torch.abs(triton_attention_hidden.float() - reference_attention_hidden.float())).item():.6f}"
    )
    print(
        f"attention_moe_attention_update_max_abs_diff: "
        f"{torch.max(torch.abs(triton_attention_update.float() - reference_update.float())).item():.6f}"
    )
    print(
        f"attention_moe_update_max_abs_diff: "
        f"{torch.max(torch.abs(triton_moe_update.float() - reference_moe_update.float())).item():.6f}"
    )
    print(
        f"attention_moe_layer_hidden_max_abs_diff: "
        f"{torch.max(torch.abs(triton_layer_hidden.float() - reference_layer_hidden.float())).item():.6f}"
    )
    print(
        f"attention_moe_key_cache_max_abs_diff: "
        f"{torch.max(torch.abs(triton_layer_key_cache.float() - reference_key_cache.float())).item():.6f}"
    )
    print(
        f"attention_moe_value_cache_max_abs_diff: "
        f"{torch.max(torch.abs(triton_layer_value_cache.float() - reference_value_cache.float())).item():.6f}"
    )
    print(f"attention_moe_logits_max_abs_diff: {torch.max(torch.abs(triton_logits.float() - reference_logits.float())).item():.6f}")
    print(f"attention_moe_topk_ids_match: {bool(torch.equal(triton_topk_ids.cpu(), reference_topk_ids.cpu()))}")
    print(
        f"attention_moe_topk_weights_max_abs_diff: "
        f"{torch.max(torch.abs(triton_topk_weights.float() - reference_topk_weights.float())).item():.6f}"
    )
    print(f"attention_moe_layer_ms: {layer_ms:.4f}")
    print(f"attention_moe_layer_ms_per_token: {layer_ms / args.tokens:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--linear-layer", type=int, default=0)
    parser.add_argument("--attention-layer", type=int, default=3)
    parser.add_argument("--attention-start-position", type=int, default=1)
    parser.add_argument("--attention-cache-positions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-scale", type=float, default=0.02)
    args = parser.parse_args()

    torch = _require_torch()
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    spec = qwen36_35b_a3b_spec()
    weights_dir, index = _download_index(args.repo_id, args.revision, args.cache_dir)
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json does not contain a valid weight_map")

    print(f"model: {spec.name}")
    print(f"repo_id: {args.repo_id}")
    print(f"revision: {args.revision}")
    print(f"device: {args.device}")
    print(f"dtype: {getattr(torch, args.dtype)}")
    print(f"tokens: {args.tokens}")
    print(f"hf_total_size_bytes: {index.get('metadata', {}).get('total_size', 'unknown')}")
    _run_linear_layer(torch, args, spec, weights_dir, weight_map, args.linear_layer)
    print("---")
    _run_attention_layer(torch, args, spec, weights_dir, weight_map, args.attention_layer)


if __name__ == "__main__":
    main()

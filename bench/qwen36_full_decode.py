"""Streaming real-weight Qwen3.6 full-layer reference decode."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from fastkernels.models import (
    flatten_qwen36_moe_weight_keys,
    load_qwen36_attention_weights_from_safetensors,
    load_qwen36_linear_attention_weights_from_safetensors,
    load_qwen36_moe_weights_from_safetensors,
    resolve_qwen36_full_weight_plan,
)
from fastkernels.models.qwen36 import Qwen36A3BSpec
from fastkernels.reference import (
    initial_qwen36_real_decode_state,
    qwen36_real_attention_moe_layer,
    qwen36_real_attention_update,
    qwen36_real_deltanet_moe_layer,
    qwen36_real_deltanet_update,
    qwen36_real_moe_update,
    qwen36_real_rms_norm,
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Qwen3.6 full decode runner") from exc
    return torch


def _require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for the Qwen3.6 full decode runner") from exc
    return hf_hub_download


def _download_json(repo_id: str, revision: str, cache_dir: str | None, filename: str) -> tuple[Path, dict[str, Any]]:
    hf_hub_download = _require_huggingface_hub()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
    )
    with path.open("r", encoding="utf-8") as handle:
        return path, json.load(handle)


def _download_keys(
    repo_id: str,
    revision: str,
    cache_dir: str | None,
    weight_map: dict[str, str],
    keys: tuple[str, ...],
) -> tuple[str, ...]:
    hf_hub_download = _require_huggingface_hub()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    shards = tuple(sorted({weight_map[key] for key in keys}))
    for shard in shards:
        hf_hub_download(repo_id=repo_id, filename=shard, revision=revision, cache_dir=cache_dir, token=token)
    return shards


def _load_selected_tensors(weights_dir: Path, weight_map: dict[str, str], keys: tuple[str, ...], device: str):
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for the Qwen3.6 full decode runner") from exc

    selected = set(keys)
    tensors = {}
    for shard in sorted({weight_map[key] for key in keys}):
        path = weights_dir / shard
        loaded = load_file(str(path), device="cpu")
        tensors.update({key: value.to(device) for key, value in loaded.items() if key in selected})
    missing = sorted(selected - set(tensors))
    if missing:
        raise KeyError(f"missing selected tensors: {missing[:8]}")
    return tensors


def _parse_token_ids(text: str) -> list[int]:
    return [int(piece.strip()) for piece in text.split(",") if piece.strip()]


def _tokenize_prompt(repo_id: str, revision: str, cache_dir: str | None, prompt: str) -> list[int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required when using --prompt instead of --prompt-token-ids") from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision, cache_dir=cache_dir, token=token)
    return list(tokenizer.encode(prompt, add_special_tokens=False))


def _layer_keys(layer) -> tuple[str, ...]:
    keys = []
    if layer.linear_attention is not None:
        keys.extend(str(value) for value in layer.linear_attention.__dict__.values())
    if layer.attention is not None:
        keys.extend(str(value) for value in layer.attention.__dict__.values())
    keys.extend(flatten_qwen36_moe_weight_keys(layer.moe))
    return tuple(dict.fromkeys(keys))


def _is_triton_four_layer_pattern(layers: tuple[Any, ...] | list[Any], index: int) -> bool:
    if index + 4 > len(layers):
        return False
    return tuple(layer.layer_kind for layer in layers[index : index + 4]) == (
        "deltanet_moe",
        "deltanet_moe",
        "deltanet_moe",
        "attention_moe",
    )


def _run_chunk(
    token_ids,
    state,
    *,
    embedding_weight,
    output_norm_weight,
    lm_head_weight,
    plan,
    weights_dir: Path,
    repo_id: str,
    revision: str,
    cache_dir: str | None,
    weight_map: dict[str, str],
    device: str,
    max_layers: int | None,
    attention_impl: str,
    deltanet_impl: str,
    moe_impl: str,
):
    torch = _require_torch()
    hidden = embedding_weight[token_ids].float()
    next_conv_states = []
    next_recurrent_states = []
    next_key_caches = []
    next_value_caches = []
    deltanet_idx = 0
    attention_idx = 0
    layers = plan.layers if max_layers is None else plan.layers[:max_layers]
    pattern_chunks = 0

    triton_moe_layer = None
    triton_deltanet_moe_layer = None
    if moe_impl == "triton":
        from fastkernels.kernels.triton import triton_qwen36_batched_moe_layer_decode

        triton_moe_layer = triton_qwen36_batched_moe_layer_decode

    triton_attention_update = None
    triton_attention_moe_layer = None
    if attention_impl == "triton":
        from fastkernels.kernels.triton import (
            triton_qwen36_batched_attention_decode,
            triton_qwen36_batched_attention_moe_layer_decode,
        )

        triton_attention_update = triton_qwen36_batched_attention_decode
        triton_attention_moe_layer = triton_qwen36_batched_attention_moe_layer_decode

    triton_deltanet_project = None
    triton_deltanet_conv = None
    triton_deltanet_recurrent_output = None
    if deltanet_impl == "triton":
        from fastkernels.kernels.triton import (
            triton_qwen36_batched_deltanet_moe_layer_decode,
            triton_qwen36_batched_deltanet_conv,
            triton_qwen36_batched_deltanet_project,
            triton_qwen36_batched_deltanet_recurrent_output,
        )

        triton_deltanet_moe_layer = triton_qwen36_batched_deltanet_moe_layer_decode
        triton_deltanet_project = triton_qwen36_batched_deltanet_project
        triton_deltanet_conv = triton_qwen36_batched_deltanet_conv
        triton_deltanet_recurrent_output = triton_qwen36_batched_deltanet_recurrent_output

    layer_pos = 0
    while layer_pos < len(layers):
        if (
            attention_impl == "triton"
            and deltanet_impl == "triton"
            and moe_impl == "triton"
            and _is_triton_four_layer_pattern(layers, layer_pos)
        ):
            for pattern_offset in range(4):
                layer = layers[layer_pos + pattern_offset]
                _download_keys(repo_id, revision, cache_dir, weight_map, _layer_keys(layer))
                moe_weights = load_qwen36_moe_weights_from_safetensors(
                    weights_dir,
                    layer.layer_idx,
                    spec=plan.spec,
                    device=device,
                )
                if layer.layer_kind == "deltanet_moe":
                    linear_weights = load_qwen36_linear_attention_weights_from_safetensors(
                        weights_dir,
                        layer.layer_idx,
                        spec=plan.spec,
                        device=device,
                    )
                    (
                        hidden,
                        _deltanet_hidden,
                        _deltanet_update,
                        _moe_update,
                        conv_state,
                        recurrent_state,
                        _logits,
                        _topk_ids,
                        _topk_weights,
                    ) = triton_deltanet_moe_layer(
                        hidden,
                        state.deltanet_conv_states[deltanet_idx],
                        state.deltanet_recurrent_states[deltanet_idx],
                        linear_weights.input_norm_weight,
                        linear_weights.in_proj_qkv_weight,
                        linear_weights.in_proj_z_weight,
                        linear_weights.in_proj_a_weight,
                        linear_weights.in_proj_b_weight,
                        linear_weights.conv1d_weight,
                        linear_weights.out_proj_weight,
                        linear_weights.linear_norm_weight,
                        linear_weights.a_log,
                        linear_weights.dt_bias,
                        moe_weights.norm_weight,
                        moe_weights.router_weight,
                        moe_weights.expert_gate_up_weight,
                        moe_weights.expert_down_weight,
                        moe_weights.shared_gate_up_weight,
                        moe_weights.shared_down_weight,
                        moe_weights.shared_expert_gate_weight,
                        qk_heads=plan.spec.deltanet_qk_heads,
                        value_heads=plan.spec.deltanet_value_heads,
                        head_dim=plan.spec.deltanet_head_dim,
                        top_k=plan.spec.num_routed_experts,
                    )
                    next_conv_states.append(conv_state)
                    next_recurrent_states.append(recurrent_state)
                    deltanet_idx += 1
                else:
                    attention_weights = load_qwen36_attention_weights_from_safetensors(
                        weights_dir,
                        layer.layer_idx,
                        spec=plan.spec,
                        device=device,
                    )
                    (
                        hidden,
                        _attention_hidden,
                        _attention_update,
                        _moe_update,
                        key_cache,
                        value_cache,
                        _logits,
                        _topk_ids,
                        _topk_weights,
                    ) = triton_attention_moe_layer(
                        hidden,
                        state.attention_key_cache[attention_idx],
                        state.attention_value_cache[attention_idx],
                        attention_weights.input_norm_weight,
                        attention_weights.q_proj_weight,
                        attention_weights.k_proj_weight,
                        attention_weights.v_proj_weight,
                        attention_weights.o_proj_weight,
                        attention_weights.q_norm_weight,
                        attention_weights.k_norm_weight,
                        moe_weights.norm_weight,
                        moe_weights.router_weight,
                        moe_weights.expert_gate_up_weight,
                        moe_weights.expert_down_weight,
                        moe_weights.shared_gate_up_weight,
                        moe_weights.shared_down_weight,
                        moe_weights.shared_expert_gate_weight,
                        start_position=state.position,
                        attention_heads=plan.spec.attention_heads,
                        kv_heads=plan.spec.attention_kv_heads,
                        head_dim=plan.spec.attention_head_dim,
                        rope_dim=plan.spec.rope_dim,
                        rope_theta=plan.spec.rope_theta,
                        top_k=plan.spec.num_routed_experts,
                    )
                    next_key_caches.append(key_cache)
                    next_value_caches.append(value_cache)
                    attention_idx += 1
                del moe_weights
                if device == "cuda":
                    torch.cuda.empty_cache()
            pattern_chunks += 1
            layer_pos += 4
            continue

        layer = layers[layer_pos]
        _download_keys(repo_id, revision, cache_dir, weight_map, _layer_keys(layer))
        moe_weights = load_qwen36_moe_weights_from_safetensors(weights_dir, layer.layer_idx, spec=plan.spec, device=device)
        if layer.layer_kind == "deltanet_moe":
            linear_weights = load_qwen36_linear_attention_weights_from_safetensors(
                weights_dir,
                layer.layer_idx,
                spec=plan.spec,
                device=device,
            )
            if deltanet_impl == "reference" and moe_impl == "reference":
                hidden, conv_state, recurrent_state = qwen36_real_deltanet_moe_layer(
                    hidden,
                    state.deltanet_conv_states[deltanet_idx],
                    state.deltanet_recurrent_states[deltanet_idx],
                    linear_weights,
                    moe_weights,
                    plan.spec,
                )
            elif deltanet_impl == "triton" and moe_impl == "triton":
                (
                    hidden,
                    _deltanet_hidden,
                    _deltanet_update,
                    _moe_update,
                    conv_state,
                    recurrent_state,
                    _logits,
                    _topk_ids,
                    _topk_weights,
                ) = triton_deltanet_moe_layer(
                    hidden,
                    state.deltanet_conv_states[deltanet_idx],
                    state.deltanet_recurrent_states[deltanet_idx],
                    linear_weights.input_norm_weight,
                    linear_weights.in_proj_qkv_weight,
                    linear_weights.in_proj_z_weight,
                    linear_weights.in_proj_a_weight,
                    linear_weights.in_proj_b_weight,
                    linear_weights.conv1d_weight,
                    linear_weights.out_proj_weight,
                    linear_weights.linear_norm_weight,
                    linear_weights.a_log,
                    linear_weights.dt_bias,
                    moe_weights.norm_weight,
                    moe_weights.router_weight,
                    moe_weights.expert_gate_up_weight,
                    moe_weights.expert_down_weight,
                    moe_weights.shared_gate_up_weight,
                    moe_weights.shared_down_weight,
                    moe_weights.shared_expert_gate_weight,
                    qk_heads=plan.spec.deltanet_qk_heads,
                    value_heads=plan.spec.deltanet_value_heads,
                    head_dim=plan.spec.deltanet_head_dim,
                    top_k=plan.spec.num_routed_experts,
                )
            else:
                if deltanet_impl == "reference":
                    update, conv_state, recurrent_state = qwen36_real_deltanet_update(
                        hidden,
                        state.deltanet_conv_states[deltanet_idx],
                        state.deltanet_recurrent_states[deltanet_idx],
                        linear_weights,
                        plan.spec,
                    )
                else:
                    mixed_qkv, z, a_logits, b_logits = triton_deltanet_project(
                        hidden,
                        linear_weights.input_norm_weight,
                        linear_weights.in_proj_qkv_weight,
                        linear_weights.in_proj_z_weight,
                        linear_weights.in_proj_a_weight,
                        linear_weights.in_proj_b_weight,
                    )
                    mixed_qkv, conv_state = triton_deltanet_conv(
                        mixed_qkv,
                        state.deltanet_conv_states[deltanet_idx],
                        linear_weights.conv1d_weight,
                    )
                    update, recurrent_state = triton_deltanet_recurrent_output(
                        mixed_qkv,
                        z,
                        a_logits,
                        b_logits,
                        state.deltanet_recurrent_states[deltanet_idx],
                        linear_weights.out_proj_weight,
                        linear_weights.linear_norm_weight,
                        linear_weights.a_log,
                        linear_weights.dt_bias,
                        qk_heads=plan.spec.deltanet_qk_heads,
                        value_heads=plan.spec.deltanet_value_heads,
                        head_dim=plan.spec.deltanet_head_dim,
                    )
                hidden = hidden.float() + update.float()
                if moe_impl == "reference":
                    hidden = hidden + qwen36_real_moe_update(hidden, moe_weights, plan.spec)
                else:
                    hidden = triton_moe_layer(
                        hidden,
                        moe_weights.norm_weight,
                        moe_weights.router_weight,
                        moe_weights.expert_gate_up_weight,
                        moe_weights.expert_down_weight,
                        moe_weights.shared_gate_up_weight,
                        moe_weights.shared_down_weight,
                        moe_weights.shared_expert_gate_weight,
                        top_k=plan.spec.num_routed_experts,
                    )[0]
            next_conv_states.append(conv_state)
            next_recurrent_states.append(recurrent_state)
            deltanet_idx += 1
        elif layer.layer_kind == "attention_moe":
            attention_weights = load_qwen36_attention_weights_from_safetensors(
                weights_dir,
                layer.layer_idx,
                spec=plan.spec,
                device=device,
            )
            if attention_impl == "reference" and moe_impl == "reference":
                hidden, key_cache, value_cache = qwen36_real_attention_moe_layer(
                    hidden,
                    state.attention_key_cache[attention_idx],
                    state.attention_value_cache[attention_idx],
                    attention_weights,
                    moe_weights,
                    plan.spec,
                    start_position=state.position,
                )
            elif attention_impl == "triton" and moe_impl == "triton":
                (
                    hidden,
                    _attention_hidden,
                    _attention_update,
                    _moe_update,
                    key_cache,
                    value_cache,
                    _logits,
                    _topk_ids,
                    _topk_weights,
                ) = triton_attention_moe_layer(
                    hidden,
                    state.attention_key_cache[attention_idx],
                    state.attention_value_cache[attention_idx],
                    attention_weights.input_norm_weight,
                    attention_weights.q_proj_weight,
                    attention_weights.k_proj_weight,
                    attention_weights.v_proj_weight,
                    attention_weights.o_proj_weight,
                    attention_weights.q_norm_weight,
                    attention_weights.k_norm_weight,
                    moe_weights.norm_weight,
                    moe_weights.router_weight,
                    moe_weights.expert_gate_up_weight,
                    moe_weights.expert_down_weight,
                    moe_weights.shared_gate_up_weight,
                    moe_weights.shared_down_weight,
                    moe_weights.shared_expert_gate_weight,
                    start_position=state.position,
                    attention_heads=plan.spec.attention_heads,
                    kv_heads=plan.spec.attention_kv_heads,
                    head_dim=plan.spec.attention_head_dim,
                    rope_dim=plan.spec.rope_dim,
                    rope_theta=plan.spec.rope_theta,
                    top_k=plan.spec.num_routed_experts,
                )
            else:
                if attention_impl == "reference":
                    update, key_cache, value_cache = qwen36_real_attention_update(
                        hidden,
                        state.attention_key_cache[attention_idx],
                        state.attention_value_cache[attention_idx],
                        attention_weights,
                        plan.spec,
                        start_position=state.position,
                    )
                else:
                    update, key_cache, value_cache = triton_attention_update(
                        hidden,
                        state.attention_key_cache[attention_idx],
                        state.attention_value_cache[attention_idx],
                        attention_weights.input_norm_weight,
                        attention_weights.q_proj_weight,
                        attention_weights.k_proj_weight,
                        attention_weights.v_proj_weight,
                        attention_weights.o_proj_weight,
                        attention_weights.q_norm_weight,
                        attention_weights.k_norm_weight,
                        start_position=state.position,
                        attention_heads=plan.spec.attention_heads,
                        kv_heads=plan.spec.attention_kv_heads,
                        head_dim=plan.spec.attention_head_dim,
                        rope_dim=plan.spec.rope_dim,
                        rope_theta=plan.spec.rope_theta,
                    )
                hidden = hidden.float() + update.float()
                if moe_impl == "reference":
                    hidden = hidden + qwen36_real_moe_update(hidden, moe_weights, plan.spec)
                else:
                    hidden = triton_moe_layer(
                        hidden,
                        moe_weights.norm_weight,
                        moe_weights.router_weight,
                        moe_weights.expert_gate_up_weight,
                        moe_weights.expert_down_weight,
                        moe_weights.shared_gate_up_weight,
                        moe_weights.shared_down_weight,
                        moe_weights.shared_expert_gate_weight,
                        top_k=plan.spec.num_routed_experts,
                    )[0]
            next_key_caches.append(key_cache)
            next_value_caches.append(value_cache)
            attention_idx += 1
        else:
            raise ValueError(f"unknown layer kind: {layer.layer_kind}")
        del moe_weights
        if device == "cuda":
            torch.cuda.empty_cache()
        layer_pos += 1

    if max_layers is not None and max_layers < len(plan.layers):
        next_conv_states.extend(state.deltanet_conv_states[deltanet_idx:])
        next_recurrent_states.extend(state.deltanet_recurrent_states[deltanet_idx:])
        next_key_caches.extend(state.attention_key_cache[attention_idx:])
        next_value_caches.extend(state.attention_value_cache[attention_idx:])

    next_state = type(state)(
        deltanet_conv_states=tuple(next_conv_states),
        deltanet_recurrent_states=tuple(next_recurrent_states),
        attention_key_cache=tuple(next_key_caches),
        attention_value_cache=tuple(next_value_caches),
        position=state.position + len(token_ids),
    )
    logits = qwen36_real_rms_norm(hidden, output_norm_weight) @ lm_head_weight.float().transpose(-1, -2)
    return logits, next_state, pattern_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-token-ids", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--attention-impl", default="reference", choices=["reference", "triton"])
    parser.add_argument("--deltanet-impl", default="reference", choices=["reference", "triton"])
    parser.add_argument("--moe-impl", default="reference", choices=["reference", "triton"])
    args = parser.parse_args()

    torch = _require_torch()
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    dtype = getattr(torch, args.dtype)
    _, config = _download_json(args.repo_id, args.revision, args.cache_dir, "config.json")
    index_path, index = _download_json(args.repo_id, args.revision, args.cache_dir, "model.safetensors.index.json")
    weights_dir = index_path.parent
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json does not contain a valid weight_map")

    spec = Qwen36A3BSpec.from_hf_config(config, name="Qwen3.6-35B-A3B")
    plan = resolve_qwen36_full_weight_plan({key: None for key in weight_map}, spec=spec)
    prompt_ids = _tokenize_prompt(args.repo_id, args.revision, args.cache_dir, args.prompt) if args.prompt else _parse_token_ids(args.prompt_token_ids)
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")

    root_keys = (plan.roots.embedding_weight, plan.roots.output_norm_weight, plan.roots.lm_head_weight)
    _download_keys(args.repo_id, args.revision, args.cache_dir, weight_map, root_keys)
    roots = _load_selected_tensors(weights_dir, weight_map, root_keys, args.device)
    embedding = roots[plan.roots.embedding_weight].to(dtype=dtype)
    output_norm = roots[plan.roots.output_norm_weight].to(dtype=dtype)
    lm_head = roots[plan.roots.lm_head_weight].to(dtype=dtype)
    state = initial_qwen36_real_decode_state(spec, max_positions=args.max_positions, device=args.device, dtype=dtype)

    start = time.perf_counter()
    logits, state, pattern_chunks = _run_chunk(
        torch.tensor(prompt_ids, device=args.device, dtype=torch.long),
        state,
        embedding_weight=embedding,
        output_norm_weight=output_norm,
        lm_head_weight=lm_head,
        plan=plan,
        weights_dir=weights_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        weight_map=weight_map,
        device=args.device,
        max_layers=args.max_layers,
        attention_impl=args.attention_impl,
        deltanet_impl=args.deltanet_impl,
        moe_impl=args.moe_impl,
    )
    total_pattern_chunks = pattern_chunks
    generated = []
    for _ in range(args.max_new_tokens):
        next_token = int(torch.argmax(logits[-1]).item())
        generated.append(next_token)
        logits, state, pattern_chunks = _run_chunk(
            torch.tensor([next_token], device=args.device, dtype=torch.long),
            state,
            embedding_weight=embedding,
            output_norm_weight=output_norm,
            lm_head_weight=lm_head,
            plan=plan,
            weights_dir=weights_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            weight_map=weight_map,
            device=args.device,
            max_layers=args.max_layers,
            attention_impl=args.attention_impl,
            deltanet_impl=args.deltanet_impl,
            moe_impl=args.moe_impl,
        )
        total_pattern_chunks += pattern_chunks
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    layers_executed = len(plan.layers) if args.max_layers is None else args.max_layers
    positions_processed = state.position
    layer_positions = layers_executed * positions_processed

    print(f"model: {spec.name}")
    print(f"repo_id: {args.repo_id}")
    print(f"revision: {args.revision}")
    print(f"device: {args.device}")
    print(f"dtype: {dtype}")
    print(f"layers_total: {len(plan.layers)}")
    print(f"layers_executed: {layers_executed}")
    print(f"prompt_tokens: {len(prompt_ids)}")
    print(f"generated_tokens: {len(generated)}")
    print(f"generated_token_ids: {','.join(str(token) for token in generated)}")
    print(f"final_position: {state.position}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"elapsed_ms_per_position: {elapsed * 1000 / max(positions_processed, 1):.4f}")
    print(f"elapsed_ms_per_layer_position: {elapsed * 1000 / max(layer_positions, 1):.4f}")
    print(f"layer_positions_per_second: {layer_positions / elapsed:.4f}")
    print(f"attention_impl: {args.attention_impl}")
    print(f"deltanet_impl: {args.deltanet_impl}")
    print(f"moe_impl: {args.moe_impl}")
    print("fastkernels_full_reference_serving_ready: True")
    print(f"fastkernels_full_triton_attention_serving_ready: {args.attention_impl == 'triton'}")
    print(
        "fastkernels_full_triton_attention_moe_layer_serving_ready: "
        f"{args.attention_impl == 'triton' and args.moe_impl == 'triton'}"
    )
    print(f"fastkernels_full_triton_deltanet_projection_serving_ready: {args.deltanet_impl == 'triton'}")
    print(f"fastkernels_full_triton_deltanet_conv_serving_ready: {args.deltanet_impl == 'triton'}")
    print(f"fastkernels_full_triton_deltanet_recurrent_output_serving_ready: {args.deltanet_impl == 'triton'}")
    print(
        "fastkernels_full_triton_deltanet_moe_layer_serving_ready: "
        f"{args.deltanet_impl == 'triton' and args.moe_impl == 'triton'}"
    )
    print(f"fastkernels_full_triton_moe_serving_ready: {args.moe_impl == 'triton'}")
    print(f"fastkernels_full_triton_four_layer_pattern_chunks: {total_pattern_chunks}")
    print(
        f"fastkernels_full_triton_four_layer_pattern_chunks_per_position: "
        f"{total_pattern_chunks / max(positions_processed, 1):.4f}"
    )
    print(f"fastkernels_full_triton_four_layer_pattern_serving_ready: {total_pattern_chunks > 0}")
    print(
        "fastkernels_full_triton_attention_deltanet_moe_staging_ready: "
        f"{args.attention_impl == 'triton' and args.deltanet_impl == 'triton' and args.moe_impl == 'triton'}"
    )
    print("fastkernels_full_triton_serving_ready: False")
    print(
        "serving_scope: streaming real Qwen3.6 weights with selectable Triton "
        "Attention -> MoE and DeltaNet -> MoE layer staging"
    )


if __name__ == "__main__":
    main()

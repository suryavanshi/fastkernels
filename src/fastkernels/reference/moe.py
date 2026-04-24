"""PyTorch reference MoE operations."""

from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("PyTorch is required for fastkernels reference MoE ops") from exc
    return torch


def reference_fused_swiglu(gate_up: Any) -> Any:
    """Return `silu(gate) * up` for a `[N, 2 * intermediate]` tensor."""

    torch = _require_torch()
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def reference_expert_histogram(topk_ids: Any, num_experts: int) -> Any:
    """Count routed tokens per expert for a `[tokens, top_k]` expert-id tensor."""

    torch = _require_torch()
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must have shape [tokens, top_k]")
    return torch.bincount(topk_ids.reshape(-1).to(torch.int64), minlength=num_experts).to(torch.int32)


def reference_routed_moe(
    hidden_states: Any,
    router_logits: Any,
    expert_gate_up_weight: Any,
    expert_down_weight: Any,
    *,
    top_k: int,
    normalize_topk: bool = True,
) -> Any:
    """Reference routed MoE forward path.

    Args:
        hidden_states: `[tokens, hidden]`.
        router_logits: `[tokens, experts]`.
        expert_gate_up_weight: `[experts, 2 * intermediate, hidden]`.
        expert_down_weight: `[experts, hidden, intermediate]`.
        top_k: Number of routed experts per token.
        normalize_topk: Re-normalize selected router weights to sum to one.

    Returns:
        Tensor with shape `[tokens, hidden]`.
    """

    torch = _require_torch()
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden]")
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    if expert_gate_up_weight.ndim != 3:
        raise ValueError("expert_gate_up_weight must have shape [experts, 2 * intermediate, hidden]")
    if expert_down_weight.ndim != 3:
        raise ValueError("expert_down_weight must have shape [experts, hidden, intermediate]")

    tokens, hidden = hidden_states.shape
    num_experts = router_logits.shape[1]
    if expert_gate_up_weight.shape[0] != num_experts:
        raise ValueError("router_logits and expert_gate_up_weight disagree on expert count")
    if expert_down_weight.shape[0] != num_experts:
        raise ValueError("router_logits and expert_down_weight disagree on expert count")
    if expert_gate_up_weight.shape[2] != hidden:
        raise ValueError("expert_gate_up_weight hidden dimension mismatch")
    if expert_down_weight.shape[1] != hidden:
        raise ValueError("expert_down_weight hidden dimension mismatch")

    route_probs = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(route_probs, k=top_k, dim=-1)
    if normalize_topk:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    output = torch.zeros_like(hidden_states)
    for token_idx in range(tokens):
        token = hidden_states[token_idx]
        for route_idx in range(top_k):
            expert_id = int(topk_ids[token_idx, route_idx])
            weight = topk_weights[token_idx, route_idx]
            gate_up = torch.matmul(expert_gate_up_weight[expert_id], token)
            activation = reference_fused_swiglu(gate_up)
            expert_out = torch.matmul(expert_down_weight[expert_id], activation)
            output[token_idx] = output[token_idx] + weight * expert_out

    return output

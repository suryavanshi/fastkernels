"""Synthetic Qwen3.6 decode reference benchmark."""

from __future__ import annotations

import argparse
import time

from fastkernels.models import synthetic_qwen36_spec
from fastkernels.reference import (
    initial_qwen36_decode_state,
    make_synthetic_qwen36_decode_weights,
    reference_qwen36_decode_step,
)


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
    args = parser.parse_args()

    torch = _require_torch()
    dtype = getattr(torch, args.dtype)
    spec = synthetic_qwen36_spec()
    if args.steps > spec.context_length:
        raise ValueError(f"steps must be <= synthetic context_length ({spec.context_length})")

    weights = make_synthetic_qwen36_decode_weights(spec, device=args.device, dtype=dtype, seed=args.seed)
    token_ids = [(args.seed + idx) % spec.vocab_size for idx in range(args.steps)]

    def run_decode():
        state = initial_qwen36_decode_state(spec, max_positions=spec.context_length, device=args.device, dtype=dtype)
        logits = None
        for token_id in token_ids:
            logits, state = reference_qwen36_decode_step(token_id, state, weights, spec)
        return logits, state

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
    print(f"steps: {args.steps}")
    print(f"final_position: {state.position}")
    print(f"logits_shape: {tuple(logits.shape)}")
    print(f"decode_ms_per_step: {elapsed * 1000 / args.steps:.4f}")


if __name__ == "__main__":
    main()

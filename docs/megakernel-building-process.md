# Megakernel Building Process

This document explains how the Qwen megakernel work is being built and what
techniques are used along the way. It is intentionally process-focused: the
repository does not yet contain a full all-layer Qwen3.6-35B-A3B megakernel for
end-to-end token generation.

The current target is the Qwen3.6-35B-A3B decode path: 40 layers, hidden size
2048, 256 routed experts, top-8 routed experts per token, one shared expert,
and a hybrid layer pattern that includes DeltaNet, full attention, and MoE.

## What "Megakernel" Means Here

A decode megakernel is the path toward doing a whole decode-layer update with as
few GPU launches as practical. The goal is to keep intermediate values in GPU
registers, shared memory, cache-friendly layouts, or persistent state instead of
round-tripping them through many small framework calls.

For this repository, the term means a staged path from small fused kernels to
larger layer-boundary kernels. It does not mean a single monolithic kernel has
already replaced all Qwen3.6 inference.

## Build From Correctness Outward

The process is deliberately conservative:

1. Define shape contracts.
   Model specs in `src/fastkernels/models/` capture hidden sizes, expert counts,
   top-k routing, layer counts, and layer kinds before kernel code depends on
   those values.

2. Keep PyTorch oracles.
   Reference implementations in `src/fastkernels/reference/` remain the source
   of truth for correctness checks. Custom kernels compare against PyTorch
   outputs with explicit tolerances.

3. Start with deterministic synthetic tests.
   Tiny and real-shaped random tensors make it possible to test kernel dataflow,
   routing behavior, and tolerances before involving large model weights.

4. Move to real shapes.
   The Qwen3.6 path then uses hidden-2048, 256 experts, top-8 routing, and
   intermediate-512 expert shapes so memory access and launch behavior resemble
   the real model.

5. Add real weight loading only at narrow boundaries.
   Safetensors index parsing downloads only the shards needed for selected
   layers. This keeps Modal and Lambda smoke tests practical while checking
   actual Hugging Face tensor names and layouts.

   The full-model planner now uses the same resolver family at whole-model
   scope, validating every root tensor, all 40 layers, and all required shards
   without downloading the entire 72GB BF16 checkpoint by default.

6. Compare at matched operator boundaries.
   vLLM comparisons are currently routed-MoE operator comparisons, not
   end-to-end serving tokens/sec comparisons. They can now use either
   real-shaped random tensors or selected downloaded real MoE layer weights.
   Full serving comparisons should wait until the same model, dtype, prompt,
   generation settings, batch size, quantization mode, and GPU setup are used.

7. Compose kernels into layer-boundary wrappers.
   Once a small kernel is correct, wrappers connect router output, routed expert
   execution, shared expert execution, residual updates, attention projections,
   and eventually state updates.

8. Reduce launches only when measurements support it.
   Fusion is useful when it removes real overhead without making parity or
   debugging fragile. Each fusion step should keep a reference path and report
   both accuracy and timing.

## Techniques Used

### Triton First

Triton is used for the current implementation because it is fast to iterate on
and easy to keep close to the PyTorch oracle. The intended path is to use Triton
to settle shape contracts, memory layouts, and fusion boundaries, then consider
CuTe, CUTLASS, or custom CUDA extensions for the hottest stable paths.

### Fused RMSNorm And Projections

The real full-attention lane includes Triton staging that starts with fused
RMSNorm plus Q/K/V projection for actual Qwen3.6 layer weights, then applies
Q/K norm, RoPE/cache update, causal attention accumulation, and output
projection against the same PyTorch oracle.

The same pattern applies to MoE: normalize or prepare a token row, project to
router logits, select experts, and keep the output contract compact enough that
the next kernel can consume it directly.

### Top-K Routing

Qwen3.6 uses 256 routed experts and top-8 routing. The router path computes
expert scores, selects the top routed experts, and produces the routing weights
used by the expert accumulation path.

Routing is tested at multiple scales:

- Tiny synthetic shapes for parity.
- Real-shaped random tensors for launch and memory behavior.
- Real Qwen3.6 MoE weights for selected layer smokes.

### Routed And Shared Expert Split

The model has routed experts and a shared expert. The current real-shape MoE
path keeps those concepts explicit:

- Routed experts process each token through the selected top-k experts.
- The shared expert contributes a dense expert path for every token.
- The layer wrapper checks the residual boundary as `hidden + MoE`.

This split makes the implementation easier to validate before deeper fusion.

### Fused Expert Math

The expert MLP path follows the gate/up/down structure used by Qwen-style MoE
blocks. The kernel work stages the expensive pieces so they can later be fused
more aggressively:

- Gate and up projections.
- SwiGLU activation.
- Down projection.
- Routing-weight accumulation.

The current design favors clear contracts and correctness over hiding every
intermediate immediately.

### Batched Decode Rows

Even when decode generates one token per sequence, benchmarks often run several
token rows at once. Batched rows help amortize launch overhead and expose memory
layout problems earlier than single-row-only tests.

The Modal real-weight smoke currently exercises multi-token rows for selected
Qwen3.6 layers instead of requiring a full model load.

### Real Safetensors Layout Support

The real-weight lane resolves Hugging Face safetensors index files and downloads
only the shard files required for the requested layer-local tensors. This is
important because the full model is large enough that naive all-weight loading
is not a useful first test on a single A100.

The loader and packer code checks actual tensor names for:

- MoE router, routed experts, shared expert, and shared expert gate.
- DeltaNet projection tensors.
- Full-attention Q/K/V/O and normalization tensors.

### Launch Count Tracking

The near-term optimization target is fewer launches across layer boundaries.
Separate kernels are acceptable while correctness is still moving, but each
measured wrapper should make launch boundaries visible so the next fusion target
is obvious.

Examples of current or near-term launch reductions include:

- Router plus routed/shared expert wrappers.
- RMSNorm plus Q/K/V projection.
- Staged Q/K norm, RoPE, KV-cache update, attention accumulation, and output
  projection, followed by measured fusion of those attention stages.
- Future DeltaNet projection and state-update fusion.

### Modal And Lambda Verification

Local machines often lack the CUDA stack needed for these tests, so GPU
verification runs on Modal and Lambda. A useful run reports:

- Unit test result.
- Device and dtype.
- Max absolute differences against PyTorch.
- Timing in milliseconds and milliseconds per token row.
- Whether weights are synthetic or real.
- Whether comparison is operator-level or full-model.

The latest documented Modal lane validates synthetic decode coverage, real MoE
weight smokes, residual layer-boundary parity, non-MoE block loading, real
full-attention decode staging, and a real Attention -> MoE residual layer
boundary. A separate Modal lane resolves the full 40-layer real-weight plan and
can run streaming full-model PyTorch reference serving on H100:8. The full
decode runner now has real-weight Triton Attention and Triton MoE switches; the
Attention path uses the real per-head Q/gate projection layout and
model-configured RoPE theta. The
full-model vLLM serving probe now has a standalone H100:8 path using the
prebuilt `vllm/vllm-openai:v0.20.0-x86_64` image. The older pip-install probe
is still available; vLLM 0.10.2 rejects this architecture before serving starts,
and pip-installing `vllm==0.20.0` into the slim fastkernels image fails because
its source build requires `CUDA_HOME`.

## Performance Reporting Rules

Tokens/sec is only meaningful when the comparison is end-to-end and matched.
Until then, this project reports narrower metrics:

- `max_abs_diff` against PyTorch reference output.
- Kernel or wrapper latency in milliseconds.
- Milliseconds per token row for decode-shaped tests.
- vLLM comparison only at the matched routed-MoE operator boundary, including
  selected real MoE layer weights when requested.

A full vLLM comparison should use the same model, weights, dtype, quantization,
prompt lengths, decode length, sampling settings, batch size, GPU type, tensor
parallelism, and cache policy.

## Current Implementation State

Implemented and tested pieces include:

- Qwen3.6 shape specs.
- Synthetic DeltaNet, attention, MoE, and layer-boundary decode references.
- Real-shape Qwen3.6 MoE router kernels.
- Real-shape routed and shared expert kernels.
- Batched full real-shape MoE wrappers.
- Indexed safetensors loading and selected real-weight MoE packing.
- Real MoE residual layer-boundary smoke tests.
- Real non-MoE block tensor loading for DeltaNet and full-attention layers.
- Real full-attention staged decode parity on actual weights.
- Real Attention -> MoE residual layer-boundary parity on actual weights.
- Full 40-layer real-weight key and shard-plan resolution.
- Streaming PyTorch reference full-model generation over real Qwen3.6 weights.
- Full-model vLLM serving benchmark entrypoint for baseline throughput.

Not implemented yet:

- Full Triton/megakernel all-layer Qwen3.6-35B-A3B token generation.
- End-to-end fastkernels tokens/sec comparison.
- Multi-GPU tensor-parallel serving harness.

## Roadmap

1. Settle the remaining real full-attention gated projection contract for the
   wider Q projection rows, then reduce launches in the verified attention
   staging path.

2. Wire real DeltaNet projections and state updates behind the same
   correctness-first harness.

3. Compose DeltaNet output into the real-weight MoE residual layer boundary;
   the real Attention -> MoE boundary is now covered by the block smoke.

4. Reduce launch count in measured steps, keeping the PyTorch reference and
   staged wrappers available.

5. Introduce CuTe, CUTLASS, or custom CUDA only after the Triton contracts and
   fusion boundaries stop shifting.

6. Run full-model vLLM baseline tests on H100:8, then compare fastkernels
   end-to-end serving metrics against that baseline once real DeltaNet and the
   40-layer runtime exist.

## Guardrails

- Do not remove PyTorch reference paths.
- Do not claim full Qwen3.6-35B-A3B inference until end-to-end real-weight
  token generation works.
- Do not treat single-operator timing as full serving tokens/sec.
- Do not load all full-precision weights on a single A100 and interpret an OOM
  as a kernel correctness failure.
- Keep Modal, Lambda, and README docs explicit about whether a run is synthetic,
  real-shaped random, selected real-weight, or full-model.

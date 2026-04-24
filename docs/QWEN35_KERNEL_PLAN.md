# Qwen3.5 MoE Kernel Plan

This document is the working execution plan for building `fastkernels` into a
kernel lab for Qwen3.5-35B-A3B and adjacent open MoE LLMs.

## Target Model

Initial target: Qwen3.5-35B-A3B.

Important kernel-facing traits:

- Total parameters: 35B.
- Active parameters: about 3B.
- Layers: 40.
- Hidden size: 2048.
- Routed experts: 256.
- Active routed experts per token: 8.
- Shared experts per token: 1.
- Expert intermediate size: 512.
- Repeating block pattern: three DeltaNet + MoE layers, then one Attention +
  MoE layer, repeated ten times.

The first optimization target is the MoE block because it combines routing,
layout transforms, many small expert matrix multiplies, activation fusion, and
weighted reduction.

## Backend Ladder

1. PyTorch reference kernels for correctness and model-shape validation.
2. Triton prototypes for fast iteration.
3. NVIDIA CuTE/CUTLASS kernels for peak performance.
4. AMD HIP/Composable Kernel kernels for ROCm parity.
5. Mojo custom ops for portable experiments and shared abstractions.
6. Persistent layer kernels.
7. Full or partial decode megakernels.

## Initial Repo Milestones

### M0: Harness And Shape Truth

- Add a Qwen3.5 model spec module.
- Add shape reports for MoE, attention, and DeltaNet paths.
- Add PyTorch reference MoE implementation.
- Add a microbenchmark that can run with or without Triton.
- Store benchmark outputs with hardware, dtype, shape, and backend metadata.

### M1: First MoE GPU Building Blocks

- Fused SwiGLU activation kernel.
- Router top-k helper.
- Token counting and expert histogram.
- Expert-major packing layout.
- Weighted reduce/unpack.

### M2: Fused MoE Prototype

- Triton decode path for small token counts.
- Triton prefill path for grouped expert GEMMs.
- CuTE grouped-GEMM implementation for NVIDIA.
- CK grouped-GEMM implementation for AMD.
- Autotune database keyed by GPU, dtype, and shape.

### M3: Layer Fusion

- Fuse RMSNorm + router where profitable.
- Fuse expert activation + down projection epilogue.
- Fuse MoE weighted reduction with residual writeback.
- Add one full DeltaNet + MoE layer kernel prototype.
- Add one full Attention + MoE layer kernel prototype.

### M4: Persistent Decode And Megakernel

- Persistent decode kernel for one repeated Qwen3.5 group:
  `3 x (DeltaNet -> MoE) + 1 x (Attention -> MoE)`.
- Extend to the full 40-layer schedule.
- Add multi-GPU expert-parallel dispatch.

## Correctness Rules

Every custom kernel must have:

- A PyTorch reference path.
- Shape assertions.
- Dtype-specific tolerances.
- A deterministic small-shape test.
- A realistic Qwen3.5-shape benchmark.

## Performance Scoreboard

Track:

- Tokens/sec for decode and prefill.
- Time/token.
- Kernel launches/token.
- HBM bandwidth.
- Tensor core utilization.
- Router and packing overhead.
- Expert imbalance.
- Power and tokens/joule.

Compare against vLLM, SGLang, KTransformers, FlashInfer where applicable, and
plain PyTorch references.

## First Kernel Surface

The first implemented GPU kernel is a fused SwiGLU activation kernel:

```text
gate_up: [N, 2 * intermediate]
out:     [N, intermediate]
out = silu(gate_up[:, :intermediate]) * gate_up[:, intermediate:]
```

This is intentionally small. It gives the project a correctness harness, Triton
integration point, and a building block that later moves into fused expert MLP
and fused MoE kernels.

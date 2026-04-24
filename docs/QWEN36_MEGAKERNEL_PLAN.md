# Qwen3.6-35B-A3B Megakernel Plan

This document tracks the path from the current `fastkernels` MoE building
blocks toward a Qwen3.6-35B-A3B inference megakernel.

## Current Status

No full Qwen3.6-35B-A3B megakernel exists in this repository yet.

The current CUDA-tested path covers:

- Qwen3.6/Qwen3.5-style MoE shape metadata.
- A synthetic PyTorch decode dataflow/reference for the Qwen3.6 layer pattern.
- PyTorch reference MoE operations.
- Triton `fused_swiglu`.
- Triton routed-expert histogram.
- Synthetic Lambda A100 correctness and timing runs for the kernels above.

This is not equivalent to the all-layer Qwen 3.5-0.8B DeltaNet megakernel
described in the Awesome Agents article. That work fuses the full recurrent /
attention / MLP stack for a much smaller 0.8B model into one CUDA launch. The
35B-A3B target has 40 layers, 256 experts, and substantially more weight and
routing complexity.

## Model Target

Hugging Face lists Qwen3.6-35B-A3B as:

- 35B total parameters, roughly 3B active.
- Hidden size 2048.
- 40 layers.
- Layout: `10 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Gated Attention -> MoE))`.
- DeltaNet: 32 V heads, 16 QK heads, head dimension 128.
- Gated attention: 16 Q heads, 2 KV heads, head dimension 256, RoPE dimension 64.
- MoE: 256 experts, 8 routed experts plus 1 shared expert, intermediate size 512.
- Native context length 262,144 tokens.

The Hugging Face serving examples use tensor parallel size 8 for SGLang and
vLLM. A single Lambda A100-40GB instance is useful for developing and validating
synthetic kernels, but it is not enough to run the full BF16 model end to end.

## Triton vs CuTe/CUTLASS

Use both, but for different phases.

Triton is the better first implementation language for this repository:

- Faster iteration from Python.
- Easy integration with the existing PyTorch reference tests.
- Good enough for routing, activation, histogram, scatter/gather, and prototype
  fused expert kernels.
- The existing repo already has Triton optional dependencies and tests.

CuTe/CUTLASS is the better final path for the fastest production megakernel:

- Better control over warp-specialized pipelines, shared-memory layout, register
  pressure, cooperative groups, and persistent scheduling.
- Better fit for one-launch or few-launch decode kernels where inter-layer
  synchronization, register-resident recurrence state, and carefully tiled GEMMs
  dominate performance.
- More work to integrate and maintain, especially once Python reference parity,
  model loading, quantization, and build tooling are included.

Recommended path:

1. Keep Triton for correctness-first kernels and performance discovery.
2. Build isolated CuTe/CUTLASS kernels once the dataflow and tensor layouts are
   stable.
3. Use the Lambda A100 instance for synthetic correctness and microbenchmarks.
4. Use an 8-GPU Lambda instance, when available, for full-model end-to-end
   validation.

## Milestones

### 1. Decode Dataflow Spec

Define a decode-only tensor contract independent of Hugging Face internals:

- Input token embedding.
- Per-layer DeltaNet recurrent state.
- Per-layer attention KV cache.
- Router logits and top-k expert ids.
- Expert weights in the expected GPU memory layout.
- Output logits or final hidden state.

Deliverable: a deterministic PyTorch reference decode step for one token and a
small synthetic model shape.

### 2. MoE Path

Extend the current kernels into a fused routed expert path:

- Router top-k selection.
- Expert histogram.
- Prefix offsets / token packing.
- Fused gate-up projection, SwiGLU, and down projection.
- Shared expert path.
- Scatter/reduce back to token order.

Deliverable: Triton MoE block matching the PyTorch reference for synthetic
Qwen3.6 shapes.

### 3. DeltaNet Path

Implement a synthetic Gated DeltaNet decode kernel:

- Q/K/V projections.
- Gating.
- Recurrent state update.
- Output projection.
- FP32 accumulation where needed.

Deliverable: Triton prototype first, followed by a CuTe/CUTLASS version if
register pressure and synchronization require lower-level control.

### 4. Attention Path

Implement Gated Attention decode:

- Q/K/V projection.
- RoPE.
- KV-cache append.
- Causal decode attention.
- Output projection.

Deliverable: parity against PyTorch for small synthetic shapes.

### 5. Layer Fusion

Fuse one full layer at a time:

- DeltaNet -> MoE.
- Attention -> MoE.
- Then 4-layer pattern.
- Then 40-layer persistent decode path.

Deliverable: staged kernels with correctness tests at each fusion boundary.

### 6. Full Model Validation

Full Qwen3.6-35B-A3B validation requires:

- Model weight access and layout conversion.
- Quantization or tensor parallelism.
- 8-GPU run for BF16 as recommended by serving examples, or a single larger /
  quantized setup if memory allows.
- Token-level numerical parity tests against Transformers, vLLM, or SGLang.
- Decode throughput and power measurements.

## Lambda Test Lane

For the current single-A100 instance, run:

```sh
cd ~/fastkernels
source .venv/bin/activate
pytest -q
python bench/qwen36_decode_reference.py --device cuda --dtype float32 --steps 8 --warmup 2
python bench/qwen35_profile.py --tokens 1 16 128
python bench/moe_microbench.py --device cuda --dtype bfloat16 --rows 2048 --tokens 4 --warmup 10 --iters 100 --skip-routed-moe
python bench/moe_microbench.py --device cuda --dtype bfloat16 --rows 8192 --tokens 4 --warmup 10 --iters 100 --skip-routed-moe
```

See `docs/LAMBDA.md` for launch, SSH, install, sync, benchmark, and terminate
instructions.

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
- Triton real-shape Qwen3.6 MoE router decode kernel for hidden-2048,
  256 experts, and top-8 routing, including batched decode-token routing.
- Triton real-shape Qwen3.6 single-expert MLP kernel for hidden-2048 and
  intermediate-512.
- Triton real-shape Qwen3.6 routed-expert-only path for matched vLLM fused-MoE
  operator comparison, including a batched selected-real-weight comparison mode.
- Triton single-token and batched real-shape Qwen3.6 top-8 routed plus shared
  expert accumulation.
- Triton full real-shape Qwen3.6 MoE decode wrapper connecting RMSNorm/router
  output to top-8 routed plus shared expert execution.
- Qwen3.6 MoE safetensors key resolver/packer for HF-style split expert
  weights in single-file or indexed sharded layouts.
- Real Qwen3.6 layer-0 MoE shard smoke that downloads only selected HF shard
  files and runs one-token parity for the current MoE wrapper contract.
- Triton fused synthetic Qwen3.6 top-2 routed + shared MoE decode block.
- Triton generalized tiny-shape synthetic MoE top-k path verified through
  top-k 8 and 256 routed experts.
- Triton staged synthetic Qwen3.6 DeltaNet decode block.
- Triton staged real-weight Qwen3.6 DeltaNet boundary for a selected DeltaNet
  layer, covering RMSNorm+QKV/Z/A/B projection, causal depthwise convolution,
  conv-state update, recurrent-state update, gated RMSNorm, and output
  projection.
- Triton staged synthetic Qwen3.6 attention decode block.
- Triton staged real-weight Qwen3.6 full-attention decode boundary for a
  selected full-attention layer, covering RMSNorm+Q/K/V projection, Q/K norm,
  RoPE/cache update, causal attention accumulation, and output projection.
- Triton staged real-weight Attention -> MoE residual layer boundary for a
  selected full-attention+MoE layer.
- Full 40-layer real-weight key-plan resolver covering root tensors, all
  DeltaNet+MoE layers, all Attention+MoE layers, and all safetensors shards.
- Modal H100:8 vLLM full-serving benchmark lane for the current end-to-end
  baseline.
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

The public `config.json` stores these language-model fields under
`text_config`, including `layer_types`, `linear_num_key_heads`,
`linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`,
`head_dim`, `partial_rotary_factor`, and `rope_parameters`.

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

## Completion Gates

Do not call the 35B-A3B megakernel complete until all of these are true:

- Full-shape 40-layer decode runs against real Qwen3.5/Qwen3.6 weights.
- DeltaNet, attention, and MoE paths all have token-level parity against a
  trusted Transformers, vLLM, or SGLang reference.
- The implementation supports the target deployment shape: tensor parallelism
  or quantization for memory, plus a documented weight layout/converter.
- Decode and prefill benchmarks report tokens/sec, time/token, launches/token,
  memory bandwidth, and power when available.
- vLLM comparison uses the same model, dtype/quantization, prompt lengths,
  generation length, batch/concurrency, GPU type, and sampling settings.

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

Status: the first fused synthetic top-2 routed + shared MoE decode block is
implemented in Triton and matches the PyTorch reference for the tiny synthetic
Qwen3.6 spec. The same kernel now has a generalized static top-k path verified
for top-k 1, 2, 4, and 8 in tiny-shape tests, plus a Modal A100
256-expert/top-8 synthetic microbench. This is routing-scale coverage, not yet
full hidden-2048/intermediate-512 Qwen3.6 MoE coverage. The current
implementation dispatches top-2 to a dedicated fast kernel and keeps the
generalized path for broader top-k coverage.

Real-shape status: a Triton router decode kernel now covers the real
hidden-2048, 256-expert, top-8 Qwen3.6 routing shape. It computes RMSNorm,
router logits, softmax, top-k expert ids, and renormalized top-k weights for one
decode token. Modal A100 parity against a PyTorch reference passes, with
`router_ms: 0.0655` for bfloat16 in the latest run.

A real-shape single-expert MLP kernel now covers hidden-2048 and
intermediate-512 gate/up projection, SwiGLU, and down projection for one decode
token. Modal A100 parity against a PyTorch reference passes, with
`expert_mlp_ms: 0.0616` for bfloat16 in the latest run.

A real-shape routed-expert-only path now provides the matched operator boundary
for comparison with vLLM's `fused_moe`, including batched token rows. Modal
A100 parity against a PyTorch reference passes, and the random-tensor
comparison against vLLM 0.10.2 reported `fastkernels_routed_moe_ms: 0.0859`,
`vllm_routed_moe_ms: 0.4841`, and matching max-abs diffs of `0.000000` for
bfloat16. The selected real-weight comparison downloads the actual layer-0 MoE
shards and reports `fastkernels_routed_moe_ms_per_token: 0.0483`,
`vllm_routed_moe_ms_per_token: 0.2306`, matching max-abs diffs of `0.000000`,
and `fastkernels_to_vllm_ms_ratio: 0.2094` for 2 bfloat16 token rows. vLLM
warned that it used a default MoE config for the exact `E=256,N=512,A100`
shape, so treat these as operator-level checkpoints rather than final serving
throughput claims.

A batched real-shape routed/shared expert path now accumulates top-8 routed
expert outputs plus the shared expert for one decode token. Modal A100 parity
against a PyTorch reference passes, with `routed_shared_moe_ms: 0.0957` for
bfloat16. This path uses two Triton launches for expert activations and the
weighted down-projection reduction, avoiding the previous Python-level expert
dispatch.

The routed/shared expert path also supports batched hidden states and per-token
routing tensors. Modal A100 parity against a PyTorch reference passes for
4 decode tokens, with `batched_routed_shared_moe_ms: 0.2471` and
`batched_routed_shared_moe_ms_per_token: 0.0618` for bfloat16.

A full real-shape MoE wrapper now connects the real router output to the
batched routed/shared expert path for one hidden-2048 decode token. Modal A100
parity against a PyTorch reference passes, with `real_moe_ms: 0.1815`,
`logits_max_abs_diff: 0.000023`, matching top-k ids, and
`topk_weights_max_abs_diff: 0.000001` for bfloat16. This is still random-tensor
real-shape validation, not real Qwen3.6 weight loading or full-model inference.
The next real-shape performance task is to reduce launches inside this wrapper
and improve the reduction tiling.

A batched full real-shape MoE wrapper now connects batched router output to the
batched routed/shared expert path for 4 hidden-2048 decode tokens. Modal A100
parity against a PyTorch reference passes, with `batched_real_moe_ms: 0.2666`,
`batched_real_moe_ms_per_token: 0.0667`, `logits_max_abs_diff: 0.000031`,
matching top-k ids, and `topk_weights_max_abs_diff: 0.000002` for bfloat16.
This is still random-tensor real-shape validation, not real Qwen3.6 weight
loading or full-model inference.

A safetensors loader smoke test now resolves HF-style keys for one generated
real-shaped Qwen3.6 MoE layer from an indexed two-shard directory, packs split
`gate_proj` and `up_proj` weights into the kernel-facing gate/up layout, and
feeds the batched full MoE wrapper. Modal A100 parity against a PyTorch
reference passes with `output_max_abs_diff: 0.000000`,
`logits_max_abs_diff: 0.000031`, matching top-k ids,
`safetensor_indexed: True`, and `safetensor_batched_real_moe_ms: 0.2159`
for 2 bfloat16 tokens. This proves the loader/packer contract, not downloaded
model-weight parity.

A real-weight smoke test now downloads the actual Qwen/Qwen3.6-35B-A3B
`model.safetensors.index.json` and only the two shard files needed for layer-0
MoE (`model-00001-of-00026.safetensors` and
`model-00002-of-00026.safetensors`). It resolves the real grouped
`model.language_model.layers.0.mlp.experts.gate_up_proj` and
`model.language_model.layers.0.mlp.experts.down_proj` tensors, packs them into
the kernel layout, and runs one-token Modal A100 parity with
`output_max_abs_diff: 0.000000`, `logits_max_abs_diff: 0.000000`, matching
top-k ids, and `real_weight_batched_moe_ms: 0.3328`.

The `shared_expert_gate.weight` path is now fused into the Triton
routed/shared MoE wrapper as a separate shared-gate scalar kernel plus gated
shared down-projection. Modal A100 parity against the real layer-0 shard smoke
passes with `shared_expert_gate_applied: True`,
`output_max_abs_diff: 0.000000`, `logits_max_abs_diff: 0.000000`, matching
top-k ids, and `real_weight_batched_moe_ms: 0.3290` for one bfloat16 token.
The real-weight smoke is now configurable over token count and layers and can
also run a residual `hidden + MoE` layer-boundary harness. The latest Modal A100
run covers layers 0 and 1 with 2 bfloat16 token rows, exact MoE update/logit
parity, exact residual layer-output parity, and
`real_weight_batched_moe_layer_ms_per_token: 0.2018` for layer 0 plus `0.1575`
for layer 1. The real non-MoE block loader now resolves and loads layer-0
`linear_attn.*` weights and layer-3 `self_attn.*` weights from the actual HF
shards. The layer-0 DeltaNet path now runs real Triton projection+conv staging
against PyTorch with `linear_triton_conv_max_abs_diff: 0.000000`,
`linear_triton_conv_state_max_abs_diff: 0.000061`, and
`linear_triton_update_max_abs_diff: 0.000000`,
`linear_triton_recurrent_state_max_abs_diff: 0.000002`,
`linear_moe_layer_hidden_max_abs_diff: 0.000000`, matching top-k ids, and
`linear_moe_layer_ms_per_token: 0.4462` for 2 token rows. The
layer-3 full-attention path now runs real Triton attention decode staging
against PyTorch with exact reported parity for Q/K/V projection, attention
update, key cache, and value cache, and
`attention_triton_decode_ms_per_token: 0.3144` for 2 token rows. The same
smoke composes the real attention update into the real layer-3 MoE residual
boundary with exact reported parity for attention-hidden, MoE update,
layer-hidden, caches, router logits, top-k ids, and top-k weights, with
`attention_moe_layer_ms_per_token: 0.4816` for 2 bfloat16 token rows.

### 3. DeltaNet Path

Implement a synthetic Gated DeltaNet decode kernel:

- Q/K/V projections.
- Gating.
- Recurrent state update.
- Output projection.
- FP32 accumulation where needed.

Deliverable: Triton prototype first, followed by a CuTe/CUTLASS version if
register pressure and synchronization require lower-level control.

Status: a staged Triton synthetic DeltaNet decode block is implemented. It
matches the PyTorch reference for the tiny synthetic Qwen3.6 spec. The default
Triton path now fuses the synthetic DeltaNet projections, recurrent-state
update, and output projection into one launch, but it is not generalized to full
35B-A3B shapes.

Real-weight status: a real-shape Triton DeltaNet staging boundary now covers
input RMSNorm plus `in_proj_qkv`, `in_proj_z`, `in_proj_a`, and `in_proj_b`
over batched decode rows, followed by causal depthwise convolution and
conv-state update, recurrent-state update, gated RMSNorm, and output
projection. The full real-weight runner can select it with
`--deltanet-impl triton`.

### 4. Attention Path

Implement Gated Attention decode:

- Q/K/V projection.
- RoPE.
- KV-cache append.
- Causal decode attention.
- Output projection.

Deliverable: parity against PyTorch for small synthetic shapes.

Status: a staged Triton synthetic attention decode block is implemented. It
covers Q/K/V projection, RoPE, KV-cache append, causal decode attention, and
output projection for the tiny synthetic spec. It matches the PyTorch reference
within dtype tolerances, but it is not yet a one-launch fused attention layer
or a full-shape 35B-A3B kernel. An experimental projection+RoPE/cache fused
launch exists, but Modal A100 timing was slower than the staged default. The
attention path now supports explicit in-place KV-cache updates for decode
benchmarks, avoiding per-step cache clones while preserving the default
copying behavior for functional-style callers.

Real-weight status: the layer-3 Qwen3.6 full-attention smoke now validates a
staged real Triton boundary for RMSNorm+Q/K/V projection, Q/K norm, RoPE/cache
update, causal attention accumulation, and output projection against PyTorch.
The current correctness-first path treats the wider real `q_proj` output as
per-head `[query, gate]` pairs and applies the sigmoid gate before `o_proj`.

### 5. Layer Fusion

Fuse one full layer at a time:

- DeltaNet -> MoE.
- Attention -> MoE.
- Then 4-layer pattern.
- Then 40-layer persistent decode path.

Deliverable: staged kernels with correctness tests at each fusion boundary.

Immediate next step: settle the remaining real gated-attention projection
contract and reduce launch count at measured layer boundaries, while keeping
the staged PyTorch-parity harness available.

Status: compositional Triton layer-boundary APIs now exist for `DeltaNet -> MoE`
and `Attention -> MoE`. The real `DeltaNet -> MoE` wrapper now composes
projection, causal convolution, recurrent/output update, residual add, and MoE
staging, and has been validated on selected downloaded Qwen3.6 weights with
exact reported parity at the DeltaNet hidden/update, MoE update, layer-hidden,
conv state, recurrent state, router-logit, and top-k outputs. The attention
side remains staged until a fused version wins in measurement, but the measured
Triton benchmark path now uses in-place attention cache updates. The real
layer-3 Attention -> MoE boundary has also been validated on selected
downloaded Qwen3.6 weights with exact reported parity at the attention update,
MoE update, final layer-hidden, cache, router-logit, and top-k outputs.

The full 40-layer weight-plan lane now resolves real Qwen3.6 metadata without
downloading all shards. Modal A100 reports 693 required tensors, 40 layers,
30 DeltaNet+MoE layers, 10 Attention+MoE layers, all 26 safetensors shards, and
the expected root embedding, final norm, and LM-head keys. This proves the
full-model key layout and shard plan. A streaming PyTorch reference runner now
loads real weights layer-by-layer and runs all 40 layers on Modal H100:8 for
one-token generation. This is not yet a Triton megakernel or throughput target.

### 6. Full Model Validation

Full Qwen3.6-35B-A3B validation requires:

- Model weight access and layout conversion.
- Quantization or tensor parallelism.
- 8-GPU run for BF16 as recommended by serving examples, or a single larger /
  quantized setup if memory allows.
- Token-level numerical parity tests against Transformers, vLLM, or SGLang.
- Decode throughput and power measurements.

Current status: streaming reference serving runs through
`bench/modal_qwen36_full_serving.py --gpu H100:8 --run-full-decode`. The full
decode runner can now select Triton `Attention -> MoE` and `DeltaNet -> MoE`
layer staging over real weights; the Attention path applies the real
per-head Q projection gate and model-configured RoPE theta. A Modal A100 full
40-layer prefill/logits smoke covering both real layer-boundary wrappers passed
with all three real-weight Triton staging flags, 10 recognized four-layer
Qwen3.6 pattern chunks, `elapsed_ms_per_layer_position: 12290.7386`, and
`layer_positions_per_second: 0.0814`. The vLLM serving probe is available
through `--run-vllm`, with the default comparison lane using the prebuilt
`vllm/vllm-openai:v0.20.0-x86_64` image on H100:8. The older pip-install probe
is kept for package debugging; vLLM 0.10.2 rejects the Qwen3.6 architecture
before serving starts, and pip-installing `vllm==0.20.0` into the slim
fastkernels image fails because its source build requires `CUDA_HOME`. The
remaining work is replacing the staged streaming path with resident weights,
fewer launches, verified full-40 Triton execution, and measured end-to-end
tokens/sec.

## Lambda Test Lane

For the current single-A100 instance, run:

```sh
cd ~/fastkernels
source .venv/bin/activate
pytest -q
python bench/qwen36_decode_reference.py --device cuda --dtype float32 --steps 8 --warmup 2
python bench/qwen36_decode_reference.py --device cuda --dtype float32 --steps 8 --warmup 2 --moe-impl triton
python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --deltanet-impl triton
python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton
python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --moe-impl triton
python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
python bench/qwen36_layer_microbench.py --layer-kind deltanet_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
python bench/qwen36_layer_microbench.py --layer-kind attention_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
python bench/qwen36_moe_topk_microbench.py --device cuda --dtype bfloat16 --experts 4 --top-k 2 --warmup 10 --iters 100
python bench/qwen36_moe_topk_microbench.py --device cuda --dtype bfloat16 --experts 256 --top-k 8 --warmup 3 --iters 10
python bench/qwen36_router_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
python bench/qwen36_expert_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
python bench/qwen36_routed_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
python bench/qwen36_batched_routed_moe_microbench.py --device cuda --dtype bfloat16 --tokens 4 --warmup 3 --iters 10
python bench/qwen36_real_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
python bench/qwen36_batched_real_moe_microbench.py --device cuda --dtype bfloat16 --tokens 4 --warmup 3 --iters 10
python bench/qwen36_safetensor_moe_smoke.py --device cuda --dtype bfloat16 --tokens 2 --warmup 2 --iters 5
python bench/qwen36_real_weight_block_smoke.py --device cuda --dtype bfloat16 --tokens 2
python bench/qwen36_real_weight_moe_smoke.py --device cuda --dtype bfloat16 --tokens 2 --warmup 1 --iters 3 --layers 0 1
python bench/qwen36_vllm_moe_compare.py --device cuda --dtype bfloat16 --warmup 5 --iters 20
python bench/qwen36_vllm_moe_compare.py --device cuda --dtype bfloat16 --tokens 2 --warmup 5 --iters 20 --real-weights --require-vllm
python bench/qwen36_full_weight_plan.py --show-layers --show-shards
python bench/qwen35_profile.py --tokens 1 16 128
python bench/moe_microbench.py --device cuda --dtype bfloat16 --rows 2048 --tokens 4 --warmup 10 --iters 100 --skip-routed-moe
python bench/moe_microbench.py --device cuda --dtype bfloat16 --rows 8192 --tokens 4 --warmup 10 --iters 100 --skip-routed-moe
```

See `docs/LAMBDA.md` for launch, SSH, install, sync, benchmark, and terminate
instructions.

## References

- Qwen/Qwen3.6-35B-A3B model card:
  <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- Qwen/Qwen3.6-35B-A3B config:
  <https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json>

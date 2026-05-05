# Modal GPU Runbook

This runbook covers the current Modal path for testing the synthetic
Qwen3.6-35B-A3B decode work from this checkout.

## Scope

The Modal runner executes `bench/qwen36_decode_reference.py` on a Modal GPU. It
can run the PyTorch reference decode path, the prototype Triton synthetic
attention/DeltaNet/MoE decode paths, layer-boundary microbenchmarks, and the
synthetic MoE top-k/expert-count microbenchmarks. It can also run the first
real-shape Qwen3.6 MoE router, single-expert MLP, and routed/shared expert
microbenchmarks, batched real-shape routed/shared expert microbenchmarks, plus
the single-token and batched full real-shape MoE wrappers and optional vLLM
routed-MoE comparison microbenchmarks. It also runs an optional generated
real-shaped indexed sharded safetensors loader smoke test and a configurable
real-weight Qwen3.6 MoE shard smoke with an optional residual layer-boundary
harness, plus a real non-MoE block loader smoke for selected `linear_attn.*`
and `self_attn.*` layers. The full-attention block smoke includes a real
Triton RMSNorm+Q/K/V projection, Q/K norm, RoPE/cache update, causal attention
accumulation, and output projection check against PyTorch, plus a staged real
Attention -> MoE residual layer-boundary check. The linear-attention block
smoke now also checks real Triton DeltaNet staging for RMSNorm+QKV/Z/A/B
projections, causal depthwise convolution, conv-state update, recurrent-state
update, gated RMSNorm, and output projection.

This is not a full real-weight Qwen3.6-35B-A3B inference test. The current
benchmark uses the tiny synthetic Qwen3.6 spec, synthetic weights, and optional
Triton attention, DeltaNet, and MoE blocks. The router microbench uses the real
Qwen3.6-35B-A3B hidden size, expert count, and top-k routing shape with random
test tensors. The expert microbench uses the real hidden size and expert
intermediate size for one selected expert with random test tensors. The
routed/shared MoE microbench uses full 256-expert real-shaped random weights
and evaluates top-8 routed expert accumulation plus the shared expert. The
vLLM comparison can also run the routed-experts-only operator boundary with
selected downloaded real MoE weights. The full
real-shape MoE microbench connects the router to the routed/shared expert path
for one hidden-2048 decode token. The vLLM comparison is routed-expert-only so
it matches vLLM's `fused_moe` operator boundary; it is not an end-to-end vLLM
serving throughput benchmark. The real-weight smoke downloads the HF index and
only the selected shard files needed for requested MoE layers, including the
real shared-expert gate. The non-MoE block smoke downloads only the selected
linear-attention and full-attention shards, checks projection shapes, runs the
real DeltaNet projection+conv+recurrent/output and full-attention decode
boundaries, and composes the attention update into the real layer-3 MoE
residual boundary. These smokes do not yet load all 40 layers or generate
tokens through fastkernels. A separate
full-serving Modal runner now resolves the complete 40-layer weight plan on
A100, can run a full-model vLLM serving baseline on H100:8, and can run the
streaming real-weight path with selectable Triton Attention, DeltaNet
projection+conv+recurrent/output, Attention -> MoE layer staging, DeltaNet ->
MoE layer staging, and MoE staging.

## Modal Setup

Install and authenticate the Modal CLI:

```sh
python3 -m pip install --user modal
modal token set
```

On this workstation the CLI has previously been available at:

```sh
/Users/kb/Library/Python/3.9/bin/modal
```

If `modal` is not on `PATH`, use that full path in the commands below.

## Runner

The Modal entrypoint is:

```sh
bench/modal_qwen36_decode.py
```

It builds a Debian/Python 3.11 image, installs pinned CUDA-compatible
PyTorch/Triton packages, copies this checkout into the image, installs the repo
editable, and runs the benchmark as a subprocess on a GPU function.

The runner defaults to:

- GPU: `A100`
- dtype: `bfloat16`
- steps: `8`
- warmup: `2`
- Attention mode: `reference`
- DeltaNet mode: `reference`
- MoE modes: `reference` and `triton`
- tests: `pytest -q` before the benchmark

## Run Commands

From the repository root:

```sh
modal run bench/modal_qwen36_decode.py
```

If the CLI is not on `PATH`:

```sh
/Users/kb/Library/Python/3.9/bin/modal run bench/modal_qwen36_decode.py
```

Run only the Triton MoE path:

```sh
modal run bench/modal_qwen36_decode.py --moe-impl triton
```

Run the staged Triton DeltaNet path with the reference MoE path:

```sh
modal run bench/modal_qwen36_decode.py --deltanet-impl triton --moe-impl reference
```

Run both Triton DeltaNet and Triton MoE:

```sh
modal run bench/modal_qwen36_decode.py --deltanet-impl triton --moe-impl triton
```

Run Triton attention, DeltaNet, and MoE together:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton
```

Run the same full synthetic path plus layer-boundary microbenches:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-layer-microbench
```

Run the same full synthetic path plus layer-boundary and MoE top-k/expert-count
microbenches:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-layer-microbench --run-moe-topk-microbench
```

Run the real-shape Qwen3.6 MoE router microbenchmark:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench
```

Run the real-shape Qwen3.6 MoE router and single-expert MLP microbenchmarks:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench
```

Run the real-shape Qwen3.6 MoE router, single-expert MLP, and routed/shared MoE
microbenchmarks:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench
```

Run the full real-shape Qwen3.6 MoE wrapper microbenchmark:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench --run-real-moe-microbench
```

Run the batched real-shape Qwen3.6 routed/shared MoE microbenchmark:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench --run-real-batched-routed-moe-microbench --run-real-moe-microbench
```

Run the batched real-shape Qwen3.6 full MoE microbenchmark:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench --run-real-batched-routed-moe-microbench --run-real-moe-microbench --run-real-batched-moe-microbench
```

Run the generated indexed sharded safetensors loader smoke test:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench --run-real-batched-routed-moe-microbench --run-real-moe-microbench --run-real-batched-moe-microbench --run-safetensor-moe-smoke
```

Run the real Qwen3.6 layer-0 MoE shard smoke test:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-weight-moe-smoke
```

Run the real-weight smoke across multiple token rows and layers:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-weight-moe-smoke --real-weight-tokens 2 --real-weight-layers 0,1
```

Run the real-weight smoke with the residual `hidden + MoE` layer harness:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-weight-layer-smoke --real-weight-tokens 2 --real-weight-layers 0,1
```

Run the real-weight block smoke for one DeltaNet layer and one full-attention
layer, including staged real Attention -> MoE layer-boundary parity:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-weight-block-smoke --real-weight-tokens 2
```

Run the optional vLLM routed-MoE comparison:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench --run-real-moe-microbench --run-vllm-moe-comparison
```

Run the optional vLLM comparison against selected real MoE layer weights:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-weight-layer-smoke --real-weight-tokens 2 --real-weight-layers 0,1 --run-real-weight-vllm-moe-comparison
```

Resolve the full real Qwen3.6 40-layer weight plan without downloading every
shard:

```sh
modal run bench/modal_qwen36_full_serving.py --gpu A100 --run-weight-plan --no-run-vllm --no-run-pytest
```

Run the full-model vLLM serving baseline on 8 H100 GPUs through the prebuilt
vLLM OpenAI image:

```sh
modal run bench/modal_qwen36_full_serving.py --gpu H100:8 --run-vllm --vllm-backend openai-image --no-run-weight-plan --no-run-pytest --max-model-len 2048 --num-prompts 2 --input-len 32 --output-len 8 --tensor-parallel-size 8
```

The older package-install probe is still available for checking resolver and
build behavior in the fastkernels runtime:

```sh
modal run bench/modal_qwen36_full_serving.py --gpu H100:8 --run-vllm --vllm-backend pip --no-run-weight-plan --no-run-pytest --max-model-len 2048 --num-prompts 2 --input-len 32 --output-len 8 --tensor-parallel-size 8
```

The pip probe installs `vllm==0.20.0` by default. Override that package spec
with `--vllm-package`, for example `--vllm-package vllm`.

A standalone Modal-docs-style vLLM runner builds from
`nvidia/cuda:13.2.0-devel-ubuntu22.04` and installs `vllm==0.20.0` with
`uv --torch-backend=cu130`:

```sh
modal run bench/modal_qwen36_vllm_serving.py --max-model-len 2048 --num-prompts 2 --input-len 32 --output-len 8 --tensor-parallel-size 8
```

On April 30, 2026, the standalone runner reached real vLLM 0.20.0 H100:8 model
startup for `Qwen/Qwen3.6-35B-A3B`: it resolved
`Qwen3_5MoeForConditionalGeneration`, used TP=8/NCCL, loaded the 26-shard
66.97 GiB checkpoint, and selected FlashAttention 3 plus FlashInfer CUTLASS
MoE. Throughput was not emitted yet: the default run stopped during graph
compile/setup, and an `--enforce-eager --gdn-prefill-backend triton` retry also
stopped during post-load engine setup. The runner now mounts a persistent
`fastkernels-qwen36-hf-cache` Modal volume and streams subprocess output so the
next run can continue with cached weights and visible init logs.

Verified the full real-weight streaming runner with Triton Attention -> MoE
and DeltaNet -> MoE layer-pattern staging on Modal A100 on May 1, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-dZ3H4nNpgVxFZL0SKAI0d6

python bench/qwen36_full_decode.py --repo-id Qwen/Qwen3.6-35B-A3B --revision main --device cuda --dtype bfloat16 --prompt-token-ids 0 --max-new-tokens 0 --max-positions 128 --moe-impl triton --attention-impl triton --deltanet-impl triton --max-layers 40
layers_total: 40
layers_executed: 40
prompt_tokens: 1
generated_tokens: 0
final_position: 1
elapsed_seconds: 491.63
elapsed_ms_per_position: 491629.5447
elapsed_ms_per_layer_position: 12290.7386
layer_positions_per_second: 0.0814
attention_impl: triton
deltanet_impl: triton
moe_impl: triton
fastkernels_full_triton_attention_serving_ready: True
fastkernels_full_triton_attention_moe_layer_serving_ready: True
fastkernels_full_triton_deltanet_projection_serving_ready: True
fastkernels_full_triton_deltanet_conv_serving_ready: True
fastkernels_full_triton_deltanet_recurrent_output_serving_ready: True
fastkernels_full_triton_deltanet_moe_layer_serving_ready: True
fastkernels_full_triton_moe_serving_ready: True
fastkernels_full_triton_four_layer_pattern_chunks: 10
fastkernels_full_triton_four_layer_pattern_chunks_per_position: 10.0000
fastkernels_full_triton_four_layer_pattern_serving_ready: True
fastkernels_full_triton_attention_deltanet_moe_staging_ready: True
fastkernels_full_triton_serving_ready: False
```

The Triton Attention path now applies the real per-head Q projection gate and
the model-configured RoPE theta. The runner also accepts `--deltanet-impl
triton`, which moves real DeltaNet input RMSNorm, QKV/Z/A/B projection, causal
depthwise convolution, conv-state staging, recurrent-state update, gated
RMSNorm, and output projection to Triton. Full Triton serving remains a staged
multi-kernel path rather than a fused megakernel.

Run the full 40-layer prefill/logits smoke with all currently available
real-weight Triton staging enabled:

```sh
modal run bench/modal_qwen36_full_serving.py --gpu A100 --run-full-decode --attention-impl triton --deltanet-impl triton --moe-impl triton --max-layers 40 --output-len 0 --prompt-token-ids 0 --no-run-weight-plan --no-run-pytest --no-run-vllm
```

Skip pytest for a faster smoke run:

```sh
modal run bench/modal_qwen36_decode.py --moe-impl triton --no-run-pytest
```

Request an exact H100 rather than allowing H100-to-H200 upgrade behavior:

```sh
modal run bench/modal_qwen36_decode.py --gpu H100!
```

Modal supports `H100!` for exact-H100 benchmarking. Plain `H100` may be
automatically upgraded to H200, which is useful for capacity but less useful
when comparing hardware-specific timings.

## Expected Output

A successful run prints the command sections and benchmark output, for example:

```text
$ pytest -q
13 passed in ...

$ python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --moe-impl reference
model: synthetic-qwen3.6-a3b
device: cuda
dtype: torch.bfloat16
attention_impl: reference
deltanet_impl: reference
moe_impl: reference
steps: 8
final_position: 8
logits_shape: (64,)
decode_ms_per_step: ...

$ python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --moe-impl triton
model: synthetic-qwen3.6-a3b
device: cuda
dtype: torch.bfloat16
attention_impl: reference
deltanet_impl: reference
moe_impl: triton
steps: 8
final_position: 8
logits_shape: (64,)
reference_max_abs_diff: 0.000000
decode_ms_per_step: ...
```

The important correctness signal for the current Triton path is
`reference_max_abs_diff`; exact-zero is expected for the tiny MoE/DeltaNet
paths, while the staged attention path is checked against dtype tolerances.

## Verified Result

Verified real-weight MoE and real-weight vLLM operator comparison on Modal A100
on April 29, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-c9zKZpalLnwXl8f4lEAwat

pytest -q
50 passed in 24.54s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.6468

real-weight layer 0 real_weight_batched_moe_ms_per_token: 0.1968
real-weight layer 0 real_weight_batched_moe_layer_ms_per_token: 0.1615
real-weight layer 1 real_weight_batched_moe_ms_per_token: 0.1157
real-weight layer 1 real_weight_batched_moe_layer_ms_per_token: 0.1305
real-weight layer parity max_abs_diff: 0.000000

real-weight vLLM comparison scope: routed_experts_only
real-weight vLLM comparison fastkernels_routed_max_abs_diff: 0.000000
real-weight vLLM comparison fastkernels_routed_moe_ms_per_token: 0.0483
real-weight vLLM comparison fastkernels_routed_moe_tokens_per_second: 20712.68
real-weight vLLM comparison vllm_version: 0.10.2
real-weight vLLM comparison vllm_routed_max_abs_diff: 0.000000
real-weight vLLM comparison vllm_routed_moe_ms_per_token: 0.2306
real-weight vLLM comparison vllm_routed_moe_tokens_per_second: 4336.93
real-weight vLLM comparison fastkernels_to_vllm_ms_ratio: 0.2094
```

vLLM emitted its default-MoE-config warning for `E=256,N=512` on A100, so this
remains an operator-boundary checkpoint rather than a final serving baseline.

Verified real-weight DeltaNet -> MoE, full-attention, and Attention -> MoE block
smoke on Modal A100 on May 1, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-yQ6FeDIoMl3WJal1yiVjAN

linear_triton_qkv_max_abs_diff: 0.000001
linear_triton_z_max_abs_diff: 0.000001
linear_triton_a_max_abs_diff: 0.000003
linear_triton_b_max_abs_diff: 0.000002
linear_triton_conv_max_abs_diff: 0.000000
linear_triton_conv_state_max_abs_diff: 0.000061
linear_triton_update_max_abs_diff: 0.000000
linear_triton_recurrent_state_max_abs_diff: 0.000002
linear_triton_full_update_ms_per_token: 0.2771
linear_moe_deltanet_hidden_max_abs_diff: 0.000000
linear_moe_deltanet_update_max_abs_diff: 0.000000
linear_moe_update_max_abs_diff: 0.000000
linear_moe_layer_hidden_max_abs_diff: 0.000000
linear_moe_conv_state_max_abs_diff: 0.000061
linear_moe_recurrent_state_max_abs_diff: 0.000002
linear_moe_logits_max_abs_diff: 0.000005
linear_moe_topk_ids_match: True
linear_moe_topk_weights_max_abs_diff: 0.000000
linear_moe_layer_ms_per_token: 0.4462

attention_decode_update_max_abs_diff: 0.000000
attention_decode_key_cache_max_abs_diff: 0.000001
attention_decode_value_cache_max_abs_diff: 0.000000
attention_triton_decode_ms_per_token: 0.3144

attention_moe_attention_hidden_max_abs_diff: 0.000000
attention_moe_attention_update_max_abs_diff: 0.000000
attention_moe_update_max_abs_diff: 0.000000
attention_moe_layer_hidden_max_abs_diff: 0.000000
attention_moe_key_cache_max_abs_diff: 0.000001
attention_moe_value_cache_max_abs_diff: 0.000000
attention_moe_logits_max_abs_diff: 0.000002
attention_moe_topk_ids_match: True
attention_moe_topk_weights_max_abs_diff: 0.000000
attention_moe_layer_ms_per_token: 0.4816
```

Verified full 40-layer Qwen3.6 weight-plan resolution on Modal A100 on
April 29, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-MboqIBec8UuY4UC4uNdnyY

layers: 40
layer_counts: {'deltanet_moe': 30, 'attention_moe': 10}
required_tensor_count: 693
required_shard_count: 26
hf_total_size_bytes: 71903645408.0
root_embedding_key: model.language_model.embed_tokens.weight
root_output_norm_key: model.language_model.norm.weight
root_lm_head_key: lm_head.weight
fastkernels_full_reference_serving_available: True
fastkernels_full_triton_serving_ready: False
fastkernels_full_triton_serving_blocker: streaming reference path has not been replaced by fused kernels
vllm_full_serving_probe_available: True
vllm_full_serving_ready: unknown_until_benchmark
recommended_vllm_tensor_parallel_size: 8
```

Verified full 40-layer streaming PyTorch reference serving on Modal H100:8 on
April 29, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-Vo5UVobsMbzEFYoJbWGuRG

model: Qwen3.6-35B-A3B
repo_id: Qwen/Qwen3.6-35B-A3B
revision: main
device: cuda
dtype: torch.bfloat16
layers_total: 40
layers_executed: 40
prompt_tokens: 1
generated_tokens: 1
generated_token_ids: 198
final_position: 2
elapsed_seconds: 454.16
fastkernels_full_reference_serving_ready: True
fastkernels_full_triton_serving_ready: False
serving_scope: streaming PyTorch reference over real Qwen3.6 weights
```

The same runner also completed a full-40 prefill/logits smoke without
generation at
https://modal.com/apps/suryavanshi/main/ap-6j3WCY55FZHDSSb0FUSgG3.

Verified the streaming real-weight path with Triton MoE sublayers on Modal A100
on April 30, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-yF7mjRvl30IxAiRCH4St7a

layers_total: 40
layers_executed: 4
prompt_tokens: 1
generated_tokens: 0
final_position: 1
elapsed_seconds: 49.76
moe_impl: triton
fastkernels_full_triton_moe_serving_ready: True
fastkernels_full_triton_serving_ready: False
```

Attempted full-model vLLM serving on Modal H100:8 on April 29, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-NNESTpgzPcXtJfVTCVoAAj

backend: vllm
vllm_version: 0.10.2
model: Qwen/Qwen3.6-35B-A3B
model_impl: auto
vllm_full_serving_supported: False
vllm_full_serving_error_type: ValidationError
vllm_full_serving_error: Model architectures ['Qwen3_5MoeForConditionalGeneration'] are not supported for now.
```

The Transformers fallback probe also fails before serving starts:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-PS0DL0O9tVDVERz3Ws7XkF

backend: vllm
vllm_version: 0.10.2
model: Qwen/Qwen3.6-35B-A3B
model_impl: transformers
vllm_full_serving_supported: False
vllm_full_serving_error_type: ValidationError
vllm_full_serving_error: The Transformers implementation of 'Qwen3_5MoeForConditionalGeneration' is not compatible with vLLM.
```

Attempted the requested `vllm==0.20.0` package comparison on Modal H100:8 on
April 30, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-oOD12jbEA9t4PZqpnNFLZ0

backend: vllm
vllm_package: vllm==0.20.0
vllm_install_supported: False
vllm_install_exit_code: 1
vllm_full_serving_supported: False
vllm_full_serving_error_type: InstallError
vllm_full_serving_error: source build requires CUDA_HOME during build metadata
```

Verified on Modal A100 on April 25, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-GAIFWyRa0tXzBThWb9MzMR

pytest -q
13 passed in 10.66s

reference bfloat16 decode_ms_per_step: 5.0975
triton bfloat16 reference_max_abs_diff: 0.000000
triton bfloat16 decode_ms_per_step: 2.8836
```

Verified staged Triton DeltaNet plus Triton MoE on Modal A100 on April 25,
2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-vx1uEKfrlLMqQIynomFcsB

pytest -q
15 passed in 16.51s

deltanet triton + moe triton reference_max_abs_diff: 0.000000
deltanet triton + moe triton decode_ms_per_step: 2.9029
```

Verified staged Triton attention, DeltaNet, and MoE on Modal A100 on April 25,
2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-vZaqGDgdbYBXjuNz4WJqKe

pytest -q
17 passed in 20.44s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000977
attention triton + deltanet triton + moe triton decode_ms_per_step: 1.3155
```

Verified layer-boundary microbenches on Modal A100 on April 25, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-EIoAiuhLAORfYaM796043P

pytest -q
21 passed in 15.83s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000977
attention triton + deltanet triton + moe triton decode_ms_per_step: 1.3880

deltanet_moe triton hidden_max_abs_diff: 0.000977
deltanet_moe triton state_max_abs_diff: 0.007812
deltanet_moe triton layer_ms: 0.2697

attention_moe triton hidden_max_abs_diff: 0.003906
attention_moe triton state_max_abs_diff: 0.000549
attention_moe triton value_cache_max_abs_diff: 0.000488
attention_moe triton layer_ms: 0.3303
```

Verified reduced-launch DeltaNet default on Modal A100 on April 26, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-Vl5zka7rZjxLj2T8Hd6nVH

pytest -q
21 passed in 11.13s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.6957

deltanet_moe triton hidden_max_abs_diff: 0.000977
deltanet_moe triton state_max_abs_diff: 0.007812
deltanet_moe triton layer_ms: 0.0909

attention_moe triton hidden_max_abs_diff: 0.000488
attention_moe triton state_max_abs_diff: 0.000549
attention_moe triton value_cache_max_abs_diff: 0.000488
attention_moe triton layer_ms: 0.2233
```

Verified generalized top-k MoE coverage on Modal A100 on April 26, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-gKxIaV5RcPcDRy0hCk4wIC

pytest -q
27 passed in 16.62s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.7675

deltanet_moe triton hidden_max_abs_diff: 0.000977
deltanet_moe triton state_max_abs_diff: 0.007812
deltanet_moe triton layer_ms: 0.0963

attention_moe triton hidden_max_abs_diff: 0.000488
attention_moe triton state_max_abs_diff: 0.000549
attention_moe triton value_cache_max_abs_diff: 0.000488
attention_moe triton layer_ms: 0.2308
```

Verified 256-expert/top-8 synthetic MoE coverage on Modal A100 on April 26,
2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-JHLkSh2mWDDtpk7b7RKeO3

pytest -q
28 passed in 29.51s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 1.1699

deltanet_moe triton hidden_max_abs_diff: 0.000977
deltanet_moe triton state_max_abs_diff: 0.007812
deltanet_moe triton layer_ms: 0.1247

attention_moe triton hidden_max_abs_diff: 0.000092
attention_moe triton state_max_abs_diff: 0.000549
attention_moe triton value_cache_max_abs_diff: 0.000488
attention_moe triton layer_ms: 0.2957

moe top-2 / 4 experts max_abs_diff: 0.000000
moe top-2 / 4 experts moe_ms: 0.1094

moe top-8 / 256 experts max_abs_diff: 0.000000
moe top-8 / 256 experts moe_ms: 0.0763
```

Verified in-place synthetic attention KV-cache updates on Modal A100 on April
27, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-V5i7e5SuWuyR3Ig8E6Wt4r

pytest -q
29 passed in 25.62s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.9347

deltanet_moe triton hidden_max_abs_diff: 0.000977
deltanet_moe triton state_max_abs_diff: 0.007812
deltanet_moe triton layer_ms: 0.1201

attention_moe triton hidden_max_abs_diff: 0.000977
attention_moe triton state_max_abs_diff: 0.000549
attention_moe triton value_cache_max_abs_diff: 0.000488
attention_moe triton layer_ms: 0.2437

moe top-2 / 4 experts max_abs_diff: 0.000000
moe top-2 / 4 experts moe_ms: 0.0563

moe top-8 / 256 experts max_abs_diff: 0.000000
moe top-8 / 256 experts moe_ms: 0.0750
```

Verified real-shape Qwen3.6 MoE router decode on Modal A100 on April 27, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-7RVhhZe6zISTwewePFZo2Y

pytest -q
32 passed in 19.34s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.7329

real Qwen3.6 router hidden_size: 2048
real Qwen3.6 router experts: 256
real Qwen3.6 router top_k: 8
real Qwen3.6 router logits_max_abs_diff: 0.000023
real Qwen3.6 router topk_ids_match: True
real Qwen3.6 router topk_weights_max_abs_diff: 0.000002
real Qwen3.6 router router_ms: 0.0672
```

Verified real-shape Qwen3.6 router plus single-expert MLP decode on Modal A100
on April 27, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-iXUfacqJHSqgtEbP0uASZp

pytest -q
34 passed in 16.97s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.6767

real Qwen3.6 router hidden_size: 2048
real Qwen3.6 router experts: 256
real Qwen3.6 router top_k: 8
real Qwen3.6 router logits_max_abs_diff: 0.000023
real Qwen3.6 router topk_ids_match: True
real Qwen3.6 router topk_weights_max_abs_diff: 0.000002
real Qwen3.6 router router_ms: 0.0690

real Qwen3.6 expert hidden_size: 2048
real Qwen3.6 expert intermediate: 512
real Qwen3.6 expert max_abs_diff: 0.000000
real Qwen3.6 expert expert_mlp_ms: 0.0627
```

Verified real-shape Qwen3.6 routed/shared MoE composition on Modal A100 on
April 27, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-AyTsQi4NzxVZOUnCvmg3Qq

pytest -q
35 passed in 16.19s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.6552

real Qwen3.6 router hidden_size: 2048
real Qwen3.6 router experts: 256
real Qwen3.6 router top_k: 8
real Qwen3.6 router logits_max_abs_diff: 0.000023
real Qwen3.6 router topk_ids_match: True
real Qwen3.6 router topk_weights_max_abs_diff: 0.000002
real Qwen3.6 router router_ms: 0.0688

real Qwen3.6 expert hidden_size: 2048
real Qwen3.6 expert intermediate: 512
real Qwen3.6 expert max_abs_diff: 0.000000
real Qwen3.6 expert expert_mlp_ms: 0.0616

real Qwen3.6 routed/shared hidden_size: 2048
real Qwen3.6 routed/shared experts: 256
real Qwen3.6 routed/shared top_k: 8
real Qwen3.6 routed/shared intermediate: 512
real Qwen3.6 routed/shared max_abs_diff: 0.000000
real Qwen3.6 routed/shared routed_shared_moe_ms: 1.2023
```

Verified batched two-launch real-shape Qwen3.6 routed/shared MoE on Modal A100
on April 27, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-qz9JM3rCDedgsFr7NVUVBZ

pytest -q
35 passed in 33.57s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.9648

real Qwen3.6 router hidden_size: 2048
real Qwen3.6 router experts: 256
real Qwen3.6 router top_k: 8
real Qwen3.6 router logits_max_abs_diff: 0.000023
real Qwen3.6 router topk_ids_match: True
real Qwen3.6 router topk_weights_max_abs_diff: 0.000002
real Qwen3.6 router router_ms: 0.0863

real Qwen3.6 expert hidden_size: 2048
real Qwen3.6 expert intermediate: 512
real Qwen3.6 expert max_abs_diff: 0.000000
real Qwen3.6 expert expert_mlp_ms: 0.0746

real Qwen3.6 routed/shared hidden_size: 2048
real Qwen3.6 routed/shared experts: 256
real Qwen3.6 routed/shared top_k: 8
real Qwen3.6 routed/shared intermediate: 512
real Qwen3.6 routed/shared max_abs_diff: 0.000000
real Qwen3.6 routed/shared routed_shared_moe_ms: 0.1211
```

Verified full real-shape Qwen3.6 MoE wrapper on Modal A100 on April 27, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-ixTieTqDIcWv03wRIKdd54

pytest -q
36 passed in 22.87s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.7128

real Qwen3.6 router hidden_size: 2048
real Qwen3.6 router experts: 256
real Qwen3.6 router top_k: 8
real Qwen3.6 router logits_max_abs_diff: 0.000023
real Qwen3.6 router topk_ids_match: True
real Qwen3.6 router topk_weights_max_abs_diff: 0.000002
real Qwen3.6 router router_ms: 0.0661

real Qwen3.6 expert hidden_size: 2048
real Qwen3.6 expert intermediate: 512
real Qwen3.6 expert max_abs_diff: 0.000000
real Qwen3.6 expert expert_mlp_ms: 0.0596

real Qwen3.6 routed/shared hidden_size: 2048
real Qwen3.6 routed/shared experts: 256
real Qwen3.6 routed/shared top_k: 8
real Qwen3.6 routed/shared intermediate: 512
real Qwen3.6 routed/shared max_abs_diff: 0.000000
real Qwen3.6 routed/shared routed_shared_moe_ms: 0.0924

real Qwen3.6 full MoE hidden_size: 2048
real Qwen3.6 full MoE experts: 256
real Qwen3.6 full MoE top_k: 8
real Qwen3.6 full MoE intermediate: 512
real Qwen3.6 full MoE output_max_abs_diff: 0.000000
real Qwen3.6 full MoE logits_max_abs_diff: 0.000023
real Qwen3.6 full MoE topk_ids_match: True
real Qwen3.6 full MoE topk_weights_max_abs_diff: 0.000001
real Qwen3.6 full MoE real_moe_ms: 0.1778
```

Verified vLLM routed-MoE operator comparison on Modal A100 on April 27, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-QoAinlMxh04ImGPRmpPPBO

pytest -q
37 passed in 20.03s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.6580

real Qwen3.6 router hidden_size: 2048
real Qwen3.6 router experts: 256
real Qwen3.6 router top_k: 8
real Qwen3.6 router logits_max_abs_diff: 0.000023
real Qwen3.6 router topk_ids_match: True
real Qwen3.6 router topk_weights_max_abs_diff: 0.000002
real Qwen3.6 router router_ms: 0.0659

real Qwen3.6 expert hidden_size: 2048
real Qwen3.6 expert intermediate: 512
real Qwen3.6 expert max_abs_diff: 0.000000
real Qwen3.6 expert expert_mlp_ms: 0.0616

real Qwen3.6 routed/shared hidden_size: 2048
real Qwen3.6 routed/shared experts: 256
real Qwen3.6 routed/shared top_k: 8
real Qwen3.6 routed/shared intermediate: 512
real Qwen3.6 routed/shared max_abs_diff: 0.000000
real Qwen3.6 routed/shared routed_shared_moe_ms: 0.0932

real Qwen3.6 full MoE hidden_size: 2048
real Qwen3.6 full MoE experts: 256
real Qwen3.6 full MoE top_k: 8
real Qwen3.6 full MoE intermediate: 512
real Qwen3.6 full MoE output_max_abs_diff: 0.000000
real Qwen3.6 full MoE logits_max_abs_diff: 0.000023
real Qwen3.6 full MoE topk_ids_match: True
real Qwen3.6 full MoE topk_weights_max_abs_diff: 0.000001
real Qwen3.6 full MoE real_moe_ms: 0.1751

vLLM comparison vllm_version: 0.10.2
vLLM comparison fastkernels_routed_max_abs_diff: 0.000000
vLLM comparison fastkernels_routed_moe_ms: 0.0859
vLLM comparison vllm_routed_max_abs_diff: 0.000000
vLLM comparison vllm_routed_moe_ms: 0.4841
vLLM comparison fastkernels_to_vllm_ms_ratio: 0.1775
```

vLLM emitted a warning that it used its default MoE config because no config
file matched `E=256,N=512,device_name=NVIDIA_A100-SXM4-40GB.json`. Do not treat
this as a final vLLM serving baseline.

Verified batched real-shape Qwen3.6 routed/shared MoE on Modal A100 on April
27, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-775CKcZ1gFDt9lS9Sjh1PO

pytest -q
38 passed in 25.86s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.7369

real Qwen3.6 router hidden_size: 2048
real Qwen3.6 router experts: 256
real Qwen3.6 router top_k: 8
real Qwen3.6 router logits_max_abs_diff: 0.000023
real Qwen3.6 router topk_ids_match: True
real Qwen3.6 router topk_weights_max_abs_diff: 0.000002
real Qwen3.6 router router_ms: 0.0663

real Qwen3.6 expert hidden_size: 2048
real Qwen3.6 expert intermediate: 512
real Qwen3.6 expert max_abs_diff: 0.000000
real Qwen3.6 expert expert_mlp_ms: 0.0619

real Qwen3.6 routed/shared hidden_size: 2048
real Qwen3.6 routed/shared experts: 256
real Qwen3.6 routed/shared top_k: 8
real Qwen3.6 routed/shared intermediate: 512
real Qwen3.6 routed/shared max_abs_diff: 0.000000
real Qwen3.6 routed/shared routed_shared_moe_ms: 0.0938

real Qwen3.6 batched routed/shared tokens: 4
real Qwen3.6 batched routed/shared hidden_size: 2048
real Qwen3.6 batched routed/shared experts: 256
real Qwen3.6 batched routed/shared top_k: 8
real Qwen3.6 batched routed/shared intermediate: 512
real Qwen3.6 batched routed/shared max_abs_diff: 0.000000
real Qwen3.6 batched routed/shared batched_routed_shared_moe_ms: 0.2480
real Qwen3.6 batched routed/shared batched_routed_shared_moe_ms_per_token: 0.0620

real Qwen3.6 full MoE hidden_size: 2048
real Qwen3.6 full MoE experts: 256
real Qwen3.6 full MoE top_k: 8
real Qwen3.6 full MoE intermediate: 512
real Qwen3.6 full MoE output_max_abs_diff: 0.000000
real Qwen3.6 full MoE logits_max_abs_diff: 0.000023
real Qwen3.6 full MoE topk_ids_match: True
real Qwen3.6 full MoE topk_weights_max_abs_diff: 0.000001
real Qwen3.6 full MoE real_moe_ms: 0.1742
```

Verified batched full real-shape Qwen3.6 MoE on Modal A100 on April 27, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-SCRpc6fMuwfdtPRTxelmV4

pytest -q
40 passed in 34.91s

attention triton + deltanet triton + moe triton reference_max_abs_diff: 0.000488
attention triton + deltanet triton + moe triton decode_ms_per_step: 0.8318

real Qwen3.6 router hidden_size: 2048
real Qwen3.6 router experts: 256
real Qwen3.6 router top_k: 8
real Qwen3.6 router logits_max_abs_diff: 0.000023
real Qwen3.6 router topk_ids_match: True
real Qwen3.6 router topk_weights_max_abs_diff: 0.000002
real Qwen3.6 router router_ms: 0.0830

real Qwen3.6 expert hidden_size: 2048
real Qwen3.6 expert intermediate: 512
real Qwen3.6 expert max_abs_diff: 0.000000
real Qwen3.6 expert expert_mlp_ms: 0.0811

real Qwen3.6 routed/shared hidden_size: 2048
real Qwen3.6 routed/shared experts: 256
real Qwen3.6 routed/shared top_k: 8
real Qwen3.6 routed/shared intermediate: 512
real Qwen3.6 routed/shared max_abs_diff: 0.000000
real Qwen3.6 routed/shared routed_shared_moe_ms: 0.1164

real Qwen3.6 batched routed/shared tokens: 4
real Qwen3.6 batched routed/shared max_abs_diff: 0.000000
real Qwen3.6 batched routed/shared batched_routed_shared_moe_ms: 0.2505
real Qwen3.6 batched routed/shared batched_routed_shared_moe_ms_per_token: 0.0626

real Qwen3.6 full MoE output_max_abs_diff: 0.000000
real Qwen3.6 full MoE logits_max_abs_diff: 0.000023
real Qwen3.6 full MoE topk_ids_match: True
real Qwen3.6 full MoE topk_weights_max_abs_diff: 0.000001
real Qwen3.6 full MoE real_moe_ms: 0.2286

real Qwen3.6 batched full MoE tokens: 4
real Qwen3.6 batched full MoE output_max_abs_diff: 0.000000
real Qwen3.6 batched full MoE logits_max_abs_diff: 0.000031
real Qwen3.6 batched full MoE topk_ids_match: True
real Qwen3.6 batched full MoE topk_weights_max_abs_diff: 0.000002
real Qwen3.6 batched full MoE batched_real_moe_ms: 0.2701
real Qwen3.6 batched full MoE batched_real_moe_ms_per_token: 0.0675
```

Verified generated indexed sharded safetensors MoE loader smoke on Modal A100
on April 28, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-ErpI7lmb9nGtwgN08ILDNO

pytest -q
43 passed in 27.62s

decode reference_max_abs_diff: 0.000488
decode decode_ms_per_step: 0.7225

router logits_max_abs_diff: 0.000023
router topk_ids_match: True
router topk_weights_max_abs_diff: 0.000002
router router_ms: 0.0655

expert max_abs_diff: 0.000000
expert expert_mlp_ms: 0.0616

routed/shared max_abs_diff: 0.000000
routed/shared routed_shared_moe_ms: 0.0957

batched routed/shared max_abs_diff: 0.000000
batched routed/shared batched_routed_shared_moe_ms: 0.2471
batched routed/shared batched_routed_shared_moe_ms_per_token: 0.0618

full MoE output_max_abs_diff: 0.000000
full MoE logits_max_abs_diff: 0.000023
full MoE topk_ids_match: True
full MoE topk_weights_max_abs_diff: 0.000001
full MoE real_moe_ms: 0.1815

batched full MoE output_max_abs_diff: 0.000000
batched full MoE logits_max_abs_diff: 0.000031
batched full MoE topk_ids_match: True
batched full MoE topk_weights_max_abs_diff: 0.000002
batched full MoE batched_real_moe_ms: 0.2666
batched full MoE batched_real_moe_ms_per_token: 0.0667

safetensor smoke resolved_norm_key: model.layers.0.post_attention_layernorm.weight
safetensor smoke resolved_router_key: model.layers.0.mlp.gate.weight
safetensor smoke safetensor_indexed: True
safetensor smoke output_max_abs_diff: 0.000000
safetensor smoke logits_max_abs_diff: 0.000031
safetensor smoke topk_ids_match: True
safetensor smoke topk_weights_max_abs_diff: 0.000000
safetensor smoke safetensor_batched_real_moe_ms: 0.2159
safetensor smoke safetensor_batched_real_moe_ms_per_token: 0.1079
```

Verified real-weight Qwen3.6 block loading plus gated multi-layer MoE shard
smoke and residual layer harness on Modal A100 on April 29, 2026:

```text
Run URL: https://modal.com/apps/suryavanshi/main/ap-5HOgeKfIAnhNNJkjAA60bK

pytest -q
49 passed in 39.79s

synthetic decode reference_max_abs_diff: 0.000488
synthetic decode decode_ms_per_step: 0.9423

real-weight block layer: 0
real-weight block layer_kind: deltanet_moe
real-weight block downloaded_shards: model-00001-of-00026.safetensors,model-00002-of-00026.safetensors
real-weight block needed_tensor_count: 11
real-weight block resolved_in_proj_qkv_key: model.language_model.layers.0.linear_attn.in_proj_qkv.weight
real-weight block resolved_in_proj_z_key: model.language_model.layers.0.linear_attn.in_proj_z.weight
real-weight block resolved_out_proj_key: model.language_model.layers.0.linear_attn.out_proj.weight
real-weight block linear_in_proj_qkv_shape: (8192, 2048)
real-weight block linear_in_proj_z_shape: (4096, 2048)
real-weight block linear_out_proj_shape: (2048, 4096)
real-weight block linear_conv1d_shape: (8192, 1, 4)
real-weight block linear_qkv_projection_shape: (2, 8192)
real-weight block linear_z_projection_shape: (2, 4096)

real-weight block layer: 3
real-weight block layer_kind: attention_moe
real-weight block downloaded_shards: model-00003-of-00026.safetensors
real-weight block needed_tensor_count: 8
real-weight block resolved_q_proj_key: model.language_model.layers.3.self_attn.q_proj.weight
real-weight block resolved_k_proj_key: model.language_model.layers.3.self_attn.k_proj.weight
real-weight block resolved_v_proj_key: model.language_model.layers.3.self_attn.v_proj.weight
real-weight block resolved_o_proj_key: model.language_model.layers.3.self_attn.o_proj.weight
real-weight block attention_q_proj_shape: (8192, 2048)
real-weight block attention_k_proj_shape: (512, 2048)
real-weight block attention_v_proj_shape: (512, 2048)
real-weight block attention_o_proj_shape: (2048, 4096)
real-weight block attention_q_projection_shape: (2, 8192)
real-weight block attention_k_projection_shape: (2, 512)
real-weight block attention_v_projection_shape: (2, 512)
real-weight block attention_triton_q_max_abs_diff: 0.000000
real-weight block attention_triton_k_max_abs_diff: 0.000000
real-weight block attention_triton_v_max_abs_diff: 0.000000
real-weight block attention_triton_project_ms: 0.2113
real-weight block attention_triton_project_ms_per_token: 0.1056

real-weight smoke repo_id: Qwen/Qwen3.6-35B-A3B
real-weight smoke revision: main
real-weight smoke tokens: 2
real-weight smoke layer: 0
real-weight smoke hf_total_size_bytes: 71903645408.0
real-weight smoke downloaded_shards: model-00001-of-00026.safetensors,model-00002-of-00026.safetensors
real-weight smoke downloaded_shard_count: 2
real-weight smoke needed_tensor_count: 8
real-weight smoke resolved_norm_key: model.language_model.layers.0.post_attention_layernorm.weight
real-weight smoke resolved_router_key: model.language_model.layers.0.mlp.gate.weight
real-weight smoke resolved_expert_gate_up_key: model.language_model.layers.0.mlp.experts.gate_up_proj
real-weight smoke resolved_expert_down_key: model.language_model.layers.0.mlp.experts.down_proj
real-weight smoke shared_expert_gate_present: True
real-weight smoke shared_expert_gate_applied: True
real-weight smoke shared_expert_gate_ungated_gap_max_abs: 0.000115
real-weight smoke output_max_abs_diff: 0.000000
real-weight smoke logits_max_abs_diff: 0.000000
real-weight smoke topk_ids_match: True
real-weight smoke topk_weights_max_abs_diff: 0.000000
real-weight smoke real_weight_batched_moe_ms: 0.3577
real-weight smoke real_weight_batched_moe_ms_per_token: 0.1788
real-weight smoke real_weight_layer_harness: True
real-weight smoke layer_output_max_abs_diff: 0.000000
real-weight smoke layer_update_max_abs_diff: 0.000000
real-weight smoke layer_logits_max_abs_diff: 0.000000
real-weight smoke layer_topk_ids_match: True
real-weight smoke layer_topk_weights_max_abs_diff: 0.000000
real-weight smoke real_weight_batched_moe_layer_ms: 0.4036
real-weight smoke real_weight_batched_moe_layer_ms_per_token: 0.2018

real-weight smoke layer: 1
real-weight smoke downloaded_shards: model-00001-of-00026.safetensors,model-00002-of-00026.safetensors
real-weight smoke resolved_norm_key: model.language_model.layers.1.post_attention_layernorm.weight
real-weight smoke resolved_router_key: model.language_model.layers.1.mlp.gate.weight
real-weight smoke resolved_expert_gate_up_key: model.language_model.layers.1.mlp.experts.gate_up_proj
real-weight smoke resolved_expert_down_key: model.language_model.layers.1.mlp.experts.down_proj
real-weight smoke shared_expert_gate_present: True
real-weight smoke shared_expert_gate_applied: True
real-weight smoke shared_expert_gate_ungated_gap_max_abs: 0.000078
real-weight smoke output_max_abs_diff: 0.000000
real-weight smoke logits_max_abs_diff: 0.000000
real-weight smoke topk_ids_match: True
real-weight smoke topk_weights_max_abs_diff: 0.000000
real-weight smoke real_weight_batched_moe_ms: 0.2863
real-weight smoke real_weight_batched_moe_ms_per_token: 0.1432
real-weight smoke real_weight_layer_harness: True
real-weight smoke layer_output_max_abs_diff: 0.000000
real-weight smoke layer_update_max_abs_diff: 0.000000
real-weight smoke layer_logits_max_abs_diff: 0.000000
real-weight smoke layer_topk_ids_match: True
real-weight smoke layer_topk_weights_max_abs_diff: 0.000000
real-weight smoke real_weight_batched_moe_layer_ms: 0.3151
real-weight smoke real_weight_batched_moe_layer_ms_per_token: 0.1575
```

## Notes

- The runner copies the local checkout into the Modal image, so uncommitted
  local code changes are included in the run.
- The image uses pinned `torch==2.7.1` and `triton==3.3.1`, matching the CUDA
  setup used for the Lambda test lane.
- No Modal Volume is required for the synthetic benchmark. Use a Volume later
  for real model weights, datasets, or durable logs.
- Modal ephemeral apps created by `modal run` stop when the command exits, so
  GPU billing should stop with the run. Check the Modal dashboard if a command
  is detached or interrupted.

## References

- Modal GPU docs: <https://modal.com/docs/guide/gpu>
- Modal image docs: <https://modal.com/docs/guide/images>
- Modal apps and entrypoints: <https://modal.com/docs/guide/apps>
- Modal run CLI: <https://modal.com/docs/reference/cli/run>
- Modal volumes: <https://modal.com/docs/guide/volumes>

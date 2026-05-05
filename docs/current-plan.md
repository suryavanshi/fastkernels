# Current Plan

## Objective

Progress from the current MoE building-block kernels toward a Qwen3.6-35B-A3B
decode megakernel, using deterministic synthetic references and Lambda GPU tests
at each step.

## Verified Baseline

On Lambda `gpu_1x_a100_sxm4`:

```text
pytest -q
13 passed in 3.25s

python bench/qwen36_decode_reference.py --device cuda --dtype float32 --steps 8 --warmup 2 --moe-impl reference
decode_ms_per_step: 5.0384

python bench/qwen36_decode_reference.py --device cuda --dtype float32 --steps 8 --warmup 2 --moe-impl triton
reference_max_abs_diff: 0.000000
decode_ms_per_step: 2.9017

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --moe-impl reference
decode_ms_per_step: 4.9950

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --moe-impl triton
reference_max_abs_diff: 0.000000
decode_ms_per_step: 2.9876

On Modal A100:

Run URL: https://modal.com/apps/suryavanshi/main/ap-c9zKZpalLnwXl8f4lEAwat

pytest -q
50 passed in 24.54s

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.6468

python bench/qwen36_real_weight_moe_smoke.py --device cuda --dtype bfloat16 --tokens 2 --warmup 1 --iters 3 --layers 0 1 --run-layer-harness
layer 0 real_weight_batched_moe_ms_per_token: 0.1968
layer 0 real_weight_batched_moe_layer_ms_per_token: 0.1615
layer 1 real_weight_batched_moe_ms_per_token: 0.1157
layer 1 real_weight_batched_moe_layer_ms_per_token: 0.1305
layer outputs/logits/top-k parity: exact within reported max_abs_diff 0.000000

python bench/qwen36_vllm_moe_compare.py --device cuda --dtype bfloat16 --tokens 2 --layer 0 --warmup 5 --iters 20 --real-weights --require-vllm
comparison_scope: routed_experts_only
fastkernels_routed_max_abs_diff: 0.000000
fastkernels_routed_moe_ms: 0.0966
fastkernels_routed_moe_ms_per_token: 0.0483
fastkernels_routed_moe_tokens_per_second: 20712.68
vllm_version: 0.10.2
vllm_routed_max_abs_diff: 0.000000
vllm_routed_moe_ms: 0.4612
vllm_routed_moe_ms_per_token: 0.2306
vllm_routed_moe_tokens_per_second: 4336.93
fastkernels_to_vllm_ms_ratio: 0.2094

Latest real-weight DeltaNet -> MoE, full-attention, and Attention -> MoE block run:

Run URL: https://modal.com/apps/suryavanshi/main/ap-yQ6FeDIoMl3WJal1yiVjAN

python bench/qwen36_real_weight_block_smoke.py --device cuda --dtype bfloat16 --tokens 2
model: Qwen3.6-35B-A3B
repo_id: Qwen/Qwen3.6-35B-A3B
revision: main
tokens: 2
hf_total_size_bytes: 71903645408.0
layer: 0
layer_kind: deltanet_moe
downloaded_shards: model-00001-of-00026.safetensors,model-00002-of-00026.safetensors
needed_tensor_count: 11
resolved_input_norm_key: model.language_model.layers.0.input_layernorm.weight
resolved_in_proj_qkv_key: model.language_model.layers.0.linear_attn.in_proj_qkv.weight
resolved_in_proj_z_key: model.language_model.layers.0.linear_attn.in_proj_z.weight
resolved_out_proj_key: model.language_model.layers.0.linear_attn.out_proj.weight
linear_in_proj_qkv_shape: (8192, 2048)
linear_in_proj_z_shape: (4096, 2048)
linear_out_proj_shape: (2048, 4096)
linear_conv1d_shape: (8192, 1, 4)
linear_qkv_projection_shape: (2, 8192)
linear_z_projection_shape: (2, 4096)
linear_a_projection_shape: (2, 32)
linear_b_projection_shape: (2, 32)
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

layer: 3
layer_kind: attention_moe
downloaded_shards: model-00003-of-00026.safetensors
needed_tensor_count: 8
resolved_input_norm_key: model.language_model.layers.3.input_layernorm.weight
resolved_q_proj_key: model.language_model.layers.3.self_attn.q_proj.weight
resolved_k_proj_key: model.language_model.layers.3.self_attn.k_proj.weight
resolved_v_proj_key: model.language_model.layers.3.self_attn.v_proj.weight
resolved_o_proj_key: model.language_model.layers.3.self_attn.o_proj.weight
attention_q_proj_shape: (8192, 2048)
attention_k_proj_shape: (512, 2048)
attention_v_proj_shape: (512, 2048)
attention_o_proj_shape: (2048, 4096)
attention_q_projection_shape: (2, 8192)
attention_k_projection_shape: (2, 512)
attention_v_projection_shape: (2, 512)
attention_q_runtime_width: 4096
attention_kv_runtime_width: 512
attention_triton_q_max_abs_diff: 0.000001
attention_triton_k_max_abs_diff: 0.000000
attention_triton_v_max_abs_diff: 0.000000
attention_triton_project_ms: 0.1825
attention_triton_project_ms_per_token: 0.0912
attention_decode_start_position: 1
attention_decode_cache_positions: 4
attention_decode_update_max_abs_diff: 0.000000
attention_decode_key_cache_max_abs_diff: 0.000001
attention_decode_value_cache_max_abs_diff: 0.000000
attention_triton_decode_ms: 0.6288
attention_triton_decode_ms_per_token: 0.3144
attention_moe_downloaded_shards: model-00003-of-00026.safetensors
attention_moe_needed_tensor_count: 8
attention_moe_resolved_norm_key: model.language_model.layers.3.post_attention_layernorm.weight
attention_moe_resolved_router_key: model.language_model.layers.3.mlp.gate.weight
attention_moe_attention_hidden_max_abs_diff: 0.000000
attention_moe_attention_update_max_abs_diff: 0.000000
attention_moe_update_max_abs_diff: 0.000000
attention_moe_layer_hidden_max_abs_diff: 0.000000
attention_moe_key_cache_max_abs_diff: 0.000001
attention_moe_value_cache_max_abs_diff: 0.000000
attention_moe_logits_max_abs_diff: 0.000002
attention_moe_topk_ids_match: True
attention_moe_topk_weights_max_abs_diff: 0.000000
attention_moe_layer_ms: 0.9632
attention_moe_layer_ms_per_token: 0.4816

Latest full 40-layer weight-plan run:

Run URL: https://modal.com/apps/suryavanshi/main/ap-MboqIBec8UuY4UC4uNdnyY

python bench/qwen36_full_weight_plan.py --repo-id Qwen/Qwen3.6-35B-A3B --revision main --show-layers --show-shards
model: Qwen3.6-35B-A3B
hidden_size: 2048
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

Latest full 40-layer streaming reference serving run:

Run URL: https://modal.com/apps/suryavanshi/main/ap-Vo5UVobsMbzEFYoJbWGuRG

python bench/qwen36_full_decode.py --repo-id Qwen/Qwen3.6-35B-A3B --revision main --device cuda --dtype bfloat16 --prompt-token-ids 0 --max-new-tokens 1 --max-positions 128
model: Qwen3.6-35B-A3B
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

The same runner also passed a full-40 prefill/logits smoke with
`generated_tokens: 0`, `final_position: 1`, and `elapsed_seconds: 1153.89`
at https://modal.com/apps/suryavanshi/main/ap-6j3WCY55FZHDSSb0FUSgG3.

Latest real streaming decode with Triton MoE sublayers:

Run URL: https://modal.com/apps/suryavanshi/main/ap-yF7mjRvl30IxAiRCH4St7a

python bench/qwen36_full_decode.py --repo-id Qwen/Qwen3.6-35B-A3B --revision main --device cuda --dtype bfloat16 --prompt-token-ids 0 --max-new-tokens 0 --max-positions 128 --moe-impl triton --max-layers 4
model: Qwen3.6-35B-A3B
layers_total: 40
layers_executed: 4
prompt_tokens: 1
generated_tokens: 0
final_position: 1
elapsed_seconds: 49.76
moe_impl: triton
fastkernels_full_triton_moe_serving_ready: True
fastkernels_full_triton_serving_ready: False

Latest real Attention -> MoE and DeltaNet -> MoE Triton layer-pattern staging smoke:

Run URL: https://modal.com/apps/suryavanshi/main/ap-dZ3H4nNpgVxFZL0SKAI0d6

python bench/qwen36_full_decode.py --repo-id Qwen/Qwen3.6-35B-A3B --revision main --device cuda --dtype bfloat16 --prompt-token-ids 0 --max-new-tokens 0 --max-positions 128 --moe-impl triton --attention-impl triton --deltanet-impl triton --max-layers 40
model: Qwen3.6-35B-A3B
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

Modal A100 pytest after the real DeltaNet -> MoE layer wrapper update:

Run URL: https://modal.com/apps/suryavanshi/main/ap-WCDGvgD4rNtocYUcdsVpCA

pytest -q
61 passed in 58.64s

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 1.3862

Attempted full-model vLLM serving baseline:

Run URL: https://modal.com/apps/suryavanshi/main/ap-NNESTpgzPcXtJfVTCVoAAj

python bench/qwen36_vllm_serving_benchmark.py --model Qwen/Qwen3.6-35B-A3B --dtype bfloat16 --tensor-parallel-size 8 --max-model-len 2048 --num-prompts 2 --input-len 32 --output-len 8
backend: vllm
vllm_version: 0.10.2
model_impl: auto
vllm_full_serving_supported: False
vllm_full_serving_error_type: ValidationError
vllm_full_serving_error: Model architectures ['Qwen3_5MoeForConditionalGeneration'] are not supported for now.

Transformers fallback probe:

Run URL: https://modal.com/apps/suryavanshi/main/ap-PS0DL0O9tVDVERz3Ws7XkF

python bench/qwen36_vllm_serving_benchmark.py --model Qwen/Qwen3.6-35B-A3B --dtype bfloat16 --tensor-parallel-size 8 --max-model-len 2048 --model-impl transformers --num-prompts 2 --input-len 32 --output-len 8
backend: vllm
vllm_version: 0.10.2
model_impl: transformers
vllm_full_serving_supported: False
vllm_full_serving_error_type: ValidationError
vllm_full_serving_error: The Transformers implementation of 'Qwen3_5MoeForConditionalGeneration' is not compatible with vLLM.

Latest vLLM package probe with `vllm==0.20.0`:

Run URL: https://modal.com/apps/suryavanshi/main/ap-oOD12jbEA9t4PZqpnNFLZ0

python -m pip install vllm==0.20.0
backend: vllm
vllm_package: vllm==0.20.0
vllm_install_supported: False
vllm_install_exit_code: 1
vllm_full_serving_supported: False
vllm_full_serving_error_type: InstallError
vllm_full_serving_error: source build requires CUDA_HOME during build metadata

The Modal vLLM baseline runner now also has a standalone H100:8 lane using the
prebuilt `vllm/vllm-openai:v0.20.0-x86_64` container image:

modal run bench/modal_qwen36_full_serving.py --gpu H100:8 --run-vllm --vllm-backend openai-image --no-run-weight-plan --no-run-pytest --max-model-len 2048 --num-prompts 2 --input-len 32 --output-len 8 --tensor-parallel-size 8

A second standalone runner follows the Modal vLLM example more directly by
building from `nvidia/cuda:13.2.0-devel-ubuntu22.04` and installing
`vllm==0.20.0` with `uv --torch-backend=cu130`:

modal run bench/modal_qwen36_vllm_serving.py --max-model-len 2048 --num-prompts 2 --input-len 32 --output-len 8 --tensor-parallel-size 8

Latest vLLM 0.20.0 full-serving attempts on April 30, 2026:

- `vllm/vllm-openai:v0.20.0-x86_64` is wired into
  `bench/modal_qwen36_full_serving.py --vllm-backend openai-image`, but this
  Modal client cannot currently bootstrap that image without adding a separate
  Python runtime, and adding that runtime hides the image's original vLLM
  Python environment.
- The standalone CUDA/uv runner imports `vllm==0.20.0` successfully on H100:8
  with `torch==2.11.0+cu130` and resolves
  `Qwen3_5MoeForConditionalGeneration`.
- Run `ap-d90hp0rXkMQGvMA8EG25oH` downloaded the 66.97 GiB checkpoint in
  347.82s, loaded all 26 shards in 8.76s, selected FlashAttention 3,
  FlashInfer GDN prefill, and FlashInfer CUTLASS MoE, then stopped during
  compile/setup before generation throughput was emitted.
- Run `ap-80rPAm8rHqDHKpBny8ievE` reused the persistent HF cache, accepted
  `--enforce-eager --gdn-prefill-backend triton`, selected Triton/FLA GDN
  prefill, loaded all 26 shards, then still stopped during post-load engine
  setup before throughput was emitted.

python bench/qwen36_real_weight_moe_smoke.py --device cuda --dtype bfloat16 --tokens 2 --warmup 1 --iters 3 --layers 0 1 --run-layer-harness
model: Qwen3.6-35B-A3B
repo_id: Qwen/Qwen3.6-35B-A3B
revision: main
tokens: 2
layer: 0
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
hf_total_size_bytes: 71903645408.0
downloaded_shards: model-00001-of-00026.safetensors,model-00002-of-00026.safetensors
downloaded_shard_count: 2
needed_tensor_count: 8
resolved_norm_key: model.language_model.layers.0.post_attention_layernorm.weight
resolved_router_key: model.language_model.layers.0.mlp.gate.weight
resolved_expert_gate_up_key: model.language_model.layers.0.mlp.experts.gate_up_proj
resolved_expert_down_key: model.language_model.layers.0.mlp.experts.down_proj
shared_expert_gate_present: True
shared_expert_gate_applied: True
shared_expert_gate_ungated_gap_max_abs: 0.000115
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000000
topk_ids_match: True
topk_weights_max_abs_diff: 0.000000
real_weight_batched_moe_ms: 0.3577
real_weight_batched_moe_ms_per_token: 0.1788
real_weight_layer_harness: True
layer_output_max_abs_diff: 0.000000
layer_update_max_abs_diff: 0.000000
layer_logits_max_abs_diff: 0.000000
layer_topk_ids_match: True
layer_topk_weights_max_abs_diff: 0.000000
real_weight_batched_moe_layer_ms: 0.4036
real_weight_batched_moe_layer_ms_per_token: 0.2018

layer: 1
downloaded_shards: model-00001-of-00026.safetensors,model-00002-of-00026.safetensors
downloaded_shard_count: 2
needed_tensor_count: 8
resolved_norm_key: model.language_model.layers.1.post_attention_layernorm.weight
resolved_router_key: model.language_model.layers.1.mlp.gate.weight
resolved_expert_gate_up_key: model.language_model.layers.1.mlp.experts.gate_up_proj
resolved_expert_down_key: model.language_model.layers.1.mlp.experts.down_proj
shared_expert_gate_present: True
shared_expert_gate_applied: True
shared_expert_gate_ungated_gap_max_abs: 0.000078
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000000
topk_ids_match: True
topk_weights_max_abs_diff: 0.000000
real_weight_batched_moe_ms: 0.2863
real_weight_batched_moe_ms_per_token: 0.1432
real_weight_layer_harness: True
layer_output_max_abs_diff: 0.000000
layer_update_max_abs_diff: 0.000000
layer_logits_max_abs_diff: 0.000000
layer_topk_ids_match: True
layer_topk_weights_max_abs_diff: 0.000000
real_weight_batched_moe_layer_ms: 0.3151
real_weight_batched_moe_layer_ms_per_token: 0.1575

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.9423

pytest -q
49 passed in 39.79s

Previous reference points:

Run URL: https://modal.com/apps/suryavanshi/main/ap-EKlDfjmGtodM0u5WMul3wB

python bench/qwen36_real_weight_block_smoke.py --device cuda --dtype bfloat16 --tokens 2
attention_q_projection_shape: (2, 8192)
attention_k_projection_shape: (2, 512)
attention_v_projection_shape: (2, 512)
pytest -q
48 passed in 37.15s

Run URL: https://modal.com/apps/suryavanshi/main/ap-J5tOEIFkK9FPtKB2qcqwUL

python bench/qwen36_real_weight_moe_smoke.py --device cuda --dtype bfloat16 --tokens 2 --warmup 1 --iters 3 --layers 0 1 --run-layer-harness
real_weight_batched_moe_layer_ms_per_token: 0.1753 for layer 0
real_weight_batched_moe_layer_ms_per_token: 0.1325 for layer 1
pytest -q
46 passed in 23.82s

Run URL: https://modal.com/apps/suryavanshi/main/ap-MqZekGjsWbJ0Gdrc3v3pxR

python bench/qwen36_real_weight_moe_smoke.py --device cuda --dtype bfloat16 --tokens 2 --warmup 1 --iters 3 --layers 0 1
real_weight_batched_moe_ms_per_token: 0.1606 for layer 0
real_weight_batched_moe_ms_per_token: 0.1307 for layer 1
pytest -q
45 passed in 23.32s

Run URL: https://modal.com/apps/suryavanshi/main/ap-7HqJkO0oSEPPVhdbdymSaD

python bench/qwen36_real_weight_moe_smoke.py --device cuda --dtype bfloat16 --tokens 1 --warmup 1 --iters 3
shared_expert_gate_present: True
shared_expert_gate_applied: True
shared_expert_gate_ungated_gap_max_abs: 0.000070
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000000
topk_ids_match: True
topk_weights_max_abs_diff: 0.000000
real_weight_batched_moe_ms: 0.3290
real_weight_batched_moe_ms_per_token: 0.3290

pytest -q
45 passed in 29.24s

Run URL: https://modal.com/apps/suryavanshi/main/ap-DN8cfhl0r892Glgwh44LhI

python bench/qwen36_real_weight_moe_smoke.py --device cuda --dtype bfloat16 --tokens 1 --warmup 1 --iters 3
shared_expert_gate_present: True
shared_expert_gate_contract_gap_max_abs: 0.000070
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000000
topk_ids_match: True
topk_weights_max_abs_diff: 0.000000
real_weight_batched_moe_ms: 0.3328
real_weight_batched_moe_ms_per_token: 0.3328

pytest -q
44 passed in 39.98s

Run URL: https://modal.com/apps/suryavanshi/main/ap-ErpI7lmb9nGtwgN08ILDNO

python bench/qwen36_safetensor_moe_smoke.py --device cuda --dtype bfloat16 --tokens 2 --warmup 2 --iters 5
model: Qwen3.6-35B-A3B
tokens: 2
layer: 0
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
resolved_norm_key: model.layers.0.post_attention_layernorm.weight
resolved_router_key: model.layers.0.mlp.gate.weight
safetensor_indexed: True
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000031
topk_ids_match: True
topk_weights_max_abs_diff: 0.000000
safetensor_batched_real_moe_ms: 0.2159
safetensor_batched_real_moe_ms_per_token: 0.1079

python bench/qwen36_batched_real_moe_microbench.py --device cuda --dtype bfloat16 --tokens 4 --warmup 3 --iters 10
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000031
topk_ids_match: True
topk_weights_max_abs_diff: 0.000002
batched_real_moe_ms: 0.2666
batched_real_moe_ms_per_token: 0.0667

pytest -q
43 passed in 27.62s

Run URL: https://modal.com/apps/suryavanshi/main/ap-VkUHN3489Dh9ZPWwzmMheW

python bench/qwen36_safetensor_moe_smoke.py --device cuda --dtype bfloat16 --tokens 2 --warmup 2 --iters 5
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000031
topk_ids_match: True
topk_weights_max_abs_diff: 0.000000
safetensor_batched_real_moe_ms: 0.2506
safetensor_batched_real_moe_ms_per_token: 0.1253

python bench/qwen36_batched_real_moe_microbench.py --device cuda --dtype bfloat16 --tokens 4 --warmup 3 --iters 10
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000031
topk_ids_match: True
topk_weights_max_abs_diff: 0.000002
batched_real_moe_ms: 0.2779
batched_real_moe_ms_per_token: 0.0695

pytest -q
42 passed in 41.03s

Run URL: https://modal.com/apps/suryavanshi/main/ap-SCRpc6fMuwfdtPRTxelmV4

python bench/qwen36_batched_real_moe_microbench.py --device cuda --dtype bfloat16 --tokens 4 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
tokens: 4
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000031
topk_ids_match: True
topk_weights_max_abs_diff: 0.000002
batched_real_moe_ms: 0.2701
batched_real_moe_ms_per_token: 0.0675

python bench/qwen36_batched_routed_moe_microbench.py --device cuda --dtype bfloat16 --tokens 4 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
tokens: 4
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
max_abs_diff: 0.000000
batched_routed_shared_moe_ms: 0.2505
batched_routed_shared_moe_ms_per_token: 0.0626

python bench/qwen36_real_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000001
real_moe_ms: 0.2286

python bench/qwen36_routed_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
max_abs_diff: 0.000000
routed_shared_moe_ms: 0.1164

python bench/qwen36_router_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000002
router_ms: 0.0830

pytest -q
40 passed in 34.91s

Run URL: https://modal.com/apps/suryavanshi/main/ap-775CKcZ1gFDt9lS9Sjh1PO

python bench/qwen36_batched_routed_moe_microbench.py --device cuda --dtype bfloat16 --tokens 4 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
tokens: 4
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
max_abs_diff: 0.000000
batched_routed_shared_moe_ms: 0.2480
batched_routed_shared_moe_ms_per_token: 0.0620

python bench/qwen36_real_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000001
real_moe_ms: 0.1742

python bench/qwen36_routed_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
max_abs_diff: 0.000000
routed_shared_moe_ms: 0.0938

python bench/qwen36_expert_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
intermediate: 512
max_abs_diff: 0.000000
expert_mlp_ms: 0.0619

python bench/qwen36_router_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000002
router_ms: 0.0663

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.7369

pytest -q
38 passed in 25.86s

Run URL: https://modal.com/apps/suryavanshi/main/ap-QoAinlMxh04ImGPRmpPPBO

python bench/qwen36_vllm_moe_compare.py --device cuda --dtype bfloat16 --warmup 5 --iters 20 --require-vllm
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
fastkernels_routed_max_abs_diff: 0.000000
fastkernels_routed_moe_ms: 0.0859
vllm_available: True
vllm_version: 0.10.2
vllm_routed_max_abs_diff: 0.000000
vllm_routed_moe_ms: 0.4841
fastkernels_to_vllm_ms_ratio: 0.1775

python bench/qwen36_real_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000001
real_moe_ms: 0.1751

python bench/qwen36_routed_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
max_abs_diff: 0.000000
routed_shared_moe_ms: 0.0932

python bench/qwen36_expert_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
intermediate: 512
max_abs_diff: 0.000000
expert_mlp_ms: 0.0616

python bench/qwen36_router_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000002
router_ms: 0.0659

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.6580

pytest -q
37 passed in 20.03s

The vLLM comparison is a routed-expert-only operator comparison over random
real-shaped tensors, not an end-to-end vLLM serving tokens/sec result. vLLM
0.10.2 warned that it used the default MoE config for this exact
`E=256,N=512,A100` shape.

Run URL: https://modal.com/apps/suryavanshi/main/ap-ixTieTqDIcWv03wRIKdd54

python bench/qwen36_real_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
output_max_abs_diff: 0.000000
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000001
real_moe_ms: 0.1778

python bench/qwen36_routed_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
max_abs_diff: 0.000000
routed_shared_moe_ms: 0.0924

python bench/qwen36_expert_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
intermediate: 512
max_abs_diff: 0.000000
expert_mlp_ms: 0.0596

python bench/qwen36_router_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000002
router_ms: 0.0661

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.7128

pytest -q
36 passed in 22.87s

python bench/qwen36_routed_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
max_abs_diff: 0.000000
routed_shared_moe_ms: 0.1211

python bench/qwen36_expert_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
intermediate: 512
max_abs_diff: 0.000000
expert_mlp_ms: 0.0746

python bench/qwen36_router_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000002
router_ms: 0.0863

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.9648

pytest -q
35 passed in 33.57s

python bench/qwen36_routed_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
intermediate: 512
max_abs_diff: 0.000000
routed_shared_moe_ms: 1.2023

python bench/qwen36_expert_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
intermediate: 512
max_abs_diff: 0.000000
expert_mlp_ms: 0.0616

python bench/qwen36_router_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000002
router_ms: 0.0688

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.6552

pytest -q
35 passed in 16.19s

python bench/qwen36_expert_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
intermediate: 512
max_abs_diff: 0.000000
expert_mlp_ms: 0.0627

python bench/qwen36_router_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000002
router_ms: 0.0690

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.6767

pytest -q
34 passed in 16.97s

python bench/qwen36_router_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
model: Qwen3.6-35B-A3B
hidden_size: 2048
experts: 256
top_k: 8
logits_max_abs_diff: 0.000023
topk_ids_match: True
topk_weights_max_abs_diff: 0.000002
router_ms: 0.0672

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.7329

pytest -q
32 passed in 19.34s

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.9347

python bench/qwen36_layer_microbench.py --layer-kind deltanet_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
hidden_max_abs_diff: 0.000977
state_max_abs_diff: 0.007812
layer_ms: 0.1201

python bench/qwen36_layer_microbench.py --layer-kind attention_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
hidden_max_abs_diff: 0.000977
state_max_abs_diff: 0.000549
value_cache_max_abs_diff: 0.000488
layer_ms: 0.2437

python bench/qwen36_moe_topk_microbench.py --device cuda --dtype bfloat16 --experts 4 --top-k 2 --warmup 10 --iters 100
max_abs_diff: 0.000000
moe_ms: 0.0563

python bench/qwen36_moe_topk_microbench.py --device cuda --dtype bfloat16 --experts 256 --top-k 8 --warmup 3 --iters 10
max_abs_diff: 0.000000
moe_ms: 0.0750

pytest -q
29 passed in 25.62s

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 1.1699

python bench/qwen36_layer_microbench.py --layer-kind deltanet_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
hidden_max_abs_diff: 0.000977
state_max_abs_diff: 0.007812
layer_ms: 0.1247

python bench/qwen36_layer_microbench.py --layer-kind attention_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
hidden_max_abs_diff: 0.000092
state_max_abs_diff: 0.000549
value_cache_max_abs_diff: 0.000488
layer_ms: 0.2957

python bench/qwen36_moe_topk_microbench.py --device cuda --dtype bfloat16 --experts 4 --top-k 2 --warmup 10 --iters 100
max_abs_diff: 0.000000
moe_ms: 0.1094

python bench/qwen36_moe_topk_microbench.py --device cuda --dtype bfloat16 --experts 256 --top-k 8 --warmup 3 --iters 10
max_abs_diff: 0.000000
moe_ms: 0.0763

pytest -q
28 passed in 29.51s

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.7675

python bench/qwen36_layer_microbench.py --layer-kind deltanet_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
hidden_max_abs_diff: 0.000977
state_max_abs_diff: 0.007812
layer_ms: 0.0963

python bench/qwen36_layer_microbench.py --layer-kind attention_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
hidden_max_abs_diff: 0.000488
state_max_abs_diff: 0.000549
value_cache_max_abs_diff: 0.000488
layer_ms: 0.2308

pytest -q
27 passed in 16.62s

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
reference_max_abs_diff: 0.000488
decode_ms_per_step: 0.6957

python bench/qwen36_layer_microbench.py --layer-kind deltanet_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
hidden_max_abs_diff: 0.000977
state_max_abs_diff: 0.007812
layer_ms: 0.0909

python bench/qwen36_layer_microbench.py --layer-kind attention_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
hidden_max_abs_diff: 0.000488
state_max_abs_diff: 0.000549
value_cache_max_abs_diff: 0.000488
layer_ms: 0.2233

python bench/moe_microbench.py --device cuda --dtype bfloat16 --rows 8192 --tokens 4 --warmup 10 --iters 100 --skip-routed-moe
torch_swiglu_ms: 0.0559
triton_swiglu_ms: 0.0442
triton_expert_histogram: correctness ok
```

## Priority TODOs

1. Grow the real Qwen3.6 path outward from routing:
   - Current real-shape Triton coverage: RMSNorm + router projection + softmax
     top-8 for one hidden-2048 token over 256 experts, plus one hidden-2048 /
     intermediate-512 expert MLP, single-token and batched routed-expert-only
     paths for vLLM comparison, single-token and batched two-launch top-8
     routed/shared expert accumulation paths, and a full real-shape MoE wrapper
     that feeds router output into routed/shared expert execution.
   - Current weight-layout coverage: HF-style split `gate_proj`/`up_proj`/
     `down_proj` expert names and real grouped `experts.gate_up_proj` /
     `experts.down_proj` names, real `linear_attn.*` names, and real
     `self_attn.*` names are resolved from single-file or indexed sharded
     safetensors and packed into kernel-facing contracts.
   - Current real-weight coverage: Modal downloads the real HF index plus only
     the shard files needed by requested layers, then runs configurable
     multi-token MoE parity and a residual `hidden + MoE` layer-boundary
     harness. The latest run covers layers 0 and 1 with 2 token rows against
     the PyTorch reference contract, including `shared_expert_gate.weight`.
     It also loads layer-0 `linear_attn.*` and layer-3 `self_attn.*` weights
     and runs PyTorch projection shape smokes for 2 token rows. The real
     full-attention path now has Triton parity against PyTorch on layer-3
     weights for RMSNorm+Q/K/V projection, Q/K norm, RoPE/cache update, causal
     attention accumulation, and output projection. The latest block smoke
     composes that attention update into the real layer-3 MoE residual boundary
     with exact reported parity for attention-hidden, MoE update, layer-hidden,
     caches, router logits, and top-k outputs. The latest vLLM lane compares
     selected real layer-0 routed-expert weights at the routed-experts-only
     operator boundary.
   - Current full-model planning coverage: Modal resolves the complete
     40-layer Qwen3.6 real-weight plan, including root tensors, all 30
     DeltaNet+MoE layers, all 10 Attention+MoE layers, 693 required tensors,
     and all 26 safetensors shards.
   - Current full-model serving coverage: Modal H100:8 runs a streaming
     PyTorch reference lane over real Qwen3.6 weights for all 40 layers and
     one-token generation. This is not yet the fast Triton serving path.
   - Next real-shape target: replace the streaming reference internals with
     verified Triton/CUDA DeltaNet, gated attention, and MoE layer kernels.
   - Keep vLLM comparison scoped to matched operator boundaries until full
     serving configs are available.
   - Keep comparing against PyTorch reference tensors before widening beyond
     the downloaded multi-layer real-weight shard smoke.

2. Collapse staged synthetic layer-boundary APIs into fewer launches:
   - Current Triton `DeltaNet -> MoE` uses a fused one-launch DeltaNet plus MoE.
   - Current Triton `Attention -> MoE` still uses staged attention plus MoE, but
     the benchmark path now updates KV cache in place instead of cloning it.
   - The attempted attention projection+RoPE/cache fusion is available as an
     experimental switch but was slower in Modal A100 timing, so it is not the
     default path.

3. Generalize fused synthetic MoE:
   - Current Triton MoE dispatches top-2 to a dedicated fast kernel and uses a
     generalized static top-k path for top-k 1, 4, and 8 coverage.
   - Modal now verifies a tiny-shape 256-expert/top-8 path; this proves routing
     scale coverage, not full hidden-2048/intermediate-512 model coverage.
   - Next performance task: specialize top-8 routing/MLP math without losing
     the new parity coverage.
   - Then scale tiling toward full Qwen3.6 hidden/intermediate dimensions.

4. Collapse staged synthetic attention into a fused layer path:
   - Current Triton attention implementation matches the reference within dtype
     tolerances but uses staged launches.
   - Keep KV-cache append and RoPE parity tests stable before fusion.

5. Decide CuTe/CUTLASS boundary:
   - Keep Triton while dataflow changes quickly.
   - Move the stable hot path to CuTe/CUTLASS if Triton cannot express the
     persistent decode schedule or if register/shared-memory control is limiting.

6. Full model path:
   - Real Hugging Face `config.json` and safetensors index parsing are in
     place for full 40-layer weight planning.
   - Config alias tests and full-weight-plan unit tests are in place.
   - Current full-reference serving lane: `bench/modal_qwen36_full_serving.py`
     on `H100:8` with `--run-full-decode`.
   - Current hybrid serving lane: the same runner can use `--moe-impl triton`
     to replace the streaming MoE sublayer with the real-weight Triton MoE
     wrapper; this is verified through the first four real layers on A100.
   - Current vLLM full-serving baseline lane: `bench/modal_qwen36_full_serving.py`
     on `H100:8` with tensor parallel size 8. The default comparison path uses
     `--vllm-backend openai-image` with `vllm/vllm-openai:v0.20.0-x86_64`.
     The older pip lane remains available for package debugging; vLLM 0.10.2
     rejects the architecture before serving starts, and pip-installing
     `vllm==0.20.0` into the slim fastkernels image fails because its source
     build requires `CUDA_HOME`.
   - Fast Triton full serving remains blocked on replacing the streaming
     reference path with verified kernels and measuring end-to-end throughput.

## Open Questions

- Which deployment target matters first: fastest single-token decode, prompt
  prefill, or mixed serving throughput?
- Should the first real-weight integration target Transformers, vLLM, SGLang,
  or a custom weight converter?
- Should the next Lambda instance be kept at A100 for cost-effective iteration,
  or upgraded only when full-model memory demands it?

## Do Not Do Yet

- Do not start with a 40-layer CuTe megakernel before the synthetic per-block
  references and tests are stable.
- Do not load the full 35B model on the single A100-40GB instance and treat OOM
  as a kernel issue.
- Do not remove the current Triton MoE tests or PyTorch references.
- Do not compare against vLLM tokens/sec until real weights, identical serving
  settings, and full-model parity are in place.

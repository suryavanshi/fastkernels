# Codex Handoff

## Goal

Build and optimize inference kernels for Qwen MoE models, with the current focus
moving from MoE building blocks toward a Qwen3.6-35B-A3B decode megakernel.

## Current Architecture

- `fastkernels.models` contains shape specs for Qwen3.5 MoE and Qwen3.6 A3B.
- `fastkernels.reference` contains PyTorch correctness oracles.
- `fastkernels.kernels.triton` contains optional Triton MoE building blocks.
- `bench/` contains CLI smoke tests and benchmarks.
- `tests/` contains CPU tests plus optional CUDA/Triton tests.

Current Qwen3.6 work started as synthetic and structurally faithful, then grew
into selected real-weight lanes and a streaming PyTorch full-reference runner.
It models the Qwen3.6 layer pattern, decode state, DeltaNet state, attention KV
cache, routed MoE, and shared expert path, and now loads real
Qwen3.6-35B-A3B weights for a slow end-to-end reference generation path.
The real-shape MoE work now covers one hidden-2048 decode token with RMSNorm,
router projection, top-8 routing, top-8 routed expert execution, and shared
expert execution over random real-shaped tensors. It also has an optional
routed-expert-only comparison against vLLM's `fused_moe` op and a batched
routed-expert-only comparison against vLLM's `fused_moe` op, including a
batched selected-real-weight comparison lane, and a batched routed/shared
expert path plus batched full MoE wrapper for multiple decode tokens. The
safetensors layout lane now resolves HF-style Qwen3.6 MoE keys
from single-file or indexed sharded layouts and packs them into the kernel
contract. The latest Modal lane downloads the real Qwen/Qwen3.6-35B-A3B index
and the selected shard files needed for requested real MoE layers, then checks
both MoE-update parity and the residual `hidden + MoE` layer boundary. It also
loads real layer-0 `linear_attn.*` weights and real layer-3 `self_attn.*`
weights, validating their packed projection contracts. The linear-attention
smoke now includes a real Triton DeltaNet staging boundary for
RMSNorm+QKV/Z/A/B projections, causal depthwise convolution/state staging,
recurrent-state update, gated RMSNorm, and output projection, and the
full-attention smoke runs a real Triton
decode boundary against PyTorch on layer-3 weights for two token rows:
RMSNorm+Q/K/V projection, Q/K norm, RoPE/cache update, causal attention
accumulation, and output projection. The real layer-3 smoke also
composes that attention update into the real MoE residual layer boundary and
checks attention-hidden, MoE update, final layer-hidden, cache, router-logit,
and top-k parity. The vLLM comparison is still scoped to the routed-experts-only
operator boundary, not end-to-end serving throughput.
The full-model planning lane now resolves all real Qwen3.6 root tensors, all
40 layer-local tensor groups, and all 26 safetensors shards. The streaming
PyTorch reference lane now serves the full 40-layer real-weight model on Modal
H100:8 for one-token generation, with selectable Triton Attention, DeltaNet
projection+conv+recurrent/output, Attention -> MoE layer staging, DeltaNet ->
MoE layer staging, and MoE staging. vLLM 0.10.2 currently fails before serving
starts for this model in both native registry and Transformers fallback modes.
The requested `vllm==0.20.0` comparison now has a standalone Modal H100:8 lane
using the prebuilt `vllm/vllm-openai:v0.20.0-x86_64` image. The older pip
comparison still fails earlier during package installation because the PyPI
source build requires `CUDA_HOME` in the slim fastkernels Modal runtime.
There is also a standalone Modal-docs-style CUDA/uv runner,
`bench/modal_qwen36_vllm_serving.py`, which imports vLLM 0.20.0 with
`torch==2.11.0+cu130` and reaches Qwen3.6 model load on H100:8. It has not yet
emitted throughput: the default run stopped during compile/setup and the
`--enforce-eager --gdn-prefill-backend triton` run stopped during post-load
engine setup.
The full Triton/megakernel runtime still does not serve the full model.

## Files Changed So Far

- `README.md`: install notes, Colab link, Lambda runbook link, current kernel scope.
- `notebooks/colab_kernel_test.ipynb`: Colab GPU test notebook.
- `docs/LAMBDA.md`: Lambda launch, SSH, install, test, benchmark, terminate runbook.
- `docs/QWEN36_MEGAKERNEL_PLAN.md`: staged Qwen3.6 megakernel plan.
- `src/fastkernels/__init__.py`: exports Qwen3.6 spec.
- `src/fastkernels/models/__init__.py`: exports Qwen3.6 helpers.
- `src/fastkernels/models/qwen36.py`: Qwen3.6-35B-A3B and synthetic specs.
- `src/fastkernels/models/qwen36_full.py`: full 40-layer real-weight key-plan
  resolver for roots, DeltaNet layers, attention layers, MoE layers, and shard
  coverage.
- `src/fastkernels/models/qwen36_weights.py`: Qwen3.6 MoE, linear-attention,
  and full-attention key resolvers, state-dict packers, and single-file/indexed
  sharded safetensors loaders for selected layers, including the real grouped
  expert tensor names and real `linear_attn.*` / `self_attn.*` tensor names.
- `src/fastkernels/reference/__init__.py`: exports Qwen3.6 decode reference helpers.
- `src/fastkernels/reference/qwen36_decode.py`: synthetic Qwen3.6 decode reference.
- `src/fastkernels/reference/qwen36_real.py`: streaming real-weight PyTorch
  reference blocks for Qwen3.6 RMSNorm, Gated DeltaNet, gated full attention,
  MoE, and layer composition.
- `src/fastkernels/kernels/triton/qwen36_router.py`: real-shape Qwen3.6 MoE router decode kernels for single-token and batched hidden-2048, 256 experts, and top-8 routing.
- `src/fastkernels/kernels/triton/qwen36_expert.py`: real-shape Qwen3.6 single-expert MLP kernel for hidden-2048/intermediate-512, single-token and batched routed-expert-only accumulation for vLLM comparison, single-token and batched two-launch top-8 routed/shared expert accumulation, and single-token plus batched full real-shape MoE wrappers that connect router output to routed/shared expert execution.
- `src/fastkernels/kernels/triton/qwen36_deltanet.py`: staged synthetic Qwen3.6 DeltaNet decode block plus a real-weight batched DeltaNet staging boundary for input RMSNorm, QKV/Z/A/B projections, causal depthwise convolution, conv-state update, recurrent-state update, gated RMSNorm, and output projection.
- `src/fastkernels/kernels/triton/qwen36_attention.py`: staged synthetic Qwen3.6 attention decode block with optional in-place KV-cache update, plus real-weight batched full-attention decode staging for RMSNorm+Q/K/V projection, Q/K norm, RoPE/cache update, causal attention accumulation, and output projection.
- `src/fastkernels/kernels/triton/qwen36_layer.py`: compositional layer-boundary wrappers, including the real-shape batched MoE residual layer harness and staged real-weight Attention -> MoE layer boundary.
- `bench/qwen36_layer_microbench.py`: synthetic layer-boundary timing entrypoint.
- `bench/qwen36_moe_topk_microbench.py`: synthetic MoE expert-count/top-k timing entrypoint.
- `bench/qwen36_router_microbench.py`: real-shape Qwen3.6 MoE router timing entrypoint.
- `bench/qwen36_expert_microbench.py`: real-shape Qwen3.6 single-expert MLP timing entrypoint.
- `bench/qwen36_routed_moe_microbench.py`: real-shape Qwen3.6 top-8 routed plus shared expert timing entrypoint.
- `bench/qwen36_batched_routed_moe_microbench.py`: batched real-shape Qwen3.6 routed/shared expert timing entrypoint.
- `bench/qwen36_real_moe_microbench.py`: full real-shape Qwen3.6 MoE wrapper timing entrypoint for router plus routed/shared experts.
- `bench/qwen36_batched_real_moe_microbench.py`: batched full real-shape Qwen3.6 MoE wrapper timing entrypoint.
- `bench/qwen36_safetensor_moe_smoke.py`: generated real-shaped indexed
  sharded safetensors loader smoke test feeding the batched full MoE wrapper.
- `bench/qwen36_real_weight_moe_smoke.py`: downloads the real Qwen3.6 HF index
  and selected layer shards, then runs configurable multi-token real-weight MoE
  parity with `shared_expert_gate.weight` applied and optional token-level
  residual layer-boundary parity.
- `bench/qwen36_real_weight_block_smoke.py`: downloads only the real shards
  needed for one linear-attention layer and one full-attention layer, validates
  tensor shapes, runs PyTorch projection shape smokes, checks the real Triton
  full-attention decode boundary against PyTorch, and checks the composed real
  Attention -> MoE residual layer boundary on real layer weights.
- `bench/qwen36_real_weight_key_probe.py`: Modal/HF helper for listing real
  layer-local key names and optional shapes from the Qwen3.6 index.
- `bench/qwen36_vllm_moe_compare.py`: optional vLLM routed-MoE operator comparison over real-shaped random tensors or selected real Qwen3.6 MoE layer weights.
- `bench/qwen36_full_weight_plan.py`: resolves the complete real Qwen3.6
  40-layer weight plan from HF config/index metadata and reports required
  tensors and shards.
- `bench/qwen36_full_decode.py`: streams real Qwen3.6 weights layer-by-layer
  and runs full 40-layer PyTorch reference prefill/decode on Modal H100:8.
- `bench/qwen36_vllm_serving_benchmark.py`: full-model vLLM serving baseline
  benchmark entrypoint for prompt/generation throughput.
- `bench/modal_qwen36_full_serving.py`: Modal A100 full-weight-plan lane and
  H100:8 streaming reference / vLLM full-serving lanes.
- `src/fastkernels/kernels/triton/moe.py`: fixed lazy Triton globals, bf16 exp upcast, cached JIT kernels.
- `bench/qwen36_decode_reference.py`: synthetic decode benchmark.
- `tests/test_qwen36_spec.py`: Qwen3.6 shape/spec tests.
- `tests/test_qwen36_decode.py`: synthetic decode state/determinism tests.
- `tests/test_triton_moe.py`: optional CUDA/Triton correctness tests.

## Important Design Decisions

- Do not claim a full Qwen3.6 megakernel exists yet.
- Use Triton first for rapid correctness-first kernel work.
- Use CuTe/CUTLASS later for maximum-performance persistent decode kernels.
- Keep the current single A100 Lambda instance for synthetic kernels and tests.
- Full BF16 Qwen3.6-35B-A3B validation likely needs tensor parallelism or
  quantization; Hugging Face examples use 8-way tensor parallel serving.
- Keep secrets out of docs and logs.

## Known Bugs / Failing Tests

- Local machine lacks `pytest` and `torch`, so local `pytest -q` and PyTorch
  benchmarks do not run without installing dependencies.
- Lambda A100 tests pass.
- Modal A100 tests pass; latest pytest run was 51 tests passed with grouped real
  HF layout packing, real linear-attention/full-attention block loading, real
  full-attention decode parity, plus a gated real-weight 2-token / 2-layer MoE
  and residual layer-boundary smoke, and a real-weight routed-expert-only vLLM
  comparison. The latest no-pytest Modal block smoke also validated the staged
  real layer-3 Attention -> MoE boundary with exact reported parity.
- Modal A100 full-weight-plan smoke passes; latest run resolved 693 tensors,
  40 layers, 30 DeltaNet+MoE layers, 10 Attention+MoE layers, and all 26
  Qwen3.6 safetensors shards.
- Modal H100:8 streaming full-reference serving passes for real Qwen3.6
  weights: latest one-token generation run produced token id `198`, final
  position `2`, and elapsed in 454.16s.
- Modal A100 streaming full-decode with Triton MoE sublayers passes through the
  first four real layers, including the first Attention+MoE layer.
- Modal A100 streaming full-decode with Triton Attention plus Triton MoE
  sublayers now passes through the first four real layers, including the first
  real Attention+MoE layer. Latest run: `ap-zN18sfejGyRPQn8LOI4pqv`, 34.57s,
  `attention_impl: triton`, `moe_impl: triton`.
- The full real-weight runner now accepts `--deltanet-impl triton`, which moves
  DeltaNet input RMSNorm, QKV/Z/A/B projection, causal depthwise convolution,
  conv-state staging, recurrent-state update, gated RMSNorm, and output
  projection to Triton, and now composes both real layer kinds through
  `DeltaNet -> MoE` and `Attention -> MoE` layer wrappers. It also recognizes
  repeated full Qwen3.6 four-layer pattern chunks. Latest full-40 prefill/logits
  run: `ap-dZ3H4nNpgVxFZL0SKAI0d6`, 491.63s, `attention_impl: triton`,
  `deltanet_impl: triton`, `moe_impl: triton`,
  `fastkernels_full_triton_four_layer_pattern_chunks: 10`,
  `elapsed_ms_per_layer_position: 12290.7386`, and
  `layer_positions_per_second: 0.0814`. This is still staged streaming
  execution, not the final fused persistent megakernel.
- Modal A100 pytest after the real DeltaNet -> MoE layer wrapper update
  passed: `ap-WCDGvgD4rNtocYUcdsVpCA`, `61 passed in 58.64s`; synthetic
  all-Triton decode reported `reference_max_abs_diff: 0.000488` and
  `decode_ms_per_step: 1.3862`.
- Modal A100 real-weight block smoke after the same update passed:
  `ap-yQ6FeDIoMl3WJal1yiVjAN`, with `linear_moe_layer_hidden_max_abs_diff:
  0.000000`, `linear_moe_logits_max_abs_diff: 0.000005`, top-k ids matching,
  and `linear_moe_layer_ms_per_token: 0.4462`.
- Modal H100:8 `vllm==0.20.0` comparison now has a prebuilt OpenAI image lane;
  the older pip lane does not reach model load because source-build metadata
  fails with `CUDA_HOME is not set`.
- Modal H100:8 standalone CUDA/uv vLLM 0.20.0 runner reaches architecture
  resolution, TP=8 NCCL setup, 26-shard weight load, and vLLM backend selection
  for Qwen3.6, but no generation throughput has been emitted yet.
- No known failing tests on Lambda or Modal.
- No full Triton/megakernel Qwen3.6 serving path exists yet. The current
  end-to-end path is the streaming real-weight runner with selectable Triton
  Attention -> MoE and DeltaNet -> MoE layer staging.

## Commands

Setup:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

GPU setup:

```sh
python -m pip install "torch==2.7.1" "triton==3.3.1" numpy
python -m pip install -e ".[triton,dev]"
```

Tests:

```sh
pytest -q
```

Benchmarks:

```sh
PYTHONPATH=src python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2
PYTHONPATH=src python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
PYTHONPATH=src python bench/qwen36_layer_microbench.py --layer-kind deltanet_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
PYTHONPATH=src python bench/qwen36_layer_microbench.py --layer-kind attention_moe --impl triton --device cuda --dtype bfloat16 --warmup 10 --iters 100
PYTHONPATH=src python bench/qwen36_moe_topk_microbench.py --device cuda --dtype bfloat16 --experts 256 --top-k 8 --warmup 3 --iters 10
PYTHONPATH=src python bench/qwen36_router_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
PYTHONPATH=src python bench/qwen36_expert_microbench.py --device cuda --dtype bfloat16 --warmup 10 --iters 100
PYTHONPATH=src python bench/qwen36_routed_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
PYTHONPATH=src python bench/qwen36_batched_routed_moe_microbench.py --device cuda --dtype bfloat16 --tokens 4 --warmup 3 --iters 10
PYTHONPATH=src python bench/qwen36_real_moe_microbench.py --device cuda --dtype bfloat16 --warmup 3 --iters 10
PYTHONPATH=src python bench/qwen36_batched_real_moe_microbench.py --device cuda --dtype bfloat16 --tokens 4 --warmup 3 --iters 10
PYTHONPATH=src python bench/qwen36_safetensor_moe_smoke.py --device cuda --dtype bfloat16 --tokens 2 --warmup 2 --iters 5
PYTHONPATH=src python bench/qwen36_real_weight_block_smoke.py --device cuda --dtype bfloat16 --tokens 2
PYTHONPATH=src python bench/qwen36_real_weight_moe_smoke.py --device cuda --dtype bfloat16 --tokens 2 --warmup 1 --iters 3 --layers 0 1 --run-layer-harness
PYTHONPATH=src python bench/qwen36_vllm_moe_compare.py --device cuda --dtype bfloat16 --warmup 5 --iters 20
PYTHONPATH=src python bench/qwen36_vllm_moe_compare.py --device cuda --dtype bfloat16 --tokens 2 --warmup 5 --iters 20 --real-weights --require-vllm
PYTHONPATH=src python bench/qwen36_full_weight_plan.py --show-layers --show-shards
PYTHONPATH=src python bench/qwen36_full_decode.py --device cuda --dtype bfloat16 --prompt-token-ids 0 --max-new-tokens 1
PYTHONPATH=src python bench/qwen36_vllm_serving_benchmark.py --tensor-parallel-size 8 --max-model-len 2048 --num-prompts 2 --input-len 32 --output-len 8
PYTHONPATH=src python bench/moe_microbench.py --device cuda --dtype bfloat16 --rows 8192 --tokens 4 --warmup 10 --iters 100 --skip-routed-moe
```

Lint:

```text
No lint command configured.
```

Dev server:

```text
No dev server configured.
```

## Lambda Notes

- API key is expected in `/Users/kb/Documents/proj/git_projs/.env` as
  `LAMBDA_API_KEY`.
- Do not include the API key value in chat, docs, commits, or logs.
- Existing tested instance from this session:
  - Type: `gpu_1x_a100_sxm4`
  - Region: `us-east-1`
  - Public IP can be retrieved from the Lambda console or API.
  - SSH key path: `/Users/kb/.ssh/id_ed25519`
- See `docs/LAMBDA.md` for launch and terminate commands.

## Current TODOs

1. Replace the streaming PyTorch full-reference serving internals with verified
   Triton layer kernels, starting with real DeltaNet and the corrected gated
   full-attention Q projection layout.
2. Reduce launch count inside the real-shape Qwen3.6 MoE wrappers; the shared
   expert gate currently adds a small Triton scalar launch.
3. Keep the vLLM comparison at matched routed-MoE operator boundaries until
   full serving configs exist.
4. Continue reducing launch count in measured-wins-only steps.
5. Specialize top-8 MoE performance after adding top-k 1/2/4/8 tiny-shape
   parity coverage, 256-expert/top-8 synthetic coverage, and restoring the
   top-2 fast path.
6. Revisit attention fusion with a lower-overhead implementation; the benchmark
   path now avoids KV-cache clones, but the first projection+RoPE/cache fusion
   attempt measured slower than the staged default.
7. Decide when to introduce a C++/CUDA/CuTe extension build.
8. Add token-level parity against Transformers or another canonical backend
   once its Qwen3.6 architecture support is usable in this environment.
9. Keep `bench/modal_qwen36_full_serving.py --gpu H100:8 --run-vllm
   --vllm-backend openai-image` as the end-to-end vLLM serving probe. On April
   29, 2026, vLLM 0.10.2 rejected the Qwen3.6 architecture in both native and
   Transformers fallback modes. On April 30, 2026, the pip-install
   `vllm==0.20.0` lane failed package installation because its source build
   requires `CUDA_HOME` in the Modal runtime.

## Things Not To Change

- Do not remove PyTorch reference implementations; they are correctness oracles.
- Do not hardcode or commit Lambda API key values.
- Do not claim full Qwen3.6-35B-A3B inference support until real weights and
  token-level parity are tested.
- Do not switch fully to CuTe/CUTLASS before the synthetic dataflow and parity
  tests are stable.
- Do not terminate or replace a running Lambda instance without confirming the
  user wants that.

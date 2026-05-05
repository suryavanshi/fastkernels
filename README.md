# fastkernels

Use AI to build and optimize inference kernels for MoE LLM inference.

The first target is Qwen3.5-35B-A3B: a 35B total / roughly 3B active MoE model
with 40 layers, hidden size 2048, 256 routed experts, 8 routed experts per
token, and 1 shared expert. The initial kernel work focuses on MoE routing,
expert layout, fused expert activations, and the path toward persistent
decode-time megakernels.

## Installation

```sh
pip install fastkernels
```

For local development or benchmarking from a checkout:

```sh
pip install -e ".[triton,dev]"
```

The Triton extra is intended for Linux CUDA environments. CPU-only systems can
still run the model-shape utilities and PyTorch reference checks with the
`bench` or `dev` extras.

On GPU hosts whose NVIDIA driver does not support CUDA 13, install a CUDA
12-compatible PyTorch/Triton pair before the editable install:

```sh
pip install "torch==2.7.1" "triton==3.3.1" numpy
pip install -e ".[triton,dev]"
```

## Usage

```python
import fastkernels

print(fastkernels.__version__)
```

## Current Kernel Work

- Qwen3.5 model-shape metadata in `fastkernels.models`.
- PyTorch reference MoE operations in `fastkernels.reference`.
- Optional Triton MoE building blocks in `fastkernels.kernels.triton`.
- Benchmark and correctness entrypoints under `bench/`.
- Detailed execution plan in `docs/QWEN35_KERNEL_PLAN.md`.
- Qwen3.6 megakernel design notes in `docs/QWEN36_MEGAKERNEL_PLAN.md`.
- Megakernel build process and fusion techniques in
  `docs/megakernel-building-process.md`.
- Modal GPU runbook in `docs/modal.md`.

This repository does not yet contain a full all-layer Qwen DeltaNet/attention
megakernel. The current tested CUDA path covers MoE building blocks such as
fused SwiGLU, routed-expert histograms, prototype Triton synthetic
attention/DeltaNet/MoE decode blocks, in-place synthetic attention cache
updates for decode benchmarking, layer-boundary wrappers, tiny-shape MoE top-k
coverage up to 256 experts/top-8 routing, and a first real-shape Qwen3.6
MoE path covering hidden-2048 / 256-expert / top-8 decode routing plus a
single hidden-2048 / intermediate-512 expert MLP, a batched two-launch
top-8 routed plus shared expert accumulation path, and a full real-shape MoE
decode wrapper that connects router output to routed/shared expert execution.
The real-shape path also has an optional routed-expert-only comparison against
vLLM's fused MoE op, including a real-weight mode that downloads only the
selected MoE layer shards and compares routed experts at the matched operator
boundary. There are also batched routed/shared expert and full MoE paths for
multiple decode tokens. There is also a safetensors loader/packer smoke lane
for HF-style Qwen3.6 MoE layer keys, including indexed sharded safetensors
directories and a configurable real-weight smoke that downloads only the
selected Qwen3.6 shard files needed for requested MoE layers, including the
real shared-expert gate and a residual `hidden + MoE` layer-boundary harness.
The real full-attention lane now also checks a Triton RMSNorm+Q/K/V projection
kernel plus staged Q/K norm, RoPE/cache update, causal attention accumulation,
and output projection against actual layer-3 Qwen3.6 weights. The same
real-weight block smoke composes the attention update into the layer-3 MoE
residual boundary and checks attention-hidden, MoE update, final layer-hidden,
cache, router-logit, and top-k parity. The real DeltaNet lane now also stages
input RMSNorm, QKV/Z/A/B projections, causal depthwise convolution, conv-state
update, recurrent-state update, gated RMSNorm, and output projection in Triton
against actual layer-0 Qwen3.6 weights, and composes that update into a real
DeltaNet -> MoE layer-boundary wrapper with layer-output/router/top-k parity.
The full-model planning lane now resolves the complete 40-layer real Qwen3.6
safetensors plan: all root tensors, all 30 DeltaNet+MoE layers, all 10
Attention+MoE layers, and all 26 model shards. The repository also has a Modal
H100:8 streaming reference runner that loads real Qwen3.6 weights and runs all
40 layers for token generation, with selectable Triton Attention, DeltaNet
projection+conv+recurrent/output, Attention -> MoE layer staging, DeltaNet ->
MoE layer staging, and MoE staging. The full Triton/megakernel serving path is
still incomplete because the current runner is still a staged multi-kernel
streaming path rather than a fused all-layer megakernel, and the
vLLM baseline now has a Modal lane based on the prebuilt
`vllm/vllm-openai:v0.20.0-x86_64` container image, avoiding the CUDA source
build failure seen when pip-installing `vllm==0.20.0` into the slim
fastkernels runtime.

## Shape Report

```sh
PYTHONPATH=src python bench/qwen35_profile.py --tokens 1 16 128
```

## MoE Microbench

```sh
PYTHONPATH=src python bench/moe_microbench.py --device cuda --dtype bfloat16
```

The microbench falls back gracefully when Triton is unavailable. The PyTorch
reference path is kept as the correctness oracle for each custom kernel.

## Qwen3.6 Synthetic Decode

Run the synthetic Qwen3.6 decode benchmark locally on a CUDA host:

```sh
PYTHONPATH=src python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --moe-impl triton
```

The decode benchmark can switch the synthetic attention, DeltaNet, and MoE paths
independently:

```sh
PYTHONPATH=src python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2 --attention-impl triton --deltanet-impl triton --moe-impl triton
```

Run the same benchmark on Modal:

```sh
modal run bench/modal_qwen36_decode.py
```

Run Modal with the layer-boundary microbenchmarks:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-layer-microbench
```

Run Modal with layer-boundary and MoE top-k/expert-count microbenchmarks:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-layer-microbench --run-moe-topk-microbench
```

Run Modal with the real-shape Qwen3.6 MoE router microbenchmark:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench
```

Run Modal with the real-shape router and single-expert MLP microbenchmarks:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench
```

Run Modal with the real-shape router, single-expert MLP, and routed/shared MoE
microbenchmarks:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench
```

Run Modal with the full real-shape MoE wrapper microbenchmark:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench --run-real-moe-microbench
```

Run Modal with the batched real-shape routed/shared MoE microbenchmark:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench --run-real-batched-routed-moe-microbench --run-real-moe-microbench
```

Run Modal with the batched real-shape full MoE microbenchmark:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench --run-real-batched-routed-moe-microbench --run-real-moe-microbench --run-real-batched-moe-microbench
```

Run Modal with the indexed safetensors MoE loader smoke test:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench --run-real-batched-routed-moe-microbench --run-real-moe-microbench --run-real-batched-moe-microbench --run-safetensor-moe-smoke
```

Run Modal with the real Qwen3.6 MoE shard smoke test:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-weight-moe-smoke
```

You can widen that smoke across layers and decode-token rows:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-weight-moe-smoke --real-weight-tokens 2 --real-weight-layers 0,1
```

Run the real-weight MoE smoke with the token-level residual layer harness:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-weight-layer-smoke --real-weight-tokens 2 --real-weight-layers 0,1
```

Run the real-weight block smoke for one DeltaNet layer and one full-attention
layer, including staged real Attention -> MoE layer-boundary parity:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-weight-block-smoke --real-weight-tokens 2
```

Run Modal with the optional vLLM routed-MoE comparison:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-router-microbench --run-real-expert-microbench --run-real-routed-moe-microbench --run-real-moe-microbench --run-vllm-moe-comparison
```

Run the optional vLLM routed-MoE comparison over selected real Qwen3.6 MoE
weights:

```sh
modal run bench/modal_qwen36_decode.py --attention-impl triton --deltanet-impl triton --moe-impl triton --run-real-weight-layer-smoke --real-weight-tokens 2 --real-weight-layers 0,1 --run-real-weight-vllm-moe-comparison
```

Resolve the full real Qwen3.6 40-layer weight plan on Modal without downloading
all shards:

```sh
modal run bench/modal_qwen36_full_serving.py --gpu A100 --run-weight-plan --no-run-vllm --no-run-pytest
```

Run the streaming full-40-layer PyTorch reference decode on 8 H100 GPUs:

```sh
modal run bench/modal_qwen36_full_serving.py --gpu H100:8 --run-full-decode --output-len 1 --prompt-token-ids 0 --no-run-weight-plan --no-run-pytest --no-run-vllm
```

Run a smaller real-weight streaming smoke with Triton MoE enabled:

```sh
modal run bench/modal_qwen36_full_serving.py --gpu A100 --run-full-decode --moe-impl triton --max-layers 4 --output-len 0 --prompt-token-ids 0 --no-run-weight-plan --no-run-pytest --no-run-vllm
```

Run the same first-four-layer smoke with Triton Attention and Triton MoE. This
covers the first real Attention+MoE layer while the DeltaNet layers still use
the reference mixer:

```sh
modal run bench/modal_qwen36_full_serving.py --gpu A100 --run-full-decode --attention-impl triton --moe-impl triton --max-layers 4 --output-len 0 --prompt-token-ids 0 --no-run-weight-plan --no-run-pytest --no-run-vllm
```

Run the same first-four-layer smoke with real-weight Triton DeltaNet
projection+conv+recurrent/output staging enabled:

```sh
modal run bench/modal_qwen36_full_serving.py --gpu A100 --run-full-decode --attention-impl triton --deltanet-impl triton --moe-impl triton --max-layers 4 --output-len 0 --prompt-token-ids 0 --no-run-weight-plan --no-run-pytest --no-run-vllm
```

Run the full-model vLLM serving baseline on 8 H100 GPUs through the prebuilt
vLLM OpenAI image:

```sh
modal run bench/modal_qwen36_full_serving.py --gpu H100:8 --run-vllm --vllm-backend openai-image --no-run-weight-plan --no-run-pytest --max-model-len 2048 --num-prompts 2 --input-len 32 --output-len 8 --tensor-parallel-size 8
```

The previous pip-install probe is still available for debugging package
resolution in the fastkernels runtime:

```sh
modal run bench/modal_qwen36_full_serving.py --gpu H100:8 --run-vllm --vllm-backend pip --no-run-weight-plan --no-run-pytest --max-model-len 2048 --num-prompts 2 --input-len 32 --output-len 8 --tensor-parallel-size 8
```

There is also a standalone Modal-docs-style vLLM runner that builds from a
CUDA devel image and installs `vllm==0.20.0` with `uv`:

```sh
modal run bench/modal_qwen36_vllm_serving.py --max-model-len 2048 --num-prompts 2 --input-len 32 --output-len 8 --tensor-parallel-size 8
```

That standalone runner reaches vLLM 0.20.0 model startup and real Qwen3.6
weight loading on H100:8. It has not emitted an end-to-end throughput number
yet; current runs stop during post-load vLLM engine setup.

See [`docs/modal.md`](docs/modal.md) for Modal authentication, GPU selection,
and current scope. The Modal path currently validates synthetic decode and
microbench coverage plus real-shape MoE routing and expert execution kernels.
It now has configurable real-weight MoE shard, residual layer-boundary,
non-MoE block loader, real full-attention decode, and real Attention -> MoE
layer-boundary smokes, plus a full 40-layer weight-plan lane. The fastkernels
runtime now has a streaming PyTorch reference lane for full real-weight
Qwen3.6-35B-A3B generation with selectable Triton Attention, DeltaNet
projection+conv+recurrent/output, Attention -> MoE layer staging, DeltaNet ->
MoE layer staging, and MoE staging. That runner now recognizes repeated
four-layer Qwen3.6 pattern chunks and reports layer-position throughput metrics
for staged real-weight Triton runs; the latest Modal A100 prefill/logits smoke
reaches all 40 real layers / 10 pattern chunks with all current Triton staging
flags enabled.
The fused full Triton megakernel serving path is not complete yet. The vLLM
comparison now has an end-to-end serving benchmark entrypoint using the
prebuilt vLLM OpenAI image, plus the earlier operator-level routed-MoE
comparison over selected real layer weights.

## Google Colab

Use the Colab notebook to install the repository on a GPU runtime, run the unit
tests, check Qwen3.5 MoE shapes, and compare the Triton MoE kernels against the
PyTorch reference path:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suryavanshi/fastkernels/blob/main/notebooks/colab_kernel_test.ipynb)

In Colab, choose `Runtime > Change runtime type > GPU` before running the
notebook. The notebook defaults to cloning the main repository, but you can set
`FASTKERNELS_REPO` or `FASTKERNELS_BRANCH` in the first setup cell when testing
a fork or branch.

## Lambda GPU Instance

For a persistent SSH workflow on Lambda Cloud, see
[`docs/LAMBDA.md`](docs/LAMBDA.md). The runbook covers launching an instance,
syncing a local checkout, installing CUDA-compatible PyTorch/Triton wheels,
running the test suite, running the kernel microbenchmarks, and terminating the
instance to stop billing.

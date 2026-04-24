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

This repository does not yet contain a full all-layer Qwen DeltaNet/attention
megakernel. The current tested CUDA path covers MoE building blocks such as
fused SwiGLU and routed-expert histograms.

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

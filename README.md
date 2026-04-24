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

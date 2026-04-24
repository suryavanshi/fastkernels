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

Current Qwen3.6 work is synthetic and structurally faithful. It models the
Qwen3.6 layer pattern, decode state, DeltaNet state, attention KV cache, routed
MoE, and shared expert path, but it does not load full Qwen3.6-35B-A3B weights.

## Files Changed So Far

- `README.md`: install notes, Colab link, Lambda runbook link, current kernel scope.
- `notebooks/colab_kernel_test.ipynb`: Colab GPU test notebook.
- `docs/LAMBDA.md`: Lambda launch, SSH, install, test, benchmark, terminate runbook.
- `docs/QWEN36_MEGAKERNEL_PLAN.md`: staged Qwen3.6 megakernel plan.
- `src/fastkernels/__init__.py`: exports Qwen3.6 spec.
- `src/fastkernels/models/__init__.py`: exports Qwen3.6 helpers.
- `src/fastkernels/models/qwen36.py`: Qwen3.6-35B-A3B and synthetic specs.
- `src/fastkernels/reference/__init__.py`: exports Qwen3.6 decode reference helpers.
- `src/fastkernels/reference/qwen36_decode.py`: synthetic Qwen3.6 decode reference.
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
- No known failing tests on Lambda.
- No full-model Qwen3.6 load or real-weight parity test exists yet.

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

1. Implement a fused Triton synthetic MoE decode block and test it against
   `reference_qwen36_moe_decode`.
2. Add benchmark flags that compare PyTorch reference versus fused Triton blocks
   inside the synthetic Qwen3.6 decode loop.
3. Prototype a synthetic DeltaNet decode kernel in Triton.
4. Decide when to introduce a C++/CUDA/CuTe extension build.
5. Add real model config ingestion for Qwen3.6 once exact Hugging Face config
   field names are validated from `config.json`.
6. Plan full-model testing on an 8-GPU Lambda instance or a quantized setup.

## Things Not To Change

- Do not remove PyTorch reference implementations; they are correctness oracles.
- Do not hardcode or commit Lambda API key values.
- Do not claim full Qwen3.6-35B-A3B inference support until real weights and
  token-level parity are tested.
- Do not switch fully to CuTe/CUTLASS before the synthetic dataflow and parity
  tests are stable.
- Do not terminate or replace a running Lambda instance without confirming the
  user wants that.

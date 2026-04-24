# AGENTS.md

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For CUDA/Triton work on Linux GPU hosts:

```sh
python -m pip install "torch==2.7.1" "triton==3.3.1" numpy
python -m pip install -e ".[triton,dev]"
```

## Test Commands

```sh
pytest -q
PYTHONPATH=src python bench/qwen35_profile.py --tokens 1 16 128
PYTHONPATH=src python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2
PYTHONPATH=src python bench/moe_microbench.py --device cuda --dtype bfloat16 --rows 8192 --tokens 4 --warmup 10 --iters 100 --skip-routed-moe
```

There is no lint command or dev server configured.

## Coding Style

- Python 3.9+ with `src/` package layout.
- Keep optional GPU dependencies lazy-imported.
- Keep PyTorch reference paths as correctness oracles for custom kernels.
- Prefer small, deterministic synthetic tests before full model work.
- Keep docs honest about what is implemented versus planned.

## Repo Structure

- `src/fastkernels/models/`: model shape specs.
- `src/fastkernels/reference/`: PyTorch reference implementations.
- `src/fastkernels/kernels/triton/`: optional Triton kernels.
- `bench/`: benchmark and smoke-test entrypoints.
- `tests/`: unit and optional CUDA/Triton tests.
- `docs/`: design plans and runbooks.

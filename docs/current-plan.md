# Current Plan

## Objective

Progress from the current MoE building-block kernels toward a Qwen3.6-35B-A3B
decode megakernel, using deterministic synthetic references and Lambda GPU tests
at each step.

## Verified Baseline

On Lambda `gpu_1x_a100_sxm4`:

```text
pytest -q
11 passed in 2.21s

python bench/qwen36_decode_reference.py --device cuda --dtype float32 --steps 8 --warmup 2
decode_ms_per_step: 4.9802

python bench/qwen36_decode_reference.py --device cuda --dtype bfloat16 --steps 8 --warmup 2
decode_ms_per_step: 5.1264

python bench/moe_microbench.py --device cuda --dtype bfloat16 --rows 8192 --tokens 4 --warmup 10 --iters 100 --skip-routed-moe
torch_swiglu_ms: 0.0559
triton_swiglu_ms: 0.0442
triton_expert_histogram: correctness ok
```

## Priority TODOs

1. Implement a fused Triton synthetic MoE decode block:
   - Input: one hidden vector, router weights/logits, expert weights, shared expert weights.
   - Output: same tensor contract as `reference_qwen36_moe_decode`.
   - Test against PyTorch reference for float32 and bfloat16 tolerances.

2. Add benchmark integration:
   - Add an option to `bench/qwen36_decode_reference.py` for reference-only vs
     fused-MoE mode.
   - Print per-step timing and correctness delta.

3. Prototype synthetic DeltaNet kernel:
   - Start with one layer, one token, fixed synthetic shapes.
   - Match `reference_qwen36_deltanet_decode`.
   - Measure register pressure and memory movement on A100.

4. Decide CuTe/CUTLASS boundary:
   - Keep Triton while dataflow changes quickly.
   - Move the stable hot path to CuTe/CUTLASS if Triton cannot express the
     persistent decode schedule or if register/shared-memory control is limiting.

5. Full model path:
   - Fetch and inspect the real Hugging Face `config.json`.
   - Add config alias tests for exact Qwen3.6 fields.
   - Plan 8-GPU or quantized Lambda validation for real weights.

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

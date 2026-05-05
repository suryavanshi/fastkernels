"""Benchmark full-model Qwen3.6 serving through vLLM."""

from __future__ import annotations

import argparse
import time


def _require_vllm():
    try:
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is required for full-model Qwen3.6 serving benchmarks") from exc
    return vllm, LLM, SamplingParams


def _make_prompts(num_prompts: int, input_len: int) -> list[str]:
    # Keep tokenization stable-ish without depending on tokenizer internals here.
    base = "Qwen3.6 MoE decode benchmark prompt "
    words = [f"tok{i}" for i in range(max(1, input_len))]
    prompt = base + " ".join(words)
    return [prompt for _ in range(num_prompts)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--model-impl", default="auto", choices=["auto", "vllm", "transformers"])
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gdn-prefill-backend", choices=["flashinfer", "triton"], default=None)
    args = parser.parse_args()

    try:
        vllm, LLM, SamplingParams = _require_vllm()
    except Exception as exc:
        print("backend: vllm")
        print("vllm_version: unknown")
        print(f"model: {args.model}")
        print(f"model_impl: {args.model_impl}")
        print("vllm_full_serving_supported: False")
        print(f"vllm_full_serving_error_type: {type(exc).__name__}")
        print(f"vllm_full_serving_error: {exc}")
        return
    prompts = _make_prompts(args.num_prompts, args.input_len)

    load_start = time.perf_counter()
    llm_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
    }
    if args.model_impl != "auto":
        llm_kwargs["model_impl"] = args.model_impl
    if args.gdn_prefill_backend is not None:
        llm_kwargs["gdn_prefill_backend"] = args.gdn_prefill_backend
    try:
        llm = LLM(**llm_kwargs)
    except Exception as exc:
        print("backend: vllm")
        print(f"vllm_version: {getattr(vllm, '__version__', 'unknown')}")
        print(f"model: {args.model}")
        print(f"model_impl: {args.model_impl}")
        print("vllm_full_serving_supported: False")
        print(f"vllm_full_serving_error_type: {type(exc).__name__}")
        print(f"vllm_full_serving_error: {exc}")
        return
    load_seconds = time.perf_counter() - load_start

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.output_len,
        ignore_eos=True,
    )

    # One warmup generate keeps load/compile out of the measured throughput.
    llm.generate(prompts[:1], sampling)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    generate_seconds = time.perf_counter() - start

    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    prompt_tokens = sum(len(output.prompt_token_ids or []) for output in outputs)
    total_tokens = prompt_tokens + generated_tokens

    print(f"backend: vllm")
    print(f"vllm_version: {getattr(vllm, '__version__', 'unknown')}")
    print(f"model: {args.model}")
    print(f"dtype: {args.dtype}")
    print(f"model_impl: {args.model_impl}")
    print(f"tensor_parallel_size: {args.tensor_parallel_size}")
    print(f"max_model_len: {args.max_model_len}")
    print(f"num_prompts: {args.num_prompts}")
    print(f"requested_input_len_words: {args.input_len}")
    print(f"requested_output_len: {args.output_len}")
    print(f"prompt_tokens: {prompt_tokens}")
    print(f"generated_tokens: {generated_tokens}")
    print(f"total_tokens: {total_tokens}")
    print(f"load_seconds: {load_seconds:.2f}")
    print(f"generate_seconds: {generate_seconds:.4f}")
    print(f"generated_tokens_per_second: {generated_tokens / generate_seconds:.2f}")
    print(f"total_tokens_per_second: {total_tokens / generate_seconds:.2f}")


if __name__ == "__main__":
    main()

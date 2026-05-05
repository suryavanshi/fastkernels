"""Run synthetic Qwen3.6 decode benchmarks on Modal GPUs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/fastkernels"

app = modal.App("fastkernels-qwen36-decode")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1",
        "triton==3.3.1",
        "numpy",
        "pytest>=8",
        "safetensors",
        "huggingface_hub[hf_xet]>=0.34.0",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path=REMOTE_REPO,
        copy=True,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            "*.egg-info",
        ],
    )
    .workdir(REMOTE_REPO)
    .run_commands("python -m pip install -e '.[dev]'")
)


def _run_checked(command: list[str]) -> str:
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=REMOTE_REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, flush=True)
    if completed.returncode:
        raise RuntimeError(
            "$ " + " ".join(command) + f"\nexited with {completed.returncode}\n" + completed.stdout
        )
    return completed.stdout


def _benchmark_commands(
    dtype: str,
    steps: int,
    warmup: int,
    attention_impl: str,
    deltanet_impl: str,
    moe_impl: str,
) -> list[list[str]]:
    impls = ["reference", "triton"] if moe_impl == "both" else [moe_impl]
    return [
        [
            "python",
            "bench/qwen36_decode_reference.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--steps",
            str(steps),
            "--warmup",
            str(warmup),
            "--attention-impl",
            attention_impl,
            "--deltanet-impl",
            deltanet_impl,
            "--moe-impl",
            impl,
        ]
        for impl in impls
    ]


def _run_suite(
    dtype: str,
    steps: int,
    warmup: int,
    attention_impl: str,
    deltanet_impl: str,
    moe_impl: str,
    run_pytest: bool,
    run_layer_microbench: bool,
    run_moe_topk_microbench: bool,
    run_real_router_microbench: bool,
    run_real_expert_microbench: bool,
    run_real_routed_moe_microbench: bool,
    run_real_batched_routed_moe_microbench: bool,
    run_real_moe_microbench: bool,
    run_real_batched_moe_microbench: bool,
    run_safetensor_moe_smoke: bool,
    run_real_weight_key_probe: bool,
    real_weight_key_probe_load_shapes: bool,
    run_real_weight_block_smoke: bool,
    run_real_weight_moe_smoke: bool,
    run_real_weight_layer_smoke: bool,
    real_weight_layers: str,
    real_weight_tokens: int,
    run_vllm_moe_comparison: bool,
    run_real_weight_vllm_moe_comparison: bool,
    vllm_package: str,
) -> str:
    sections = []
    if run_pytest:
        sections.append("$ pytest -q\n" + _run_checked(["pytest", "-q"]))

    for command in _benchmark_commands(dtype, steps, warmup, attention_impl, deltanet_impl, moe_impl):
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_layer_microbench:
        for layer_kind in ("deltanet_moe", "attention_moe"):
            command = [
                "python",
                "bench/qwen36_layer_microbench.py",
                "--layer-kind",
                layer_kind,
                "--impl",
                "triton",
                "--device",
                "cuda",
                "--dtype",
                dtype,
                "--warmup",
                "10",
                "--iters",
                "100",
            ]
            sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_moe_topk_microbench:
        for experts, top_k, warmup_iters, bench_iters in ((4, 2, 10, 100), (256, 8, 3, 10)):
            command = [
                "python",
                "bench/qwen36_moe_topk_microbench.py",
                "--device",
                "cuda",
                "--dtype",
                dtype,
                "--experts",
                str(experts),
                "--top-k",
                str(top_k),
                "--warmup",
                str(warmup_iters),
                "--iters",
                str(bench_iters),
            ]
            sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_real_router_microbench:
        command = [
            "python",
            "bench/qwen36_router_microbench.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--warmup",
            "10",
            "--iters",
            "100",
        ]
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_real_expert_microbench:
        command = [
            "python",
            "bench/qwen36_expert_microbench.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--warmup",
            "10",
            "--iters",
            "100",
        ]
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_real_routed_moe_microbench:
        command = [
            "python",
            "bench/qwen36_routed_moe_microbench.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--warmup",
            "3",
            "--iters",
            "10",
        ]
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_real_batched_routed_moe_microbench:
        command = [
            "python",
            "bench/qwen36_batched_routed_moe_microbench.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--tokens",
            "4",
            "--warmup",
            "3",
            "--iters",
            "10",
        ]
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_real_moe_microbench:
        command = [
            "python",
            "bench/qwen36_real_moe_microbench.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--warmup",
            "3",
            "--iters",
            "10",
        ]
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_real_batched_moe_microbench:
        command = [
            "python",
            "bench/qwen36_batched_real_moe_microbench.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--tokens",
            "4",
            "--warmup",
            "3",
            "--iters",
            "10",
        ]
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_safetensor_moe_smoke:
        command = [
            "python",
            "bench/qwen36_safetensor_moe_smoke.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--tokens",
            "2",
            "--warmup",
            "2",
            "--iters",
            "5",
        ]
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_real_weight_key_probe:
        command = [
            "python",
            "bench/qwen36_real_weight_key_probe.py",
            "--layers",
        ]
        layers = [layer for layer in real_weight_layers.replace(",", " ").split() if layer]
        command.extend(layers or ["0", "3"])
        if real_weight_key_probe_load_shapes:
            command.append("--load-shapes")
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_real_weight_block_smoke:
        command = [
            "python",
            "bench/qwen36_real_weight_block_smoke.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--tokens",
            str(real_weight_tokens),
        ]
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_real_weight_moe_smoke or run_real_weight_layer_smoke:
        command = [
            "python",
            "bench/qwen36_real_weight_moe_smoke.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--tokens",
            str(real_weight_tokens),
            "--warmup",
            "1",
            "--iters",
            "3",
        ]
        layers = [layer for layer in real_weight_layers.replace(",", " ").split() if layer]
        if layers:
            command.append("--layers")
            command.extend(layers)
        if run_real_weight_layer_smoke:
            command.append("--run-layer-harness")
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_vllm_moe_comparison:
        install_command = ["python", "-m", "pip", "install", vllm_package]
        sections.append("$ " + " ".join(install_command) + "\n" + _run_checked(install_command))
        command = [
            "python",
            "bench/qwen36_vllm_moe_compare.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--warmup",
            "5",
            "--iters",
            "20",
            "--require-vllm",
        ]
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))
    if run_real_weight_vllm_moe_comparison:
        install_command = ["python", "-m", "pip", "install", vllm_package]
        sections.append("$ " + " ".join(install_command) + "\n" + _run_checked(install_command))
        layers = [layer for layer in real_weight_layers.replace(",", " ").split() if layer]
        layer = layers[0] if layers else "0"
        command = [
            "python",
            "bench/qwen36_vllm_moe_compare.py",
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--tokens",
            str(real_weight_tokens),
            "--layer",
            layer,
            "--warmup",
            "5",
            "--iters",
            "20",
            "--real-weights",
            "--require-vllm",
        ]
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command))

    return "\n".join(sections)


@app.function(image=image, gpu="A100", timeout=1800)
def run_on_a100(
    dtype: str,
    steps: int,
    warmup: int,
    attention_impl: str,
    deltanet_impl: str,
    moe_impl: str,
    run_pytest: bool,
    run_layer_microbench: bool,
    run_moe_topk_microbench: bool,
    run_real_router_microbench: bool,
    run_real_expert_microbench: bool,
    run_real_routed_moe_microbench: bool,
    run_real_batched_routed_moe_microbench: bool,
    run_real_moe_microbench: bool,
    run_real_batched_moe_microbench: bool,
    run_safetensor_moe_smoke: bool,
    run_real_weight_key_probe: bool,
    real_weight_key_probe_load_shapes: bool,
    run_real_weight_block_smoke: bool,
    run_real_weight_moe_smoke: bool,
    run_real_weight_layer_smoke: bool,
    real_weight_layers: str,
    real_weight_tokens: int,
    run_vllm_moe_comparison: bool,
    run_real_weight_vllm_moe_comparison: bool,
    vllm_package: str,
) -> str:
    return _run_suite(
        dtype,
        steps,
        warmup,
        attention_impl,
        deltanet_impl,
        moe_impl,
        run_pytest,
        run_layer_microbench,
        run_moe_topk_microbench,
        run_real_router_microbench,
        run_real_expert_microbench,
        run_real_routed_moe_microbench,
        run_real_batched_routed_moe_microbench,
        run_real_moe_microbench,
        run_real_batched_moe_microbench,
        run_safetensor_moe_smoke,
        run_real_weight_key_probe,
        real_weight_key_probe_load_shapes,
        run_real_weight_block_smoke,
        run_real_weight_moe_smoke,
        run_real_weight_layer_smoke,
        real_weight_layers,
        real_weight_tokens,
        run_vllm_moe_comparison,
        run_real_weight_vllm_moe_comparison,
        vllm_package,
    )


@app.function(image=image, gpu="H100!", timeout=1800)
def run_on_h100(
    dtype: str,
    steps: int,
    warmup: int,
    attention_impl: str,
    deltanet_impl: str,
    moe_impl: str,
    run_pytest: bool,
    run_layer_microbench: bool,
    run_moe_topk_microbench: bool,
    run_real_router_microbench: bool,
    run_real_expert_microbench: bool,
    run_real_routed_moe_microbench: bool,
    run_real_batched_routed_moe_microbench: bool,
    run_real_moe_microbench: bool,
    run_real_batched_moe_microbench: bool,
    run_safetensor_moe_smoke: bool,
    run_real_weight_key_probe: bool,
    real_weight_key_probe_load_shapes: bool,
    run_real_weight_block_smoke: bool,
    run_real_weight_moe_smoke: bool,
    run_real_weight_layer_smoke: bool,
    real_weight_layers: str,
    real_weight_tokens: int,
    run_vllm_moe_comparison: bool,
    run_real_weight_vllm_moe_comparison: bool,
    vllm_package: str,
) -> str:
    return _run_suite(
        dtype,
        steps,
        warmup,
        attention_impl,
        deltanet_impl,
        moe_impl,
        run_pytest,
        run_layer_microbench,
        run_moe_topk_microbench,
        run_real_router_microbench,
        run_real_expert_microbench,
        run_real_routed_moe_microbench,
        run_real_batched_routed_moe_microbench,
        run_real_moe_microbench,
        run_real_batched_moe_microbench,
        run_safetensor_moe_smoke,
        run_real_weight_key_probe,
        real_weight_key_probe_load_shapes,
        run_real_weight_block_smoke,
        run_real_weight_moe_smoke,
        run_real_weight_layer_smoke,
        real_weight_layers,
        real_weight_tokens,
        run_vllm_moe_comparison,
        run_real_weight_vllm_moe_comparison,
        vllm_package,
    )


@app.local_entrypoint()
def main(
    gpu: str = "A100",
    dtype: str = "bfloat16",
    steps: int = 8,
    warmup: int = 2,
    attention_impl: str = "reference",
    deltanet_impl: str = "reference",
    moe_impl: str = "both",
    run_pytest: bool = True,
    run_layer_microbench: bool = False,
    run_moe_topk_microbench: bool = False,
    run_real_router_microbench: bool = False,
    run_real_expert_microbench: bool = False,
    run_real_routed_moe_microbench: bool = False,
    run_real_batched_routed_moe_microbench: bool = False,
    run_real_moe_microbench: bool = False,
    run_real_batched_moe_microbench: bool = False,
    run_safetensor_moe_smoke: bool = False,
    run_real_weight_key_probe: bool = False,
    real_weight_key_probe_load_shapes: bool = False,
    run_real_weight_block_smoke: bool = False,
    run_real_weight_moe_smoke: bool = False,
    run_real_weight_layer_smoke: bool = False,
    real_weight_layers: str = "0",
    real_weight_tokens: int = 1,
    run_vllm_moe_comparison: bool = False,
    run_real_weight_vllm_moe_comparison: bool = False,
    vllm_package: str = "vllm==0.20.0",
) -> None:
    """Run the Modal benchmark entrypoint."""

    if attention_impl not in {"reference", "triton"}:
        raise ValueError("attention_impl must be one of: reference, triton")
    if deltanet_impl not in {"reference", "triton"}:
        raise ValueError("deltanet_impl must be one of: reference, triton")
    if moe_impl not in {"reference", "triton", "both"}:
        raise ValueError("moe_impl must be one of: reference, triton, both")

    gpu_key = gpu.upper()
    if gpu_key == "A100":
        output = run_on_a100.remote(
            dtype,
            steps,
            warmup,
            attention_impl,
            deltanet_impl,
            moe_impl,
            run_pytest,
            run_layer_microbench,
            run_moe_topk_microbench,
            run_real_router_microbench,
            run_real_expert_microbench,
            run_real_routed_moe_microbench,
            run_real_batched_routed_moe_microbench,
            run_real_moe_microbench,
            run_real_batched_moe_microbench,
            run_safetensor_moe_smoke,
            run_real_weight_key_probe,
            real_weight_key_probe_load_shapes,
            run_real_weight_block_smoke,
            run_real_weight_moe_smoke,
            run_real_weight_layer_smoke,
            real_weight_layers,
            real_weight_tokens,
            run_vllm_moe_comparison,
            run_real_weight_vllm_moe_comparison,
            vllm_package,
        )
    elif gpu_key in {"H100", "H100!"}:
        output = run_on_h100.remote(
            dtype,
            steps,
            warmup,
            attention_impl,
            deltanet_impl,
            moe_impl,
            run_pytest,
            run_layer_microbench,
            run_moe_topk_microbench,
            run_real_router_microbench,
            run_real_expert_microbench,
            run_real_routed_moe_microbench,
            run_real_batched_routed_moe_microbench,
            run_real_moe_microbench,
            run_real_batched_moe_microbench,
            run_safetensor_moe_smoke,
            run_real_weight_key_probe,
            real_weight_key_probe_load_shapes,
            run_real_weight_block_smoke,
            run_real_weight_moe_smoke,
            run_real_weight_layer_smoke,
            real_weight_layers,
            real_weight_tokens,
            run_vllm_moe_comparison,
            run_real_weight_vllm_moe_comparison,
            vllm_package,
        )
    else:
        raise ValueError("gpu must be A100, H100, or H100!")

    print(output)

"""Run full-model Qwen3.6 vLLM serving benchmarks on Modal."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/fastkernels"
CUDA_VERSION = "13.2.0"
CUDA_IMAGE = f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04"
VLLM_PACKAGE = "vllm==0.20.0"
TORCH_BACKEND = "cu130"

app = modal.App("fastkernels-qwen36-vllm-serving")
hf_cache = modal.Volume.from_name("fastkernels-qwen36-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .uv_pip_install(
        VLLM_PACKAGE,
        "huggingface_hub[hf_transfer]>=0.34.0",
        extra_options=f"--torch-backend={TORCH_BACKEND}",
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
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
)


def _run_checked(command: list[str], timeout: int | None = None) -> str:
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=REMOTE_REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    print(completed.stdout, flush=True)
    if completed.returncode:
        raise RuntimeError(
            "$ " + " ".join(command) + f"\nexited with {completed.returncode}\n" + completed.stdout
        )
    return completed.stdout


def _run_streaming(command: list[str], timeout: int | None = None) -> str:
    print("$ " + " ".join(command), flush=True)
    start = time.monotonic()
    completed_output: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=REMOTE_REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if line:
            completed_output.append(line)
            print(line, end="", flush=True)
        elif process.poll() is not None:
            break
        if timeout is not None and time.monotonic() - start > timeout:
            process.kill()
            tail = f"\ncommand timed out after {timeout} seconds\n"
            completed_output.append(tail)
            print(tail, flush=True)
            break
        time.sleep(0.1)
    remaining = process.stdout.read()
    if remaining:
        completed_output.append(remaining)
        print(remaining, end="", flush=True)
    return_code = process.wait()
    output = "".join(completed_output)
    if return_code:
        raise RuntimeError(
            "$ " + " ".join(command) + f"\nexited with {return_code}\n" + output
        )
    return output


@app.function(
    image=image,
    gpu="H100:8",
    timeout=10800,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_vllm_on_h100_8(
    model: str,
    dtype: str,
    tensor_parallel_size: int,
    max_model_len: int,
    model_impl: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    enforce_eager: bool,
    gdn_prefill_backend: Optional[str],
) -> str:
    sections = [
        f"vllm_backend: cuda-uv\n",
        f"vllm_base_image: {CUDA_IMAGE}\n",
        f"vllm_package: {VLLM_PACKAGE}\n",
        f"torch_backend: {TORCH_BACKEND}\n",
    ]
    sections.append(
        "$ python -c import sys, vllm; print(sys.executable); print(getattr(vllm, '__version__', 'unknown'))\n"
        + _run_checked(
            [
                "python",
                "-c",
                "import sys, vllm; print(sys.executable); print(getattr(vllm, '__version__', 'unknown'))",
            ],
            timeout=300,
        )
    )
    command = [
        "python",
        "bench/qwen36_vllm_serving_benchmark.py",
        "--model",
        model,
        "--dtype",
        dtype,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--model-impl",
        model_impl,
        "--num-prompts",
        str(num_prompts),
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
    ]
    if enforce_eager:
        command.append("--enforce-eager")
    if gdn_prefill_backend:
        command.extend(["--gdn-prefill-backend", gdn_prefill_backend])
    try:
        sections.append("$ " + " ".join(command) + "\n" + _run_streaming(command, timeout=7200))
    finally:
        hf_cache.commit()
    return "\n".join(sections)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.6-35B-A3B",
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 8,
    max_model_len: int = 2048,
    model_impl: str = "auto",
    num_prompts: int = 2,
    input_len: int = 32,
    output_len: int = 8,
    enforce_eager: bool = False,
    gdn_prefill_backend: Optional[str] = None,
) -> None:
    """Run the vLLM full-model serving benchmark on Modal H100:8."""

    output = run_vllm_on_h100_8.remote(
        model,
        dtype,
        tensor_parallel_size,
        max_model_len,
        model_impl,
        num_prompts,
        input_len,
        output_len,
        enforce_eager,
        gdn_prefill_backend,
    )
    print(output)

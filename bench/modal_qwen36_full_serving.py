"""Run full-model Qwen3.6 planning and vLLM serving benchmarks on Modal."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/fastkernels"
VLLM_OPENAI_IMAGE = "vllm/vllm-openai:v0.20.0-x86_64"
VLLM_OPENAI_PYTHON = "/usr/local/bin/python3.12"

app = modal.App("fastkernels-qwen36-full-serving")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1",
        "triton==3.3.1",
        "huggingface_hub[hf_xet]>=0.34.0",
        "numpy",
        "pytest>=8",
        "safetensors",
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

vllm_openai_image = (
    modal.Image.from_registry(VLLM_OPENAI_IMAGE, add_python="3.11")
    .entrypoint([])
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
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
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


def _run_capture(command: list[str], timeout: int | None = None) -> tuple[int, str]:
    print("$ " + " ".join(command), flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=REMOTE_REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        output = f"{type(exc).__name__}: {exc}\n"
        print(output, flush=True)
        return 127, output
    print(completed.stdout, flush=True)
    return completed.returncode, completed.stdout


def _run_suite(
    model: str,
    revision: str,
    cache_dir: Optional[str],
    dtype: str,
    tensor_parallel_size: int,
    max_model_len: int,
    model_impl: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    prompt_token_ids: str,
    max_positions: int,
    max_layers: Optional[int],
    attention_impl: str,
    deltanet_impl: str,
    moe_impl: str,
    vllm_package: str,
    run_pytest: bool,
    run_weight_plan: bool,
    download_shards: bool,
    run_full_decode: bool,
    run_vllm: bool,
) -> str:
    sections = []
    if run_pytest:
        sections.append("$ pytest -q\n" + _run_checked(["pytest", "-q"], timeout=900))
    if run_weight_plan:
        command = [
            "python",
            "bench/qwen36_full_weight_plan.py",
            "--repo-id",
            model,
            "--revision",
            revision,
            "--show-layers",
            "--show-shards",
        ]
        if cache_dir:
            command.extend(["--cache-dir", cache_dir])
        if download_shards:
            command.append("--download-shards")
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command, timeout=7200))
    if run_full_decode:
        command = [
            "python",
            "bench/qwen36_full_decode.py",
            "--repo-id",
            model,
            "--revision",
            revision,
            "--device",
            "cuda",
            "--dtype",
            dtype,
            "--prompt-token-ids",
            prompt_token_ids,
            "--max-new-tokens",
            str(output_len),
            "--max-positions",
            str(max_positions),
            "--moe-impl",
            moe_impl,
        ]
        if attention_impl != "reference":
            command.extend(["--attention-impl", attention_impl])
        if deltanet_impl != "reference":
            command.extend(["--deltanet-impl", deltanet_impl])
        if cache_dir:
            command.extend(["--cache-dir", cache_dir])
        if max_layers is not None:
            command.extend(["--max-layers", str(max_layers)])
        sections.append("$ " + " ".join(command) + "\n" + _run_checked(command, timeout=10800))
    if run_vllm:
        install_command = ["python", "-m", "pip", "install", vllm_package]
        code, install_output = _run_capture(install_command, timeout=1800)
        section = "$ " + " ".join(install_command) + "\n" + install_output
        if code:
            section += (
                "\nbackend: vllm\n"
                f"vllm_package: {vllm_package}\n"
                "vllm_install_supported: False\n"
                f"vllm_install_exit_code: {code}\n"
                "vllm_full_serving_supported: False\n"
                "vllm_full_serving_error_type: InstallError\n"
            )
            sections.append(section)
        else:
            sections.append(section)
            sections.append(
                _run_vllm_benchmark(
                    model,
                    dtype,
                    tensor_parallel_size,
                    max_model_len,
                    model_impl,
                    num_prompts,
                    input_len,
                    output_len,
                )
            )
    return "\n".join(sections)


def _run_vllm_benchmark(
    model: str,
    dtype: str,
    tensor_parallel_size: int,
    max_model_len: int,
    model_impl: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    python_executable: str = "python",
) -> str:
    command = [
        python_executable,
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
    return "$ " + " ".join(command) + "\n" + _run_checked(command, timeout=7200)


@app.function(image=image, gpu="H100:8", timeout=10800)
def run_on_h100_8(
    model: str,
    revision: str,
    cache_dir: Optional[str],
    dtype: str,
    tensor_parallel_size: int,
    max_model_len: int,
    model_impl: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    prompt_token_ids: str,
    max_positions: int,
    max_layers: Optional[int],
    attention_impl: str,
    deltanet_impl: str,
    moe_impl: str,
    vllm_package: str,
    run_pytest: bool,
    run_weight_plan: bool,
    download_shards: bool,
    run_full_decode: bool,
    run_vllm: bool,
) -> str:
    return _run_suite(
        model,
        revision,
        cache_dir,
        dtype,
        tensor_parallel_size,
        max_model_len,
        model_impl,
        num_prompts,
        input_len,
        output_len,
        prompt_token_ids,
        max_positions,
        max_layers,
        attention_impl,
        deltanet_impl,
        moe_impl,
        vllm_package,
        run_pytest,
        run_weight_plan,
        download_shards,
        run_full_decode,
        run_vllm,
    )


@app.function(image=vllm_openai_image, gpu="H100:8", timeout=10800)
def run_vllm_openai_image_on_h100_8(
    model: str,
    dtype: str,
    tensor_parallel_size: int,
    max_model_len: int,
    model_impl: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
) -> str:
    section = (
        f"vllm_backend: openai-image\n"
        f"vllm_container_image: {VLLM_OPENAI_IMAGE}\n"
    )
    probe_command = [
        VLLM_OPENAI_PYTHON,
        "-c",
        "import sys, vllm; print(sys.executable); print(getattr(vllm, '__version__', 'unknown'))",
    ]
    code, probe_output = _run_capture(probe_command, timeout=300)
    section += "$ " + " ".join(probe_command) + "\n" + probe_output
    if code:
        return (
            section
            + "\nbackend: vllm\n"
            + f"vllm_container_image: {VLLM_OPENAI_IMAGE}\n"
            + "vllm_image_import_supported: False\n"
            + f"vllm_image_import_exit_code: {code}\n"
            + "vllm_full_serving_supported: False\n"
            + "vllm_full_serving_error_type: ImportError\n"
        )
    return section + _run_vllm_benchmark(
        model,
        dtype,
        tensor_parallel_size,
        max_model_len,
        model_impl,
        num_prompts,
        input_len,
        output_len,
        python_executable=VLLM_OPENAI_PYTHON,
    )


@app.function(image=image, gpu="A100", timeout=3600)
def run_on_a100_plan(
    model: str,
    revision: str,
    cache_dir: Optional[str],
    dtype: str,
    tensor_parallel_size: int,
    max_model_len: int,
    model_impl: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    prompt_token_ids: str,
    max_positions: int,
    max_layers: Optional[int],
    attention_impl: str,
    deltanet_impl: str,
    moe_impl: str,
    vllm_package: str,
    run_pytest: bool,
    run_weight_plan: bool,
    download_shards: bool,
    run_full_decode: bool,
    run_vllm: bool,
) -> str:
    if run_vllm:
        raise ValueError("A100 plan lane is for metadata/shard planning only; use gpu='H100:8' for vLLM serving")
    if run_full_decode and max_layers is None:
        raise ValueError("A100 full decode must set --max-layers; use gpu='H100:8' for all 40 layers")
    return _run_suite(
        model,
        revision,
        cache_dir,
        dtype,
        tensor_parallel_size,
        max_model_len,
        model_impl,
        num_prompts,
        input_len,
        output_len,
        prompt_token_ids,
        max_positions,
        max_layers,
        attention_impl,
        deltanet_impl,
        moe_impl,
        vllm_package,
        run_pytest,
        run_weight_plan,
        download_shards,
        run_full_decode,
        run_vllm,
    )


@app.local_entrypoint()
def main(
    gpu: str = "A100",
    model: str = "Qwen/Qwen3.6-35B-A3B",
    revision: str = "main",
    cache_dir: Optional[str] = None,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 8,
    max_model_len: int = 4096,
    model_impl: str = "auto",
    num_prompts: int = 4,
    input_len: int = 64,
    output_len: int = 16,
    prompt_token_ids: str = "0",
    max_positions: int = 128,
    max_layers: Optional[int] = None,
    attention_impl: str = "reference",
    deltanet_impl: str = "reference",
    moe_impl: str = "reference",
    vllm_package: str = "vllm==0.20.0",
    vllm_backend: str = "openai-image",
    run_pytest: bool = False,
    run_weight_plan: bool = True,
    download_shards: bool = False,
    run_full_decode: bool = False,
    run_vllm: bool = False,
) -> None:
    """Run the full-model planning or vLLM serving Modal entrypoint."""

    gpu_key = gpu.upper()
    vllm_backend_key = vllm_backend.lower().replace("_", "-")
    if vllm_backend_key not in {"openai-image", "pip"}:
        raise ValueError("vllm_backend must be openai-image or pip")
    if gpu_key == "A100":
        output = run_on_a100_plan.remote(
            model,
            revision,
            cache_dir,
            dtype,
            tensor_parallel_size,
            max_model_len,
            model_impl,
            num_prompts,
            input_len,
            output_len,
            prompt_token_ids,
            max_positions,
            max_layers,
            attention_impl,
            deltanet_impl,
            moe_impl,
            vllm_package,
            run_pytest,
            run_weight_plan,
            download_shards,
            run_full_decode,
            run_vllm,
        )
    elif gpu_key in {"H100:8", "8XH100"}:
        if (
            run_vllm
            and vllm_backend_key == "openai-image"
            and not run_pytest
            and not run_weight_plan
            and not download_shards
            and not run_full_decode
        ):
            output = run_vllm_openai_image_on_h100_8.remote(
                model,
                dtype,
                tensor_parallel_size,
                max_model_len,
                model_impl,
                num_prompts,
                input_len,
                output_len,
            )
        else:
            if run_vllm and vllm_backend_key == "openai-image":
                raise ValueError(
                    "The openai-image vLLM lane only runs a standalone vLLM "
                    "benchmark. Disable weight-plan/pytest/full-decode flags "
                    "or use --vllm-backend pip."
                )
            output = run_on_h100_8.remote(
                model,
                revision,
                cache_dir,
                dtype,
                tensor_parallel_size,
                max_model_len,
                model_impl,
                num_prompts,
                input_len,
                output_len,
                prompt_token_ids,
                max_positions,
                max_layers,
                attention_impl,
                deltanet_impl,
                moe_impl,
                vllm_package,
                run_pytest,
                run_weight_plan,
                download_shards,
                run_full_decode,
                run_vllm,
            )
    else:
        raise ValueError("gpu must be A100 or H100:8")
    print(output)

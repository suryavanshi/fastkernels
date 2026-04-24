# Lambda Cloud GPU Runbook

Use this runbook to launch a Lambda Cloud GPU instance, SSH into it, and run the
`fastkernels` tests and Triton kernel benchmarks.

## Current Scope

`fastkernels` currently tests Qwen3.5-35B-A3B shape metadata plus MoE
building-block kernels:

- `triton_fused_swiglu`
- `triton_expert_histogram`
- PyTorch reference routed-MoE smoke checks

It does not yet include a full all-layer Qwen DeltaNet/attention megakernel like
the Qwen 3.5-0.8B CUDA megakernel described in the Awesome Agents article.

## Prerequisites

- Lambda Cloud account with credits.
- Lambda API key stored locally, for example in:

  ```sh
  /Users/kb/Documents/proj/git_projs/.env
  ```

  with:

  ```sh
  LAMBDA_API_KEY=...
  ```

- An SSH public key registered in Lambda Cloud. The local private key used in
  testing was:

  ```sh
  /Users/kb/.ssh/id_ed25519
  ```

- Recommended instance for correctness and early performance work:
  `gpu_1x_a100_sxm4` in `us-east-1`.

## Launch

You can launch through the Lambda Cloud console, or use the API:

```sh
set -a
source /Users/kb/Documents/proj/git_projs/.env
set +a

curl -fsS \
  -A "curl/8.7.1" \
  -X POST \
  -H "Authorization: Bearer ${LAMBDA_API_KEY}" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  --data '{
    "region_name": "us-east-1",
    "instance_type_name": "gpu_1x_a100_sxm4",
    "ssh_key_names": ["mojo"],
    "name": "fastkernels-a100-test",
    "quantity": 1
  }' \
  https://cloud.lambdalabs.com/api/v1/instance-operations/launch
```

The response contains `instance_ids`. Poll the instance until it is `active`:

```sh
INSTANCE_ID=<INSTANCE_ID>

curl -fsS \
  -A "curl/8.7.1" \
  -H "Authorization: Bearer ${LAMBDA_API_KEY}" \
  -H "Accept: application/json" \
  "https://cloud.lambdalabs.com/api/v1/instances/${INSTANCE_ID}"
```

Record the returned public `ip`.

## SSH

```sh
ssh -i /Users/kb/.ssh/id_ed25519 ubuntu@<INSTANCE_IP>
```

Check the GPU:

```sh
nvidia-smi
python3 --version
```

## Put the Repo on the Instance

For committed code:

```sh
git clone https://github.com/suryavanshi/fastkernels.git
cd fastkernels
```

For local uncommitted changes from this workspace:

```sh
rsync -az --delete \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  -e "ssh -i /Users/kb/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new" \
  /Users/kb/Documents/proj/git_projs/fastkernels/ \
  ubuntu@<INSTANCE_IP>:~/fastkernels/
```

Then SSH in and enter the checkout:

```sh
ssh -i /Users/kb/.ssh/id_ed25519 ubuntu@<INSTANCE_IP>
cd ~/fastkernels
```

## Install

Use a CUDA-12-compatible PyTorch/Triton pair unless the Lambda image has a newer
driver that supports CUDA 13 wheels:

```sh
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install "torch==2.7.1" "triton==3.3.1" numpy
python -m pip install -e ".[triton,dev]"
```

Verify CUDA:

```sh
python - <<'PY'
import torch
import triton

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("triton", triton.__version__)
PY
```

Expected shape from the tested A100 instance:

```text
torch 2.7.1+cu126 cuda 12.6
cuda_available True
device NVIDIA A100-SXM4-40GB
triton 3.3.1
```

## Run Tests

```sh
pytest -q
```

On a CUDA instance this runs the CPU tests plus optional Triton correctness
tests. On a non-CUDA host, the Triton tests are skipped.

## Run Shape Report

```sh
python bench/qwen35_profile.py --tokens 1 16 128
```

## Run Kernel Benchmarks

Fast MoE building-block check:

```sh
python bench/moe_microbench.py \
  --device cuda \
  --dtype bfloat16 \
  --rows 2048 \
  --tokens 4 \
  --warmup 10 \
  --iters 100 \
  --skip-routed-moe
```

Larger SwiGLU timing:

```sh
python bench/moe_microbench.py \
  --device cuda \
  --dtype bfloat16 \
  --rows 8192 \
  --tokens 4 \
  --warmup 10 \
  --iters 100 \
  --skip-routed-moe
```

Tiny routed-MoE reference smoke test:

```sh
python bench/moe_microbench.py \
  --device cuda \
  --dtype bfloat16 \
  --rows 512 \
  --tokens 2 \
  --hidden 256 \
  --intermediate 128 \
  --experts 16 \
  --top-k 4 \
  --warmup 3 \
  --iters 10
```

## Known Good Result

On `gpu_1x_a100_sxm4` in `us-east-1`:

```text
5 passed in 2.31s

rows=2048:
triton_fused_swiglu: correctness ok
torch_swiglu_ms: 0.0229
triton_swiglu_ms: 0.0442
triton_expert_histogram: correctness ok

rows=8192:
triton_fused_swiglu: correctness ok
torch_swiglu_ms: 0.0560
triton_swiglu_ms: 0.0444
triton_expert_histogram: correctness ok
```

## Troubleshooting

If `torch.cuda.is_available()` is false and PyTorch reports that the NVIDIA
driver is too old, check the installed PyTorch CUDA version:

```sh
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
PY
```

If it shows a CUDA 13 build such as `+cu130`, reinstall the CUDA 12 pair:

```sh
python -m pip install --force-reinstall "torch==2.7.1" "triton==3.3.1" numpy
```

## Stop Billing

Terminate the instance from the Lambda Cloud console, or use:

```sh
curl -fsS \
  -A "curl/8.7.1" \
  -X POST \
  -H "Authorization: Bearer ${LAMBDA_API_KEY}" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  --data "{\"instance_ids\": [\"${INSTANCE_ID}\"]}" \
  https://cloud.lambdalabs.com/api/v1/instance-operations/terminate
```


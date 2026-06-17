#!/usr/bin/env bash
set -euo pipefail

# Driver 560.28.03 advertises CUDA 12.6 and cannot load CUDA 13.x PyTorch
# wheels. Keep the analysis environment on an official, reproducible cu126
# build unless explicitly overridden.
TORCH_VERSION="${TORCH_VERSION:-2.12.0}"
PYTORCH_CUDA="${PYTORCH_CUDA:-cu126}"
FORCE_TORCH_REINSTALL="${FORCE_TORCH_REINSTALL:-0}"
INDEX_URL="https://download.pytorch.org/whl/${PYTORCH_CUDA}"

command -v python >/dev/null || {
  echo "ERROR: activate the target virtual environment first" >&2
  exit 1
}
command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi not found" >&2
  exit 1
}

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader |
  sed -n '1p' | tr -d '[:space:]')"
reported_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p')"
echo "NVIDIA driver=${driver_version}; nvidia-smi CUDA capability=${reported_cuda:-unknown}"
echo "Required PyTorch build: torch=${TORCH_VERSION}, ${PYTORCH_CUDA}"
if [[ -n "${reported_cuda}" ]] && ! python - "${reported_cuda}" <<'PY'
import sys

version = tuple(map(int, sys.argv[1].split(".")[:2]))
raise SystemExit(0 if version >= (12, 6) else 1)
PY
then
  echo "ERROR: ${PYTORCH_CUDA} requires a driver reporting CUDA 12.6 or newer" >&2
  exit 1
fi

compatible=0
if [[ "${FORCE_TORCH_REINSTALL}" != "1" ]]; then
  if TORCH_VERSION="${TORCH_VERSION}" PYTORCH_CUDA="${PYTORCH_CUDA}" python - <<'PY'
import os
import sys

try:
    import torch
except Exception:
    sys.exit(1)

expected_version = os.environ["TORCH_VERSION"]
expected_cuda = os.environ["PYTORCH_CUDA"].removeprefix("cu")
expected_cuda = f"{int(expected_cuda) // 10}.{int(expected_cuda) % 10}"
installed_version = torch.__version__.split("+", 1)[0]
sys.exit(
    0 if installed_version == expected_version
    and torch.version.cuda == expected_cuda
    and torch.cuda.is_available()
    else 1
)
PY
  then
    compatible=1
  fi
fi

stale_cuda_packages="$(python - <<'PY'
from importlib.metadata import distributions

names = {
    dist.metadata["Name"]
    for dist in distributions()
    if dist.metadata["Name"]
    and dist.metadata["Name"].lower().startswith("nvidia-")
    and dist.metadata["Name"].lower().endswith("-cu13")
}
print(" ".join(sorted(names)))
PY
)"
if [[ -n "${stale_cuda_packages}" ]]; then
  echo "Removing stale CUDA 13 runtime packages: ${stale_cuda_packages}"
  # Package names cannot contain spaces; intentional word splitting is safe.
  python -m pip uninstall -y ${stale_cuda_packages}
fi

if (( compatible )); then
  echo "Compatible PyTorch CUDA build is already installed."
else
  echo "Replacing incompatible PyTorch packages (for example, cu130) ..."
  python -m pip uninstall -y torch torchvision torchaudio || true
  python -m pip install --no-cache-dir "torch==${TORCH_VERSION}" \
    --index-url "${INDEX_URL}"
fi

TORCH_VERSION="${TORCH_VERSION}" PYTORCH_CUDA="${PYTORCH_CUDA}" python - <<'PY'
import os
import torch
from importlib.metadata import distributions

expected_version = os.environ["TORCH_VERSION"]
expected_cuda_digits = os.environ["PYTORCH_CUDA"].removeprefix("cu")
expected_cuda = f"{int(expected_cuda_digits) // 10}.{int(expected_cuda_digits) % 10}"
installed_version = torch.__version__.split("+", 1)[0]

print("torch", torch.__version__)
print("torch CUDA runtime", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("visible GPUs", torch.cuda.device_count())
if installed_version != expected_version or torch.version.cuda != expected_cuda:
    raise SystemExit(
        f"Expected torch {expected_version} with CUDA {expected_cuda}, "
        f"found {torch.__version__} with CUDA {torch.version.cuda}"
    )
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA remains unavailable after installing the compatible wheel")

stale = sorted(
    dist.metadata["Name"]
    for dist in distributions()
    if dist.metadata["Name"]
    and dist.metadata["Name"].lower().startswith("nvidia-")
    and dist.metadata["Name"].lower().endswith("-cu13")
)
if stale:
    raise SystemExit(f"Stale CUDA 13 packages remain installed: {stale}")

x = torch.arange(1024, device="cuda", dtype=torch.float32)
value = (x.square().mean()).item()
print("GPU smoke tensor", torch.cuda.get_device_name(0), value)
PY

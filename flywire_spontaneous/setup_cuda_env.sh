#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

PY="${PYTHON:-python3.10}"
VENV="${VENV:-.venv-cuda}"
command -v "${PY}" >/dev/null || {
  echo "ERROR: ${PY} not found; set PYTHON=/path/to/python3.10" >&2
  exit 1
}
command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi not found" >&2
  exit 1
}
. ./activate_cuda_toolkit.sh

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader |
  sed -n '1p' | tr -d '[:space:]')"
reported_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p')"
nvcc_release="$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
echo "NVIDIA driver=${driver_version}; nvidia-smi CUDA capability=${reported_cuda:-unknown}; nvcc=${nvcc_release:-unknown}"

"${PY}" -m venv "${VENV}"
. "${VENV}/bin/activate"
python -m pip install -U pip wheel
bash install_torch_cuda.sh
python -m pip install -r requirements_cuda.txt
# Re-check after resolving third-party dependencies so none can replace cu126.
bash install_torch_cuda.sh
python -m pip check
bash setup_critical_init.sh

python - <<'PY'
import brian2
import brian2cuda
import torch
from importlib.metadata import version

print("brian2", brian2.__version__)
print("brian2cuda", version("Brian2Cuda"))
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "gpus", torch.cuda.device_count())
print("torch CUDA runtime", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable in this environment")
brian2cuda.example_run(directory="b2_cuda_example")
PY

echo "CUDA environment ready. Activate with: source ${VENV}/bin/activate"

#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

GPU="${GPU:-0}"
W_BG="${W_BG:-5.0}"
GI="${GI:-0.9}"
BUILD_DIR="${BUILD_DIR:-b2_cuda_smoke_g${GPU}}"

. ./activate_cuda_toolkit.sh
echo "nvcc=$(command -v nvcc)"
nvcc --version
nvcc --list-gpu-code | grep -x "sm_86"
case "${BUILD_DIR}" in
  b2_cuda_smoke_*) rm -rf -- "${BUILD_DIR}" ;;
  *)
    echo "ERROR: smoke BUILD_DIR must start with b2_cuda_smoke_" >&2
    exit 2
    ;;
esac
bash download_data.sh
CUDA_VISIBLE_DEVICES="${GPU}" python run_spontaneous.py --cuda --calibrate \
  --build-dir "${BUILD_DIR}" \
  --t 2 --burn 1 --rec 300 \
  --noise poisson --r-bg 3000 --w-bg "${W_BG}" --gi "${GI}" --seed 0

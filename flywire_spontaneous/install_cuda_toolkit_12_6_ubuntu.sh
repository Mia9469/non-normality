#!/usr/bin/env bash
set -euo pipefail

# Install the CUDA 12.6 toolkit without replacing the working NVIDIA driver.
# The default Conda method needs no sudo and installs into the user's home.
METHOD="${INSTALL_METHOD:-conda}"
TOOLKIT_DIR="${CUDA_TOOLKIT_DIR:-${HOME}/.local/cuda-12.6}"
CUDA_LABEL="${CUDA_LABEL:-cuda-12.6.3}"
CONDA_CHANNEL_ARGS=(
  --override-channels
  --strict-channel-priority
  --channel "nvidia/label/${CUDA_LABEL}"
  --channel nvidia
  --channel conda-forge
)

install_conda() {
  command -v conda >/dev/null || {
    echo "ERROR: conda not found; activate base Conda or set INSTALL_METHOD=apt if sudo is available" >&2
    exit 1
  }
  echo "Installing NVIDIA CUDA toolkit ${CUDA_LABEL} into ${TOOLKIT_DIR} (no sudo)"
  if [[ -d "${TOOLKIT_DIR}/conda-meta" ]]; then
    conda install --yes --prefix "${TOOLKIT_DIR}" \
      "${CONDA_CHANNEL_ARGS[@]}" "cuda-toolkit=12.6.3"
  else
    conda create --yes --prefix "${TOOLKIT_DIR}" \
      "${CONDA_CHANNEL_ARGS[@]}" "cuda-toolkit=12.6.3"
  fi
  test -x "${TOOLKIT_DIR}/bin/nvcc"
  "${TOOLKIT_DIR}/bin/nvcc" --list-gpu-code | grep -qx "sm_86"
}

install_apt() {
  if [[ ! -r /etc/os-release ]]; then
    echo "ERROR: apt method requires Ubuntu" >&2
    exit 1
  fi
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "ERROR: apt method detected ${ID:-unknown}, not Ubuntu" >&2
    exit 1
  fi
  case "${VERSION_ID:-}" in
    20.04|22.04|24.04) repo="ubuntu${VERSION_ID//./}" ;;
    *)
      echo "ERROR: unsupported Ubuntu ${VERSION_ID:-unknown}" >&2
      exit 1
      ;;
  esac

  keyring="/tmp/cuda-keyring_1.1-1_all.deb"
  url="https://developer.download.nvidia.com/compute/cuda/repos/${repo}/x86_64/cuda-keyring_1.1-1_all.deb"
  echo "Installing system CUDA toolkit 12.6 from ${url}"
  wget -O "${keyring}" "${url}"
  sudo dpkg -i "${keyring}"
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-6
}

case "${METHOD}" in
  conda) install_conda ;;
  apt) install_apt ;;
  *)
    echo "ERROR: INSTALL_METHOD must be conda or apt" >&2
    exit 2
    ;;
esac

echo
echo "CUDA toolkit installed. Verify with:"
echo "  source ./activate_cuda_toolkit.sh"
echo "  which nvcc"
echo "  nvcc --version"
echo "  nvcc --list-gpu-code | grep -x sm_86"

#!/usr/bin/env bash

# Source this file before Brian2CUDA commands. It selects a CUDA toolkit that
# can compile RTX 3090 (sm_86) code and is not newer than the installed driver.
_activate_cuda_toolkit() {
  local driver_cuda root nvcc_bin version
  local -a candidates=()

  command -v nvidia-smi >/dev/null || {
    echo "ERROR: nvidia-smi not found" >&2
    return 1
  }
  driver_cuda="$(nvidia-smi |
    sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p')"
  [[ -n "${driver_cuda}" ]] || {
    echo "ERROR: could not determine the driver's CUDA capability" >&2
    return 1
  }

  for root in \
    "${CUDA_TOOLKIT_ROOT:-}" \
    "${CUDA_PATH:-}" \
    "${CUDA_HOME:-}" \
    "${HOME}/.local/cuda-12.6" \
    /usr/local/cuda-12.6 \
    /usr/local/cuda-12 \
    /usr/local/cuda
  do
    if [[ -n "${root}" && -x "${root}/bin/nvcc" ]]; then
      candidates+=("${root}")
    fi
  done
  if command -v nvcc >/dev/null; then
    nvcc_bin="$(readlink -f "$(command -v nvcc)")"
    candidates+=("$(dirname "$(dirname "${nvcc_bin}")")")
  fi

  for root in "${candidates[@]}"; do
    version="$("${root}/bin/nvcc" --version |
      sed -n 's/.*release \([0-9.]*\).*/\1/p')"
    [[ -n "${version}" ]] || continue
    if python3 - "${version}" "${driver_cuda}" <<'PY'
import sys

toolkit = tuple(map(int, sys.argv[1].split(".")[:2]))
driver = tuple(map(int, sys.argv[2].split(".")[:2]))
# CUDA 11.1 introduced Ampere sm_86 compilation support.
raise SystemExit(0 if (11, 1) <= toolkit <= driver else 1)
PY
    then
      if ! "${root}/bin/nvcc" --list-gpu-code 2>/dev/null |
        grep -qx "sm_86"
      then
        continue
      fi
      export CUDA_PATH="${root}"
      export CUDA_HOME="${root}"
      export PATH="${CUDA_PATH}/bin:${PATH}"
      export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${CUDA_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      echo "Using CUDA toolkit ${version} at ${CUDA_PATH} (driver capability ${driver_cuda})"
      return 0
    fi
  done

  echo "ERROR: no compatible nvcc found; RTX 3090 needs sm_86 and this driver supports toolkits through CUDA ${driver_cuda}." >&2
  echo "Install CUDA 12.6 with: bash install_cuda_toolkit_12_6_ubuntu.sh" >&2
  return 1
}

if ! _activate_cuda_toolkit; then
  unset -f _activate_cuda_toolkit
  return 1 2>/dev/null || exit 1
fi
unset -f _activate_cuda_toolkit

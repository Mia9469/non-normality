#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

R_BG="${R_BG:-3000}"
R_INH="${R_INH:-}"
BG_SOURCES="${BG_SOURCES:-100}"
PAIRS="${PAIRS:-3:0.70 4:0.80 5:0.90 6:0.90 8:0.90}"
CONTROLS="${CONTROLS:-original}"
T="${T:-10}"
BURN="${BURN:-2}"
REC="${REC:-1000}"
SEED="${SEED:-0}"
BUILD_ROOT="${BUILD_ROOT:-b2_calibration}"
SIM_BACKEND="${SIM_BACKEND:-cpp}"
ORIGINAL_GAIN="${ORIGINAL_GAIN:-1.0}"
SYMMETRIZED_GAIN="${SYMMETRIZED_GAIN:-1.0}"

bash download_data.sh
case "${SIM_BACKEND}" in
  cpp) backend_args=(--cpp) ;;
  cuda)
    . ./activate_cuda_toolkit.sh
    backend_args=(--cuda)
    ;;
  runtime) backend_args=() ;;
  *) echo "SIM_BACKEND must be cpp, cuda, or runtime" >&2; exit 2 ;;
esac

for control in ${CONTROLS}; do
  case "${control}" in
    original) recurrent_gain="${ORIGINAL_GAIN}" ;;
    symmetrized) recurrent_gain="${SYMMETRIZED_GAIN}" ;;
    *) echo "unknown connectome control: ${control}" >&2; exit 2 ;;
  esac
  for pair in ${PAIRS}; do
    w_bg="${pair%%:*}"
    gi="${pair##*:}"
    tag="r${R_BG}_w${w_bg}_gi${gi}_${control}"
    echo
    echo "===== calibration ${tag} ====="
    args=(
      "${backend_args[@]}" --calibrate --build-dir "${BUILD_ROOT}_${tag}"
      --t "${T}" --burn "${BURN}" --rec "${REC}"
      --noise poisson --r-bg "${R_BG}" --bg-sources "${BG_SOURCES}"
      --w-bg "${w_bg}" --gi "${gi}" --connectome-control "${control}"
      --recurrent-gain "${recurrent_gain}"
      --seed "${SEED}"
    )
    if [[ -n "${R_INH}" ]]; then
      args+=(--r-inh "${R_INH}")
    fi
    if ! python run_spontaneous.py "${args[@]}"; then
      echo "calibration failed for ${tag}; continuing parameter sweep" >&2
    fi
  done
done

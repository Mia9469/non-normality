#!/usr/bin/env bash
set -euo pipefail

# Match the symmetric null's spontaneous operating point using one global
# recurrent-weight multiplier. This preserves exact matrix symmetry.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

GPU="${GPU:-0}"
GAINS="${GAINS:-0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90}"
R_BG="${R_BG:-3000}"
R_INH="${R_INH:-}"
W_BG="${W_BG:-5.0}"
GI="${GI:-0.9}"
BG_SOURCES="${BG_SOURCES:-100}"
T="${T:-20}"
BURN="${BURN:-5}"
REC="${REC:-3000}"
SEED="${SEED:-0}"
BUILD_ROOT="${BUILD_ROOT:-b2_cuda_gain_calibration}"
LOG="${LOG:-recurrent_gain_calibration_w${W_BG}_gi${GI}_s${SEED}.log}"

. ./activate_cuda_toolkit.sh
bash download_data.sh
exec > >(tee "${LOG}") 2>&1

run_gain() {
  local control="$1" gain="$2" tag
  local -a args
  tag="${control}_rg${gain}_w${W_BG}_gi${GI}_s${SEED}"
  echo
  echo "===== recurrent-gain calibration: ${tag} ====="
  args=(
    --cuda --calibrate --build-dir "${BUILD_ROOT}_${tag}"
    --t "${T}" --burn "${BURN}" --rec "${REC}"
    --noise poisson --r-bg "${R_BG}" --bg-sources "${BG_SOURCES}"
    --w-bg "${W_BG}" --gi "${GI}" --connectome-control "${control}"
    --recurrent-gain "${gain}" --seed "${SEED}"
  )
  if [[ -n "${R_INH}" ]]; then
    args+=(--r-inh "${R_INH}")
  fi
  if ! CUDA_VISIBLE_DEVICES="${GPU}" python run_spontaneous.py "${args[@]}"; then
    echo "gain calibration failed for ${tag}; continuing sweep" >&2
  fi
}

run_gain original 1.0
for gain in ${GAINS}; do
  run_gain symmetrized "${gain}"
done
echo "Saved calibration log: ${LOG}"

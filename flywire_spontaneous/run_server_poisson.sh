#!/usr/bin/env bash
set -euo pipefail

# Calibrate W_BG/GI with short runs before launching this script.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

R_BG="${R_BG:-3000}"
R_INH="${R_INH:-}"
W_BG="${W_BG:-5.0}"
GI="${GI:-0.9}"
BG_SOURCES="${BG_SOURCES:-100}"
CONTROLS="${CONTROLS:-original symmetrized}"
ORIGINAL_GAIN="${ORIGINAL_GAIN:-1.0}"
SYMMETRIZED_GAIN="${SYMMETRIZED_GAIN:-}"
TAG="${TAG:-r${R_BG}_w${W_BG}_gi${GI}}"
DURATION="${DURATION:-1800}"
BURN="${BURN:-60}"
REC="${REC:-3000}"
SEEDS="${SEEDS:-0 1 2}"
BUILD_ROOT="${BUILD_ROOT:-b2_build}"
RATE_MIN="${RATE_MIN:-0.5}"
RATE_MAX="${RATE_MAX:-8.0}"
MEDIAN_MIN="${MEDIAN_MIN:-1.0}"
MEDIAN_MAX="${MEDIAN_MAX:-3.0}"
P95_MAX="${P95_MAX:-50.0}"
SILENT_MAX="${SILENT_MAX:-0.3}"
BLOCK_CV_MAX="${BLOCK_CV_MAX:-0.1}"
PREFLIGHT_T="${PREFLIGHT_T:-10}"
PREFLIGHT_REC="${PREFLIGHT_REC:-1000}"
SIM_BACKEND="${SIM_BACKEND:-cpp}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-auto}"

bash download_data.sh
bash setup_critical_init.sh

background_args=(
  --noise poisson --r-bg "${R_BG}" --bg-sources "${BG_SOURCES}"
  --w-bg "${W_BG}" --gi "${GI}"
)
case "${SIM_BACKEND}" in
  cpp) backend_args=(--cpp) ;;
  cuda)
    . ./activate_cuda_toolkit.sh
    backend_args=(--cuda)
    ;;
  runtime) backend_args=() ;;
  *) echo "SIM_BACKEND must be cpp, cuda, or runtime" >&2; exit 2 ;;
esac
if [[ -n "${R_INH}" ]]; then
  background_args+=(--r-inh "${R_INH}")
fi

gain_for_control() {
  case "$1" in
    original) printf '%s\n' "${ORIGINAL_GAIN}" ;;
    symmetrized)
      if [[ -z "${SYMMETRIZED_GAIN}" ]]; then
        echo "ERROR: set SYMMETRIZED_GAIN after running calibrate_recurrent_gain.sh" >&2
        return 1
      fi
      printf '%s\n' "${SYMMETRIZED_GAIN}"
      ;;
    *) echo "ERROR: unknown connectome control: $1" >&2; return 1 ;;
  esac
}
for control in ${CONTROLS}; do
  gain_for_control "${control}" >/dev/null
done

read -r first_seed _ <<< "${SEEDS}"
summaries=()
for control in ${CONTROLS}; do
  gain="$(gain_for_control "${control}")"
  control_tag="${TAG}_${control}_rg${gain}"
  python run_spontaneous.py "${backend_args[@]}" --calibrate \
    --cpp-dir "${BUILD_ROOT}_${control_tag}_preflight" \
    --t "${PREFLIGHT_T}" --burn 2 --rec "${PREFLIGHT_REC}" \
    "${background_args[@]}" --connectome-control "${control}" \
    --recurrent-gain "${gain}" \
    --seed "${first_seed}" --accept-rate-min "${RATE_MIN}" \
    --accept-rate-max "${RATE_MAX}" \
    --accept-median-rate-min "${MEDIAN_MIN}" \
    --accept-median-rate-max "${MEDIAN_MAX}" \
    --accept-p95-rate-max "${P95_MAX}" \
    --accept-silent-max "${SILENT_MAX}" --accept-block-cv-max "${BLOCK_CV_MAX}"

  first=1
  analyses=()
  for seed in ${SEEDS}; do
    output="spont_poisson_${control_tag}_s${seed}.npz"
    analysis="spont_poisson_${control_tag}_s${seed}_analysis.json"
    analyses+=("${analysis}")
    python run_spontaneous.py "${backend_args[@]}" \
      --cpp-dir "${BUILD_ROOT}_${control_tag}_s${seed}" \
      --t "${DURATION}" --burn "${BURN}" \
      "${background_args[@]}" --connectome-control "${control}" \
      --recurrent-gain "${gain}" \
      --rec "${REC}" --bin 45.454545 --seed "${seed}" --out "${output}" \
      --accept-rate-min "${RATE_MIN}" --accept-rate-max "${RATE_MAX}" \
      --accept-median-rate-min "${MEDIAN_MIN}" \
      --accept-median-rate-max "${MEDIAN_MAX}" \
      --accept-p95-rate-max "${P95_MAX}" \
      --accept-silent-max "${SILENT_MAX}" --accept-block-cv-max "${BLOCK_CV_MAX}"
    if [[ "${first}" -eq 1 ]]; then
      python analyze_spontaneous.py \
        --in "${output}" \
        --con 2023_03_23_connectivity_630_final.parquet \
        --out "${analysis}" --device "${ANALYSIS_DEVICE}"
      first=0
    else
      python analyze_spontaneous.py \
        --in "${output}" --skip-structure --out "${analysis}" \
        --device "${ANALYSIS_DEVICE}"
    fi
  done

  summary="spont_poisson_${control_tag}_summary.json"
  summaries+=("${summary}")
  python aggregate_analyses.py "${analyses[@]}" --out "${summary}"
done

if [[ "${#summaries[@]}" -eq 2 ]]; then
  python compare_controls.py "${summaries[@]}" \
    --out "spont_poisson_${TAG}_original_rg${ORIGINAL_GAIN}_symmetrized_rg${SYMMETRIZED_GAIN}_control_comparison.json"
fi

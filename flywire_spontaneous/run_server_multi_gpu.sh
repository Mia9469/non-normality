#!/usr/bin/env bash
set -euo pipefail

# Parallelize independent whole-brain runs across GPUs. One simulation remains
# single-GPU; seeds and connectome controls are the parallel unit.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
MAX_JOBS="${MAX_JOBS:-4}"
SIM_BACKEND="${SIM_BACKEND:-cuda}"
R_BG="${R_BG:-3000}"
R_INH="${R_INH:-}"
W_BG="${W_BG:-5.0}"
GI="${GI:-0.9}"
BG_SOURCES="${BG_SOURCES:-100}"
CONTROLS="${CONTROLS:-original symmetrized}"
ORIGINAL_GAIN="${ORIGINAL_GAIN:-1.0}"
SYMMETRIZED_GAIN="${SYMMETRIZED_GAIN:-}"
SEEDS="${SEEDS:-0 1 2}"
TAG="${TAG:-r${R_BG}_w${W_BG}_gi${GI}}"
DURATION="${DURATION:-1800}"
BURN="${BURN:-60}"
REC="${REC:-3000}"
BUILD_ROOT="${BUILD_ROOT:-b2_cuda}"
LOG_ROOT="${LOG_ROOT:-logs_${TAG}}"
RATE_MIN="${RATE_MIN:-0.5}"
RATE_MAX="${RATE_MAX:-8.0}"
MEDIAN_MIN="${MEDIAN_MIN:-1.0}"
MEDIAN_MAX="${MEDIAN_MAX:-3.0}"
P95_MAX="${P95_MAX:-50.0}"
SILENT_MAX="${SILENT_MAX:-0.3}"
BLOCK_CV_MAX="${BLOCK_CV_MAX:-0.1}"
PREFLIGHT_T="${PREFLIGHT_T:-10}"
PREFLIGHT_REC="${PREFLIGHT_REC:-1000}"
PREFLIGHT_BURN="${PREFLIGHT_BURN:-2}"  # raise for high-gain runs with slow transients

command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi not found" >&2
  exit 1
}
python -c "import torch; assert torch.cuda.is_available()"
case "${SIM_BACKEND}" in
  cuda)
    . ./activate_cuda_toolkit.sh
    python -c "import brian2cuda"
    simulation_backend_args=(--cuda)
    ;;
  cpp)
    simulation_backend_args=(--cpp)
    ;;
  *)
    echo "SIM_BACKEND must be cuda or cpp" >&2
    exit 2
    ;;
esac
bash download_data.sh
bash setup_critical_init.sh
mkdir -p "${LOG_ROOT}"

read -r -a gpu_ids <<< "${GPU_IDS}"
read -r -a seeds <<< "${SEEDS}"
read -r -a controls <<< "${CONTROLS}"
if [[ ! "${MAX_JOBS}" =~ ^[0-9]+$ ]]; then
  echo "MAX_JOBS must be an integer" >&2
  exit 2
fi
if (( ${#gpu_ids[@]} == 0 || MAX_JOBS < 1 || MAX_JOBS > ${#gpu_ids[@]} )); then
  echo "MAX_JOBS must be between 1 and the number of GPU_IDS" >&2
  exit 2
fi
first_seed="${seeds[0]}"

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
for control in "${controls[@]}"; do
  gain_for_control "${control}" >/dev/null
done

background_args=(
  --noise poisson --r-bg "${R_BG}" --bg-sources "${BG_SOURCES}"
  --w-bg "${W_BG}" --gi "${GI}"
)
if [[ -n "${R_INH}" ]]; then
  background_args+=(--r-inh "${R_INH}")
fi
acceptance_args=(
  --accept-rate-min "${RATE_MIN}" --accept-rate-max "${RATE_MAX}"
  --accept-median-rate-min "${MEDIAN_MIN}"
  --accept-median-rate-max "${MEDIAN_MAX}"
  --accept-p95-rate-max "${P95_MAX}"
  --accept-silent-max "${SILENT_MAX}"
  --accept-block-cv-max "${BLOCK_CV_MAX}"
)

echo "Preflight: one short CUDA run per connectome control"
for index in "${!controls[@]}"; do
  control="${controls[index]}"
  gain="$(gain_for_control "${control}")"
  gpu="${gpu_ids[$((index % ${#gpu_ids[@]}))]}"
  control_tag="${TAG}_${control}_rg${gain}"
  CUDA_VISIBLE_DEVICES="${gpu}" python run_spontaneous.py \
    "${simulation_backend_args[@]}" --calibrate \
    --build-dir "${BUILD_ROOT}_${control_tag}_preflight" \
    --t "${PREFLIGHT_T}" --burn "${PREFLIGHT_BURN}" --rec "${PREFLIGHT_REC}" \
    "${background_args[@]}" "${acceptance_args[@]}" \
    --connectome-control "${control}" --recurrent-gain "${gain}" \
    --seed "${first_seed}" \
    2>&1 | tee "${LOG_ROOT}/${control_tag}_preflight.log"
done

jobs=()
for control in "${controls[@]}"; do
  for seed in "${seeds[@]}"; do
    jobs+=("${control}:${seed}")
  done
done

wait_batch() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "At least one parallel job failed; inspect ${LOG_ROOT}" >&2
    exit 1
  fi
}

run_simulation() {
  local control="$1" seed="$2" gpu="$3"
  local gain control_tag
  gain="$(gain_for_control "${control}")"
  control_tag="${TAG}_${control}_rg${gain}"
  local output="spont_poisson_${control_tag}_s${seed}.npz"
  CUDA_VISIBLE_DEVICES="${gpu}" python run_spontaneous.py \
    "${simulation_backend_args[@]}" \
    --build-dir "${BUILD_ROOT}_${control_tag}_s${seed}" \
    --t "${DURATION}" --burn "${BURN}" --rec "${REC}" --bin 45.454545 \
    "${background_args[@]}" "${acceptance_args[@]}" \
    --connectome-control "${control}" --recurrent-gain "${gain}" \
    --seed "${seed}" --out "${output}" \
    >"${LOG_ROOT}/${control_tag}_s${seed}_simulation.log" 2>&1
}

run_analysis() {
  local control="$1" seed="$2" gpu="$3"
  local gain control_tag
  gain="$(gain_for_control "${control}")"
  control_tag="${TAG}_${control}_rg${gain}"
  local output="spont_poisson_${control_tag}_s${seed}.npz"
  local analysis="spont_poisson_${control_tag}_s${seed}_analysis.json"
  local structure_args=()
  if [[ "${seed}" != "${first_seed}" ]]; then
    structure_args=(--skip-structure)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python analyze_spontaneous.py \
    --device cuda --in "${output}" --out "${analysis}" "${structure_args[@]}" \
    >"${LOG_ROOT}/${control_tag}_s${seed}_analysis.log" 2>&1
}

run_phase() {
  local phase="$1"
  local pids=()
  local index token control seed gpu
  echo "Starting ${phase} phase with MAX_JOBS=${MAX_JOBS}, GPUs=${GPU_IDS}"
  for index in "${!jobs[@]}"; do
    token="${jobs[index]}"
    control="${token%%:*}"
    seed="${token##*:}"
    gpu="${gpu_ids[$((index % ${#gpu_ids[@]}))]}"
    echo "${phase}: control=${control} seed=${seed} gpu=${gpu}"
    if [[ "${phase}" == "simulation" ]]; then
      run_simulation "${control}" "${seed}" "${gpu}" &
    else
      run_analysis "${control}" "${seed}" "${gpu}" &
    fi
    pids+=("$!")
    if (( ${#pids[@]} == MAX_JOBS )); then
      wait_batch "${pids[@]}"
      pids=()
    fi
  done
  if (( ${#pids[@]} )); then
    wait_batch "${pids[@]}"
  fi
}

run_phase simulation
run_phase analysis

summaries=()
for control in "${controls[@]}"; do
  gain="$(gain_for_control "${control}")"
  control_tag="${TAG}_${control}_rg${gain}"
  analyses=()
  for seed in "${seeds[@]}"; do
    analyses+=("spont_poisson_${control_tag}_s${seed}_analysis.json")
  done
  summary="spont_poisson_${control_tag}_summary.json"
  summaries+=("${summary}")
  python aggregate_analyses.py "${analyses[@]}" --out "${summary}"
done
if [[ "${#summaries[@]}" -eq 2 ]]; then
  python compare_controls.py "${summaries[@]}" \
    --out "spont_poisson_${TAG}_original_rg${ORIGINAL_GAIN}_symmetrized_rg${SYMMETRIZED_GAIN}_control_comparison.json"
fi

echo "Completed multi-GPU workflow; logs are in ${LOG_ROOT}"

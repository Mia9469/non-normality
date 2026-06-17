#!/usr/bin/env bash
set -euo pipefail

# Recurrence-dominance calibration grid.
#
# Walks the network from background-dominated -> recurrence-dominated by
# sweeping recurrent_gain UP and w_bg (background fluctuation amplitude) DOWN,
# for BOTH connectome controls. recurrent_gain=0 gives the background-only rate
# floor at each w_bg, so "recurrence excess" = rate(gain)/rate(0) is measurable.
#
# Short --calibrate runs (no save, no acceptance guards -> every cell reports).
# Read the table from parse_sweep.py, pick matched-rate cells that span the
# recurrence axis, then run production with run_server_multi_gpu.sh
# (ORIGINAL_GAIN/SYMMETRIZED_GAIN + W_BG=...).
#
# Parallel unit = one (control, w_bg, gain) cell; cells fan out across GPU_IDS.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

GPU_IDS="${GPU_IDS:-0 6 7}"
MAX_JOBS="${MAX_JOBS:-3}"
# recurrence axis: 0 = background-only floor, then an amplification ladder.
# symmetrized couples ~4x harder (denser, eta=1) so it uses a lower ladder.
ORIG_GAINS="${ORIG_GAINS:-0 1 2 4 8}"
SYM_GAINS="${SYM_GAINS:-0 0.25 0.5 1 2}"
W_BGS="${W_BGS:-5 2 1}"                 # background fluctuation ladder (mV in w_syn)
R_BG="${R_BG:-3000}"
GI="${GI:-0.9}"
BG_SOURCES="${BG_SOURCES:-100}"
SEED="${SEED:-0}"
T="${T:-30}"
BURN="${BURN:-8}"
REC="${REC:-2000}"
BUILD_ROOT="${BUILD_ROOT:-b2_cuda_recsweep}"
LOG_ROOT="${LOG_ROOT:-logs_recsweep_s${SEED}}"

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found" >&2; exit 1; }
. ./activate_cuda_toolkit.sh
python -c "import brian2cuda"
bash download_data.sh
mkdir -p "${LOG_ROOT}"

read -r -a gpu_ids <<< "${GPU_IDS}"
(( ${#gpu_ids[@]} >= 1 )) || { echo "need at least one GPU" >&2; exit 2; }
(( MAX_JOBS >= 1 && MAX_JOBS <= ${#gpu_ids[@]} )) || {
  echo "MAX_JOBS must be between 1 and number of GPU_IDS" >&2; exit 2; }

gains_for_control() {
  case "$1" in
    original)    printf '%s' "${ORIG_GAINS}" ;;
    symmetrized) printf '%s'  "${SYM_GAINS}" ;;
    *) echo "unknown control: $1" >&2; return 1 ;;
  esac
}

# Build the (control, w_bg, gain) job list.
jobs=()
for control in original symmetrized; do
  for wbg in ${W_BGS}; do
    for gain in $(gains_for_control "${control}"); do
      jobs+=("${control}:${wbg}:${gain}")
    done
  done
done
echo "Recurrence sweep: ${#jobs[@]} cells across GPUs [${GPU_IDS}], MAX_JOBS=${MAX_JOBS}"

run_cell() {
  local control="$1" wbg="$2" gain="$3" gpu="$4"
  local tag="${control}_w${wbg}_rg${gain}"
  CUDA_VISIBLE_DEVICES="${gpu}" python run_spontaneous.py \
    --cuda --calibrate --build-dir "${BUILD_ROOT}_${tag}" \
    --t "${T}" --burn "${BURN}" --rec "${REC}" \
    --noise poisson --r-bg "${R_BG}" --bg-sources "${BG_SOURCES}" \
    --w-bg "${wbg}" --gi "${GI}" \
    --connectome-control "${control}" --recurrent-gain "${gain}" \
    --seed "${SEED}" \
    >"${LOG_ROOT}/${tag}.log" 2>&1 \
    && echo "ok   ${tag} (gpu ${gpu})" \
    || echo "FAIL ${tag} (gpu ${gpu}) -- see ${LOG_ROOT}/${tag}.log" >&2
}

pids=()
for i in "${!jobs[@]}"; do
  token="${jobs[i]}"; control="${token%%:*}"; rest="${token#*:}"
  wbg="${rest%%:*}"; gain="${rest##*:}"
  gpu="${gpu_ids[$((i % ${#gpu_ids[@]}))]}"
  run_cell "${control}" "${wbg}" "${gain}" "${gpu}" &
  pids+=("$!")
  if (( ${#pids[@]} == MAX_JOBS )); then
    wait "${pids[@]}" || true
    pids=()
  fi
done
(( ${#pids[@]} )) && { wait "${pids[@]}" || true; }

echo
echo "=== sweep done; building table ==="
python parse_sweep.py "${LOG_ROOT}" --out "${LOG_ROOT}/recurrence_sweep_table.md"
echo "Table: ${LOG_ROOT}/recurrence_sweep_table.md"

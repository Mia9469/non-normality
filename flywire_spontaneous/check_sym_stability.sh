#!/usr/bin/env bash
set -euo pipefail

# Map where the symmetric null is graded-and-reproducible vs multistable/runaway.
# Same (control, w_bg, gain) is run across several seeds; large seed-to-seed
# rate spread = multistability (the symmetric operator's real leading eigenvalue
# drives a sharp runaway/bistability threshold the asymmetric net lacks).
# Short --calibrate runs (no save, no acceptance guards). REC matches production
# so the mean is measured on the same sampling as the real runs.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

GPU_IDS="${GPU_IDS:-0 6 7}"
MAX_JOBS="${MAX_JOBS:-3}"
SYM_GAINS="${SYM_GAINS:-0.30 0.35 0.40 0.50 0.60}"
ORIG_GAINS="${ORIG_GAINS:-2 4}"          # reference points to match against
SEEDS="${SEEDS:-0 1 2}"
W_BG="${W_BG:-5}"
R_BG="${R_BG:-3000}"
GI="${GI:-0.9}"
BG_SOURCES="${BG_SOURCES:-100}"
T="${T:-40}"
BURN="${BURN:-10}"
REC="${REC:-3000}"
BUILD_ROOT="${BUILD_ROOT:-b2_cuda_symstab}"
LOG_ROOT="${LOG_ROOT:-logs_symstab}"

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found" >&2; exit 1; }
. ./activate_cuda_toolkit.sh
python -c "import brian2cuda"
bash download_data.sh
mkdir -p "${LOG_ROOT}"

read -r -a gpu_ids <<< "${GPU_IDS}"
(( MAX_JOBS >= 1 && MAX_JOBS <= ${#gpu_ids[@]} )) || {
  echo "MAX_JOBS must be between 1 and number of GPU_IDS" >&2; exit 2; }

jobs=()
for seed in ${SEEDS}; do
  for gain in ${SYM_GAINS}; do jobs+=("symmetrized:${gain}:${seed}"); done
  for gain in ${ORIG_GAINS}; do jobs+=("original:${gain}:${seed}"); done
done
echo "Stability check: ${#jobs[@]} runs across GPUs [${GPU_IDS}]"

run_cell() {
  local control="$1" gain="$2" seed="$3" gpu="$4"
  local tag="${control}_rg${gain}_s${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" python run_spontaneous.py \
    --cuda --calibrate --build-dir "${BUILD_ROOT}_${tag}" \
    --t "${T}" --burn "${BURN}" --rec "${REC}" \
    --noise poisson --r-bg "${R_BG}" --bg-sources "${BG_SOURCES}" \
    --w-bg "${W_BG}" --gi "${GI}" \
    --connectome-control "${control}" --recurrent-gain "${gain}" \
    --seed "${seed}" >"${LOG_ROOT}/${tag}.log" 2>&1 \
    && echo "ok   ${tag} (gpu ${gpu})" \
    || echo "FAIL ${tag} (gpu ${gpu})" >&2
}

pids=()
for i in "${!jobs[@]}"; do
  t="${jobs[i]}"; control="${t%%:*}"; rest="${t#*:}"; gain="${rest%%:*}"; seed="${rest##*:}"
  gpu="${gpu_ids[$((i % ${#gpu_ids[@]}))]}"
  run_cell "${control}" "${gain}" "${seed}" "${gpu}" &
  pids+=("$!")
  (( ${#pids[@]} == MAX_JOBS )) && { wait "${pids[@]}" || true; pids=(); }
done
(( ${#pids[@]} )) && { wait "${pids[@]}" || true; }

echo
echo "=== rate vs gain vs seed (spread across seeds = multistability) ==="
printf "%-13s %6s %5s %9s %9s %7s\n" control gain seed meanHz medianHz silent
for f in "${LOG_ROOT}"/*.log; do
  base="$(basename "${f}" .log)"
  control="${base%%_rg*}"; rest="${base#*_rg}"; gain="${rest%%_s*}"; seed="${rest##*_s}"
  line="$(grep -m1 'mean rate=' "${f}" || true)"
  pct="$(grep -m1 'percentiles' "${f}" || true)"
  mean="$(sed -E 's/.*mean rate=([0-9.]+)Hz.*/\1/' <<<"${line}")"
  silent="$(sed -E 's/.*silent fraction=([0-9.]+).*/\1/' <<<"${line}")"
  median="$(sed -E 's/.*%=\[ *[0-9.eE+-]+ +[0-9.eE+-]+ +([0-9.eE+-]+).*/\1/' <<<"${pct}")"
  printf "%-13s %6s %5s %9s %9s %7s\n" "${control}" "${gain}" "${seed}" "${mean:-NA}" "${median:-NA}" "${silent:-NA}"
done | sort -k1,1 -k2,2n -k3,3n

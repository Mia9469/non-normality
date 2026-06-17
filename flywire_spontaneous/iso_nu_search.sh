#!/usr/bin/env bash
set -euo pipefail

# iso-nu search: scan ONE connectome control over a recurrent_gain ladder,
# running the FULL pipeline (sim -> official SVCA2/DMD analysis) per gain, so we
# can find the gain whose functional nu MATCHES the other control's nu. Then we
# ask: at matched nu, do the symmetry-SPECIFIC readouts (DMD rotation,
# max|Im lambda|, kappa(V)) also coincide? If yes -> an eta=-0.04 vs eta=+1 pair
# that is functionally indistinguishable = empirical degeneracy.
#
# Defaults scan symmetrized UP toward the asymmetric net's nu (orig rg2 ~0.234,
# orig rg4 ~higher). Reuse existing A/B1/B2 analyses for the original control;
# only the missing higher-gain symmetrized points need new runs.
#
# Parallel unit = one gain cell, fanned across GPU_IDS (default the free 2 3 4 5).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

GPU_IDS="${GPU_IDS:-2 3 4 5}"
MAX_JOBS="${MAX_JOBS:-4}"
CONTROL="${CONTROL:-symmetrized}"
GAINS="${GAINS:-0.6 0.7 0.85 1.0}"
R_BG="${R_BG:-3000}"
GI="${GI:-0.9}"
W_BG="${W_BG:-5}"
BG_SOURCES="${BG_SOURCES:-100}"
SEED="${SEED:-0}"
T="${T:-1800}"                 # match A/B1/B2 so nu is comparable
BURN="${BURN:-60}"
REC="${REC:-3000}"
BLOCK_CV_MAX="${BLOCK_CV_MAX:-0.1}"
BUILD_ROOT="${BUILD_ROOT:-b2_cuda_isonu}"
LOG_ROOT="${LOG_ROOT:-logs_isonu_${CONTROL}}"

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found" >&2; exit 1; }
. ./activate_cuda_toolkit.sh
python -c "import brian2cuda"
bash download_data.sh
bash setup_critical_init.sh
mkdir -p "${LOG_ROOT}"

read -r -a gpu_ids <<< "${GPU_IDS}"
(( ${#gpu_ids[@]} >= 1 )) || { echo "need >=1 GPU" >&2; exit 2; }
(( MAX_JOBS >= 1 && MAX_JOBS <= ${#gpu_ids[@]} )) || {
  echo "MAX_JOBS must be 1..#GPU_IDS" >&2; exit 2; }

run_job() {
  local gain="$1" gpu="$2"
  local tag="iso_${CONTROL}_rg${gain}_s${SEED}"
  local npz="${tag}.npz" js="${tag}_analysis.json"
  {
    CUDA_VISIBLE_DEVICES="${gpu}" python run_spontaneous.py \
      --cuda --build-dir "${BUILD_ROOT}_${tag}" \
      --t "${T}" --burn "${BURN}" --rec "${REC}" --bin 45.454545 \
      --noise poisson --r-bg "${R_BG}" --bg-sources "${BG_SOURCES}" \
      --w-bg "${W_BG}" --gi "${GI}" \
      --connectome-control "${CONTROL}" --recurrent-gain "${gain}" \
      --seed "${SEED}" --accept-block-cv-max "${BLOCK_CV_MAX}" --out "${npz}" \
    && CUDA_VISIBLE_DEVICES="${gpu}" python analyze_spontaneous.py \
      --device cuda --in "${npz}" --out "${js}" --skip-structure
  } >"${LOG_ROOT}/${tag}.log" 2>&1 \
    && echo "ok   ${tag} (gpu ${gpu})" \
    || echo "FAIL ${tag} (gpu ${gpu}) -- see ${LOG_ROOT}/${tag}.log" >&2
}

pids=()
i=0
for gain in ${GAINS}; do
  gpu="${gpu_ids[$((i % ${#gpu_ids[@]}))]}"
  run_job "${gain}" "${gpu}" &
  pids+=("$!"); i=$((i + 1))
  if (( ${#pids[@]} == MAX_JOBS )); then
    wait "${pids[@]}" || true; pids=()
  fi
done
(( ${#pids[@]} )) && { wait "${pids[@]}" || true; }

echo
echo "=== iso-nu scan done; building table (include existing analyses too) ==="
python parse_iso_nu.py iso_*_analysis.json *_analysis.json --out iso_nu_table.md || true
echo "Table: iso_nu_table.md"

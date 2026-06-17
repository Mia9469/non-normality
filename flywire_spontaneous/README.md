# FlyWire/Shiu spontaneous-activity forward-model calibration

This directory runs the Shiu et al. (2024) whole-brain LIF model on the
FlyWire-derived connectome, then applies the public Pachitariu et al. analysis
code. It is a forward-model calibration, not an equality test between a
structural and functional reciprocity scalar.

Three objects must remain distinct:

1. the signed synaptic matrix `W`;
2. the nonlinear LIF dynamics with filtering, delay, threshold and reset;
3. the z-scored, partially observed, PCA-reduced DMD propagator.

The valid test is whether the functional pipeline changes between the original
directed connectome and an explicitly defined directionality null. Agreement or
disagreement between `eta(W)` and a DMD-coordinate `eta` is not a recovery
criterion.

## Setup

Run on the simulation server:

```bash
bash setup_env.sh
source .venv/bin/activate
bash download_data.sh
```

`setup_critical_init.sh` clones the official
[`MouseLand/critical_init`](https://github.com/MouseLand/critical_init) repository
and pins commit `2c15edf4e770165fc3962dcc3920c8bcaf555bed`.
`analyze_spontaneous.py` directly calls its `powerlaw.SVCA2`,
`powerlaw.fit_powerlaw_exp`, `powerlaw.compute_evals`, and `lyapun.dmd`.

## Background input and calibration

The released model is silent without input. The primary condition adds
independent excitatory and inhibitory Poisson synaptic streams to every neuron:

- `--r-bg` and `--r-inh` set their total event rates;
- `--gi` sets inhibitory/excitatory event-weight ratio and mainly controls net
  mean drive;
- `--w-bg` scales both event weights and mainly controls fluctuation size.

This is a balanced-E/I-like operating-point model, not a claim about the fly's
unknown in-vivo spontaneous drive. The same released recurrent weights, time
constants, threshold, reset and delay are retained.

The old `w_bg=1.5, gi=0.9` setting has an isolated-neuron threshold distance of
about `9.4 SD` and is expected to be silent. Start with the analytical grid:

```bash
bash calibrate_poisson.sh
```

The default pairs span threshold distances of roughly `2.7` to `1.0 SD`.
Select a pair only after inspecting mean rate, per-neuron percentiles, silent
fraction, Fano factor and ten-block drift. Do not force the population mean into
1--3 Hz when a stable high-rate tail pulls it above the typical-neuron rate.
The default long-run acceptance window instead requires median neuron rate
1--3 Hz, mean rate 0.5--8 Hz, P95 below 50 Hz, silent fraction below 0.3 and
block-rate CV below 0.1. These are operating-point checks, not fitting targets
for the exponent.

The printed pooled binned Fano mixes temporal variability with differences in
mean rate across neurons. Use the median active-neuron Fano as the cleaner
single-neuron irregularity diagnostic.

To lower the rate while approximately preserving fluctuation size, hold
`W_BG` fixed and raise `GI` in small increments:

```bash
PAIRS="5:0.90 5:0.92 5:0.94 5:0.96" bash calibrate_poisson.sh
```

Retain the original stable operating point and one lower-rate point as a
robustness comparison rather than tuning to one exact mean.

To calibrate both graph conditions:

```bash
CONTROLS="original symmetrized" bash calibrate_poisson.sh
```

## Directionality control

`--connectome-control original` uses the released signed FlyWire graph.
`--connectome-control symmetrized` uses
`W_sym=(W+W.T)/2` as a mathematical symmetry null.

The symmetric null changes reciprocal support and can violate Dale's law. It is
therefore useful only for asking whether the functional pipeline is sensitive
to directed interactions; it is not a second biological connectome. Unscaled
`W_sym` can also have a very different spectral radius and firing-rate regime
despite preserving the global signed-weight sum. Comparing that run directly
would confound directionality with operating point.

Use one global recurrent multiplier to match the symmetric null to the original
graph's spontaneous regime. A scalar multiplier preserves exact symmetry,
whereas neuron-wise row or column normalization would not:

```bash
GPU=0 GAINS="0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90" \
T=20 BURN=5 REC=3000 bash calibrate_recurrent_gain.sh
```

Choose a gain that approximately matches the original mean, median, P95,
silent fraction and Fano factor without using SVCA2 or DMD results. Report the
selected scalar and residual rate differences. The unmatched `gain=1`
symmetric run remains a stress test, not the primary directionality control.

## Long run

After calibration, launch three seeds of both graph conditions:

```bash
W_BG=5.0 GI=0.9 SYMMETRIZED_GAIN=<calibrated_gain> \
bash run_server_poisson.sh
```

The script performs a short preflight for each condition before any 30-minute
run, rejects out-of-range or overly silent simulations, analyzes each seed, and
writes a separate summary per graph condition plus a direct functional-metric
contrast. Configure it with `R_BG`,
`R_INH`, `W_BG`, `GI`, `BG_SOURCES`, `ORIGINAL_GAIN`, `SYMMETRIZED_GAIN`,
`CONTROLS`, `DURATION`, `REC`, `SEEDS`,
`RATE_MIN`, `RATE_MAX`, `MEDIAN_MIN`, `MEDIAN_MAX`, `P95_MAX`, `SILENT_MAX`,
and `BLOCK_CV_MAX`.

## GPU acceleration

There are two independent GPU opportunities:

1. `analyze_spontaneous.py` runs official SVCA2, covariance eigendecomposition
   and DMD on CUDA when `--device cuda` is selected.
2. Whole-brain simulation can optionally use Brian2CUDA's `cuda_standalone`
   backend via `run_spontaneous.py --cuda`.

A single Brian2CUDA simulation uses one GPU; it is not distributed across
multiple GPUs. Brian2CUDA remains an experimental optional backend; the CUDA
environment pins PyPI `Brian2Cuda==1.0a4` with its required Brian2 2.6.0.
Use the eight GPUs for independent seeds and graph controls.
Create a separate CUDA environment and run a smoke test first:

```bash
bash setup_cuda_env.sh
source .venv-cuda/bin/activate
GPU=0 bash smoke_cuda.sh
```

The server with NVIDIA driver `560.28.03` reports CUDA 12.6 capability. The
setup therefore pins the official `torch==2.12.0` `cu126` wheel instead of
allowing pip to select a CUDA 13.x wheel. To repair an existing environment
that reports a version such as `torch 2.12.0+cu130`:

```bash
source .venv-cuda/bin/activate
bash install_torch_cuda.sh
python -m pip check
GPU=0 bash smoke_cuda.sh
```

`install_torch_cuda.sh` removes incompatible Torch packages, installs from the
official PyTorch `cu126` wheel index, and performs a real GPU tensor operation.
Brian2CUDA additionally compiles with the system `nvcc`; use a CUDA 12.6
toolkit module and do not load CUDA 13.x on this driver.

If Brian2CUDA reports `Value 'sm_86' is not defined`, `/usr/bin/nvcc` is an
old toolkit that predates RTX 3090 support. Without sudo access, install the
CUDA 12.6 toolkit into `$HOME/.local/cuda-12.6` using the dedicated Conda
environment created by the installer:

```bash
bash install_cuda_toolkit_12_6_ubuntu.sh
source ./activate_cuda_toolkit.sh
which nvcc
nvcc --version
GPU=0 bash smoke_cuda.sh
```

The installer uses `--override-channels` with NVIDIA and conda-forge only, so
it does not query Anaconda's `defaults` channels or require accepting their
Terms of Service.

The selected compiler should be `$HOME/.local/cuda-12.6/bin/nvcc`. Users with
sudo can instead run `INSTALL_METHOD=apt bash
install_cuda_toolkit_12_6_ubuntu.sh` for a system installation. All CUDA
workflow entry points source `activate_cuda_toolkit.sh` automatically and
refuse compilers that cannot target `sm_86`.

Short calibration can also use GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 SIM_BACKEND=cuda \
PAIRS="5:0.90 5:0.92 5:0.94 5:0.96" bash calibrate_poisson.sh
```

Then run the multi-GPU workflow:

```bash
GPU_IDS="0 1 2 3 4 5 6 7" MAX_JOBS=4 \
SEEDS="0 1 2" W_BG=5.0 GI=0.9 SYMMETRIZED_GAIN=<calibrated_gain> \
bash run_server_multi_gpu.sh
```

Set `GPU_IDS` to GPUs that are actually idle. For example, if GPUs 1-5 are
already occupied, begin with `GPU_IDS="0 6 7" MAX_JOBS=3`.

`MAX_JOBS=4` is conservative because every process separately loads the
14.7-million-edge original graph or 25-million-edge symmetric null. Monitor
host RAM and GPU memory with `htop` and `nvidia-smi`; increase to
`MAX_JOBS=8 SEEDS="0 1 2 3"` only after a short workflow succeeds. Compare a
short CUDA run with the CPU baseline statistically, not spike-for-spike, because
the random-number implementations can differ between backends.

If Brian2CUDA does not support a model operation on the server, retain
multi-process C++ simulation while still using CUDA for all analyses:

```bash
SIM_BACKEND=cpp MAX_JOBS=4 bash run_server_multi_gpu.sh
```

The sequential workflow also accepts:

```bash
SIM_BACKEND=cuda ANALYSIS_DEVICE=cuda bash run_server_poisson.sh
```

Only a random neuron subset is monitored, avoiding storage of all whole-brain
spike events. The primary readout is spike count binned at approximately 22 Hz;
sampled membrane voltage is a secondary latent-state diagnostic.

## Functional analysis

```bash
python analyze_spontaneous.py \
  --in spont_poisson_r3000_w5.0_gi0.9_original_s0.npz \
  --con 2023_03_23_connectivity_630_final.parquet \
  --out spont_poisson_r3000_w5.0_gi0.9_original_s0_analysis.json
```

The pipeline uses neuron-wise z-scoring, one official random-split `SVCA2` call
because soma coordinates are absent, the official `1/rank`-weighted exponent
fit, and official PCA-reduced DMD at approximately 0.23 s. It also reports
eigenvector condition number, rotation statistics, and the
numerical-minus-spectral-abscissa gap. An all-silent or insufficiently active
run is rejected before these functions are loaded.

The primary spike DMD defaults reproduce the official mouse-analysis choices
`delta=5` bins, `lam=0.1`, `nt=5000`, and `n_comps=1000`. Voltage analysis is an
explicit secondary diagnostic and uses a smaller ridge.

The strongest valid result is a sensitivity comparison across the original and
symmetrized forward models, operating points, seeds and partial observations.
Structural and fitted-propagator metrics must be reported separately.

Data source: Edmond/MPG doi:10.17617/3.CZODIW, Shiu et al. (2024).

# Non-normality and the identifiability of symmetric critical dynamics

Reproduction **code** for the analyses behind the *Matters Arising* "Cortical
population activity does not identify symmetric critical dynamics" (arising from
M. Pachitariu *et al.*, *Nature* 2026). The manuscript itself is not included in
this repository.

The argument is that two reported observables of spontaneous cortical
activity — a covariance power-law exponent (ν ≈ 0.7–0.85) and near-real
dynamic-mode-decomposition (DMD) eigenvalues — do **not** identify symmetric,
critically normalized recurrent dynamics:

1. **The exponent is degenerate.** ν is set jointly by reciprocity η and the
   spectral abscissa α_eff, so the cortical band is reproduced along a whole
   curve of (η, α_eff) pairs, not a single symmetric-critical point.
2. **Near-real DMD eigenvalues do not imply symmetry.** A non-symmetric,
   non-normal operator `A = T S T⁻¹` has identical real eigenvalues, zero DMD
   rotation, and passes corresponding covariance, rank–timescale and scaled
   zero-shot-memory analyses alongside its symmetric parent.
3. **A connectome with known ground truth illustrates the non-uniqueness.** In a whole-brain
   *Drosophila* leaky-integrate-and-fire model on the FlyWire connectome,
   degree-preserving nulls vary structural reciprocity over two orders of
   magnitude at fixed in/out-degree, weight multiset and Dale sign; across the
   tested graph controls and operating points, neither the DMD spectrum nor the
   covariance exponent provides a one-to-one ordering of the known reciprocity.

## Layout

```
nature_matters_arising/      Analytic reproduction (Gap 1 / Gap 2)
  ma_degeneracy_surface.py            Gap 1: ν(η, α_eff) Lyapunov degeneracy surface
  ma_real_spectrum_counterexample.py  Gap 2: real-spectrum non-normal counterexample
  ma_sparse_spatial_counterexample.py Gap 2 robustness (sparse, spatial)
  ma_effective_operator_from_data.py  Effective-operator estimator + validation
  ma_probe_diagnostic.py, ma_controls.py
  make_fig_v4.py                      Rebuilds the analytic figure
  *.json                              Precomputed numerical outputs (figures rebuild offline)

flywire_spontaneous/         FlyWire / Shiu ground-truth test (GPU; Brian2)
  run_spontaneous.py         Whole-brain LIF simulation + spontaneous regime
  connectome_controls.py     original / symmetrized / degree-preserving nulls
  analyze_spontaneous.py     Calls the pinned MouseLand/critical_init SVCA2/DMD
  sweep_recurrence.sh, iso_nu_search.sh, rate_matched_compare.py
  make_fig_identifiability.py
  precomputed/              42 per-run analysis JSONs, figure, manifests, hashes
  setup_*.sh, download_data.sh, requirements*.txt, README.md
```

## Analytic results (CPU)

Requires Python 3.10+ with `numpy`, `scipy`, `matplotlib`.

```bash
cd nature_matters_arising
python ma_degeneracy_surface.py            # Gap 1 surface  -> ma_degeneracy_surface.json
python ma_real_spectrum_counterexample.py  # Gap 2          -> ma_real_spectrum_results.json
python ma_sparse_spatial_counterexample.py # Gap 2 robustness
python ma_effective_operator_from_data.py  # estimator validation
mkdir -p figures && python make_fig_v4.py  # rebuilds the analytic figure
```

## FlyWire ground-truth test (GPU)

Runs the published Shiu *Drosophila* LIF model (Brian2 / brian2cuda) on the
FlyWire connectome. The connectivity data (~86 MB) and the official analysis
code are fetched by the setup scripts and are **not** stored here.

```bash
cd flywire_spontaneous
bash setup_env.sh                 # venv + deps + pinned MouseLand/critical_init
bash download_data.sh             # FlyWire connectivity + completeness files
python connectome_controls.py --con 2023_03_23_connectivity_630_final.parquet --control degree_reciprocal
python connectome_controls.py --con 2023_03_23_connectivity_630_final.parquet --control degree_randomized
GPU_IDS="0 6 7" bash sweep_recurrence.sh
CONTROL=degree_reciprocal GAINS="1.5 2.0 2.5" bash iso_nu_search.sh
CONTROL=degree_randomized GAINS="1.5 2.0 2.5" bash iso_nu_search.sh
python make_fig_identifiability.py iso_*_analysis.json spont_poisson_recdom*_analysis.json \
  --R original=0.265 symmetrized=1.0 degree_reciprocal=0.690 degree_randomized=0.006 \
  --out fig_identifiability
```

The archived per-run analysis outputs rebuild the deposited figure without
rerunning the GPU simulation:

```bash
python make_fig_identifiability.py precomputed/analysis/*_analysis.json \
  --out fig_identifiability
```

See `flywire_spontaneous/README.md` for the full server workflow.

## External data and code

- **FlyWire connectome** (Shiu *et al.*, *Nature* 2024): fetched by
  `flywire_spontaneous/download_data.sh`.
- **SVCA2 / DMD pipeline**: the pinned public `MouseLand/critical_init`
  implementation is cloned by `flywire_spontaneous/setup_critical_init.sh`; the
  analysis calls it directly.

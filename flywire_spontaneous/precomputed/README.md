# FlyWire precomputed analysis outputs

This directory deposits the numerical outputs behind the FlyWire
identifiability figure without the large simulation `.npz` files.

- `analysis/`: 42 per-run `analyze_spontaneous.py` JSON outputs.
- `manifest.json`: machine-readable control, gain, seed, plotted statistics,
  file hashes and pinned official-analysis commit.
- `figure/`: deposited PDF/PNG, plus the plotting summary generated from the
  42 runs.
- `iso_nu_table.md`: human-readable table of the same operating-point sweep.
- `server_code_and_null_sha256.txt`: hashes recorded on the simulation server,
  including both degree-preserving null Parquet files.
- `SHA256SUMS`: hashes for every file in this directory.

The degree-preserving nulls use `FLYWIRE_NULL_SEED=0` and
`FLYWIRE_NULL_PASSES=5`. Their reciprocated-edge fractions are:

| control | R |
|---|---:|
| degree randomized | 0.006 |
| original | 0.265 |
| degree reciprocal | 0.690 |
| symmetrized | 1.000 |

The sweep covers multiple operating points, including exploratory high-gain
stress tests. It should not be interpreted as one matched low-rate biological
regime. The symmetrized graph also changes edge count, degree, Dale sign and
topology, so its contrast is not a single-factor intervention on reciprocity or
density.

From `flywire_spontaneous/`, rebuild the figure with:

```bash
python make_fig_identifiability.py precomputed/analysis/*_analysis.json \
  --out fig_identifiability
```

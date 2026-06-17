#!/usr/bin/env python3
"""Aggregate FlyWire/Shiu forward-model calibration runs across seeds."""

import argparse
import json
from pathlib import Path

import numpy as np


FUNCTIONAL_METRICS = {
    "svca2_nu": ("svca2", "nu"),
    "direct_nu": ("direct", "nu"),
    "rotation_median": ("dmd", "rotation_median"),
    "rotation_p95": ("dmd", "rotation_p95"),
    "eigenvector_condition": ("dmd", "eigenvector_condition"),
    "numerical_minus_spectral_abscissa": (
        "dmd", "numerical_minus_spectral_abscissa"
    ),
    "max_abs_imag": ("dmd", "max_abs_imag"),
}
SIMULATION_METRICS = [
    "recurrent_gain",
    "mean_rate_hz", "rate_p05_hz", "rate_median_hz", "rate_p95_hz",
    "silent_fraction", "pooled_binned_fano", "median_active_neuron_fano",
    "target_rate_band_fraction", "high_rate_fraction", "block_rate_cv"
]


def summarize(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return {
        "n": int(len(x)),
        "mean": float(x.mean()) if len(x) else None,
        "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0 if len(x) else None,
        "min": float(x.min()) if len(x) else None,
        "max": float(x.max()) if len(x) else None,
    }


def nested(row, path):
    for key in path:
        row = row[key]
    return row


def format_stat(stat, digits):
    if stat["mean"] is None:
        return "NA"
    return f"{stat['mean']:.{digits}f}+/-{stat['sd']:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("spont_poisson_summary.json"))
    args = parser.parse_args()
    runs = []
    for path in args.inputs:
        with open(path) as handle:
            runs.append(json.load(handle))
    controls = {run["simulation"].get("connectome_control", "original") for run in runs}
    if len(controls) != 1:
        raise ValueError(
            "refusing to aggregate different connectome controls: "
            + ", ".join(sorted(controls))
        )

    names = [row["name"] for row in runs[0]["functional"]]
    result = {
        "interpretation": (
            "Across-run forward-model sensitivity summary; no structural-to-DMD "
            "eta equality criterion is applied."
        ),
        "inputs": [str(path) for path in args.inputs],
        "connectome_control": next(iter(controls)),
        "simulation": {
            key: summarize([run["simulation"].get(key) for run in runs
                            if run["simulation"].get(key) is not None])
            for key in SIMULATION_METRICS
        },
        "functional": {},
    }
    for index, name in enumerate(names):
        result["functional"][name] = {
            label: summarize([nested(run["functional"][index], path) for run in runs])
            for label, path in FUNCTIONAL_METRICS.items()
        }
    structural = [run["structural"] for run in runs if "structural" in run]
    if structural:
        result["structural"] = structural[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)

    print(f"aggregated {len(runs)} runs")
    for name, metrics in result["functional"].items():
        print(
            f"{name}: SVCA2 nu={format_stat(metrics['svca2_nu'], 3)}, "
            f"rotation p95={format_stat(metrics['rotation_p95'], 3)}, "
            f"kappa(V)={format_stat(metrics['eigenvector_condition'], 1)}"
        )
    print("saved", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare original and symmetric-null FlyWire forward-model summaries."""

import argparse
import json
from pathlib import Path


def metric_delta(left, right):
    left_mean = left.get("mean")
    right_mean = right.get("mean")
    return {
        "original_mean": left_mean,
        "symmetrized_mean": right_mean,
        "symmetrized_minus_original": (
            right_mean - left_mean
            if left_mean is not None and right_mean is not None else None
        ),
    }


def format_delta(value, digits):
    return "NA" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs=2)
    parser.add_argument("--out", type=Path, default=Path("spont_control_comparison.json"))
    args = parser.parse_args()

    runs = {}
    for path in args.inputs:
        with open(path) as handle:
            row = json.load(handle)
        runs[row["connectome_control"]] = row
    if set(runs) != {"original", "symmetrized"}:
        raise ValueError("inputs must be one original and one symmetrized summary")

    original = runs["original"]
    symmetric = runs["symmetrized"]
    result = {
        "interpretation": (
            "Sensitivity contrast for matched forward-model workflows. Deltas "
            "do not estimate structural eta from functional data."
        ),
        "inputs": [str(path) for path in args.inputs],
        "simulation": {
            key: metric_delta(value, symmetric["simulation"][key])
            for key, value in original["simulation"].items()
        },
        "functional": {},
    }
    for activity, metrics in original["functional"].items():
        result["functional"][activity] = {
            key: metric_delta(value, symmetric["functional"][activity][key])
            for key, value in metrics.items()
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)

    print("symmetrized minus original")
    for activity, metrics in result["functional"].items():
        print(
            f"{activity}: "
            f"SVCA2 nu={format_delta(metrics['svca2_nu']['symmetrized_minus_original'], 3)}, "
            f"rotation p95={format_delta(metrics['rotation_p95']['symmetrized_minus_original'], 3)}, "
            f"kappa(V)={format_delta(metrics['eigenvector_condition']['symmetrized_minus_original'], 1)}"
        )
    print("saved", args.out)


if __name__ == "__main__":
    main()

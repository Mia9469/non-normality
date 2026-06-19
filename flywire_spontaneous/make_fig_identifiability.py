#!/usr/bin/env python3
"""Summary figure for the FlyWire forward-model sensitivity analysis.

Panel A reports a DMD spectral diagnostic against known structural reciprocity.
Panel B reports the SVCA2 exponent across operating points. The figure is
descriptive: the graph transformations and firing-rate distributions are not a
single-factor causal intervention on reciprocity.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "degree_randomized": "#1565C0",
    "original": "#212121",
    "degree_reciprocal": "#2E7D32",
    "symmetrized": "#C62828",
}
LABELS = {
    "degree_randomized": "degree-null, recip.↓ (random)",
    "original": "original (real)",
    "degree_reciprocal": "degree-null, recip.↑",
    "symmetrized": "symmetrized (η=1; 1.70× edges)",
}
DEFAULT_R = {
    "degree_randomized": 0.006,
    "original": 0.265,
    "degree_reciprocal": 0.690,
    "symmetrized": 1.000,
}


def readout(func, name):
    if isinstance(func, dict):
        return func.get(name, {})
    for f in func:
        if f.get("name") == name:
            return f
    return {}


def load_rows(files):
    rows = []
    for f in files:
        try:
            j = json.loads(Path(f).read_text())
        except Exception as exc:
            raise ValueError(f"cannot read analysis JSON {f}: {exc}") from exc
        sim = j.get("simulation", {})
        b = readout(j.get("functional", []), "binned_spikes_primary")
        dmd = b.get("dmd", {})
        row = {
            "control": sim.get("connectome_control", "?"),
            "gain": sim.get("recurrent_gain", float("nan")),
            "seed": sim.get("seed", -1),
            "rate": sim.get("mean_rate_hz", float("nan")),
            "nu": b.get("svca2", {}).get("nu", float("nan")),
            "max_imag": dmd.get("max_abs_imag", float("nan")),
            "rot": dmd.get("rotation_p95", float("nan")),
            "rotation_count": dmd.get("rotation_count", 0),
            "file": str(f),
        }
        if row["control"] not in COLORS:
            continue
        required = ("gain", "rate", "nu", "max_imag")
        if not all(np.isfinite(row[key]) for key in required):
            raise ValueError(f"missing finite figure value in {f}")
        rows.append(row)
    missing = set(COLORS) - {row["control"] for row in rows}
    if missing:
        raise ValueError("missing controls: " + ", ".join(sorted(missing)))
    return rows


def sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--R", nargs="+", default=[],
                    help="control=reciprocity, e.g. original=0.27 symmetrized=1.0")
    ap.add_argument("--out", default="fig_identifiability")
    args = ap.parse_args()
    R = dict(DEFAULT_R)
    for kv in args.R:
        k, v = kv.split("=")
        if k not in COLORS:
            raise ValueError(f"unknown control in --R: {k}")
        R[k] = float(v)
    rows = load_rows(args.json)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.0))
    summary = {
        "inputs": [str(path) for path in args.json],
        "n_runs": len(rows),
        "structural_reciprocity": R,
        "symmetrized_edge_ratio": 25015346 / 14687178,
        "controls": {},
    }

    # ---- Panel A: max|Im| against structural reciprocity ----
    rotating_label_used = False
    for control in COLORS:
        rs = [r for r in rows if r["control"] == control]
        rotating = [
            r for r in rs
            if np.isfinite(r["rot"]) and abs(r["rot"]) > 0.05
        ]
        nonrotating = [r for r in rs if r not in rotating]
        offsets = np.linspace(-0.008, 0.008, max(1, len(nonrotating)))
        axA.scatter(
            R[control] + offsets,
            [r["max_imag"] for r in nonrotating],
            s=20, color=COLORS[control], alpha=0.28, linewidths=0,
        )
        values = [r["max_imag"] for r in nonrotating]
        axA.errorbar(
            R[control], np.mean(values), yerr=sample_sd(values),
            fmt="o", ms=8, color=COLORS[control], ecolor=COLORS[control],
            label=LABELS[control], capsize=3, zorder=4,
        )
        for r in rotating:
            axA.scatter(
                R[control], r["max_imag"], marker="*", s=180,
                color=COLORS[control], zorder=5,
                label=("rotating high-gain runs" if not rotating_label_used else None),
            )
            rotating_label_used = True
        summary["controls"][control] = {
            "n_runs": len(rs),
            "n_rotating_runs": len(rotating),
            "nonrotating_max_abs_imag_mean": float(np.mean(values)),
            "nonrotating_max_abs_imag_sd": sample_sd(values),
        }
    pooled_nonrotating = [
        r["max_imag"] for r in rows
        if not (np.isfinite(r["rot"]) and abs(r["rot"]) > 0.05)
    ]
    axA.axhline(np.mean(pooled_nonrotating), color="0.65", ls=":", lw=1)
    axA.annotate(
        "high-gain symmetrized\nrotation = 1.4-2.0",
        (1.0, 0.54), xytext=(-8, 0), textcoords="offset points",
        fontsize=7.5, ha="right", va="center",
    )
    axA.set_xlabel("structural reciprocity  R")
    axA.set_ylabel(r"DMD  $\max|\mathrm{Im}\,\lambda|$  (binned)")
    axA.set_title("(a) DMD diagnostic is not one-to-one with R", fontsize=10)
    axA.set_xlim(-0.03, 1.04)
    axA.set_ylim(0.14, 0.64)
    axA.legend(frameon=False, fontsize=7.2, loc="upper left")

    # ---- Panel B: nu(binned) vs rate, per control (mean +/- sd over seeds) ----
    for control in COLORS:
        rs = [r for r in rows if r["control"] == control]
        if not rs:
            continue
        by_gain = {}
        for r in rs:
            by_gain.setdefault(r["gain"], []).append(r)
        pts = []
        for grp in by_gain.values():
            rate = [x["rate"] for x in grp]
            nu = [x["nu"] for x in grp]
            pts.append((np.mean(rate), np.mean(nu),
                        sample_sd(nu), sample_sd(rate),
                        grp[0]["gain"], len(grp)))
        pts.sort()
        rate, nu, nuerr, rterr = (
            np.asarray(c, float) for c in zip(*[point[:4] for point in pts])
        )
        axB.errorbar(rate, nu, yerr=nuerr, xerr=rterr, fmt="o-",
                     color=COLORS[control], lw=1.8, ms=6, capsize=2,
                     label=LABELS[control])
        summary["controls"][control]["gain_groups"] = [
            {
                "gain": float(point[4]),
                "n": int(point[5]),
                "rate_mean_hz": float(point[0]),
                "rate_sd_hz": float(point[3]),
                "nu_mean": float(point[1]),
                "nu_sd": float(point[2]),
            }
            for point in pts
        ]
    axB.set_xlabel("mean firing rate (Hz)")
    axB.set_ylabel(r"$\nu$  (binned SVCA2 exponent)")
    axB.set_title(r"(b) $\nu$ depends on operating point and graph control",
                  fontsize=10)
    axB.legend(frameon=False, fontsize=7.5, loc="center right")

    # No figure-level title: the manuscript caption carries it (avoids
    # duplicating the caption's bold lead). Panel titles remain.
    fig.tight_layout()
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    Path(args.out + "_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("saved", args.out + ".pdf /.png /_summary.json",
          f"({len(rows)} runs)")


if __name__ == "__main__":
    main()

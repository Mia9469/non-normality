#!/usr/bin/env python3
"""Summary figure: functional readouts do not identify connectivity reciprocity.

Panel A: the symmetry-specific DMD readout (max|Im lambda|, and rotation where
present) vs structural reciprocity R, across degree-matched controls and the
density-doubled symmetrized null. max|Im| is flat across the whole reciprocity
range; rotation appears only in the synchronous symmetrized-rg1.0 outlier (and
points the wrong way). -> the spectral symmetry readout is blind to reciprocity.

Panel B: nu(binned SVCA2) vs firing rate, one series per control. The
degree-matched nulls track the real network (rising with recurrence); only the
density-doubled symmetrized null is pinned flat at ~0.13 -> its apparent nu
"separation" is density, not reciprocity. Original and the high-reciprocity
degree null overlap -> nu does not identify reciprocity.

Reads analyze_spontaneous JSONs; reciprocity R per control supplied on the CLI
(degree-null analyses are run with --skip-structure).
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
    "symmetrized": "symmetrized (η=1, 2× density)",
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
        except Exception:
            continue
        sim = j.get("simulation", {})
        b = readout(j.get("functional", []), "binned_spikes_primary")
        dmd = b.get("dmd", {})
        rows.append({
            "control": sim.get("connectome_control", "?"),
            "gain": sim.get("recurrent_gain", float("nan")),
            "rate": sim.get("mean_rate_hz", float("nan")),
            "nu": b.get("svca2", {}).get("nu", float("nan")),
            "max_imag": dmd.get("max_abs_imag", float("nan")),
            "rot": dmd.get("rotation_p95", float("nan")),
        })
    return [r for r in rows if r["control"] in COLORS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--R", nargs="+", required=True,
                    help="control=reciprocity, e.g. original=0.27 symmetrized=1.0")
    ap.add_argument("--out", default="fig_identifiability")
    args = ap.parse_args()
    R = {}
    for kv in args.R:
        k, v = kv.split("=")
        R[k] = float(v)
    rows = load_rows(args.json)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- Panel A: max|Im| (and rotation outliers) vs reciprocity R ----
    for control in COLORS:
        rs = [r for r in rows if r["control"] == control]
        if not rs or control not in R:
            continue
        normal = [r["max_imag"] for r in rs if r["max_imag"] < 0.3]
        outl = [r for r in rs if r["max_imag"] >= 0.3]
        if normal:
            axA.errorbar(R[control], np.mean(normal),
                         yerr=(np.std(normal) if len(normal) > 1 else 0),
                         fmt="o", ms=9, color=COLORS[control],
                         label=LABELS[control], capsize=3)
        for r in outl:
            axA.scatter(R[control], r["max_imag"], marker="*", s=220,
                        color=COLORS[control], zorder=5)
            axA.annotate("synchronous\n(rotation=%.1f)" % (r["rot"] or 0),
                         (R[control], r["max_imag"]), textcoords="offset points",
                         xytext=(-10, -28), fontsize=7.5, ha="center")
    axA.axhline(np.mean([r["max_imag"] for r in rows if r["max_imag"] < 0.3]),
                color="0.6", ls=":", lw=1,
                label="flat across reciprocity")
    axA.set_xlabel("structural reciprocity  R")
    axA.set_ylabel(r"DMD  $\max|\mathrm{Im}\,\lambda|$  (binned)")
    axA.set_title("(a) symmetry readout is blind to reciprocity", fontsize=10)
    axA.legend(frameon=False, fontsize=7.5, loc="center left")

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
                        np.std(nu) if len(nu) > 1 else 0.0,
                        np.std(rate) if len(rate) > 1 else 0.0))
        pts.sort()
        rate, nu, nuerr, rterr = (np.array(c) for c in zip(*pts))
        axB.errorbar(rate, nu, yerr=nuerr, xerr=rterr, fmt="o-",
                     color=COLORS[control], lw=1.8, ms=6, capsize=2,
                     label=LABELS[control])
    axB.set_xlabel("mean firing rate (Hz)")
    axB.set_ylabel(r"$\nu$  (binned SVCA2 exponent)")
    axB.set_title(r"(b) $\nu$: symmetrized flat = density, not reciprocity",
                  fontsize=10)
    axB.legend(frameon=False, fontsize=7.5, loc="center right")

    fig.suptitle("Functional readouts do not recover structural reciprocity",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    print("saved", args.out + ".pdf /.png", f"({len(rows)} runs)")


if __name__ == "__main__":
    main()

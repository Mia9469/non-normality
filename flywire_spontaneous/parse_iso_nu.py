#!/usr/bin/env python3
"""Tabulate nu and the symmetry-specific readouts vs recurrent_gain.

Reads any analyze_spontaneous JSON outputs (iso-nu scan runs plus existing
A/B1/B2 analyses), and builds one table per connectome control:

  gain | seed | rate | nu(binned) | nu(voltage) | rotation_p95 | max|Im| |
  eta_pca | kappa(V)

Use it to find iso-nu pairs (an original gain and a symmetrized gain with the
SAME nu) and read off whether their rotation / max|Im| / kappa(V) also coincide.
"""
import argparse
import json
from pathlib import Path


def readout(func, name):
    for f in func:
        if f.get("name") == name:
            return f
    return {}


def extract(path):
    try:
        j = json.loads(Path(path).read_text())
    except Exception:
        return None
    sim = j.get("simulation", {})
    func = j.get("functional", [])
    b = readout(func, "binned_spikes_primary")
    v = readout(func, "membrane_voltage_secondary")
    bd = b.get("dmd", {})
    return {
        "control": sim.get("connectome_control", "?"),
        "gain": sim.get("recurrent_gain", float("nan")),
        "seed": sim.get("seed", -1),
        "rate": sim.get("mean_rate_hz", float("nan")),
        "nu_bin": b.get("svca2", {}).get("nu", float("nan")),
        "nu_volt": v.get("svca2", {}).get("nu", float("nan")),
        "rot_p95": bd.get("rotation_p95", float("nan")),
        "max_imag": bd.get("max_abs_imag", float("nan")),
        "eta_pca": bd.get("eta_pca_coordinate", float("nan")),
        "kappaV": bd.get("eigenvector_condition", float("nan")),
        "file": Path(path).name,
    }


def fmt(x, p=3):
    try:
        if x != x:  # nan
            return "NA"
        return f"{x:.{p}f}"
    except (TypeError, ValueError):
        return "NA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows, seen = [], set()
    for f in args.files:
        r = extract(f)
        if r is None:
            continue
        key = (r["control"], r["gain"], r["seed"], r["file"])
        # de-dup if the same file matched multiple globs
        dedup = (r["control"], r["gain"], r["seed"])
        if dedup in seen:
            continue
        seen.add(dedup)
        rows.append(r)

    lines = ["# iso-nu: nu and symmetry readouts vs gain", "",
             "Find an original gain and a symmetrized gain with equal "
             "`nu(binned)`; then compare `rotation_p95`, `max|Im|`, `eta_pca`, "
             "`kappa(V)`. Coincidence at matched nu = empirical degeneracy.", ""]
    for control in sorted({r["control"] for r in rows}):
        sub = sorted((r for r in rows if r["control"] == control),
                     key=lambda r: (r["gain"], r["seed"]))
        lines += [f"## {control}", "",
                  "| gain | seed | rate Hz | nu(bin) | nu(volt) | rot_p95 | "
                  "max|Im| | eta_pca | kappa(V) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            lines.append(
                f"| {fmt(r['gain'],2)} | {r['seed']} | {fmt(r['rate'],2)} | "
                f"{fmt(r['nu_bin'])} | {fmt(r['nu_volt'])} | {fmt(r['rot_p95'])} | "
                f"{fmt(r['max_imag'])} | {fmt(r['eta_pca'])} | {fmt(r['kappaV'],1)} |"
            )
        lines.append("")

    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.write_text(table + "\n")
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Parse recurrence-sweep calibration logs into a comparison table.

Reads logs_recsweep_*/<control>_w<wbg>_rg<gain>.log, extracts the firing-regime
diagnostics printed by run_spontaneous.py --calibrate, and emits a markdown
table per control plus a "rec_excess" = mean_rate / mean_rate(gain=0, same w_bg)
that quantifies how much recurrence (vs background) drives the activity.
"""
import argparse
import re
from pathlib import Path

NAME = re.compile(r"^(original|symmetrized)_w([0-9.]+)_rg([0-9.]+)\.log$")
RATE = re.compile(r"mean rate=([0-9.]+)Hz silent fraction=([0-9.]+) "
                  r"pooled binned Fano=([0-9.]+)")
PCT = re.compile(r"percentiles \[5,25,50,75,95\]%=\[([^\]]+)\]")
FRAC = re.compile(r"1-3Hz=([0-9.]+) >20Hz=([0-9.]+); "
                  r"median active-neuron Fano=([0-9.naN]+)")
CV = re.compile(r"block-rate CV=([0-9.]+)")


def parse_log(path):
    text = path.read_text()
    rate = RATE.search(text)
    pct = PCT.search(text)
    frac = FRAC.search(text)
    cv = CV.search(text)
    if not (rate and pct and frac and cv):
        return None
    p = [float(x) for x in pct.group(1).split()]
    return {
        "mean": float(rate.group(1)),
        "silent": float(rate.group(2)),
        "fano": float(rate.group(3)),
        "median": p[2],
        "p95": p[4],
        "band": float(frac.group(1)),
        "high": float(frac.group(2)),
        "cv": float(cv.group(1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cells = {}  # (control, wbg, gain) -> metrics
    for f in sorted(args.logdir.glob("*.log")):
        m = NAME.match(f.name)
        if not m:
            continue
        control, wbg, gain = m.group(1), float(m.group(2)), float(m.group(3))
        parsed = parse_log(f)
        cells[(control, wbg, gain)] = parsed

    lines = ["# Recurrence-dominance sweep",
             "",
             "`rec_excess` = mean_rate / mean_rate(gain=0, same w_bg); "
             ">1 means recurrence adds drive, large = recurrence-dominated.",
             ""]
    for control in ("original", "symmetrized"):
        keys = sorted(k for k in cells if k[0] == control)
        lines += [f"## {control}", "",
                  "| w_bg | gain | mean Hz | median | p95 | 1-3Hz | silent | "
                  "Fano | blockCV | rec_excess |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for key in keys:
            _, wbg, gain = key
            d = cells[key]
            if d is None:
                lines.append(f"| {wbg:g} | {gain:g} | FAILED/parse | | | | | | | |")
                continue
            floor = cells.get((control, wbg, 0.0))
            exc = (d["mean"] / floor["mean"]) if floor and floor.get("mean") else float("nan")
            lines.append(
                f"| {wbg:g} | {gain:g} | {d['mean']:.2f} | {d['median']:.2f} | "
                f"{d['p95']:.1f} | {d['band']:.2f} | {d['silent']:.3f} | "
                f"{d['fano']:.2f} | {d['cv']:.3f} | "
                f"{exc:.1f}x |" if exc == exc else
                f"| {wbg:g} | {gain:g} | {d['mean']:.2f} | {d['median']:.2f} | "
                f"{d['p95']:.1f} | {d['band']:.2f} | {d['silent']:.3f} | "
                f"{d['fano']:.2f} | {d['cv']:.3f} | - |"
            )
        lines.append("")

    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.write_text(table + "\n")
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

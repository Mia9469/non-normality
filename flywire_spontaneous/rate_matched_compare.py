#!/usr/bin/env python3
"""Rate-matched SVCA2 comparison of original vs symmetrized controls.

SVCA2 z-scores then DROPS silent neurons, so the two controls feed SVCA
DIFFERENT active subpopulations (original keeps a hub tail; symmetrized is
near-uniform). A bare nu difference is therefore confounded by the differing
rate distributions / SNR, not necessarily by connectivity symmetry.

This script removes that confound: for each seed-paired (original, symmetrized)
npz, it builds rate-matched active-neuron subsets (equal per-quantile-bin
counts, so the two subsets share a rate histogram), runs the PINNED official
SVCA2 on each, and fits nu. If the nu gap survives matching it is a genuine
correlation-structure difference; if it collapses it was the distribution.

Reuses the official-code loader and SVCA/fit wrappers from analyze_spontaneous,
so the SVCA2 call is byte-identical to the main pipeline.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_spontaneous import (
    load_official_critical_init,
    resolve_device,
    svca2_spectrum,
    fit_powerlaw,
    DEFAULT_CRITICAL_INIT,
)


def load_activity_and_rates(path):
    """Return (activity[time, neurons] float32, neuron_rates[neurons])."""
    data = np.load(path, allow_pickle=False)
    spikes = np.asarray(data["spike_counts"])          # [neurons, time]
    activity = spikes.T.astype(np.float32)             # [time, neurons]
    if "neuron_rates" in data:
        rates = np.asarray(data["neuron_rates"], float)
    else:
        dt_s = float(data["dt_bin"]) / 1000.0
        rates = spikes.sum(axis=1) / (spikes.shape[1] * dt_s)
    return activity, rates


def rate_matched_indices(rates_a, rates_b, active_a, active_b, n_bins, rng):
    """Equal-per-bin subsample so both subsets share a rate histogram."""
    ra, rb = rates_a[active_a], rates_b[active_b]
    pooled = np.concatenate([ra, rb])
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] = np.inf
    keep_a, keep_b = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        ia = active_a[(ra >= lo) & (ra < hi)]
        ib = active_b[(rb >= lo) & (rb < hi)]
        take = min(len(ia), len(ib))
        if take == 0:
            continue
        keep_a.append(rng.choice(ia, take, replace=False))
        keep_b.append(rng.choice(ib, take, replace=False))
    return (np.concatenate(keep_a) if keep_a else np.array([], int),
            np.concatenate(keep_b) if keep_b else np.array([], int))


def nu_on_subset(activity, idx, official_powerlaw, device, repeats, seed,
                 rank_min, rank_max):
    spectrum = svca2_spectrum(activity[:, idx], official_powerlaw, device,
                              repeats, seed)
    return fit_powerlaw(spectrum, official_powerlaw, rank_min, rank_max)["nu"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, nargs="+", required=True,
                    help="original-control npz files, seed-ordered")
    ap.add_argument("--symmetrized", type=Path, nargs="+", required=True,
                    help="symmetrized-control npz files, same length/order")
    ap.add_argument("--out", type=Path, default=Path("rate_matched_compare.json"))
    ap.add_argument("--bins", type=int, default=20, help="rate quantile bins")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    ap.add_argument("--svca-repeats", type=int, default=1)
    ap.add_argument("--rank-min", type=int, default=10)
    ap.add_argument("--rank-max", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--critical-init-root", type=Path,
                    default=DEFAULT_CRITICAL_INIT)
    args = ap.parse_args()
    if len(args.original) != len(args.symmetrized):
        raise ValueError("--original and --symmetrized need equal counts")

    powerlaw, lyapun, commit = load_official_critical_init(args.critical_init_root)
    device = resolve_device(lyapun.torch, args.device)

    pairs = []
    for fo, fs in zip(args.original, args.symmetrized):
        rng = np.random.default_rng(args.seed)
        act_o, rate_o = load_activity_and_rates(fo)
        act_s, rate_s = load_activity_and_rates(fs)
        active_o = np.where(act_o.std(axis=0) > 1e-9)[0]
        active_s = np.where(act_s.std(axis=0) > 1e-9)[0]

        # raw (unmatched) nu on each full active set, for reference
        nu_o_raw = nu_on_subset(act_o, active_o, powerlaw, device,
                                args.svca_repeats, args.seed,
                                args.rank_min, args.rank_max)
        nu_s_raw = nu_on_subset(act_s, active_s, powerlaw, device,
                                args.svca_repeats, args.seed,
                                args.rank_min, args.rank_max)

        idx_o, idx_s = rate_matched_indices(rate_o, rate_s, active_o, active_s,
                                            args.bins, rng)
        nu_o = nu_on_subset(act_o, idx_o, powerlaw, device, args.svca_repeats,
                            args.seed, args.rank_min, args.rank_max)
        nu_s = nu_on_subset(act_s, idx_s, powerlaw, device, args.svca_repeats,
                            args.seed, args.rank_min, args.rank_max)

        pair = {
            "original_file": str(fo), "symmetrized_file": str(fs),
            "n_matched": int(len(idx_o)),
            "matched_rate_median_orig": float(np.median(rate_o[idx_o])),
            "matched_rate_median_sym": float(np.median(rate_s[idx_s])),
            "matched_rate_p95_orig": float(np.percentile(rate_o[idx_o], 95)),
            "matched_rate_p95_sym": float(np.percentile(rate_s[idx_s], 95)),
            "nu_raw_original": nu_o_raw, "nu_raw_symmetrized": nu_s_raw,
            "nu_raw_delta": nu_s_raw - nu_o_raw,
            "nu_matched_original": nu_o, "nu_matched_symmetrized": nu_s,
            "nu_matched_delta": nu_s - nu_o,
        }
        pairs.append(pair)
        print(f"{Path(fo).name} vs {Path(fs).name}: n_matched={pair['n_matched']} "
              f"| raw nu {nu_o:.3f}->{nu_o_raw:.3f}/{nu_s_raw:.3f} "
              f"Δ={pair['nu_raw_delta']:+.3f} "
              f"| MATCHED nu {nu_o:.3f}/{nu_s:.3f} Δ={pair['nu_matched_delta']:+.3f}")

    def agg(key):
        vals = np.array([p[key] for p in pairs], float)
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}

    summary = {
        "n_pairs": len(pairs),
        "bins": args.bins,
        "nu_raw_delta": agg("nu_raw_delta"),
        "nu_matched_delta": agg("nu_matched_delta"),
        "nu_matched_original": agg("nu_matched_original"),
        "nu_matched_symmetrized": agg("nu_matched_symmetrized"),
        "n_matched": agg("n_matched"),
        "pairs": pairs,
        "official_commit": commit,
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nraw   Δnu = {summary['nu_raw_delta']['mean']:+.3f} "
          f"± {summary['nu_raw_delta']['std']:.3f}")
    print(f"matched Δnu = {summary['nu_matched_delta']['mean']:+.3f} "
          f"± {summary['nu_matched_delta']['std']:.3f}  "
          f"(n_matched≈{summary['n_matched']['mean']:.0f})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

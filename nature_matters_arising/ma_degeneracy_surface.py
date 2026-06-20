#!/usr/bin/env python3
"""Continuous-time reciprocity--criticality degeneracy surface.

This follows the dynamical system used by Pachitariu et al.:

    tau dx/dt = (-I + A) x + xi(t).

For a non-symmetric continuous-time operator, distance to instability is set by
the spectral abscissa alpha(A) = max Re eig(A), not by max |eig(A)|.  We draw
elliptic-law matrices with controlled reciprocity eta, scale each realization so
that alpha(A) equals a requested alpha_eff < 1, solve the continuous Lyapunov
equation, and fit the covariance power-law exponent using the rank weighting in
the Article's public code.

The output demonstrates that nu is a joint function of reciprocity and distance
to the continuous-time stability boundary.  It must not be interpreted as an
estimator that separately recovers either quantity.
"""
import argparse
import json
import os

import numpy as np
from scipy.linalg import solve_continuous_lyapunov


def make_operator_eta(n, rng, eta):
    """Return a real elliptic-law matrix with pairwise reciprocity eta."""
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    out = np.empty((n, n))
    iu = np.triu_indices(n, 1)
    il = (iu[1], iu[0])
    out[iu] = a[iu]
    out[il] = eta * a[iu] + np.sqrt(max(0.0, 1.0 - eta**2)) * b[iu]
    np.fill_diagonal(out, a.diagonal())
    return out / np.sqrt(n)


def scale_to_abscissa(a, alpha_eff):
    """Scale A so max Re eig(A) == alpha_eff, the CT stability coordinate."""
    alpha = float(np.max(np.linalg.eigvals(a).real))
    if alpha <= 0:
        raise ValueError("operator has non-positive spectral abscissa")
    return a * (alpha_eff / alpha)


def stationary_covariance(a):
    """Solve (A-I) C + C (A-I)^T = -I."""
    drift = a - np.eye(a.shape[0])
    cov = solve_continuous_lyapunov(drift, -np.eye(a.shape[0]))
    return 0.5 * (cov + cov.T)


def fit_powerlaw(eigenvalues, rank_min=10, rank_max=None):
    """Released-code-matched 1/rank-weighted log--log power-law fit."""
    values = np.sort(np.asarray(eigenvalues, float))[::-1]
    values = values[values > 0]
    if rank_max is None:
        rank_max = min(500, len(values) // 2)
    rank_max = min(rank_max, len(values))
    if rank_max <= rank_min:
        return float("nan"), rank_max

    indices = np.arange(rank_min, rank_max, dtype=int)
    ranks = indices + 1.0
    x = np.column_stack((-np.log(ranks), np.ones_like(ranks)))
    y = np.log(values[indices])
    weight = (1.0 / ranks)[:, None]
    beta = np.linalg.solve(x.T @ (x * weight), (weight * x).T @ y)
    return float(beta[0]), rank_max


def operator_metrics(a):
    eig = np.linalg.eigvals(a)
    return {
        "alpha_eff": float(np.max(eig.real)),
        "spectral_radius": float(np.max(np.abs(eig))),
        "max_imag": float(np.max(np.abs(eig.imag))),
    }


def run(args):
    etas = np.asarray(args.etas, float)
    alphas = np.asarray(args.alphas, float)
    rng = np.random.default_rng(args.seed)
    nu = np.full((len(etas), len(alphas)), np.nan)
    nu_std = np.full_like(nu, np.nan)
    radius = np.full_like(nu, np.nan)

    print(f"N={args.N} navg={args.navg} weighted rank window "
          f"[{args.rank_min},{args.rank_max or min(500, args.N // 2)})")
    print(f"{'eta':>5} {'alpha':>7} {'nu':>8} {'sd':>7} {'radius':>8}")
    for i, eta in enumerate(etas):
        for j, alpha_eff in enumerate(alphas):
            vals, radii = [], []
            for _ in range(args.navg):
                a = scale_to_abscissa(make_operator_eta(args.N, rng, eta),
                                      alpha_eff)
                cov = stationary_covariance(a)
                exponent, _ = fit_powerlaw(
                    np.linalg.eigvalsh(cov), args.rank_min, args.rank_max
                )
                vals.append(exponent)
                radii.append(operator_metrics(a)["spectral_radius"])
            nu[i, j] = np.mean(vals)
            nu_std[i, j] = np.std(vals)
            radius[i, j] = np.mean(radii)
            print(f"{eta:>5.2f} {alpha_eff:>7.3f} {nu[i,j]:>8.3f} "
                  f"{nu_std[i,j]:>7.3f} {radius[i,j]:>8.3f}")

    return {
        "model": "tau dx/dt = (-I + A)x + xi",
        "criticality_coordinate": "alpha_eff = max Re eig(A); stable iff alpha_eff < 1",
        "etas": etas.tolist(),
        "alphas": alphas.tolist(),
        "nu": nu.tolist(),
        "nu_std": nu_std.tolist(),
        "mean_spectral_radius": radius.tolist(),
        "N": args.N,
        "navg": args.navg,
        "rank_min": args.rank_min,
        "rank_max": args.rank_max or min(500, args.N // 2),
    }


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    etas = np.asarray(result["etas"])
    alphas = np.asarray(result["alphas"])
    nu = np.asarray(result["nu"])
    order = np.argsort(etas)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.pcolormesh(
        alphas, etas[order], nu[order], shading="nearest", cmap="viridis"
    )
    try:
        cs = ax.contour(alphas, etas[order], nu[order],
                        levels=[2 / 3, 0.85, 1.0, 1.25],
                        colors="w", linewidths=1.0)
        ax.clabel(cs, fmt="%.2f", fontsize=7)
        ax.contourf(alphas, etas[order], nu[order],
                    levels=[0.70, 0.85], colors=["red"], alpha=0.25)
    except ValueError:
        pass
    ax.set_xlabel(r"spectral abscissa $\alpha_{\rm eff}=\max\Re\lambda(A)$")
    ax.set_ylabel(r"reciprocity $\eta$  (1 = symmetric, 0 = independent)")
    ax.set_title(r"Continuous-time degeneracy: $\nu(\eta,\alpha_{\rm eff})$")
    fig.colorbar(im, ax=ax, label=r"covariance exponent $\nu$")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print("figure ->", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=600)
    parser.add_argument("--navg", type=int, default=8)
    parser.add_argument("--rank-min", type=int, default=10)
    parser.add_argument("--rank-max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--etas", type=float, nargs="+",
                        default=[1.0, 0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.90, 0.95, 0.98, 0.99, 0.995, 0.998])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="ma_degeneracy_surface.json")
    parser.add_argument("--fig", default="figures/ma_degeneracy_surface.png")
    args = parser.parse_args()
    if args.smoke:
        args.N = 120
        args.navg = 2
        args.etas = [1.0, 0.5, 0.0]
        args.alphas = [0.95, 0.99]
        args.rank_max = 60

    result = run(args)
    base = os.path.dirname(__file__) or "."
    out = os.path.join(base, args.out)
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)
    print("saved ->", out)
    plot(result, os.path.join(base, args.fig))


if __name__ == "__main__":
    main()

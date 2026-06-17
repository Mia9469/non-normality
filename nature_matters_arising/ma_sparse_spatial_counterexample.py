#!/usr/bin/env python3
"""Exactly sparse, spatial, real-spectrum non-normal counterexample.

Start from a symmetric sparse matrix S whose connection probability and weight
scale decay with distance on a ring.  Apply a positive diagonal, non-orthogonal
similarity transform:

    A_g = D_g S D_g^{-1},   D_g = diag(exp(g z_i)).

Unlike a local shear T=I+L, whose inverse is generally dense, diagonal
similarity preserves every structural zero exactly:

    A_ij = 0 iff S_ij = 0.

It also preserves the full real spectrum and the distance-dependent support,
while generically making the directed weights non-symmetric and non-normal.
"""
import argparse
import json
import os

import numpy as np


def ring_distance(n):
    idx = np.arange(n)
    raw = np.abs(idx[:, None] - idx[None, :])
    return np.minimum(raw, n - raw) / n


def powerlaw_kernel(distance, d0, exponent):
    return (1.0 + distance / d0) ** (-exponent)


def make_sparse_spatial_symmetric(n, target_density, d0, exponent, rng):
    """Symmetric matrix with distance-dependent Bernoulli support and weights."""
    distance = ring_distance(n)
    kernel = powerlaw_kernel(distance, d0, exponent)
    np.fill_diagonal(kernel, 0.0)
    iu = np.triu_indices(n, 1)
    scale = target_density / np.mean(kernel[iu])
    probability = np.clip(scale * kernel, 0.0, 1.0)
    support_upper = rng.random(len(iu[0])) < probability[iu]

    s = np.zeros((n, n), dtype=float)
    weights = rng.standard_normal(np.count_nonzero(support_upper))
    weights *= np.sqrt(kernel[iu][support_upper])
    rows = iu[0][support_upper]
    cols = iu[1][support_upper]
    s[rows, cols] = weights
    s[cols, rows] = weights
    eigvals, eigvecs = np.linalg.eigh(s)
    s *= 0.998 / eigvals[-1]
    eigvals *= 0.998 / eigvals[-1]
    return s, eigvals, eigvecs, distance, probability


def diagonal_similarity(s, eigvecs, z, gain):
    d = np.exp(gain * z)
    a = (d[:, None] * s) / d[None, :]
    eigenvectors = d[:, None] * eigvecs
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    return a, eigenvectors, float(d.max() / d.min())


def reciprocity(a):
    iu = np.triu_indices(a.shape[0], 1)
    x, y = a[iu], a.T[iu]
    return float(2.0 * np.sum(x * y) / (np.sum(x * x) + np.sum(y * y) + 1e-30))


def radial_profile(a, distance, bins):
    out = []
    abs_a = np.abs(a)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (distance >= lo) & (distance < hi)
        out.append(float(abs_a[mask].mean()) if np.any(mask) else 0.0)
    return out


def metrics(a, eigvals, eigenvectors, distance, bins):
    exact_density = float(np.count_nonzero(a) / a.size)
    asym = float(np.linalg.norm(a - a.T) / (2 * np.linalg.norm(a) + 1e-30))
    return {
        "density_exact": exact_density,
        "eta": reciprocity(a),
        "asymmetry": asym,
        "kappa_eigenvectors": float(np.linalg.cond(eigenvectors)),
        "max_imag_eigenvalue": float(np.max(np.abs(np.asarray(eigvals).imag))),
        "radial_profile": radial_profile(a, distance, bins),
    }


def run(args):
    rng = np.random.default_rng(args.seed)
    s, eigvals, eigvecs, distance, probability = make_sparse_spatial_symmetric(
        args.N, args.density, args.d0, args.exponent, rng
    )
    z = rng.standard_normal(args.N)
    z = (z - z.mean()) / z.std()
    bins = np.linspace(0, 0.5, args.profile_bins + 1)
    result = {
        "construction": "A_g = D_g S D_g^{-1}; D_g diagonal positive",
        "N": args.N,
        "target_density": args.density,
        "distance_kernel": f"(1 + d/{args.d0})^-{args.exponent}",
        "distance_bins": bins.tolist(),
        "mean_support_probability": float(
            probability[np.triu_indices(args.N, 1)].mean()
        ),
        "conditions": {},
    }

    print(f"{'gain':>6} {'density':>9} {'eta':>8} {'asym':>8} "
          f"{'kappa(V)':>10} {'max|Im|':>10}")
    for gain in args.gains:
        a, vectors, transform_condition = diagonal_similarity(s, eigvecs, z, gain)
        row = metrics(a, eigvals, vectors, distance, bins)
        row["transform_condition"] = transform_condition
        result["conditions"][f"{gain:g}"] = row
        print(f"{gain:>6.2f} {100*row['density_exact']:>8.3f}% "
              f"{row['eta']:>8.3f} {row['asymmetry']:>8.3f} "
              f"{row['kappa_eigenvectors']:>10.2f} "
              f"{row['max_imag_eigenvalue']:>10.1e}")
    return result, s, z, distance


def plot(result, s, z, distance, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains = [float(x) for x in result["conditions"]]
    show_gain = gains[-1]
    d = np.exp(show_gain * z)
    shown = (d[:, None] * s) / d[None, :]
    bins = np.asarray(result["distance_bins"])
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))
    sl = slice(0, min(150, s.shape[0]))
    vmax = np.max(np.abs(shown[sl, sl]))
    im = axes[0].imshow(shown[sl, sl], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(
        f"(a) exact sparse support ({100*result['conditions'][f'{show_gain:g}']['density_exact']:.2f}%)"
    )
    axes[0].set_xlabel("neuron j")
    axes[0].set_ylabel("neuron i")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for key, row in result["conditions"].items():
        axes[1].semilogy(centers, np.asarray(row["radial_profile"]) + 1e-15,
                         "o-", ms=3, label=f"g={key}")
    axes[1].set_xlabel("ring distance")
    axes[1].set_ylabel(r"mean $|A_{ij}|$")
    axes[1].set_title("(b) distance dependence retained")
    axes[1].legend(frameon=False, fontsize=8)

    eta = [result["conditions"][f"{g:g}"]["eta"] for g in gains]
    kappa = [result["conditions"][f"{g:g}"]["kappa_eigenvectors"] for g in gains]
    axes[2].plot(gains, eta, "o-", color="#1565C0", label=r"$\eta$")
    axes[2].set_xlabel("diagonal gain g")
    axes[2].set_ylabel(r"reciprocity $\eta$", color="#1565C0")
    axes[2].tick_params(axis="y", colors="#1565C0")
    twin = axes[2].twinx()
    twin.semilogy(gains, kappa, "s--", color="#8E24AA", label=r"$\kappa(V)$")
    twin.set_ylabel(r"$\kappa(V)$", color="#8E24AA")
    twin.tick_params(axis="y", colors="#8E24AA")
    axes[2].set_title("(c) non-normality without fill-in")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=220, bbox_inches="tight")
    print("saved", path + ".pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--density", type=float, default=0.005)
    parser.add_argument("--d0", type=float, default=0.02)
    parser.add_argument("--exponent", type=float, default=2.0)
    parser.add_argument("--gains", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--profile-bins", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="ma_sparse_spatial.json")
    parser.add_argument("--fig", default="figures/ma_sparse_spatial")
    args = parser.parse_args()
    if args.smoke:
        args.N = 240
        args.density = 0.01

    result, s, z, distance = run(args)
    base = os.path.dirname(__file__) or "."
    with open(os.path.join(base, args.out), "w") as handle:
        json.dump(result, handle, indent=2)
    plot(result, s, z, distance, os.path.join(base, args.fig))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Real-spectrum non-normal counterexamples for Pachitariu et al. (2026).

The original Article uses near-real DMD eigenvalues as evidence for symmetric
dynamics.  A real-spectrum operator need not be symmetric.  We construct

    A_g = T_g S T_g^{-1},

where S is a critically normalized symmetric Wigner matrix and T_g is a
non-orthogonal positive-definite change of basis.  A_g has exactly the same
real eigenvalues as S at every g, while becoming increasingly non-symmetric and
non-normal.

This script tests, in the Article's continuous-time model,

    tau dx/dt = (-I + A) x + white noise,

whether these counterexamples reproduce the observables used in the Article:
the covariance power-law exponent, low DMD rotation, the PC
variance--timescale relation, and a time-independent zero-shot memory decoder.
"""
import argparse
import json
import os
import warnings

import numpy as np
from scipy.linalg import solve_continuous_lyapunov


HERE = os.path.dirname(__file__) or "."
# Some macOS Accelerate builds emit spurious matmul RuntimeWarnings while
# returning finite, correct BLAS results. Keep real numerical warnings visible.
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)


def symmetric_operator(n, rng, rho=0.998):
    """Centred symmetric random connectivity, scaled as in the Article."""
    upper = rng.uniform(-1.0, 1.0, size=(n, n))
    s = np.triu(upper, 1)
    s = s + s.T
    np.fill_diagonal(s, 0.0)
    evals, evecs = np.linalg.eigh(s)
    s *= rho / evals[-1]
    evals *= rho / evals[-1]
    return s, evals, evecs


def real_spectrum_transform(s, evals, evecs, rng, gain):
    """Return A=T S T^-1 with a dense SPD, non-orthogonal T."""
    q, _ = np.linalg.qr(rng.standard_normal(s.shape))
    z = rng.standard_normal(s.shape[0])
    z = (z - z.mean()) / z.std()
    d = np.exp(gain * z)
    t = (q * d) @ q.T
    ti = (q * (1.0 / d)) @ q.T
    a = t @ s @ ti
    # Known eigendecomposition; avoids numerical ambiguity for a real spectrum.
    v = t @ evecs
    vi = evecs.T @ ti
    return a, evals.copy(), v, vi, float(d.max() / d.min())


def ginibre_operator(n, rng, rho=0.998):
    """Independent non-symmetric control, scaled by maximum real eigenvalue."""
    a = rng.uniform(-1.0, 1.0, size=(n, n))
    np.fill_diagonal(a, 0.0)
    evals, v = np.linalg.eig(a)
    a *= rho / evals.real.max()
    evals, v = np.linalg.eig(a)
    vi = np.linalg.inv(v)
    return a, evals, v, vi


def weighted_power_law(evals, n_min=10, n_max=None):
    """Match the released fit_powerlaw_exp implementation."""
    values = np.sort(np.asarray(evals).real)[::-1]
    if n_max is None:
        n_max = min(500, len(values) // 2)
    stop = min(n_max, len(values))
    indices = np.arange(n_min, stop, dtype=int)
    ranks = indices + 1.0
    y = np.log(np.abs(values[indices]))[:, None]
    x = np.column_stack((-np.log(ranks), np.ones_like(ranks)))
    weight = (1.0 / ranks)[:, None]
    beta = np.linalg.solve(x.T @ (x * weight), (weight * x).T @ y)
    return float(beta[0, 0]), int(stop)


def reciprocity(a):
    iu = np.triu_indices(a.shape[0], 1)
    return float(np.mean(a[iu] * a.T[iu]) /
                 (np.mean(a[iu] ** 2) + 1e-30))


def asymmetry_fraction(a):
    return float(np.linalg.norm(a - a.T) / (2.0 * np.linalg.norm(a) + 1e-30))


def transition_from_eig(evals, v, vi, delay, tau):
    b = (v * np.exp((evals - 1.0) * delay / tau)) @ vi
    return np.real_if_close(b, tol=1000).real


def summarize_dmd_rotations(b_evals):
    """Rotations per tenfold attenuation from DMD eigenvalues."""
    keep = (b_evals.real > 0.25) & (np.abs(b_evals) < 1.0)
    if not np.any(keep):
        return 0.0, 0.0, 0
    z = b_evals[keep]
    rotations = (np.log(0.1) / np.log(np.abs(z))) * (
        np.abs(np.angle(z)) / (2.0 * np.pi)
    )
    return (float(np.median(rotations)), float(np.quantile(rotations, 0.95)),
            int(len(rotations)))


def dmd_rotations(evals, delay=0.23, tau=0.02):
    """Rotations per tenfold attenuation from exact full-state DMD eigenvalues."""
    return summarize_dmd_rotations(np.exp((evals - 1.0) * delay / tau))


def operator_metrics(a, evals, v, vi, tau=0.02):
    """Continuous-time covariance, DMD, symmetry, and PC-timescale metrics."""
    m = a - np.eye(a.shape[0])
    cov = solve_continuous_lyapunov(m, -np.eye(a.shape[0]))
    cov = 0.5 * (cov + cov.T)
    ce, cu = np.linalg.eigh(cov)
    order = np.argsort(ce)[::-1]
    ce, cu = ce[order], cu[:, order]
    nu, rank_max = weighted_power_law(ce)

    # Integrated autocorrelation time of each covariance PC:
    # integral_0^infty corr_i(t) dt = -tau * u_i^T (A-I)^-1 u_i.
    solve_modes = np.linalg.solve(m, cu)
    times = -tau * np.sum(cu * solve_modes, axis=0)
    k = min(rank_max, len(times))
    valid = (ce[:k] > 0) & (times[:k] > 0)
    r_timescale = float(np.corrcoef(np.log(ce[:k][valid]),
                                    np.log(times[:k][valid]))[0, 1])

    # Article-like fixed-lag PC autocorrelation relation.
    b_lag = transition_from_eig(evals, v, vi, 0.23, tau)
    acg = np.sum(cu * (b_lag @ cu), axis=0)
    valid_acg = (ce[:k] > 0) & (acg[:k] > 0)
    r_fixed_lag = float(np.corrcoef(np.log(ce[:k][valid_acg]),
                                    np.log(acg[:k][valid_acg]))[0, 1])

    sd = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    # z = Z^-1 x, so A_z = Z^-1 A Z.
    a_z = (a * sd[None, :]) / sd[:, None]
    cov_z = cov / (sd[:, None] * sd[None, :])
    _, pca_z = np.linalg.eigh(0.5 * (cov_z + cov_z.T))
    pca_k = min(200, a.shape[0] // 2)
    pca_z = pca_z[:, -pca_k:]
    b_z = (b_lag * sd[None, :]) / sd[:, None]
    reduced_dmd_evals = np.linalg.eigvals(pca_z.T @ b_z @ pca_z)
    pca_rot_med, pca_rot_q95, pca_n_rot = summarize_dmd_rotations(
        reduced_dmd_evals
    )
    rot_med, rot_q95, n_rot = dmd_rotations(evals, tau=tau)
    # Eigenvector-level non-normality: the statistic an eigenvector-based test
    # (available from the same DMD modes) would use, unlike eigenvalue rotation.
    # For A=T S T^-1 with S symmetric, the eigenvectors are columns of T V_S, so
    # cond(v) equals cond(T); it is 1 for symmetric operators and grows with g.
    evec_condition = float(np.linalg.cond(v))
    departure_from_normality = float(np.linalg.norm(a.T @ a - a @ a.T))
    return {
        "nu": nu,
        "fit_rank_max": rank_max,
        "eta": reciprocity(a),
        "asymmetry": asymmetry_fraction(a),
        "eigvec_condition": evec_condition,
        "departure_from_normality": departure_from_normality,
        "eta_zscore": reciprocity(a_z),
        "asymmetry_zscore": asymmetry_fraction(a_z),
        "max_imag_eigenvalue": float(np.max(np.abs(evals.imag))),
        "max_real_eigenvalue": float(np.max(evals.real)),
        "dmd_rotation_median": rot_med,
        "dmd_rotation_q95": rot_q95,
        "dmd_modes_retained": n_rot,
        "pca_dmd_dimension": pca_k,
        "pca_dmd_rotation_median": pca_rot_med,
        "pca_dmd_rotation_q95": pca_rot_q95,
        "pca_dmd_modes_retained": pca_n_rot,
        "variance_timescale_r": r_timescale,
        "variance_fixed_lag_r": r_fixed_lag,
        "transform_condition": None,
        "covariance_spectrum": ce[:min(300, len(ce))].tolist(),
        "pc_timescales": times[:min(300, len(times))].tolist(),
        "pc_fixed_lag_acg": acg[:min(300, len(acg))].tolist(),
        "operator_eigenvalues_real": evals.real[:min(500, len(evals))].tolist(),
        "operator_eigenvalues_imag": evals.imag[:min(500, len(evals))].tolist(),
    }


def response_from_eig(inputs, projection, delays, evals, v, vi, tau):
    """Noise-free response to a brief input, up to a common scale factor."""
    base = inputs @ projection.T @ vi.T
    decay = np.exp((evals[None, :] - 1.0) * delays[:, None] / tau)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        response = (base * decay) @ v.T
    return np.real_if_close(response, tol=1000).real


def nearest_feature_accuracy(pred, targets):
    pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    targets = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.argmax(pred @ targets.T, axis=1) ==
                         np.arange(len(targets))))


def decode_features(x_train, u_train, x_test, u_test, ridge=1e-2):
    w = np.linalg.solve(x_train.T @ x_train + ridge * np.eye(x_train.shape[1]),
                        x_train.T @ u_train)
    pred = x_test @ w
    cosine = np.sum(pred * u_test, axis=1) / (
        np.linalg.norm(pred, axis=1) * np.linalg.norm(u_test, axis=1) + 1e-12
    )
    return nearest_feature_accuracy(pred, u_test), float(np.mean(cosine))


def memory_benchmark(seed, n, gain, delays, tau=0.02, feature_dim=40,
                     n_train=800, n_test=300, noise_sd=0.01):
    """Original-style zero-shot feature decoder with fixed or random readout time."""
    rng = np.random.default_rng(seed + 100_000)
    s, se, su = symmetric_operator(n, rng)
    real_a, real_e, real_v, real_vi, _ = real_spectrum_transform(
        s, se, su, rng, gain
    )
    gin_a, gin_e, gin_v, gin_vi = ginibre_operator(n, rng)
    conditions = {
        "symmetric": (s, se, su, su.T),
        "real_spectrum_nonnormal": (real_a, real_e, real_v, real_vi),
        "ginibre": (gin_a, gin_e, gin_v, gin_vi),
    }

    u_train = rng.standard_normal((n_train, feature_dim))
    u_test = rng.standard_normal((n_test, feature_dim))
    u_train /= np.linalg.norm(u_train, axis=1, keepdims=True)
    u_test /= np.linalg.norm(u_test, axis=1, keepdims=True)
    projection = rng.standard_normal((n, feature_dim)) / np.sqrt(feature_dim)

    out = {}
    for name, (_, evals, v, vi) in conditions.items():
        rows = []
        for delay in delays:
            # Shared random delays and readout noise across conditions.
            local = np.random.default_rng(seed * 10_000 + int(delay * 10_000) + 9)
            random_train = local.uniform(0.0, delay, n_train)
            random_test = local.uniform(0.0, delay, n_test)
            noise_train = noise_sd * local.standard_normal((n_train, n))
            noise_test = noise_sd * local.standard_normal((n_test, n))

            fixed_train = response_from_eig(
                u_train, projection, np.full(n_train, delay), evals, v, vi, tau
            ) + noise_train
            fixed_test = response_from_eig(
                u_test, projection, np.full(n_test, delay), evals, v, vi, tau
            ) + noise_test
            fixed_acc, fixed_cos = decode_features(
                fixed_train, u_train, fixed_test, u_test
            )

            mixed_train = response_from_eig(
                u_train, projection, random_train, evals, v, vi, tau
            ) + noise_train
            mixed_test = response_from_eig(
                u_test, projection, random_test, evals, v, vi, tau
            ) + noise_test
            mixed_acc, mixed_cos = decode_features(
                mixed_train, u_train, mixed_test, u_test
            )
            rows.append({
                "max_delay_s": float(delay),
                "fixed_time_accuracy": fixed_acc,
                "fixed_time_cosine": fixed_cos,
                "time_independent_accuracy": mixed_acc,
                "time_independent_cosine": mixed_cos,
            })
        out[name] = rows
    return out


def summarize(results):
    print("\n== Continuous-time observable summary ==")
    for name in results["conditions"]:
        rows = results["conditions"][name]
        keys = ["nu", "asymmetry", "eta", "asymmetry_zscore", "eta_zscore",
                "dmd_rotation_median", "variance_timescale_r"]
        vals = {k: np.array([r[k] for r in rows], float) for k in keys}
        print(f"{name:>25}: " + "  ".join(
            f"{k}={vals[k].mean():.3f}+/-{vals[k].std(ddof=1):.3f}"
            for k in keys
        ))
    print("\n== Time-independent zero-shot accuracy ==")
    for name in results["memory"]:
        by_delay = {}
        for seed_rows in results["memory"][name]:
            for row in seed_rows:
                by_delay.setdefault(row["max_delay_s"], []).append(
                    row["time_independent_accuracy"]
                )
        print(f"{name:>25}: " + "  ".join(
            f"{d:g}s={np.mean(v):.3f}" for d, v in sorted(by_delay.items())
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--gains", type=float, nargs="*",
                    default=[0.0, 0.10, 0.20, 0.25, 0.30, 0.40])
    ap.add_argument("--main-gain", type=float, default=0.30)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--memory-n", type=int, default=300)
    ap.add_argument("--memory-seeds", type=int, default=6)
    ap.add_argument("--memory-delays", type=float, nargs="*",
                    default=[0.1, 0.2, 0.5, 1.0, 2.0])
    ap.add_argument("--memory-noise", type=float, default=0.001)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="ma_real_spectrum_results.json")
    args = ap.parse_args()
    if args.smoke:
        args.n = 140
        args.seeds = 2
        args.gains = [0.0, args.main_gain, 0.4]
        args.memory_n = 120
        args.memory_seeds = 2
        args.memory_delays = [0.1, 0.5]

    result = {
        "config": vars(args).copy(),
        "construction": "A_g = T_g S T_g^{-1}; T_g dense SPD; continuous-time OU",
        "conditions": {f"real_gain_{g:.2f}": [] for g in args.gains},
        "memory": {
            "symmetric": [],
            "real_spectrum_nonnormal": [],
            "ginibre": [],
        },
    }
    result["conditions"]["ginibre"] = []

    for seed in range(args.seeds):
        print(f"observable seed {seed + 1}/{args.seeds}")
        rng = np.random.default_rng(seed)
        s, se, su = symmetric_operator(args.n, rng)
        for gain in args.gains:
            a, ae, av, avi, cond = real_spectrum_transform(s, se, su, rng, gain)
            metrics = operator_metrics(a, ae, av, avi, tau=args.tau)
            metrics["seed"] = seed
            metrics["gain"] = gain
            metrics["transform_condition"] = cond
            result["conditions"][f"real_gain_{gain:.2f}"].append(metrics)
        a, ae, av, avi = ginibre_operator(args.n, rng)
        metrics = operator_metrics(a, ae, av, avi, tau=args.tau)
        metrics["seed"] = seed
        metrics["gain"] = None
        result["conditions"]["ginibre"].append(metrics)

    for seed in range(args.memory_seeds):
        print(f"memory seed {seed + 1}/{args.memory_seeds}")
        rows = memory_benchmark(
            seed, args.memory_n, args.main_gain, args.memory_delays, tau=args.tau,
            noise_sd=args.memory_noise
        )
        for name, values in rows.items():
            result["memory"][name].append(values)

    path = os.path.join(HERE, args.out)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {path}")
    summarize(result)


if __name__ == "__main__":
    main()

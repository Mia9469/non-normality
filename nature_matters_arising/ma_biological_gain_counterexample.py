#!/usr/bin/env python3
"""Biologically-grounded real-spectrum non-normal counterexample (no fine-tuning).

The Authors argue that non-symmetric matrices with real eigenvalues "will not
arise from simple, random connectivity rules" and require "careful construction".
We show the opposite with the single most generic biological ingredient:
heterogeneous neuronal gain.

The Article's interaction matrix A is itself an *effective* operator,
A_eff = G W_syn, with G = diag(gain) the per-neuron gain (their linearisation
about a fixed point). Take W_syn = the Authors' own symmetric random matrix,
let each neuron have a heterogeneous positive gain g_i (lognormal -- the
default once neurons differ in excitability, size, input drive or
neuromodulatory state; NOT tuned), and apply one global critical normalization
after gain dressing. Then

    A_eff = c G W_syn = (cG)^{1/2} [(cG)^{1/2} W_syn (cG)^{1/2}] (cG)^{-1/2}

is *similar to a symmetric matrix*, so:
  * its eigenvalues are exactly real (DMD rotations = 0), and its spectral
    abscissa is set by the same scalar critical normalization used in random
    matrix models, yet
  * it is non-symmetric and non-normal, with reciprocity eta = 1/(1+CV_g^2)
    falling smoothly below 1 as gain heterogeneity grows.

For the covariance calculation we use the firing-rate linearization's upstream
input-noise form, Q = G^2, because isotropic fluctuations in h enter the
linearized dynamics as G delta h. We confirm that the covariance power-law
exponent stays in the cortical/brainwide band (0.7-0.85) for ordinary gain
heterogeneity. Thus the real-spectrum + power-law signature does NOT identify
symmetry: it is produced, untuned, by gain heterogeneity acting on the Article's
own symmetric substrate -- and exact synaptic symmetry is itself a strong
assumption under Dale's law, whereas gain heterogeneity is biologically
unavoidable.
"""
import numpy as np
import scipy.linalg as la

rng = np.random.default_rng(0)
N = 400
RHO = 0.998          # target spectral abscissa after gain dressing
TAU = 0.02           # s
DT = 0.23            # s, DMD lag (matches the Article)
SIGMAS = [0.0, 0.20, 0.25, 0.30, 0.35, 0.40]   # sd of log gain
N_SEED = 6


def symmetric_parent(n, rng):
    S = rng.standard_normal((n, n)) / np.sqrt(n)
    S = (S + S.T) / 2.0
    S *= RHO / np.max(np.linalg.eigvalsh(S))   # scale top eigenvalue to RHO
    return S


def power_law_exponent(cov, n_min=10, n_max=500):
    """Released-code 1/rank-weighted log-log fit on the covariance spectrum."""
    ce = np.sort(np.linalg.eigvalsh(cov))[::-1]
    stop = min(n_max, len(ce) // 2)
    idx = np.arange(n_min, stop)
    ranks = idx + 1.0
    y = np.log(ce[idx])
    x = np.column_stack((-np.log(ranks), np.ones_like(ranks)))
    w = 1.0 / ranks
    beta = np.linalg.solve(x.T @ (x * w[:, None]), (x * w[:, None]).T @ y)
    return float(beta[0])


def reciprocity(a):
    iu = np.triu_indices_from(a, 1)
    num = 2.0 * np.sum(a[iu] * a.T[iu])
    den = np.sum(a[iu] ** 2) + np.sum(a.T[iu] ** 2)
    return float(num / den)


def asymmetry(a):
    return float(np.linalg.norm(a - a.T) / (2 * np.linalg.norm(a)))


def dmd_max_rotation(eigvals_a):
    """Rotations per tenfold attenuation of the DMD propagator eigenvalues.

    The DMD operator B = expm((A-I)dt/tau) has eigenvalues exp((lambda_A-1)dt/tau),
    so we evaluate them from A's eigenvalues directly (exact; avoids ill-conditioned
    matrix exponentials of highly non-normal operators).
    """
    mu = np.exp((eigvals_a - 1.0) * DT / TAU)
    keep = np.abs(mu) > 0.25
    mu = mu[keep]
    rot = (np.abs(np.angle(mu)) / (2 * np.pi)) / (-np.log10(np.abs(mu)) + 1e-30)
    return float(np.max(rot)) if rot.size else 0.0


rows = {s: [] for s in SIGMAS}
for seed in range(N_SEED):
    r = np.random.default_rng(seed)
    S = symmetric_parent(N, r)
    for sg in SIGMAS:
        g = np.exp(sg * r.standard_normal(N) - sg ** 2 / 2.0)   # lognormal, mean 1
        A0 = g[:, None] * S                                      # unnormalized G W_syn
        alpha0 = float(np.max(np.linalg.eigvals(A0).real))
        A = (RHO / alpha0) * A0                                  # global critical normalization
        ev = np.linalg.eigvals(A)
        # Upstream input noise: tau dr = (-I + G W)r + G dh, with cov(dh)=I.
        Q = np.diag(g ** 2)
        cov = la.solve_continuous_lyapunov(A - np.eye(N), -Q)
        cov = 0.5 * (cov + cov.T)
        _, V = np.linalg.eig(A)
        rows[sg].append({
            "eta": reciprocity(A),
            "asym": asymmetry(A),
            "alpha0": alpha0,
            "alpha": float(np.max(ev.real)),
            "max_imag": float(np.max(np.abs(ev.imag))),
            "dmd_rot": dmd_max_rotation(ev),
            "kappaV": float(np.linalg.cond(V)),
            "nu": power_law_exponent(cov),
            "cv": float(np.std(g) / np.mean(g)),
        })


def stat(s, key):
    v = np.array([d[key] for d in rows[s]])
    return v.mean(), v.std()


print(f"Biological gain-heterogeneity counterexample (N={N}, {N_SEED} seeds, "
      f"critical-normalized alpha_eff={RHO}, upstream-noise Q=G^2)")
print(f"{'sigma_g':>8} {'eta':>14} {'asymmetry':>14} {'alpha':>14} "
      f"{'max|Im l|':>12} {'DMD rot':>10} {'kappa(V)':>12} {'nu':>14}")
for s in SIGMAS:
    eta, eta_s = stat(s, "eta")
    asy, asy_s = stat(s, "asym")
    al, al_s = stat(s, "alpha")
    mi, mi_s = stat(s, "max_imag")
    rt, rt_s = stat(s, "dmd_rot")
    kv, kv_s = stat(s, "kappaV")
    nu, nu_s = stat(s, "nu")
    print(f"{s:>8.2f} {eta:>7.3f}+/-{eta_s:<5.3f} {asy:>7.3f}+/-{asy_s:<5.3f} "
          f"{al:>7.3f}+/-{al_s:<5.3f} {mi:>11.2e} {rt:>9.3f} {kv:>6.1f}+/-{kv_s:<5.1f} "
          f"{nu:>7.3f}+/-{nu_s:<5.3f}")
print(f"\nanalytic eta = 1/(1+CV_g^2) = exp(-sigma^2): "
      f"{[round(float(np.exp(-s**2)), 3) for s in SIGMAS]}")

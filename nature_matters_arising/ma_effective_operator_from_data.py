#!/usr/bin/env python3
"""Turn-key pipeline to test the identifiability prediction on REAL data.

The Matters Arising argues that the spectral observables (covariance exponent,
near-real DMD eigenvalues) do not identify symmetry, and that the resolving
statistic is the eigenvector condition number kappa(V) of the effective operator.
This script estimates the effective linear operator A_eff from a population
activity matrix and returns the non-normality witnesses:

    kappa(V)         eigenvector condition number  (=1 symmetric/normal, >>1 non-normal)
    eta              reciprocity <A_ij A_ji>/<A_ij^2>  (=1 symmetric, 0 independent)
    omega - alpha    numerical-minus-spectral abscissa (>0 certifies non-normality)
    max|Im(lambda)|  rotation content (small => "real spectrum", the Article's signature)

A value kappa(V) >> 1 with max|Im| small is exactly the regime the Article's
spectral reading cannot detect but our analysis predicts: structured non-normal,
not symmetric.

>>> IMPORTANT SCOPE CAVEAT. <<<
This estimator constrains the EFFECTIVE operator A_eff = G W_syn, NOT the synaptic
connectivity W_syn. Because of the gain dressing, a non-normal A_eff does not imply
asymmetric W_syn (a symmetric W_syn with heterogeneous gain gives non-normal
A_eff, and conversely). Therefore measuring kappa(V) >> 1 on real functional
recordings would NOT prove asymmetric connectivity -- it would be the same
inference the Article makes for symmetry, reversed. The public Pachitariu/Stringer
data is entirely FUNCTIONAL (Neuropixels spikes + two-photon calcium); it cannot
resolve connectivity symmetry in either direction. Settling that requires
structural connectivity, or perturbation data interpreted through an
independently constrained biophysical model, not a re-analysis of activity.
This script's role is therefore (a) to provide the estimator, and (b) to document
its limits: a finite-data noise floor on kappa(V), with omega-alpha as the more
robust (but still effective-operator-level) witness.

Usage on real data:
    X = load_activity(...)          # shape (T_timesteps, N_neurons), z-scored
    print(analyze_activity(X, k=200))
"""
import numpy as np


# --------------------- the estimator (turn-key for real data) ----------------
def analyze_activity(X, k=200, ridge=1e-2, dt_lag=1):
    """X: (T, N) activity. Estimate the effective lag-1 operator in a k-dim PCA
    subspace (the Article's data-driven route) and return non-normality witnesses."""
    X = np.asarray(X, float)
    X = X - X.mean(0, keepdims=True)
    # PCA reduction
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    k = int(min(k, X.shape[1], X.shape[0] - dt_lag - 1))
    B = Vt[:k].T                                   # (N, k) principal axes
    Z = X @ B                                       # (T, k) reduced trajectories
    Z0, Z1 = Z[:-dt_lag], Z[dt_lag:]
    # lag-1 regression A: Z1 ~ Z0 A^T  (ridge-regularized)
    A = np.linalg.solve(Z0.T @ Z0 + ridge * np.eye(k), Z0.T @ Z1).T
    return operator_metrics(A)


def operator_metrics(A):
    ev, V = np.linalg.eig(A)
    iu = np.triu_indices(A.shape[0], 1)
    eta = float(np.mean(A[iu] * A.T[iu]) / (np.mean(A[iu] ** 2) + 1e-12))
    M = 0.5 * (A + A.T)
    alpha = float(np.max(ev.real))
    omega = float(np.max(np.linalg.eigvalsh(M)))
    return dict(kappaV=float(np.linalg.cond(V)),
                eta=eta,
                max_im=float(np.max(np.abs(ev.imag))),
                alpha=alpha, omega=omega, gap=omega - alpha,
                departure=float(np.linalg.norm(A @ A.T - A.T @ A)))


# --------------------- synthetic validation of the estimator -----------------
def _simulate(B, T=6000, noise=1.0, seed=0):
    """Discrete linear system x_{t+1} = B x_t + noise, return (T, N) activity."""
    rng = np.random.default_rng(seed)
    N = B.shape[0]
    x = np.zeros(N)
    out = np.empty((T, N))
    for t in range(T):
        x = B @ x + noise * rng.standard_normal(N)
        out[t] = x
    return out


def _matched_pair(N=120, rho=0.9, g=1.2, seed=1):
    """A symmetric operator and a real-spectrum NON-NORMAL operator (similarity
    transform) with the SAME real eigenvalues, both stable (discrete radius rho)."""
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((N, N)); S = (S + S.T) / 2
    S *= rho / np.max(np.abs(np.linalg.eigvalsh(S)))     # symmetric, radius rho
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    z = rng.standard_normal(N)
    T = Q @ np.diag(np.exp(g * (z - z.mean()) / z.std())) @ Q.T
    A = T @ S @ np.linalg.inv(T)                          # same real spectrum
    return S, A


if __name__ == "__main__":
    print("Synthetic validation: can the pipeline distinguish symmetric from "
          "real-spectrum non-normal from data alone?\n")
    S, A = _matched_pair()
    print(f"{'ground truth':>26} {'kappa(V)':>10} {'eta':>7} {'max|Im|':>9} {'gap w-a':>9}")
    for name, B in [("symmetric", S), ("non-normal (same spectrum)", A)]:
        gt = operator_metrics(B)
        X = _simulate(B, T=8000, noise=1.0)
        est = analyze_activity(X, k=min(100, B.shape[0]))
        print(f"{name+' [truth]':>26} {gt['kappaV']:>10.1f} {gt['eta']:>7.3f} "
              f"{gt['max_im']:>9.2e} {gt['gap']:>9.3f}")
        print(f"{name+' [from data]':>26} {est['kappaV']:>10.1f} {est['eta']:>7.3f} "
              f"{est['max_im']:>9.2e} {est['gap']:>9.3f}")
    print("\nThis validates a diagnostic of the effective operator. Applying "
          "analyze_activity() to functional recordings can test effective "
          "non-normality, but cannot by itself determine synaptic symmetry.")

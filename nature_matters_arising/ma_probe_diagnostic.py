#!/usr/bin/env python3
"""Is the point-2 undershoot a CODE BUG, a regime effect, or a spiking effect?

We compare three things for a SYMMETRIC effective operator J rescaled to a series
of spectral radii rho -> 1, all measured with the same fit window:

  analytic   : exponent of the exact Lyapunov covariance  Sigma_k = sigma^2/(1-lam_k^2)
               (the ground truth the 2/3 theory is about)
  linear-sim : simulate the LINEAR recurrence  u_{t+1} = J u_t + sigma*eps,
               estimate nu from the membrane covariance (tests the MEASUREMENT)
  spiking-sim: simulate the LIF recurrence  u_{t+1} = J s_t + sigma*eps,
               s = spike(u-th), u -= s*th, estimate nu from the SPIKE covariance
               (tests the effect of the spiking nonlinearity)

Interpretation:
  * analytic ~ 2/3 only as rho->1 (so rho=0.95 is EXPECTED to undershoot).
  * if linear-sim tracks analytic -> the covariance/fit measurement is correct
    (no estimation bug).
  * the gap spiking-sim minus linear-sim is the genuine spiking flattening -- a
    real modelling effect, not a code error.

    python ma_probe_diagnostic.py
"""
import argparse, os, sys
import numpy as np
import torch

SUPP = os.path.join(os.path.dirname(__file__), "..", "prl_cv_criticality", "supplemental")
sys.path.insert(0, SUPP)
from toy_nu_sweep import discrete_lyap_symmetric, fit_nu as fit_nu_eig


def sym_operator(N, rho, rng):
    g = torch.from_numpy(rng.standard_normal((N, N)).astype(np.float32)) / np.sqrt(N)
    W = (g + g.T) / np.sqrt(2.0)
    lam = torch.linalg.eigvalsh(W).abs().max()
    return W * (rho / (lam + 1e-9))


@torch.no_grad()
def sim(J, mode, sigma, T, burn, trials, dev):
    N = J.shape[0]; J = J.to(dev); th = 1.0
    u = torch.zeros(trials, N, device=dev); s = torch.zeros(trials, N, device=dev)
    out = []
    for t in range(T):
        eps = sigma * torch.randn(trials, N, device=dev)
        if mode == "linear":
            u = u @ J.T + eps
            out.append(u.clone())
        else:                                            # spiking LIF
            u = s @ J.T + eps
            s = (u >= th).float()
            u = u - s * th
            out.append(s)
    X = torch.stack(out)[burn:].permute(1, 0, 2).reshape(-1, N).cpu().numpy()
    return X


def nu_cov(X, n0, n1):
    X = X - X.mean(0, keepdims=True)
    var = X.var(0); X = X[:, var > 1e-9]
    if X.shape[1] < n0 + 2:
        return float("nan")
    cov = X.T @ X / max(1, X.shape[0] - 1)
    ev = np.sort(np.linalg.eigvalsh(0.5 * (cov + cov.T)).clip(min=0))[::-1]
    n1 = min(n1, len(ev)); n = np.arange(n0, n1 + 1.0)
    a, _ = np.polyfit(np.log(n), np.log(ev[n0 - 1:n1] + 1e-30), 1)
    return -a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--burn", type=int, default=120)
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n0", type=int, default=2)
    ap.add_argument("--n1", type=int, default=120)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    rhos = [0.90, 0.95, 0.98, 0.99, 0.995]
    print(f"device={dev}  N={a.N}  sigma={a.sigma}  window [{a.n0},{a.n1}]   (2/3={2/3:.3f})\n")
    print(f"{'rho':>7} | {'analytic':>9} {'linear-sim':>11} {'spiking-sim':>12}  "
          f"| {'spike-linear gap':>16}")
    print("-" * 72)
    for rho in rhos:
        an, li, sp = [], [], []
        for sd in range(a.seeds):
            rng = np.random.default_rng(sd)
            J = sym_operator(a.N, rho, rng)
            an.append(fit_nu_eig(discrete_lyap_symmetric(J.numpy()), a.n0, a.n1)[0])
            li.append(nu_cov(sim(J, "linear", a.sigma, a.T, a.burn, a.trials, dev), a.n0, a.n1))
            sp.append(nu_cov(sim(J, "spiking", a.sigma, a.T, a.burn, a.trials, dev), a.n0, a.n1))
        an, li, sp = np.array(an), np.array(li), np.array(sp)
        print(f"{rho:>7.3f} | {np.nanmean(an):>9.3f} {np.nanmean(li):>11.3f} "
              f"{np.nanmean(sp):>12.3f}  | {np.nanmean(sp) - np.nanmean(li):>16.3f}")
    print("\nIf analytic ~ linear-sim across rho -> the measurement is correct (no bug);")
    print("the spiking-sim deficit is a real nonlinearity effect, and 2/3 needs rho->1.")


if __name__ == "__main__":
    main()

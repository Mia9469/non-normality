#!/usr/bin/env python3
"""Matters-Arising controls for the two gaps a referee will hit first.

It reuses the trained recurrent-LIF infrastructure in
prl_cv_criticality/supplemental/recurrent_snn_scale.py and adds the two
measurements the current draft is missing:

  CONTROL A  -- UNTRAINED baseline (the decisive control for "2/3 emerges from
                training"). For each init regime {subcritical, standard, critical,
                supercritical} and seed, we measure the spontaneous exponent
                nu_spont and the operator symmetry BEFORE any training, then again
                AFTER training. If nu_spont is already ~2/3 at a symmetric
                near-critical init, the honest claim is "training MAINTAINS 2/3"
                (and the exponent does NOT distinguish init from learning) --
                which is exactly the non-identifiability point of the revised MA.

  CONTROL B  -- DMD symmetry of the TRAINED operator, measured the same way the
                original Article does (Fig. 3): fit a lagged linear map on the
                PC-reduced spontaneous states and read the complex parts of its
                eigenvalues as "rotations per tenfold attenuation". This closes
                the loop between point (2) [trained nets reach the band] and point
                (1) [the band diagnoses symmetry] by showing the trained operator
                is in fact near-symmetric (low rotation), rather than merely
                asserting it.

Run smoke (CPU, minutes):
    python ma_controls.py --smoke
Full (GPU):
    python ma_controls.py --epochs 30 --hidden 512 --seeds 5 \
        --inits subcritical standard critical supercritical
"""
import argparse, json, os, sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

SUPP = os.path.join(os.path.dirname(__file__), "..", "prl_cv_criticality", "supplemental")
sys.path.insert(0, SUPP)
from recurrent_snn_scale import (  # noqa: E402  reuse the exact model used in Fig 1b
    RecurrentSNN, loaders, to_seq, reg_loss, jac_stats, evaluate, fit_nu,
)

INIT_RHO = {"subcritical": 0.5, "standard": 0.9, "critical": 1.0, "supercritical": 1.3}


# ----------------------------- spontaneous probe -----------------------------
@torch.no_grad()
def spont_states(model, dev, N, in_dim=28, T=200, burn=40, trials=64, noise=0.2):
    """Return (a) pooled z-scored states for the covariance exponent, and
    (b) a list of per-trial state arrays (T-burn, N_active) for DMD (so the
    lagged pairs never cross a trial boundary)."""
    x = noise * torch.randn(trials, T, in_dim, device=dev)
    _, S, _ = model(x)                                   # (T, B, N)
    A = S[burn:].permute(1, 0, 2).float().cpu().numpy()  # (B, T', N)
    flat = A.reshape(-1, N)
    flat = flat[np.isfinite(flat).all(axis=1)]
    if flat.shape[0] < 10:
        return None, None, 0.0
    mu = flat.mean(0, keepdims=True)
    var = flat.var(0); active = var > 1e-9
    sd = np.sqrt(var) + 1e-9
    pooled = (flat[:, active] - mu[:, active]) / sd[active]
    per_trial = [((A[b] - mu) / sd)[:, active] for b in range(A.shape[0])]
    return pooled, per_trial, float(active.mean())


def nu_from_pooled(pooled):
    if pooled is None or pooled.shape[1] < 6:
        return float("nan"), float("nan")
    cov = pooled.T @ pooled / max(1, pooled.shape[0] - 1)
    ev = np.linalg.eigvalsh(0.5 * (cov + cov.T)).clip(min=0)
    return fit_nu(ev, 2, min(200, pooled.shape[1] // 3))


# ------------------------------- DMD (Control B) -----------------------------
def dmd_rotation(per_trial, dt=1, k=100, ridge=0.1, mag_min=0.25):
    """Mirror of Pachitariu et al. Fig. 3: lagged DMD on PC-reduced states.

    Fit  Z_{t+dt} = A Z_t  (ridge) in a k-dim PCA subspace, take eig(A), and
    summarise the *rotational* content with the paper's metric
        n_rot10(lambda) = log(0.1)/log|lambda| * angle(lambda)
    (rotations accrued per tenfold amplitude attenuation). Symmetric dynamics
    -> near-real eigenvalues -> n_rot10 ~ 0; non-normal -> larger n_rot10.
    Returns median n_rot10, mean |Im|/|lambda|, and the fraction with |Im|>0.05.
    """
    if per_trial is None or len(per_trial) == 0:
        return dict(n_rot10_med=float("nan"), imag_frac=float("nan"),
                    imag_mean=float("nan"), k=0)
    # keep only usable trials (divergent/supercritical runs can produce inf/nan
    # *or* huge-but-finite values that overflow the float matmuls below). Sanitise
    # non-finite entries and clip to a generous range (z-scored states are O(10)).
    def _clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e4, 1e4)
    per_trial = [_clean(X) for X in per_trial if X.shape[0] > dt + 2]
    if len(per_trial) == 0:
        return dict(n_rot10_med=float("nan"), imag_frac=float("nan"),
                    imag_mean=float("nan"), k=0)
    Xall = np.concatenate(per_trial, 0)
    Xall = Xall - Xall.mean(0, keepdims=True)
    k = int(min(k, Xall.shape[1], Xall.shape[0] - 1))
    if k < 3:
        return dict(n_rot10_med=float("nan"), imag_frac=float("nan"),
                    imag_mean=float("nan"), k=k)
    # global PCA basis
    _, _, Vt = np.linalg.svd(Xall, full_matrices=False)
    P = Vt[:k].T
    Z0, Z1 = [], []
    for X in per_trial:
        Z = (X - X.mean(0, keepdims=True)) @ P
        Z0.append(Z[:-dt]); Z1.append(Z[dt:])
    Z0 = np.concatenate(Z0, 0); Z1 = np.concatenate(Z1, 0)
    G = Z0.T @ Z0 + ridge * np.eye(k)
    A = np.linalg.solve(G, Z0.T @ Z1).T                  # Z1 ~ Z0 A^T
    ev = np.linalg.eigvals(A)
    mag = np.abs(ev); ang = np.abs(np.angle(ev))
    keep = mag > mag_min
    if keep.sum() == 0:
        keep = mag > 0
    magk, angk, evk = mag[keep], ang[keep], ev[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        n_rot10 = np.log(0.1) / np.log(np.clip(magk, 1e-6, 0.999999)) * angk
    imag_ratio = np.abs(evk.imag) / (np.abs(evk) + 1e-12)
    return dict(n_rot10_med=float(np.nanmedian(n_rot10)),
                imag_frac=float(np.mean(np.abs(evk.imag) > 0.05)),
                imag_mean=float(np.mean(imag_ratio)), k=k)


# --------------------------- one init x seed cell ----------------------------
def measure(model, dev, a):
    pooled, per_trial, alive = spont_states(model, dev, a.hidden, in_dim=a.in_dim)
    nu, r2 = nu_from_pooled(pooled)
    _, rho, asym = jac_stats(model, dev, a.in_dim)        # Frobenius anti-sym fraction
    dmd = dmd_rotation(per_trial, dt=a.dmd_dt, k=a.dmd_k)
    return dict(nu_spont=nu, nu_r2=r2, rho_eff=rho, asym=asym,
                dead=1 - alive, **{f"dmd_{kk}": vv for kk, vv in dmd.items()})


def train_inplace(model, name, a, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    (tr, te), _ = loaders(a.batch, a.smoke, a.task, nclass=a.classes, in_dim=a.in_dim)
    for _ in range(a.epochs):
        model.train()
        for img, y in tr:
            x = to_seq(img).to(dev); y = y.to(dev)
            logits, S, G = model(x)
            loss = F.cross_entropy(logits, y) + reg_loss(name, S, model, G, a)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    acc, _ = evaluate(model, te, dev)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--cond", default="baseline",
                    help="regularizer condition used during training (baseline/sigma2_fm/pop_cv/spectral)")
    ap.add_argument("--inits", nargs="*", default=["subcritical", "standard", "critical", "supercritical"])
    ap.add_argument("--task", choices=["mnist", "synth", "clusters"], default="mnist")
    ap.add_argument("--dmd-dt", type=int, default=1)
    ap.add_argument("--dmd-k", type=int, default=100)
    ap.add_argument("--out", default="ma_controls_results.json")
    c = ap.parse_args()
    if c.smoke:
        c.epochs, c.hidden, c.seeds = 2, 128, 2
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    in_dim = 28 if c.task == "mnist" else 28
    a = SimpleNamespace(epochs=c.epochs, hidden=c.hidden, batch=c.batch, lr=2e-3,
                        smoke=c.smoke, task=c.task, classes=10, in_dim=in_dim,
                        lam_cv=0.02, lam_var=0.5, lam_spec=0.02, gamma=20.0,
                        mu_target=0.10, rho_target=1.02, dmd_dt=c.dmd_dt, dmd_k=c.dmd_k,
                        init_rho=0.9)
    print(f"device={dev} hidden={c.hidden} epochs={c.epochs} seeds={c.seeds} cond={c.cond}")
    print(f"{'init':>13} {'seed':>4} {'phase':>9} {'acc':>6} {'rho':>6} {'asym':>6} "
          f"{'nu':>6} {'R2':>5} {'dmd_rot':>8} {'dmd_im%':>7}")

    results = {}
    for init in c.inits:
        a.init_rho = INIT_RHO[init]
        cells = {"init": [], "trained": []}
        for seed in range(c.seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            model = RecurrentSNN(a.in_dim, a.hidden, a.classes, init_rho=a.init_rho).to(dev)
            m0 = measure(model, dev, a); m0["acc"] = float("nan")           # UNTRAINED
            print(f"{init:>13} {seed:>4} {'init':>9} {m0['acc']:>6} {m0['rho_eff']:>6.3f} "
                  f"{m0['asym']:>6.3f} {m0['nu_spont']:>6.3f} {m0['nu_r2']:>5.2f} "
                  f"{m0['dmd_n_rot10_med']:>8.3f} {100*m0['dmd_imag_frac']:>6.1f}")
            acc = train_inplace(model, c.cond, a, dev)                       # TRAIN
            m1 = measure(model, dev, a); m1["acc"] = acc
            print(f"{init:>13} {seed:>4} {'trained':>9} {m1['acc']:>6.3f} {m1['rho_eff']:>6.3f} "
                  f"{m1['asym']:>6.3f} {m1['nu_spont']:>6.3f} {m1['nu_r2']:>5.2f} "
                  f"{m1['dmd_n_rot10_med']:>8.3f} {100*m1['dmd_imag_frac']:>6.1f}")
            cells["init"].append(m0); cells["trained"].append(m1)
        results[init] = cells

    # ---- compact summary: did 2/3 already exist before training? ----
    def agg(cells, key):
        v = [d[key] for d in cells if np.isfinite(d[key])]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)
    print("\n== nu_spont: untrained vs trained (mean +/- s.d. over seeds) ==")
    print(f"{'init':>13} {'nu_init':>14} {'nu_trained':>14} {'asym_tr':>9} {'dmd_rot_tr':>11}")
    for init in c.inits:
        ni, si = agg(results[init]['init'], 'nu_spont')
        nt, st = agg(results[init]['trained'], 'nu_spont')
        at, _ = agg(results[init]['trained'], 'asym')
        dr, _ = agg(results[init]['trained'], 'dmd_n_rot10_med')
        print(f"{init:>13} {ni:>7.3f}+/-{si:<5.3f} {nt:>7.3f}+/-{st:<5.3f} {at:>9.3f} {dr:>11.3f}")
    print(f"\nReference: nu=2/3={2/3:.3f}; cortex band 0.68-0.78")
    print("Read-out: if nu_init is already ~2/3 at a symmetric near-critical init,")
    print("the exponent does NOT distinguish initialization from learning (degeneracy).")

    json.dump(results, open(os.path.join(os.path.dirname(__file__) or ".", c.out), "w"), indent=2)
    print(f"saved -> {c.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the V4 combined Matters Arising figure.

Four panels synthesising the two identifiability gaps:
  a -- Degeneracy surface nu(eta, alpha_eff)         [Gap 1]
  b -- Operator eigenvalue spectra                    [Gap 2]
  c -- Covariance exponent vs. asymmetry sweep        [Gap 2 validation]
  d -- Time-independent zero-shot memory              [Gap 2 validation]

Uses data from ma_degeneracy_surface.json and ma_real_spectrum_results.json.
"""
import json
import os

import os as _os
if _os.environ.get("MPLBACKEND") != "pdf":
    _os.environ["MPLBACKEND"] = "pdf"
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(__file__) or "."
DEGEN = os.path.join(HERE, "ma_degeneracy_surface.json")
REAL = os.path.join(HERE, "ma_real_spectrum_results.json")
OUT = os.path.join(HERE, "figures", "fig_v4")

CORTEX = (0.68, 0.78)
TWO3 = 2 / 3

plt.rcParams.update({
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


def load_json(path):
    with open(path) as f:
        return json.load(f)


def panel_a(ax, d):
    """Continuous-time degeneracy surface nu(eta, alpha_eff) — Gap 1."""
    if d is None:
        ax.text(0.5, 0.5, "run ma_degeneracy_surface.py", ha="center",
                va="center", transform=ax.transAxes, color="0.5", fontsize=9)
        ax.set_title("a", loc="left", fontweight="bold")
        return

    etas = np.array(d["etas"])
    alphas = np.array(d["alphas"])
    NU = np.array(d["nu"])

    # Order etas increasing for imshow (origin="lower")
    order = np.argsort(etas)
    etas_sorted = etas[order]
    NU_sorted = NU[order]

    im = ax.imshow(NU_sorted, origin="lower", aspect="auto", cmap="viridis",
                   extent=[alphas.min(), alphas.max(), etas_sorted.min(),
                           etas_sorted.max()])

    # Iso-nu contours
    try:
        cs = ax.contour(alphas, etas_sorted, NU_sorted,
                        levels=[TWO3, 0.78, 1.0, 1.25],
                        colors="w", linewidths=1.0)
        ax.clabel(cs, fmt="%.2f", fontsize=7)
        # Highlight cortical band as a filled band + bright outline so it
        # reads as a diagonal curve (not a point) in print.
        ax.contourf(alphas, etas_sorted, NU_sorted,
                    levels=list(CORTEX), colors=["red"], alpha=0.30)
        ax.contour(alphas, etas_sorted, NU_sorted,
                   levels=list(CORTEX), colors="red", linewidths=1.3)
    except Exception:
        pass

    ax.set_xlabel(r"spectral abscissa $\alpha_{\rm eff}$")
    ax.set_ylabel(r"reciprocity $\eta$  (1 = symmetric)")
    ax.set_title("a", loc="left", fontweight="bold")

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"covariance exponent $\nu$")

    # Annotate the two anchoring columns
    ax.text(0.999, 1.02, r"$\eta{=}1\to 2/3$", ha="right", va="bottom",
            fontsize=6.5, color="white", fontweight="bold")
    ax.text(0.999, 0.02, r"$\eta{=}0\to 1.25$", ha="right", va="bottom",
            fontsize=6.5, color="white", fontweight="bold")


def panel_b(ax, d):
    """Operator eigenvalue spectra — Gap 2."""
    sym_key = "real_gain_0.00"
    main_key = "real_gain_0.30"
    gin_key = "ginibre"

    colors = {sym_key: "#1f77b4", main_key: "#d62728", gin_key: "#777777"}
    labels = {
        sym_key: "symmetric",
        main_key: "real-spectrum non-normal",
        gin_key: "Ginibre",
    }
    markers = {sym_key: "o", main_key: "x", gin_key: "."}
    sizes = {sym_key: 9, main_key: 9, gin_key: 5}
    zorders = {gin_key: 1, sym_key: 2, main_key: 3}

    for key in [gin_key, sym_key, main_key]:
        if key not in d["conditions"] or not d["conditions"][key]:
            continue
        row = d["conditions"][key][0]
        re_vals = np.array(row["operator_eigenvalues_real"])
        im_vals = np.array(row["operator_eigenvalues_imag"])
        ax.scatter(re_vals, im_vals, s=sizes[key], marker=markers[key],
                   color=colors[key], alpha=0.72, linewidths=0.7,
                   label=labels[key], zorder=zorders[key])

    ax.axhline(0, color="0.8", lw=0.6)
    ax.set(xlabel=r"$\mathrm{Re}(\lambda_A)$",
           ylabel=r"$\mathrm{Im}(\lambda_A)$",
           title="b  Real spectra do not identify symmetry")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(0.98, 0.04,
            "DMD rotations = 0\nfor both real-spectrum models",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))


def panel_c(ax, d):
    """Covariance exponent vs asymmetry — Gap 2 validation."""
    gain_keys = [k for k in d["conditions"] if k.startswith("real_gain_")]
    gain_keys.sort(key=lambda x: float(x.rsplit("_", 1)[1]))

    gains = np.array([float(k.rsplit("_", 1)[1]) for k in gain_keys])
    nu_vals = np.array([[r["nu"] for r in d["conditions"][k]]
                         for k in gain_keys])
    asym_vals = np.array([[r["asymmetry"] for r in d["conditions"][k]]
                           for k in gain_keys])

    ax.axhspan(CORTEX[0], CORTEX[1], color="#ef9a9a", alpha=0.30,
               label="cortical band")
    ax.errorbar(asym_vals.mean(1), nu_vals.mean(1),
                xerr=asym_vals.std(1), yerr=nu_vals.std(1),
                color="#d62728", marker="o", ms=4, capsize=2)

    for x, y, g in zip(asym_vals.mean(1), nu_vals.mean(1), gains):
        ax.text(x + 0.008, y, f"{g:g}", fontsize=6.5, color="#8b1a1a")

    ax.set(xlabel=r"operator asymmetry  $\|A-A^\top\|_F/(2\|A\|_F)$",
           ylabel=r"covariance exponent $\nu$",
           title="c  Scaling persists away from symmetry")
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def panel_d(ax, d):
    """Time-independent zero-shot memory — Gap 2 validation."""
    memory_colors = {
        "symmetric": "#1f77b4",
        "real_spectrum_nonnormal": "#d62728",
        "ginibre": "#777777",
    }
    memory_labels = {
        "symmetric": "symmetric",
        "real_spectrum_nonnormal": "real-spectrum non-normal",
        "ginibre": "Ginibre",
    }

    for name, seed_rows in d["memory"].items():
        delays = np.array([r["max_delay_s"] for r in seed_rows[0]])
        values = np.array([[r["time_independent_accuracy"] for r in rows]
                           for rows in seed_rows])
        ax.plot(delays, values.mean(0), "-o", color=memory_colors[name],
                ms=3.5, lw=1.4, label=memory_labels[name])
        ax.fill_between(delays,
                        values.mean(0) - values.std(0),
                        values.mean(0) + values.std(0),
                        color=memory_colors[name], alpha=0.15)

    ax.set(xlabel="maximum randomized readout delay (s)",
           ylabel="zero-shot accuracy",
           title="d  Time-independent zero-shot memory",
           ylim=(0, 1.03))
    ax.legend(frameon=False, fontsize=7)


def main():
    dd = load_json(DEGEN) if os.path.exists(DEGEN) else None
    if dd is not None and "alphas" not in dd:
        print("STALE degeneracy surface: rerun ma_degeneracy_surface.py "
              "for the continuous-time alpha_eff grid")
        dd = None
    dr = load_json(REAL) if os.path.exists(REAL) else None

    fig = plt.figure(figsize=(11.0, 7.5))
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[1.0, 1.0],
                  hspace=0.35, wspace=0.32,
                  left=0.07, right=0.96, top=0.95, bottom=0.07)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, dd)
    panel_b(ax_b, dr)
    panel_c(ax_c, dr)
    panel_d(ax_d, dr)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # PDF first (doesn't need PIL), then try PNG
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print(f"saved {OUT}.pdf")
    try:
        fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
        print(f"saved {OUT}.png")
    except Exception as e:
        print(f"PNG skipped (PIL/iCloud issue): {e}")
    print(f"  degeneracy surface: {'ok' if dd else 'MISSING'}")
    print(f"  counterexample:     {'ok' if dr else 'MISSING'}")


if __name__ == "__main__":
    main()

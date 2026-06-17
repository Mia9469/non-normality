#!/usr/bin/env python3
"""Analyze Shiu/FlyWire activity with the official MouseLand/critical_init code.

The primary functional readout is approximately 22 Hz binned spikes. This
script directly loads and calls the pinned official implementations:

  * powerlaw.SVCA2
  * powerlaw.fit_powerlaw_exp
  * lyapun.dmd

Structural W and the fitted DMD propagator remain different mathematical
objects; their reciprocity scalars are not expected to be equal.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np

from connectome_controls import load_connectome


HERE = Path(__file__).resolve().parent
DEFAULT_CRITICAL_INIT = HERE / "critical_init_official"
PINNED_CRITICAL_INIT_COMMIT = "2c15edf4e770165fc3962dcc3920c8bcaf555bed"


def load_official_critical_init(root):
    """Load the pinned, unmodified official powerlaw.py and lyapun.py."""
    root = root.resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            [
                "git", "-C", str(root), "status", "--porcelain", "--",
                "powerlaw.py", "lyapun.py",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        raise RuntimeError(f"{root} is not a verifiable official git checkout") from exc
    if commit != PINNED_CRITICAL_INIT_COMMIT:
        raise RuntimeError(
            f"critical_init commit {commit} does not match pinned "
            f"{PINNED_CRITICAL_INIT_COMMIT}; run setup_critical_init.sh"
        )
    if dirty:
        raise RuntimeError(
            "official critical_init powerlaw.py or lyapun.py has local changes; "
            "use a clean pinned checkout"
        )
    modules = {}
    for name in ["powerlaw", "lyapun"]:
        path = root / f"{name}.py"
        if not path.exists():
            raise FileNotFoundError(
                f"missing official critical_init source: {path}. "
                "Run setup_critical_init.sh."
            )
        spec = importlib.util.spec_from_file_location(f"critical_init_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["powerlaw"], modules["lyapun"], commit


def resolve_device(torch, requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def zscore_neurons(x):
    """Input time x neurons; return z-scored time x active-neurons."""
    x = np.asarray(x, np.float64)
    if not np.isfinite(x).all():
        raise ValueError("activity contains NaN or infinite values")
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    keep = sd > 1e-9
    return x[:, keep] / sd[keep], keep


def official_ready_activity(x):
    """Return active-neuron x time float32 activity, as official code expects."""
    xz, keep = zscore_neurons(x)
    return np.asarray(xz.T, np.float32), keep


def fit_powerlaw(values, official_powerlaw, rank_min=10, rank_max=500):
    """Call official powerlaw.fit_powerlaw_exp without changing valid inputs."""
    values = np.asarray(values, float)
    stop = min(rank_max, len(values) // 2, len(values))
    if stop <= rank_min:
        return {"nu": float("nan"), "rank_min": rank_min, "rank_max": stop}
    indices = np.arange(rank_min, stop, dtype=int)
    selected = values[indices]
    if not np.isfinite(selected).all() or np.any(selected == 0):
        raise ValueError("power-law fit range contains zero or non-finite values")
    alpha, _ = official_powerlaw.fit_powerlaw_exp(values.copy(), indices.copy())
    return {"nu": float(alpha), "rank_min": rank_min, "rank_max": stop}


def direct_spectrum(x, official_powerlaw, device):
    sp, _ = official_ready_activity(x)
    tensor = official_powerlaw.torch.from_numpy(sp).to(device)
    values, _ = official_powerlaw.compute_evals(tensor)
    return values


def svca2_spectrum(x, official_powerlaw, device, repeats=1, seed=0):
    """Call official SVCA2; repeats>1 averages independent official splits."""
    sp, _ = official_ready_activity(x)
    tensor = official_powerlaw.torch.from_numpy(sp).to(device)
    spectra = []
    for repeat in range(repeats):
        np.random.seed(seed + repeat)
        spectra.append(official_powerlaw.SVCA2(tensor))
    length = min(map(len, spectra))
    return np.mean([spectrum[:length] for spectrum in spectra], axis=0)


def rotation_summary(eigenvalues, threshold=0.25):
    """Match the Article's DMD rotation calculation on the positive half-plane."""
    eigenvalues = np.asarray(eigenvalues)
    keep = np.abs(eigenvalues) > threshold
    values = eigenvalues[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        rotations = (np.angle(values) / (2 * np.pi)) / (
            -np.log10(np.abs(values))
        )
    rotations = rotations[(values.imag >= 0) & np.isfinite(rotations)]
    percentiles = (
        np.percentile(rotations, [5, 25, 50, 75, 95])
        if len(rotations) else np.full(5, np.nan)
    )
    return {
        "rotation_count": int(len(rotations)),
        "rotation_p05": float(percentiles[0]),
        "rotation_p25": float(percentiles[1]),
        "rotation_median": float(percentiles[2]),
        "rotation_p75": float(percentiles[3]),
        "rotation_p95": float(percentiles[4]),
        "unstable_fraction": float(np.mean(np.abs(eigenvalues) >= 1.0)),
    }


def dmd(x, dt_s, official_lyapun, device, lag_s=0.23, modes=1000,
        ridge=0.1, nt=5000):
    """Call official lyapun.dmd, then add coordinate-dependent diagnostics."""
    sp, _ = official_ready_activity(x)
    lag = max(1, int(round(lag_s / dt_s)))
    if sp.shape[1] - lag <= 50:
        raise ValueError("DMD requires more than 50 usable samples after the lag")
    operator, eigenvalues, eigenvectors = official_lyapun.dmd(
        sp, lam=ridge, delta=lag, nt=nt, n_comps=modes, device=device
    )
    count = operator.shape[0]
    iu = np.triu_indices(count, 1)
    eta_coordinate = float(
        2.0 * np.sum(operator[iu] * operator.T[iu])
        / (np.sum(operator[iu] ** 2) + np.sum(operator.T[iu] ** 2) + 1e-30)
    )
    symmetric_part = 0.5 * (operator + operator.T)
    result = {
        "lag_bins": lag,
        "lag_s": lag * dt_s,
        "modes": count,
        "ridge": ridge,
        "official_nt": nt,
        "max_abs_imag": float(np.max(np.abs(eigenvalues.imag))),
        "eta_pca_coordinate": eta_coordinate,
        "eigenvector_condition": float(np.linalg.cond(eigenvectors)),
        "numerical_minus_spectral_abscissa": float(
            np.max(np.linalg.eigvalsh(symmetric_part))
            - np.max(eigenvalues.real)
        ),
    }
    result.update(rotation_summary(eigenvalues))
    return result


def bin_spikes(spike_i, spike_t, subset, dt_s, duration):
    bins = int(np.ceil(duration / dt_s))
    position = {int(neuron): j for j, neuron in enumerate(subset)}
    keep = np.isin(spike_i, subset)
    ids, times = spike_i[keep], spike_t[keep]
    columns = np.fromiter((position[int(x)] for x in ids), int, count=len(ids))
    rows = np.minimum((times / dt_s).astype(int), bins - 1)
    matrix = np.zeros((bins, len(subset)), np.float32)
    np.add.at(matrix, (rows, columns), 1.0)
    return matrix


def structural_metrics(path, control):
    df = load_connectome(path, control)
    df = df[df.pre != df.post]
    n = int(max(df.pre.max(), df.post.max()) + 1)
    reverse = df.rename(columns={
        "pre": "post", "post": "pre",
        "unsigned": "unsigned_reverse", "signed": "signed_reverse",
    })
    paired = df.merge(reverse, on=["pre", "post"], how="left").fillna(0)
    density = float(len(df) / (n * (n - 1)))
    reciprocal_fraction = float(np.mean(paired["unsigned_reverse"] != 0))
    out = {
        "control": control,
        "N": n,
        "directed_edges": int(len(df)),
        "density": density,
        "reciprocated_edge_fraction": reciprocal_fraction,
        "reciprocal_support_enrichment_vs_independent": reciprocal_fraction / density,
    }
    for label in ["unsigned", "signed"]:
        x = paired[label].to_numpy(np.float64)
        y = paired[f"{label}_reverse"].to_numpy(np.float64)
        out[f"eta_{label}"] = float(np.sum(x * y) / (np.sum(x * x) + 1e-30))
        out[f"cosine_reciprocity_{label}"] = float(
            np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y) + 1e-30)
        )
    return out


def analyze_activity(name, x, dt_s, args, ridge, official_powerlaw,
                     official_lyapun, device):
    svca = svca2_spectrum(
        x, official_powerlaw, device, args.svca_repeats, args.seed
    )
    direct = direct_spectrum(x, official_powerlaw, device)
    return {
        "name": name,
        "shape_time_by_neuron": list(x.shape),
        "active_neurons": int(np.count_nonzero(np.std(x, axis=0) > 1e-9)),
        "svca2": fit_powerlaw(
            svca, official_powerlaw, args.rank_min, args.rank_max
        ),
        "direct": fit_powerlaw(
            direct, official_powerlaw, args.rank_min, args.rank_max
        ),
        "dmd": dmd(
            x, dt_s, official_lyapun, device, args.lag, args.dmd_modes,
            ridge, args.dmd_nt
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", type=Path, default=Path("spont_poisson.npz"))
    parser.add_argument("--con", type=Path,
                        default=HERE / "2023_03_23_connectivity_630_final.parquet")
    parser.add_argument("--out", type=Path, default=Path("spont_analysis.json"))
    parser.add_argument("--skip-structure", action="store_true")
    parser.add_argument("--critical-init-root", type=Path,
                        default=DEFAULT_CRITICAL_INIT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--svca-repeats", type=int, default=1,
                        help="1 exactly matches one official SVCA2 call")
    parser.add_argument("--rank-min", type=int, default=10)
    parser.add_argument("--rank-max", type=int, default=500)
    parser.add_argument("--dmd-modes", type=int, default=1000)
    parser.add_argument("--dmd-nt", type=int, default=5000,
                        help="official mouse-analysis lyapun.dmd block length")
    parser.add_argument("--lag", type=float, default=0.23, help="DMD lag (s)")
    parser.add_argument("--ridge-spikes", type=float, default=0.1)
    parser.add_argument("--ridge-voltage", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-active-neurons", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.svca_repeats < 1:
        raise ValueError("--svca-repeats must be at least 1")
    if args.dmd_modes < 1 or args.dmd_nt < 1:
        raise ValueError("--dmd-modes and --dmd-nt must be at least 1")
    data = np.load(args.input, allow_pickle=False)
    dt_s = float(data["dt_bin"]) / 1000.0
    duration = float(data["t_run"])
    subset = np.asarray(data["sub"])
    if "spike_counts" in data:
        spikes = np.asarray(data["spike_counts"]).T
    else:
        spikes = bin_spikes(
            np.asarray(data["spk_i"]), np.asarray(data["spk_t"]),
            subset, dt_s, duration,
        )
    active_spikes = int(np.count_nonzero(np.std(spikes, axis=0) > 1e-9))
    if active_spikes < args.min_active_neurons:
        raise RuntimeError(
            f"only {active_spikes} spike-count neurons are active; refusing "
            "SVCA2/DMD analysis. Recalibrate Poisson background first."
        )
    official_powerlaw, official_lyapun, official_commit = (
        load_official_critical_init(args.critical_init_root)
    )
    device = resolve_device(official_lyapun.torch, args.device)
    voltage = np.asarray(data["v"]).T
    rate_percentiles = (
        np.asarray(data["rate_percentiles"], float)
        if "rate_percentiles" in data else np.full(5, np.nan)
    )
    connectome_control = (
        str(data["connectome_control"]) if "connectome_control" in data else "original"
    )
    result = {
        "interpretation": (
            "Forward-model calibration. Structural eta(W) and eta of a "
            "z-scored/PCA-reduced DMD propagator are not expected to be equal."
        ),
        "official_critical_init": {
            "repository": "https://github.com/MouseLand/critical_init",
            "commit": official_commit,
            "SVCA2": "powerlaw.SVCA2",
            "fit_powerlaw_exp": "powerlaw.fit_powerlaw_exp",
            "compute_evals": "powerlaw.compute_evals",
            "dmd": "lyapun.dmd",
            "device": str(device),
            "svca_repeats": args.svca_repeats,
            "single_call_exact": args.svca_repeats == 1,
        },
        "simulation": {
            "input": str(args.input),
            "noise": str(data["noise"]),
            "simulation_backend": (
                str(data["simulation_backend"])
                if "simulation_backend" in data else "unknown"
            ),
            "connectome_control": connectome_control,
            "recurrent_gain": (
                float(data["recurrent_gain"]) if "recurrent_gain" in data else 1.0
            ),
            "duration_s": duration,
            "dt_s": dt_s,
            "mean_rate_hz": float(data["mean_rate"]),
            "rate_p05_hz": float(rate_percentiles[0]),
            "rate_p25_hz": float(rate_percentiles[1]),
            "rate_median_hz": float(rate_percentiles[2]),
            "rate_p75_hz": float(rate_percentiles[3]),
            "rate_p95_hz": float(rate_percentiles[4]),
            "silent_fraction": float(data["silent_fraction"]) if "silent_fraction" in data else None,
            "binned_fano": float(data["binned_fano"]) if "binned_fano" in data else None,
            "pooled_binned_fano": (
                float(data["pooled_binned_fano"])
                if "pooled_binned_fano" in data
                else float(data["binned_fano"]) if "binned_fano" in data else None
            ),
            "median_active_neuron_fano": (
                float(data["median_active_neuron_fano"])
                if "median_active_neuron_fano" in data else None
            ),
            "target_rate_band_fraction": (
                float(data["target_rate_band_fraction"])
                if "target_rate_band_fraction" in data else None
            ),
            "high_rate_fraction": (
                float(data["high_rate_fraction"])
                if "high_rate_fraction" in data else None
            ),
            "block_rate_cv": float(data["block_rate_cv"]),
            "recorded_neurons": int(len(subset)),
            "seed": int(data["seed"]) if "seed" in data else None,
            "r_bg_hz": float(data["r_bg"]) if "r_bg" in data else None,
            "bg_sources": int(data["bg_sources"]) if "bg_sources" in data else None,
            "w_bg_in_w_syn": float(data["w_bg"]) if "w_bg" in data else None,
            "r_inh_hz": float(data["r_inh"]) if "r_inh" in data else None,
            "gi": float(data["gi"]) if "gi" in data else None,
            "poisson_threshold_z": (
                float(data["poisson_threshold_z"])
                if "poisson_threshold_z" in data else None
            ),
        },
        "functional": [
            analyze_activity(
                "binned_spikes_primary", spikes, dt_s, args, args.ridge_spikes,
                official_powerlaw, official_lyapun, device
            ),
            analyze_activity(
                "membrane_voltage_secondary", voltage, dt_s, args,
                args.ridge_voltage, official_powerlaw, official_lyapun, device
            ),
        ],
    }
    if not args.skip_structure:
        result["structural"] = structural_metrics(args.con, connectome_control)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)

    print(f"=== official critical_init {official_commit} on {device} ===")
    if "structural" in result:
        s = result["structural"]
        print(f"structural: density={s['density']:.6f}, "
              f"reciprocated={s['reciprocated_edge_fraction']:.3f}, "
              f"enrichment={s['reciprocal_support_enrichment_vs_independent']:.1f}x, "
              f"eta_unsigned={s['eta_unsigned']:.3f}, eta_signed={s['eta_signed']:.3f}")
    for row in result["functional"]:
        dmd_row = row["dmd"]
        print(f"{row['name']}: active={row['active_neurons']}, "
              f"SVCA2 nu={row['svca2']['nu']:.3f}, "
              f"direct nu={row['direct']['nu']:.3f}, "
              f"DMD lag={dmd_row['lag_s']:.3f}s, "
              f"rotation median={dmd_row['rotation_median']:.3f}, "
              f"p95={dmd_row['rotation_p95']:.3f}, "
              f"kappa(V)={dmd_row['eigenvector_condition']:.1f}")
    print("saved", args.out)


if __name__ == "__main__":
    main()

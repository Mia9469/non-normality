#!/usr/bin/env python3
"""Run the Shiu et al. whole-brain LIF model with independent Poisson background.

The published model is silent without input. This script adds many independent
Poisson synaptic background sources to every neuron, using the model's synaptic
variable g and decay time. The background therefore provides a controlled
operating point for a forward-model calibration experiment; it is not claimed
to reproduce the fly's unknown in-vivo spontaneous drive.

Only a random subset is monitored.  Recording every spike from all 127,400
neurons for a 20--40 minute run would require storing hundreds of millions of
events and is unnecessary for the Article-style population analysis.
"""
import argparse
from copy import deepcopy
from pathlib import Path
from textwrap import dedent

import brian2 as b2
import numpy as np
import pandas as pd
from brian2 import (Hz, Network, NeuronGroup, PoissonInput, SpikeMonitor,
                    StateMonitor, Synapses, mV, ms, second)

from model import default_params
from connectome_controls import CONTROL_CHOICES, load_connectome


HERE = Path(__file__).resolve().parent
DEFAULT_COMP = HERE / "2023_03_23_completeness_630_final.csv"
DEFAULT_CON = HERE / "2023_03_23_connectivity_630_final.parquet"


def validate_inputs(comp_path, con_path):
    if not comp_path.exists():
        raise FileNotFoundError(f"missing completeness file: {comp_path}")
    if not con_path.exists():
        raise FileNotFoundError(f"missing connectivity file: {con_path}")
    try:
        pd.read_parquet(con_path, columns=["Presynaptic_Index"]).head(1)
    except Exception as exc:
        raise RuntimeError(
            f"{con_path} is not a readable complete Parquet file. "
            "Re-run download_data.sh before starting the simulation."
        ) from exc


def choose_subset(n, count, rng):
    return np.sort(rng.choice(n, min(count, n), replace=False))


def build_poisson(params, args, rng):
    """Default Shiu model plus independent per-neuron Poisson input into g."""
    df_comp = pd.read_csv(args.comp, index_col=0)
    df_con = load_connectome(args.con, args.connectome_control)
    equations = params["eqs"] + "\nspike_count : integer"
    reset = params["eq_rst"] + "; spike_count += 1"
    neu = NeuronGroup(
        len(df_comp), model=equations, method="linear",
        threshold=params["eq_th"], reset=reset, refractory="rfc",
        namespace=params, name="default_neurons",
    )
    neu.v = params["v_0"]
    neu.g = 0
    neu.rfc = params["t_rfc"]
    neu.spike_count = 0
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                   delay=params["t_dly"], name="default_synapses")
    syn.connect(i=df_con["pre"].values, j=df_con["post"].values)
    syn.w = df_con["signed"].values * args.recurrent_gain * params["w_syn"]

    n = len(neu)
    subset = choose_subset(n, args.rec, rng)
    # Balanced-E/I-like background. Excitatory and
    # inhibitory Poisson into g with +w and -gi*w. The NET mean is set by the
    # imbalance (1 - gi) -> firing rate; the TOTAL variance by (1 + gi^2) -> CV.
    # So gi (toward 1) lowers the rate at fixed fluctuation size, decoupling rate
    # from irregularity. Independent per neuron, so Q stays ~ diagonal.
    r_inh = args.r_inh if args.r_inh is not None else args.r_bg
    background_exc = PoissonInput(
        neu, "g", N=args.bg_sources, rate=(args.r_bg / args.bg_sources) * Hz,
        weight=args.w_bg * params["w_syn"],
    )
    background_inh = PoissonInput(
        neu, "g", N=args.bg_sources, rate=(r_inh / args.bg_sources) * Hz,
        weight=-args.gi * args.w_bg * params["w_syn"],
    )
    # Brian2 2.5 cannot record an arbitrary subset in SpikeMonitor. A cumulative
    # integer sampled by StateMonitor gives exact binned counts without storing
    # every whole-brain spike event. record=False retains only per-neuron totals.
    spike_monitor = SpikeMonitor(neu, record=False)
    state_monitor = StateMonitor(neu, ["v", "spike_count"],
                                 record=subset, dt=args.bin * ms)
    net = Network(neu, syn, background_exc, background_inh,
                  spike_monitor, state_monitor)
    return net, spike_monitor, state_monitor, subset, n, len(df_con)


def build_ou(params, args, rng):
    """Independent OU-current robustness control; Poisson remains the default."""
    df_comp = pd.read_csv(args.comp, index_col=0)
    df_con = load_connectome(args.con, args.connectome_control)
    p = deepcopy(params)
    p["sigma_ou"] = args.sigma_ou * mV
    p["tau_ou"] = args.tau_ou * ms
    equations = dedent("""
        dv/dt = (v_0 - v + g + I_ou) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau                     : volt (unless refractory)
        dI_ou/dt = -I_ou/tau_ou + sigma_ou*sqrt(2/tau_ou)*xi : volt
        spike_count : integer
        rfc : second
    """)
    reset = p["eq_rst"] + "; spike_count += 1"
    neu = NeuronGroup(
        len(df_comp), model=equations, method="euler",
        threshold=p["eq_th"], reset=reset, refractory="rfc",
        namespace=p, name="default_neurons",
    )
    neu.v = p["v_0"]
    neu.g = 0
    neu.rfc = p["t_rfc"]
    neu.spike_count = 0
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                   delay=p["t_dly"], name="default_synapses")
    syn.connect(i=df_con["pre"].values, j=df_con["post"].values)
    syn.w = df_con["signed"].values * args.recurrent_gain * p["w_syn"]

    n = len(neu)
    subset = choose_subset(n, args.rec, rng)
    spike_monitor = SpikeMonitor(neu, record=False)
    state_monitor = StateMonitor(neu, ["v", "spike_count"],
                                 record=subset, dt=args.bin * ms)
    net = Network(neu, syn, spike_monitor, state_monitor)
    return net, spike_monitor, state_monitor, subset, n, len(df_con)


def block_rates(binned_counts, dt_s, blocks=10):
    chunks = np.array_split(np.asarray(binned_counts), blocks, axis=0)
    rates = np.asarray([
        chunk.sum() / (len(chunk) * dt_s * binned_counts.shape[1])
        for chunk in chunks if len(chunk)
    ])
    return rates, float(rates.std() / (rates.mean() + 1e-12))


def poisson_moment_estimate(params, args):
    """Isolated-neuron shot-noise moments (balanced E/I) before recurrent effects.
    Net mean rate (E-I) sets the operating point; total rate (E+I) sets the
    variance. threshold_z is how many SD[v] the mean sits below threshold; aim for
    ~1.5-3 for a fluctuation-driven, irregular regime."""
    weight_mv = args.w_bg * float(params["w_syn"] / mV)
    tau_s = float(params["tau"] / second)
    membrane_s = float(params["t_mbr"] / second)
    threshold_gap_mv = float((params["v_th"] - params["v_0"]) / mV)
    r_inh = args.r_inh if args.r_inh is not None else args.r_bg
    rate_mean = args.r_bg - args.gi * r_inh          # net (E - I) -> operating point
    rate_var = args.r_bg + args.gi**2 * r_inh        # total (E + I) -> fluctuations
    mean_g = rate_mean * weight_mv * tau_s
    sd_g = np.sqrt(rate_var * weight_mv**2 * tau_s / 2)
    sd_v = np.sqrt(
        rate_var * weight_mv**2 * tau_s**2 / (2 * (membrane_s + tau_s))
    )
    threshold_z = (threshold_gap_mv - mean_g) / (sd_v + 1e-30)
    return weight_mv, mean_g, sd_g, sd_v, threshold_z


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=1800.0,
                        help="recording duration after burn-in (s)")
    parser.add_argument("--burn", type=float, default=60.0, help="burn-in duration (s)")
    parser.add_argument("--noise", choices=["poisson", "ou"], default="poisson")
    parser.add_argument("--r-bg", type=float, default=3000.0,
                        help="[Poisson] total independent input rate per neuron (Hz)")
    parser.add_argument("--bg-sources", type=int, default=100,
                        help="[Poisson] independent background sources per neuron")
    parser.add_argument("--w-bg", type=float, default=5.0,
                        help="[Poisson] excitatory event weight in units of w_syn "
                             "(fluctuation size -> CV/Fano)")
    parser.add_argument("--gi", type=float, default=0.9,
                        help="[Poisson] inhibitory/excitatory background weight ratio "
                             "(balanced E/I). gi<1 net depolarizing; RAISE toward 1 to "
                             "lower the mean (rate) at fixed fluctuation size. This is "
                             "the rate knob; --w-bg is the irregularity knob.")
    parser.add_argument("--r-inh", type=float, default=None,
                        help="[Poisson] inhibitory background rate per neuron (Hz); "
                             "default = --r-bg")
    parser.add_argument(
        "--connectome-control", choices=CONTROL_CHOICES, default="original",
        help="original FlyWire graph or an explicitly mathematical symmetric null",
    )
    parser.add_argument(
        "--recurrent-gain", type=float, default=1.0,
        help="global multiplier on recurrent weights; preserves symmetry when "
             "applied to the symmetric null",
    )
    parser.add_argument("--sigma-ou", type=float, default=2.0, help="[OU] noise sd (mV)")
    parser.add_argument("--tau-ou", type=float, default=5.0, help="[OU] correlation time (ms)")
    parser.add_argument("--rec", type=int, default=3000, help="number of neurons monitored")
    parser.add_argument("--bin", type=float, default=1000.0 / 22.0,
                        help="state sampling and later spike-bin width (ms)")
    parser.add_argument("--seed", type=int, default=0)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--cpp", action="store_true", help="use Brian2 cpp_standalone")
    backend.add_argument("--cuda", action="store_true",
                         help="use Brian2CUDA cuda_standalone")
    parser.add_argument("--cpp-dir", "--build-dir", dest="build_dir",
                        default="b2_build")
    parser.add_argument("--calibrate", action="store_true",
                        help="run and print diagnostics without saving")
    parser.add_argument("--accept-rate-min", type=float, default=None,
                        help="fail after simulation if mean rate is below this Hz")
    parser.add_argument("--accept-rate-max", type=float, default=None,
                        help="fail after simulation if mean rate is above this Hz")
    parser.add_argument("--accept-median-rate-min", type=float, default=None,
                        help="fail if median neuron rate is below this Hz")
    parser.add_argument("--accept-median-rate-max", type=float, default=None,
                        help="fail if median neuron rate is above this Hz")
    parser.add_argument("--accept-p95-rate-max", type=float, default=None,
                        help="fail if the 95th-percentile neuron rate exceeds this Hz")
    parser.add_argument("--accept-silent-max", type=float, default=None,
                        help="fail if the monitored silent-neuron fraction exceeds this")
    parser.add_argument("--accept-block-cv-max", type=float, default=None,
                        help="fail if ten-block population-rate CV exceeds this")
    parser.add_argument("--allow-silent", action="store_true",
                        help="allow saving an all-silent run (normally rejected)")
    parser.add_argument("--comp", type=Path, default=DEFAULT_COMP)
    parser.add_argument("--con", type=Path, default=DEFAULT_CON)
    parser.add_argument("--out", type=Path, default=Path("spont_poisson.npz"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bg_sources < 1:
        raise ValueError("--bg-sources must be at least 1")
    if args.rec < 1 or args.bin <= 0 or args.t <= args.bin / 1000 or args.burn < 0:
        raise ValueError("--rec must be positive, --bin positive, --burn non-negative, "
                         "and --t longer than one bin")
    if args.r_bg < 0 or args.w_bg < 0 or args.gi < 0:
        raise ValueError("--r-bg, --w-bg and --gi must be non-negative")
    if not np.isfinite(args.recurrent_gain) or args.recurrent_gain < 0:
        raise ValueError("--recurrent-gain must be finite and non-negative")
    if args.r_inh is not None and args.r_inh < 0:
        raise ValueError("--r-inh must be non-negative")
    if args.accept_silent_max is not None and not 0 <= args.accept_silent_max <= 1:
        raise ValueError("--accept-silent-max must lie in [0, 1]")
    nonnegative_acceptance = {
        "--accept-rate-min": args.accept_rate_min,
        "--accept-rate-max": args.accept_rate_max,
        "--accept-median-rate-min": args.accept_median_rate_min,
        "--accept-median-rate-max": args.accept_median_rate_max,
        "--accept-p95-rate-max": args.accept_p95_rate_max,
        "--accept-block-cv-max": args.accept_block_cv_max,
    }
    for name, value in nonnegative_acceptance.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")
    if (args.accept_rate_min is not None and args.accept_rate_max is not None
            and args.accept_rate_min > args.accept_rate_max):
        raise ValueError("--accept-rate-min cannot exceed --accept-rate-max")
    if (args.accept_median_rate_min is not None
            and args.accept_median_rate_max is not None
            and args.accept_median_rate_min > args.accept_median_rate_max):
        raise ValueError(
            "--accept-median-rate-min cannot exceed --accept-median-rate-max"
        )
    args.comp = args.comp.resolve()
    args.con = args.con.resolve()
    validate_inputs(args.comp, args.con)
    if args.cuda:
        try:
            import brian2cuda  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--cuda requires Brian2CUDA; run setup_cuda_env.sh"
            ) from exc
        b2.set_device("cuda_standalone", build_on_run=True, directory=args.build_dir)
        simulation_backend = "cuda_standalone"
    elif args.cpp:
        b2.set_device("cpp_standalone", build_on_run=True, directory=args.build_dir)
        simulation_backend = "cpp_standalone"
    else:
        simulation_backend = "runtime"
    b2.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    params = deepcopy(default_params)
    builder = build_poisson if args.noise == "poisson" else build_ou
    net, spike_monitor, state_monitor, subset, n, edge_count = builder(
        params, args, rng
    )

    total = args.burn + args.t
    if args.noise == "poisson":
        weight_mv, mean_g, sd_g, sd_v, threshold_z = poisson_moment_estimate(
            params, args
        )
        r_inh = args.r_inh if args.r_inh is not None else args.r_bg
        noise_desc = (f"r_exc={args.r_bg:g}Hz r_inh={r_inh:g}Hz "
                      f"w_bg={weight_mv:g}mV gi={args.gi:g} "
                      f"sources={args.bg_sources:d} "
                      f"net E[g]={mean_g:.3f}mV SD[g]={sd_g:.3f}mV "
                      f"isolated SD[v]={sd_v:.3f}mV threshold_z={threshold_z:.2f}")
    else:
        noise_desc = f"sigma_ou={args.sigma_ou:g}mV tau_ou={args.tau_ou:g}ms"
    print(f">>> backend={simulation_backend} noise={args.noise} "
          f"control={args.connectome_control} recurrent_gain={args.recurrent_gain:g} "
          f"N={n} edges={edge_count} monitored={len(subset)} "
          f"burn={args.burn:g}s record={args.t:g}s bin={args.bin:g}ms")
    print(f">>> {noise_desc}")
    net.run(total * second, report="text")

    state_t_all = np.asarray(state_monitor.t / second)
    keep_state = state_t_all >= args.burn
    state_t = state_t_all[keep_state] - args.burn
    voltage_all = np.asarray(state_monitor.v / mV, dtype=np.float32)[:, keep_state]
    cumulative = np.asarray(state_monitor.spike_count, dtype=np.int64)[:, keep_state]
    # Each column is the exact count accumulated since the previous sampled bin.
    spike_counts = np.diff(cumulative, axis=1).astype(np.int16)
    voltage = voltage_all[:, 1:]
    state_t = state_t[1:]
    dt_s = args.bin / 1000.0
    rate = float(spike_counts.sum() / (spike_counts.size * dt_s))
    rates, block_cv = block_rates(spike_counts.T, dt_s)
    neuron_rates = spike_counts.sum(axis=1) / (spike_counts.shape[1] * dt_s)
    silent_fraction = float(np.mean(neuron_rates == 0))
    rate_percentiles = np.percentile(neuron_rates, [5, 25, 50, 75, 95])
    pooled_fano = float(np.var(spike_counts) / (np.mean(spike_counts) + 1e-12))
    neuron_fano = np.var(spike_counts, axis=1) / (
        np.mean(spike_counts, axis=1) + 1e-12
    )
    active_neuron_fano = neuron_fano[neuron_rates > 0]
    median_active_neuron_fano = float(
        np.median(active_neuron_fano) if len(active_neuron_fano) else np.nan
    )
    target_band_fraction = float(np.mean((neuron_rates >= 1) & (neuron_rates <= 3)))
    high_rate_fraction = float(np.mean(neuron_rates > 20))

    print(f">>> monitored-subset mean rate={rate:.3f}Hz "
          f"silent fraction={silent_fraction:.3f} pooled binned Fano={pooled_fano:.3f}")
    print(f">>> neuron-rate percentiles [5,25,50,75,95]%="
          f"{np.array2string(rate_percentiles, precision=3)} Hz")
    print(f">>> rate fractions: 1-3Hz={target_band_fraction:.3f} "
          f">20Hz={high_rate_fraction:.3f}; "
          f"median active-neuron Fano={median_active_neuron_fano:.3f}")
    print(f">>> ten-block rates={np.array2string(rates, precision=3)}")
    print(f">>> block-rate CV={block_cv:.3f}; inspect for drift/bursts before analysis")
    if args.accept_rate_min is not None and rate < args.accept_rate_min:
        raise RuntimeError(
            f"mean rate {rate:.3f} Hz is below accepted minimum "
            f"{args.accept_rate_min:.3f} Hz"
        )
    if args.accept_rate_max is not None and rate > args.accept_rate_max:
        raise RuntimeError(
            f"mean rate {rate:.3f} Hz is above accepted maximum "
            f"{args.accept_rate_max:.3f} Hz"
        )
    median_rate = float(rate_percentiles[2])
    p95_rate = float(rate_percentiles[4])
    if (args.accept_median_rate_min is not None
            and median_rate < args.accept_median_rate_min):
        raise RuntimeError(
            f"median neuron rate {median_rate:.3f} Hz is below accepted minimum "
            f"{args.accept_median_rate_min:.3f} Hz"
        )
    if (args.accept_median_rate_max is not None
            and median_rate > args.accept_median_rate_max):
        raise RuntimeError(
            f"median neuron rate {median_rate:.3f} Hz exceeds accepted maximum "
            f"{args.accept_median_rate_max:.3f} Hz"
        )
    if args.accept_p95_rate_max is not None and p95_rate > args.accept_p95_rate_max:
        raise RuntimeError(
            f"95th-percentile neuron rate {p95_rate:.3f} Hz exceeds accepted "
            f"maximum {args.accept_p95_rate_max:.3f} Hz"
        )
    if args.accept_silent_max is not None and silent_fraction > args.accept_silent_max:
        raise RuntimeError(
            f"silent fraction {silent_fraction:.3f} exceeds accepted maximum "
            f"{args.accept_silent_max:.3f}"
        )
    if args.accept_block_cv_max is not None and block_cv > args.accept_block_cv_max:
        raise RuntimeError(
            f"block-rate CV {block_cv:.3f} exceeds accepted maximum "
            f"{args.accept_block_cv_max:.3f}"
        )
    if args.calibrate:
        print(">>> calibration run: not saving")
        return
    if rate == 0 and not args.allow_silent:
        raise RuntimeError(
            "all monitored neurons are silent; refusing to save an analysis run. "
            "Run calibrate_poisson.sh or adjust --w-bg/--gi/--r-bg."
        )

    df_comp = pd.read_csv(args.comp, index_col=0)
    flyids = np.asarray(df_comp.index)[subset]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        spike_counts=spike_counts,
        v=voltage,
        v_t=state_t.astype(np.float32),
        sub=subset.astype(np.int32),
        flyids=flyids.astype(np.int64),
        N=n,
        dt_bin=args.bin,
        t_run=args.t,
        burn=args.burn,
        mean_rate=rate,
        neuron_rates=neuron_rates.astype(np.float32),
        rate_percentiles=rate_percentiles.astype(np.float32),
        silent_fraction=silent_fraction,
        binned_fano=pooled_fano,
        pooled_binned_fano=pooled_fano,
        median_active_neuron_fano=median_active_neuron_fano,
        target_rate_band_fraction=target_band_fraction,
        high_rate_fraction=high_rate_fraction,
        block_rates=rates.astype(np.float32),
        block_rate_cv=block_cv,
        noise=args.noise,
        simulation_backend=simulation_backend,
        connectome_control=args.connectome_control,
        recurrent_gain=args.recurrent_gain,
        edge_count=edge_count,
        r_bg=args.r_bg,
        r_inh=(args.r_inh if args.r_inh is not None else args.r_bg),
        gi=args.gi,
        bg_sources=args.bg_sources,
        w_bg=args.w_bg,
        sigma_ou=args.sigma_ou,
        tau_ou=args.tau_ou,
        poisson_weight_mv=(weight_mv if args.noise == "poisson" else np.nan),
        poisson_mean_g_mv=(mean_g if args.noise == "poisson" else np.nan),
        poisson_sd_g_mv=(sd_g if args.noise == "poisson" else np.nan),
        poisson_sd_v_mv=(sd_v if args.noise == "poisson" else np.nan),
        poisson_threshold_z=(threshold_z if args.noise == "poisson" else np.nan),
        seed=args.seed,
    )
    print(">>> saved", args.out)


if __name__ == "__main__":
    main()

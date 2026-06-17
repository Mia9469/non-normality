"""Matched connectome transformations used by simulation and analysis.

Controls
--------
- ``original``      : the real FlyWire graph (eta_signed ~= -0.04, asymmetric).
- ``symmetrized``   : 0.5(W+W^T), the eta=1 null. BUT it also doubles edge count
                      (density/degree change), so any functional difference vs
                      original confounds reciprocity with density.
- ``degree_randomized`` / ``degree_reciprocal`` : degree-preserving nulls that
                      change ONLY the wiring at fixed in/out degree, weight
                      multiset, and per-neuron Dale sign. Maslov-Sneppen
                      target-swaps: pairs (a->b),(c->d) -> (a->d),(c->b), so each
                      edge keeps its presynaptic neuron (hence its sign) and only
                      the postsynaptic target moves. ``degree_randomized`` mixes
                      freely (reciprocity -> chance); ``degree_reciprocal`` biases
                      swaps to RAISE topological reciprocity toward the maximum
                      achievable at the fixed degree sequences. Comparing original
                      vs these isolates the effect of reciprocity from density:
                      if a functional readout (nu) separates original from
                      symmetrized but NOT from a degree-preserving null, the
                      separation was density, not reciprocity.

The degree-preserving nulls are expensive to build, so they are cached to a
parquet next to the source file (keyed by control + seed) and reused. Build/
inspect them standalone with ``python connectome_controls.py --con <file>
--control degree_reciprocal``.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


CONTROL_CHOICES = (
    "original",
    "symmetrized",
    "degree_randomized",
    "degree_reciprocal",
)
_DEGREE_NULLS = ("degree_randomized", "degree_reciprocal")


def _read_edges(path):
    return pd.read_parquet(
        path,
        columns=[
            "Presynaptic_Index",
            "Postsynaptic_Index",
            "Connectivity",
            "Excitatory x Connectivity",
        ],
    ).rename(
        columns={
            "Presynaptic_Index": "pre",
            "Postsynaptic_Index": "post",
            "Connectivity": "unsigned",
            "Excitatory x Connectivity": "signed",
        }
    )


def _symmetrized(edges):
    from scipy import sparse

    n = int(max(edges["pre"].max(), edges["post"].max()) + 1)

    def symmetric_sparse(values):
        matrix = sparse.csr_matrix(
            (values, (edges["pre"].to_numpy(), edges["post"].to_numpy())),
            shape=(n, n),
        )
        matrix = 0.5 * (matrix + matrix.T)
        matrix.eliminate_zeros()
        return matrix

    signed = symmetric_sparse(edges["signed"].to_numpy(np.float64)).tocoo()
    pre = signed.row.copy()
    post = signed.col.copy()
    signed_values = signed.data.copy()
    del signed
    unsigned = symmetric_sparse(edges["unsigned"].to_numpy(np.float64))
    unsigned_values = np.asarray(unsigned[pre, post]).ravel()
    return pd.DataFrame(
        {"pre": pre, "post": post,
         "unsigned": unsigned_values, "signed": signed_values}
    )


def _reciprocity(pre, post, n):
    """Topological reciprocated-edge fraction R/m."""
    keys = set((pre.astype(np.int64) * n + post.astype(np.int64)).tolist())
    rev = (post.astype(np.int64) * n + pre.astype(np.int64))
    return float(np.mean([k in keys for k in rev.tolist()]))


def _degree_preserving_null(edges, mode, n_passes, seed):
    """Maslov-Sneppen directed target-swap preserving in/out degree + weights.

    Drops self-loops first (the model excludes them). Swaps only postsynaptic
    targets, so the presynaptic neuron (and its Dale sign) and the weight stay
    attached to each edge. ``degree_randomized`` accepts any valid swap;
    ``degree_reciprocal`` accepts swaps with non-negative reciprocity change,
    using reciprocity-biased proposals to climb toward maximum reciprocity.
    """
    import random

    edges = edges[edges["pre"] != edges["post"]].reset_index(drop=True)
    n = int(max(edges["pre"].max(), edges["post"].max()) + 1)
    pre = edges["pre"].to_numpy(np.int64)
    post = edges["post"].to_numpy(np.int64).copy()
    m = len(pre)
    rng = random.Random(seed)

    # edge_set: presence of directed key u*n+v; targets[v]: edge indices ->v;
    # in_neigh[a]: source nodes u with edge u->a (for reciprocity bias)
    edge_set = set((pre * n + post).tolist())
    targets, in_neigh = {}, {}
    for idx in range(m):
        targets.setdefault(int(post[idx]), []).append(idx)
        in_neigh.setdefault(int(post[idx]), []).append(int(pre[idx]))

    contains = edge_set.__contains__
    add, discard = edge_set.add, edge_set.discard
    reciprocal = mode == "degree_reciprocal"
    attempts = n_passes * m
    accepted = 0
    report_every = max(1, attempts // 10)

    for step in range(attempts):
        i = rng.randrange(m)
        a, b = int(pre[i]), int(post[i])
        if reciprocal:
            ins = in_neigh.get(a)
            if not ins:
                continue
            d = ins[rng.randrange(len(ins))]      # (d->a) exists => (a->d) reciprocates
            tg = targets.get(d)
            if not tg:
                continue
            j = tg[rng.randrange(len(tg))]
        else:
            j = rng.randrange(m)
            d = int(post[j])
        c = int(pre[j])
        if i == j or len({a, b, c, d}) != 4:
            continue
        k_ad, k_cb = a * n + d, c * n + b
        if contains(k_ad) or contains(k_cb):           # no parallel edges
            continue
        if reciprocal:
            delta = ((contains(d * n + a) + contains(b * n + c))
                     - (contains(b * n + a) + contains(d * n + c)))
            if delta < 0:
                continue
        # apply target swap: (a->b),(c->d) -> (a->d),(c->b)
        discard(a * n + b); discard(c * n + d)
        add(k_ad); add(k_cb)
        post[i] = d; post[j] = b
        targets[b].remove(i); targets.setdefault(d, []).append(i)
        targets[d].remove(j); targets.setdefault(b, []).append(j)
        in_neigh[b].remove(a); in_neigh.setdefault(d, []).append(a)
        in_neigh[d].remove(c); in_neigh.setdefault(b, []).append(c)
        accepted += 1
        if (step + 1) % report_every == 0:
            print(f"  {mode}: {step + 1}/{attempts} attempts, "
                  f"{accepted} accepted, R={_reciprocity(pre, post, n):.3f}")

    return pd.DataFrame(
        {"pre": pre, "post": post,
         "unsigned": edges["unsigned"].to_numpy(np.float64),
         "signed": edges["signed"].to_numpy(np.float64)}
    )


def _cache_path(path, control, seed):
    path = Path(path)
    return path.parent / f"{path.stem}__{control}__seed{seed}.parquet"


def load_connectome(path, control="original"):
    """Load the signed FlyWire graph and optionally form a matched null."""
    path = Path(path)
    if control == "original":
        return _read_edges(path)
    if control == "symmetrized":
        return _symmetrized(_read_edges(path))
    if control in _DEGREE_NULLS:
        seed = int(os.environ.get("FLYWIRE_NULL_SEED", "0"))
        n_passes = int(os.environ.get("FLYWIRE_NULL_PASSES", "5"))
        cache = _cache_path(path, control, seed)
        if cache.exists():
            return pd.read_parquet(cache)
        print(f">>> building {control} null (seed={seed}, passes={n_passes}); "
              f"caching to {cache.name}")
        null = _degree_preserving_null(_read_edges(path), control, n_passes, seed)
        null.to_parquet(cache, index=False)
        return null
    raise ValueError(f"unknown connectome control: {control}")


def _main():
    ap = argparse.ArgumentParser(description="Build/inspect a connectome control")
    ap.add_argument("--con", type=Path, required=True)
    ap.add_argument("--control", choices=CONTROL_CHOICES, default="degree_reciprocal")
    args = ap.parse_args()
    real = _read_edges(args.con)
    real = real[real["pre"] != real["post"]].reset_index(drop=True)
    n = int(max(real["pre"].max(), real["post"].max()) + 1)
    df = load_connectome(args.con, args.control)
    # degree-preservation check (only meaningful for degree nulls)
    out_real = np.bincount(real["pre"].to_numpy(), minlength=n)
    in_real = np.bincount(real["post"].to_numpy(), minlength=n)
    out_new = np.bincount(df["pre"].to_numpy(), minlength=n)
    in_new = np.bincount(df["post"].to_numpy(), minlength=n)
    print(f"control={args.control}  edges: real={len(real)} null={len(df)}")
    print(f"reciprocity R: real={_reciprocity(real['pre'].to_numpy(), real['post'].to_numpy(), n):.3f} "
          f"null={_reciprocity(df['pre'].to_numpy(), df['post'].to_numpy(), n):.3f}")
    if args.control in _DEGREE_NULLS:
        print(f"out-degree preserved: {np.array_equal(out_real, out_new)}; "
              f"in-degree preserved: {np.array_equal(in_real, in_new)}")
        print(f"signed-weight multiset preserved: "
              f"{np.array_equal(np.sort(real['signed'].to_numpy()), np.sort(df['signed'].to_numpy()))}")


if __name__ == "__main__":
    _main()

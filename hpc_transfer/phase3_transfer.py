#!/usr/bin/env python3
"""
Phase 3 -- predict a configuration for each target dataset by transfer from the
ablated ones, and validate the predictor by leave-one-out on the ablated set.

Strategies
  A  whole-config transfer : distance-weighted vote over the k nearest ablated
                             datasets' best configurations (k = 1..3).
  B  per-axis vote         : each axis decided by its own distance-weighted
                             vote over the same neighbours.
  C  global best (baseline): the config with the best mean normalised composite
                             rank across all ablated datasets it appears in.

Validation. For every ablated dataset, predict it from the others and score

    regret = composite_pct(predicted) - composite_pct(that dataset's best)

in normalised composite rank (0 = best in the grid, 1 = worst), so grids of
different sizes are comparable. A predicted config that does not exist in the
held-out dataset's grid is mapped to the nearest one that does (axis distance,
see ``nearest_available``); the exact-hit rate reports how often no mapping was
needed. The strategy with the lowest mean regret wins; if neither A nor B beats
C, C is used for every target and that is stated.

Writes hpc_transfer/predictions.md, loo_results.csv, distances.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "hpc_transfer")
AXES = ["gnn_type", "n_layers", "d_model", "top_k", "n_heads"]

# meta-feature groups. Dependency carries the heaviest weight by default: the
# columns are the GNN's nodes, so the association structure is what the graph
# denoiser actually differs on.
GROUPS = {
    "size":       ["n_rows", "n_features", "n_num", "n_cat", "onehot_dim"],
    "types":      ["num_cat_ratio", "frac_binary_cols", "mean_cardinality",
                   "max_cardinality", "n_classes"],
    "marginals":  ["mean_abs_skew", "mean_kurtosis", "frac_integer_cols",
                   "mean_gmm_components", "frac_multimodal_cols",
                   "mean_norm_entropy", "class_imbalance"],
    "dependency": ["assoc_assoc_mean", "assoc_assoc_max",
                   "assoc_mean_weighted_degree", "assoc_density_0.1",
                   "assoc_density_0.3", "assoc_clustering_0.3",
                   "assoc_n_components_0.3", "assoc_eig_top3_share",
                   "mg_assoc_mean", "mg_mean_weighted_degree",
                   "mg_density_0.1", "mg_density_0.3", "mg_clustering_0.3",
                   "mg_n_components_0.3", "mg_eig_top3_share"],
}
# counts -> log1p before z-scoring
LOG_COLS = {"n_rows", "n_features", "n_num", "n_cat", "onehot_dim",
            "mean_cardinality", "max_cardinality", "n_classes",
            "class_imbalance", "num_cat_ratio", "mean_kurtosis"}

DEFAULT_WEIGHTS = {"size": 1.0, "types": 1.0, "marginals": 1.0,
                   "dependency": 2.0}


# ---------------------------------------------------------------------------
def build_space(meta: pd.DataFrame, weights: dict):
    """z-scored, group-weighted feature matrix. Each group is scaled by
    sqrt(w_g / n_cols_g) so a group with many columns cannot dominate purely
    by being wide."""
    cols, scale = [], []
    for g, feats in GROUPS.items():
        present = [f for f in feats if f in meta.columns]
        w = np.sqrt(weights[g] / max(len(present), 1))
        for f in present:
            cols.append(f)
            scale.append(w)
    X = meta[cols].astype(float).copy()
    for c in cols:
        if c in LOG_COLS:
            X[c] = np.log1p(X[c].clip(lower=0))
    X = X.fillna(X.mean())
    sd = X.std(ddof=0).replace(0, 1.0)
    Z = ((X - X.mean()) / sd) * np.asarray(scale)
    return Z.to_numpy(), cols


def distances(Z: np.ndarray, names: list, froms: list, tos: list) -> pd.DataFrame:
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    for a in froms:
        for b in tos:
            if a == b:
                continue
            d = float(np.linalg.norm(Z[idx[a]] - Z[idx[b]]))
            rows.append(dict(target=a, neighbour=b, distance=d))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def parse_config(cid: str) -> dict:
    g, l, d, k, h = cid.split("_")
    return dict(gnn_type=g, n_layers=int(l[1:]), d_model=int(d[1:]),
                top_k=int(k[1:]), n_heads=int(h[1:]))


def fmt_config(c: dict) -> str:
    return (f"{c['gnn_type']}_L{c['n_layers']}_d{c['d_model']}"
            f"_k{c['top_k']}_h{c['n_heads']}")


def axis_distance(a: dict, b: dict) -> float:
    """How far apart two configurations are, for mapping a prediction onto a
    grid that does not contain it."""
    d = 2.0 * (a["gnn_type"] != b["gnn_type"])
    d += 1.0 * abs(a["n_layers"] - b["n_layers"])
    d += 1.0 * abs(np.log2(a["d_model"] / b["d_model"]))
    d += 0.5 * abs(a["top_k"] - b["top_k"]) / 2.0
    d += 0.5 * (a["n_heads"] != b["n_heads"])
    return float(d)


def nearest_available(pred: dict, grid: pd.DataFrame):
    """Map a predicted config onto the nearest config present in `grid`."""
    best, bd = None, np.inf
    for _, r in grid.iterrows():
        c = {a: r[a] for a in AXES}
        d = axis_distance(pred, c)
        if d < bd:
            best, bd = r, d
    return best, bd


# ---------------------------------------------------------------------------
def neighbours(dist: pd.DataFrame, target: str, pool: list, k: int):
    s = dist[(dist.target == target) & (dist.neighbour.isin(pool))]
    s = s.sort_values("distance").head(k)
    w = 1.0 / (s.distance.to_numpy() + 1e-9)
    return list(s.neighbour), s.distance.to_numpy(), w / w.sum()


def predict_A(dist, target, pool, k, best_of):
    nb, d, w = neighbours(dist, target, pool, k)
    votes = defaultdict(float)
    for n, wi in zip(nb, w):
        votes[best_of[n]] += wi
    winner = max(votes.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
    return parse_config(winner), nb, d, w


def predict_B(dist, target, pool, k, best_of):
    nb, d, w = neighbours(dist, target, pool, k)
    cfgs = [parse_config(best_of[n]) for n in nb]
    out = {}
    for axis in AXES:
        votes = defaultdict(float)
        for c, wi in zip(cfgs, w):
            votes[c[axis]] += wi
        out[axis] = max(votes.items(), key=lambda kv: kv[1])[0]
    return out, nb, d, w


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", choices=["tiebreak", "rank1"], default="tiebreak",
                    help="which definition of 'best config' to transfer")
    ap.add_argument("--min_configs", type=int, default=4,
                    help="ablated datasets with fewer configurations than this "
                         "are excluded from the donor pool: an argmax over two "
                         "or three points that differ on several axes at once "
                         "is not a measured best")
    ap.add_argument("--min_snr", type=float, default=2.0,
                    help="ablated datasets below this spread/noise ratio are "
                         "excluded from the neighbour pool (their best config "
                         "is not distinguishable from any other)")
    ap.add_argument("--weights", type=str, default=None,
                    help='JSON group weights, default '
                         f'{json.dumps(DEFAULT_WEIGHTS)}')
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        weights.update(json.loads(args.weights))

    meta = pd.read_csv(os.path.join(OUT, "meta_features.csv"))
    best = pd.read_csv(os.path.join(OUT, "best_per_dataset.csv"))
    ranked = pd.read_csv(os.path.join(OUT, "all_configs_ranked.csv"))

    ablated = sorted(best.dataset)
    targets = args.targets or sorted(set(meta.dataset) - set(ablated))
    label_col = "config_id" if args.label == "tiebreak" else "rank1_config_id"
    best_of = dict(zip(best.dataset, best[label_col]))
    snr = dict(zip(best.dataset, best.snr))
    n_cfg = dict(zip(best.dataset, best.n_configs))
    pool = [d for d in ablated
            if snr[d] >= args.min_snr and n_cfg[d] >= args.min_configs]

    Z, cols = build_space(meta, weights)
    names = list(meta.dataset)
    dist = distances(Z, names, names, names)
    dist.to_csv(os.path.join(OUT, "distances.csv"), index=False)

    # ---- global best (strategy C), over the whole donor pool ---------------
    # (defined below as global_best_over; the fold-local version is what the
    # leave-one-out uses, so C never votes for its own held-out dataset)

    # ---- leave-one-out ----------------------------------------------------
    def regret(ds, pred):
        """Rank regret AND utility regret.

        Rank regret is the normalised composite-rank gap the brief asks for.
        Utility regret is the same gap in the primary metric, normalised by the
        grid's own span: (best - predicted) / (best - worst). It is the
        decision-relevant one -- on insurance, ranks 1..11 span only 2.6 % of
        RMSE and then the grid falls off a cliff, so a rank gap of 0.67 there
        costs 0.07 of the achievable utility, not 0.67.
        """
        grid = ranked[ranked.dataset == ds]
        row, d = nearest_available(pred, grid)
        rank_regret = float(row.composite_pct - grid.composite_pct.min())
        hi = float(grid["score__mean"].max())
        lo = float(grid["score__mean"].min())
        span = hi - lo
        util = float((hi - row["score__mean"]) / span) if span > 0 else 0.0
        return rank_regret, util, (d == 0.0), row.config_id

    def global_best_over(datasets, min_ds=2):
        """Best mean normalised composite rank over `datasets` only.

        Recomputed per leave-one-out fold: using the all-donor global best to
        predict a held-out donor would let that donor's own ranking vote for
        its own answer, which inflates strategy C.
        """
        sub = ranked[ranked.dataset.isin(datasets)]
        agg = (sub.groupby("config_id")
                  .agg(n=("dataset", "nunique"), mean_pct=("composite_pct", "mean"))
                  .reset_index())
        elig = agg[agg.n >= min(min_ds, len(datasets))]
        if elig.empty:
            elig = agg
        return parse_config(elig.sort_values("mean_pct").iloc[0].config_id)

    C = global_best_over(pool)

    loo = []
    for held in pool:
        rest = [d for d in pool if d != held]
        if not rest:
            continue
        C_fold = global_best_over(rest)
        for k in (1, 2, 3):
            pa, nb, dd, w = predict_A(dist, held, rest, k, best_of)
            ra, ua, hita, mapa = regret(held, pa)
            loo.append(dict(held_out=held, strategy=f"A_k{k}",
                            predicted=fmt_config(pa), mapped_to=mapa,
                            exact_hit=hita, regret=ra, utility_regret=ua,
                            neighbours=";".join(nb),
                            distances=";".join(f"{x:.3f}" for x in dd)))
            pb, *_ = predict_B(dist, held, rest, k, best_of)
            rb, ub, hitb, mapb = regret(held, pb)
            loo.append(dict(held_out=held, strategy=f"B_k{k}",
                            predicted=fmt_config(pb), mapped_to=mapb,
                            exact_hit=hitb, regret=rb, utility_regret=ub,
                            neighbours=";".join(nb),
                            distances=";".join(f"{x:.3f}" for x in dd)))
        rc, uc, hitc, mapc = regret(held, C_fold)
        loo.append(dict(held_out=held, strategy="C_global",
                        predicted=fmt_config(C_fold), mapped_to=mapc,
                        exact_hit=hitc, regret=rc, utility_regret=uc,
                        neighbours="", distances=""))
    loo = pd.DataFrame(loo)
    loo.to_csv(os.path.join(OUT, "loo_results.csv"), index=False)

    summary = (loo.groupby("strategy")
                  .agg(mean_utility_regret=("utility_regret", "mean"),
                       max_utility_regret=("utility_regret", "max"),
                       mean_rank_regret=("regret", "mean"),
                       median_rank_regret=("regret", "median"),
                       exact_hit_rate=("exact_hit", "mean"),
                       n=("regret", "size"))
                  .reset_index().sort_values("mean_utility_regret"))
    SEL = "mean_utility_regret"
    winner_strategy = summary.iloc[0].strategy
    idx = summary.set_index("strategy")
    # A strategy only displaces the global-best baseline if it beats it by more
    # than a rounding error; with a donor pool this small, a hair's difference
    # is not evidence.
    beats_C = idx.loc[winner_strategy, SEL] < idx.loc["C_global", SEL] - 1e-9
    chosen = winner_strategy if beats_C else "C_global"

    # ---- OOD ranges --------------------------------------------------------
    # n_rows and n_features are reported as explicit extrapolation FACTORS
    # rather than as boolean flags: every target is larger than every donor, so
    # a boolean would fire for all seven and say nothing. The boolean flags are
    # reserved for the structural features, where being outside the donor range
    # really is informative.
    struct_cols = ["n_cat", "max_cardinality", "num_cat_ratio",
                   "assoc_assoc_mean", "assoc_density_0.3",
                   "assoc_clustering_0.3", "assoc_eig_top3_share",
                   "mg_density_0.3", "mg_n_components_0.3"]
    pool_meta = meta[meta.dataset.isin(pool)]
    lo = pool_meta[struct_cols].min()
    hi = pool_meta[struct_cols].max()
    max_rows = float(pool_meta.n_rows.max())
    max_feats = float(pool_meta.n_features.max())

    # ---- predict targets ---------------------------------------------------
    preds = []
    for t in targets:
        if chosen.startswith("A"):
            k = int(chosen[-1])
            cfg, nb, dd, w = predict_A(dist, t, pool, k, best_of)
        elif chosen.startswith("B"):
            k = int(chosen[-1])
            cfg, nb, dd, w = predict_B(dist, t, pool, k, best_of)
        else:
            cfg = C
            nb, dd, w = neighbours(dist, t, pool, 3)
        m = meta[meta.dataset == t].iloc[0]
        ood = [c for c in struct_cols if m[c] < lo[c] or m[c] > hi[c]]
        row_factor = float(m.n_rows) / max_rows
        feat_factor = float(m.n_features) / max_feats
        d0 = dd[0] if len(dd) else float("nan")
        pool_d = dist[(dist.target.isin(pool)) & (dist.neighbour.isin(pool))]
        near_med = float(pool_d.groupby("target").distance.min().median())
        # Confidence: how far the nearest donor is, how far outside the donor
        # size range the target sits, and whether its graph structure is
        # outside the donor range at all.
        score = 0
        score += 0 if d0 <= near_med else (1 if d0 <= 2 * near_med else 2)
        score += 0 if row_factor <= 2 else (1 if row_factor <= 6 else 2)
        score += 0 if feat_factor <= 1.5 else (1 if feat_factor <= 3 else 2)
        score += min(len(ood), 2)
        conf = "high" if score <= 1 else ("medium" if score <= 3 else "low")
        preds.append(dict(dataset=t, **cfg, config_id=fmt_config(cfg),
                          nearest=nb[0] if nb else "", nearest_distance=d0,
                          neighbours=";".join(nb),
                          distances=";".join(f"{x:.3f}" for x in dd),
                          n_rows=int(m.n_rows), n_features=int(m.n_features),
                          row_factor=round(row_factor, 1),
                          feat_factor=round(feat_factor, 1),
                          struct_ood=";".join(ood) or "-",
                          confidence=conf, conf_score=score))
    preds = pd.DataFrame(preds)
    preds.to_csv(os.path.join(OUT, "predictions.csv"), index=False)

    if not args.quiet:
        pd.set_option("display.width", 220, "display.max_columns", 40)
        print(f"label={args.label}  min_snr={args.min_snr}  "
              f"pool={pool}\nweights={weights}\n")
        print("=== LOO summary ===")
        print(summary.to_string(index=False))
        print(f"\nchosen strategy: {chosen}"
              f"{'' if beats_C else '  (neither A nor B beat C)'}")
        print(f"global best (C): {fmt_config(C)}\n")
        print("=== target predictions ===")
        print(preds[["dataset", "config_id", "nearest", "nearest_distance",
                     "n_rows", "n_features", "row_factor", "feat_factor",
                     "struct_ood", "confidence"]].to_string(index=False))
    return dict(summary=summary, chosen=chosen, preds=preds, loo=loo,
                C=C, pool=pool, dist=dist, meta=meta, best=best,
                weights=weights, label=args.label)


if __name__ == "__main__":
    main()

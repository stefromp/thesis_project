#!/usr/bin/env python3
"""
Phase 1 -- best ablation configuration per ablated dataset.

Reads the finished ablation summaries in ``tabddpm_ablation/*.json`` (the
single source of truth; the same files ``ablation/existing_results.py`` and
``hpc/adaptive_manifest.py`` read) and produces, per dataset:

  * a direction-aware rank of every configuration on the UTILITY metrics only,
  * a composite = weighted mean rank,
  * a winner, or a tie set broken in favour of the cheaper configuration,
  * a global best over all ablated datasets,
  * per-axis marginal effects, computed on MATCHED PAIRS so the effect of one
    axis is not confounded with the others.

Utility metrics, by task type (nothing else is scored -- no fidelity, no
privacy; those are reported alongside for information only):

    regression   r2                     (up)
    binclass     accuracy               (up)
                 macro avg/f1-score     (up)
                 roc_auc                (up)
    multiclass   accuracy               (up)
                 macro avg/f1-score     (up)

``rmse`` is deliberately NOT scored next to ``r2``: within one dataset
r2 = 1 - mse/var(y) is a strictly decreasing function of rmse, so ranking on
both would double-weight one signal. ``weighted avg/f1-score`` is dropped for
the same reason -- on these datasets it tracks accuracy almost exactly.

Writes:
    hpc_transfer/best_per_dataset.csv
    hpc_transfer/all_configs_ranked.csv
    hpc_transfer/axis_marginals.csv

Nothing outside hpc_transfer/ is written or modified.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "hpc_transfer")

AXES = ["gnn_type", "n_layers", "d_model", "top_k", "n_heads"]

# metric -> +1 if higher is better, -1 if lower is better
UTILITY = {
    "regression": {"r2": +1},
    "binclass":   {"accuracy": +1, "macro avg/f1-score": +1, "roc_auc": +1},
    "multiclass": {"accuracy": +1, "macro avg/f1-score": +1},
}

# Reported but never scored (kept so the table stays auditable).
CONTEXT = ["score", "rmse", "wasserstein_mean", "mia_auc",
           "dcr_tabddpm_mean", "dcr_baseline"]

# The single metric the noise test is applied to: accuracy for classification,
# -rmse for regression (verified against the files, not assumed).
PRIMARY = "score"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def task_type(ds: str) -> str:
    with open(os.path.join(REPO, "data", ds, "info.json")) as fh:
        return json.load(fh)["task_type"]


def load_records() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, "tabddpm_ablation", "*.json"))):
        for rec in json.load(open(path)):
            if rec.get("status") != "done":
                continue
            row = {a: rec[a] for a in AXES}
            row.update(dataset=rec["dataset"], exp_name=rec["exp_name"],
                       use_attention=rec["use_attention"], steps=rec["steps"],
                       n_gen_seeds=rec["n_gen_seeds"], n_clf_seeds=rec["n_clf_seeds"],
                       source=os.path.basename(path))
            for key, val in rec["metrics"].items():
                row[f"{key}__mean"] = val["mean"]
                row[f"{key}__std"] = val["std"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df["task_type"] = df["dataset"].map(task_type)
    df["config_id"] = df.apply(
        lambda r: f"{r.gnn_type}_L{r.n_layers}_d{r.d_model}_k{r.top_k}_h{r.n_heads}",
        axis=1)
    return df


def dataset_cfg(ds: str) -> dict:
    """The dataset's DATASET_CFG, loaded the way hpc_wide/wide_manifest.py does
    it: import ablation/<ds with '-'->'_'>_ablation.py and read the dict."""
    import importlib
    ablation_dir = os.path.join(REPO, "ablation")
    if ablation_dir not in sys.path:
        sys.path.insert(0, ablation_dir)
    return importlib.import_module(f"{ds.replace('-', '_')}_ablation").DATASET_CFG


# ---------------------------------------------------------------------------
# Parameter count -- the cost used to break a tie. Real counts from the repo's
# own denoiser, not a proxy.
# ---------------------------------------------------------------------------
def param_counts(df: pd.DataFrame) -> dict:
    sys.path.insert(0, os.path.join(REPO, "ablation"))
    sys.path.insert(0, REPO)
    from ablation_runner import _setup_zero_rtdl_stubs  # noqa
    _setup_zero_rtdl_stubs()
    import torch  # noqa: F401
    from tab_ddpm.graph_denoiser import GraphAwareDenoiser

    out = {}
    for ds, sub in df.groupby("dataset"):
        cfg = dataset_cfg(ds)
        tt = task_type(ds)
        d_num = cfg["num_numerical_features"]
        if tt == "regression" and not cfg["is_y_cond"]:
            d_num += 1
        cat_path = os.path.join(REPO, "data", ds, "X_cat_train.npy")
        if os.path.exists(cat_path):
            X = np.load(cat_path, allow_pickle=True)
            cat_sizes = [len(set(X[:, i].tolist())) for i in range(X.shape[1])]
        else:
            cat_sizes = []
        cat_sizes = [k for k in cat_sizes if k > 0]
        d_in = int(sum(cat_sizes)) + d_num
        for _, r in sub.iterrows():
            key = (ds, r.config_id)
            if key in out:
                continue
            den = GraphAwareDenoiser(
                d_in=d_in, num_classes=cfg["num_classes"],
                is_y_cond=cfg["is_y_cond"], d_num=d_num, cat_sizes=cat_sizes,
                d_model=int(r.d_model), n_layers=int(r.n_layers),
                n_heads=int(r.n_heads), graph_mode="dynamic",
                top_k=int(r.top_k), gnn_type=r.gnn_type, dropout=0.0,
                use_attention=bool(r.use_attention),
            )
            out[key] = sum(p.numel() for p in den.parameters() if p.requires_grad)
    return out


# ---------------------------------------------------------------------------
# Rank + composite
# ---------------------------------------------------------------------------
def rank_dataset(sub: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Direction-aware ranks (1 = best) and the weighted-mean-rank composite."""
    tt = sub["task_type"].iloc[0]
    metrics = {m: d for m, d in UTILITY[tt].items() if f"{m}__mean" in sub.columns
               and sub[f"{m}__mean"].notna().all()}
    sub = sub.copy()
    used, wts = [], []
    for m, direction in metrics.items():
        col = sub[f"{m}__mean"] * direction          # higher is better now
        sub[f"rank__{m}"] = col.rank(ascending=False, method="average")
        used.append(m)
        wts.append(weights.get(m, 1.0))
    wts = np.asarray(wts, dtype=float)
    wts = wts / wts.sum()
    sub["composite_rank"] = sum(
        w * sub[f"rank__{m}"] for w, m in zip(wts, used))
    n = len(sub)
    # normalised: 0 = best, 1 = worst; comparable across differently sized grids
    sub["composite_pct"] = ((sub["composite_rank"] - 1) / (n - 1)) if n > 1 else 0.0
    sub.attrs["metrics_used"] = used
    return sub


def decide_winner(sub: pd.DataFrame, params: dict, sigma: float):
    """Winner if its PRIMARY-metric lead beats combined noise, else a tie set.

    The noise here is the std stored in the ablation files, which is spread
    over 5 generation x 10 classifier seeds of a SINGLE trained model
    (ablation_runner trains once per cell, then re-samples). It therefore does
    not contain training-seed variance and is an optimistic noise floor -- the
    tie sets below are, if anything, too small.
    """
    s = sub.sort_values(["composite_rank", f"{PRIMARY}__mean"],
                        ascending=[True, False]).reset_index(drop=True)
    best = s.iloc[0]
    if len(s) == 1:
        return best, s.iloc[[0]], "single-config grid"

    def combined(a, b):
        return sigma * float(np.hypot(a[f"{PRIMARY}__std"], b[f"{PRIMARY}__std"]))

    tie_mask = [
        (best[f"{PRIMARY}__mean"] - row[f"{PRIMARY}__mean"]) <= combined(best, row)
        for _, row in s.iterrows()
    ]
    ties = s[np.array(tie_mask)]
    if len(ties) == 1:
        return best, ties, "clear winner (lead > combined seed noise)"
    # cheapest member of the tie set wins
    ties = ties.copy()
    ties["n_params"] = [params[(r.dataset, r.config_id)] for _, r in ties.iterrows()]
    cheapest = ties.sort_values(
        ["n_params", "composite_rank"]).iloc[0]
    return cheapest, ties, f"tie set of {len(ties)} broken on parameter count"


# ---------------------------------------------------------------------------
# Matched-pair axis marginals
# ---------------------------------------------------------------------------
def axis_marginals(ranked: pd.DataFrame) -> pd.DataFrame:
    """Mean change in normalised composite rank when one axis value is swapped,
    holding the other four axes fixed. Negative = the 'to' value is better."""
    out = []
    for ds, sub in ranked.groupby("dataset"):
        for axis in AXES:
            others = [a for a in AXES if a != axis]
            for _, grp in sub.groupby(others):
                if len(grp) < 2:
                    continue
                for a, b in combinations(grp.itertuples(), 2):
                    va, vb = getattr(a, axis), getattr(b, axis)
                    if va == vb:
                        continue
                    lo, hi = sorted([(va, a), (vb, b)], key=lambda t: str(t[0]))
                    out.append(dict(
                        dataset=ds, axis=axis,
                        from_value=lo[0], to_value=hi[0],
                        delta_pct=hi[1].composite_pct - lo[1].composite_pct,
                        delta_primary=getattr(hi[1], f"{PRIMARY}__mean")
                        - getattr(lo[1], f"{PRIMARY}__mean"),
                    ))
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    agg = (df.groupby(["dataset", "axis", "from_value", "to_value"])
             .agg(n_pairs=("delta_pct", "size"),
                  mean_delta_pct=("delta_pct", "mean"),
                  mean_delta_primary=("delta_primary", "mean"))
             .reset_index())
    return agg.sort_values(["axis", "dataset", "from_value", "to_value"])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="tie band in combined standard deviations (default 1.0)")
    ap.add_argument("--min_datasets_global", type=int, default=3,
                    help="a config must appear in this many datasets to be "
                         "eligible for the global best (default 3)")
    ap.add_argument("--weights", type=str, default=None,
                    help='JSON dict of per-metric weights, e.g. '
                         '\'{"accuracy":2,"macro avg/f1-score":1}\'')
    args = ap.parse_args()
    weights = json.loads(args.weights) if args.weights else {}

    df = load_records()
    print(f"loaded {len(df)} finished runs over {df.dataset.nunique()} datasets")

    print("counting parameters (real counts from GraphAwareDenoiser) ...")
    params = param_counts(df)

    ranked, best_rows = [], []
    for ds, sub in df.groupby("dataset"):
        r = rank_dataset(sub, weights)
        used = r.attrs["metrics_used"]
        r["metrics_used"] = ";".join(used)
        r["n_configs"] = len(r)
        r["n_params"] = [params[(ds, c)] for c in r.config_id]
        ranked.append(r)
        winner, ties, reason = decide_winner(r, params, args.sigma)
        rank1 = r.sort_values(["composite_rank", f"{PRIMARY}__mean"],
                              ascending=[True, False]).iloc[0]
        # Signal quality: how far the grid spreads relative to the per-cell
        # noise. Below ~2 the dataset cannot discriminate between configs at
        # all, and its "best config" carries no transferable information.
        spread = float(r[f"{PRIMARY}__mean"].max() - r[f"{PRIMARY}__mean"].min())
        med_std = float(r[f"{PRIMARY}__std"].median())
        snr = spread / med_std if med_std > 0 else float("nan")
        best_rows.append(dict(
            dataset=ds, task_type=winner.task_type, n_configs=len(r),
            metrics_used=";".join(used),
            **{a: winner[a] for a in AXES},
            config_id=winner.config_id, exp_name=winner.exp_name,
            n_params=params[(ds, winner.config_id)],
            rank1_config_id=rank1.config_id,
            rank1_n_params=params[(ds, rank1.config_id)],
            composite_rank=round(float(winner.composite_rank), 3),
            primary_mean=float(winner[f"{PRIMARY}__mean"]),
            primary_std=float(winner[f"{PRIMARY}__std"]),
            grid_spread=spread, median_cell_std=med_std,
            snr=round(snr, 2),
            discriminative=("yes" if snr > 4 else
                            "weak" if snr > 2 else "no"),
            tie_set_size=len(ties),
            tie_set=";".join(sorted(ties.config_id)),
            decision=reason,
        ))

    ranked = pd.concat(ranked, ignore_index=True)
    best = pd.DataFrame(best_rows).sort_values("dataset")

    # ---- global best -------------------------------------------------------
    cov = ranked.groupby("config_id").agg(
        n_datasets=("dataset", "nunique"),
        mean_pct=("composite_pct", "mean"),
        mean_rank=("composite_rank", "mean"),
        datasets=("dataset", lambda s: ";".join(sorted(s)))).reset_index()
    eligible = cov[cov.n_datasets >= args.min_datasets_global].sort_values("mean_pct")

    os.makedirs(OUT, exist_ok=True)
    keep = (["dataset", "task_type", "config_id", "exp_name", *AXES,
             "n_configs", "n_params", "composite_rank", "composite_pct",
             "metrics_used"]
            + [c for c in ranked.columns if c.startswith("rank__")]
            + [f"{m}__mean" for m in [PRIMARY] + CONTEXT if f"{m}__mean" in ranked.columns]
            + [f"{m}__std" for m in [PRIMARY] if f"{m}__std" in ranked.columns])
    keep = list(dict.fromkeys(keep))
    ranked[keep].sort_values(["dataset", "composite_rank"]).to_csv(
        os.path.join(OUT, "all_configs_ranked.csv"), index=False)
    best.to_csv(os.path.join(OUT, "best_per_dataset.csv"), index=False)
    marg = axis_marginals(ranked)
    marg.to_csv(os.path.join(OUT, "axis_marginals.csv"), index=False)
    cov.sort_values("mean_pct").to_csv(
        os.path.join(OUT, "global_config_coverage.csv"), index=False)

    # ---- report ------------------------------------------------------------
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n=== best per dataset ===")
    print(best[["dataset", "task_type", "n_configs", "config_id",
                "rank1_config_id", "n_params", "snr", "discriminative",
                "tie_set_size", "decision"]]
          .to_string(index=False))
    print("\n=== global best (configs present in >= "
          f"{args.min_datasets_global} datasets, 0 = best) ===")
    print(eligible.head(10).to_string(index=False))
    print("\nwrote:", ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()

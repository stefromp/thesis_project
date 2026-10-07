#!/usr/bin/env python3
"""
Phase 2 -- dataset meta-features for every TabDDPM dataset (ablated + target).

Everything here is computed on the TRAIN SPLIT ONLY (``X_num_train.npy``,
``X_cat_train.npy``, ``y_train.npy``). The validation and test splits are never
opened, so nothing leaks into the transfer decision.

The node set matches what the denoiser actually sees, built the way
``scripts/train.py`` / ``lib.make_dataset`` build it:

    one node per numerical column
  + one node for y when the task is regression and is_y_cond is False
    (the label is concatenated into X_num in that case)
  + one node per categorical column, regardless of its cardinality

Two association matrices are produced over that node set.

1. ``assoc_*``  -- the specification matrix, mixed-type:
       num-num  |Spearman rho|
       cat-cat  Cramer's V
       num-cat  correlation ratio eta
   Rank-based and non-parametric, so it is not distorted by the heavy skew
   several of these datasets have.

2. ``mg_*``     -- the graph the repository itself builds, by calling
   ``tab_ddpm.graph_builder.build_static_adjacency`` (|Pearson| num-num,
   Cramer's V cat-cat, normalised MI mixed). Note the ablation runs all use
   graph_mode="dynamic", where the adjacency is learned from node embeddings
   and re-formed every block -- so no fixed matrix is "the" graph the GNN sees.
   This is the repo's own data-derived graph and the closest auditable
   analogue; it is reported so the meta-features can be compared against the
   structure the static mode would impose.

Writes hpc_transfer/meta_features.csv.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "hpc_transfer")

GMM_MAX_COMPONENTS = 5
GMM_SUBSAMPLE = 10_000      # rows used for the BIC search; fixed seed
RNG_SEED = 0
THRESHOLDS = (0.1, 0.3)
EIG_TOPK = 3


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------
def dataset_cfg(ds: str) -> dict:
    """DATASET_CFG from ablation/<ds>_ablation.py (the repo's own convention)."""
    d = os.path.join(REPO, "ablation")
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module(f"{ds.replace('-', '_')}_ablation").DATASET_CFG


def load_train(ds: str):
    """X_num, X_cat, y and info for the TRAIN split only."""
    root = os.path.join(REPO, "data", ds)
    info = json.load(open(os.path.join(root, "info.json")))

    def maybe(name):
        p = os.path.join(root, name)
        return np.load(p, allow_pickle=True) if os.path.exists(p) else None

    X_num = maybe("X_num_train.npy")
    X_cat = maybe("X_cat_train.npy")
    y = maybe("y_train.npy")
    X_num = None if X_num is None else X_num.astype(np.float64)
    return X_num, X_cat, y, info


# ---------------------------------------------------------------------------
# Pairwise association measures
# ---------------------------------------------------------------------------
def cramers_v(a, b) -> float:
    """Bias-corrected Cramer's V between two categorical vectors."""
    from scipy.stats import chi2_contingency
    a = np.asarray(a).astype(str)
    b = np.asarray(b).astype(str)
    ca, cb = pd.factorize(a)[0], pd.factorize(b)[0]
    table = np.zeros((ca.max() + 1, cb.max() + 1), dtype=np.int64)
    np.add.at(table, (ca, cb), 1)
    table = table[table.sum(1) > 0][:, table.sum(0) > 0]
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    phi2 = chi2 / n
    r, k = table.shape
    # Bergsma-Wicher bias correction
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def correlation_ratio(cat, num) -> float:
    """eta: fraction of the numerical variance explained by the category."""
    cat = pd.factorize(np.asarray(cat).astype(str))[0]
    num = np.asarray(num, dtype=np.float64)
    ok = np.isfinite(num)
    cat, num = cat[ok], num[ok]
    if num.size == 0 or np.allclose(num, num[0]):
        return 0.0
    grand = num.mean()
    ss_between = 0.0
    for g in np.unique(cat):
        v = num[cat == g]
        ss_between += v.size * (v.mean() - grand) ** 2
    ss_total = ((num - grand) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def spearman_abs(a, b) -> float:
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return 0.0
    rho = spearmanr(a[ok], b[ok]).statistic
    return 0.0 if not np.isfinite(rho) else float(abs(rho))


def build_assoc(num_cols, cat_cols) -> np.ndarray:
    """Mixed association matrix over [numerical nodes..., categorical nodes...].

    num-num |Spearman|, cat-cat Cramer's V, num-cat correlation ratio eta.
    Diagonal is 1. Symmetric, entries in [0, 1].
    """
    cols = list(num_cols) + list(cat_cols)
    n_num = len(num_cols)
    N = len(cols)
    A = np.eye(N, dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            if i < n_num and j < n_num:
                v = spearman_abs(cols[i], cols[j])
            elif i >= n_num and j >= n_num:
                v = cramers_v(cols[i], cols[j])
            else:
                num, cat = (cols[i], cols[j]) if i < n_num else (cols[j], cols[i])
                v = correlation_ratio(cat, num)
            A[i, j] = A[j, i] = float(np.clip(v, 0.0, 1.0))
    return A


# ---------------------------------------------------------------------------
# Graph statistics on an association matrix
# ---------------------------------------------------------------------------
def graph_stats(A: np.ndarray, prefix: str) -> dict:
    import networkx as nx
    N = A.shape[0]
    off = ~np.eye(N, dtype=bool)
    vals = A[off]
    out = {
        f"{prefix}_n_nodes": N,
        f"{prefix}_assoc_mean": float(vals.mean()) if vals.size else 0.0,
        f"{prefix}_assoc_max": float(vals.max()) if vals.size else 0.0,
        f"{prefix}_mean_weighted_degree": float((A * off).sum(1).mean()),
    }
    for t in THRESHOLDS:
        adj = (A >= t) & off
        n_edges = adj.sum() / 2
        possible = N * (N - 1) / 2
        out[f"{prefix}_density_{t}"] = float(n_edges / possible) if possible else 0.0
    # clustering + components on the 0.3-thresholded unweighted graph
    adj3 = (A >= 0.3) & off
    G = nx.from_numpy_array(adj3.astype(int))
    out[f"{prefix}_clustering_0.3"] = float(nx.average_clustering(G))
    out[f"{prefix}_n_components_0.3"] = int(nx.number_connected_components(G))
    # spectral concentration of the association matrix
    ev = np.abs(np.linalg.eigvalsh((A + A.T) / 2))
    ev = np.sort(ev)[::-1]
    k = min(EIG_TOPK, N)
    out[f"{prefix}_eig_top{EIG_TOPK}_share"] = (
        float(ev[:k].sum() / ev.sum()) if ev.sum() > 0 else 0.0)
    return out


# ---------------------------------------------------------------------------
# Column-level statistics
# ---------------------------------------------------------------------------
def numerical_stats(X: np.ndarray) -> dict:
    from scipy.stats import skew, kurtosis
    from sklearn.mixture import GaussianMixture
    if X is None or X.shape[1] == 0:
        return {"mean_abs_skew": 0.0, "mean_kurtosis": 0.0,
                "frac_integer_cols": 0.0, "mean_gmm_components": 0.0,
                "frac_multimodal_cols": 0.0}
    sk, ku, is_int, ncomp = [], [], [], []
    rng = np.random.default_rng(RNG_SEED)
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        sk.append(abs(float(skew(col))))
        ku.append(float(kurtosis(col)))
        is_int.append(bool(np.all(np.equal(np.mod(col, 1), 0))))
        # multimodality: components chosen by BIC, capped
        s = col if col.size <= GMM_SUBSAMPLE else rng.choice(
            col, GMM_SUBSAMPLE, replace=False)
        s = s.reshape(-1, 1)
        if np.allclose(s, s[0]):
            ncomp.append(1)
            continue
        s = (s - s.mean()) / (s.std() + 1e-12)
        best_bic, best_k = np.inf, 1
        for k in range(1, GMM_MAX_COMPONENTS + 1):
            try:
                gm = GaussianMixture(n_components=k, covariance_type="full",
                                     random_state=RNG_SEED, reg_covar=1e-4,
                                     max_iter=100).fit(s)
                bic = gm.bic(s)
            except Exception:
                continue
            if bic < best_bic:
                best_bic, best_k = bic, k
        ncomp.append(best_k)
    return {
        "mean_abs_skew": float(np.mean(sk)) if sk else 0.0,
        "mean_kurtosis": float(np.mean(ku)) if ku else 0.0,
        "frac_integer_cols": float(np.mean(is_int)) if is_int else 0.0,
        "mean_gmm_components": float(np.mean(ncomp)) if ncomp else 0.0,
        "frac_multimodal_cols": float(np.mean([c > 1 for c in ncomp])) if ncomp else 0.0,
    }


def categorical_stats(X) -> dict:
    if X is None or X.shape[1] == 0:
        return {"mean_cardinality": 0.0, "max_cardinality": 0,
                "mean_norm_entropy": 0.0, "frac_binary_cols": 0.0,
                "onehot_dim_cat": 0}
    cards, ents, binary = [], [], []
    for j in range(X.shape[1]):
        vals, counts = np.unique(X[:, j].astype(str), return_counts=True)
        c = len(vals)
        cards.append(c)
        binary.append(c == 2)
        p = counts / counts.sum()
        h = -(p * np.log(p + 1e-300)).sum()
        ents.append(h / np.log(c) if c > 1 else 0.0)
    return {
        "mean_cardinality": float(np.mean(cards)),
        "max_cardinality": int(np.max(cards)),
        "mean_norm_entropy": float(np.mean(ents)),
        "frac_binary_cols": float(np.mean(binary)),
        "onehot_dim_cat": int(np.sum(cards)),
    }


# ---------------------------------------------------------------------------
def compute(ds: str) -> dict:
    X_num, X_cat, y, info = load_train(ds)
    cfg = dataset_cfg(ds)
    task = info["task_type"]
    is_y_cond = bool(cfg["is_y_cond"])

    # --- node set, exactly as the denoiser builds it -----------------------
    num_cols = [] if X_num is None else [X_num[:, j] for j in range(X_num.shape[1])]
    y_is_node = (task == "regression" and not is_y_cond)
    if y_is_node:
        num_cols.append(np.asarray(y, dtype=np.float64))
    cat_cols = [] if X_cat is None else [X_cat[:, j] for j in range(X_cat.shape[1])]

    n_num, n_cat = len(num_cols), len(cat_cols)
    n_rows = (X_num if X_num is not None else X_cat).shape[0]

    num_mat = np.column_stack(num_cols) if num_cols else None
    nstats = numerical_stats(num_mat)
    cstats = categorical_stats(np.column_stack(cat_cols) if cat_cols else None)

    row = dict(
        dataset=ds, task_type=task, is_y_cond=is_y_cond, y_is_node=y_is_node,
        n_rows=n_rows, n_num=n_num, n_cat=n_cat, n_features=n_num + n_cat,
        onehot_dim=n_num + cstats["onehot_dim_cat"],
        num_cat_ratio=(n_num / n_cat) if n_cat else float(n_num),
        **nstats, **cstats,
    )

    # --- target ------------------------------------------------------------
    if task == "regression":
        row["class_imbalance"] = float("nan")
        row["n_classes"] = 0
    else:
        _, counts = np.unique(np.asarray(y), return_counts=True)
        row["class_imbalance"] = float(counts.max() / counts.min())
        row["n_classes"] = int(counts.size)

    # --- dependency graph: specification matrix ----------------------------
    A = build_assoc(num_cols, cat_cols)
    row.update(graph_stats(A, "assoc"))

    # --- dependency graph: the repo's own static builder -------------------
    sys.path.insert(0, REPO)
    from tab_ddpm.graph_builder import build_static_adjacency
    cat_sizes = [int(len(np.unique(c.astype(str)))) for c in cat_cols]
    Xc_int = None
    if cat_cols:
        Xc_int = np.column_stack(
            [pd.factorize(np.asarray(c).astype(str))[0] for c in cat_cols])
    adj = build_static_adjacency(
        num_mat, Xc_int, d_num=n_num, cat_sizes=cat_sizes,
        threshold=0.1).numpy()
    row.update(graph_stats(adj, "mg"))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="default: every dataset with an ablation entry point")
    args = ap.parse_args()

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(
            d for d in os.listdir(os.path.join(REPO, "data"))
            if os.path.isdir(os.path.join(REPO, "data", d))
            and os.path.exists(os.path.join(
                REPO, "ablation", f"{d.replace('-', '_')}_ablation.py")))

    rows = []
    for ds in datasets:
        print(f"  {ds} ...", flush=True)
        rows.append(compute(ds))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "meta_features.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(df)} datasets x {df.shape[1]} columns)")
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(df[["dataset", "task_type", "n_rows", "n_num", "n_cat", "n_features",
              "assoc_n_nodes", "assoc_assoc_mean", "assoc_density_0.3",
              "mg_density_0.3", "mg_n_components_0.3"]].to_string(index=False))


if __name__ == "__main__":
    main()

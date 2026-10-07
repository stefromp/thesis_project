#!/usr/bin/env python3
"""Dependency fidelity of the three top-k rules (fixed / Jenks / entmax-1.5).

One command rebuilds every output in this directory:

    python3 analysis/dependency_fidelity/dependency_fidelity.py

Read-only on results and data. Nothing is sampled or trained; nothing is imputed.

Step 1 (this file, so far): per generation seed, compare each synthetic sample
with the REAL TRAIN split, both in original scale, on
  * association distance -- full association matrix over every column, target
    included: Pearson (num-num), correlation ratio (cat-num), Theil's U
    (cat-cat, asymmetric U(row | col), as dython defines it). Reported as the
    Frobenius norm of real - synthetic and the mean absolute difference over
    off-diagonal entries. Lower is better.
  * SDMetrics QualityReport: "Column Shapes" and "Column Pair Trends" (1 = best).
and the same metrics for REAL TEST vs REAL TRAIN (the floor).

Sample-size protocol (so the floor and the synthetic numbers share their noise):
  reference  = real train, all rows (capped at REF_CAP by a seeded subsample)
  compared   = n_cmp = min(n_test, CMP_CAP) rows: all of real test for the
               floor; a seeded subsample of each synthetic gen seed, with the
               SAME indices for every rule (synthetic sizes are equal across
               rules within a dataset).

Dataset lists, subsets, rule names and result trees come from
analysis/topk_comparison/compare_topk.py.
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "topk_comparison"))
import compare_topk as ct  # noqa: E402  (shared loaders, subsets, labels)

warnings.filterwarnings("ignore")

REPO = ct.REPO
RESULTS_ROOT = ct.RESULTS_ROOT
OUT = Path(os.environ.get("DEPFID_OUT", HERE))
DATA = REPO / "data"

CMP_CAP = 20_000
REF_CAP = 100_000
SUB_SEED = 0
N_WORKERS = min(8, os.cpu_count() or 1)



def rule_trees(variant: str) -> dict:
    """rule -> result trees in priority order, for the dataset's variant as
    chosen by compare_topk.choose_variant ("catfix" only when all three rules
    are complete in *_catfix; never mixed with the original runs)."""
    t = ct.result_trees(variant)
    return {"fixed": tuple(x for x in (t["fixed_perseed"], t["fixed_dir"]) if x),
            "jenks": (t["jenks"],), "entmax15": (t["entmax15"],)}

# metric -> (+1 higher is better | -1 lower is better, label)
METRICS = {"assoc_frobenius": (-1, "association distance, Frobenius"),
           "assoc_mean_abs": (-1, "association distance, mean |Δ| per pair"),
           "column_pair_trends": (+1, "SDMetrics Column Pair Trends"),
           "column_shapes": (+1, "SDMetrics Column Shapes"),
           "column_pair_trends_qr": (+1, "SDMetrics QualityReport Pair Trends, per-table bins (reference only)")}

# file:line evidence that X_*_train / y_train are original scale.
SCALE_EVIDENCE = (
    ("scripts/sample.py", 146, "X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])",
     "categoricals are inverse-transformed to their original labels"),
    ("scripts/sample.py", 150, "np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])",
     "`X_num_unnorm` is saved BEFORE the inverse transform (model space)"),
    ("scripts/sample.py", 151, "X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])",
     "numericals are inverse-transformed to original scale"),
    ("scripts/sample.py", 162, "y_gen = X_num[:, 0]",
     "regression without y-conditioning: y is column 0 of the inverse-transformed numericals"),
    ("scripts/sample.py", 165, "X_num = round_columns(X_num_real, X_num, disc_cols)",
     "numericals that are discrete in the real data are rounded to real values"),
    ("scripts/sample.py", 169, "np.save(os.path.join(parent_dir, 'X_num_train'), X_num)", "saved as `X_num_train`"),
    ("scripts/sample.py", 171, "np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)", "saved as `X_cat_train`"),
    ("scripts/sample.py", 172, "np.save(os.path.join(parent_dir, 'y_train'), y_gen)", "saved as `y_train`"),
    ("ablation/ablation_runner.py", 462, "from sample import sample as tabddpm_sample",
     "the ablation runner uses exactly this function"),
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def dataset_info(ds: str) -> dict:
    info = json.load(open(DATA / ds / "info.json"))
    cfg = (REPO / "ablation" / f"{ds.replace('-', '_')}_ablation.py").read_text()
    m = re.search(r'"num_samples":\s*([0-9_]+)', cfg)
    return {"task": info["task_type"], "n_num": int(info.get("n_num_features", 0)),
            "n_cat": int(info.get("n_cat_features", 0)),
            "num_samples": int(m.group(1).replace("_", "")) if m else None}


def to_frame(xn, xc, y, task: str) -> pd.DataFrame:
    """Columns in pipeline order: numericals, categoricals, then the target.
    Target is numerical for regression, categorical otherwise."""
    cols = {}
    if xn is not None:
        for i in range(xn.shape[1]):
            cols[f"num_{i}"] = xn[:, i].astype(float)
    if xc is not None:
        for j in range(xc.shape[1]):
            cols[f"cat_{j}"] = xc[:, j].astype(str)
    cols["target"] = y.astype(float) if task == "regression" else y.astype(str)
    return pd.DataFrame(cols)


def load_real(ds: str, part: str, info: dict) -> pd.DataFrame:
    d = DATA / ds
    xn = np.load(d / f"X_num_{part}.npy", allow_pickle=True) if (d / f"X_num_{part}.npy").exists() else None
    xc = np.load(d / f"X_cat_{part}.npy", allow_pickle=True) if (d / f"X_cat_{part}.npy").exists() else None
    return to_frame(xn, xc, np.load(d / f"y_{part}.npy", allow_pickle=True), info["task"])


def validate_seed(gdir: Path, info: dict, real_train: pd.DataFrame) -> dict:
    """Fully load one gen seed and check it against the real schema."""
    need = ["X_num_train.npy", "y_train.npy"] if info["n_num"] else ["y_train.npy"]
    if info["n_cat"]:
        need.append("X_cat_train.npy")
    present = sorted(p.name for p in gdir.glob("*.npy") if "unnorm" not in p.name)
    out = {"files": ";".join(present), "ok": False, "problem": ""}
    missing = [f for f in need if f not in present]
    if missing:
        out["problem"] = "missing " + ",".join(missing)
        return out
    try:
        arrs = {f: np.load(gdir / f, allow_pickle=True) for f in need}
    except Exception as exc:  # truncated / being written
        out["problem"] = f"unreadable ({type(exc).__name__})"
        return out
    rows = {len(a) for a in arrs.values()}
    if len(rows) != 1:
        out["problem"] = f"row counts differ {sorted(rows)}"
        return out
    n = rows.pop()
    df = to_frame(arrs.get("X_num_train.npy"), arrs.get("X_cat_train.npy"), arrs["y_train.npy"], info["task"])
    if list(df.columns) != list(real_train.columns):
        out["problem"] = f"columns {list(df.columns)} != real {list(real_train.columns)}"
        return out
    if info["num_samples"] and n != info["num_samples"]:
        out["problem"] = f"rows {n} != num_samples {info['num_samples']}"
        return out
    unseen = {c: int((~df[c].isin(set(real_train[c]))).sum()) for c in df.columns
              if df[c].dtype == object}
    out.update(ok=True, rows=n, unseen_categories=sum(unseen.values()),
               nonfinite=int((~np.isfinite(df.select_dtypes("number").to_numpy())).sum()))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s)[0]


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def theils_u(x: np.ndarray, y: np.ndarray) -> float:
    """U(x | y) for integer codes, as dython.nominal.theils_u."""
    hx = _entropy(np.bincount(x))
    if hx == 0:
        return 1.0
    joint = pd.crosstab(x, y).to_numpy()
    py = joint.sum(0) / joint.sum()
    h_x_given_y = sum(py[j] * _entropy(joint[:, j]) for j in range(joint.shape[1]))
    return float((hx - h_x_given_y) / hx)


def correlation_ratio(cat: np.ndarray, num: np.ndarray) -> float:
    """eta, as dython.nominal.correlation_ratio (0 when the numeric is constant)."""
    counts = np.bincount(cat)
    sums = np.bincount(cat, weights=num)
    means = sums[counts > 0] / counts[counts > 0]
    mu = num.mean()
    num_ = (counts[counts > 0] * (means - mu) ** 2).sum()
    den = ((num - mu) ** 2).sum()
    return float(np.sqrt(num_ / den)) if den > 0 else 0.0


def association_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = list(df.columns)
    is_num = [df[c].dtype != object for c in cols]
    k = len(cols)
    A = np.eye(k)
    codes = {c: _codes(df[c]) for c, n in zip(cols, is_num) if not n}
    num_idx = [i for i in range(k) if is_num[i]]
    if num_idx:
        X = df.iloc[:, num_idx].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(X, rowvar=False) if len(num_idx) > 1 else np.ones((1, 1))
        for a, i in enumerate(num_idx):
            for b, j in enumerate(num_idx):
                if i != j:
                    A[i, j] = C[a, b]
    for i in range(k):
        for j in range(k):
            if i == j or (is_num[i] and is_num[j]):
                continue
            ci, cj = cols[i], cols[j]
            if not is_num[i] and not is_num[j]:
                A[i, j] = theils_u(codes[ci], codes[cj])
            elif not is_num[i]:
                A[i, j] = correlation_ratio(codes[ci], df[cj].to_numpy(float))
            else:
                A[i, j] = correlation_ratio(codes[cj], df[ci].to_numpy(float))
    return A


def sdmetrics_scores(real: pd.DataFrame, syn: pd.DataFrame) -> dict:
    from sdmetrics.reports.single_table import QualityReport
    md = {"columns": {c: {"sdtype": "numerical" if real[c].dtype != object else "categorical"}
                      for c in real.columns}}
    rep = QualityReport()
    # sdmetrics >= 0.2x averages Column Pair Trends only over pairs whose REAL
    # correlation > 0.5 / association > 0.3 (e.g. 4 of 66 pairs on cardio).
    # The repo pins sdmetrics 0.13.0 for the TabDiff suite, which averages all
    # pairs; zero thresholds reproduce that.
    rep.real_correlation_threshold = 0
    rep.real_association_threshold = 0
    rep.generate(real, syn, md, verbose=False)
    props = rep.get_properties().set_index("Property")["Score"]
    cpt, n_scored, n_total = pair_trends_shared_bins(real, syn)
    return {"column_shapes": float(props["Column Shapes"]),
            "column_pair_trends": cpt,
            "cpt_pairs_scored": n_scored, "cpt_pairs_total": n_total,
            # QualityReport's own value, kept for transparency only: its
            # property bins real and synthetic numerical columns on EACH TABLE'S
            # OWN range (column_pair_trends.py:297-298 with bin_edges=None), so
            # a bin index means different values in the two tables whenever
            # their ranges differ (cardio num_3: max 14020 vs 2000 -> 0.017).
            "column_pair_trends_qr": float(props["Column Pair Trends"])}


def pair_trends_shared_bins(real: pd.DataFrame, syn: pd.DataFrame) -> tuple[float, int, int]:
    """Column Pair Trends as SDMetrics defines it per pair, over ALL pairs:
    CorrelationSimilarity (Pearson) for num-num, ContingencySimilarity
    otherwise, with numerical columns binned on the REAL column's edges
    (sdmetrics.utils.discretize_column, 10 equal-width bins) for both tables."""
    from sdmetrics.column_pairs import ContingencySimilarity, CorrelationSimilarity
    from sdmetrics.utils import discretize_column
    cols = list(real.columns)
    is_num = {c: real[c].dtype != object for c in cols}
    rb, sb = real.copy(), syn.copy()
    for c in cols:
        if is_num[c]:
            rb[c], sb[c] = discretize_column(real[c], syn[c])
    scores = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            try:
                if is_num[a] and is_num[b]:
                    s = CorrelationSimilarity.compute(real[[a, b]], syn[[a, b]], coefficient="Pearson")
                else:
                    s = ContingencySimilarity.compute(rb[[a, b]].astype(str), sb[[a, b]].astype(str))
            except Exception:
                s = np.nan
            scores.append(s)
    scores = np.asarray(scores, float)
    ok = np.isfinite(scores)
    return (float(scores[ok].mean()) if ok.all() else np.nan), int(ok.sum()), len(scores)


def all_metrics(real_ref: pd.DataFrame, A_ref: np.ndarray, cmp: pd.DataFrame) -> dict:
    A = association_matrix(cmp)
    D = A_ref - A
    off = ~np.eye(len(D), dtype=bool)
    finite = np.isfinite(D)
    out = {"n_nan_pairs": int((~finite & off).sum())}
    # Undefined entries (e.g. a constant synthetic column) are NOT filled in:
    # the distances are then left undefined and n_nan_pairs says why.
    if out["n_nan_pairs"]:
        out.update(assoc_frobenius=np.nan, assoc_mean_abs=np.nan)
    else:
        out.update(assoc_frobenius=float(np.sqrt((D ** 2).sum())),
                   assoc_mean_abs=float(np.abs(D[off]).mean()))
    out.update(sdmetrics_scores(real_ref, cmp))
    return out, A


# ---------------------------------------------------------------------------
# Work units (run in a pool; each loads what it needs)
# ---------------------------------------------------------------------------
_REF_CACHE: dict = {}


def reference(ds: str):
    if ds not in _REF_CACHE:
        info = dataset_info(ds)
        tr = load_real(ds, "train", info)
        if len(tr) > REF_CAP:
            tr = tr.sample(REF_CAP, random_state=SUB_SEED).reset_index(drop=True)
        te = load_real(ds, "test", info)
        n_cmp = min(len(te), CMP_CAP)
        _REF_CACHE[ds] = (info, tr, association_matrix(tr), te, n_cmp)
    return _REF_CACHE[ds]


def job_floor(ds: str) -> dict:
    info, tr, A_ref, te, n_cmp = reference(ds)
    te = te.sample(n_cmp, random_state=SUB_SEED) if len(te) > n_cmp else te
    m, A = all_metrics(tr, A_ref, te.reset_index(drop=True))
    return {"dataset": ds, "n_ref": len(tr), "n_cmp": len(te), **m, "_A": A, "_A_ref": A_ref}


def job_seed(args) -> dict:
    ds, rule, gdir = args
    info, tr, A_ref, te, n_cmp = reference(ds)
    gdir = Path(gdir)
    arr = {f: np.load(gdir / f, allow_pickle=True) for f in ("X_num_train.npy", "X_cat_train.npy", "y_train.npy")
           if (gdir / f).exists()}
    syn = to_frame(arr.get("X_num_train.npy"), arr.get("X_cat_train.npy"), arr["y_train.npy"], info["task"])
    idx = np.random.default_rng(SUB_SEED).choice(len(syn), size=min(n_cmp, len(syn)), replace=False)
    syn = syn.iloc[np.sort(idx)].reset_index(drop=True)
    m, A = all_metrics(tr, A_ref, syn)
    return {"dataset": ds, "rule": rule, "gen_seed": int(gdir.name.split("_")[-1]),
            "n_ref": len(tr), "n_cmp": len(syn), **m, "_A": A}


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
def inventory(cells: dict) -> tuple[pd.DataFrame, list]:
    rows, jobs = [], []
    for ds, cinfo in cells.items():
        if not (DATA / ds / "info.json").exists():
            continue
        info = dataset_info(ds)
        subset = "held_out" if cinfo["prov"] == "transfer" else "development"
        real_train = load_real(ds, "train", info)
        trees = rule_trees(cinfo["variant"])
        for rule in ct.RULES:
            combo, tree = cinfo["exp_name"], None
            for t in trees[rule]:
                cand = sorted((RESULTS_ROOT / ds / t).glob("gnn_*"))
                cand = [c for c in cand if c.is_dir() and (combo is None or c.name == combo)]
                if cand:
                    combo, tree = cand[0].name, cand[0]
                    break
            row = {"dataset": ds, "subset": subset, "rule": rule, "combo": combo or "",
                   "tree": ct.rel(tree.parent) if tree else "", "n_gen_seed_dirs": 0, "n_intact": 0,
                   "intact_seeds": "", "files_found": "", "problems": "", "unseen_categories": 0,
                   "nonfinite_values": 0, "variant": cinfo["variant"],
                   "cat_input_affected": cinfo["cat_affected"], "catfix_note": cinfo["catfix_note"]}
            if tree is None:
                row["status"] = ("pending" if rule == "fixed" and subset == "development" else "missing")
                rows.append(row)
                continue
            gdirs = sorted(tree.glob("gen_seed_*"))
            intact, probs, files = [], [], set()
            for g in gdirs:
                v = validate_seed(g, info, real_train)
                files |= set(v["files"].split(";")) - {""}
                if v["ok"]:
                    intact.append(g)
                    row["unseen_categories"] += v["unseen_categories"]
                    row["nonfinite_values"] += v["nonfinite"]
                else:
                    probs.append(f"{g.name}: {v['problem']}")
            row.update(n_gen_seed_dirs=len(gdirs), n_intact=len(intact),
                       intact_seeds=";".join(g.name.split("_")[-1] for g in intact),
                       files_found=";".join(sorted(files)), problems=" | ".join(probs))
            if len(intact) >= ct.EXPECTED_GEN:
                row["status"] = "complete"
            elif intact:
                row["status"] = "partial"
            elif gdirs:
                row["status"] = "corrupt"
            else:
                row["status"] = "missing"
            rows.append(row)
            jobs += [(ds, rule, str(g)) for g in intact]
    return pd.DataFrame(rows), jobs


# ---------------------------------------------------------------------------
# Step 2: comparisons between rules
# ---------------------------------------------------------------------------
STEP2_METRICS = ("column_pair_trends", "assoc_mean_abs", "column_shapes")


def step2_deltas(long: pd.DataFrame) -> pd.DataFrame:
    """Per dataset x pair x metric: delta (positive = first rule better), 95%
    bootstrap CI over gen seeds (independent resampling, N_BOOT), outcome."""
    rng = np.random.default_rng(ct.RNG_SEED)
    rows = []
    for (ds, metric), g in long[long.metric.isin(STEP2_METRICS)].groupby(["dataset", "metric"]):
        sign = METRICS[metric][0]
        vals = {r: g[g.rule == r].value.dropna().to_numpy(float) for r in ct.RULES}
        sub = g.subset.iloc[0]
        for a, b in ct.PAIRS:
            va, vb = vals[a], vals[b]
            row = {"dataset": ds, "subset": sub, "rule": a, "vs": b, "metric": metric,
                   "mean_rule": va.mean() if len(va) else np.nan, "mean_vs": vb.mean() if len(vb) else np.nan,
                   "n_rule": len(va), "n_vs": len(vb), "delta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                   "outcome": "missing"}
            if len(va) and len(vb):
                row["delta"] = sign * (va.mean() - vb.mean())
                ia = rng.integers(0, len(va), size=(ct.N_BOOT, len(va)))
                ib = rng.integers(0, len(vb), size=(ct.N_BOOT, len(vb)))
                boot = sign * (va[ia].mean(1) - vb[ib].mean(1))
                row["ci_lo"], row["ci_hi"] = np.percentile(boot, [2.5, 97.5])
                if min(len(va), len(vb)) < ct.EXPECTED_GEN:
                    row["outcome"] = "partial"
                else:
                    row["outcome"] = "win" if row["ci_lo"] > 0 else ("loss" if row["ci_hi"] < 0 else "tie")
            rows.append(row)
    return pd.DataFrame(rows)


def step2_across(pdm: pd.DataFrame, d2: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats
    rows = []
    complete = {ds for ds, g in cov.groupby("dataset") if (g.status == "complete").all()}
    for metric in STEP2_METRICS:
        sign = METRICS[metric][0]
        for sub, _, _ in ct.SUBSETS:
            use = sorted(ds for ds in complete
                         if sub == "pooled" or cov[cov.dataset == ds].subset.iloc[0] == sub)
            base = {"metric": metric, "subset": sub, "n_datasets": len(use), "datasets": ";".join(use)}
            if not use:
                rows.append({**base, "kind": "none"})
                continue
            M = np.array([[sign * pdm[(pdm.dataset == d) & (pdm.rule == r) & (pdm.metric == metric)]["mean"].iloc[0]
                           for r in ct.RULES] for d in use])
            avg = np.vstack([stats.rankdata(-row) for row in M]).mean(0)
            for r, a in zip(ct.RULES, avg):
                rows.append({**base, "kind": "avg_rank", "a": r, "value": a})
            fp = np.nan
            if len(use) >= 3:
                try:
                    fp = stats.friedmanchisquare(*M.T).pvalue
                except ValueError:
                    pass
            rows.append({**base, "kind": "friedman", "p_value": fp})
            wl = []
            for a, b in ct.PAIRS:
                d = d2[(d2.metric == metric) & (d2.rule == a) & (d2.vs == b) & d2.dataset.isin(use)]
                x = d.delta.dropna().to_numpy()
                p = np.nan
                if len(x) and np.any(x != 0):
                    try:
                        p = stats.wilcoxon(x).pvalue
                    except ValueError:
                        pass
                oc = d.outcome.value_counts()
                wl.append({**base, "kind": "wilcoxon", "a": a, "b": b, "value": float(np.median(x)) if len(x) else np.nan,
                           "p_value": p, "wins": int(oc.get("win", 0)), "ties": int(oc.get("tie", 0)),
                           "losses": int(oc.get("loss", 0))})
            for w, ph in zip(wl, ct.holm([w["p_value"] for w in wl])):
                w["p_holm"] = ph
            rows.extend(wl)
    return pd.DataFrame(rows)


def step2_figure(d2: pd.DataFrame, fl: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (OUT / "figures").mkdir(exist_ok=True)
    order = (d2[["dataset", "subset"]].drop_duplicates()
             .assign(_o=lambda x: x.subset.map({"held_out": 0, "development": 1}))
             .sort_values(["_o", "dataset"]))
    dss = order.dataset.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(dss) + 1.4), sharey=True)
    for ax, metric in zip(axes, ("column_pair_trends", "assoc_mean_abs")):
        for off, (a, b) in zip((-0.15, 0.15), (("jenks", "fixed"), ("entmax15", "fixed"))):
            d = d2[(d2.metric == metric) & (d2.rule == a) & (d2.vs == b)].set_index("dataset")
            for i, ds in enumerate(dss):
                if ds not in d.index or not np.isfinite(d.loc[ds, "delta"]):
                    continue
                r = d.loc[ds]
                ax.errorbar(r.delta, i + off, xerr=[[r.delta - r.ci_lo], [r.ci_hi - r.delta]], fmt="o", ms=4,
                            color=ct.C_RULE[a], ecolor=ct.C_RULE[a], elinewidth=1.2, capsize=0,
                            label=f"{ct.RULE_LABEL[a]} − fixed" if i == 0 or ds == dss[0] else None)
        ax.axvline(0, color=ct.C_TEXT2, lw=0.8)
        n_ho = (order.subset == "held_out").sum()
        ax.axhline(n_ho - 0.5, color=ct.C_GRID, lw=1)
        ax.set_title(f"{METRICS[metric][1]}\nΔ vs fixed (positive = better)", fontsize=9, color=ct.C_TEXT)
        ct._style(ax)
    axes[0].set_yticks(range(len(dss)), [f"{d} ({'held-out' if s == 'held_out' else 'dev'})"
                                         for d, s in zip(dss, order.subset)], fontsize=8)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "figures" / "dependency_delta_vs_fixed.png", dpi=180)
    plt.close(fig)


def step2_section(d2: pd.DataFrame, acr: pd.DataFrame, cov: pd.DataFrame) -> list[str]:
    L = ["## Step 2: rule comparisons\n",
         "Δ = rule − reference, signed so that **positive = better** (for the association distance, a smaller "
         "distance). 95% CI: bootstrap over the 5 gen seeds of each rule, resampled independently "
         f"({ct.N_BOOT:,} draws). win/loss = CI excludes 0. Held-out first; development rows are in-sample "
         "w.r.t. configuration selection. Figure: `figures/dependency_delta_vs_fixed.png`.\n"]
    L.append("### Across datasets\n")
    L.append("| metric | subset | n | avg rank fixed | Jenks | entmax-1.5 | Friedman p |")
    L.append("|---|---|---|---|---|---|---|")
    for metric in STEP2_METRICS:
        for sub, _, lab in ct.SUBSETS:
            a = acr[(acr.metric == metric) & (acr.subset == sub)]
            if a.empty or (a.kind == "none").all():
                continue
            rk = a[a.kind == "avg_rank"].set_index("a").value
            fp = a[a.kind == "friedman"].p_value.iloc[0]
            fps = f"{fp:.3f}" if np.isfinite(fp) else "n/a"
            L.append(f"| {METRICS[metric][1]} | {lab} | {int(a.n_datasets.iloc[0])} | "
                     + " | ".join(f"{rk[r]:.2f}" for r in ct.RULES) + f" | {fps} |")
    L.append("\n| metric | subset | pair | median Δ | Wilcoxon Holm p | W/T/L |")
    L.append("|---|---|---|---|---|---|")
    for metric in STEP2_METRICS:
        for sub, _, lab in ct.SUBSETS:
            for _, w in acr[(acr.metric == metric) & (acr.subset == sub) & (acr.kind == "wilcoxon")].iterrows():
                ph = f"{w.p_holm:.3f}" if np.isfinite(w.p_holm) else "n/a"
                L.append(f"| {METRICS[metric][1]} | {lab} | {ct.RULE_LABEL[w.a]} vs {ct.RULE_LABEL[w.b]} | "
                         f"{w.value:+.4f} | {ph} | {int(w.wins)}/{int(w.ties)}/{int(w.losses)} |")
    L.append("\nWith n = 4 held-out datasets the smallest attainable two-sided Wilcoxon p is 0.125.\n")
    L.append("### Per dataset: Column Pair Trends and association distance\n")
    L.append("| dataset | subset | metric | Jenks − fixed | entmax − fixed | entmax − Jenks |")
    L.append("|---|---|---|---|---|---|")
    order = (d2[["dataset", "subset"]].drop_duplicates()
             .assign(_o=lambda x: x.subset.map({"held_out": 0, "development": 1})).sort_values(["_o", "dataset"]))
    for ds, sub in zip(order.dataset, order.subset):
        for metric in ("column_pair_trends", "assoc_mean_abs"):
            cells_ = []
            for a, b in ct.PAIRS:
                r = d2[(d2.dataset == ds) & (d2.metric == metric) & (d2.rule == a) & (d2.vs == b)]
                if r.empty or not np.isfinite(r.delta.iloc[0]):
                    cells_.append("—")
                    continue
                r = r.iloc[0]
                cells_.append(f"{r.delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] {r.outcome}")
            L.append(f"| {ds} | {sub.replace('_', '-')} | {METRICS[metric][1].split(',')[0]} | " + " | ".join(cells_) + " |")
    L.append("")
    return L


# ---------------------------------------------------------------------------
def fmt(m, s):
    return "—" if not np.isfinite(m) else f"{m:.4f} ± {s:.4f}" if np.isfinite(s) else f"{m:.4f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, cells = ct.discover()
    cov, jobs = inventory(cells)
    cov.to_csv(OUT / "coverage.csv", index=False)

    datasets = sorted({j[0] for j in jobs})
    with Pool(N_WORKERS) as pool:
        floors = pool.map(job_floor, datasets)
        seeds = pool.map(job_seed, jobs, chunksize=1)

    fl = pd.DataFrame([{k: v for k, v in f.items() if not k.startswith("_")} for f in floors])
    fl.to_csv(OUT / "floor.csv", index=False)
    ps = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in seeds])
    sub = cov.set_index(["dataset", "rule"])["subset"]
    ps.insert(1, "subset", [sub[(d, r)] for d, r in zip(ps.dataset, ps.rule)])
    long = ps.melt(id_vars=["dataset", "subset", "rule", "gen_seed", "n_ref", "n_cmp", "n_nan_pairs",
                            "cpt_pairs_scored", "cpt_pairs_total"],
                   value_vars=list(METRICS), var_name="metric", value_name="value")
    long.sort_values(["dataset", "rule", "metric", "gen_seed"]).to_csv(OUT / "per_seed_metrics.csv", index=False)
    # association matrices, kept for the Step-2 heatmaps
    np.savez_compressed(OUT / "assoc_matrices.npz",
                        **{f"{f['dataset']}__real_train": f["_A_ref"] for f in floors},
                        **{f"{f['dataset']}__real_test": f["_A"] for f in floors},
                        **{f"{s['dataset']}__{s['rule']}__{s['gen_seed']}": s["_A"] for s in seeds})

    # per dataset x rule x metric: mean +/- sd over gen seeds, and vs the floor
    flx = fl.set_index("dataset")
    rows = []
    for (ds, rule, metric), g in long.groupby(["dataset", "rule", "metric"]):
        c = cov[(cov.dataset == ds) & (cov.rule == rule)].iloc[0]
        v = g.value.dropna().to_numpy()
        mean = v.mean() if len(v) else np.nan
        sd = v.std(ddof=1) if len(v) > 1 else np.nan
        f = flx.loc[ds, metric]
        rel_ = (mean / f if METRICS[metric][0] < 0 else (1 - mean) / (1 - f)) if f not in (0, 1) else np.nan
        rows.append({"dataset": ds, "subset": c.subset, "rule": rule, "metric": metric, "mean": mean, "sd": sd,
                     "n_gen_seeds": len(v), "status": c.status, "floor": f,
                     "rel_to_floor": rel_, "n_cmp": int(g.n_cmp.iloc[0]), "n_ref": int(g.n_ref.iloc[0])})
    pdm = pd.DataFrame(rows)
    pdm.to_csv(OUT / "per_dataset.csv", index=False)
    d2 = step2_deltas(long)
    d2.to_csv(OUT / "deltas.csv", index=False)
    acr = step2_across(pdm, d2, cov)
    acr.to_csv(OUT / "across_datasets.csv", index=False)
    step2_figure(d2, fl)
    write_report(cov, fl, pdm, step2_section(d2, acr, cov))

    # ---- terminal summary --------------------------------------------------
    print("Coverage (status):")
    print(cov.pivot(index="dataset", columns="rule", values="status")[list(ct.RULES)].to_string())
    print(f"\nComputed {len(seeds)} synthetic gen seeds + {len(floors)} floors.")
    pt = pdm[pdm.metric == "column_pair_trends"].pivot(index="dataset", columns="rule", values="mean")
    pt["floor"] = flx["column_pair_trends"]
    print("\nColumn Pair Trends (mean over gen seeds; 1 = best):")
    print(pt.round(4).to_string())
    print(f"\nWrote outputs to {OUT}")


def write_report(cov, fl, pdm, step2_lines=()):
    L = ["# Dependency fidelity of the top-k rules\n",
         f"_Generated by `dependency_fidelity.py` from `{ct.rel(RESULTS_ROOT)}` and `data/`. Rerun to refresh. "
         f"Variant mode: {ct.VARIANT_MODE}; excluded (still running): {', '.join(sorted(ct.EXCLUDE)) or 'none'}._\n"]
    L.append("## Which files are original scale\n")
    L.append("Synthetic samples are read from `gen_seed_<k>/X_num_train.npy`, `X_cat_train.npy`, `y_train.npy`; "
             "`X_*_unnorm.npy` are never used.\n")
    for path, line, text, what in SCALE_EVIDENCE:
        L.append(f"- {ct.cite(path, line, text)} — {what}")
    L.append("\nThe runner does not pass `disbalance`, so labels are drawn from the empirical class distribution "
             "and synthetic label frequencies are comparable with real ones.\n")

    L.append("## Protocol\n")
    L.append(f"- Reference = real **train** split (what the model learned from), all rows, capped at {REF_CAP:,} "
             f"by a seeded subsample. Compared set = n_cmp = min(n_test, {CMP_CAP:,}) rows.")
    L.append("- Floor = real **test** vs real train at the same n_cmp: the best a generator can realistically reach, "
             "since an independent real sample of that size scores this.")
    L.append(f"- Synthetic: a seeded subsample (seed {SUB_SEED}) of n_cmp rows per gen seed, identical indices for "
             "every rule. Synthetic sets are 1–8× the real train size, so no gen seed is used in full.")
    L.append("- Column types come from `data/<ds>/info.json`. The target is a column: categorical for "
             "classification, numerical for regression.")
    L.append("- Association matrix: Pearson (num–num), correlation ratio (cat–num), Theil's U (cat–cat, asymmetric "
             "U(row|col)), implemented here after dython's definitions (dython is not installed). Undefined entries "
             "(e.g. a constant synthetic column) are not filled: the distance is then left undefined.")
    L.append("- **Column Pair Trends is computed pair by pair, not taken from QualityReport.** sdmetrics 0.28.1 "
             "(installed) differs from the 0.13.0 the repo pins for the TabDiff suite in two ways that break the "
             "report value: (1) QualityReport averages only pairs whose real correlation > 0.5 / association > 0.3 "
             "(4 of 66 pairs on cardio); (2) its Pair Trends property bins numerical columns on each table's own "
             "range (`column_pair_trends.py:297-298`, `bin_edges=None`), so a bin means different values in real and "
             "synthetic whenever their ranges differ. Here: all pairs; Pearson `CorrelationSimilarity` for num–num; "
             "`ContingencySimilarity` otherwise, with numerical columns binned on the real column's edges "
             "(`sdmetrics.utils.discretize_column`, 10 equal-width bins). The QualityReport value (all pairs, "
             "per-table bins) is kept as `column_pair_trends_qr` for reference; on cardio it puts real test vs "
             "real train at 0.73 against 0.99 with shared bins. Column Shapes is taken from QualityReport "
             "(KS / TV complement; no binning involved).")
    L.append("- **This Pair Trends is a custom version (all pairs, bins shared from the real column). It is not "
             "comparable with the \"Trend\" numbers reported in the TabDiff paper**, which come from SDMetrics' own "
             "report; compare rules with each other and with the real-real floor only.")
    L.append("- Relative to floor: distance ÷ floor distance; for SDMetrics, (1 − score) ÷ (1 − floor score). "
             "1 = as good as an independent real sample; larger = worse.\n")

    L.append("## Coverage\n")
    L.append("| dataset | subset | fixed | Jenks | entmax-1.5 |")
    L.append("|---|---|---|---|---|")
    pv = cov.set_index(["dataset", "rule"])
    order = cov.assign(_o=cov.subset.map({"held_out": 0, "development": 1})).sort_values(["_o", "dataset"])
    for ds in order.dataset.unique():
        sub = cov[cov.dataset == ds].subset.iloc[0].replace("_", "-")
        cells_ = []
        for r in ct.RULES:
            c = pv.loc[(ds, r)]
            cells_.append(c.status + (f" ({c.n_intact}/{c.n_gen_seed_dirs})" if c.n_gen_seed_dirs else ""))
        mark = " §" if cov[cov.dataset == ds].cat_input_affected.iloc[0] else ""
        L.append(f"| {ds}{mark} | {sub} | " + " | ".join(cells_) + " |")
    probs = cov[cov.problems != ""]
    if len(probs):
        L.append("\nProblems found:\n")
        for _, r in probs.iterrows():
            L.append(f"- {r.dataset} / {r.rule}: {r.problems}")
    unseen = cov[cov.unseen_categories > 0]
    if len(unseen):
        L.append("\nSynthetic categorical values not present in real train (kept, counted): "
                 + "; ".join(f"{r.dataset}/{r.rule} {int(r.unseen_categories)}" for _, r in unseen.iterrows()) + ".")
    L.append("")

    L.append("## Results per dataset (mean ± sd over gen seeds)\n")
    for metric in ("column_pair_trends", "assoc_mean_abs", "column_shapes", "assoc_frobenius"):
        sign, label = METRICS[metric]
        L.append(f"### {label} ({'higher' if sign > 0 else 'lower'} is better)\n")
        L.append("| dataset | subset | n_cmp | fixed | Jenks | entmax-1.5 | real-real floor |")
        L.append("|---|---|---|---|---|---|---|")
        d = pdm[pdm.metric == metric]
        for ds in order.dataset.unique():
            if ds not in set(fl.dataset):
                continue
            g = d[d.dataset == ds].set_index("rule")
            vals = {}
            for r in ct.RULES:
                vals[r] = g.loc[r, "mean"] if r in g.index else np.nan
            have = [r for r in ct.RULES if np.isfinite(vals[r])]
            best = max(have, key=lambda r: sign * vals[r]) if have else None
            cells_ = []
            for r in ct.RULES:
                if r not in g.index:
                    cells_.append(pv.loc[(ds, r)].status)
                    continue
                x = g.loc[r]
                t = fmt(x["mean"], x["sd"])
                t = f"**{t}**" if r == best else t
                if x.status == "partial":
                    t += f" † (n={int(x.n_gen_seeds)})"
                cells_.append(t)
            f = fl.set_index("dataset").loc[ds]
            mark = " §" if cov[cov.dataset == ds].cat_input_affected.iloc[0] else ""
            L.append(f"| {ds}{mark} | {cov[cov.dataset == ds].subset.iloc[0].replace('_', '-')} | {int(f.n_cmp)} | "
                     + " | ".join(cells_) + f" | {f[metric]:.4f} |")
        L.append("")
    L.append("§ = trained with the categorical-input bug (cat_input=log; see 'Categorical-input bug'). "
             "Bold = best mean in the row among the rules present; † partial (<5 gen seeds). Development rows are "
             "in-sample with respect to configuration selection (see topk_comparison REPORT). Fixed on an ablated "
             "dataset is `ablation_fixed_seed0/` (numerical-only) or `ablation_catfix/`; on the original trees of a "
             "categorical ablated dataset it is a grid summary without samples (pending).\n")

    L.extend(step2_lines)

    import cat_input_evidence
    L.extend(cat_input_evidence.section(cov))

    L.append("## Caveats\n")
    below = []
    for metric in ("column_pair_trends", "assoc_mean_abs", "column_shapes"):
        d = pdm[pdm.metric == metric]
        b = d[d.rel_to_floor < 1]
        below.append(f"{METRICS[metric][1]} {len(b)}/{len(d)}"
                     + (f" ({', '.join(f'{r.dataset}/{r.rule}' for _, r in b.iterrows())})" if len(b) else ""))
    L.append("- **The floor is not a strict bound.** A model fit to the train split can reproduce train-specific "
             "structure more closely than an independent real sample does, so a rule can score better than "
             "real test vs real train. Cells better than the floor: " + "; ".join(below) + ". Read "
             "`rel_to_floor` < 1 as \"within real sampling noise of train\", not as better than real data.")
    n_sub = cov[cov.status.isin(["complete", "partial"])].groupby("subset").dataset.nunique().to_dict()
    L.append(f"- **n per subset** (datasets with at least one rule computed): held-out {n_sub.get('held_out', 0)}, "
             f"development {n_sub.get('development', 0)}. Held-out cells complete for all three rules: "
             + ", ".join(sorted(ds for ds, g in cov[cov.subset == "held_out"].groupby("dataset")
                                if (g.status == "complete").all())) + ".")
    part = cov[cov.status == "partial"]
    if len(part):
        L.append("- **Partial cells** (fewer than 5 intact gen seeds, shown with †): "
                 + "; ".join(f"{r.dataset}/{r.rule} {r.n_intact}/5" for _, r in part.iterrows()) + ".")
    miss = cov[cov.status.isin(["missing", "corrupt"])]
    if len(miss):
        L.append("- **Missing or corrupt:** " + "; ".join(f"{r.dataset}/{r.rule} ({r.status})"
                                                          for _, r in miss.iterrows()) + ".")
    L.append("- **Sub-sampling:** every synthetic gen seed is sub-sampled to n_cmp rows (column in the tables); "
             f"real train is used in full up to {REF_CAP:,} rows. The seed-to-seed sd therefore mixes generator "
             "variability with sub-sampling noise at that n.")
    L.append("- **n = 5 gen seeds of one trained model per cell**: training-seed variance is not in any sd.\n")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

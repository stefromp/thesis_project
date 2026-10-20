#!/usr/bin/env python3
"""
Per-pair Trend error, per-column Shape error and marginals of the Route-A synthetic
tables (seed 0, 20 tables per run), for figures 6-7 (analysis/summary/fig_pairs_marginals.py).

    # eval venv (TabDiff's evaluator: sdmetrics 0.18.0)
    python analysis/tabdiff/pair_and_marginals.py --exp_root $TRANSFER_ROOT \
        [--datasets beijing,default,magic] [--arms fixed,jenks,entmax15,dense]

Code path: for every table, TabDiff's own TabMetrics(real.csv, test.csv, val, info, cpu,
["density"]).evaluate_density -- the function its Trainer.report_test calls for "shape" and
"trend" (same real reference = data/tabdiff_gnn/<target>/real.csv, same info.json metadata,
same reorder() column typing, same sdmetrics QualityReport). Its extras hold the report's
"Column Shapes" and "Column Pair Trends" details; nothing is re-implemented.
  error_ab = 1 - score_ab    (pair, ColumnPairTrends)     error_c = 1 - score_c (column)
Self-checks (fail loudly): per table, 100 x mean over pairs of error_ab equals that table's
"trend" in tabdiff_eval/all_results.csv, and 100 x mean over columns of error_c its "shape",
both to 1e-6.

Marginals: numerical columns on 50 equal-width bins between the REAL column's 0.5 % and
99.5 % quantiles; real counts once, synthetic counts summed over the 20 tables; both divided
by (all rows x bin width), so each curve integrates to its in-window fraction. Categorical
columns: category frequencies, real vs the mean over the 20 tables.

Writes, per run, <run>/analysis/ (small files only; the run's results are never touched):
  pair_trend_error.csv    col_a, col_b, type_a, type_b, metric (correlation|contingency),
                          err_mean, err_std (ddof=1 over the tables), n_tables
  column_shape_error.csv  column, type, metric, err_mean, err_std, n_tables
  marginals.csv           column, type, bin_left, bin_right, category, real, syn
  pair_marginals_meta.json  self-check deviations, versions, sample files (sha256/16)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import route_a_arms as RA  # noqa: E402
C = RA.C                                    # tabdiff_gnn/config.py
import tabdiff_eval as TE  # noqa: E402     (hpc_transfer/tabdiff_study, on sys.path through config)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATASETS = ("beijing", "default", "magic")
TOL = 1e-6
N_BINS = 50
Q_LO, Q_HI = 0.005, 0.995
METRIC_NAME = {"CorrelationSimilarity": "correlation", "ContingencySimilarity": "contingency"}
OUT_DIR = "analysis"


# ---------------------------------------------------------------------------
# Columns in TabDiff's evaluation order (metrics.reorder): numerical block (the
# regression target appended), then categorical block (classification target appended)
# ---------------------------------------------------------------------------
def eval_columns(info: dict) -> list[dict]:
    num, cat, tgt = list(info["num_col_idx"]), list(info["cat_col_idx"]), list(info["target_col_idx"])
    if info["task_type"] == "regression":
        num += tgt
    else:
        cat += tgt
    names = info["column_names"]
    meta = {int(k): v["sdtype"] for k, v in info["metadata"]["columns"].items()}
    return [{"pos": p, "idx": i, "name": names[i], "type": meta[i]} for p, i in enumerate(num + cat)]


def sample_paths(run: str, n_tables: int) -> tuple[list[str], str]:
    """tabdiff_eval/all_samples/ (what report_test evaluated); samples/ if that is absent.
    Where both exist they must be byte-identical."""
    a = [os.path.join(run, "tabdiff_eval", "all_samples", f"samples_{i}.csv") for i in range(n_tables)]
    b = [os.path.join(run, "samples", f"samples_{i}.csv") for i in range(n_tables)]
    have_a, have_b = all(map(os.path.exists, a)), all(map(os.path.exists, b))
    if have_a and have_b:
        bad = [x for x, y in zip(a, b) if C.sha16(x) != C.sha16(y)]
        if bad:
            raise RuntimeError(f"{run}: all_samples/ and samples/ differ: {bad}")
    if have_a:
        return a, "tabdiff_eval/all_samples"
    if have_b:
        return b, "samples"
    raise FileNotFoundError(f"{run}: fewer than {n_tables} tables in tabdiff_eval/all_samples/ and samples/")


# ---------------------------------------------------------------------------
# One table through TabDiff's evaluate_density
# ---------------------------------------------------------------------------
def table_details(metrics, syn: pd.DataFrame, cols: list[dict]):
    """(shape_score, trend_score, column frame, pair frame) of one synthetic table."""
    out, extras = metrics.evaluate_density(syn.copy())
    name = {c["pos"]: c["name"] for c in cols}
    typ = {c["pos"]: c["type"] for c in cols}
    sh = extras["shapes"]
    shapes = pd.DataFrame({"column": [name[int(c)] for c in sh["Column"]],
                           "type": [typ[int(c)] for c in sh["Column"]],
                           "metric": sh["Metric"].to_numpy(), "err": 1.0 - sh["Score"].to_numpy(float)})
    tr = extras["trends"]
    unknown = set(tr["Metric"]) - set(METRIC_NAME)
    if unknown:
        raise RuntimeError(f"unexpected Column Pair Trends metric(s) {unknown}")
    pairs = pd.DataFrame({"col_a": [name[int(c)] for c in tr["Column 1"]],
                          "col_b": [name[int(c)] for c in tr["Column 2"]],
                          "type_a": [typ[int(c)] for c in tr["Column 1"]],
                          "type_b": [typ[int(c)] for c in tr["Column 2"]],
                          "metric": [METRIC_NAME[m] for m in tr["Metric"]],
                          "err": 1.0 - tr["Score"].to_numpy(float)})
    return float(out["density/Shape"]), float(out["density/Trend"]), shapes, pairs


def check(what: str, got: float, want: float, tol: float = TOL) -> float:
    d = abs(got - want)
    if not np.isfinite(got) or not d <= tol:
        raise AssertionError(f"SELF-CHECK FAILED {what}: {got!r} vs {want!r} (|diff| {d:.3e} > {tol:g})")
    return d


def aggregate(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """err_mean / err_std (ddof=1, TabDiff's aggregation) over the tables, per key."""
    allf = pd.concat([f.assign(table=i) for i, f in enumerate(frames)], ignore_index=True)
    g = allf.groupby(keys, sort=False)["err"]
    out = g.agg(err_mean="mean", err_std=lambda x: x.std(ddof=1), n_tables="count").reset_index()
    return out


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------
def numeric_edges(real: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(real, [Q_LO, Q_HI])
    if not hi > lo:                         # (near-)constant window: fall back to the full range
        lo, hi = float(np.min(real)), float(np.max(real))
    if not hi > lo:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, N_BINS + 1)


def _density(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """counts per bin / (all rows x bin width): values outside the window count in the
    denominator, so the curve integrates to the in-window fraction."""
    x = x[np.isfinite(x)]
    c, _ = np.histogram(x, edges)
    return c, len(x)


def marginals(real: pd.DataFrame, syns: list[pd.DataFrame], cols: list[dict]) -> pd.DataFrame:
    rows = []
    for c in cols:
        r = real.iloc[:, c["idx"]]
        if c["type"] == "numerical":
            rv = r.to_numpy(float)
            edges = numeric_edges(rv[np.isfinite(rv)])
            w = np.diff(edges)
            rc, rn = _density(rv, edges)
            sc, sn = np.zeros(N_BINS), 0
            for s in syns:
                cc, nn = _density(s.iloc[:, c["idx"]].to_numpy(float), edges)
                sc, sn = sc + cc, sn + nn
            for j in range(N_BINS):
                rows.append({"column": c["name"], "type": "numerical", "bin_left": edges[j],
                             "bin_right": edges[j + 1], "category": "", "real": rc[j] / (rn * w[j]),
                             "syn": sc[j] / (sn * w[j])})
        else:
            rf = r.astype(str).value_counts(normalize=True)
            sf = pd.concat([s.iloc[:, c["idx"]].astype(str).value_counts(normalize=True).rename(i)
                            for i, s in enumerate(syns)], axis=1).fillna(0.0).mean(axis=1)
            cats = list(rf.index) + [k for k in sf.sort_values(ascending=False).index if k not in rf.index]
            for k in cats:
                rows.append({"column": c["name"], "type": "categorical", "bin_left": np.nan,
                             "bin_right": np.nan, "category": k, "real": float(rf.get(k, 0.0)),
                             "syn": float(sf.get(k, 0.0))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def process_run(run: str, ds: str, n_tables: int, metrics_factory, repo: str = REPO) -> dict:
    files = C.data_files(C.target_name(ds), repo)
    info = json.load(open(files["info"]))
    cols = eval_columns(info)
    ar = pd.read_csv(os.path.join(run, "tabdiff_eval", "all_results.csv"), index_col=0)
    if len(ar) != n_tables:
        raise RuntimeError(f"{run}: all_results.csv has {len(ar)} rows, expected {n_tables}")
    paths, where = sample_paths(run, n_tables)
    metrics = metrics_factory(files, info)
    real = pd.read_csv(files["real"])
    shapes, pairs, syns, dev = [], [], [], {"trend": 0.0, "shape": 0.0, "trend_property": 0.0,
                                             "shape_property": 0.0}
    for i, p in enumerate(paths):
        syn = pd.read_csv(p)
        if list(syn.columns) != list(real.columns):
            raise RuntimeError(f"{p}: columns {list(syn.columns)} != real.csv {list(real.columns)}")
        s_score, t_score, sh, pr = table_details(metrics, syn, cols)
        t_mean, s_mean = 100 * pr["err"].mean(), 100 * sh["err"].mean()
        # the report's property score is the mean of its details ...
        dev["trend_property"] = max(dev["trend_property"], check(f"{run} table {i} trend property",
                                                                  t_mean, 100 * (1 - t_score)))
        dev["shape_property"] = max(dev["shape_property"], check(f"{run} table {i} shape property",
                                                                  s_mean, 100 * (1 - s_score)))
        # ... and equals what TabDiff's report_test wrote for this table
        dev["trend"] = max(dev["trend"], check(f"{run} table {i} trend vs all_results.csv",
                                               t_mean, float(ar.loc[i, "trend"])))
        dev["shape"] = max(dev["shape"], check(f"{run} table {i} shape vs all_results.csv",
                                               s_mean, float(ar.loc[i, "shape"])))
        shapes.append(sh)
        pairs.append(pr)
        syns.append(syn)
        print(f"    table {i:2d}: trend {t_mean:.6f} shape {s_mean:.6f}  (self-checks ok)", flush=True)
    out = os.path.join(run, OUT_DIR)
    os.makedirs(out, exist_ok=True)
    aggregate(pairs, ["col_a", "col_b", "type_a", "type_b", "metric"]).to_csv(
        os.path.join(out, "pair_trend_error.csv"), index=False)
    aggregate(shapes, ["column", "type", "metric"]).to_csv(os.path.join(out, "column_shape_error.csv"), index=False)
    marginals(real, syns, cols).to_csv(os.path.join(out, "marginals.csv"), index=False)
    meta = {"dataset": ds, "run": run, "n_tables": n_tables, "samples_from": where,
            "samples_sha256_16": [C.sha16(p) for p in paths], "real": files["real"],
            "real_sha256_16": C.sha16(files["real"]), "info_sha256_16": C.sha16(files["info"]),
            "self_check_tol": TOL, "self_check_max_abs_dev": dev,
            "nan_pair_scores": int(sum(p["err"].isna().sum() for p in pairs)),
            "finished_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return meta


def tabdiff_metrics_factory(repo: str = REPO):
    """TabMetrics with the density metric only, exactly as report_tables builds it."""
    root = os.path.join(repo, C.TABDIFF_REL)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tabdiff.metrics import TabMetrics

    def make(files, info):
        return TabMetrics(files["real"], files["test"], files.get("val"), info, "cpu", ["density"])
    return make


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", required=True, help="the directory holding exp_tabdiff_gnn/")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--arms", default=",".join(RA.ARMS))
    ap.add_argument("--n_tables", type=int, default=C.N_TABLES)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--no_check_pins", action="store_true", help="skip the evaluator package pins (tests only)")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    td = C.verify_tabdiff(a.repo)
    versions = TE._versions(["sdmetrics", "pandas", "numpy", "sklearn"])
    if not a.no_check_pins:
        TE._check_pins(versions, {"sdmetrics": C.S.EVAL_PINS["sdmetrics"]})
    factory = tabdiff_metrics_factory(a.repo)
    done, missing = [], []
    for ds in a.datasets.split(","):
        for arm in a.arms.split(","):
            if not RA.has_arm(ds, arm):
                continue
            run = RA.arm_dir(a.exp_root, ds, arm, 0, a.repo)
            if run is None or not os.path.exists(os.path.join(run, "tabdiff_eval", "all_results.csv")):
                missing.append(f"{ds}/{arm}")
                print(f"  [skip] {ds}/{arm}: no evaluated run on disk", flush=True)
                continue
            print(f"  [{ds}/{arm}] {run}", flush=True)
            meta = process_run(run, ds, a.n_tables, factory, a.repo)
            meta.update({"arm": arm, "rule_token": RA.rule_token(ds, arm, a.repo), "versions": versions,
                         "tabdiff": {"commit": td["commit"]},
                         "producer": "analysis/tabdiff/pair_and_marginals.py"})
            with open(os.path.join(run, OUT_DIR, "pair_marginals_meta.json"), "w") as fh:
                json.dump(meta, fh, indent=2)
            d = meta["self_check_max_abs_dev"]
            print(f"  [{ds}/{arm}] self-checks passed on {a.n_tables} tables: max |diff| trend {d['trend']:.2e}, "
                  f"shape {d['shape']:.2e}", flush=True)
            done.append(f"{ds}/{arm}")
    print(f"done: {', '.join(done) or 'none'}; not on disk: {', '.join(missing) or 'none'}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())

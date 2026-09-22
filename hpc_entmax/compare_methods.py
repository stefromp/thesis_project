#!/usr/bin/env python3
"""Fixed top-k (baseline) vs Jenks vs sparsemax vs entmax-1.5, side by side.

Reads, never writes, the result trees:

    baseline   tabddpm_ablation/*.json                  fixed top-k (Kaggle);
                                                         only mean +/- std on disk
    jenks      exp/<ds>/ablation_adaptive_jenks/<combo>/
    sparsemax  exp/<ds>/ablation_adaptive_sparsemax/<combo>/
    entmax15   exp/<ds>/ablation_adaptive_entmax15/<combo>/

and writes, into hpc_entmax/ (never into hpc/):

    method_comparison.csv      one row per BASELINE combination: each metric as
                               mean/std for all four methods, plus the mean
                               sampling-phase budget k of each adaptive method
    method_comparison_wtl.csv  win/tie/loss counts per dataset x metric x pair
    budget_curves.csv          sampling-phase k_mean per t-bucket, per dataset x
                               adaptive method, averaged over combinations

Rows. The adaptive modes do not use top_k, so baseline combinations that differ
only in top_k collapse onto one adaptive run (hpc/adaptive_manifest.py, read
only). method_comparison.csv keeps one row per baseline combination -- abalone
has 19 -- and repeats the adaptive cells across the top_k values they replace.

Win / tie / loss, for one metric, method A vs method B on one row:
    tie   if |mean_A - mean_B| < max(std_A, std_B)
    win   otherwise, if A is better in the metric's direction
    loss  otherwise
Directions: higher is better for score (accuracy / F1 for classification,
-RMSE for regression) and dcr_tabddpm_mean; lower is better for
wasserstein_mean and mia_auc.
    adaptive vs baseline : counted over the per-baseline-combination rows
    adaptive vs adaptive : counted over the unique adaptive combinations
Rows where either side is missing or NaN are excluded and counted as n_na.

Every std is over generation (x classifier) seeds of ONE trained model, and the
baseline exists on disk only as mean +/- std, so all comparisons are unpaired
and descriptive. Nothing here is a significance test.

    python hpc_entmax/compare_methods.py --repo_root "$REPO_ROOT"
    python hpc_entmax/compare_methods.py --repo_root "$REPO_ROOT" --datasets abalone

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import json
import math
import os
import sys
from collections import defaultdict

# Every dataset with a job script in hpc_entmax/jobs/. A method with no runs
# for a dataset (e.g. Jenks outside wilt/abalone/gesture) shows as 0/N.
DEFAULT_DATASETS = ("wilt", "abalone", "gesture", "buddy", "churn2", "diabetes", "insurance")

# (column label, topk_mode or None for the baseline)
METHODS = (("baseline", None),
           ("jenks", "adaptive_jenks"),
           ("sparsemax", "adaptive_sparsemax"),
           ("entmax15", "adaptive_entmax15"))
ADAPTIVE = tuple(label for label, mode in METHODS if mode)

# (metric, +1 = higher is better, -1 = lower is better)
METRICS = (("score", +1),
           ("wasserstein_mean", -1),
           ("mia_auc", -1),
           ("dcr_tabddpm_mean", +1))

# (A, B, rows): A's wins/ties/losses against B.
PAIRS = (("jenks", "baseline", "baseline_rows"),
         ("sparsemax", "baseline", "baseline_rows"),
         ("entmax15", "baseline", "baseline_rows"),
         ("sparsemax", "jenks", "unique_combos"),
         ("entmax15", "jenks", "unique_combos"),
         ("entmax15", "sparsemax", "unique_combos"))
SHORT = {"baseline": "base", "jenks": "jnk", "sparsemax": "spm", "entmax15": "e15"}

NAN = float("nan")


def _setup_paths(repo_root: str) -> None:
    for sub in ("hpc", "ablation"):
        path = os.path.join(repo_root, sub)
        if path not in sys.path:
            sys.path.insert(0, path)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _metric(metrics: dict, key: str):
    value = (metrics or {}).get(key)
    if isinstance(value, dict):
        mean, std = value.get("mean", NAN), value.get("std", NAN)
        return (mean if _finite(mean) else NAN, std if _finite(std) else NAN)
    return (NAN, NAN)


def load_baseline(repo_root: str) -> dict:
    """(dataset, exp_name) -> metrics, from every finished baseline entry."""
    out = {}
    for path in sorted(glob.glob(os.path.join(repo_root, "tabddpm_ablation", "*.json"))):
        with open(path) as fh:
            try:
                entries = json.load(fh)
            except json.JSONDecodeError:
                continue
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and entry.get("status") == "done":
                out[(entry.get("dataset"), entry.get("exp_name"))] = entry.get("metrics", {})
    return out


def load_adaptive(repo_root: str, dataset: str, mode: str, exp_name: str):
    """(metrics or None, k_sample, {t_bucket: k}) for one adaptive run."""
    run_dir = os.path.join(repo_root, "exp", dataset, f"ablation_{mode}", exp_name)
    metrics = None
    results = os.path.join(run_dir, "results_full_averaged.json")
    if os.path.exists(results):
        with open(results) as fh:
            metrics = json.load(fh)

    # Sampling rows only: that is the phase that generated the evaluated data.
    # n_samples-weighted over blocks and gen seeds.
    total, count = 0.0, 0
    per_bucket = defaultdict(lambda: [0.0, 0])
    stats = os.path.join(run_dir, "topk_stats.csv")
    if os.path.exists(stats):
        with open(stats, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("phase") != "sample":
                    continue
                try:
                    k, n, bucket = float(row["k_mean"]), int(row["n_samples"]), int(row["t_bucket"])
                except (KeyError, TypeError, ValueError):
                    continue
                total += k * n
                count += n
                per_bucket[bucket][0] += k * n
                per_bucket[bucket][1] += n
    k_sample = total / count if count else NAN
    curve = {b: s / n for b, (s, n) in per_bucket.items() if n}
    return metrics, k_sample, curve


def n_nodes(repo_root: str, dataset: str) -> int:
    """Graph size N as GraphAwareDenoiser builds it: one node per numerical and
    categorical feature, plus y as a numerical node for regression without y
    conditioning (abalone). The budget k ranges over 1 .. N-1."""
    cfg = importlib.import_module(f"{dataset}_ablation").DATASET_CFG
    with open(os.path.join(repo_root, cfg["real_data_path"], "info.json")) as fh:
        info = json.load(fh)
    extra = int(info.get("task_type") == "regression" and not cfg["is_y_cond"])
    return int(cfg["num_numerical_features"]) + int(info.get("n_cat_features", 0)) + extra


def wtl(rows, a: str, b: str, metric: str, direction: int):
    wins = ties = losses = n_na = 0
    for row in rows:
        ma, sa = row[f"{metric}_{a}_mean"], row[f"{metric}_{a}_std"]
        mb, sb = row[f"{metric}_{b}_mean"], row[f"{metric}_{b}_std"]
        if not all(_finite(x) for x in (ma, sa, mb, sb)):
            n_na += 1
            continue
        diff = ma - mb
        if abs(diff) < max(sa, sb):
            ties += 1
        elif diff * direction > 0:
            wins += 1
        else:
            losses += 1
    return wins, ties, losses, n_na


def _fmt(x, digits=4):
    return f"{x:.{digits}f}" if _finite(x) else "--"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", required=True)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--out_dir", default=None, help="default: <repo_root>/hpc_entmax")
    args = p.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_dir = os.path.abspath(args.out_dir or os.path.join(repo_root, "hpc_entmax"))
    if os.path.commonpath([out_dir, os.path.join(repo_root, "hpc")]) == os.path.join(repo_root, "hpc"):
        p.error("refusing to write under hpc/ -- that directory belongs to the Jenks runs")
    os.makedirs(out_dir, exist_ok=True)

    _setup_paths(repo_root)
    import adaptive_manifest  # hpc/, read-only
    manifest = adaptive_manifest.adaptive_manifest(repo_root)
    baseline = load_baseline(repo_root)

    combo_rows, wtl_rows, curve_rows, warnings = [], [], [], []
    summary = {}

    for dataset in args.datasets:
        if dataset not in manifest:
            warnings.append(f"{dataset}: not in hpc/adaptive_manifest.py -- skipped")
            continue
        N = n_nodes(repo_root, dataset)
        unique_rows, per_row = [], []
        found = {label: 0 for label in ADAPTIVE}
        k_by_method = defaultdict(list)
        curves = defaultdict(lambda: defaultdict(list))   # method -> bucket -> [k per combo]

        for run in manifest[dataset]:
            adaptive_name = adaptive_manifest.exp_dir_name(run)
            cells = {}
            for label, mode in METHODS:
                if mode is None:
                    continue
                metrics, k_sample, curve = load_adaptive(repo_root, dataset, mode, adaptive_name)
                if metrics is not None:
                    found[label] += 1
                    for metric, _ in METRICS:
                        if not _finite(_metric(metrics, metric)[0]):
                            warnings.append(f"{dataset}/{label}/{adaptive_name}: {metric} is NaN/missing"
                                            + (" -- run hpc_entmax/backfill_privacy.py" if label == "jenks" else ""))
                for metric, _ in METRICS:
                    cells[f"{metric}_{label}_mean"], cells[f"{metric}_{label}_std"] = _metric(metrics, metric)
                cells[f"k_sample_{label}"] = k_sample
                if _finite(k_sample):
                    k_by_method[label].append(k_sample)
                for bucket, k in curve.items():
                    curves[label][bucket].append(k)

            unique_rows.append(dict(cells))
            for top_k in run["baseline_top_ks"]:
                base_name = adaptive_manifest.exp_dir_name({**run, "top_k": top_k})
                base_metrics = baseline.get((dataset, base_name))
                if base_metrics is None:
                    warnings.append(f"{dataset}: baseline entry {base_name} not found")
                row = {"dataset": dataset, "gnn_type": run["gnn_type"],
                       "n_layers": run["n_layers"], "d_model": run["d_model"],
                       "n_heads": run["n_heads"], "baseline_top_k": top_k,
                       "baseline_exp_name": base_name, "adaptive_exp_name": adaptive_name,
                       "n_nodes": N}
                for metric, _ in METRICS:
                    row[f"{metric}_baseline_mean"], row[f"{metric}_baseline_std"] = \
                        _metric(base_metrics, metric)
                row.update(cells)
                per_row.append(row)

        combo_rows += per_row

        # ---- win / tie / loss ---------------------------------------------
        for metric, direction in METRICS:
            for a, b, basis in PAIRS:
                rows = per_row if basis == "baseline_rows" else unique_rows
                w, t, l, na = wtl(rows, a, b, metric, direction)
                wtl_rows.append({"dataset": dataset, "metric": metric,
                                 "better": "higher" if direction > 0 else "lower",
                                 "method_a": a, "method_b": b, "counted_over": basis,
                                 "wins": w, "ties": t, "losses": l, "n_na": na,
                                 "n_rows": len(rows)})

        # ---- budget curves -------------------------------------------------
        for label in ADAPTIVE:
            for bucket in sorted(curves[label]):
                ks = curves[label][bucket]
                mean = sum(ks) / len(ks)
                std = math.sqrt(sum((k - mean) ** 2 for k in ks) / len(ks))
                curve_rows.append({"dataset": dataset, "method": label, "t_bucket": bucket,
                                   "k_mean": mean, "k_std_over_combos": std,
                                   "n_combos": len(ks), "n_nodes": N, "k_max": N - 1,
                                   "k_frac": mean / (N - 1) if N > 1 else NAN})

        summary[dataset] = {"n_rows": len(per_row), "n_unique": len(unique_rows),
                            "found": found, "N": N,
                            "k": {m: (sum(v) / len(v) if v else NAN) for m, v in k_by_method.items()}}

    # ---- write ---------------------------------------------------------------
    combo_cols = ["dataset", "gnn_type", "n_layers", "d_model", "n_heads",
                  "baseline_top_k", "baseline_exp_name", "adaptive_exp_name", "n_nodes"]
    for metric, _ in METRICS:
        for label, _ in METHODS:
            combo_cols += [f"{metric}_{label}_mean", f"{metric}_{label}_std"]
    combo_cols += [f"k_sample_{label}" for label in ADAPTIVE]

    paths = {"combo": os.path.join(out_dir, "method_comparison.csv"),
             "wtl": os.path.join(out_dir, "method_comparison_wtl.csv"),
             "curves": os.path.join(out_dir, "budget_curves.csv")}
    for key, rows, cols in (
            ("combo", combo_rows, combo_cols),
            ("wtl", wtl_rows, ["dataset", "metric", "better", "method_a", "method_b",
                               "counted_over", "wins", "ties", "losses", "n_na", "n_rows"]),
            ("curves", curve_rows, ["dataset", "method", "t_bucket", "k_mean",
                                    "k_std_over_combos", "n_combos", "n_nodes", "k_max", "k_frac"])):
        with open(paths[key], "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: ("" if isinstance(row.get(c), float) and math.isnan(row[c])
                                     else row.get(c, "")) for c in cols})

    # ---- compact summary -----------------------------------------------------
    print("=" * 96)
    print("  Fixed top-k vs Jenks vs sparsemax vs entmax-1.5   (W/T/L of the first method; "
          "tie = |diff| < max std)")
    print("  Unpaired, descriptive only -- not a significance test.")
    print("=" * 96)
    pair_labels = [f"{SHORT[a]}-vs-{SHORT[b]}" for a, b, _ in PAIRS]
    for dataset, info in summary.items():
        found = "  ".join(f"{m} {info['found'][m]}/{info['n_unique']}" for m in ADAPTIVE)
        ks = "  ".join(f"{m} {_fmt(info['k'].get(m, NAN), 2)}" for m in ADAPTIVE)
        print(f"\n  {dataset}   ({info['n_rows']} baseline rows, {info['n_unique']} unique combos, "
              f"N={info['N']} nodes, k in 1..{info['N'] - 1})")
        print(f"    runs found      : {found}")
        print(f"    mean k (sample) : {ks}")
        print(f"    {'metric':18s} {'better':6s} " + " ".join(f"{lab:>13s}" for lab in pair_labels))
        for metric, direction in METRICS:
            cells = []
            for a, b, _ in PAIRS:
                r = next(r for r in wtl_rows if r["dataset"] == dataset and r["metric"] == metric
                         and r["method_a"] == a and r["method_b"] == b)
                done = r["n_rows"] - r["n_na"]
                cells.append(f"{r['wins']}/{r['ties']}/{r['losses']}" if done else "--")
            print(f"    {metric:18s} {'higher' if direction > 0 else 'lower':6s} "
                  + " ".join(f"{c:>13s}" for c in cells))

    if warnings:
        print("\n  WARNINGS")
        for w in warnings:
            print(f"    {w}")
    print()
    for key in ("combo", "wtl", "curves"):
        print(f"  -> {paths[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

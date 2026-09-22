#!/usr/bin/env python3
"""How many neighbours does each adaptive rule keep? Budget k vs graph size N.

Reads every run's topk_stats.csv (never writes into a run directory):

    wide     wide_runs/<protocol>/exp/<ds>/ablation_<mode>/<combo>/   (hpc_wide/)
    main     exp/<ds>/ablation_<mode>/<combo>/                         (hpc/, hpc_entmax/)

"main" (on by default, --no_main to skip) adds the manifest datasets (wilt ...
gesture, N = 5-32), so one table covers k against N from 5 to 52 nodes.
Smoke runs (wide_runs/smoke/) are ignored.

k is the number of kept off-diagonal edges per node (1 .. N-1), averaged over
nodes and samples as DynamicAdjacency logs it; every aggregate below is
n_samples-weighted. Writes, into hpc_wide/ (never into hpc/):

    budget_runs.csv      one row per run: N, k at the start and end of
                         training, k when sampling (overall, at t near 0 and
                         t near T), k / (N-1)
    budget_curves.csv    sampling-phase k per run x block x t-bucket
    budget_training.csv  training-phase k per run x logged step (trajectory)

    python3 hpc_wide/budget_summary.py --repo_root "$REPO_ROOT"
    python3 hpc_wide/budget_summary.py --repo_root "$REPO_ROOT" --no_main --datasets miniboone king

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict

MODES = ("adaptive_jenks", "adaptive_sparsemax", "adaptive_entmax15")
SHORT = {"adaptive_jenks": "jenks", "adaptive_sparsemax": "sparsemax", "adaptive_entmax15": "entmax15"}
EARLY, LATE = 0.10, 0.90   # first / last 10% of the logged training steps

NAN = float("nan")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _wmean(pairs):
    """n_samples-weighted mean of (k, n) pairs."""
    total = sum(k * n for k, n in pairs)
    count = sum(n for _, n in pairs)
    return total / count if count else NAN


def read_stats(path: str):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append({"phase": row["phase"], "step": int(row["step"]),
                             "block": int(row["block"]) if row["block"] not in ("", "None") else -1,
                             "t_bucket": int(row["t_bucket"]), "k": float(row["k_mean"]),
                             "n": int(row["n_samples"])})
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def find_runs(repo_root: str, include_main: bool, datasets):
    """(source, dataset, mode, exp_name, run_dir) for every run with a topk_stats.csv."""
    patterns = [("wide_budget", os.path.join(repo_root, "wide_runs", "budget", "exp")),
                ("wide_full", os.path.join(repo_root, "wide_runs", "full", "exp"))]
    if include_main:
        patterns.append(("main", os.path.join(repo_root, "exp")))
    runs = []
    for source, root in patterns:
        for mode in MODES:
            for stats in sorted(glob.glob(os.path.join(root, "*", f"ablation_{mode}", "*", "topk_stats.csv"))):
                run_dir = os.path.dirname(stats)
                dataset = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
                if datasets and dataset not in datasets:
                    continue
                runs.append((source, dataset, mode, os.path.basename(run_dir), run_dir))
    return runs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", required=True)
    p.add_argument("--datasets", nargs="+", default=None, help="default: every dataset found")
    p.add_argument("--no_main", action="store_true", help="skip the manifest runs under exp/")
    p.add_argument("--out_dir", default=None, help="default: <repo_root>/hpc_wide")
    args = p.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_dir = os.path.abspath(args.out_dir or os.path.join(repo_root, "hpc_wide"))
    hpc_dir = os.path.join(repo_root, "hpc")
    if os.path.commonpath([out_dir, hpc_dir]) == hpc_dir:
        p.error("refusing to write under hpc/ -- that directory belongs to the Jenks runs")
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(repo_root, "hpc_wide"))
    import wide_manifest

    node_cache = {}

    def nodes(dataset):
        if dataset not in node_cache:
            try:
                node_cache[dataset] = wide_manifest.n_nodes(repo_root, dataset)
            except Exception:
                node_cache[dataset] = None
        return node_cache[dataset]

    run_rows, curve_rows, train_rows = [], [], []
    for source, dataset, mode, exp_name, run_dir in find_runs(repo_root, not args.no_main,
                                                             set(args.datasets or [])):
        stats = read_stats(os.path.join(run_dir, "topk_stats.csv"))
        N = nodes(dataset)
        k_max = N - 1 if N else None
        train = [r for r in stats if r["phase"] == "train"]
        sample = [r for r in stats if r["phase"] == "sample"]

        k_early = k_late = NAN
        if train:
            last = max(r["step"] for r in train)
            k_early = _wmean([(r["k"], r["n"]) for r in train if r["step"] <= EARLY * last])
            k_late = _wmean([(r["k"], r["n"]) for r in train if r["step"] >= LATE * last])
        k_sample = _wmean([(r["k"], r["n"]) for r in sample])
        k_t0 = _wmean([(r["k"], r["n"]) for r in sample if r["t_bucket"] == 0])
        k_t9 = _wmean([(r["k"], r["n"]) for r in sample if r["t_bucket"] == 9])
        gnn_type = exp_name.split("_")[1] if exp_name.startswith("gnn_") else ""
        finished = os.path.exists(os.path.join(run_dir, "results_full_averaged.json"))

        run_rows.append({
            "source": source, "dataset": dataset, "n_nodes": N if N else "", "k_max": k_max or "",
            "mode": SHORT[mode], "gnn_type": gnn_type, "exp_name": exp_name, "finished": finished,
            "k_train_early": k_early, "k_train_late": k_late, "k_sample": k_sample,
            "k_sample_frac": k_sample / k_max if k_max and _finite(k_sample) else NAN,
            "k_sample_t0": k_t0, "k_sample_t9": k_t9,
            "n_train_rows": len(train), "n_sample_rows": len(sample)})

        by_bt = defaultdict(list)
        for r in sample:
            by_bt[(r["block"], r["t_bucket"])].append((r["k"], r["n"]))
        for (block, bucket), pairs in sorted(by_bt.items()):
            k = _wmean(pairs)
            curve_rows.append({"source": source, "dataset": dataset, "n_nodes": N or "",
                               "mode": SHORT[mode], "exp_name": exp_name, "block": block,
                               "t_bucket": bucket, "k_mean": k,
                               "k_frac": k / k_max if k_max else NAN})
        by_step = defaultdict(list)
        for r in train:
            by_step[r["step"]].append((r["k"], r["n"]))
        for step, pairs in sorted(by_step.items()):
            train_rows.append({"source": source, "dataset": dataset, "mode": SHORT[mode],
                               "exp_name": exp_name, "step": step, "k_mean": _wmean(pairs)})

    run_rows.sort(key=lambda r: (r["n_nodes"] if r["n_nodes"] != "" else 999, r["dataset"],
                                 MODES.index(next(m for m in MODES if SHORT[m] == r["mode"])),
                                 r["exp_name"], r["source"]))

    def write(name, rows, cols):
        path = os.path.join(out_dir, name)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: ("" if isinstance(row.get(c), float) and math.isnan(row[c])
                                     else row.get(c, "")) for c in cols})
        return path

    paths = [
        write("budget_runs.csv", run_rows,
              ["source", "dataset", "n_nodes", "k_max", "mode", "gnn_type", "exp_name", "finished",
               "k_train_early", "k_train_late", "k_sample", "k_sample_frac", "k_sample_t0",
               "k_sample_t9", "n_train_rows", "n_sample_rows"]),
        write("budget_curves.csv", curve_rows,
              ["source", "dataset", "n_nodes", "mode", "exp_name", "block", "t_bucket",
               "k_mean", "k_frac"]),
        write("budget_training.csv", train_rows,
              ["source", "dataset", "mode", "exp_name", "step", "k_mean"]),
    ]

    # ---- compact summary: dataset x mode, mean over combos -------------------
    def fmt(x, d=2):
        return f"{x:.{d}f}" if _finite(x) else "--"

    print("=" * 100)
    print("  Adaptive budget k (kept neighbours per node) vs graph size N -- mean over combinations")
    print("  k_sample = sampling phase; t0 / t9 = t near 0 / near T; frac = k_sample / (N-1)")
    print("=" * 100)
    print(f"  {'dataset':12s} {'N':>3s} {'source':12s} {'mode':10s} {'runs':>4s} "
          f"{'k_train end':>11s} {'k_sample':>9s} {'[min-max]':>13s} {'frac':>6s} {'t0':>6s} {'t9':>6s}")
    groups = defaultdict(list)
    for r in run_rows:
        groups[(r["n_nodes"], r["dataset"], r["source"], r["mode"])].append(r)
    for (N, dataset, source, mode), rows in groups.items():
        def avg(key):
            vals = [r[key] for r in rows if _finite(r[key])]
            return sum(vals) / len(vals) if vals else NAN
        ks = [r["k_sample"] for r in rows if _finite(r["k_sample"])]
        rng = f"[{min(ks):.1f}-{max(ks):.1f}]" if ks else "--"
        print(f"  {dataset:12s} {str(N):>3s} {source:12s} {mode:10s} {len(rows):4d} "
              f"{fmt(avg('k_train_late')):>11s} {fmt(avg('k_sample')):>9s} {rng:>13s} "
              f"{fmt(avg('k_sample_frac')):>6s} {fmt(avg('k_sample_t0')):>6s} {fmt(avg('k_sample_t9')):>6s}")
    if not run_rows:
        print("  (no topk_stats.csv found)")
    unfinished = [r for r in run_rows if not r["finished"]]
    if unfinished:
        print(f"\n  {len(unfinished)} run(s) still in progress (partial topk_stats.csv included): "
              + ", ".join(f"{r['dataset']}/{r['mode']}/{r['gnn_type']}" for r in unfinished[:8])
              + (" ..." if len(unfinished) > 8 else ""))
    print()
    for path in paths:
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

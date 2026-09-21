#!/usr/bin/env python3
"""Collect the adaptive-top-k runs into CSVs for merging with the baseline.

Walks ``<exp_root>/exp/<dataset>/ablation_adaptive_jenks/<combo>/`` for every
run the deduped manifest expects, and writes:

  hpc/results.csv
      One row per adaptive run. Column layout matches
      ablation_runner.summary_to_csv() -- the same flattening the baseline
      CSV uses (config columns, then <metric>_mean / <metric>_std per metric)
      -- so the two concatenate directly. Adaptive-only columns are appended
      AFTER those, where they simply come out empty for baseline rows.

  hpc/topk_curves.csv
      Every run's topk_stats.csv concatenated, with dataset/exp_name columns
      prepended. Full step resolution, for the budget-vs-t figure.

  hpc/topk_budget_summary.csv
      Mean adaptive budget per (dataset, block, t_bucket), weighted by
      n_samples -- which is why that column exists. Also printed.

Missing and failed runs are reported, never silently skipped.

Stdlib only: no dependency beyond what the repo already needs.

    python3 hpc/collect_results.py --repo_root /work/pr019/atvoulo/Rompos/thesis_project
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

ADAPTIVE_SUBDIR = "ablation_adaptive_jenks"
RESULTS_FILE = "results_full_averaged.json"
TOPK_STATS_FILE = "topk_stats.csv"
RUN_META_FILE = "run_meta.json"

# Exactly ablation_runner.summary_to_csv()'s config columns, in its order.
CONFIG_COLS = ["dataset", "exp_name", "gnn_type", "n_layers", "d_model",
               "top_k", "n_heads", "use_attention", "steps",
               "n_gen_seeds", "n_clf_seeds", "status"]

# Appended after the metric columns; empty for baseline rows on a merge.
ADAPTIVE_COLS = ["topk_mode", "baseline_top_ks", "k_min", "topk_warmup_steps",
                 "k_mean_train", "k_mean_sample", "gpu_name", "git_commit"]


def _load_helpers(repo_root: str):
    for sub in ("hpc", "ablation"):
        path = os.path.join(repo_root, sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import adaptive_manifest  # noqa: E402
    return adaptive_manifest


def _protocol_constants(repo_root: str):
    """steps / seed counts actually used, from the module the runs imported.

    The job scripts deliberately never pass --steps/--n_gen_seeds/
    --n_clf_seeds, so these constants are what every run was trained and
    evaluated with.
    """
    ablation_dir = os.path.join(repo_root, "ablation")
    if ablation_dir not in sys.path:
        sys.path.insert(0, ablation_dir)
    try:
        import ablation_runner
        return (ablation_runner.ABLATION_STEPS,
                ablation_runner.N_GEN_SEEDS,
                ablation_runner.N_CLF_SEEDS)
    except Exception as exc:  # pragma: no cover
        print(f"  [WARN] could not import ablation_runner for the protocol "
              f"constants ({exc}); leaving those columns empty")
        return ("", "", "")


def _read_topk_stats(path: str):
    """Return (rows, mean_k_by_phase). rows keep full resolution."""
    rows = []
    weighted = defaultdict(lambda: [0.0, 0])  # phase -> [sum(k*n), sum(n)]
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    k_mean = float(row["k_mean"])
                    n_samples = int(row["n_samples"])
                    row["_k_mean"] = k_mean
                    row["_n_samples"] = n_samples
                except (KeyError, ValueError, TypeError):
                    continue
                rows.append(row)
                acc = weighted[row.get("phase", "")]
                acc[0] += k_mean * n_samples
                acc[1] += n_samples
    except OSError:
        return [], {}
    means = {phase: (tot / n if n else None) for phase, (tot, n) in weighted.items()}
    return rows, means


def _failed_runs(repo_root: str) -> dict:
    """exp_name -> last failure line, from hpc/logs/<dataset>_failures.txt."""
    failures = {}
    logs_dir = os.path.join(repo_root, "hpc", "logs")
    if not os.path.isdir(logs_dir):
        return failures
    for name in sorted(os.listdir(logs_dir)):
        if not name.endswith("_failures.txt"):
            continue
        with open(os.path.join(logs_dir, name)) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    failures[(parts[1], parts[2])] = line.rstrip("\n")
    return failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", required=True,
                   help="Repository root (contains exp/, ablation/, hpc/)")
    p.add_argument("--exp_root", default=None,
                   help="Root containing exp/ (default: --repo_root)")
    p.add_argument("--out_dir", default=None,
                   help="Where to write the CSVs (default: <repo_root>/hpc)")
    args = p.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    exp_root = os.path.abspath(args.exp_root or repo_root)
    out_dir = os.path.abspath(args.out_dir or os.path.join(repo_root, "hpc"))
    os.makedirs(out_dir, exist_ok=True)

    adaptive_manifest = _load_helpers(repo_root)
    manifest = adaptive_manifest.adaptive_manifest(repo_root)
    steps, n_gen_seeds, n_clf_seeds = _protocol_constants(repo_root)
    failures = _failed_runs(repo_root)

    result_rows = []
    metric_cols = []
    curve_rows = []
    budget = defaultdict(lambda: [0.0, 0])   # (dataset, block, t_bucket) -> [sum, n]
    missing, failed, found = [], [], 0

    for dataset in adaptive_manifest.datasets_cheapest_first(manifest):
        for run in manifest[dataset]:
            exp_name = adaptive_manifest.exp_dir_name(run)
            run_dir = os.path.join(exp_root, "exp", dataset,
                                   ADAPTIVE_SUBDIR, exp_name)
            results_path = os.path.join(run_dir, RESULTS_FILE)

            if (dataset, exp_name) in failures:
                failed.append((dataset, exp_name, failures[(dataset, exp_name)]))

            if not os.path.exists(results_path):
                missing.append((dataset, exp_name))
                continue
            found += 1

            with open(results_path) as fh:
                metrics = json.load(fh)

            meta = {}
            meta_path = os.path.join(run_dir, RUN_META_FILE)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as fh:
                        meta = json.load(fh)
                except (OSError, ValueError):
                    meta = {}

            stats_rows, phase_means = _read_topk_stats(
                os.path.join(run_dir, TOPK_STATS_FILE))
            for row in stats_rows:
                curve_rows.append({
                    "dataset": dataset,
                    "exp_name": exp_name,
                    "step": row.get("step", ""),
                    "block": row.get("block", ""),
                    "t_bucket": row.get("t_bucket", ""),
                    "k_mean": row["_k_mean"],
                    "n_samples": row["_n_samples"],
                    "phase": row.get("phase", ""),
                    "tag": row.get("tag", ""),
                })
                key = (dataset, row.get("block", ""), row.get("t_bucket", ""))
                budget[key][0] += row["_k_mean"] * row["_n_samples"]
                budget[key][1] += row["_n_samples"]

            csv_row = {
                "dataset": dataset,
                "exp_name": exp_name,
                "gnn_type": run["gnn_type"],
                "n_layers": run["n_layers"],
                "d_model": run["d_model"],
                "top_k": run["top_k"],
                "n_heads": run["n_heads"],
                "use_attention": run["use_attention"],
                "steps": steps,
                "n_gen_seeds": n_gen_seeds,
                "n_clf_seeds": n_clf_seeds,
                "status": "done",
                "topk_mode": meta.get("topk_mode", "adaptive_jenks"),
                "baseline_top_ks": ",".join(str(k) for k in run["baseline_top_ks"]),
                "k_min": meta.get("k_min", ""),
                "topk_warmup_steps": meta.get("topk_warmup_steps", ""),
                "k_mean_train": phase_means.get("train", ""),
                "k_mean_sample": phase_means.get("sample", ""),
                "gpu_name": meta.get("gpu_name", ""),
                "git_commit": meta.get("git_commit", ""),
            }
            for metric, value in metrics.items():
                if isinstance(value, dict) and "mean" in value:
                    pairs = [(f"{metric}_mean", value["mean"]),
                             (f"{metric}_std", value.get("std"))]
                else:
                    pairs = [(metric, value)]
                for col, val in pairs:
                    if col not in metric_cols:
                        metric_cols.append(col)
                    csv_row[col] = val
            result_rows.append(csv_row)

    # ---- hpc/results.csv ---------------------------------------------------
    results_csv = os.path.join(out_dir, "results.csv")
    with open(results_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CONFIG_COLS + metric_cols + ADAPTIVE_COLS)
        writer.writeheader()
        writer.writerows(result_rows)

    # ---- hpc/topk_curves.csv ----------------------------------------------
    curves_csv = os.path.join(out_dir, "topk_curves.csv")
    curve_cols = ["dataset", "exp_name", "step", "block", "t_bucket",
                  "k_mean", "n_samples", "phase", "tag"]
    with open(curves_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=curve_cols)
        writer.writeheader()
        writer.writerows(curve_rows)

    # ---- hpc/topk_budget_summary.csv --------------------------------------
    budget_csv = os.path.join(out_dir, "topk_budget_summary.csv")

    def _sort_key(key):
        dataset, block, bucket = key
        def _num(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return -1
        return (dataset, _num(block), _num(bucket))

    with open(budget_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "block", "t_bucket", "k_mean", "n_samples"])
        for key in sorted(budget, key=_sort_key):
            total, n = budget[key]
            writer.writerow([key[0], key[1], key[2],
                             (total / n) if n else "", n])

    # ---- report ------------------------------------------------------------
    expected = sum(len(runs) for runs in manifest.values())
    print("=" * 70)
    print("  Adaptive-top-k collection")
    print("=" * 70)
    print(f"  expected runs : {expected}")
    print(f"  found         : {found}")
    print(f"  missing       : {len(missing)}")
    print(f"  failed (log)  : {len(failed)}")
    print(f"  exp root      : {exp_root}")
    print()
    print(f"  -> {results_csv}   ({len(result_rows)} rows)")
    print(f"  -> {curves_csv}   ({len(curve_rows)} rows)")
    print(f"  -> {budget_csv}   ({len(budget)} rows)")

    if missing:
        print()
        print("  MISSING (no results_full_averaged.json):")
        for dataset, exp_name in missing:
            print(f"    {dataset:12s} {exp_name}")
    if failed:
        print()
        print("  FAILED (recorded in hpc/logs/):")
        for dataset, exp_name, line in failed:
            print(f"    {dataset:12s} {exp_name}  [{line}]")

    # ---- per-dataset metric summary (mean +/- std over seeds) --------------
    print()
    print("=" * 70)
    print("  Per-dataset summary (mean +/- std as written by the run)")
    print("=" * 70)
    headline = ["macro avg/f1-score", "r2", "score", "dcr_tabddpm_mean",
                "wasserstein_mean", "mia_auc"]
    by_dataset = defaultdict(list)
    for row in result_rows:
        by_dataset[row["dataset"]].append(row)

    for dataset in sorted(by_dataset):
        rows = by_dataset[dataset]
        print(f"\n  {dataset}  ({len(rows)} run(s))")
        for metric in headline:
            values = [r[f"{metric}_mean"] for r in rows
                      if isinstance(r.get(f"{metric}_mean"), (int, float))]
            if not values:
                continue
            mean = sum(values) / len(values)
            if len(values) > 1:
                var = sum((v - mean) ** 2 for v in values) / len(values)
                spread = f" +/- {math.sqrt(var):.4f}"
            else:
                spread = ""
            print(f"    {metric:22s} {mean:9.4f}{spread}   (over {len(values)} run(s))")

    # ---- adaptive budget ---------------------------------------------------
    print()
    print("=" * 70)
    print("  Mean adaptive budget per dataset / block / t_bucket")
    print("  (n_samples-weighted; t_bucket 0 = t near 0, 9 = t near T)")
    print("=" * 70)
    current = None
    for key in sorted(budget, key=_sort_key):
        dataset, block, bucket = key
        total, n = budget[key]
        if dataset != current:
            current = dataset
            print(f"\n  {dataset}")
            print(f"    {'block':>6s} {'t_bucket':>9s} {'k_mean':>9s} {'n':>12s}")
        print(f"    {block:>6s} {bucket:>9s} {total / n if n else 0:9.3f} {n:>12d}")

    if not budget:
        print("\n  (no topk_stats.csv rows found -- "
              "were the runs made with --topk_mode adaptive_jenks?)")

    return 0 if (not missing and not failed) else 1


if __name__ == "__main__":
    sys.exit(main())

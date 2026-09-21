#!/usr/bin/env python3
"""Deduped run list for the adaptive-top-k re-run.

The baseline in ``tabddpm_ablation/*.json`` swept a top_k axis. In
``--topk_mode adaptive_jenks`` the budget is chosen per node from the data, so
top_k is unused except during warm-up -- baseline combinations that differ only
in top_k collapse onto a single adaptive run.

This module therefore takes the exact combination list that
``ablation/existing_results.load_manifest()`` reads out of the baseline
summaries (never a re-derived cartesian product -- see that module's docstring
for why) and groups it on the five axes that still matter:

    (gnn_type, n_layers, d_model, n_heads, use_attention)

Each group yields ONE adaptive run. The representative top_k is ``min(top_k)``
over the group -- an explicit, order-independent rule, since JSON entry order
is not a documented invariant. Every collapsed top_k is carried alongside, so
run_meta.json can record which baseline rows the run is comparable against.

No new run-definition file is introduced: the baseline summaries remain the
single source of truth.

Usage (bash, one tab-separated line per adaptive run):
    python3 hpc/adaptive_manifest.py --repo_root /path/to/repo --dataset wilt
        -> gnn_type  n_layers  d_model  top_k  n_heads  use_attention  baseline_top_ks

    python3 hpc/adaptive_manifest.py --repo_root /path/to/repo --list_datasets
        -> one dataset name per line, cheapest (fewest runs) first

    python3 hpc/adaptive_manifest.py --repo_root /path/to/repo --summary
"""

from __future__ import annotations

import argparse
import os
import sys

# Datasets excluded from this re-run by design: the baseline for these two is
# not being revisited.
EXCLUDED_DATASETS = ("adult", "california")

# Grouping key: everything that still changes the model in adaptive mode.
DEDUP_FIELDS = ("gnn_type", "n_layers", "d_model", "n_heads", "use_attention")


def _load_existing_results(repo_root: str):
    ablation_dir = os.path.join(repo_root, "ablation")
    if ablation_dir not in sys.path:
        sys.path.insert(0, ablation_dir)
    import existing_results  # noqa: E402  (path set above)
    return existing_results


def adaptive_manifest(repo_root: str) -> dict:
    """Map dataset -> list of adaptive runs.

    Each run is a dict with the five dedup axes, the representative ``top_k``
    (min over the group, used only for warm-up) and ``baseline_top_ks``, the
    sorted list of baseline top_k values the run collapses.
    """
    existing_results = _load_existing_results(repo_root)
    results_dir = os.path.join(repo_root, existing_results.DEFAULT_RESULTS_DIR)
    baseline = existing_results.load_manifest(results_dir, status="done")

    manifest: dict = {}
    for dataset, combos in baseline.items():
        if dataset in EXCLUDED_DATASETS:
            continue

        groups: dict = {}
        order: list = []
        for gnn_type, n_layers, d_model, top_k, n_heads, use_attention in combos:
            key = (gnn_type, n_layers, d_model, n_heads, use_attention)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(int(top_k))

        runs = []
        for key in order:
            gnn_type, n_layers, d_model, n_heads, use_attention = key
            top_ks = sorted(set(groups[key]))
            runs.append({
                "gnn_type":        gnn_type,
                "n_layers":        int(n_layers),
                "d_model":         int(d_model),
                "n_heads":         int(n_heads),
                "use_attention":   bool(use_attention),
                "top_k":           min(top_ks),   # representative, warm-up only
                "baseline_top_ks": top_ks,
                "n_baseline_combos": len(groups[key]),
            })
        manifest[dataset] = runs

    return manifest


def exp_dir_name(run: dict) -> str:
    """Reproduce ablation_runner.exp_dir_name() for a run dict.

    Kept in sync with that function deliberately: collect_results.py needs to
    find the output directory without importing the training stack.
    """
    suffix = "" if run["use_attention"] else "_noattn"
    return (f"gnn_{run['gnn_type']}_layers{run['n_layers']}_dim{run['d_model']}"
            f"_topk{run['top_k']}_heads{run['n_heads']}{suffix}")


def datasets_cheapest_first(manifest: dict) -> list:
    """Dataset names ordered by run count, so a short job reports back first."""
    return sorted(manifest, key=lambda ds: (len(manifest[ds]), ds))


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", required=True,
                   help="Repository root (contains tabddpm_ablation/ and ablation/)")
    p.add_argument("--dataset", default=None,
                   help="Emit the tab-separated run list for this dataset")
    p.add_argument("--list_datasets", action="store_true",
                   help="Emit dataset names, cheapest (fewest runs) first")
    p.add_argument("--summary", action="store_true",
                   help="Human-readable summary of the dedup")
    args = p.parse_args()

    manifest = adaptive_manifest(args.repo_root)

    if args.list_datasets:
        for dataset in datasets_cheapest_first(manifest):
            print(dataset)
        return 0

    if args.summary:
        total_baseline = 0
        total_runs = 0
        for dataset in datasets_cheapest_first(manifest):
            runs = manifest[dataset]
            n_baseline = sum(r["n_baseline_combos"] for r in runs)
            total_baseline += n_baseline
            total_runs += len(runs)
            print(f"{dataset:12s} {n_baseline:3d} baseline combos -> "
                  f"{len(runs):3d} adaptive runs")
            for run in runs:
                print(f"    {exp_dir_name(run):48s} "
                      f"baseline_top_ks={run['baseline_top_ks']}")
        print(f"{'TOTAL':12s} {total_baseline:3d} baseline combos -> "
              f"{total_runs:3d} adaptive runs")
        return 0

    if args.dataset:
        if args.dataset not in manifest:
            print(f"error: no baseline results for dataset {args.dataset!r}; "
                  f"known: {sorted(manifest)}", file=sys.stderr)
            return 1
        for run in manifest[args.dataset]:
            print("\t".join([
                run["gnn_type"],
                str(run["n_layers"]),
                str(run["d_model"]),
                str(run["top_k"]),
                str(run["n_heads"]),
                "1" if run["use_attention"] else "0",
                ",".join(str(k) for k in run["baseline_top_ks"]),
            ]))
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(_main())

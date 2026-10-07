#!/usr/bin/env python3
"""
Best ablation configuration per ABLATED dataset, for the adaptive-Jenks runs.

The nine datasets in ``tabddpm_ablation/`` already have fixed-top-k baselines,
so their adaptive-Jenks run needs only ONE configuration each: the best one the
ablation found. "Best" here is the repository's own definition --
``ablation/existing_results.load_best()``, which ranks by ``metrics.score.mean``
(accuracy for classification, NEGATIVE RMSE for regression, so higher is always
better).

This is deliberately NOT what ``hpc/adaptive_manifest.py`` does. That module
replays every recorded combination (48 runs over 7 datasets) so each adaptive
run has a paired fixed baseline across the whole grid. Here the scope is one
run per dataset, 9 in total, covering all nine ablated datasets including the
two ``hpc/`` excludes by design (``EXCLUDED_DATASETS = ("adult", "california")``).

Nothing in ``hpc/`` is read differently or modified. Submit either study, not
both -- they would write the same directories.

Because the run keeps the best configuration's OWN ``top_k``, the output
directory name matches that configuration's fixed baseline exactly, so the
adaptive/fixed pair lines up by name in the tables.

    python3 hpc_transfer/best_manifest.py --summary
    python3 hpc_transfer/best_manifest.py --dataset abalone   # TSV
"""
from __future__ import annotations

import argparse
import os
import sys

# Datasets with finished ablation results. Derived, not hardcoded -- see
# load_best() below; this tuple only fixes the submission order (cheapest
# first, by training-set size).
SUBMIT_ORDER = ("diabetes", "insurance", "abalone", "wilt", "churn2",
                "gesture", "buddy", "california", "adult")


def _load_existing_results(repo_root: str):
    ablation_dir = os.path.join(repo_root, "ablation")
    if ablation_dir not in sys.path:
        sys.path.insert(0, ablation_dir)
    import existing_results  # noqa: E402  (path set above)
    return existing_results


def best_manifest(repo_root: str) -> dict:
    """Map dataset -> the single best combination, as a run dict.

    ``load_best`` returns a list per dataset (it keeps every combination tied on
    ``score.mean``). Ties are broken by taking the first, which is the order the
    summary file records; in this study no dataset actually ties.
    """
    existing_results = _load_existing_results(repo_root)
    results_dir = os.path.join(repo_root, existing_results.DEFAULT_RESULTS_DIR)
    best = existing_results.load_best(results_dir)

    manifest: dict = {}
    for dataset, combos in best.items():
        if not combos:
            continue
        gnn_type, n_layers, d_model, top_k, n_heads, use_attention = combos[0]
        manifest[dataset] = {
            "gnn_type":        gnn_type,
            "n_layers":        int(n_layers),
            "d_model":         int(d_model),
            "top_k":           int(top_k),
            "n_heads":         int(n_heads),
            "use_attention":   bool(use_attention),
            "n_tied":          len(combos),
        }
    return manifest


def exp_dir_name(run: dict) -> str:
    """Reproduce ablation_runner.exp_dir_name() without importing the stack."""
    suffix = "" if run["use_attention"] else "_noattn"
    return (f"gnn_{run['gnn_type']}_layers{run['n_layers']}_dim{run['d_model']}"
            f"_topk{run['top_k']}_heads{run['n_heads']}{suffix}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_root", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    manifest = best_manifest(args.repo_root)

    if args.summary or not args.dataset:
        order = [d for d in SUBMIT_ORDER if d in manifest]
        order += [d for d in sorted(manifest) if d not in SUBMIT_ORDER]
        for ds in order:
            r = manifest[ds]
            tie = "" if r["n_tied"] == 1 else f"   ({r['n_tied']} tied on score)"
            print(f"{ds:12s} {exp_dir_name(r)}{tie}")
        print(f"\n{len(manifest)} datasets, one adaptive_jenks run each")
        return 0

    if args.dataset not in manifest:
        print(f"error: no ablation results for '{args.dataset}'; "
              f"known: {sorted(manifest)}", file=sys.stderr)
        return 1

    r = manifest[args.dataset]
    print("\t".join([r["gnn_type"], str(r["n_layers"]), str(r["d_model"]),
                     str(r["top_k"]), str(r["n_heads"]),
                     "1" if r["use_attention"] else "0"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

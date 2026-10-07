#!/usr/bin/env python3
"""
Collect the transfer runs into the ablation's own table format.

Reads every
    <transfer_root>/exp/<dataset>/ablation/<combo>/results_full_averaged.json
and writes, into hpc_transfer/:

Every config in hpc_transfer/configs/ is collected, in the modes its
``RUN_MODES`` line declares:

  * the 7 NEW datasets -- the transferred configuration, in every top-k rule
    (``fixed`` is the baseline they never had, the adaptive rules the comparison);
  * the 9 ABLATED datasets -- their own ablation winner, in the adaptive rules
    only, because the fixed baseline already exists in tabddpm_ablation/.

A ``group`` column separates "transfer" from "ablated_best".

The fixed baseline is read from
``exp/<ds>/ablation/`` and the adaptive-Jenks run from
``exp/<ds>/ablation_adaptive_jenks/``, the sibling tree ablation_runner derives
from --topk_mode. A ``topk_mode`` column separates them, so the fixed-vs-Jenks
comparison hpc/ makes on the ablated datasets extends to these seven.

  * transfer_ablation.json  -- one entry per run, in exactly the schema of
    tabddpm_ablation/<ds>_ablation.json (dataset, exp_name, the five axes,
    use_attention, steps, n_gen_seeds, n_clf_seeds, status, metrics), so the
    new results can be dropped alongside the ablation summaries and read by
    ablation/existing_results.py without special-casing.
  * results.csv             -- the same rows flattened by
    ablation_runner.summary_to_csv(), the function the ablation's own CSV uses,
    so the column layout is identical by construction rather than by hand.
  * combined.csv            -- the ablation's 88 runs and these runs in one
    table, with a `study` column separating them.
  * run_meta.csv            -- provenance per run (strategy, neighbours,
    distance, confidence, GPU, git commit) from each run_meta.json.

Missing and failed runs are reported rather than silently skipped, and the
exit status is non-zero if either list is non-empty -- same contract as
hpc/collect_results.py.

Usage:
    python3 hpc_transfer/collect_results.py \
        --repo_root "$REPO_ROOT" --transfer_root "$HPC_ROOT/transfer_runs"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every top-k rule a run can use. "fixed" is the baseline -- the counterpart of
# tabddpm_ablation/ for a dataset that was never ablated. The three adaptive
# two adaptive rules are the comparison: Jenks (as hpc/ runs it) and
# entmax-1.5 (alpha 1.5, as hpc_entmax/ runs it). sparsemax (alpha 2.0) is
# deliberately not part of this study.
# Which of these a dataset is actually run in comes from its config's
# RUN_MODES line, not from this tuple.
MODES = ("fixed", "adaptive_jenks", "adaptive_entmax15")


def read_config(repo_root: str, dataset: str) -> dict:
    """The five axes for a dataset, parsed out of hpc_transfer/configs/<ds>.sh."""
    path = os.path.join(repo_root, "hpc_transfer", "configs", f"{dataset}.sh")
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.split("#")[0].strip().strip('"').strip("'")
            if key in ("GNN_TYPE", "N_LAYERS", "D_MODEL", "TOP_K", "N_HEADS",
                       "USE_ATTENTION", "RUN_MODES", "STRATEGY"):
                out[key] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_root", default=REPO_DEFAULT)
    ap.add_argument("--transfer_root", default=None,
                    help="default: $HPC_ROOT/transfer_runs, else "
                         "<repo_root>/../transfer_runs")
    ap.add_argument("--baseline_dir", default=None,
                    help="default: <repo_root>/tabddpm_ablation")
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    transfer_root = args.transfer_root or os.path.join(
        os.environ.get("HPC_ROOT", os.path.dirname(repo_root)), "transfer_runs")
    baseline_dir = args.baseline_dir or os.path.join(repo_root, "tabddpm_ablation")
    out_dir = os.path.join(repo_root, "hpc_transfer")

    sys.path.insert(0, os.path.join(repo_root, "ablation"))
    from ablation_runner import (ABLATION_STEPS, N_GEN_SEEDS, N_CLF_SEEDS,
                                 exp_dir_name, summary_to_csv)

    datasets = sorted(
        f[:-3] for f in os.listdir(os.path.join(out_dir, "configs"))
        if f.endswith(".sh"))

    summary, metas, missing, failed = [], [], [], []
    for ds in datasets:
        cfg = read_config(repo_root, ds)
        gnn, nl, dm = cfg["GNN_TYPE"], int(cfg["N_LAYERS"]), int(cfg["D_MODEL"])
        tk, nh = int(cfg["TOP_K"]), int(cfg["N_HEADS"])
        ua = bool(int(cfg["USE_ATTENTION"]))
        name = exp_dir_name(gnn, nl, dm, tk, nh, ua)

        modes = cfg.get("RUN_MODES", " ".join(MODES)).split()
        group = ("transfer" if cfg.get("STRATEGY") == "C_global"
                 else "ablated_best")
        for mode in modes:
            subdir = "ablation" if mode == "fixed" else f"ablation_{mode}"
            run_dir = os.path.join(transfer_root, "exp", ds, subdir, name)
            results_file = os.path.join(run_dir, "results_full_averaged.json")

            entry = {"dataset": ds, "exp_name": name, "gnn_type": gnn,
                     "n_layers": nl, "d_model": dm, "top_k": tk, "n_heads": nh,
                     "use_attention": ua, "steps": ABLATION_STEPS,
                     "n_gen_seeds": N_GEN_SEEDS, "n_clf_seeds": N_CLF_SEEDS,
                     "topk_mode": mode, "group": group}
            if os.path.exists(results_file):
                entry["status"] = "done"
                with open(results_file) as fh:
                    entry["metrics"] = json.load(fh)
            else:
                entry["status"] = "missing"
                missing.append(f"{ds}/{subdir}/{name}")
            summary.append(entry)

            meta_file = os.path.join(run_dir, "run_meta.json")
            if os.path.exists(meta_file):
                metas.append(json.load(open(meta_file)))

    for path in glob.glob(os.path.join(out_dir, "logs", "*_failures.txt")):
        for line in open(path):
            if line.strip():
                failed.append(line.rstrip("\n"))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "transfer_ablation.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    summary_to_csv(summary, os.path.join(out_dir, "results.csv"))

    if metas:
        import csv
        cols = sorted({k for m in metas for k in m})
        with open(os.path.join(out_dir, "run_meta.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(metas)

    # --- one table with the ablation ---------------------------------------
    combined = []
    for path in sorted(glob.glob(os.path.join(baseline_dir, "*.json"))):
        for rec in json.load(open(path)):
            rec = dict(rec)
            rec["study"] = "ablation"
            combined.append(rec)
    for rec in summary:
        rec = dict(rec)
        rec["study"] = f"{rec['group']}_{rec['topk_mode']}"
        combined.append(rec)
    # The ablation rows are all fixed-mode; label them so the column is total.
    for rec in combined:
        rec.setdefault("topk_mode", "fixed")
        rec.setdefault("group", "ablation")
    combined_csv = os.path.join(out_dir, "combined.csv")
    summary_to_csv(combined, combined_csv)
    # summary_to_csv does not know about `study`; re-add it as the first column
    import csv as _csv
    with open(combined_csv) as fh:
        rows = list(_csv.DictReader(fh))
    for row, rec in zip(rows, combined):
        row["study"] = rec["study"]
        row["topk_mode"] = rec["topk_mode"]
        row["group"] = rec["group"]
    with open(combined_csv, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["study", "group", "topk_mode"]
                            + [c for c in rows[0]
                               if c not in ("study", "group", "topk_mode")])
        w.writeheader()
        w.writerows(rows)

    n_done = sum(1 for e in summary if e["status"] == "done")
    print(f"\n  runs: done={n_done}/{len(summary)}  missing={len(missing)}")
    for group in ("transfer", "ablated_best"):
        sub = [e for e in summary if e["group"] == group]
        for mode in MODES:
            ms = [e for e in sub if e["topk_mode"] == mode]
            if not ms:
                continue
            d = sum(1 for e in ms if e["status"] == "done")
            print(f"    {group:12s} {mode:16s} done={d}/{len(ms)}")
    print(f"  combined table: {len(combined)} rows "
          f"({len(combined) - len(summary)} ablation + {len(summary)} transfer)")
    if missing:
        print("\n  MISSING:")
        for m in missing:
            print(f"    {m}")
    if failed:
        print("\n  FAILED (from hpc_transfer/logs/):")
        for f in failed:
            print(f"    {f}")
    print(f"\n  wrote -> {out_dir}/transfer_ablation.json, results.csv, "
          f"combined.csv" + (", run_meta.csv" if metas else ""))
    return 1 if (missing or failed) else 0


if __name__ == "__main__":
    sys.exit(main())

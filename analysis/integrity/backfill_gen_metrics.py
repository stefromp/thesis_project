#!/usr/bin/env python3
"""Recompute wasserstein_mean and mia_auc for split-runner (catfix) runs.

hpc_transfer/split_runner.py called compute_wasserstein_mean(real_data_path,
gen_dir) and compute_membership_inference_auc(real_data_path, gen_dir), one
positional argument short of their (dataset_name, real_data_path, gen_dir)
signature. Both raised TypeError, were caught by the per-metric try/except and
only printed a [WARN]; per_seed_scores.gen<k>.csv therefore has utility and both
DCRs but no Wasserstein and no MIA for every *_catfix run.

Both functions are deterministic (Wasserstein has no randomness; MIA uses
seed=0) and read only the saved gen_seed_<k>/*.npy and data/<ds>. Recomputing
locally reproduces the stored values of non-split runs to 1e-10 (checked on
cardio/ablation, abalone/ablation_adaptive_jenks, wilt/ablation_fixed_seed0,
gen seeds 0 and 3). The results tree is not modified: output is
posthoc_gen_metrics.csv next to this file, in per_seed_scores format plus a
`run_dir` column; compare_topk.load_run_dir merges it for runs missing them.

    python3 analysis/integrity/backfill_gen_metrics.py [--check]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "transfer_runs" / "exp"
OUT = HERE / "posthoc_gen_metrics.csv"
SKIP = {"higgs-small", "miniboone"}
KEYS = ("wasserstein_mean", "mia_auc")


def _fns():
    sys.path.insert(0, str(REPO / "analysis" / "dependency_fidelity"))
    from cat_input_local_check import _setup  # stubs zero/rtdl/category_encoders
    _setup()
    from eval_tabddpm_protocol import compute_wasserstein_mean, compute_membership_inference_auc
    return {"wasserstein_mean": compute_wasserstein_mean, "mia_auc": compute_membership_inference_auc}


def stored(run: Path) -> pd.DataFrame:
    files = sorted(run.glob("per_seed_scores.csv")) + sorted(run.glob("per_seed_scores.gen*.csv"))
    if not files:
        return pd.DataFrame(columns=["gen_seed", "clf_seed", "metric", "value"])
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return raw[raw.clf_seed.isna()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="also recompute for runs that HAVE the metrics and report the max abs diff")
    args = ap.parse_args()
    fns = _fns()
    rows, checks = [], []
    for run in sorted(ROOT.glob("*/*/gnn_*")):
        ds = run.parts[-3]
        if ds in SKIP or not run.is_dir():
            continue
        have = stored(run)
        for g in sorted(run.glob("gen_seed_*")):
            seed = int(g.name.split("_")[-1])
            for key in KEYS:
                prev = have[(have.gen_seed == seed) & (have.metric == key)]
                if len(prev) and not args.check:
                    continue
                v = float(fns[key](ds, str(REPO / "data" / ds), str(g)))
                if len(prev):
                    checks.append(abs(v - prev.value.iloc[-1]))
                else:
                    rows.append({"run_dir": str(run.relative_to(ROOT)), "gen_seed": seed,
                                 "clf_seed": None, "metric": key, "value": v})
                    print(f"{run.relative_to(ROOT)} gen{seed} {key}={v:.6f}", flush=True)
    pd.DataFrame(rows, columns=["run_dir", "gen_seed", "clf_seed", "metric", "value"]).to_csv(OUT, index=False)
    print(f"wrote {len(rows)} rows -> {OUT.relative_to(REPO)}")
    if checks:
        print(f"check: {len(checks)} stored values reproduced, max |diff| = {max(checks):.3g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Real-vs-real self-check of the export path + TabDiff's evaluator. Run with the
EVAL venv's python (and the synthcity venv's for --quality_python).

    python selfcheck.py --target tabdiff_shoppers --work /tmp/tdcheck [--n_runs 2]

1. Round trip: the REAL train split, laid out exactly like a sampler output
   (X_num_train.npy / X_cat_train.npy / y_train.npy) and written through
   export.export_seed, must parse back equal to TabDiff's real.csv (same
   columns, dtypes, categories; numbers equal to float32 precision).
2. n_runs "synthetic" tables = bootstrap resamples of the real train split
   (seed i, n_train rows like TabDiff's tables -- synthcity's alpha/beta needs
   equal sizes), written through the same export path into
   <work>/exp_tabdiff/<target>/selfcheck/real_boot/samples/, then tabdiff_eval.py main (or dcr for a *_dcr
   target) on them. Expected: Shape/Trend errors ~0-2 %, C2ST ~1, MLE close to
   the "Real" row of TabDiff's Table 3, DCR ~100 % (the rows ARE train rows).
3. --quality_python: tabdiff_eval.py quality with that interpreter.

--cpu_xgb_patch: ONLY for machines whose xgboost has no CUDA build (macOS):
sets tree_method='hist' in eval/mle/mle.py's model grid IN MEMORY for this
check. Never used by the study's jobs; recorded in the output.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study as S  # noqa: E402
import export as E  # noqa: E402
import tabdiff_eval as TE  # noqa: E402


def _fake_gen(data_dir: str, gen_dir: str, rows=None) -> None:
    os.makedirs(gen_dir, exist_ok=True)
    for name in ("X_num_train.npy", "X_cat_train.npy", "y_train.npy"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            a = np.load(p, allow_pickle=True)
            np.save(os.path.join(gen_dir, name), a if rows is None else a[rows])


def round_trip(target: str, work: str, repo: str) -> None:
    data = S.data_dir(target, repo)
    gen = os.path.join(work, "roundtrip", target)
    _fake_gen(data, gen)
    out = E.export_seed(gen, data, os.path.join(gen, "export.csv"))
    info = E.load_info(data)
    real = pd.read_csv(os.path.join(data, info["tabdiff"]["real"]))
    syn = pd.read_csv(out)
    assert list(real.columns) == list(syn.columns)
    cols = info["column_names"]
    numeric = {cols[i] for i in info["num_col_idx"]}
    if info["task_type"] == "regression":
        numeric.add(cols[info["target_col_idx"][0]])
    for c in real.columns:
        if c in numeric:
            # int vs float is allowed: TabDiff's real.csv keeps an integer
            # regression target (news ' shares') as int, and TabDiff's own
            # sampler writes it as float too (trainer.split_num_cat_target).
            assert syn[c].dtype.kind in "fi" and real[c].dtype.kind in "fi", (c, syn[c].dtype)
        else:
            assert real[c].dtype == syn[c].dtype, (c, real[c].dtype, syn[c].dtype)
        if c in numeric:
            np.testing.assert_allclose(syn[c].to_numpy(), real[c].to_numpy(), rtol=1e-6, atol=1e-6)
        else:
            assert (syn[c].to_numpy() == real[c].to_numpy()).all(), c
    print(f"  [round trip] {target}: real train -> export -> read_csv == TabDiff real.csv "
          f"({len(real)} rows, {len(real.columns)} columns)")


def _patch_cpu_xgb(tabdiff_root: str) -> None:
    sys.path.insert(0, tabdiff_root)
    import eval.mle.mle as mle
    for specs in mle._MODELS.values():
        for spec in specs:
            if spec.get("kwargs", {}).get("tree_method") == ["gpu_hist"]:
                spec["kwargs"]["tree_method"] = ["hist"]
    print("  [cpu_xgb_patch] tree_method gpu_hist -> hist IN MEMORY (local check only)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="tabdiff_shoppers")
    ap.add_argument("--work", required=True)
    ap.add_argument("--repo_root", default=S.REPO)
    ap.add_argument("--n_runs", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cpu_xgb_patch", action="store_true")
    ap.add_argument("--no_pin_check", action="store_true")
    ap.add_argument("--quality_python", default=None)
    args = ap.parse_args(argv)

    repo = os.path.abspath(args.repo_root)
    _, dcr = S.parse_target(args.target)
    round_trip(args.target, args.work, repo)

    data = S.data_dir(args.target, repo)
    n = E.load_info(data)["train_size"]
    run = os.path.join(S.results_root(args.work), args.target, "selfcheck", "real_boot")
    for i in range(args.n_runs):
        rows = np.random.default_rng(i).choice(n, n, replace=True)
        gen = os.path.join(run, f"gen_seed_{i}")
        _fake_gen(data, gen, rows)
        E.export_seed(gen, data, os.path.join(run, "samples", f"samples_{i}.csv"))
    print(f"  [export] {args.n_runs} bootstrap tables -> {run}/samples/")

    if args.cpu_xgb_patch and not dcr:
        _patch_cpu_xgb(os.path.join(repo, S.TABDIFF_REL))
    out = TE.run_report("dcr" if dcr else "main", run, args.target, repo, args.device,
                        args.n_runs, check_pins=not args.no_pin_check)
    avg = pd.read_csv(os.path.join(out, "avg_std.csv"), index_col=0)
    print(avg.to_string())
    if args.quality_python and not dcr:
        cmd = [args.quality_python, os.path.join(HERE, "tabdiff_eval.py"), "quality", "--run_dir", run,
               "--target", args.target, "--repo_root", repo, "--n_runs", str(args.n_runs)]
        if args.no_pin_check:
            cmd.append("--no_pin_check")
        subprocess.run(cmd, check=True)
        print(pd.read_csv(os.path.join(out, "avg_std.csv"), index_col=0).to_string())
    with open(os.path.join(out, "selfcheck.json"), "w") as fh:
        json.dump({"target": args.target, "n_runs": args.n_runs, "cpu_xgb_patch": args.cpu_xgb_patch,
                   "avg_std": pd.read_csv(os.path.join(out, "avg_std.csv"), index_col=0).to_dict()},
                  fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

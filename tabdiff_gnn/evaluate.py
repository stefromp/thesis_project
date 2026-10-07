#!/usr/bin/env python3
"""
Route-A evaluation through the TabDiff study's evaluation path
(hpc_transfer/tabdiff_study/tabdiff_eval.py: report_tables / quality_tables
-- TabDiff's own TabMetrics + Trainer.report_test + eval/eval_quality.py),
with TabDiff's digests (study pins + Route-A pins) and the evaluator package
pins checked first.

    # eval venv
    python tabdiff_gnn/evaluate.py main    --run_dir RUN --target tabdiff_adult [--device cuda:0]
    # synthcity venv
    python tabdiff_gnn/evaluate.py quality --run_dir RUN --target tabdiff_adult

Inputs come from data/tabdiff_gnn/<target>/ (real.csv = TabDiff's train
split, test.csv, and val.csv only for diabetes, where TabDiff's MLE uses it).
Tables: RUN/samples/samples_<k>.csv. Outputs: RUN/tabdiff_eval/
(all_results.csv, avg_std.csv, mle2.csv, quality.csv, eval_meta.json). No DCR in this study.
Only standard-library imports at module level (the quality step runs in the
synthcity venv).
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402
import tabdiff_eval as TE  # noqa: E402  (on sys.path through config)

EVAL_DIR = "tabdiff_eval"


def sample_paths(run_dir: str, n_runs: int) -> list:
    return TE._sample_paths(run_dir, n_runs)


def run(step: str, run_dir: str, target: str, device: str = "cpu", n_runs: int = C.N_TABLES,
        check_pins: bool = True, repo: str = C.REPO) -> str:
    C.parse_target(target)
    if step not in ("main", "quality"):
        raise ValueError(f"step {step!r}: Route A evaluates 'main' and 'quality' only (no DCR)")
    td = C.verify_tabdiff(repo)
    files = C.data_files(target, repo)
    inp = {k: files[k] for k in ("info", "real", "test", "val")}
    out_dir = os.path.join(run_dir, EVAL_DIR)
    tabdiff_root = os.path.join(repo, C.TABDIFF_REL)
    paths = sample_paths(run_dir, n_runs)
    if step == "quality":
        versions = TE._versions(["synthcity", "pandas", "numpy", "sklearn", "torch"])
        if check_pins:
            TE._check_pins(versions, C.S.QUALITY_PINS)
        return TE.quality_tables(out_dir, C.tabdiff_dataname(target), inp, tabdiff_root, n_runs,
                                 {"tabdiff": td, "versions": versions, "study": C.STUDY})
    versions = TE._versions(["sdmetrics", "xgboost", "pandas", "numpy", "sklearn", "torch"])
    if check_pins:
        TE._check_pins(versions, C.S.EVAL_PINS)
    return TE.report_tables(step, out_dir, inp, paths, tabdiff_root, device, n_runs,
                            {"tabdiff": td, "versions": versions, "study": C.STUDY,
                             "tabdiff_dataname": C.tabdiff_dataname(target)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["main", "quality"])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--exp_root", default=None, help="if given, --run_dir must lie in its Route-A tree")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_runs", type=int, default=C.N_TABLES)
    ap.add_argument("--no_pin_check", action="store_true", help="local smoke tests only")
    a = ap.parse_args(argv)
    if a.exp_root:
        C.assert_in_tree(a.run_dir, a.exp_root)
    out = run(a.step, a.run_dir, a.target, a.device, a.n_runs, not a.no_pin_check)
    print(f"  [OK] {a.step} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

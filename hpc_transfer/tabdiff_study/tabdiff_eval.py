#!/usr/bin/env python3
"""
Thin wrapper around TabDiff's OWN evaluation code (third_party/TabDiff, pinned
commit, digests verified before anything runs). No metric is implemented here.
runner.py calls it as stages of the dataset job; it also runs on its own to
re-evaluate existing tables (--exp_root + --target, or --run_dir).

    # eval venv (sdmetrics 0.18.0, xgboost 2.1.4):
    python tabdiff_eval.py main    --run_dir RUN --target tabdiff_adult     [--device cpu]
    python tabdiff_eval.py dcr     --run_dir RUN --target tabdiff_adult_dcr  --device cuda:0
    # synthcity venv (synthcity 0.2.12):
    python tabdiff_eval.py quality --run_dir RUN --target tabdiff_adult

main     TabMetrics(real.csv, test.csv, val=None, info, device,
         ["density", "mle", "c2st"]) on each of the 20 exported tables, fed
         through TabDiff's Trainer.report_test, which averages and writes
         all_results.csv / avg_std.csv / all_samples/samples_<i>.csv exactly
         as in their `main.py --mode test --report`.
dcr      Same with ["dcr"] on a *_dcr target, through Trainer.report_test_dcr
         (all_results.csv / avg_std.csv / dcr.csv), as `--dataname <ds>_dcr`.
quality  TabDiff's eval/eval_quality.py run AS A SCRIPT (unchanged) in a small
         directory that mirrors the layout it expects (data/<name>/info.json,
         synthetic/<name>/real.csv, eval/report_runs/<exp>/<name>/all_samples),
         symlinked to our files. It appends alpha / beta to avg_std.csv and
         writes quality.csv.

Adaptations (all here, none in TabDiff's files):
  1. paths: real/test/info from data/<target>/tabdiff/, samples from
     <run>/samples/, outputs to <run>/tabdiff_eval/ (their result_save_path);
  2. device: --device (cpu for main/quality, the job's GPU for dcr);
  3. a stub object stands in for Trainer.self: evaluate_generation() reads our
     table i with pd.read_csv (as theirs re-reads samples.csv) instead of
     sampling;
  4. np.random.seed(i) before table i: their MLE val split uses the unseeded
     global RNG -- seeding fixes the state, not the procedure;
  5. package pins (study.EVAL_PINS / QUALITY_PINS).
Only standard-library imports at module level: the quality step runs in the
synthcity venv, the others in the eval venv.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study as S  # noqa: E402  (stdlib-only module)

EXP_NAME = "tabdiff_transfer"     # their --exp_name, used only for the quality dir layout


_DIST = {"sklearn": "scikit-learn"}     # import name -> distribution name


def _versions(mods):
    """Installed distribution versions (package metadata: synthcity has no
    __version__), plus whether the module imports at all."""
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for m in mods:
        try:
            importlib.import_module(m)
            out[m] = metadata.version(_DIST.get(m, m))
        except Exception as exc:          # recorded, not fatal here
            out[m] = f"unavailable ({type(exc).__name__})"
    return out


def _check_pins(versions: dict, pins: dict) -> None:
    bad = {k: (versions.get(k), v) for k, v in pins.items() if versions.get(k) != v}
    if bad:
        raise RuntimeError("evaluator packages differ from the study pins: "
                           + ", ".join(f"{k} {a} != {b}" for k, (a, b) in bad.items())
                           + " -- see hpc_transfer/tabdiff_study/README.md (install)")


def _sample_paths(run_dir: str, n_runs: int):
    paths = [os.path.join(run_dir, "samples", f"samples_{i}.csv") for i in range(n_runs)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {n_runs} tables missing, e.g. {missing[0]}")
    return paths


def mle2_from_extras(mle_extra: dict, task_type: str) -> dict:
    """MLE-1 / MLE-2 of ONE table from the per-table MLE details TabDiff's
    evaluate_mle already returns (extras['mle']; nothing is recomputed):
    binclass AUROC (= TabDiff's 'mle') and binary F1 of the F1-selected
    XGBClassifier; regression RMSE (= 'mle') and R2 of the R2-selected
    XGBRegressor -- the MLE-1/MLE-2 pair GATD reports."""
    if task_type == "regression":
        return {"rmse": float(mle_extra["best_rmse_scores"]["XGBRegressor"]["RMSE"]),
                "r2": float(mle_extra["best_r2_scores"]["XGBRegressor"]["r2"])}
    return {"auroc": float(mle_extra["best_auroc_scores"]["XGBClassifier"]["roc_auc"]),
            "f1": float(mle_extra["best_f1_scores"]["XGBClassifier"]["binary_f1"])}


class _Stub:
    """Stands in for TabDiff's Trainer in report_test / report_test_dcr: same
    attribute and method names, evaluating our tables instead of sampling."""

    def __init__(self, metrics, paths, result_save_path):
        self.metrics = metrics
        self.paths = paths
        self.result_save_path = result_save_path
        self.i = 0
        self.mle2 = []                              # per-table MLE-1/MLE-2 (main step)

    def evaluate_generation(self, save_metric_details=False, plot_density=False, ema=False):
        import numpy as np
        import pandas as pd
        path = self.paths[self.i]
        np.random.seed(self.i)                      # adaptation 4
        syn_df_loaded = pd.read_csv(path)
        out_metrics, extras = self.metrics.evaluate(syn_df_loaded)
        if "mle" in extras:
            self.mle2.append(mle2_from_extras(extras["mle"], self.metrics.info["task_type"]))
        self.i += 1
        return out_metrics, extras, syn_df_loaded


def _write_meta(out_dir: str, step: str, extra: dict) -> None:
    p = os.path.join(out_dir, "eval_meta.json")
    meta = json.load(open(p)) if os.path.exists(p) else {}
    meta[step] = {"finished_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  **extra}
    with open(p, "w") as fh:
        json.dump(meta, fh, indent=2)


def run_report(step: str, run_dir: str, target: str, repo: str, device: str, n_runs: int,
               check_pins: bool = True) -> str:
    tabdiff_root = os.path.join(repo, S.TABDIFF_REL)
    td = S.verify_tabdiff(tabdiff_root)
    versions = _versions(["sdmetrics", "xgboost", "pandas", "numpy", "sklearn", "torch"])
    if check_pins:
        _check_pins(versions, S.EVAL_PINS)
    _, dcr = S.parse_target(target)
    if (step == "dcr") != dcr:
        raise ValueError(f"step {step!r} does not fit target {target} "
                         f"(dcr runs only on *_dcr targets, main only on the others)")
    inp = dict(S.eval_inputs(target, repo))
    # val=None: these datasets have no val.csv in TabDiff either, so their MLE
    # carves a validation set out of the synthetic table.
    inp["val"] = None
    out_dir = os.path.join(run_dir, "tabdiff_eval")
    return report_tables(step, out_dir, inp, _sample_paths(run_dir, n_runs), tabdiff_root, device,
                         n_runs, {"tabdiff": td, "versions": versions,
                                  "tabdiff_dataname": S.tabdiff_dataname(target)})


def report_tables(step: str, out_dir: str, inp: dict, paths: list, tabdiff_root: str, device: str,
                  n_runs: int, meta: dict) -> str:
    """TabDiff's Trainer.report_test (step 'main') / report_test_dcr ('dcr') on
    the given tables. inp = {info, real, test, val (None unless TabDiff has a
    val.csv for the dataset)}. The caller has verified the TabDiff digests and
    the package pins. Also writes mle2.csv (per-table MLE-1/MLE-2, read from the
    same MLE call)."""
    import csv
    with open(inp["info"]) as fh:
        info = json.load(fh)
    os.makedirs(out_dir, exist_ok=True)
    if tabdiff_root not in sys.path:
        sys.path.insert(0, tabdiff_root)             # their packages first
    from tabdiff.metrics import TabMetrics
    from tabdiff.trainer import Trainer
    metric_list = ["dcr"] if step == "dcr" else ["density", "mle", "c2st"]
    metrics = TabMetrics(inp["real"], inp["test"], inp.get("val"), info, device, metric_list)
    stub = _Stub(metrics, paths, out_dir)
    if step == "dcr":
        Trainer.report_test_dcr(stub, n_runs)
    else:
        Trainer.report_test(stub, n_runs)
        if stub.mle2:
            with open(os.path.join(out_dir, "mle2.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["run"] + list(stub.mle2[0]))
                w.writeheader()
                for i, r in enumerate(stub.mle2):
                    w.writerow({"run": i, **r})
    _write_meta(out_dir, step, {**meta, "device": device, "metric_list": metric_list, "n_runs": n_runs,
                                "val_csv": inp.get("val"),
                                "adaptations": __doc__.split("Adaptations")[1].split("Only standard")[0].strip()})
    return out_dir


def run_quality(run_dir: str, target: str, repo: str, n_runs: int, check_pins: bool = True) -> str:
    tabdiff_root = os.path.join(repo, S.TABDIFF_REL)
    td = S.verify_tabdiff(tabdiff_root)
    versions = _versions(["synthcity", "pandas", "numpy", "sklearn", "torch"])
    if check_pins:
        _check_pins(versions, S.QUALITY_PINS)
    out_dir = os.path.join(run_dir, "tabdiff_eval")
    _sample_paths(run_dir, n_runs)                   # our tables are all there
    return quality_tables(out_dir, S.tabdiff_dataname(target), S.eval_inputs(target, repo),
                          tabdiff_root, n_runs, {"tabdiff": td, "versions": versions})


def quality_tables(out_dir: str, name: str, inp: dict, tabdiff_root: str, n_runs: int, meta: dict) -> str:
    """TabDiff's eval/eval_quality.py (alpha / beta), run as a script, on the
    tables report_tables() copied to <out_dir>/all_samples."""
    if not os.path.exists(os.path.join(out_dir, "avg_std.csv")):
        raise FileNotFoundError(f"{out_dir}/avg_std.csv missing: run the 'main' (or 'dcr') step first "
                                f"(eval_quality.py appends to it)")
    samples = os.path.join(out_dir, "all_samples")
    if len([f for f in os.listdir(samples) if f.endswith(".csv")]) != n_runs:
        raise RuntimeError(f"{samples} should hold exactly {n_runs} tables (written by report_test)")
    cwd = os.path.join(out_dir, "_tabdiff_cwd")
    shutil.rmtree(cwd, ignore_errors=True)
    for d in (f"data/{name}", f"synthetic/{name}", f"eval/report_runs/{EXP_NAME}"):
        os.makedirs(os.path.join(cwd, d), exist_ok=True)
    os.symlink(inp["info"], os.path.join(cwd, "data", name, "info.json"))
    os.symlink(inp["real"], os.path.join(cwd, "synthetic", name, "real.csv"))
    os.symlink(out_dir, os.path.join(cwd, "eval", "report_runs", EXP_NAME, name))
    cmd = [sys.executable, os.path.join(tabdiff_root, "eval", "eval_quality.py"),
           "--dataname", name, "--exp_name", EXP_NAME]
    print("  [quality] cwd=" + cwd + "\n  " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"eval_quality.py exited {r.returncode}")
    shutil.rmtree(cwd, ignore_errors=True)
    _write_meta(out_dir, "quality", {**meta, "n_runs": n_runs,
                                     "script": "eval/eval_quality.py (unchanged, run as a script)"})
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["main", "dcr", "quality"])
    ap.add_argument("--run_dir", default=None,
                    help="default: study.run_dir(--exp_root, --target)")
    ap.add_argument("--target", required=True)
    ap.add_argument("--repo_root", default=S.REPO)
    ap.add_argument("--exp_root", default=None,
                    help="if given, --run_dir must lie in <exp_root>/exp_tabdiff/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_runs", type=int, default=S.N_SAMPLE_RUNS)
    ap.add_argument("--no_pin_check", action="store_true",
                    help="local smoke tests only; recorded in eval_meta.json versions")
    args = ap.parse_args(argv)
    S.parse_target(args.target)
    if args.run_dir is None:
        if not args.exp_root:
            ap.error("give --run_dir, or --exp_root to derive it")
        args.run_dir = S.run_dir(args.exp_root, args.target, args.repo_root)
    if args.exp_root:
        S.assert_in_study_tree(args.run_dir, args.exp_root)
    if args.step == "quality":
        out = run_quality(args.run_dir, args.target, args.repo_root, args.n_runs, not args.no_pin_check)
    else:
        out = run_report(args.step, args.run_dir, args.target, args.repo_root, args.device,
                         args.n_runs, not args.no_pin_check)
    print(f"  [OK] {args.step} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

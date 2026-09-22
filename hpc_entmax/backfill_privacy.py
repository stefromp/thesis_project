#!/usr/bin/env python3
"""Recompute the privacy metrics that the Jenks runs lost to sklearn 1.1.3.

The ARIS venv pins scikit-learn 1.1.3, and scripts/eval_tabddpm_protocol.py
built OneHotEncoder(sparse_output=False), which only exists from 1.2. Every
metric that one-hot encodes a categorical column raised on every gen seed:

    wilt, gesture : dcr_baseline      (the class label is folded in as a category)
    abalone       : dcr_tabddpm_mean, dcr_baseline, mia_auc   (Sex is categorical)

ablation_runner caught the error, so results_full_averaged.json holds NaN for
those keys (or no key at all, for dcr_baseline).

Nothing needs retraining. The runner never deletes gen_seed_*/, and these
metrics are deterministic functions of those files (exact nearest neighbours,
MIA subsample seed 0). With the _encode fallback in place, this recomputes
exactly what the run would have written.

For every finished run of --topk_mode on --datasets:
  * a TabDDPM-protocol metric that is missing or NaN is recomputed over all
    gen seeds and aggregated exactly as ablation_runner does (mean, population
    std); finite metrics are never touched;
  * the untouched original is kept once as results_full_averaged.orig.json;
  * per-seed values and the environment go to privacy_backfill.json.

    python hpc_entmax/backfill_privacy.py --repo_root "$REPO_ROOT" \\
        --datasets wilt abalone gesture             # --topk_mode adaptive_jenks (default)
    python hpc_entmax/backfill_privacy.py --repo_root "$REPO_ROOT" \\
        --datasets wilt abalone gesture --check     # report only; exit 1 on any NaN

Runs without results_full_averaged.json (not finished yet) are listed and
skipped; they do not make --check fail.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import math
import os
import platform
import shutil
import sys

RESULTS_FILE = "results_full_averaged.json"
BACKUP_FILE = "results_full_averaged.orig.json"
SIDECAR_FILE = "privacy_backfill.json"

# ablation_runner's tabddpm_order: the keys it writes, in the order it writes them.
PROTOCOL_ORDER = ("dcr_tabddpm_mean", "dcr_baseline", "wasserstein_mean", "mia_auc")
MODES = ("adaptive_jenks", "adaptive_sparsemax", "adaptive_entmax15")


def _setup_paths(repo_root: str) -> None:
    for sub in ("", "scripts", "ablation", "hpc"):
        path = os.path.join(repo_root, sub) if sub else repo_root
        if path not in sys.path:
            sys.path.insert(0, path)


def _status(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if not isinstance(value, dict) or "mean" not in value:
        return "missing"
    mean = value["mean"]
    if not isinstance(mean, (int, float)) or not math.isfinite(mean):
        return "NaN"
    return "ok"


def _merged(original: dict, updates: dict) -> dict:
    """original with updates applied; a NEW key is placed where ablation_runner
    would have written it (protocol order), so the file reads like a normal run."""
    pending = [k for k in PROTOCOL_ORDER if k in updates and k not in original]
    out = {}
    for key, value in original.items():
        if key in PROTOCOL_ORDER:
            for new in list(pending):
                if PROTOCOL_ORDER.index(new) < PROTOCOL_ORDER.index(key):
                    out[new] = updates[new]
                    pending.remove(new)
        out[key] = updates.get(key, value)
    for new in pending:
        out[new] = updates[new]
    return out


def _dump(obj, path: str) -> None:
    # Same format as lib.dump_json (indent=4, trailing newline); atomic.
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(json.dumps(obj, indent=4) + "\n")
    os.replace(tmp, path)


def _gen_dirs(run_dir: str, n_expected: int):
    dirs = [os.path.join(run_dir, f"gen_seed_{i}") for i in range(n_expected)]
    missing = [d for d in dirs if not os.path.exists(os.path.join(d, "y_train.npy"))]
    return dirs, missing


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--topk_mode", default="adaptive_jenks", choices=MODES)
    p.add_argument("--check", action="store_true",
                   help="Only report metric status; exit 1 if a finished run has a NaN/missing metric")
    p.add_argument("--dry_run", action="store_true",
                   help="Recompute and print, but write nothing")
    args = p.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    _setup_paths(repo_root)
    import adaptive_manifest  # hpc/, read-only
    manifest = adaptive_manifest.adaptive_manifest(repo_root)

    fns = None
    if not args.check:
        import numpy as np
        import sklearn
        import ablation_runner
        from eval_tabddpm_protocol import (compute_dcr_baseline,
                                           compute_dcr_tabddpm_mean,
                                           compute_membership_inference_auc,
                                           compute_wasserstein_mean)
        fns = {"dcr_tabddpm_mean": compute_dcr_tabddpm_mean,
               "dcr_baseline": compute_dcr_baseline,
               "wasserstein_mean": compute_wasserstein_mean,
               "mia_auc": compute_membership_inference_auc}
        n_gen_seeds = ablation_runner.N_GEN_SEEDS
        print(f"sklearn {sklearn.__version__}   numpy {np.__version__}   "
              f"mode {args.topk_mode}   gen seeds {n_gen_seeds}"
              + ("   DRY RUN" if args.dry_run else ""))

    bad_after, unfinished, errors, changed, would_change = [], [], [], 0, 0
    for dataset in args.datasets:
        if dataset not in manifest:
            errors.append(f"{dataset}: not in hpc/adaptive_manifest.py")
            continue
        real_data_path = os.path.join(
            repo_root, importlib.import_module(f"{dataset}_ablation").DATASET_CFG["real_data_path"])
        print(f"\n{dataset}")
        for run in manifest[dataset]:
            exp_name = adaptive_manifest.exp_dir_name(run)
            run_dir = os.path.join(repo_root, "exp", dataset, f"ablation_{args.topk_mode}", exp_name)
            results_path = os.path.join(run_dir, RESULTS_FILE)
            if not os.path.exists(results_path):
                unfinished.append((dataset, exp_name))
                print(f"  {exp_name:46s} not finished (no {RESULTS_FILE})")
                continue
            with open(results_path) as fh:
                metrics = json.load(fh)
            status = {k: _status(metrics, k) for k in PROTOCOL_ORDER}
            todo = [k for k, s in status.items() if s != "ok"]
            label = "  ".join(f"{k}={s}" for k, s in status.items())

            if args.check or not todo:
                print(f"  {exp_name:46s} {label}")
                if todo:
                    bad_after.append((dataset, exp_name, todo))
                continue

            dirs, missing = _gen_dirs(run_dir, n_gen_seeds)
            if missing:
                errors.append(f"{dataset}/{exp_name}: gen dirs incomplete: {missing}")
                bad_after.append((dataset, exp_name, todo))
                print(f"  {exp_name:46s} SKIPPED: incomplete gen_seed dirs")
                continue

            updates, record, failed = {}, {}, False
            for key in todo:
                values = []
                for gen_dir in dirs:
                    try:
                        value = fns[key](dataset, real_data_path, gen_dir)
                    except Exception as exc:  # report, never write a partial result
                        errors.append(f"{dataset}/{exp_name} {key} {os.path.basename(gen_dir)}: "
                                      f"{type(exc).__name__}: {exc}")
                        failed = True
                        break
                    if value is not None:
                        values.append(float(value))
                if failed:
                    break
                if not values:          # e.g. Wasserstein on a dataset without numericals
                    continue
                arr = np.array(values)
                updates[key] = {"mean": float(arr.mean()), "std": float(arr.std())}
                record[key] = {"previous": status[key], "per_seed": values, **updates[key]}

            if failed:
                bad_after.append((dataset, exp_name, todo))
                print(f"  {exp_name:46s} FAILED (see errors below); nothing written")
                continue

            print(f"  {exp_name:46s} recomputed "
                  + "  ".join(f"{k}={v['mean']:.4f}+/-{v['std']:.4f} (was {status[k]})"
                              for k, v in updates.items()))
            if args.dry_run:
                would_change += 1
                continue

            backup = os.path.join(run_dir, BACKUP_FILE)
            if not os.path.exists(backup):
                shutil.copy2(results_path, backup)
            _dump(_merged(metrics, updates), results_path)

            sidecar_path = os.path.join(run_dir, SIDECAR_FILE)
            sidecar = {}
            if os.path.exists(sidecar_path):
                with open(sidecar_path) as fh:
                    sidecar = json.load(fh)
            sidecar.setdefault("recomputed", {}).update(record)
            sidecar.update({
                "reason": ("scikit-learn < 1.2 on ARIS: OneHotEncoder(sparse_output=...) "
                           "raised TypeError on every gen seed; recomputed from the saved "
                           "gen_seed_*/ files with the _encode fallback"),
                "original_backup": BACKUP_FILE,
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
                "script": "hpc_entmax/backfill_privacy.py",
                "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            _dump(sidecar, sidecar_path)
            changed += 1

    print()
    if args.dry_run:
        print(f"dry run        : {would_change} run(s) would be rewritten; nothing written")
    elif not args.check:
        print(f"runs rewritten : {changed}")
    print(f"not finished   : {len(unfinished)}"
          + (f"  ({', '.join(f'{d}/{e}' for d, e in unfinished)})" if unfinished else ""))
    if errors:
        print("errors:")
        for err in errors:
            print(f"  {err}")
    if bad_after:
        print(f"finished runs still NaN/missing: {len(bad_after)}")
        for dataset, exp_name, keys in bad_after:
            print(f"  {dataset}/{exp_name}: {', '.join(keys)}")
        return 1
    if not args.dry_run:
        print("all finished runs have finite " + ", ".join(PROTOCOL_ORDER))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

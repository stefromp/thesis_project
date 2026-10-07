#!/usr/bin/env python3
"""
Split a combination's generation seeds across SLURM jobs.

Why: with 5 generation seeds x 1000 diffusion steps, miniboone (~40-98 h) and
fb-comments (~77-188 h) cannot finish inside the 24 h walltime, and
ablation_runner passes ``checkpoint_every=0``, so a walltime kill loses the
whole run. Training is a single pass; only sampling + evaluation repeat per
generation seed, and each seed is independent.

WHY SPLITTING IS EXACTLY EQUIVALENT TO THE MONOLITHIC RUN
---------------------------------------------------------
``ablation_runner._run_one`` calls ``tabddpm_sample(..., seed=gen_seed)`` and
``scripts/sample.py`` opens with ``zero.improve_reproducibility(seed)``, which
re-seeds ``random``, ``numpy``, ``torch`` and ``torch.cuda`` from that value
alone. Generation seed k therefore depends only on k and on the saved weights
-- never on RNG state left behind by seeds 0..k-1. The same holds for the
classifier loop, which re-seeds from ``clf_seed``. So running seed k in its own
process, against the same ``model_ema.pt``, reproduces the monolithic run's
seed k. This module never re-derives a seed: it passes the same integer to the
same function.

HOW THIS MODULE AVOIDS DUPLICATING ablation_runner
--------------------------------------------------
``_run_one`` builds ``model_params`` / ``T_dict_train`` / ``graph_params``
inline, so they cannot be imported. Instead of copying them, this module
*captures* them: it calls the real ``_run_one`` with ``train.train`` patched to
record its keyword arguments and abort before any work happens. Every value
then comes from the original code path and tracks changes to it automatically.

Everything that computes anything is imported from ``ablation_runner`` and
called unchanged: ``tabddpm_train``, ``tabddpm_sample``, ``train_eval_model``,
``_load_raw_real`` / ``_load_raw_syn``, ``_append_per_seed``, ``exp_dir_name``,
the ``eval_tabddpm_protocol`` metrics and every module constant. What this
module re-expresses is orchestration only: which seeds to run, whether to
train or load, and how the per-seed files are laid out and merged.

``T_dict_eval`` and the metric-wiring inside the generation loop are the two
places that must be mirrored, because they are inline in ``_run_one`` and are
not reachable through the capture. ``tests/test_split_runner_backcompat.py``
pins both with a source-digest guard that fails loudly if ``_run_one`` changes.

ablation/ablation_runner.py IS NOT MODIFIED. hpc/, hpc_entmax/, hpc_wide/ and
hpc_density/ do not import this module.

Usage
-----
    # train + generation seed 0
    python3 hpc_transfer/split_runner.py --dataset miniboone \\
        --gnn_type gatv2 --n_layers 3 --d_model 128 --top_k 5 --n_heads 8 \\
        --use_attention 0 --topk_mode adaptive_entmax15 \\
        --gen_seed_ids 0 \\
        --repo_root "$REPO_ROOT" --data_root "$REPO_ROOT" --exp_root "$ROOT"

    # generation seed 3, reusing that checkpoint
    python3 hpc_transfer/split_runner.py ... --gen_seed_ids 3 --load_checkpoint

    # merge the shards into results_full_averaged.json once all seeds are in
    python3 hpc_transfer/split_runner.py ... --merge_only
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

REPO_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHARD_GLOB = "per_seed_scores.gen*.csv"
SHARD_FMT = "per_seed_scores.gen{}.csv"
SEED_METRICS_FMT = "metrics_gen{}.json"


# ---------------------------------------------------------------------------
# ablation_runner access
# ---------------------------------------------------------------------------
def _load_runner(repo_root: str):
    """Import ablation_runner with the same environment it sets up for itself."""
    for sub in ("ablation", "scripts"):
        p = os.path.join(repo_root, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import ablation_runner as ar
    ar._setup_zero_rtdl_stubs()
    ar._setup_repo_path(repo_root)
    return ar


class _Captured(Exception):
    """Raised by the patched trainer to stop _run_one once it has built its
    parameters. Carries those parameters."""

    def __init__(self, kwargs: dict):
        super().__init__("captured")
        self.kwargs = kwargs


def capture_context(ar, run_kwargs: dict) -> dict:
    """Run the real ``_run_one`` far enough to build its parameters, then stop.

    Returns the exact keyword arguments ``_run_one`` would have passed to
    ``train.train`` -- which includes ``parent_dir``, ``real_data_path``,
    ``model_params``, ``T_dict``, ``graph_params`` and every training
    hyperparameter. Nothing is trained, sampled or evaluated.
    """
    import train as train_mod

    real_train = train_mod.train
    real_remove = os.remove

    def _capturing_train(**kwargs):
        raise _Captured(kwargs)

    def _guarded_remove(path, *a, **kw):
        # _run_one deletes per_seed_scores.csv on entry so a partial file from
        # an interrupted attempt cannot mix two runs' seeds. During capture
        # nothing is being run, and a shard job must not wipe the consolidated
        # file, so the delete is suppressed here only.
        if os.path.basename(path) == ar.PER_SEED_FILE:
            return None
        return real_remove(path, *a, **kw)

    train_mod.train = _capturing_train
    os.remove = _guarded_remove
    try:
        ar._run_one(**{**run_kwargs, "skip_if_done": False})
    except _Captured as exc:
        return exc.kwargs
    finally:
        train_mod.train = real_train
        os.remove = real_remove
    raise RuntimeError(
        "capture failed: ablation_runner._run_one returned without calling "
        "train.train. Its structure changed -- rerun "
        "tests/test_split_runner_backcompat.py and update split_runner.py.")


def parent_dir_for(ar, run_kwargs: dict) -> str:
    """Where a combination's results live, without importing the training stack.

    File layout only, so --scan and --merge_only work on a login node where
    `lib` (and its category_encoders -> statsmodels chain) may not import. The
    directory NAME comes from ablation_runner.exp_dir_name and the
    subdirectory from ablation_runner.results_subdir -- the same function
    _run_one calls, so nothing is mirrored (pinned by
    tests/..._backcompat.py, subdir_rule).
    """
    dir_name = ar.exp_dir_name(
        run_kwargs["gnn_type"], run_kwargs["n_layers"], run_kwargs["d_model"],
        run_kwargs["top_k"], run_kwargs["n_heads"], run_kwargs["use_attention"])
    subdir = ar.results_subdir(run_kwargs.get("topk_mode", "fixed"),
                               run_kwargs.get("cat_input", "log"),
                               run_kwargs.get("train_setup", "ablation"))
    return os.path.join(run_kwargs["exp_root"], "exp",
                        run_kwargs["dataset_name"], subdir, dir_name)


def eval_T_dict(dataset_cfg: dict, seed: int) -> dict:
    """Mirror of ``_run_one``'s ``T_dict_eval``.

    MIRRORED, not imported: it is built inline in ``_run_one`` and is not
    reachable through the capture (it is first used after training). The
    source-digest guard in tests/test_split_runner_backcompat.py fails if
    ``_run_one`` changes, so this cannot drift silently.
    """
    return dict(
        seed=seed,
        normalization=dataset_cfg.get("eval_normalization", None),
        num_nan_policy=None,
        cat_nan_policy=None,
        cat_min_frequency=None,
        cat_encoding=None,
        y_policy="default",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_split(ar, run_kwargs: dict, gen_seed_ids, load_checkpoint: bool,
              force_retrain: bool = False, train_only: bool = False) -> dict:
    """Train (or load) once, then run only ``gen_seed_ids``.

    Each generation seed writes its own shard so concurrent jobs never append
    to the same file. Returns a summary dict.
    """
    import gc

    import torch

    import train as train_mod
    import sample as sample_mod
    from tab_ddpm import topk_logging

    if ar.EVALUATOR == "xgboost":
        from eval_xgboost import train_xgboost as train_eval_model
        eval_results_json = "results_xgboost.json"
    else:
        from eval_catboost import train_catboost as train_eval_model
        eval_results_json = "results_catboost.json"
    from eval_tabddpm_protocol import (
        compute_dcr_tabddpm_mean, compute_dcr_baseline,
        compute_wasserstein_mean, compute_membership_inference_auc,
    )
    import lib

    cap = capture_context(ar, run_kwargs)
    parent_dir = cap["parent_dir"]
    real_data_path = cap["real_data_path"]
    dataset_cfg = run_kwargs["dataset_cfg"]
    dataset_name = run_kwargs["dataset_name"]
    ckpt = os.path.join(parent_dir, "model_ema.pt")
    # Same lookup _run_one makes (imported, not mirrored): the sampling batch
    # size is the only setup value the capture does not carry.
    setup, _ = ar.training_setup(dataset_name, dataset_cfg,
                                 run_kwargs.get("train_setup", "ablation"),
                                 run_kwargs["repo_root"])

    # ---- training ---------------------------------------------------------
    if load_checkpoint:
        if not os.path.exists(ckpt):
            raise SystemExit(
                f"--load_checkpoint given but no checkpoint at {ckpt}.\n"
                f"Training never finished for this combination, so there is "
                f"nothing to load. Submit the seed-0 job (which trains) first, "
                f"or drop --load_checkpoint to retrain here.")
        # A "log" and a "prob" checkpoint have identical tensor shapes, so
        # loading the wrong one would run silently with the wrong model.
        # (The *_catfix directory suffix already separates them; this also
        # catches a checkpoint copied or moved by hand.)
        # Verified before anything is sampled: the record written after
        # training must match what this job asks for. Without a record the
        # checkpoint predates it: cat_input counts as 'log' and topk_mode
        # cannot be checked (only the tree name vouches for it).
        wanted = run_kwargs.get("cat_input", "log")
        wanted_mode = run_kwargs.get("topk_mode", "fixed")
        record = ar.checkpoint_record(parent_dir)
        trained = (record or {}).get("cat_input", "log")
        if trained != wanted:
            raise SystemExit(
                f"refusing to load {ckpt}: it was trained with cat_input="
                f"{trained!r} but this job requests cat_input={wanted!r}.\n"
                f"(Checkpoints without {ar.DENOISER_CONFIG_FILE} predate the "
                f"option and count as 'log'.) Point the job at the matching "
                f"results tree, or drop --load_checkpoint to retrain.")
        wanted_setup = run_kwargs.get("train_setup", "ablation")
        trained_setup = (record or {}).get("train_setup", "ablation")
        if trained_setup != wanted_setup:
            raise SystemExit(
                f"refusing to load {ckpt}: it was trained with train_setup="
                f"{trained_setup!r} but this job requests train_setup="
                f"{wanted_setup!r} ({ar.DENOISER_CONFIG_FILE}; absent key = "
                f"'ablation').")
        if record is not None and record.get("topk_mode") != wanted_mode:
            raise SystemExit(
                f"refusing to load {ckpt}: it was trained with topk_mode="
                f"{record.get('topk_mode')!r} but this job requests "
                f"topk_mode={wanted_mode!r} ({ar.DENOISER_CONFIG_FILE}).")
        print(f"  [LOAD] {ckpt} (cat_input={trained}, topk_mode="
              f"{(record or {}).get('topk_mode', 'unrecorded')}) -- skipping training")
    elif os.path.exists(ckpt) and not force_retrain:
        # Training would overwrite model_ema.pt. Any generation seed already
        # finished came from the OLD weights, so the run would end up averaging
        # seeds from two different models -- and nothing downstream could tell.
        state = scan_state(ar, parent_dir)
        raise SystemExit(
            f"refusing to train: {ckpt} already exists.\n"
            f"  generation seeds already done with those weights: "
            f"{state['done'] or 'none'}\n"
            f"Training now would overwrite the checkpoint and mix seeds from "
            f"two different models.\n"
            f"  - to sample the missing seeds from the existing model, pass "
            f"--load_checkpoint (or --resume)\n"
            f"  - to deliberately start over, pass --force_retrain, which "
            f"discards every seed already done")
    else:
        if force_retrain and os.path.exists(ckpt):
            print(f"  [FORCE] overwriting {ckpt} and discarding any finished "
                  f"generation seeds")
        print(f"  [TRAIN] {run_kwargs['dataset_name']} -- "
              f"{cap['steps']} steps, seed {cap['seed']}")
        train_mod.train(**cap)          # the real trainer, untouched kwargs
        # Same record _run_one writes after training (checkpoint_cat_input).
        with open(os.path.join(parent_dir, ar.DENOISER_CONFIG_FILE), "w") as fh:
            json.dump(ar.denoiser_record(run_kwargs.get("cat_input", "log"),
                                         run_kwargs.get("topk_mode", "fixed"),
                                         run_kwargs.get("train_setup", "ablation")), fh, indent=2)

    X_num_real, X_cat_real, cat_sizes, cat_enc = ar._load_raw_real(real_data_path)
    T_dict_eval = eval_T_dict(dataset_cfg, run_kwargs["seed"])

    if train_only:
        # Training is the only thing this job does. model_ema.pt is now on
        # disk, so every generation seed can run as its own parallel job with
        # --load_checkpoint. Used where train + one seed would not fit in the
        # walltime together.
        print(f"  [TRAIN-ONLY] checkpoint at {ckpt}")
        return dict(parent_dir=parent_dir, done=[], skipped=[], trained=True)

    done, skipped = [], []
    for gen_seed in gen_seed_ids:
        seed_metrics_path = os.path.join(parent_dir, SEED_METRICS_FMT.format(gen_seed))
        if os.path.exists(seed_metrics_path):
            print(f"  [SKIP] gen_seed={gen_seed} -- {os.path.basename(seed_metrics_path)} present")
            skipped.append(gen_seed)
            continue

        gen_dir = os.path.join(parent_dir, f"gen_seed_{gen_seed}")
        os.makedirs(gen_dir, exist_ok=True)
        shard = os.path.join(parent_dir, SHARD_FMT.format(gen_seed))
        if os.path.exists(shard):
            os.remove(shard)
        # A seed cut off mid-way left a partial block in the consolidated file.
        # Drop it, or re-running would give this seed duplicate clf rows.
        n_dropped = drop_seed_rows(
            os.path.join(parent_dir, ar.PER_SEED_FILE), gen_seed)
        if n_dropped:
            print(f"    dropped {n_dropped} partial row(s) for gen_seed="
                  f"{gen_seed} from {ar.PER_SEED_FILE}")

        gen_metrics: dict = {}
        test_metrics: dict = {}

        topk_logging.set_tag(str(gen_seed))

        # --- sample: the ORIGINAL call, with the ORIGINAL seed derivation ---
        sample_mod.sample(
            parent_dir=gen_dir,
            real_data_path=real_data_path,
            model_path=ckpt,
            model_type="mlp",
            model_params=ar.deepcopy(cap["model_params"]),
            T_dict=cap["T_dict"],
            num_numerical_features=cap["num_numerical_features"],
            device=torch.device(run_kwargs["device"]),
            seed=gen_seed,
            change_val=False,
            graph_params=cap["graph_params"],
            num_timesteps=cap["num_timesteps"],
            gaussian_loss_type=cap["gaussian_loss_type"],
            scheduler=cap["scheduler"],
            num_samples=dataset_cfg["num_samples"],
            batch_size=setup["sample_batch_size"],
        )

        # --- per-gen-seed metrics (MIRRORED wiring, delegated computation) ---
        ar._load_raw_syn(gen_dir, cat_enc)
        for key, fn in (
            ("dcr_tabddpm_mean", lambda: compute_dcr_tabddpm_mean(
                dataset_name, real_data_path, gen_dir)),
            ("dcr_baseline", lambda: compute_dcr_baseline(
                dataset_name, real_data_path, gen_dir)),
            ("wasserstein_mean", lambda: compute_wasserstein_mean(
                dataset_name, real_data_path, gen_dir)),
            ("mia_auc", lambda: compute_membership_inference_auc(
                dataset_name, real_data_path, gen_dir)),
        ):
            try:
                gen_metrics[key] = [fn()]
            except Exception as exc:                        # noqa: BLE001
                print(f"    [WARN] {key} failed for gen_seed={gen_seed}: {exc}")

        if ar.TABDIFF_FIDELITY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                from eval_tabdiff_fidelity import compute_tabdiff_fidelity
                fid = compute_tabdiff_fidelity(gen_dir, real_data_path,
                                               device=run_kwargs["device"],
                                               metrics=ar.TABDIFF_METRICS,
                                               seed=gen_seed)
            except Exception as exc:                        # noqa: BLE001
                print(f"    [WARN] TabDiff fidelity failed for "
                      f"gen_seed={gen_seed}: {exc}")
                fid = {}
            for key, value in fid.items():
                gen_metrics[key] = [value]

        ar._append_per_seed(shard, gen_seed, None, gen_metrics, {})

        # --- classifier loop: the ORIGINAL call and seed derivation ---------
        for clf_seed in range(ar.N_CLF_SEEDS):
            train_eval_model(
                parent_dir=gen_dir,
                real_data_path=real_data_path,
                eval_type="synthetic",
                T_dict={**T_dict_eval, "seed": clf_seed},
                seed=clf_seed,
                change_val=False,
            )
            report = lib.load_json(os.path.join(gen_dir, eval_results_json))
            before = {k: len(v) for k, v in test_metrics.items()}
            for metric, value in report["metrics"]["test"].items():
                if isinstance(value, (int, float)):
                    test_metrics.setdefault(metric, []).append(value)
                elif isinstance(value, dict) and metric in ("macro avg", "weighted avg"):
                    for sub, subval in value.items():
                        if isinstance(subval, (int, float)):
                            test_metrics.setdefault(f"{metric}/{sub}", []).append(subval)
            ar._append_per_seed(shard, gen_seed, clf_seed, test_metrics, before)

        with open(seed_metrics_path, "w") as fh:
            json.dump({"gen_seed": gen_seed, "gen_metrics": gen_metrics,
                       "test_metrics": test_metrics}, fh, indent=2)
        print(f"  [OK] gen_seed={gen_seed}")
        done.append(gen_seed)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return dict(parent_dir=parent_dir, done=done, skipped=skipped)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_shards(ar, parent_dir: str, require_all: bool = True) -> dict:
    """Rebuild results_full_averaged.json and per_seed_scores.csv from shards.

    Order-independent: the shards are read in generation-seed order whatever
    order the jobs finished in, so the consolidated per_seed_scores.csv is
    byte-comparable with a monolithic run's.
    """
    import numpy as np

    present = {}
    for path in glob.glob(os.path.join(parent_dir, SEED_METRICS_FMT.format("*"))):
        d = json.load(open(path))
        present[int(d["gen_seed"])] = d
    missing = [s for s in range(ar.N_GEN_SEEDS) if s not in present]
    if missing and require_all:
        raise SystemExit(f"cannot merge: generation seeds {missing} are missing "
                         f"from {parent_dir}")

    gen_all: dict = {}
    test_all: dict = {}
    for s in sorted(present):
        for k, v in present[s]["gen_metrics"].items():
            gen_all.setdefault(k, []).extend(v)
        for k, v in present[s]["test_metrics"].items():
            test_all.setdefault(k, []).extend(v)

    averaged: dict = {}
    for src in (test_all, gen_all):
        for metric, values in src.items():
            arr = np.array(values, dtype=float)
            averaged[metric] = {"mean": float(arr.mean()), "std": float(arr.std())}

    with open(os.path.join(parent_dir, "results_full_averaged.json"), "w") as fh:
        json.dump(averaged, fh, indent=2)

    # Consolidated long-format file, in generation-seed order. hpc_density's
    # density_curve.py and common.sh both read this exact name.
    rows = []
    for s in sorted(present):
        shard = os.path.join(parent_dir, SHARD_FMT.format(s))
        if not os.path.exists(shard):
            continue
        with open(shard, newline="") as fh:
            r = csv.reader(fh)
            header = next(r, None)
            rows.extend(list(r))
    if rows:
        with open(os.path.join(parent_dir, ar.PER_SEED_FILE), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["gen_seed", "clf_seed", "metric", "value"])
            w.writerows(rows)

    print(f"  merged {len(present)}/{ar.N_GEN_SEEDS} generation seeds "
          f"-> results_full_averaged.json ({len(averaged)} metrics), "
          f"{ar.PER_SEED_FILE} ({len(rows)} rows)")
    if missing:
        print(f"  WARNING: partial merge, missing generation seeds {missing}")
    return averaged


# ---------------------------------------------------------------------------
# Legacy consolidated file: read, prune, import
# ---------------------------------------------------------------------------
def read_legacy_rows(path: str):
    """(gen_seed, clf_seed, metric, value) rows of a consolidated file."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            if len(row) < 4:
                continue
            try:
                g = int(row[0])
            except ValueError:
                continue
            out.append((g, row[1], row[2], row[3]))
    return out


def drop_seed_rows(path: str, seed: int) -> int:
    """Remove every row of one generation seed from the consolidated file.

    A seed cut off mid-way leaves a partial block behind -- some of its
    classifier seeds, not all. Re-running it would append a second, complete
    block, and that seed would then carry duplicate clf rows and be counted
    twice by anything that averages the file. So its rows go first.

    Returns how many rows were dropped.
    """
    rows = read_legacy_rows(path)
    if not rows:
        return 0
    keep = [r for r in rows if r[0] != seed]
    dropped = len(rows) - len(keep)
    if not dropped:
        return 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gen_seed", "clf_seed", "metric", "value"])
        w.writerows(keep)
    return dropped


def import_legacy_seed(ar, parent_dir: str, seed: int) -> bool:
    """Turn a COMPLETE legacy block into this module's per-seed files.

    A monolithic run that hit the walltime leaves finished seeds in
    per_seed_scores.csv and nothing else. Without this they would have no
    shard, and merge_shards -- which rebuilds everything from shards -- would
    silently drop them. Writing the shard and metrics_gen<k>.json makes the
    merge complete and keeps the work.
    """
    legacy = os.path.join(parent_dir, ar.PER_SEED_FILE)
    rows = [r for r in read_legacy_rows(legacy) if r[0] == seed]
    if not rows:
        return False
    clf_seeds = {r[1] for r in rows if r[1] != ""}
    if len(clf_seeds) < ar.N_CLF_SEEDS:
        return False        # partial: not importable, must be re-run

    gen_metrics: dict = {}
    test_metrics: dict = {}
    for _g, clf, metric, value in rows:
        target = gen_metrics if clf == "" else test_metrics
        target.setdefault(metric, []).append(float(value))

    with open(os.path.join(parent_dir, SEED_METRICS_FMT.format(seed)), "w") as fh:
        json.dump({"gen_seed": seed, "gen_metrics": gen_metrics,
                   "test_metrics": test_metrics, "imported_from": ar.PER_SEED_FILE},
                  fh, indent=2)
    shard = os.path.join(parent_dir, SHARD_FMT.format(seed))
    with open(shard, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gen_seed", "clf_seed", "metric", "value"])
        w.writerows(rows)
    return True


# ---------------------------------------------------------------------------
# State scan -- which generation seeds already have metrics (point 4)
# ---------------------------------------------------------------------------
def scan_state(ar, parent_dir: str) -> dict:
    """Which generation seeds are done, and is there a checkpoint to reuse.

    Handles BOTH layouts:

    * new  -- ``metrics_gen<k>.json``, written by this module per shard;
    * old  -- a partial ``per_seed_scores.csv`` left by a monolithic run that
              hit the walltime. ``_run_one`` appends as it goes, so the rows of
              the seeds that finished survive. A seed counts as done when it
              has N_CLF_SEEDS distinct classifier seeds in that file.

    ``model_ema.pt`` is written by scripts/train.py only after the full
    training loop, so its presence means training completed.
    """
    ckpt = os.path.join(parent_dir, "model_ema.pt")
    state = dict(parent_dir=parent_dir,
                 has_checkpoint=os.path.exists(ckpt),
                 checkpoint=ckpt,
                 done=[], missing=[], source={})

    for k in range(ar.N_GEN_SEEDS):
        if os.path.exists(os.path.join(parent_dir, SEED_METRICS_FMT.format(k))):
            state["done"].append(k)
            state["source"][k] = "metrics_gen%d.json" % k

    legacy = os.path.join(parent_dir, ar.PER_SEED_FILE)
    if os.path.exists(legacy):
        clf_seen: dict = {}
        with open(legacy, newline="") as fh:
            r = csv.reader(fh)
            next(r, None)
            for row in r:
                if len(row) < 4:
                    continue
                try:
                    g = int(row[0])
                except ValueError:
                    continue
                if row[1] != "":
                    clf_seen.setdefault(g, set()).add(row[1])
        for k, seeds in clf_seen.items():
            if k in state["done"] or k >= ar.N_GEN_SEEDS:
                continue
            if len(seeds) >= ar.N_CLF_SEEDS:
                # Complete legacy block: materialise its shard, so
                # merge_shards -- which rebuilds from shards only -- keeps it.
                if import_legacy_seed(ar, parent_dir, k):
                    state["done"].append(k)
                    state["source"][k] = (f"{ar.PER_SEED_FILE} "
                                          f"({len(seeds)} clf seeds, imported)")
            else:
                state.setdefault("partial", {})[k] = len(seeds)

    state["done"] = sorted(state["done"])
    state["missing"] = [k for k in range(ar.N_GEN_SEEDS) if k not in state["done"]]
    return state


def print_scan(ar, state: dict) -> None:
    print(f"  run            : {state['parent_dir']}")
    print(f"  checkpoint     : "
          + ("model_ema.pt present -- seeds can be loaded"
             if state["has_checkpoint"] else
             "MISSING -- training never finished"))
    print(f"  generation seeds done    : {state['done'] or '-'}")
    for k in state["done"]:
        print(f"      {k}: from {state['source'][k]}")
    print(f"  generation seeds missing : {state['missing'] or '-'}")
    for k, n in sorted(state.get("partial", {}).items()):
        print(f"      {k}: PARTIAL -- {n}/{ar.N_CLF_SEEDS} classifier seeds in "
              f"{ar.PER_SEED_FILE}; those rows are dropped before it re-runs")
    if state["missing"] and not state["has_checkpoint"]:
        print()
        print("  NOTE: there is no checkpoint, so the missing seeds cannot be")
        print("  sampled without retraining. Retraining reproduces the original")
        print("  model only if training is deterministic given --seed. The RNGs")
        print("  are fully re-seeded (scripts/train.py:113) and the shuffle uses")
        print("  torch.randperm off that RNG, so it is deterministic on CPU; on")
        print("  GPU it additionally depends on what libzero's")
        print("  improve_reproducibility sets (cudnn.deterministic / benchmark).")
        print("  Verify on the cluster with:")
        print("      python -c \"import zero,inspect;"
              "print(inspect.getsource(zero.improve_reproducibility))\"")
        print("  If it does not set cudnn.deterministic=True and")
        print("  benchmark=False, a retrained model is statistically equivalent")
        print("  but NOT bit-identical, and the seeds already finished should be")
        print("  discarded rather than mixed with the new ones.")


# ---------------------------------------------------------------------------
def parse_seed_ids(text: str, n_gen_seeds: int):
    ids = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    bad = [i for i in ids if not 0 <= i < n_gen_seeds]
    if bad:
        raise SystemExit(f"--gen_seed_ids out of range 0..{n_gen_seeds - 1}: {bad}")
    return sorted(set(ids))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gen_seed_ids", default=None,
                    help="comma list, e.g. '0' or '1,2'. Omitted = every seed, "
                         "which is the monolithic behaviour.")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="run only the generation seeds that have no metrics "
                         "yet (implies --load_checkpoint when a checkpoint "
                         "exists)")
    ap.add_argument("--load_checkpoint", action="store_true", default=False,
                    help="skip training and sample from the saved model_ema.pt")
    ap.add_argument("--merge_only", action="store_true", default=False,
                    help="only rebuild results_full_averaged.json from shards")
    ap.add_argument("--allow_partial_merge", action="store_true", default=False)
    ap.add_argument("--train_only", action="store_true", default=False,
                    help="train, save the checkpoint, run no generation seed. "
                         "The seeds then run as independent jobs with "
                         "--load_checkpoint.")
    ap.add_argument("--force_retrain", action="store_true", default=False,
                    help="train even though model_ema.pt exists, discarding "
                         "every generation seed already done with it")
    ap.add_argument("--scan_shell", action="store_true", default=False,
                    help="print the scan as shell assignments for eval")
    ap.add_argument("--scan", action="store_true", default=False,
                    help="report which generation seeds already have metrics "
                         "and whether a checkpoint exists; runs nothing")
    ap.add_argument("--repo_root", default=REPO_DEFAULT)
    ours, rest = ap.parse_known_args()

    ar = _load_runner(ours.repo_root)

    import importlib
    mod = importlib.import_module(f"{ours.dataset.replace('-', '_')}_ablation")
    args = ar.make_parser(mod.DATASET_NAME).parse_args(rest)
    # Same mechanism run_ablation uses: --steps / --n_gen_seeds / --n_clf_seeds
    # / --evaluator / --tabdiff_fidelity patch the module constants. Anything
    # left at None is ignored, so omitting them keeps the ablation's budget.
    ar.apply_cli_overrides(args)
    ar.check_train_setup_args(args)

    combo = (args.gnn_type, args.n_layers, args.d_model, args.top_k, args.n_heads)
    if any(v is None for v in combo):
        raise SystemExit("pass --gnn_type --n_layers --d_model --top_k --n_heads")

    run_kwargs = dict(
        gnn_type=args.gnn_type, n_layers=args.n_layers, d_model=args.d_model,
        top_k=args.top_k, n_heads=args.n_heads,
        use_attention=bool(args.use_attention),
        dataset_name=mod.DATASET_NAME, dataset_cfg=mod.DATASET_CFG,
        repo_root=os.path.abspath(args.repo_root),
        data_root=os.path.abspath(args.data_root or args.repo_root),
        exp_root=os.path.abspath(args.exp_root or args.repo_root),
        device=args.device, seed=args.seed,
        topk_mode=args.topk_mode, k_min=args.k_min,
        topk_warmup_steps=args.topk_warmup_steps,
        topk_log_every=args.topk_log_every,
        cat_input=args.cat_input,
        train_setup=args.train_setup,
    )

    # --scan and --merge_only touch files only, so they never import the
    # training stack -- they run on a login node.
    if ours.scan or ours.scan_shell:
        state = scan_state(ar, parent_dir_for(ar, run_kwargs))
        if ours.scan_shell:
            # Machine-readable, for `eval "$(... --scan_shell)"` in
            # submit_split.sh, which has to know whether to submit a training
            # job at all.
            print(f"HAS_CHECKPOINT={1 if state['has_checkpoint'] else 0}")
            print(f"DONE_SEEDS='{' '.join(str(k) for k in state['done'])}'")
            print(f"MISSING_SEEDS='{' '.join(str(k) for k in state['missing'])}'")
            print(f"PARTIAL_SEEDS='{' '.join(str(k) for k in sorted(state.get('partial', {})))}'")
            print(f"RUN_DIR='{state['parent_dir']}'")
        else:
            print_scan(ar, state)
        return 0

    if ours.merge_only:
        merge_shards(ar, parent_dir_for(ar, run_kwargs),
                     require_all=not ours.allow_partial_merge)
        return 0

    if ours.train_only:
        if ours.load_checkpoint:
            raise SystemExit("--train_only and --load_checkpoint are opposites")
        print("  mode: train only, no generation seed")
        run_split(ar, run_kwargs, [], False, ours.force_retrain, train_only=True)
        return 0

    load_ckpt = ours.load_checkpoint
    if ours.resume:
        state = scan_state(ar, parent_dir_for(ar, run_kwargs))
        print_scan(ar, state)
        if not state["missing"]:
            print("  nothing to resume")
            return 0
        ids = state["missing"]
        load_ckpt = load_ckpt or state["has_checkpoint"]
    else:
        ids = (list(range(ar.N_GEN_SEEDS)) if ours.gen_seed_ids is None
               else parse_seed_ids(ours.gen_seed_ids, ar.N_GEN_SEEDS))
    print(f"  generation seeds : {ids}  (of {ar.N_GEN_SEEDS})")

    out = run_split(ar, run_kwargs, ids, load_ckpt, ours.force_retrain)

    have = len(glob.glob(os.path.join(out["parent_dir"],
                                      SEED_METRICS_FMT.format("*"))))
    if have >= ar.N_GEN_SEEDS:
        merge_shards(ar, out["parent_dir"])
    else:
        print(f"  {have}/{ar.N_GEN_SEEDS} generation seeds present; "
              f"run --merge_only once the rest finish")
    return 0


if __name__ == "__main__":
    sys.exit(main())

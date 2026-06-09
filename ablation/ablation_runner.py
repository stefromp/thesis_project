"""
Shared runner for GNN ablation studies across all 16 datasets.

Evaluation protocol — matches the default notebook exactly:
  N_GEN_SEEDS = 5  : sample the trained model 5 times (different gen seeds)
  N_CLF_SEEDS = 10 : run CatBoost 10 times per generated dataset

  Per gen_seed:
    1. Sample synthetic data
    2. Fidelity metrics  (eval_fidelity.compute_fidelity_metrics)
    3. DCR privacy metric (pairwise distance to closest real record)
    4. CatBoost x N_CLF_SEEDS -> collect test-split metrics

  Aggregation: mean +/- std over all 50 CatBoost runs (utility) and 5 gen runs
  (fidelity / privacy). Saved to results_full_averaged.json.

Ablation grid (per dataset):
  gnn_type : gcn | gat | gatv2 | gin
  n_layers : 2, 3, 4
  d_model  : 128, 256, 512, 1024
  top_k    : 0, 3  (sparsity_top_k for DynamicAdjacency; 0 = dense)
  n_heads  :  4, 8    (GCN/GIN ignore this; only one value run for those)

Training: 20 000 steps, no intermediate checkpoints.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import traceback
import argparse
from collections import defaultdict
from copy import deepcopy
from typing import Optional

# ---------------------------------------------------------------------------
# Ablation + evaluation constants
# ---------------------------------------------------------------------------

ABLATION_STEPS = 20_000
N_GEN_SEEDS    = 5
N_CLF_SEEDS    = 10

GNN_TYPES = ["gcn", "gat", "gatv2", "gin"]
N_LAYERS  = [2, 3, 4]
D_MODELS  = [128, 256, 512, 1024]
TOP_KS    = [0, 3]
N_HEADS   = [4, 8]

_HEADLESS_GNNS = {"gcn", "gin"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exp_dir_name(gnn_type: str, n_layers: int, d_model: int,
                 top_k: int, n_heads: int) -> str:
    return (f"gnn_{gnn_type}_layers{n_layers}_dim{d_model}"
            f"_topk{top_k}_heads{n_heads}")


def iter_ablation_grid():
    """Yield all (gnn_type, n_layers, d_model, top_k, n_heads) combinations.

    GCN and GIN have no attention heads; only the first n_heads value (= 2)
    is run for them to avoid producing identical redundant jobs.
    """
    for combo in itertools.product(GNN_TYPES, N_LAYERS, D_MODELS, TOP_KS, N_HEADS):
        gnn_type, _, _, _, n_heads = combo
        if gnn_type in _HEADLESS_GNNS and n_heads != N_HEADS[0]:
            continue
        yield combo


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def _setup_zero_rtdl_stubs() -> None:
    """Inject minimal zero / rtdl stubs when those packages are not installed."""
    import random
    import time
    import types

    import numpy as np
    import torch
    import torch.nn as nn

    if "zero" not in sys.modules:
        zero_mod        = types.ModuleType("zero")
        zero_random_mod = types.ModuleType("zero.random")
        zero_hw_mod     = types.ModuleType("zero.hardware")

        zero_random_mod.get_state  = lambda: None
        zero_random_mod.set_state  = lambda _: None
        zero_hw_mod.get_gpus_info  = lambda: {}
        zero_mod.random            = zero_random_mod
        zero_mod.hardware          = zero_hw_mod
        zero_mod.iter_batches      = lambda batch, n_: (
            batch[i: i + n_] for i in range(0, len(batch), n_)
        )

        class _Timer:
            def __init__(self):
                self._t = None
            def run(self):
                self._t = time.time()
            def __call__(self):
                return time.time() - self._t if self._t else 0.0

        zero_mod.Timer = _Timer

        def _repro(s: int = 0) -> None:
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

        zero_mod.improve_reproducibility = _repro
        sys.modules["zero"]          = zero_mod
        sys.modules["zero.random"]   = zero_random_mod
        sys.modules["zero.hardware"] = zero_hw_mod

    if "rtdl" not in sys.modules:
        rtdl_mod = types.ModuleType("rtdl")
        for _n in ("CLSToken", "NumericalFeatureTokenizer",
                   "CategoricalFeatureTokenizer"):
            setattr(rtdl_mod, _n, type(_n, (nn.Module,), {}))
        sys.modules["rtdl"] = rtdl_mod


def _setup_repo_path(repo_root: str) -> None:
    for p in [repo_root, os.path.join(repo_root, "scripts")]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# DCR (inlined to avoid resample_privacy.py's heavy import chain)
# ---------------------------------------------------------------------------

def _compute_dcr(real_data_path: str, gen_dir: str) -> float:
    """Median distance-to-closest-record between synthetic and real data.

    Matches privacy_metrics() from resample_privacy.py but avoids triggering
    its full module-level imports (smote, eval_seeds, etc.).
    """
    import numpy as np
    import lib
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.metrics import pairwise_distances

    task_type = lib.load_json(os.path.join(real_data_path, "info.json"))["task_type"]
    X_num_r, X_cat_r, y_r = lib.read_pure_data(real_data_path, "train")
    X_num_f, X_cat_f, y_f = lib.read_pure_data(gen_dir, "train")

    if task_type == "regression":
        X_num_r = np.concatenate([X_num_r, y_r[:, None]], axis=1)
        X_num_f = np.concatenate([X_num_f, y_f[:, None]], axis=1)
    else:
        lbl_r = y_r[:, None].astype(int).astype(str)
        lbl_f = y_f[:, None].astype(int).astype(str)
        if X_cat_r is None:
            X_cat_r, X_cat_f = lbl_r, lbl_f
        else:
            X_cat_r = np.concatenate([X_cat_r, lbl_r], axis=1)
            X_cat_f = np.concatenate([X_cat_f, lbl_f], axis=1)

    if len(y_r) > 50_000:
        ix = np.random.choice(len(y_r), 50_000, replace=False)
        X_num_r = X_num_r[ix]
        X_cat_r = X_cat_r[ix] if X_cat_r is not None else None
    if len(y_f) > 50_000:
        ix = np.random.choice(len(y_f), 50_000, replace=False)
        X_num_f = X_num_f[ix]
        X_cat_f = X_cat_f[ix] if X_cat_f is not None else None

    mm  = MinMaxScaler().fit(X_num_r)
    X_r = mm.transform(X_num_r)
    X_f = mm.transform(X_num_f)

    if X_cat_r is not None:
        ohe   = OneHotEncoder(handle_unknown="ignore").fit(X_cat_r)
        cat_r = ohe.transform(X_cat_r).toarray() / (2 ** 0.5)
        cat_f = ohe.transform(X_cat_f).toarray() / (2 ** 0.5)
        X_r   = np.concatenate([X_r, cat_r], axis=1)
        X_f   = np.concatenate([X_f, cat_f], axis=1)

    dist_rf   = pairwise_distances(X_f, Y=X_r, metric="l2", n_jobs=-1)
    min_dists = dist_rf.min(axis=1)
    return float(np.median(min_dists))


# ---------------------------------------------------------------------------
# Raw data loader for fidelity metrics
# ---------------------------------------------------------------------------

def _load_raw_real(real_data_path: str):
    """Load raw (unnormalised) real training arrays for fidelity computation.

    Returns (X_num_raw, X_cat_raw_int, cat_sizes, ordinal_encoder_or_None).
    """
    import numpy as np
    from sklearn.preprocessing import OrdinalEncoder

    X_num_raw = np.load(os.path.join(real_data_path, "X_num_train.npy"),
                        allow_pickle=True).astype(np.float32)

    cat_path = os.path.join(real_data_path, "X_cat_train.npy")
    if os.path.exists(cat_path):
        _X_cat = np.load(cat_path, allow_pickle=True)
        if _X_cat.dtype.kind in ("U", "S", "O"):
            enc       = OrdinalEncoder(dtype=np.float64)
            X_cat_raw = enc.fit_transform(_X_cat).astype(np.int64)
        else:
            enc       = None
            X_cat_raw = _X_cat.astype(np.int64)
        n_cat     = X_cat_raw.shape[1]
        cat_sizes = [int(X_cat_raw[:, i].max()) + 1 for i in range(n_cat)]
    else:
        X_cat_raw = None
        cat_sizes = []
        enc       = None

    return X_num_raw, X_cat_raw, cat_sizes, enc


def _load_raw_syn(gen_dir: str, cat_enc):
    """Load raw synthetic arrays saved by sample.py, applying the same encoding."""
    import numpy as np

    X_num_syn = np.load(os.path.join(gen_dir, "X_num_train.npy"),
                        allow_pickle=True).astype(np.float32)

    cat_path = os.path.join(gen_dir, "X_cat_train.npy")
    if os.path.exists(cat_path):
        _X_cat = np.load(cat_path, allow_pickle=True)
        if cat_enc is not None:
            X_cat_syn = cat_enc.transform(_X_cat).astype(np.int64)
        else:
            X_cat_syn = _X_cat.astype(np.int64)
    else:
        X_cat_syn = None

    return X_num_syn, X_cat_syn


# ---------------------------------------------------------------------------
# Single ablation run  (train -> multi-seed sample+eval loop)
# ---------------------------------------------------------------------------

def _run_one(
    *,
    gnn_type: str,
    n_layers: int,
    d_model: int,
    top_k: int,
    n_heads: int,
    dataset_name: str,
    dataset_cfg: dict,
    repo_root: str,
    data_root: str,
    exp_root: str,
    device: str,
    seed: int,
    skip_if_done: bool,
) -> dict:
    """Train once, then evaluate with N_GEN_SEEDS x N_CLF_SEEDS protocol.

    Returns the aggregated results dict (also written to results_full_averaged.json).
    """
    import numpy as np
    import torch
    import lib
    from train import train as tabddpm_train
    from sample import sample as tabddpm_sample
    from eval_catboost import train_catboost
    from eval_fidelity import compute_fidelity_metrics

    dir_name       = exp_dir_name(gnn_type, n_layers, d_model, top_k, n_heads)
    parent_dir     = os.path.join(exp_root, "exp", dataset_name, "ablation", dir_name)
    real_data_path = os.path.join(data_root, dataset_cfg["real_data_path"])
    results_file   = os.path.join(parent_dir, "results_full_averaged.json")

    if skip_if_done and os.path.exists(results_file):
        print(f"  [SKIP] {dir_name} -- already done")
        with open(results_file) as fh:
            return json.load(fh)

    os.makedirs(parent_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  DATASET : {dataset_name}")
    print(f"  RUN     : {dir_name}")
    print(f"{'=' * 72}")

    model_params = dict(
        num_classes=dataset_cfg["num_classes"],
        is_y_cond=dataset_cfg["is_y_cond"],
        rtdl_params=dict(
            d_layers=list(dataset_cfg["rtdl_d_layers"]),
            dropout=0.0,
        ),
    )

    T_dict_train = dict(
        seed=seed,
        normalization=dataset_cfg["train_normalization"],
        num_nan_policy=None,
        cat_nan_policy=None,
        cat_min_frequency=None,
        cat_encoding=None,
        y_policy="default",
    )

    # Eval T_dict normalization comes from each dataset's ddpm_cb_best eval.T config.
    T_dict_eval = dict(
        seed=seed,
        normalization=dataset_cfg.get("eval_normalization", None),
        num_nan_policy=None,
        cat_nan_policy=None,
        cat_min_frequency=None,
        cat_encoding=None,
        y_policy="default",
    )

    graph_params = dict(
        enabled=True,
        mode="dynamic",
        gnn_type=gnn_type,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        sparsity_top_k=top_k,
        dropout=0.0,
    )

    # ---- 1. Train (single run) ---------------------------------------------
    tabddpm_train(
        parent_dir=parent_dir,
        real_data_path=real_data_path,
        model_type="mlp",
        model_params=deepcopy(model_params),
        T_dict=T_dict_train,
        num_numerical_features=dataset_cfg["num_numerical_features"],
        device=torch.device(device),
        seed=seed,
        change_val=False,
        graph_params=graph_params,
        num_timesteps=dataset_cfg["num_timesteps"],
        gaussian_loss_type="mse",
        scheduler=dataset_cfg["scheduler"],
        lr=dataset_cfg["lr"],
        weight_decay=dataset_cfg["weight_decay"],
        batch_size=dataset_cfg["batch_size"],
        steps=ABLATION_STEPS,
        checkpoint_every=0,
    )

    # Load raw real data once (shared across all gen seeds for fidelity)
    X_num_real, X_cat_real, cat_sizes, cat_enc = _load_raw_real(real_data_path)

    # ---- 2. Multi-seed evaluation loop -------------------------------------
    all_test_metrics: dict = defaultdict(list)
    all_gen_metrics:  dict = defaultdict(list)

    for gen_seed in range(N_GEN_SEEDS):
        gen_dir = os.path.join(parent_dir, f"gen_seed_{gen_seed}")
        os.makedirs(gen_dir, exist_ok=True)

        # 2a. Generate synthetic dataset
        tabddpm_sample(
            parent_dir=gen_dir,
            real_data_path=real_data_path,
            model_path=os.path.join(parent_dir, "model.pt"),
            model_type="mlp",
            model_params=deepcopy(model_params),
            T_dict=T_dict_train,
            num_numerical_features=dataset_cfg["num_numerical_features"],
            device=torch.device(device),
            seed=gen_seed,
            change_val=False,
            graph_params=graph_params,
            num_timesteps=dataset_cfg["num_timesteps"],
            gaussian_loss_type="mse",
            scheduler=dataset_cfg["scheduler"],
            num_samples=dataset_cfg["num_samples"],
            batch_size=dataset_cfg["sample_batch_size"],
        )

        # 2b. Fidelity metrics
        X_num_syn, X_cat_syn = _load_raw_syn(gen_dir, cat_enc)
        fidelity = compute_fidelity_metrics(
            X_real_num=X_num_real,
            X_syn_num=X_num_syn,
            X_real_cat=X_cat_real,
            X_syn_cat=X_cat_syn,
            cat_sizes=cat_sizes,
            seed=gen_seed,
        )
        for name, val in fidelity.items():
            if val is not None:
                all_gen_metrics[name].append(val)

        # 2c. DCR privacy metric
        try:
            dcr = _compute_dcr(real_data_path, gen_dir)
            all_gen_metrics["DCR"].append(dcr)
        except Exception as exc:
            print(f"    [WARN] DCR failed for gen_seed={gen_seed}: {exc}")

        print(
            f"  [gen_seed={gen_seed}]"
            f"  ColumnWise={fidelity['ColumnWise_score']:.2f}"
            f"  PairWise={fidelity['PairWise_score']:.2f}"
            f"  Coverage={fidelity.get('Coverage', float('nan')):.4f}"
            + (f"  DCR={all_gen_metrics['DCR'][-1]:.4f}"
               if all_gen_metrics["DCR"] else "")
        )

        # 2d. CatBoost utility: 10 classifier seeds
        for clf_seed in range(N_CLF_SEEDS):
            clf_eval_T = {**T_dict_eval, "seed": clf_seed}
            train_catboost(
                parent_dir=gen_dir,
                real_data_path=real_data_path,
                eval_type="synthetic",
                T_dict=clf_eval_T,
                seed=clf_seed,
                change_val=False,
            )
            run_report = lib.load_json(os.path.join(gen_dir, "results_catboost.json"))
            for metric, value in run_report["metrics"]["test"].items():
                all_test_metrics[metric].append(value)

    # ---- 3. Aggregate with mean +/- std ------------------------------------
    averaged: dict = {}

    print(
        f"\n  CatBoost utility  "
        f"({N_GEN_SEEDS} gen x {N_CLF_SEEDS} clf seeds = "
        f"{N_GEN_SEEDS * N_CLF_SEEDS} runs)"
    )
    for metric, values in all_test_metrics.items():
        arr = np.array(values)
        averaged[metric] = {"mean": float(arr.mean()), "std": float(arr.std())}
        print(f"    {metric:20s}: {arr.mean():.4f} +/- {arr.std():.4f}")

    fidelity_order = [
        ("ColumnWise_score", "Column-wise  [up]"),
        ("KST",              "KST          [down]"),
        ("TVD",              "TVD          [down]"),
        ("PairWise_score",   "Pair-wise    [up]"),
        ("DPCM",             "DPCM         [down]"),
        ("DCSM",             "DCSM         [down]"),
        ("mixed",            "Mixed        [down]"),
        ("Coverage",         "Coverage     [up]"),
        ("beta_Recall",      "beta-Recall  [up]"),
    ]
    print(f"\n  Fidelity  (averaged over {N_GEN_SEEDS} generation seeds)")
    for key, label in fidelity_order:
        if key in all_gen_metrics:
            arr = np.array(all_gen_metrics[key])
            averaged[key] = {"mean": float(arr.mean()), "std": float(arr.std())}
            print(f"    {label}: {arr.mean():.4f} +/- {arr.std():.4f}")

    if "DCR" in all_gen_metrics:
        arr = np.array(all_gen_metrics["DCR"])
        averaged["DCR"] = {"mean": float(arr.mean()), "std": float(arr.std())}
        print(f"\n  Privacy  DCR: {arr.mean():.4f} +/- {arr.std():.4f}")

    lib.dump_json(averaged, results_file)
    print(f"\n  [DONE] {dir_name}  ->  {results_file}")
    return averaged


# ---------------------------------------------------------------------------
# Main entry-point called by each dataset script
# ---------------------------------------------------------------------------

def aggregate_summary(
    dataset_name: str,
    args: argparse.Namespace,
) -> list:
    """Build ablation_summary.json from per-combo results_full_averaged.json files.

    Safe to call after all SLURM array tasks finish — reads only per-combo
    files that are never shared between tasks, so there is no race condition.
    """
    exp_root     = os.path.abspath(args.exp_root if args.exp_root else args.repo_root)
    ablation_dir = os.path.join(exp_root, "exp", dataset_name, "ablation")
    summary_path = os.path.join(ablation_dir, "ablation_summary.json")
    os.makedirs(ablation_dir, exist_ok=True)

    summary: list = []
    for gnn_type, n_layers, d_model, top_k, n_heads in iter_ablation_grid():
        name         = exp_dir_name(gnn_type, n_layers, d_model, top_k, n_heads)
        results_file = os.path.join(ablation_dir, name, "results_full_averaged.json")
        entry: dict = {
            "dataset":     dataset_name,
            "exp_name":    name,
            "gnn_type":    gnn_type,
            "n_layers":    n_layers,
            "d_model":     d_model,
            "top_k":       top_k,
            "n_heads":     n_heads,
            "steps":       ABLATION_STEPS,
            "n_gen_seeds": N_GEN_SEEDS,
            "n_clf_seeds": N_CLF_SEEDS,
        }
        if os.path.exists(results_file):
            with open(results_file) as fh:
                entry["status"]  = "done"
                entry["metrics"] = json.load(fh)
        else:
            entry["status"] = "missing"
        summary.append(entry)

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    n_done    = sum(1 for e in summary if e["status"] == "done")
    n_missing = sum(1 for e in summary if e["status"] == "missing")
    print(f"  [{dataset_name}] Aggregated: done={n_done}  missing={n_missing}"
          f"  -> {summary_path}")
    return summary


def run_ablation(
    dataset_name: str,
    dataset_cfg: dict,
    args: argparse.Namespace,
    single_combo: Optional[tuple] = None,
) -> list:
    """
    Run ablation study for one dataset.

    Parameters
    ----------
    dataset_name  : e.g. "higgs-small"
    dataset_cfg   : dict with keys: real_data_path, num_numerical_features,
                    num_classes, is_y_cond, rtdl_d_layers, num_timesteps,
                    scheduler, lr, weight_decay, batch_size, num_samples,
                    sample_batch_size, train_normalization
    args          : parsed argparse namespace (see make_parser())
    single_combo  : optional (gnn_type, n_layers, d_model, top_k, n_heads)
                    tuple for cluster single-job mode; if None, run all.
    """
    if getattr(args, "aggregate", False):
        return aggregate_summary(dataset_name, args)

    repo_root = os.path.abspath(args.repo_root)
    data_root = os.path.abspath(args.data_root if args.data_root else repo_root)
    exp_root  = os.path.abspath(args.exp_root  if args.exp_root  else repo_root)
    skip_flag = args.skip_if_done and not getattr(args, "no_skip", False)

    _setup_zero_rtdl_stubs()
    _setup_repo_path(repo_root)

    ablation_dir = os.path.join(exp_root, "exp", dataset_name, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    # SLURM single-combo path: never touch the shared summary file.
    # Each combo writes only to its own results_full_averaged.json, so
    # parallel array tasks cannot race. Run aggregate_summary() afterwards.
    if single_combo is not None:
        gnn_type, n_layers, d_model, top_k, n_heads = single_combo
        name = exp_dir_name(gnn_type, n_layers, d_model, top_k, n_heads)
        print(f"\n[1/1] {name}")
        try:
            result = _run_one(
                gnn_type=gnn_type,
                n_layers=n_layers,
                d_model=d_model,
                top_k=top_k,
                n_heads=n_heads,
                dataset_name=dataset_name,
                dataset_cfg=dataset_cfg,
                repo_root=repo_root,
                data_root=data_root,
                exp_root=exp_root,
                device=args.device,
                seed=args.seed,
                skip_if_done=skip_flag,
            )
        except Exception:
            traceback.print_exc()
            raise
        return [result]

    # Sequential (local / single-node) mode: no concurrency, safe to
    # read-modify-write the shared summary after each combo completes.
    summary_path = os.path.join(ablation_dir, "ablation_summary.json")
    summary: list = []
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            summary = json.load(fh)
    done_names = {e["exp_name"] for e in summary if e.get("status") == "done"}

    combos = list(iter_ablation_grid())
    total  = len(combos)

    for idx, (gnn_type, n_layers, d_model, top_k, n_heads) in enumerate(combos, 1):
        name = exp_dir_name(gnn_type, n_layers, d_model, top_k, n_heads)
        print(f"\n[{idx}/{total}] {name}")

        if skip_flag and name in done_names:
            print(f"  [SKIP] already in summary")
            continue

        entry: dict = {
            "dataset":     dataset_name,
            "exp_name":    name,
            "gnn_type":    gnn_type,
            "n_layers":    n_layers,
            "d_model":     d_model,
            "top_k":       top_k,
            "n_heads":     n_heads,
            "steps":       ABLATION_STEPS,
            "n_gen_seeds": N_GEN_SEEDS,
            "n_clf_seeds": N_CLF_SEEDS,
            "status":      "pending",
        }

        try:
            result = _run_one(
                gnn_type=gnn_type,
                n_layers=n_layers,
                d_model=d_model,
                top_k=top_k,
                n_heads=n_heads,
                dataset_name=dataset_name,
                dataset_cfg=dataset_cfg,
                repo_root=repo_root,
                data_root=data_root,
                exp_root=exp_root,
                device=args.device,
                seed=args.seed,
                skip_if_done=skip_flag,
            )
            entry["status"]  = "done"
            entry["metrics"] = result
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"]  = str(exc)
            traceback.print_exc()

        summary = [e for e in summary if e.get("exp_name") != name]
        summary.append(entry)
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  Summary -> {summary_path}")

    n_done   = sum(1 for e in summary if e.get("status") == "done")
    n_failed = sum(1 for e in summary if e.get("status") == "failed")
    print(f"\nFinished. done={n_done}  failed={n_failed}  summary={summary_path}")
    return summary


# ---------------------------------------------------------------------------
# Shared argument parser
# ---------------------------------------------------------------------------

def make_parser(dataset_name: str = "") -> argparse.ArgumentParser:
    desc = f"Ablation study{f' for {dataset_name}' if dataset_name else ''}."
    p = argparse.ArgumentParser(description=desc)

    grp = p.add_argument_group("Ablation combination (all required for single-job mode)")
    grp.add_argument("--gnn_type", type=str, choices=GNN_TYPES, default=None,
                     metavar="TYPE", help="gcn | gat | gatv2 | gin")
    grp.add_argument("--n_layers", type=int, choices=N_LAYERS,  default=None, metavar="N")
    grp.add_argument("--d_model",  type=int, choices=D_MODELS,  default=None, metavar="D")
    grp.add_argument("--top_k",    type=int, choices=TOP_KS,    default=None, metavar="K",
                     help="sparsity_top_k (0 = dense)")
    grp.add_argument("--n_heads",  type=int, choices=N_HEADS,   default=None, metavar="H",
                     help="Attention heads (ignored by gcn/gin)")

    env = p.add_argument_group("Environment")
    env.add_argument("--repo_root", type=str, default=".",
                     help="Absolute path to repository root")
    env.add_argument("--data_root", type=str, default=None,
                     help="Root directory containing data/ (default: repo_root)")
    env.add_argument("--exp_root",  type=str, default=None,
                     help="Root directory for exp/ outputs (default: repo_root)")
    env.add_argument("--device",    type=str, default="cuda:0")
    env.add_argument("--seed",      type=int, default=0)
    env.add_argument("--no_skip",   action="store_true", default=False,
                     help="Re-run even if results already exist")

    p.add_argument("--aggregate", action="store_true", default=False,
                   help="Build ablation_summary.json from finished per-combo results "
                        "(run this after all SLURM array tasks complete)")

    p.set_defaults(skip_if_done=True)
    return p

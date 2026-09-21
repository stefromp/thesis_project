#!/usr/bin/env python3
"""
Run the full ablation study across all 16 datasets sequentially.

Usage (single node, runs everything one after another):
    python ablation/run_all_ablation.py \
        --repo_root /path/to/repo \
        --device cuda:0

Usage (cluster — submit one SLURM array job per dataset):
    python ablation/run_all_ablation.py --submit_slurm \
        --repo_root /path/to/repo \
        --partition gpu --time 06:00:00

Pass --datasets to restrict to a subset:
    python ablation/run_all_ablation.py \
        --datasets higgs-small miniboone cardio \
        --repo_root /path/to/repo --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import traceback
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ablation_runner import (
    run_ablation,
    iter_ablation_grid,
    aggregate_summary,
    apply_cli_overrides,
    describe_config,
)
from existing_results import (
    DEFAULT_RESULTS_DIR,
    load_manifest,
    load_best,
    load_datasets,
    describe_manifest,
)

# ---------------------------------------------------------------------------
# All dataset configurations (16 TabDDPM-tuned + 4 TabDiff-paper additions)
# ---------------------------------------------------------------------------

ALL_DATASETS: dict = {
    "abalone": {
        "real_data_path":          "data/abalone/",
        "num_numerical_features":  7,
        "num_classes":             0,
        "is_y_cond":               False,
        "rtdl_d_layers":           [512, 128],
        "num_timesteps":           100,
        "scheduler":               "cosine",
        "lr":                      0.0003442062504112041,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             20800,
        "sample_batch_size":       10000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "adult": {
        "real_data_path":          "data/adult/",
        "num_numerical_features":  6,
        "num_classes":             2,
        "is_y_cond":               True,
        "rtdl_d_layers":           [256, 1024, 1024, 1024, 1024, 256],
        "num_timesteps":           100,
        "scheduler":               "cosine",
        "lr":                      0.0020099410620098234,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             216000,
        "sample_batch_size":       10000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    # NOTE: beijing / magic / news / shoppers (the TabDiff-paper datasets added
    # via scripts/process_new_datasets.py) have no TabDDPM-tuned config.
    # rtdl_d_layers/lr are sensible defaults borrowed from similarly sized
    # datasets; the ablation overrides lr/batch_size/num_timesteps anyway.
    # num_samples follows the repo convention of ~8x the train split size.
    "beijing": {
        "real_data_path":          "data/beijing/",
        "num_numerical_features":  10,
        "num_classes":             0,
        "is_y_cond":               False,
        "rtdl_d_layers":           [128, 512, 512, 512, 512, 256],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.001,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             213800,
        "sample_batch_size":       20000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "buddy": {
        "real_data_path":          "data/buddy/",
        "num_numerical_features":  4,
        "num_classes":             3,
        "is_y_cond":               True,
        "rtdl_d_layers":           [1024, 512, 512, 512],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.0004168262823315286,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             48000,
        "sample_batch_size":       10000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "california": {
        "real_data_path":          "data/california/",
        "num_numerical_features":  8,
        "num_classes":             0,
        "is_y_cond":               False,
        "rtdl_d_layers":           [512, 256, 256, 256, 256, 128],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.0013275991211473216,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             52800,
        "sample_batch_size":       8192,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "cardio": {
        "real_data_path":          "data/cardio/",
        "num_numerical_features":  5,
        "num_classes":             2,
        "is_y_cond":               True,
        "rtdl_d_layers":           [512, 1024, 1024, 1024, 1024, 1024],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.0005642428536156398,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             360000,
        "sample_batch_size":       20000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "churn2": {
        "real_data_path":          "data/churn2/",
        "num_numerical_features":  7,
        "num_classes":             2,
        "is_y_cond":               True,
        "rtdl_d_layers":           [512, 1024, 1024, 1024, 1024, 512],
        "num_timesteps":           100,
        "scheduler":               "cosine",
        "lr":                      0.0008057094292475385,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             26000,
        "sample_batch_size":       10000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "default": {
        "real_data_path":          "data/default/",
        # data/default has 14 numerical + 9 categorical columns (see info.json);
        # the previous value of 20 did not match the committed arrays.
        "num_numerical_features":  14,
        "num_classes":             2,
        "is_y_cond":               True,
        "rtdl_d_layers":           [256, 1024, 1024, 1024, 1024, 512],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.00046818967784044777,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             76800,
        "sample_batch_size":       10000,
        "train_normalization":     "minmax",
        "eval_normalization":      "minmax",
    },
    "diabetes": {
        "real_data_path":          "data/diabetes/",
        "num_numerical_features":  8,
        "num_classes":             2,
        "is_y_cond":               True,
        "rtdl_d_layers":           [128, 512],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      1.1510940031144828e-05,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             500,
        "sample_batch_size":       10000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "fb-comments": {
        "real_data_path":          "data/fb-comments/",
        "num_numerical_features":  36,
        "num_classes":             0,
        "is_y_cond":               False,
        "rtdl_d_layers":           [512, 1024],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.0006338119731278414,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             1264000,
        "sample_batch_size":       150000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "gesture": {
        "real_data_path":          "data/gesture/",
        "num_numerical_features":  32,
        "num_classes":             5,
        "is_y_cond":               True,
        "rtdl_d_layers":           [128, 512, 512, 1024],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.002805595109954435,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             52000,
        "sample_batch_size":       10000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "higgs-small": {
        "real_data_path":          "data/higgs-small/",
        "num_numerical_features":  28,
        "num_classes":             2,
        "is_y_cond":               True,
        "rtdl_d_layers":           [256, 1024, 1024, 1024, 1024, 512],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.0010482394930684048,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             502000,
        "sample_batch_size":       60000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "house": {
        "real_data_path":          "data/house/",
        "num_numerical_features":  16,
        "num_classes":             0,
        "is_y_cond":               False,
        "rtdl_d_layers":           [128, 512, 512, 512, 512, 256],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.0013926185951764255,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             116000,
        "sample_batch_size":       30000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "insurance": {
        "real_data_path":          "data/insurance/",
        "num_numerical_features":  3,
        "num_classes":             0,
        "is_y_cond":               False,
        "rtdl_d_layers":           [256, 512, 512, 512, 512, 256],
        "num_timesteps":           100,
        "scheduler":               "cosine",
        "lr":                      0.0011121628249569867,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             7200,
        "sample_batch_size":       10000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "king": {
        "real_data_path":          "data/king/",
        "num_numerical_features":  17,
        "num_classes":             0,
        "is_y_cond":               False,
        "rtdl_d_layers":           [256, 1024, 1024, 1024, 1024, 256],
        "num_timesteps":           100,
        "scheduler":               "cosine",
        "lr":                      0.0007444465590958975,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             27600,
        "sample_batch_size":       20000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "magic": {
        "real_data_path":          "data/magic/",
        "num_numerical_features":  10,
        "num_classes":             2,
        "is_y_cond":               True,
        "rtdl_d_layers":           [512, 1024, 1024, 1024, 1024, 512],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.001,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             97400,
        "sample_batch_size":       10000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "miniboone": {
        "real_data_path":          "data/miniboone/",
        "num_numerical_features":  50,
        "num_classes":             2,
        "is_y_cond":               True,
        "rtdl_d_layers":           [512, 1024, 1024, 1024],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.0023518278056159554,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             664000,
        "sample_batch_size":       20000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "news": {
        "real_data_path":          "data/news/",
        "num_numerical_features":  44,
        "num_classes":             0,
        "is_y_cond":               False,
        "rtdl_d_layers":           [512, 1024],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.001,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             203000,
        "sample_batch_size":       20000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "shoppers": {
        "real_data_path":          "data/shoppers/",
        "num_numerical_features":  10,
        "num_classes":             2,
        "is_y_cond":               True,
        "rtdl_d_layers":           [512, 1024, 1024, 1024, 1024, 512],
        "num_timesteps":           1000,
        "scheduler":               "cosine",
        "lr":                      0.001,
        "weight_decay":            0.0,
        "batch_size":              4096,
        "num_samples":             63200,
        "sample_batch_size":       10000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
    "wilt": {
        "real_data_path":          "data/wilt/",
        "num_numerical_features":  5,
        "num_classes":             2,
        "is_y_cond":               True,
        "rtdl_d_layers":           [1024, 512, 512, 512, 512, 512, 512, 128],
        "num_timesteps":           100,
        "scheduler":               "cosine",
        "lr":                      0.00010707356429215857,
        "weight_decay":            0.0,
        "batch_size":              256,
        "num_samples":             24800,
        "sample_batch_size":       10000,
        "train_normalization":     "quantile",
        "eval_normalization":      None,
    },
}

# ---------------------------------------------------------------------------
# TabDiff-comparable splits
# ---------------------------------------------------------------------------

TABDIFF_SPLIT_SUFFIX = "_tabdiff_split"


def tabdiff_split_config(dataset_name: str, data_root: str = ".") -> dict:
    """Config for running ``dataset_name`` on its TabDiff-comparable split.

    Comparing a number to TabDiff's published tables requires reproducing the
    setup those numbers came from, and three things differ from the TabDDPM
    config in ALL_DATASETS:

      1. the SPLIT — TabDiff partitions train/val differently (adult: 28943 /
         3618 vs TabDDPM's 26048 / 6513). Shape, Trend, C2ST, DCR and
         alpha/beta are all measured against the TRAIN split, so a different
         partition means a different reference distribution;
      2. the SCHEMA — TabDiff types some columns differently (news carries 45
         numerical features here, not the TabDDPM count);
      3. the SAMPLE SIZE — TabDiff generates exactly |train| synthetic rows,
         where this repo defaults to ~8x for ML efficiency.

    This mirrors what notebooks_tabdiff/news-tabdiff.ipynb does by hand, and is
    derived from the split's own info.json so it cannot drift from the data.
    """
    import numpy as np

    base = dataset_name[:-len(TABDIFF_SPLIT_SUFFIX)] \
        if dataset_name.endswith(TABDIFF_SPLIT_SUFFIX) else dataset_name
    split_dir_name = base + TABDIFF_SPLIT_SUFFIX
    split_dir = os.path.join(data_root, "data", split_dir_name)

    if base not in ALL_DATASETS:
        raise SystemExit(f"Unknown dataset: {base}")
    if not os.path.isdir(split_dir):
        raise SystemExit(
            f"No TabDiff split at {split_dir}. Build it with "
            f"scripts/make_tabdiff_splits.py, or drop --tabdiff_split.")

    with open(os.path.join(split_dir, "info.json")) as fh:
        info = json.load(fh)

    cfg = dict(ALL_DATASETS[base])
    cfg["real_data_path"] = f"data/{split_dir_name}/"
    cfg["num_numerical_features"] = int(info["n_num_features"])

    train_size = int(info["train_size"])
    cfg["num_samples"] = train_size          # TabDiff: |synthetic| == |train|
    cfg["sample_batch_size"] = min(cfg["sample_batch_size"], train_size)

    if info["task_type"] == "regression":
        cfg["num_classes"] = 0
        cfg["is_y_cond"] = False
    else:
        y = np.load(os.path.join(split_dir, "y_train.npy"), allow_pickle=True)
        cfg["num_classes"] = int(len(np.unique(y)))
        cfg["is_y_cond"] = True

    return cfg


def resolve_dataset(dataset_name: str, data_root: str = ".") -> dict:
    """Config for a dataset name, TabDiff-split variants included."""
    if dataset_name.endswith(TABDIFF_SPLIT_SUFFIX):
        return tabdiff_split_config(dataset_name, data_root)
    if dataset_name not in ALL_DATASETS:
        raise SystemExit(f"Unknown dataset: {dataset_name}")
    return ALL_DATASETS[dataset_name]


def available_tabdiff_splits(data_root: str = ".") -> list:
    """Base dataset names that have a TabDiff-comparable split on disk."""
    import glob
    found = glob.glob(os.path.join(data_root, "data", "*" + TABDIFF_SPLIT_SUFFIX))
    names = sorted(os.path.basename(p)[:-len(TABDIFF_SPLIT_SUFFIX)] for p in found)
    return [n for n in names if n in ALL_DATASETS]


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

# One array task per grid combination. The task reads its own line from the
# combos file, so adding/removing combinations only changes --array.
#
# `python -u ablation/run_all_ablation.py --dataset X` is used rather than the
# per-dataset wrapper scripts: beijing / magic / news / shoppers have no
# wrapper, and this keeps a single code path for all 20 datasets.
_SLURM_ARRAY_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=abl_{dataset_slug}
#SBATCH --array=1-{n_combos}{throttle}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --partition={partition}
{extra_directives}#SBATCH --output={log_dir}/abl_{dataset_slug}_%A_%a.out
#SBATCH --error={log_dir}/abl_{dataset_slug}_%A_%a.err

set -euo pipefail

REPO="{repo_root}"
cd "$REPO"
{env_setup}
# Unbuffered stdout so the .out file tracks a running job instead of flushing
# everything at exit.
export PYTHONUNBUFFERED=1

COMBO=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{combos_file}")
if [ -z "$COMBO" ]; then
    echo "No combo on line ${{SLURM_ARRAY_TASK_ID}} of {combos_file}" >&2
    exit 1
fi
read -r GNN_TYPE N_LAYERS D_MODEL TOP_K N_HEADS USE_ATTENTION <<< "$COMBO"

echo "host=$(hostname)  task=${{SLURM_ARRAY_TASK_ID}}  combo=$COMBO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u ablation/run_all_ablation.py \\
    --dataset       {dataset_name} \\
    --gnn_type      "$GNN_TYPE" \\
    --n_layers      "$N_LAYERS" \\
    --d_model       "$D_MODEL" \\
    --top_k         "$TOP_K" \\
    --n_heads       "$N_HEADS" \\
    --use_attention "$USE_ATTENTION" \\
    --repo_root     "$REPO" \\
    --data_root     "{data_root}" \\
    --exp_root      "{exp_root}" \\
    --device        {device}{eval_flags}
"""

# Submitted with --dependency=afterany:<array job id>: it must run even if some
# array tasks failed, so the summary records them as "missing" rather than
# blocking the whole collection step.
_SLURM_AGGREGATE_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=agg_{dataset_slug}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --partition={partition}
{extra_directives}#SBATCH --output={log_dir}/agg_{dataset_slug}_%j.out
#SBATCH --error={log_dir}/agg_{dataset_slug}_%j.err

set -euo pipefail

REPO="{repo_root}"
cd "$REPO"
{env_setup}
export PYTHONUNBUFFERED=1

python -u ablation/run_all_ablation.py \\
    --datasets  {dataset_name} \\
    --aggregate --csv \\
    --repo_root "$REPO" \\
    --exp_root  "{exp_root}"{grid_flags}
"""


def _eval_flags(args: argparse.Namespace) -> str:
    """Render the evaluation-protocol flags for the generated batch script.

    Only flags that were actually passed are forwarded, so the runner's own
    defaults stay in charge of everything else.
    """
    parts = []
    if args.evaluator:
        parts.append(f"--evaluator {args.evaluator}")
    if args.tabdiff_fidelity is True:
        parts.append("--tabdiff_fidelity")
    elif args.tabdiff_fidelity is False:
        parts.append("--no_tabdiff_fidelity")
    if args.tabdiff_metrics:
        parts.append("--tabdiff_metrics " + " ".join(args.tabdiff_metrics))
    for flag in ("steps", "n_gen_seeds", "n_clf_seeds"):
        value = getattr(args, flag)
        if value is not None:
            parts.append(f"--{flag} {value}")
    if args.no_skip:
        parts.append("--no_skip")
    return "".join(f" \\\n    {p}" for p in parts)


def _grid_flags(args: argparse.Namespace) -> str:
    """Grid overrides, needed by the aggregate job so it looks for the same
    combinations the array job produced."""
    parts = []
    for flag in ("grid_gnn_types", "grid_n_layers", "grid_d_models",
                 "grid_top_ks", "grid_n_heads", "grid_attentions"):
        values = getattr(args, flag)
        if values:
            parts.append(f"--{flag} " + " ".join(str(v) for v in values))
    return "".join(f" \\\n    {p}" for p in parts)


def _manifest_flags(args: argparse.Namespace) -> str:
    """Forward the manifest selection to the aggregate job, so it sweeps the
    same combination list rather than the default cartesian grid."""
    if args.include_recorded:
        return (f" \\\n    --include_recorded "
                f"{os.path.abspath(args.include_recorded)}")
    if not args.combos_from:
        return ""
    parts = [f"--combos_from {os.path.abspath(args.combos_from)}"]
    if args.best:
        parts.append("--best")
    return "".join(f" \\\n    {p}" for p in parts)


def _env_setup_block(args: argparse.Namespace) -> str:
    """Build the module-load / environment-activation lines.

    --env_setup points at a shell file that is sourced (edit it once, every
    job picks the change up); --modules and --conda_env are shortcuts that
    write the usual two lines inline.
    """
    lines = []
    if args.env_setup:
        lines.append(f'source "{os.path.abspath(args.env_setup)}"')
    for module in args.modules:
        lines.append(f"module load {module}")
    if args.conda_env:
        # `conda activate` needs the shell hook in a non-interactive job.
        lines.append('eval "$(conda shell.bash hook)"')
        lines.append(f"conda activate {args.conda_env}")
    if args.venv:
        lines.append(f'source "{os.path.abspath(args.venv)}/bin/activate"')
    if not lines:
        return ("# No environment setup requested. Pass --env_setup / --modules /\n"
                "# --conda_env / --venv if the compute nodes need it.\n")
    return "\n".join(lines) + "\n"


def _extra_directives(args: argparse.Namespace) -> str:
    lines = []
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    if args.qos:
        lines.append(f"#SBATCH --qos={args.qos}")
    if args.mail:
        lines.append(f"#SBATCH --mail-user={args.mail}")
        lines.append("#SBATCH --mail-type=END,FAIL")
    return "".join(line + "\n" for line in lines)


def _write_combos_file(path: str, combos: Optional[list] = None) -> int:
    """One combination per line: gnn_type n_layers d_model top_k n_heads attn.

    use_attention is written as 0/1 rather than Python's True/False because the
    batch script feeds it straight to `--use_attention`, which takes an int.

    ``combos`` overrides the cartesian grid — the manifest path passes the exact
    combinations recorded in tabddpm_ablation/.
    """
    combos = list(iter_ablation_grid()) if combos is None else list(combos)
    with open(path, "w") as fh:
        for gnn_type, n_layers, d_model, top_k, n_heads, use_attention in combos:
            fh.write(f"{gnn_type} {n_layers} {d_model} {top_k} {n_heads} "
                     f"{int(use_attention)}\n")
    return len(combos)


def _sbatch(script_path: str, dependency: str = "") -> str:
    """Submit and return the job id ('' on failure)."""
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(script_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    sbatch FAILED:\n{result.stderr.strip()}")
        return ""
    # "Submitted batch job 12345"
    return result.stdout.strip().split()[-1]


def _submit_slurm(dataset_name: str, args: argparse.Namespace,
                  repo_root: str, data_root: str, exp_root: str,
                  log_dir: str, combos: Optional[list] = None) -> None:
    slug        = dataset_name.replace("-", "_")
    script_dir  = os.path.join(repo_root, args.script_dir)
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    combos_file  = os.path.join(script_dir, f"combos_{slug}.txt")
    array_script = os.path.join(script_dir, f"submit_{slug}.sh")
    agg_script   = os.path.join(script_dir, f"aggregate_{slug}.sh")

    n_combos = _write_combos_file(combos_file, combos)
    if n_combos == 0:
        print(f"  [{dataset_name}] empty grid — nothing to submit")
        return

    common = dict(
        dataset_slug=slug,
        dataset_name=dataset_name,
        partition=args.partition,
        log_dir=log_dir,
        repo_root=repo_root,
        exp_root=exp_root,
        env_setup=_env_setup_block(args),
        extra_directives=_extra_directives(args),
    )

    with open(array_script, "w") as fh:
        fh.write(_SLURM_ARRAY_TEMPLATE.format(
            n_combos=n_combos,
            throttle=f"%{args.max_parallel}" if args.max_parallel else "",
            gres=args.gres,
            cpus=args.cpus,
            mem=args.mem,
            time=args.time,
            combos_file=combos_file,
            data_root=data_root,
            device=args.device,
            eval_flags=_eval_flags(args),
            **common,
        ))
    os.chmod(array_script, 0o755)

    with open(agg_script, "w") as fh:
        fh.write(_SLURM_AGGREGATE_TEMPLATE.format(
            grid_flags=_manifest_flags(args) or _grid_flags(args), **common))
    os.chmod(agg_script, 0o755)

    if args.dry_run:
        print(f"  [{dataset_name}] {n_combos} combos — scripts written, not submitted:")
        print(f"      {array_script}")
        print(f"      {agg_script}")
        return

    job_id = _sbatch(array_script)
    if not job_id:
        return
    print(f"  [{dataset_name}] array job {job_id} — {n_combos} tasks", end="")

    if args.no_aggregate_job:
        print()
        return
    agg_id = _sbatch(agg_script, dependency=f"afterany:{job_id}")
    print(f"  + aggregate job {agg_id}" if agg_id else "  (aggregate job not submitted)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the GNN ablation study across datasets, locally or on SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sel = p.add_argument_group("Dataset selection")
    sel.add_argument("--datasets", nargs="+", default=None, metavar="NAME",
                     help="Subset of datasets to run (default: all 20)")
    sel.add_argument("--tabdiff_split", action="store_true", default=False,
                     help="Run on the TabDiff-comparable splits in "
                          "data/<ds>_tabdiff_split/ instead of the TabDDPM splits: "
                          "TabDiff's train/val partition, its column schema, and "
                          "|synthetic| == |train|. REQUIRED if the metrics are going "
                          "to be compared against TabDiff's published tables. "
                          "Results land under exp/<ds>_tabdiff_split/ so they are "
                          "never conflated with the TabDDPM-split runs.")
    sel.add_argument("--datasets_from", type=str, default=None, metavar="DIR",
                     nargs="?", const=DEFAULT_RESULTS_DIR,
                     help="Run the datasets that have results in this directory "
                          f"(bare flag = {DEFAULT_RESULTS_DIR}/), sweeping the full "
                          "notebook grid. Use this to EXTEND those datasets; use "
                          "--combos_from to replay only the exact combinations "
                          "already recorded.")
    sel.add_argument("--combos_from", type=str, default=None, metavar="DIR",
                     nargs="?", const=DEFAULT_RESULTS_DIR,
                     help="Replay the exact combinations recorded in this results "
                          f"directory (bare flag = {DEFAULT_RESULTS_DIR}/) instead of "
                          "sweeping the cartesian grid. Selects the datasets too, "
                          "unless --datasets narrows them further.")
    sel.add_argument("--include_recorded", type=str, default=None, metavar="DIR",
                     nargs="?", const=DEFAULT_RESULTS_DIR,
                     help="Sweep the notebook grid AND add any combination in this "
                          f"results directory (bare flag = {DEFAULT_RESULTS_DIR}/) "
                          "that the grid does not contain. The notebooks narrowed "
                          "the grid for some datasets, so 17 of the 88 recorded "
                          "runs (wilt's heads8, adult/buddy's dim128/256, gesture's "
                          "gat) sit outside it and would otherwise never get the "
                          "TabDiff metrics.")
    sel.add_argument("--best", action="store_true", default=False,
                     help="With --combos_from, keep only the best-scoring "
                          "combination per dataset (ranked on metrics.score). "
                          "Nine jobs — the recommended first run.")
    sel.add_argument("--dataset", type=str, default=None, metavar="NAME",
                     help="Single-combination mode (used by the SLURM array "
                          "script): run one dataset with the --gnn_type ... "
                          "combination given below")

    combo = p.add_argument_group("Ablation combination (single-combination mode)")
    combo.add_argument("--gnn_type", type=str, default=None, metavar="TYPE")
    combo.add_argument("--n_layers", type=int, default=None, metavar="N")
    combo.add_argument("--d_model",  type=int, default=None, metavar="D")
    combo.add_argument("--top_k",    type=int, default=None, metavar="K")
    combo.add_argument("--n_heads",  type=int, default=None, metavar="H")
    combo.add_argument("--use_attention", type=int, choices=[0, 1], default=None,
                       metavar="A")

    env = p.add_argument_group("Environment")
    env.add_argument("--repo_root", type=str, default=".",
                     help="Path to repository root")
    env.add_argument("--data_root", type=str, default=None,
                     help="Path containing data/ (default: repo_root). On HPC this "
                          "is often a fast scratch filesystem.")
    env.add_argument("--exp_root",  type=str, default=None,
                     help="Path for exp/ outputs (default: repo_root)")
    env.add_argument("--device",    type=str, default="cuda:0")
    env.add_argument("--seed",      type=int, default=0)
    env.add_argument("--no_skip",   action="store_true", default=False,
                     help="Re-run even if results already exist")

    ev = p.add_argument_group("Evaluation protocol")
    ev.add_argument("--evaluator", type=str, choices=["catboost", "xgboost"],
                    default=None,
                    help="ML-efficiency model (default: the runner's catboost)")
    ev.add_argument("--tabdiff_fidelity", dest="tabdiff_fidelity",
                    action="store_const", const=True, default=None,
                    help="Also compute the TabDiff-paper suite: Shape, Trend "
                         "(pair-wise feature correlations), C2ST, TabDiff DCR, "
                         "alpha-precision / beta-recall")
    ev.add_argument("--no_tabdiff_fidelity", dest="tabdiff_fidelity",
                    action="store_const", const=False)
    ev.add_argument("--tabdiff_metrics", nargs="+", default=None, metavar="M",
                    choices=["density", "c2st", "dcr", "quality"])
    ev.add_argument("--steps",       type=int, default=None, metavar="S")
    ev.add_argument("--n_gen_seeds", type=int, default=None, metavar="G")
    ev.add_argument("--n_clf_seeds", type=int, default=None, metavar="C")
    ev.add_argument("--smoke", action="store_true", default=False,
                    help="Plumbing check: 200 steps, 1 generation seed, 1 classifier "
                         "seed. Verifies training -> sampling -> metrics end to end "
                         "in minutes. Numbers are meaningless; do not report them.")

    gr = p.add_argument_group("Grid overrides")
    gr.add_argument("--grid_gnn_types", nargs="+", type=str, default=None, metavar="T")
    gr.add_argument("--grid_n_layers",  nargs="+", type=int, default=None, metavar="N")
    gr.add_argument("--grid_d_models",  nargs="+", type=int, default=None, metavar="D")
    gr.add_argument("--grid_top_ks",    nargs="+", type=int, default=None, metavar="K")
    gr.add_argument("--grid_n_heads",   nargs="+", type=int, default=None, metavar="H")
    gr.add_argument("--grid_attentions", nargs="+", type=int, choices=[0, 1],
                    default=None, metavar="A")

    agg = p.add_argument_group("Aggregation")
    agg.add_argument("--aggregate", action="store_true", default=False,
                     help="Build ablation_summary.json for each dataset from finished "
                          "per-combo results. Run after all array tasks complete.")
    agg.add_argument("--csv", action="store_true", default=False,
                     help="With --aggregate, also write a flat ablation_summary.csv")

    sl = p.add_argument_group("SLURM submission")
    sl.add_argument("--submit_slurm", action="store_true", default=False,
                    help="Submit one SLURM array job per dataset")
    sl.add_argument("--dry_run", action="store_true", default=False,
                    help="Write the batch scripts but do not call sbatch")
    sl.add_argument("--partition", type=str, default="gpu")
    sl.add_argument("--time",      type=str, default="12:00:00",
                    help="Wall-clock limit PER COMBINATION (one array task)")
    sl.add_argument("--gres",      type=str, default="gpu:1")
    sl.add_argument("--cpus",      type=int, default=8)
    sl.add_argument("--mem",       type=str, default="48G")
    sl.add_argument("--max_parallel", type=int, default=None, metavar="N",
                    help="Cap concurrently running array tasks (--array=1-M%%N)")
    sl.add_argument("--account",   type=str, default=None)
    sl.add_argument("--qos",       type=str, default=None)
    sl.add_argument("--mail",      type=str, default=None,
                    help="Email for END/FAIL notifications")
    sl.add_argument("--log_dir",   type=str, default="slurm_logs")
    sl.add_argument("--script_dir", type=str, default="slurm_scripts",
                    help="Where generated combos/submit scripts are written")
    sl.add_argument("--no_aggregate_job", action="store_true", default=False,
                    help="Do not chain an aggregation job after the array job")

    envg = p.add_argument_group("Compute-node environment (for generated scripts)")
    envg.add_argument("--env_setup", type=str, default=None, metavar="FILE",
                      help="Shell file sourced at the top of every job "
                           "(see ablation/hpc/env.sh.example)")
    envg.add_argument("--modules", nargs="+", default=[], metavar="M",
                      help="`module load` these, e.g. --modules cuda/11.8 python/3.11")
    envg.add_argument("--conda_env", type=str, default=None, metavar="NAME",
                      help="`conda activate` this environment")
    envg.add_argument("--venv", type=str, default=None, metavar="DIR",
                      help="Activate this virtualenv directory")

    return p


def _resolve_manifest(args: argparse.Namespace) -> Optional[dict]:
    """dataset -> list of combinations, or None when sweeping the grid."""
    if not args.combos_from:
        return None

    loader = load_best if args.best else load_manifest
    manifest = loader(args.combos_from)
    if not manifest:
        raise SystemExit(f"No usable results found in {args.combos_from}/")

    if args.datasets:
        unknown = set(args.datasets) - set(manifest)
        if unknown:
            raise SystemExit(
                f"Not in the manifest: {sorted(unknown)}. "
                f"Available: {sorted(manifest)}")
        manifest = {d: manifest[d] for d in args.datasets}

    return manifest


def _apply_smoke(args: argparse.Namespace) -> None:
    """Shrink the protocol to a plumbing check, without silently overriding
    anything the caller set explicitly."""
    if not args.smoke:
        return
    if args.steps is None:
        args.steps = 200
    if args.n_gen_seeds is None:
        args.n_gen_seeds = 1
    if args.n_clf_seeds is None:
        args.n_clf_seeds = 1
    print("[SMOKE] plumbing check — "
          f"steps={args.steps} gen_seeds={args.n_gen_seeds} "
          f"clf_seeds={args.n_clf_seeds}. Metric values are NOT publishable.\n")


def main() -> None:
    p = _build_parser()
    args = p.parse_args()

    _apply_smoke(args)

    # Grid overrides must take effect before anything enumerates combinations
    # (the combos file, the run counts, the aggregation sweep).
    apply_cli_overrides(args)

    repo_root = os.path.abspath(args.repo_root)
    data_root = os.path.abspath(args.data_root if args.data_root else repo_root)
    exp_root  = os.path.abspath(args.exp_root  if args.exp_root  else repo_root)
    log_dir   = os.path.join(repo_root, args.log_dir)

    # ---- Single-combination mode (one SLURM array task) --------------------
    if args.dataset:
        name = args.dataset
        if args.tabdiff_split and not name.endswith(TABDIFF_SPLIT_SUFFIX):
            name += TABDIFF_SPLIT_SUFFIX
        base = name[:-len(TABDIFF_SPLIT_SUFFIX)] if name.endswith(TABDIFF_SPLIT_SUFFIX) else name
        if base not in ALL_DATASETS:
            p.error(f"Unknown dataset: {base}. Choose from: {list(ALL_DATASETS)}")
        missing = [n for n in ("gnn_type", "n_layers", "d_model", "top_k", "n_heads")
                   if getattr(args, n) is None]
        if missing:
            p.error("--dataset needs the full combination; missing: "
                    + ", ".join("--" + m for m in missing))

        runner_args = argparse.Namespace(**vars(args), skip_if_done=True)
        runner_args.aggregate = False
        combo = (args.gnn_type, args.n_layers, args.d_model, args.top_k,
                 args.n_heads,
                 True if args.use_attention is None else bool(args.use_attention))
        run_ablation(name, resolve_dataset(name, data_root), runner_args,
                     single_combo=combo)
        return

    manifest = _resolve_manifest(args)

    if manifest is not None:
        datasets_to_run = list(manifest)
    elif args.datasets:
        datasets_to_run = args.datasets
    elif args.datasets_from:
        datasets_to_run = load_datasets(args.datasets_from)
        if not datasets_to_run:
            p.error(f"No results found in {args.datasets_from}/")
    elif args.tabdiff_split:
        datasets_to_run = available_tabdiff_splits(data_root)
    else:
        datasets_to_run = list(ALL_DATASETS.keys())

    if args.tabdiff_split:
        have = set(available_tabdiff_splits(data_root))
        missing_split = [d for d in datasets_to_run if d not in have]
        if missing_split:
            p.error(f"No TabDiff split for {missing_split}. "
                    f"Available: {sorted(have)}. TabDiff's paper only covers "
                    f"those datasets — the rest have no published numbers to "
                    f"compare against.")
        datasets_to_run = [d + TABDIFF_SPLIT_SUFFIX for d in datasets_to_run]

    unknown = [d for d in datasets_to_run
               if (d[:-len(TABDIFF_SPLIT_SUFFIX)] if d.endswith(TABDIFF_SPLIT_SUFFIX)
                   else d) not in ALL_DATASETS]
    if unknown:
        p.error(f"Unknown dataset(s): {sorted(unknown)}. "
                f"Choose from: {list(ALL_DATASETS)}")

    # The grid plus whatever the notebooks ran outside it. Built here (rather
    # than in _resolve_manifest) because it needs the resolved dataset list, and
    # before the aggregation branch so the summary covers the union too.
    if manifest is None and args.include_recorded:
        recorded = load_manifest(args.include_recorded, status=None)
        grid = list(iter_ablation_grid())
        in_grid = set(grid)
        manifest = {
            ds: grid + [c for c in recorded.get(ds, []) if c not in in_grid]
            for ds in datasets_to_run
        }

    # ---- Aggregation -------------------------------------------------------
    if args.aggregate:
        for ds in datasets_to_run:
            aggregate_summary(ds, args,
                              combos=manifest[ds] if manifest else None)
        print("\nAggregation complete.")
        return

    if manifest is not None:
        total = sum(len(c) for c in manifest.values())
        print(describe_manifest(manifest))
    else:
        total = len(datasets_to_run) * len(list(iter_ablation_grid()))
        print(f"Datasets      : {len(datasets_to_run)}  {datasets_to_run}")
        print(f"Combos each   : {len(list(iter_ablation_grid()))}")

    print(f"Total runs    : {total}")
    print(f"Mode          : {'SLURM array jobs' if args.submit_slurm else 'sequential'}")
    print(describe_config(show_grid=manifest is None))
    print()

    # ---- SLURM submission --------------------------------------------------
    if args.submit_slurm:
        for ds in datasets_to_run:
            _submit_slurm(ds, args, repo_root=repo_root, data_root=data_root,
                          exp_root=exp_root, log_dir=log_dir,
                          combos=manifest[ds] if manifest else None)
        print(f"\nLogs    -> {log_dir}/")
        print(f"Scripts -> {os.path.join(repo_root, args.script_dir)}/")
        return

    # ---- Sequential (login node / single workstation) -----------------------
    runner_args = argparse.Namespace(**vars(args), skip_if_done=True)
    runner_args.aggregate = False

    for i, ds in enumerate(datasets_to_run, 1):
        print(f"\n{'#' * 72}")
        print(f"# Dataset {i}/{len(datasets_to_run)}: {ds}")
        print(f"{'#' * 72}")
        if manifest is None:
            run_ablation(ds, resolve_dataset(ds, data_root), runner_args,
                         single_combo=None)
            continue
        # Manifest mode has no cartesian grid to sweep, so drive the exact
        # combinations one at a time and collect them at the end.
        for j, combo in enumerate(manifest[ds], 1):
            print(f"\n[{ds}] combination {j}/{len(manifest[ds])}: {combo}")
            try:
                run_ablation(ds, resolve_dataset(ds, data_root), runner_args,
                             single_combo=combo)
            except Exception:
                traceback.print_exc()
        aggregate_summary(ds, runner_args, combos=manifest[ds])

    print("\nAll datasets complete.")


if __name__ == "__main__":
    main()

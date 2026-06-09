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
import os
import sys
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ablation_runner import make_parser, run_ablation, iter_ablation_grid, aggregate_summary

# ---------------------------------------------------------------------------
# All 16 dataset configurations
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
        "num_numerical_features":  20,
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
# SLURM helpers
# ---------------------------------------------------------------------------

_SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=abl_{dataset_slug}
#SBATCH --array=1-{n_combos}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --output={log_dir}/abl_{dataset_slug}_%a.out
#SBATCH --error={log_dir}/abl_{dataset_slug}_%a.err

REPO={repo_root}
COMBO=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {combos_file})
read GNN_TYPE N_LAYERS D_MODEL TOP_K N_HEADS <<< $COMBO

python ablation/{script_name} \\
    --gnn_type $GNN_TYPE \\
    --n_layers $N_LAYERS \\
    --d_model  $D_MODEL  \\
    --top_k    $TOP_K    \\
    --n_heads  $N_HEADS  \\
    --repo_root {repo_root} \\
    --exp_root  {exp_root} \\
    --device    cuda:0
"""


def _write_combos_file(path: str) -> int:
    combos = list(iter_ablation_grid())
    with open(path, "w") as fh:
        for combo in combos:
            fh.write(" ".join(str(x) for x in combo) + "\n")
    return len(combos)


def _submit_slurm(dataset_name: str, repo_root: str, exp_root: str,
                  partition: str, time_limit: str, log_dir: str) -> None:
    slug        = dataset_name.replace("-", "_")
    script_name = f"{slug}_ablation.py"
    combos_file = os.path.join(repo_root, f"combos_{slug}.txt")
    slurm_file  = os.path.join(repo_root, f"submit_{slug}.sh")

    n_combos = _write_combos_file(combos_file)

    os.makedirs(log_dir, exist_ok=True)
    script_body = _SLURM_TEMPLATE.format(
        dataset_slug=slug,
        n_combos=n_combos,
        time=time_limit,
        partition=partition,
        log_dir=log_dir,
        repo_root=repo_root,
        exp_root=exp_root,
        combos_file=combos_file,
        script_name=script_name,
    )
    with open(slurm_file, "w") as fh:
        fh.write(script_body)

    result = subprocess.run(["sbatch", slurm_file], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  [{dataset_name}] submitted {n_combos} jobs  — {result.stdout.strip()}")
    else:
        print(f"  [{dataset_name}] sbatch FAILED:\n{result.stderr}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run ablation study for all 16 datasets."
    )
    p.add_argument("--datasets", nargs="+", default=None,
                   metavar="NAME",
                   help="Subset of datasets to run (default: all 16)")
    p.add_argument("--repo_root", type=str, default=".",
                   help="Path to repository root")
    p.add_argument("--data_root", type=str, default=None,
                   help="Path containing data/ folder (default: repo_root)")
    p.add_argument("--exp_root",  type=str, default=None,
                   help="Path for exp/ outputs (default: repo_root)")
    p.add_argument("--device",    type=str, default="cuda:0")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--no_skip",   action="store_true", default=False,
                   help="Re-run even if results already exist")

    # Aggregation mode (run after SLURM array jobs finish)
    p.add_argument("--aggregate", action="store_true", default=False,
                   help="Build ablation_summary.json for each dataset from finished "
                        "per-combo results_full_averaged.json files. Run this after "
                        "all SLURM array tasks complete.")

    # SLURM mode
    p.add_argument("--submit_slurm", action="store_true", default=False,
                   help="Submit one SLURM array job per dataset instead of running sequentially")
    p.add_argument("--partition", type=str, default="gpu",
                   help="SLURM partition name (default: gpu)")
    p.add_argument("--time",      type=str, default="06:00:00",
                   help="SLURM time limit per job (default: 06:00:00)")
    p.add_argument("--log_dir",   type=str, default="slurm_logs",
                   help="Directory for SLURM .out/.err files (default: slurm_logs)")

    args = p.parse_args()

    datasets_to_run = args.datasets if args.datasets else list(ALL_DATASETS.keys())
    unknown = set(datasets_to_run) - set(ALL_DATASETS)
    if unknown:
        p.error(f"Unknown dataset(s): {unknown}. "
                f"Choose from: {list(ALL_DATASETS)}")

    repo_root = os.path.abspath(args.repo_root)
    exp_root  = os.path.abspath(args.exp_root  if args.exp_root  else repo_root)
    log_dir   = os.path.join(repo_root, args.log_dir)

    n = len(datasets_to_run)
    combos_per_ds = len(list(iter_ablation_grid()))
    print(f"Datasets      : {n}")
    print(f"Combos each   : {combos_per_ds}")
    print(f"Total runs    : {n * combos_per_ds}")
    print(f"Mode          : {'SLURM array jobs' if args.submit_slurm else 'sequential'}")
    print()

    if args.aggregate:
        agg_args = argparse.Namespace(repo_root=args.repo_root, exp_root=args.exp_root)
        for ds in datasets_to_run:
            aggregate_summary(ds, agg_args)
        print("\nAggregation complete.")
        return

    if args.submit_slurm:
        for ds in datasets_to_run:
            _submit_slurm(
                dataset_name=ds,
                repo_root=repo_root,
                exp_root=exp_root,
                partition=args.partition,
                time_limit=args.time,
                log_dir=log_dir,
            )
        print(f"\nAll jobs submitted. Logs -> {log_dir}/")
        return

    # Sequential mode — build a shared argparse Namespace and call run_ablation
    runner_args = argparse.Namespace(
        repo_root=args.repo_root,
        data_root=args.data_root,
        exp_root=args.exp_root,
        device=args.device,
        seed=args.seed,
        skip_if_done=True,
        no_skip=args.no_skip,
    )

    for i, ds in enumerate(datasets_to_run, 1):
        print(f"\n{'#' * 72}")
        print(f"# Dataset {i}/{n}: {ds}")
        print(f"{'#' * 72}")
        run_ablation(ds, ALL_DATASETS[ds], runner_args, single_combo=None)

    print("\nAll datasets complete.")


if __name__ == "__main__":
    main()

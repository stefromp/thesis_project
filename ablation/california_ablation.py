#!/usr/bin/env python3
"""
Ablation study for CALIFORNIA dataset.

Varies: gnn_type, n_layers, d_model (hidden dim), top_k (sparsity), n_heads.
Fixes : all CB-best hyperparameters (lr, batch_size, diffusion params, etc.).
Steps : 20 000  (no intermediate checkpoints).
Eval  : CatBoost on synthetic data (identical to the default notebook).

Results are saved in:
  {exp_root}/exp/california/ablation/{exp_name}/results_catboost.json

A consolidated summary across all runs lives in:
  {exp_root}/exp/california/ablation/ablation_summary.json

Usage — single combination (cluster job):
    python ablation/california_ablation.py \
        --gnn_type gcn --n_layers 3 --d_model 256 --top_k 3 --n_heads 4 \
        --repo_root /path/to/repo --device cuda:0

Usage — run full grid sequentially (local or single-node cluster):
    python ablation/california_ablation.py \
        --repo_root /path/to/repo --device cuda:0
"""

from __future__ import annotations

import os
import sys

# Make ablation_runner importable whether the script is run from repo root
# or from within the ablation/ directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ablation_runner import run_ablation, make_parser  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset configuration — derived from exp/california/ddpm_cb_best/config.toml
# ---------------------------------------------------------------------------
DATASET_NAME = "california"

DATASET_CFG: dict = {
    # Paths
    "real_data_path":          "data/california/",
    # Model
    "num_numerical_features":  8,
    "num_classes":             0,
    "is_y_cond":               False,
    "rtdl_d_layers":           [512, 256, 256, 256, 256, 128],
    # Diffusion
    "num_timesteps":           1000,
    "scheduler":               "cosine",
    # Optimiser
    "lr":                      0.0013275991211473216,
    "weight_decay":            0.0,
    "batch_size":              4096,
    # Sampling
    "num_samples":             52800,
    "sample_batch_size":       8192,
    # Transformations
    "train_normalization":     "quantile",
    "eval_normalization":      None,
}


if __name__ == "__main__":
    args = make_parser(DATASET_NAME).parse_args()

    # Single-combination mode when all five ablation parameters are supplied.
    single_combo = None
    _all_set = all(
        v is not None
        for v in [args.gnn_type, args.n_layers, args.d_model, args.top_k, args.n_heads]
    )
    if _all_set:
        single_combo = (args.gnn_type, args.n_layers, args.d_model,
                        args.top_k, args.n_heads)

    run_ablation(DATASET_NAME, DATASET_CFG, args, single_combo=single_combo)

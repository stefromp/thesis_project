#!/usr/bin/env bash
#
# Submission lines for the categorical-input fix reruns (CAT_INPUT=prob).
#
# NOTHING HERE IS EXECUTED. Every sbatch line is commented out on purpose.
#
# 24 jobs = 8 datasets with categorical columns x 3 rules, one configuration
# per job, same configuration / protocol / resources as the original
# hpc_transfer jobs. Results go to *_catfix sibling trees:
#
#   fixed              exp/<ds>/ablation_catfix/
#   adaptive_jenks     exp/<ds>/ablation_adaptive_jenks_catfix/
#   adaptive_entmax15  exp/<ds>/ablation_adaptive_entmax15_catfix/
#
# Upload first (the preflight in common.sh refuses older code):
#   tab_ddpm/graph_denoiser.py  scripts/utils_train.py  ablation/ablation_runner.py
#   hpc_transfer/common.sh  hpc_transfer/split_runner.py  hpc_transfer/jobs_catfix/
#
# The fixed runs of the development datasets (abalone ... insurance) also give
# the per-seed fixed baseline they never had; for those five datasets the
# ablation_fixed_seed0 jobs in analysis/topk_comparison/hpc_fixed_perseed/
# are superseded (they would reproduce the log-input model).
#
# fb-comments (15 categorical columns) is NOT here: it is still running with
# cat_input=log and is affected too; decide separately.

cd "$(dirname "$0")/jobs_catfix" || exit 1

# ---- priority 1: cardio, default, king --------------------
# sbatch cardio_fixed_catfix.sh
# sbatch cardio_jenks_catfix.sh
# sbatch cardio_entmax_catfix.sh
# sbatch default_fixed_catfix.sh
# sbatch default_jenks_catfix.sh
# sbatch default_entmax_catfix.sh
# sbatch king_fixed_catfix.sh
# sbatch king_jenks_catfix.sh
# sbatch king_entmax_catfix.sh

# ---- priority 2: abalone, adult, buddy, churn2, insurance --------------------
# sbatch abalone_fixed_catfix.sh
# sbatch abalone_jenks_catfix.sh
# sbatch abalone_entmax_catfix.sh
# sbatch adult_fixed_catfix.sh
# sbatch adult_jenks_catfix.sh
# sbatch adult_entmax_catfix.sh
# sbatch buddy_fixed_catfix.sh
# sbatch buddy_jenks_catfix.sh
# sbatch buddy_entmax_catfix.sh
# sbatch churn2_fixed_catfix.sh
# sbatch churn2_jenks_catfix.sh
# sbatch churn2_entmax_catfix.sh
# sbatch insurance_fixed_catfix.sh
# sbatch insurance_jenks_catfix.sh
# sbatch insurance_entmax_catfix.sh

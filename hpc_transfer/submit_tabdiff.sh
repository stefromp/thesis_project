#!/usr/bin/env bash
#
# Submission loop for the TabDiff-transfer study. NOTHING HERE IS EXECUTED:
# every sbatch line is commented out on purpose. One job per dataset (train ->
# 20 tables -> TabDiff evaluation -> _dcr model -> 20 tables -> DCR), cheapest
# first so a problem shows up early. A resubmitted job skips finished stages.
# Regenerate jobs + this file: python3 hpc_transfer/tabdiff_study/make_jobs.py
# Before the first submission run the cluster tests in
# hpc_transfer/tabdiff_study/README.md (venvs, preflight, dry run).

cd "$(dirname "$0")/jobs_tabdiff" || exit 1

# magic: 2 h walltime (estimate 0.7-1.1 GPU-h)
# sbatch magic.sh
# shoppers: 2 h walltime (estimate 1.2-1.4 GPU-h)
# sbatch shoppers.sh
# adult: 3 h walltime (estimate 1.6-1.6 GPU-h)
# sbatch adult.sh
# default: 14 h walltime (estimate 10.2-10.2 GPU-h)
# sbatch default.sh
# beijing: 5 h walltime (estimate 3.5-3.5 GPU-h)
# sbatch beijing.sh
# news: 31 h walltime (estimate 10.8-23.4 GPU-h)
# sbatch news.sh

# total: ~41.2 GPU-h (pessimistic, before the x1.3 walltime margin)

# Analysis afterwards (rerunnable): python3 analysis/tabdiff/compare.py

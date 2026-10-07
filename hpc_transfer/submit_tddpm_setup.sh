#!/usr/bin/env bash
#
# Submission lines for the BUDGET-MATCHED runs (TRAIN_SETUP=tabddpm, fixed rule).
#
# NOTHING HERE IS EXECUTED. Every sbatch line is commented out on purpose.
#
# Same configuration, cat_input, seeds, EMA, sampling size and CatBoost config
# as the run the current analysis uses for each dataset; lr, steps, batch size,
# weight decay, num_timesteps, scheduler and sampling batch come from the
# authors' exp/<ds>/ddpm_cb_best/config.toml (ablation_runner.training_setup).
# Results go to *_tddpm_setup[_catfix] sibling trees:
#   cardio, king (categorical)            exp/<ds>/ablation_tddpm_setup_catfix/
#   house, gesture, california, higgs-small, miniboone   exp/<ds>/ablation_tddpm_setup/
# Walltimes: hpc_transfer/jobs_tddpm_setup/walltime_estimates.csv
# (regenerate the jobs with: python3 hpc_transfer/make_tddpm_setup_jobs.py).
# Analysis afterwards: python3 analysis/tddpm_setup/compare_setup.py
#
# default is excluded on purpose: the local data/default is not the authors'
# (20 num + 3 cat, val/test swapped -- see analysis/config_comparison).

cd "$(dirname "$0")/jobs_tddpm_setup" || exit 1

# ---- the five target datasets (one job each: train + 5 seeds + aggregate) ----
# cheapest first, so a divergence or setup problem shows up within 2 h
# sbatch california_fixed_tddpm.sh     #  2 h
# sbatch king_fixed_tddpm.sh           #  5 h   (num_timesteps 100)
# sbatch house_fixed_tddpm.sh          #  6 h
# sbatch gesture_fixed_tddpm.sh        #  6 h
# sbatch cardio_fixed_tddpm.sh         #  8 h

# ---- expensive, decide separately ----
# sbatch higgs-small_fixed_tddpm.sh    # 38 h, single job
# miniboone does not fit 48 h as one job: split chain (train-only, then one
# job per seed, then merge), printed first and submitted only with --submit:
# TIME=21:00:00 TRAIN_ONLY=1 ../submit_split.sh miniboone_fixed_tddpm.sh
# TIME=21:00:00 TRAIN_ONLY=1 ../submit_split.sh miniboone_fixed_tddpm.sh --submit

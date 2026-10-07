#!/usr/bin/env bash
#
# Submission lines for hpc_transfer.
#
# NOTHING HERE IS EXECUTED. Every sbatch line is commented out on purpose.
#
# 39 job scripts, 39 runs, one configuration per job. What varies across a
# dataset's jobs is ONLY the k-selection rule.
#
#   GROUP A -- the 9 ABLATED datasets, 2 adaptive rules = 18 runs
#     <ds>_jenks.sh     adaptive_jenks
#     <ds>_entmax.sh    adaptive_entmax15
#     Each runs that dataset's OWN ablation winner (existing_results.load_best(),
#     highest metrics.score.mean). No fixed run: that baseline is already in
#     tabddpm_ablation/, and each job keeps the winner's own top_k, so the
#     output directory name matches the baseline exactly.
#
#   GROUP B -- the 7 NEW datasets, 3 rules = 21 runs
#     <ds>_transfer.sh        fixed              (the baseline they never had)
#     <ds>_transfer_jenks.sh  adaptive_jenks
#     <ds>_entmax.sh          adaptive_entmax15
#     No ablation results exist for these, so the configuration is the one
#     predicted in predictions.md.
#
# Rules and their result trees (ablation_runner derives each from --topk_mode,
# so the rules of one configuration can never collide):
#
#   fixed              exp/<ds>/ablation/
#   adaptive_jenks     exp/<ds>/ablation_adaptive_jenks/
#   adaptive_entmax15  exp/<ds>/ablation_adaptive_entmax15/
#
# sparsemax (alpha 2.0) is NOT part of this study.
#
# !! entmax-1.5 needs a package that is NOT in requirements.txt !!
#     pip install entmax==1.3
#   common.sh preflights it (import entmax, --topk_mode accepted, one
#   DynamicAdjacency forward) before creating anything, so a missing dependency
#   costs no allocation. Check it once interactively first:
#     ./hpc_transfer/smoke_test.sh wilt adaptive_entmax15
#
# !! OVERLAP WITH THE STUDIES THAT HAVE ALREADY RUN !!
#   hpc/        adaptive_jenks,    every combo of 7 ablated datasets
#   hpc_entmax/ entmax15+sparsemax, every combo of the same 7
#   Those 7 are: abalone buddy churn2 diabetes gesture insurance wilt.
#   For them, GROUP A adds nothing on entmax-1.5 -- the results exist.
#   It DOES add something on Jenks if hpc/ ran before the plain-Jenks fix
#   (commit 3b8b8b3) reached the cluster, because gated and plain Jenks give
#   different numbers. adult and california are covered by neither study.
#   Check what you already have before submitting:
#     for d in abalone buddy churn2 diabetes gesture insurance wilt adult california; do
#       for m in adaptive_jenks adaptive_entmax15; do
#         f=$(ls -d "$REPO_ROOT/exp/$d/ablation_$m"/*/results_full_averaged.json 2>/dev/null | wc -l)
#         echo "$d $m $f"
#       done
#     done
#
# Before anything:
#     sinfo -p gpu -o "%N %G"
#     export GRES=gpu:1
#
# Estimates are per job from hpc_transfer/resource_estimates.py. The rule
# changes cost by under 3 % (the adjacency rules are a small part of the
# traffic), so one row covers both rules of a dataset -- but that is 2 or 3
# jobs each, not one. entmax-1.5 is the heavier rule on memory, and the
# measured peaks quoted there are already entmax-1.5 figures.
#
# ---------------------------------------------------------------------------
# GROUP A -- 9 ablated datasets x 2 rules = 18 runs
#
#   dataset      config          N   synth rows   per job, opt - pess  fits 24h
#   diabetes     gin_L2_d32       8       500     1.0 -    1.0 h   yes
#   wilt         gatv2_L2_d64     5    24,800     1.0 -    1.0 h   yes
#   insurance    gatv2_L2_d64     7     7,200     1.0 -    1.1 h   yes
#   abalone      gatv2_L3_d64     9    20,800     1.1 -    1.2 h   yes
#   churn2       gatv2_L2_d128   11    26,000     1.2 -    1.4 h   yes
#   buddy        gcn_L2_d256      9    48,000     1.2 -    1.6 h   yes
#   california   gcn_L2_d256      9    52,800     1.3 -    1.7 h   yes
#   adult        gcn_L2_d128     14   216,000     2.3 -    3.4 h   yes
#   gesture      gat_L3_d128     32    52,000     3.5 -    7.1 h   yes
#
# ---------------------------------------------------------------------------
# GROUP B -- 7 new datasets x 3 rules = 21 runs
#
#   dataset      config          N   synth rows   per job, opt - pess  fits 24h
#   king         gatv2_L3_d128   21    27,600     1.8 -    3.1 h   yes
#   default      gatv2_L3_d128   23    76,800     2.5 -    4.8 h   yes
#   cardio       gatv2_L3_d128   11   360,000     2.5 -    4.1 h   yes
#   house        gatv2_L3_d128   17   116,000     2.6 -    4.2 h   yes
#   higgs-small  gatv2_L3_d128   28   502,000    10.6 -   24.2 h   marginal
#   miniboone    gatv2_L3_d128   50   664,000    40.5 -   97.5 h   NO
#   fb-comments  gatv2_L3_d128   52 1,264,000    76.7 -  188.0 h   NO
#
# CALIBRATE ON diabetes OR wilt (GROUP A, ~1 h), then king (GROUP B). If king
# runs much over ~3 h, every estimate scales up and higgs-small stops being
# marginal.
# ---------------------------------------------------------------------------

set -euo pipefail
echo "This script does not submit anything. Open it and uncomment what you want." >&2
exit 0

# === GROUP A: 9 ablated datasets ==========================================
# --- adaptive_jenks ---
# sbatch --gres="${GRES}" hpc_transfer/jobs/diabetes_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/wilt_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/insurance_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/abalone_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/churn2_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/buddy_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/california_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/adult_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/gesture_jenks.sh
# --- adaptive_entmax15 ---
# sbatch --gres="${GRES}" hpc_transfer/jobs/diabetes_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/wilt_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/insurance_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/abalone_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/churn2_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/buddy_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/california_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/adult_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/gesture_entmax.sh

# === GROUP B: 7 new datasets ==============================================
# Only the four that fit are listed here. higgs-small is marginal and
# miniboone / fb-comments do not fit -- see the bottom of this file. That
# applies to EVERY rule, so it is 3 and 6 jobs you cannot run as-is.
# --- fixed (baseline) ---
# sbatch --gres="${GRES}" hpc_transfer/jobs/king_transfer.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/default_transfer.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/cardio_transfer.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/house_transfer.sh
# --- adaptive_jenks ---
# sbatch --gres="${GRES}" hpc_transfer/jobs/king_transfer_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/default_transfer_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/cardio_transfer_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/house_transfer_jenks.sh
# --- adaptive_entmax15 ---
# sbatch --gres="${GRES}" hpc_transfer/jobs/king_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/default_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/cardio_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/house_entmax.sh

# === marginal: fits optimistically, over the limit pessimistically =========
# ONE configuration per job and no intermediate checkpoints
# (checkpoint_every=0), so a walltime kill loses the entire run. Submit only
# after king has given you a measured scaling factor.
# sbatch --gres="${GRES}" hpc_transfer/jobs/higgs-small_transfer.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/higgs-small_transfer_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/higgs-small_entmax.sh

# === do NOT submit as-is: 1.7x and 4x the walltime limit ==================
# Sampling dominates (rows x 1000 diffusion steps x 5 generation seeds) and is
# the same cost in every rule, so all 6 of these miss.
# Options, cheapest first:
#   1. --n_gen_seeds 2 instead of 5: miniboone ~17-40 h, fb-comments ~31-76 h.
#      Only fb-comments still misses. Error bars widen, no longer 5-seed
#      comparable.
#   2. Cap the synthetic rows as hpc_wide/ does (BUDGET_MAX_NUM_SAMPLES=50000,
#      applied in memory by hpc_wide/run_combo.py). The metrics are then on a
#      capped synthetic set and NOT comparable to tabddpm_ablation/.
#   3. A longer walltime on a partition that allows it, if one exists.
# Neither 1 nor 2 is applied by default: the protocol was chosen to be directly
# comparable to the ablation.
#
# sbatch --gres="${GRES}" hpc_transfer/jobs/miniboone_transfer.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/miniboone_transfer_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/miniboone_entmax.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/fb-comments_transfer.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/fb-comments_transfer_jenks.sh
# sbatch --gres="${GRES}" hpc_transfer/jobs/fb-comments_entmax.sh

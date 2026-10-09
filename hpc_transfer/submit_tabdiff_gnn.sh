#!/usr/bin/env bash
#
# Submission lines for Route A (hpc_transfer/jobs_tabdiff_gnn/): AttnGNN as the
# denoiser inside TabDiff's framework.
#
# NOTHING HERE IS EXECUTED. Every sbatch line is commented out on purpose.
#
# One job script per dataset, <ds>_tabdiff_gnn.sh: train -> 20 tables -> TabDiff's
# evaluation (+ MLE-2) -> alpha/beta -> wNMIS / C2ST(XGB). One configuration: AttnGNN V2,
# d_model 256, GNN type / layers / top_k / heads from hpc_transfer/configs/<source>.sh.
# No TabDiff (transformer) runs -- TabDiff's numbers are the reported ones -- and no DCR.
#
# news trains with the resumable copy of TabDiff's Trainer.run_loop (its 8000 epochs do
# not fit one 48 h job): news_tabdiff_gnn_train.sh is submitted once per training part,
# each part saves its run state before its budget and exits 0, the next resumes; then
# news_tabdiff_gnn_sample_eval.sh. If training needs one more part than planned, the
# sample_eval job refuses to start ("stage 'train' is not done"): submit the train
# script once more, then sample_eval again.
#
# The training seed is SEED (default 0). Seeds 1-2 are double-commented (##): submit
# them only after confirmation; they get their own job name and logs.
#
# Estimates per seed (hours on one A100, before the x1.3 walltime margin):
#
#   dataset   config            N  n_train   estimate   walltime
#   magic     gatv2_L2_d256     11   17,117     3.2 h   48 h
#   shoppers  gcn_L2_d256       18   11,097     2.6 h   48 h
#   adult     gcn_L2_d256       15   32,561     4.2 h   48 h
#   beijing   gatv2_L3_d256     12   37,581     6.9 h   48 h
#   default   gatv2_L3_d256     24   27,000    12.5 h   48 h
#   news      gatv2_L3_d256     48   35,679    55.9 h   2x 48 h + 48 h
#   diabetes  gcn_L2_d256       37   61,059    15.3 h   48 h
#
# Re-size from a measured A100 time (timing probe, tabdiff_gnn/README.md):
#     python3 tabdiff_gnn/make_jobs.py --probe news=<s/epoch>
#
# Before anything:
#     sinfo -p gpu -o "%N %G"
#     export GRES=gpu:1
#     OUT=/work/pr019/atvoulo/Rompos/jobs_out
# ---------------------------------------------------------------------------

set -euo pipefail
echo "This script does not submit anything. Open it and uncomment what you want." >&2
exit 0

# === block 1: AttnGNN d256 seed 0: magic, shoppers, adult, beijing ========
# --- magic (~3.2 GPU-h)
# sbatch --gres="${GRES}" hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn.sh
# --- shoppers (~2.6 GPU-h)
# sbatch --gres="${GRES}" hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn.sh
# --- adult (~4.2 GPU-h)
# sbatch --gres="${GRES}" hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn.sh
# --- beijing (~6.9 GPU-h)
# sbatch --gres="${GRES}" hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn.sh

# === block 2: AttnGNN d256 seed 0: default, news (resumable 3-job chain), diabetes ===
# --- default (~12.5 GPU-h)
# sbatch --gres="${GRES}" hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn.sh
# --- news (~55.9 GPU-h)
# jid=$(sbatch --parsable --gres="${GRES}" hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
# jid=$(sbatch --parsable --gres="${GRES}" --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_train_p2 --output=${OUT}/news_tabdiff_gnn_train_p2.out --error=${OUT}/news_tabdiff_gnn_train_p2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
# jid=$(sbatch --parsable --gres="${GRES}" --dependency=afterok:${jid} hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_sample_eval.sh)
# --- diabetes (~15.3 GPU-h)
# sbatch --gres="${GRES}" hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn.sh

# === ABLATION (NOT the main protocol; seed 0; compare with the main run) ========
# --- beijing topk_dense: top_k = N - 1 (~6.9 GPU-h)
# sbatch --gres="${GRES}" hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn_ablation_topk_dense.sh

# === training seed 1 -- ONLY AFTER CONFIRMATION ==============================
# --- magic seed 1 (~3.2 GPU-h)
## SEED=1 sbatch --gres="${GRES}" --job-name=magic_tabdiff_gnn_s1 --output=${OUT}/magic_tabdiff_gnn_s1.out --error=${OUT}/magic_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn.sh
# --- shoppers seed 1 (~2.6 GPU-h)
## SEED=1 sbatch --gres="${GRES}" --job-name=shoppers_tabdiff_gnn_s1 --output=${OUT}/shoppers_tabdiff_gnn_s1.out --error=${OUT}/shoppers_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn.sh
# --- adult seed 1 (~4.2 GPU-h)
## SEED=1 sbatch --gres="${GRES}" --job-name=adult_tabdiff_gnn_s1 --output=${OUT}/adult_tabdiff_gnn_s1.out --error=${OUT}/adult_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn.sh
# --- beijing seed 1 (~6.9 GPU-h)
## SEED=1 sbatch --gres="${GRES}" --job-name=beijing_tabdiff_gnn_s1 --output=${OUT}/beijing_tabdiff_gnn_s1.out --error=${OUT}/beijing_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn.sh
# --- default seed 1 (~12.5 GPU-h)
## SEED=1 sbatch --gres="${GRES}" --job-name=default_tabdiff_gnn_s1 --output=${OUT}/default_tabdiff_gnn_s1.out --error=${OUT}/default_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn.sh
# --- news seed 1 (~55.9 GPU-h)
## jid=$(SEED=1 sbatch --parsable --gres="${GRES}" --job-name=news_tabdiff_gnn_train_s1 --output=${OUT}/news_tabdiff_gnn_train_s1.out --error=${OUT}/news_tabdiff_gnn_train_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
## jid=$(SEED=1 sbatch --parsable --gres="${GRES}" --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_train_p2_s1 --output=${OUT}/news_tabdiff_gnn_train_p2_s1.out --error=${OUT}/news_tabdiff_gnn_train_p2_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
## jid=$(SEED=1 sbatch --parsable --gres="${GRES}" --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_sample_eval_s1 --output=${OUT}/news_tabdiff_gnn_sample_eval_s1.out --error=${OUT}/news_tabdiff_gnn_sample_eval_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_sample_eval.sh)
# --- diabetes seed 1 (~15.3 GPU-h)
## SEED=1 sbatch --gres="${GRES}" --job-name=diabetes_tabdiff_gnn_s1 --output=${OUT}/diabetes_tabdiff_gnn_s1.out --error=${OUT}/diabetes_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn.sh

# === training seed 2 -- ONLY AFTER CONFIRMATION ==============================
# --- magic seed 2 (~3.2 GPU-h)
## SEED=2 sbatch --gres="${GRES}" --job-name=magic_tabdiff_gnn_s2 --output=${OUT}/magic_tabdiff_gnn_s2.out --error=${OUT}/magic_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn.sh
# --- shoppers seed 2 (~2.6 GPU-h)
## SEED=2 sbatch --gres="${GRES}" --job-name=shoppers_tabdiff_gnn_s2 --output=${OUT}/shoppers_tabdiff_gnn_s2.out --error=${OUT}/shoppers_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn.sh
# --- adult seed 2 (~4.2 GPU-h)
## SEED=2 sbatch --gres="${GRES}" --job-name=adult_tabdiff_gnn_s2 --output=${OUT}/adult_tabdiff_gnn_s2.out --error=${OUT}/adult_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn.sh
# --- beijing seed 2 (~6.9 GPU-h)
## SEED=2 sbatch --gres="${GRES}" --job-name=beijing_tabdiff_gnn_s2 --output=${OUT}/beijing_tabdiff_gnn_s2.out --error=${OUT}/beijing_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn.sh
# --- default seed 2 (~12.5 GPU-h)
## SEED=2 sbatch --gres="${GRES}" --job-name=default_tabdiff_gnn_s2 --output=${OUT}/default_tabdiff_gnn_s2.out --error=${OUT}/default_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn.sh
# --- news seed 2 (~55.9 GPU-h)
## jid=$(SEED=2 sbatch --parsable --gres="${GRES}" --job-name=news_tabdiff_gnn_train_s2 --output=${OUT}/news_tabdiff_gnn_train_s2.out --error=${OUT}/news_tabdiff_gnn_train_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
## jid=$(SEED=2 sbatch --parsable --gres="${GRES}" --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_train_p2_s2 --output=${OUT}/news_tabdiff_gnn_train_p2_s2.out --error=${OUT}/news_tabdiff_gnn_train_p2_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
## jid=$(SEED=2 sbatch --parsable --gres="${GRES}" --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_sample_eval_s2 --output=${OUT}/news_tabdiff_gnn_sample_eval_s2.out --error=${OUT}/news_tabdiff_gnn_sample_eval_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_sample_eval.sh)
# --- diabetes seed 2 (~15.3 GPU-h)
## SEED=2 sbatch --gres="${GRES}" --job-name=diabetes_tabdiff_gnn_s2 --output=${OUT}/diabetes_tabdiff_gnn_s2.out --error=${OUT}/diabetes_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn.sh

# GPU-h (estimate, before the x1.3 margin): seed 0 100.6; ablation 6.9; seeds 1-2 201.2 more
# Analysis afterwards: python3 analysis/tabdiff/compare.py --route_a

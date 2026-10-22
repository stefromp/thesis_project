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
# Rule arms (adaptive_jenks, adaptive_entmax15): same as the fixed arm except the
# neighbour selection; scripts <ds>_tabdiff_gnn_<jenks|entmax15>.sh, results ..._<rule>_...
#
# Every sbatch line carries what the gpu partition requires: --gres=gpu:a100:1 --mem=24G
# and a realistic --time (estimate x1.3, min 2 h, cap 48 h; 48 h for a resumable
# training part); command-line flags override the scripts' #SBATCH directives.
#
# Estimates per seed (hours on one A100, before the x1.3 walltime margin):
#
#   rule     dataset   config            N  n_train   estimate   --time
#   fixed    magic     gatv2_L2_d256     11   17,117     3.2 h   5 h
#   fixed    shoppers  gcn_L2_d256       18   11,097     2.6 h   4 h
#   fixed    adult     gcn_L2_d256       15   32,561     4.2 h   6 h
#   fixed    beijing   gatv2_L3_d256     12   37,581     6.9 h   10 h
#   fixed    default   gatv2_L3_d256     24   27,000    12.5 h   17 h
#   fixed    news      gatv2_L3_d256     48   35,679    55.9 h   2x 48 h + 5 h
#   fixed    diabetes  gcn_L2_d256       37   61,059    15.3 h   20 h
#   jenks    beijing   gatv2_L3_d256     12   37,581     7.0 h   10 h
#   jenks    magic     gatv2_L2_d256     11   17,117     3.2 h   5 h
#   jenks    shoppers  gcn_L2_d256       18   11,097     2.6 h   4 h
#   jenks    adult     gcn_L2_d256       15   32,561     4.2 h   6 h
#   jenks    default   gatv2_L3_d256     24   27,000    12.4 h   17 h
#   jenks    diabetes  gcn_L2_d256       37   61,059    15.6 h   21 h
#   jenks    news      gatv2_L3_d256     48   35,679    56.3 h   2x 48 h + 5 h
#   entmax15 beijing   gatv2_L3_d256     12   37,581     7.0 h   10 h
#   entmax15 magic     gatv2_L2_d256     11   17,117     3.2 h   5 h
#   entmax15 shoppers  gcn_L2_d256       18   11,097     2.6 h   4 h
#   entmax15 adult     gcn_L2_d256       15   32,561     4.2 h   6 h
#   entmax15 default   gatv2_L3_d256     24   27,000    12.5 h   17 h
#   entmax15 diabetes  gcn_L2_d256       37   61,059    15.8 h   21 h
#   entmax15 news      gatv2_L3_d256     48   35,679    58.5 h   2x 48 h + 5 h
#
# Re-size from a measured A100 time (timing probe, tabdiff_gnn/README.md):
#     python3 tabdiff_gnn/make_jobs.py --probe news=<s/epoch>
#
# Before anything (seed 1-2 lines use it for their own logs):
#     OUT=/work/pr019/atvoulo/Rompos/jobs_out
# ---------------------------------------------------------------------------

set -euo pipefail
echo "This script does not submit anything. Open it and uncomment what you want." >&2
exit 0

# === block 1: AttnGNN d256 seed 0: magic, shoppers, adult, beijing ========
# --- magic (~3.2 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=5:00:00 hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn.sh
# --- shoppers (~2.6 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=4:00:00 hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn.sh
# --- adult (~4.2 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=6:00:00 hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn.sh
# --- beijing (~6.9 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=10:00:00 hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn.sh

# === block 2: AttnGNN d256 seed 0: default, news (resumable 3-job chain), diabetes ===
# --- default (~12.5 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=17:00:00 hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn.sh
# --- news (~55.9 GPU-h)
# jid=$(sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
# jid=$(sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_train_p2 --output=${OUT}/news_tabdiff_gnn_train_p2.out --error=${OUT}/news_tabdiff_gnn_train_p2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
# jid=$(sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=5:00:00 --dependency=afterok:${jid} hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_sample_eval.sh)
# --- diabetes (~15.3 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=20:00:00 hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn.sh

# === block 3: AttnGNN d256 + adaptive_jenks, seed 0 =======================
# --- beijing (~7.0 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=10:00:00 hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn_jenks.sh
# --- magic (~3.2 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=5:00:00 hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn_jenks.sh
# --- shoppers (~2.6 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=4:00:00 hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn_jenks.sh
# --- adult (~4.2 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=6:00:00 hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn_jenks.sh
# --- default (~12.4 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=17:00:00 hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn_jenks.sh
# --- diabetes (~15.6 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=21:00:00 hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn_jenks.sh
# --- news (~56.3 GPU-h)
# jid=$(sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_jenks_train.sh)
# jid=$(sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_jenks_train_p2 --output=${OUT}/news_tabdiff_gnn_jenks_train_p2.out --error=${OUT}/news_tabdiff_gnn_jenks_train_p2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_jenks_train.sh)
# jid=$(sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=5:00:00 --dependency=afterok:${jid} hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_jenks_sample_eval.sh)

# === block 4: AttnGNN d256 + adaptive_entmax15, seed 0 ====================
# --- beijing (~7.0 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=10:00:00 hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn_entmax15.sh
# --- magic (~3.2 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=5:00:00 hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn_entmax15.sh
# --- shoppers (~2.6 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=4:00:00 hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn_entmax15.sh
# --- adult (~4.2 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=6:00:00 hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn_entmax15.sh
# --- default (~12.5 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=17:00:00 hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn_entmax15.sh
# --- diabetes (~15.8 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=21:00:00 hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn_entmax15.sh
# --- news (~58.5 GPU-h)
# jid=$(sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_entmax15_train.sh)
# jid=$(sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_entmax15_train_p2 --output=${OUT}/news_tabdiff_gnn_entmax15_train_p2.out --error=${OUT}/news_tabdiff_gnn_entmax15_train_p2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_entmax15_train.sh)
# jid=$(sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=5:00:00 --dependency=afterok:${jid} hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_entmax15_sample_eval.sh)

# === ABLATION (NOT the main protocol; seed 0; compare with the main run) ========
# --- beijing topk_dense: top_k = N - 1 (~6.9 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=10:00:00 hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn_ablation_topk_dense.sh
# --- magic topk_dense: top_k = N - 1 (~3.2 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=05:00:00 hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn_ablation_topk_dense.sh
# --- magic depth3 fixed: n_layers = 3 (~4.6 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=06:00:00 hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn_ablation_depth3_topk3.sh
# --- magic depth3 jenks: n_layers = 3 (~4.6 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=06:00:00 hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn_ablation_depth3_jenks.sh
# --- magic depth3 entmax15: n_layers = 3 (~4.6 GPU-h)
# sbatch --gres=gpu:a100:1 --mem=24G --time=06:00:00 hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn_ablation_depth3_entmax15.sh

# === training seed 1 -- ONLY AFTER CONFIRMATION ==============================
# --- magic fixed seed 1 (~3.2 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=5:00:00 --job-name=magic_tabdiff_gnn_s1 --output=${OUT}/magic_tabdiff_gnn_s1.out --error=${OUT}/magic_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn.sh
# --- shoppers fixed seed 1 (~2.6 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=4:00:00 --job-name=shoppers_tabdiff_gnn_s1 --output=${OUT}/shoppers_tabdiff_gnn_s1.out --error=${OUT}/shoppers_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn.sh
# --- adult fixed seed 1 (~4.2 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=6:00:00 --job-name=adult_tabdiff_gnn_s1 --output=${OUT}/adult_tabdiff_gnn_s1.out --error=${OUT}/adult_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn.sh
# --- beijing fixed seed 1 (~6.9 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=10:00:00 --job-name=beijing_tabdiff_gnn_s1 --output=${OUT}/beijing_tabdiff_gnn_s1.out --error=${OUT}/beijing_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn.sh
# --- default fixed seed 1 (~12.5 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=17:00:00 --job-name=default_tabdiff_gnn_s1 --output=${OUT}/default_tabdiff_gnn_s1.out --error=${OUT}/default_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn.sh
# --- news fixed seed 1 (~55.9 GPU-h)
## jid=$(SEED=1 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --job-name=news_tabdiff_gnn_train_s1 --output=${OUT}/news_tabdiff_gnn_train_s1.out --error=${OUT}/news_tabdiff_gnn_train_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
## jid=$(SEED=1 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_train_p2_s1 --output=${OUT}/news_tabdiff_gnn_train_p2_s1.out --error=${OUT}/news_tabdiff_gnn_train_p2_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
## jid=$(SEED=1 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=5:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_sample_eval_s1 --output=${OUT}/news_tabdiff_gnn_sample_eval_s1.out --error=${OUT}/news_tabdiff_gnn_sample_eval_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_sample_eval.sh)
# --- diabetes fixed seed 1 (~15.3 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=20:00:00 --job-name=diabetes_tabdiff_gnn_s1 --output=${OUT}/diabetes_tabdiff_gnn_s1.out --error=${OUT}/diabetes_tabdiff_gnn_s1.err hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn.sh
# --- beijing jenks seed 1 (~7.0 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=10:00:00 --job-name=beijing_tabdiff_gnn_jenks_s1 --output=${OUT}/beijing_tabdiff_gnn_jenks_s1.out --error=${OUT}/beijing_tabdiff_gnn_jenks_s1.err hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn_jenks.sh
# --- magic jenks seed 1 (~3.2 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=5:00:00 --job-name=magic_tabdiff_gnn_jenks_s1 --output=${OUT}/magic_tabdiff_gnn_jenks_s1.out --error=${OUT}/magic_tabdiff_gnn_jenks_s1.err hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn_jenks.sh
# --- shoppers jenks seed 1 (~2.6 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=4:00:00 --job-name=shoppers_tabdiff_gnn_jenks_s1 --output=${OUT}/shoppers_tabdiff_gnn_jenks_s1.out --error=${OUT}/shoppers_tabdiff_gnn_jenks_s1.err hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn_jenks.sh
# --- adult jenks seed 1 (~4.2 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=6:00:00 --job-name=adult_tabdiff_gnn_jenks_s1 --output=${OUT}/adult_tabdiff_gnn_jenks_s1.out --error=${OUT}/adult_tabdiff_gnn_jenks_s1.err hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn_jenks.sh
# --- default jenks seed 1 (~12.4 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=17:00:00 --job-name=default_tabdiff_gnn_jenks_s1 --output=${OUT}/default_tabdiff_gnn_jenks_s1.out --error=${OUT}/default_tabdiff_gnn_jenks_s1.err hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn_jenks.sh
# --- diabetes jenks seed 1 (~15.6 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=21:00:00 --job-name=diabetes_tabdiff_gnn_jenks_s1 --output=${OUT}/diabetes_tabdiff_gnn_jenks_s1.out --error=${OUT}/diabetes_tabdiff_gnn_jenks_s1.err hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn_jenks.sh
# --- news jenks seed 1 (~56.3 GPU-h)
## jid=$(SEED=1 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --job-name=news_tabdiff_gnn_jenks_train_s1 --output=${OUT}/news_tabdiff_gnn_jenks_train_s1.out --error=${OUT}/news_tabdiff_gnn_jenks_train_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_jenks_train.sh)
## jid=$(SEED=1 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_jenks_train_p2_s1 --output=${OUT}/news_tabdiff_gnn_jenks_train_p2_s1.out --error=${OUT}/news_tabdiff_gnn_jenks_train_p2_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_jenks_train.sh)
## jid=$(SEED=1 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=5:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_jenks_sample_eval_s1 --output=${OUT}/news_tabdiff_gnn_jenks_sample_eval_s1.out --error=${OUT}/news_tabdiff_gnn_jenks_sample_eval_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_jenks_sample_eval.sh)
# --- beijing entmax15 seed 1 (~7.0 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=10:00:00 --job-name=beijing_tabdiff_gnn_entmax15_s1 --output=${OUT}/beijing_tabdiff_gnn_entmax15_s1.out --error=${OUT}/beijing_tabdiff_gnn_entmax15_s1.err hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn_entmax15.sh
# --- magic entmax15 seed 1 (~3.2 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=5:00:00 --job-name=magic_tabdiff_gnn_entmax15_s1 --output=${OUT}/magic_tabdiff_gnn_entmax15_s1.out --error=${OUT}/magic_tabdiff_gnn_entmax15_s1.err hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn_entmax15.sh
# --- shoppers entmax15 seed 1 (~2.6 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=4:00:00 --job-name=shoppers_tabdiff_gnn_entmax15_s1 --output=${OUT}/shoppers_tabdiff_gnn_entmax15_s1.out --error=${OUT}/shoppers_tabdiff_gnn_entmax15_s1.err hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn_entmax15.sh
# --- adult entmax15 seed 1 (~4.2 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=6:00:00 --job-name=adult_tabdiff_gnn_entmax15_s1 --output=${OUT}/adult_tabdiff_gnn_entmax15_s1.out --error=${OUT}/adult_tabdiff_gnn_entmax15_s1.err hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn_entmax15.sh
# --- default entmax15 seed 1 (~12.5 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=17:00:00 --job-name=default_tabdiff_gnn_entmax15_s1 --output=${OUT}/default_tabdiff_gnn_entmax15_s1.out --error=${OUT}/default_tabdiff_gnn_entmax15_s1.err hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn_entmax15.sh
# --- diabetes entmax15 seed 1 (~15.8 GPU-h)
## SEED=1 sbatch --gres=gpu:a100:1 --mem=24G --time=21:00:00 --job-name=diabetes_tabdiff_gnn_entmax15_s1 --output=${OUT}/diabetes_tabdiff_gnn_entmax15_s1.out --error=${OUT}/diabetes_tabdiff_gnn_entmax15_s1.err hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn_entmax15.sh
# --- news entmax15 seed 1 (~58.5 GPU-h)
## jid=$(SEED=1 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --job-name=news_tabdiff_gnn_entmax15_train_s1 --output=${OUT}/news_tabdiff_gnn_entmax15_train_s1.out --error=${OUT}/news_tabdiff_gnn_entmax15_train_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_entmax15_train.sh)
## jid=$(SEED=1 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_entmax15_train_p2_s1 --output=${OUT}/news_tabdiff_gnn_entmax15_train_p2_s1.out --error=${OUT}/news_tabdiff_gnn_entmax15_train_p2_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_entmax15_train.sh)
## jid=$(SEED=1 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=5:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_entmax15_sample_eval_s1 --output=${OUT}/news_tabdiff_gnn_entmax15_sample_eval_s1.out --error=${OUT}/news_tabdiff_gnn_entmax15_sample_eval_s1.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_entmax15_sample_eval.sh)

# === training seed 2 -- ONLY AFTER CONFIRMATION ==============================
# --- magic fixed seed 2 (~3.2 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=5:00:00 --job-name=magic_tabdiff_gnn_s2 --output=${OUT}/magic_tabdiff_gnn_s2.out --error=${OUT}/magic_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn.sh
# --- shoppers fixed seed 2 (~2.6 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=4:00:00 --job-name=shoppers_tabdiff_gnn_s2 --output=${OUT}/shoppers_tabdiff_gnn_s2.out --error=${OUT}/shoppers_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn.sh
# --- adult fixed seed 2 (~4.2 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=6:00:00 --job-name=adult_tabdiff_gnn_s2 --output=${OUT}/adult_tabdiff_gnn_s2.out --error=${OUT}/adult_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn.sh
# --- beijing fixed seed 2 (~6.9 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=10:00:00 --job-name=beijing_tabdiff_gnn_s2 --output=${OUT}/beijing_tabdiff_gnn_s2.out --error=${OUT}/beijing_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn.sh
# --- default fixed seed 2 (~12.5 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=17:00:00 --job-name=default_tabdiff_gnn_s2 --output=${OUT}/default_tabdiff_gnn_s2.out --error=${OUT}/default_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn.sh
# --- news fixed seed 2 (~55.9 GPU-h)
## jid=$(SEED=2 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --job-name=news_tabdiff_gnn_train_s2 --output=${OUT}/news_tabdiff_gnn_train_s2.out --error=${OUT}/news_tabdiff_gnn_train_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
## jid=$(SEED=2 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_train_p2_s2 --output=${OUT}/news_tabdiff_gnn_train_p2_s2.out --error=${OUT}/news_tabdiff_gnn_train_p2_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_train.sh)
## jid=$(SEED=2 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=5:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_sample_eval_s2 --output=${OUT}/news_tabdiff_gnn_sample_eval_s2.out --error=${OUT}/news_tabdiff_gnn_sample_eval_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_sample_eval.sh)
# --- diabetes fixed seed 2 (~15.3 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=20:00:00 --job-name=diabetes_tabdiff_gnn_s2 --output=${OUT}/diabetes_tabdiff_gnn_s2.out --error=${OUT}/diabetes_tabdiff_gnn_s2.err hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn.sh
# --- beijing jenks seed 2 (~7.0 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=10:00:00 --job-name=beijing_tabdiff_gnn_jenks_s2 --output=${OUT}/beijing_tabdiff_gnn_jenks_s2.out --error=${OUT}/beijing_tabdiff_gnn_jenks_s2.err hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn_jenks.sh
# --- magic jenks seed 2 (~3.2 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=5:00:00 --job-name=magic_tabdiff_gnn_jenks_s2 --output=${OUT}/magic_tabdiff_gnn_jenks_s2.out --error=${OUT}/magic_tabdiff_gnn_jenks_s2.err hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn_jenks.sh
# --- shoppers jenks seed 2 (~2.6 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=4:00:00 --job-name=shoppers_tabdiff_gnn_jenks_s2 --output=${OUT}/shoppers_tabdiff_gnn_jenks_s2.out --error=${OUT}/shoppers_tabdiff_gnn_jenks_s2.err hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn_jenks.sh
# --- adult jenks seed 2 (~4.2 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=6:00:00 --job-name=adult_tabdiff_gnn_jenks_s2 --output=${OUT}/adult_tabdiff_gnn_jenks_s2.out --error=${OUT}/adult_tabdiff_gnn_jenks_s2.err hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn_jenks.sh
# --- default jenks seed 2 (~12.4 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=17:00:00 --job-name=default_tabdiff_gnn_jenks_s2 --output=${OUT}/default_tabdiff_gnn_jenks_s2.out --error=${OUT}/default_tabdiff_gnn_jenks_s2.err hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn_jenks.sh
# --- diabetes jenks seed 2 (~15.6 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=21:00:00 --job-name=diabetes_tabdiff_gnn_jenks_s2 --output=${OUT}/diabetes_tabdiff_gnn_jenks_s2.out --error=${OUT}/diabetes_tabdiff_gnn_jenks_s2.err hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn_jenks.sh
# --- news jenks seed 2 (~56.3 GPU-h)
## jid=$(SEED=2 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --job-name=news_tabdiff_gnn_jenks_train_s2 --output=${OUT}/news_tabdiff_gnn_jenks_train_s2.out --error=${OUT}/news_tabdiff_gnn_jenks_train_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_jenks_train.sh)
## jid=$(SEED=2 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_jenks_train_p2_s2 --output=${OUT}/news_tabdiff_gnn_jenks_train_p2_s2.out --error=${OUT}/news_tabdiff_gnn_jenks_train_p2_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_jenks_train.sh)
## jid=$(SEED=2 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=5:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_jenks_sample_eval_s2 --output=${OUT}/news_tabdiff_gnn_jenks_sample_eval_s2.out --error=${OUT}/news_tabdiff_gnn_jenks_sample_eval_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_jenks_sample_eval.sh)
# --- beijing entmax15 seed 2 (~7.0 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=10:00:00 --job-name=beijing_tabdiff_gnn_entmax15_s2 --output=${OUT}/beijing_tabdiff_gnn_entmax15_s2.out --error=${OUT}/beijing_tabdiff_gnn_entmax15_s2.err hpc_transfer/jobs_tabdiff_gnn/beijing_tabdiff_gnn_entmax15.sh
# --- magic entmax15 seed 2 (~3.2 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=5:00:00 --job-name=magic_tabdiff_gnn_entmax15_s2 --output=${OUT}/magic_tabdiff_gnn_entmax15_s2.out --error=${OUT}/magic_tabdiff_gnn_entmax15_s2.err hpc_transfer/jobs_tabdiff_gnn/magic_tabdiff_gnn_entmax15.sh
# --- shoppers entmax15 seed 2 (~2.6 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=4:00:00 --job-name=shoppers_tabdiff_gnn_entmax15_s2 --output=${OUT}/shoppers_tabdiff_gnn_entmax15_s2.out --error=${OUT}/shoppers_tabdiff_gnn_entmax15_s2.err hpc_transfer/jobs_tabdiff_gnn/shoppers_tabdiff_gnn_entmax15.sh
# --- adult entmax15 seed 2 (~4.2 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=6:00:00 --job-name=adult_tabdiff_gnn_entmax15_s2 --output=${OUT}/adult_tabdiff_gnn_entmax15_s2.out --error=${OUT}/adult_tabdiff_gnn_entmax15_s2.err hpc_transfer/jobs_tabdiff_gnn/adult_tabdiff_gnn_entmax15.sh
# --- default entmax15 seed 2 (~12.5 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=17:00:00 --job-name=default_tabdiff_gnn_entmax15_s2 --output=${OUT}/default_tabdiff_gnn_entmax15_s2.out --error=${OUT}/default_tabdiff_gnn_entmax15_s2.err hpc_transfer/jobs_tabdiff_gnn/default_tabdiff_gnn_entmax15.sh
# --- diabetes entmax15 seed 2 (~15.8 GPU-h)
## SEED=2 sbatch --gres=gpu:a100:1 --mem=24G --time=21:00:00 --job-name=diabetes_tabdiff_gnn_entmax15_s2 --output=${OUT}/diabetes_tabdiff_gnn_entmax15_s2.out --error=${OUT}/diabetes_tabdiff_gnn_entmax15_s2.err hpc_transfer/jobs_tabdiff_gnn/diabetes_tabdiff_gnn_entmax15.sh
# --- news entmax15 seed 2 (~58.5 GPU-h)
## jid=$(SEED=2 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --job-name=news_tabdiff_gnn_entmax15_train_s2 --output=${OUT}/news_tabdiff_gnn_entmax15_train_s2.out --error=${OUT}/news_tabdiff_gnn_entmax15_train_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_entmax15_train.sh)
## jid=$(SEED=2 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=48:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_entmax15_train_p2_s2 --output=${OUT}/news_tabdiff_gnn_entmax15_train_p2_s2.out --error=${OUT}/news_tabdiff_gnn_entmax15_train_p2_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_entmax15_train.sh)
## jid=$(SEED=2 sbatch --parsable --gres=gpu:a100:1 --mem=24G --time=5:00:00 --dependency=afterok:${jid} --job-name=news_tabdiff_gnn_entmax15_sample_eval_s2 --output=${OUT}/news_tabdiff_gnn_entmax15_sample_eval_s2.out --error=${OUT}/news_tabdiff_gnn_entmax15_sample_eval_s2.err hpc_transfer/jobs_tabdiff_gnn/news_tabdiff_gnn_entmax15_sample_eval.sh)

# GPU-h (estimate, before the x1.3 margin): seed 0 305.7 (fixed 100.6, rules 205.1); ablation 23.8; seeds 1-2 611.3 more
# Analysis afterwards: python3 analysis/tabdiff/compare.py --route_a

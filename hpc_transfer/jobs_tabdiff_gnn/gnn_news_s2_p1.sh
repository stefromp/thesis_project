#!/bin/bash -l
#SBATCH --job-name=tdg_gnn_news_s2_p1
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/tdg_gnn_news_s2_p1.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/tdg_gnn_news_s2_p1.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
##SBATCH --account=pr019022_gpu

# ROUTE A (TabDiff framework, only the denoiser differs) -- news (regression), arm gnn:
#   denoiser attngnn V2: gatv2 L3 d256 top-k 5 (fixed) heads 8, dynamic graph, no global attention, dim_t 128, shared noise embedding; GNN source king
#   training seed 2; 20 tables per model (generation seeds 0..19)
# Stages of this job (each leaves <run>/stages/<stage>.done; resubmitting skips them):
#   train (52.79 h)
# CHAIN part 1/3 (resumable training: pauses + saves its run state before 46.5 h, the next part resumes): submit as sbatch gnn_news_s2_p1.sh
# Protocol  : TabDiff tabdiff_configs.toml unchanged (8000 epochs, batch 4096, lr 1e-3 +
#             ReduceLROnPlateau, EMA 0.997, best EMA loss after epoch 4000, 50 sampling steps);
#             routine eval every 2000 epochs off (no-effect deviation)
# Trainer   : resumable copy of TabDiff's Trainer.run_loop (news only; evidence tests/test_tabdiff_gnn.py)
# Evaluation: TabDiff's evaluator (digests checked) + MLE-2 + alpha/beta + tabpc wNMIS / C2ST(XGB);
#             no DCR in this study
# Data      : data/tabdiff_gnn/tabdiff_news/ (TabDiff's process_dataset.py @ 5ecdb33)
# Results   : $TRANSFER_ROOT/exp_tabdiff_gnn/tabdiff_news/attngnn/v2_gatv2_layers3_dim256_topk5_heads8_dimt128_noise-shared/seed2/
# Walltime 48:00:00: stage estimates 26.40 h x 1.3, rounded up (min 2, cap 48); ref default 0.2710 s/step (A100) x CPU ratio 8.08 = 2.189 s/step; 23.7 s/epoch

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

module purge
module load hdf5
module load python/3.10
module load cuda/11.8
module load cmake

export TORCH_CUDA_ARCH_LIST="8.0"

set -uo pipefail

HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
EVAL_PYTHON="${EVAL_PYTHON:-${HPC_ROOT}/venvs/tabdiff_eval310/bin/python}"
SYNTHCITY_PYTHON="${SYNTHCITY_PYTHON:-${HPC_ROOT}/venvs/tabdiff_synthcity310/bin/python}"
TABPC_PYTHON="${TABPC_PYTHON:-${HPC_ROOT}/venvs/tabpc310/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${EVAL_PYTHON}}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${HPC_ROOT}/transfer_runs}"
DEVICE="${DEVICE:-cuda:0}"

cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; exit 1; }

DS="news"
DENOISER="attngnn"
SEED="2"
STAGE_GROUPS="train"
TRAIN_BUDGET_H="46.5"

export HPC_ROOT REPO_ROOT EVAL_PYTHON SYNTHCITY_PYTHON TABPC_PYTHON TRAIN_PYTHON TRANSFER_ROOT DEVICE
export DS DENOISER SEED STAGE_GROUPS TRAIN_BUDGET_H

# shellcheck source=common_tabdiff_gnn.sh
. "${REPO_ROOT}/hpc_transfer/jobs_tabdiff_gnn/common_tabdiff_gnn.sh"

run_route_a

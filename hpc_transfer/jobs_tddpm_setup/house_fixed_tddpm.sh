#!/bin/bash -l

#SBATCH --job-name=house_tddpm
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/house_fixed_tddpm.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/house_fixed_tddpm.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time=6:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
##SBATCH --account=pr019022_gpu

# BUDGET-MATCHED RUN -- house, topk_mode fixed, TRAIN_SETUP=tabddpm.
#
# Same configuration as the run the current analysis uses for house
# (hpc_transfer/configs/house.sh, cat_input=log, training seed 0, 5 gen x 10
# clf seeds, EMA, same synthetic size, CatBoost) EXCEPT the training setup,
# which is read from the TabDDPM authors' exp/house/ddpm_cb_best/config.toml:
#   lr                0.0013926185951764255    exp/house/ddpm_cb_best/config.toml:31 [train.main] lr
#   steps             30000                    exp/house/ddpm_cb_best/config.toml:30 [train.main] steps
#   batch_size        4096                     exp/house/ddpm_cb_best/config.toml:33 [train.main] batch_size
#   weight_decay      0.0                      exp/house/ddpm_cb_best/config.toml:32 [train.main] weight_decay
#   num_timesteps     1000                     exp/house/ddpm_cb_best/config.toml:25 [diffusion_params] num_timesteps
#   sample_batch_size 30000                    exp/house/ddpm_cb_best/config.toml:46 [sample] batch_size
#   scheduler         'cosine'                 exp/house/ddpm_cb_best/config.toml:27 [diffusion_params] scheduler
# (ablation_runner.training_setup; the values and their file:line also land in
# run_meta.json / train_setup.json.) Training aborts with divergence.json if
# the loss becomes non-finite or explodes; nothing else about training changes.
#
# Results: $TRANSFER_ROOT/exp/house/ablation_tddpm_setup/<combo>/
# -- a *_tddpm_setup sibling tree; no existing result is touched.
#
# Walltime 6:00:00: measured 0.94 h training (transfer_runs/exp/house/ablation/gnn_gatv2_layers3_dim128_topk5_heads8_noattn) x 30000/20000 steps + 5 x 0.45 h per seed (median of 4 consecutive seed gaps) = 3.7 h, x 1.5 margin, rounded up.

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
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabddpm310/bin/python}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${HPC_ROOT}/transfer_runs}"

cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; exit 1; }

TOPK_MODE="${TOPK_MODE:-fixed}"
CAT_INPUT="${CAT_INPUT:-log}"
TRAIN_SETUP="${TRAIN_SETUP:-tabddpm}"
TOPK_WARMUP_STEPS="${TOPK_WARMUP_STEPS:-0}"
K_MIN="${K_MIN:-1}"
TOPK_LOG_EVERY="${TOPK_LOG_EVERY:-100}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

DATASET="house"

export HPC_ROOT REPO_ROOT PYTHON_BIN TRANSFER_ROOT DATASET TOPK_MODE CAT_INPUT \
       TRAIN_SETUP TOPK_WARMUP_STEPS K_MIN TOPK_LOG_EVERY DEVICE SEED

# shellcheck source=../common.sh
. "${REPO_ROOT}/hpc_transfer/common.sh"

run_transfer

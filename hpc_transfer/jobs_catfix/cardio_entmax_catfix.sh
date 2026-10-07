#!/bin/bash -l

#SBATCH --job-name=cardio_entmax_cf
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/cardio_entmax_catfix.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/cardio_entmax_catfix.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --partition=gpu

# ===========================================================================
# TODO(GPU) -- same as every hpc_transfer job: "gpu:a100:1" is rejected on
# this cluster. Find the accepted string with `sinfo -p gpu -o "%N %G"` and
# either uncomment the line below or pass --gres on the sbatch command line.
##SBATCH --gres=gpu:1
# ===========================================================================
##SBATCH --account=pr019022_gpu

# CATEGORICAL-INPUT FIX RERUN (priority 1) -- cardio, topk_mode adaptive_entmax15.
#
# Identical to the original hpc_transfer job for this dataset and rule
# (configuration from hpc_transfer/configs/cardio.sh, 20 000 steps, 5 gen x 10
# clf seeds, training seed 0, same resources) except CAT_INPUT=prob: the GNN
# denoiser's categorical nodes see one-hot probabilities instead of the log
# one-hot (see tab_ddpm/graph_denoiser.py, cat_input). Results go to
#   $TRANSFER_ROOT/exp/cardio/ablation_adaptive_entmax15_catfix/<combo>/
# -- a *_catfix sibling tree; the original results are never touched.
# run_transfer runs a preflight that aborts if the uploaded code lacks the
# option.

# shellcheck disable=SC2268,SC2086
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

# Run from the repo root whatever directory sbatch was called from: parts of
# the pipeline used to read repo files relative to the working directory.
cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; exit 1; }

TOPK_MODE="${TOPK_MODE:-adaptive_entmax15}"
CAT_INPUT="${CAT_INPUT:-prob}"
TOPK_WARMUP_STEPS="${TOPK_WARMUP_STEPS:-0}"
K_MIN="${K_MIN:-1}"
TOPK_LOG_EVERY="${TOPK_LOG_EVERY:-100}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

DATASET="cardio"

export HPC_ROOT REPO_ROOT PYTHON_BIN TRANSFER_ROOT DATASET TOPK_MODE CAT_INPUT \
       TOPK_WARMUP_STEPS K_MIN TOPK_LOG_EVERY DEVICE SEED

# shellcheck source-path=SCRIPTDIR
# shellcheck source=../common.sh
. "${REPO_ROOT}/hpc_transfer/common.sh"

run_transfer

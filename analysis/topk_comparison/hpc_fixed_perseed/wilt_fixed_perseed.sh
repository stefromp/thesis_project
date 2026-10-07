#!/bin/bash -l

#SBATCH --job-name=wilt_fixed_ps
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/wilt_fixed_perseed.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/wilt_fixed_perseed.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# NOTE: hpc_transfer/jobs/*.sh use --mem=120G and leave --gres commented out,
# because "gpu:a100:1" was rejected on this cluster ("invalid GRES format or
# GPU type"). If submission fails, override on the command line, e.g.
#   sbatch --gres=gpu:1 wilt_fixed_perseed.sh
##SBATCH --account=pr019022_gpu

# Fixed top-k, per-seed rerun at TRAINING SEED 0 (same as the grid), for wilt's selected configuration
# (hpc_transfer/configs/wilt.sh). Results end up in
#   $RESULTS_ROOT/exp/wilt/ablation_fixed_seed0/<combo>/
# Adds per-seed values only; see common_perseed.sh for what it does not fix.

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
RESULTS_ROOT="${RESULTS_ROOT:-${HPC_ROOT}/transfer_runs}"
DEVICE="${DEVICE:-cuda:0}"

DATASET="wilt"

export HPC_ROOT REPO_ROOT PYTHON_BIN RESULTS_ROOT DATASET DEVICE

# shellcheck source=common_perseed.sh
. "${REPO_ROOT}/analysis/topk_comparison/hpc_fixed_perseed/common_perseed.sh"

run_fixed_perseed

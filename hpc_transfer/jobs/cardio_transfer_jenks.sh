#!/bin/bash -l

#SBATCH --job-name=cardio_transfer_jenks
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/cardio_transfer_jenks.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/cardio_transfer_jenks.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --partition=gpu

# ===========================================================================
# TODO(GPU) -- FILL THIS IN BEFORE SUBMITTING. This is the ONLY line to edit.
# "gpu:a100:1" is rejected on this cluster ("invalid GRES format or GPU type").
# Find the accepted string with:      sinfo -p gpu -o "%N %G"
# then uncomment the line below with that value, e.g. #SBATCH --gres=gpu:1
#
##SBATCH --gres=gpu:1
#
# Alternatively leave it commented and pass it on the sbatch command line
# (GRES=... ) -- a command-line --gres overrides this directive, so you only
# have to get it right in one place.
# ===========================================================================

# Enable if submission is rejected for a missing account association
# (association: account=pr019022_gpu, partitions gpu and mig).
##SBATCH --account=pr019022_gpu

# NOTE: the #SBATCH --output/--error paths above are literal on purpose.
# SLURM parses these directives before any shell expansion, so they cannot use
# $HPC_ROOT. If you move HPC_ROOT, update those two lines too -- they are the
# only place in this file where the root is spelled out twice.

# Verbatim from the cluster's job template -- kept byte-for-byte on purpose,
# so the x-prefix comparison and unquoted expansion are intentional here.
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

# --- not -e: the failure must reach the log, not abort the shell ------------
set -uo pipefail

# ---------------------------------------------------------------------------
# Paths. Defined once, overridable from the environment.
#
# TRANSFER_ROOT is a results tree of its own: this study never reads or writes
# $REPO_ROOT/exp/, so it cannot disturb the ablation or the hpc/ adaptive runs.
# ---------------------------------------------------------------------------
HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabddpm310/bin/python}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${HPC_ROOT}/transfer_runs}"

# Run from the repo root whatever directory sbatch was called from: parts of
# the pipeline used to read repo files relative to the working directory.
cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Top-k rule: adaptive_jenks -- a per-node budget from a plain 2-class Jenks
# split of the raw scores, exactly what hpc/ runs on the seven ablated
# datasets. Results land in
# $TRANSFER_ROOT/exp/<ds>/ablation_adaptive_jenks/<combo>/, a sibling of the
# fixed run's tree that ablation_runner derives from --topk_mode.
# ---------------------------------------------------------------------------
TOPK_MODE="${TOPK_MODE:-adaptive_jenks}"

# ---------------------------------------------------------------------------
# Warm-up: number of training steps that keep the fixed top-k path before the
# adaptive rule takes over. 0 = adaptive from step 0. Change it HERE only.
# The counter is in optimizer-step units.
# ---------------------------------------------------------------------------
TOPK_WARMUP_STEPS="${TOPK_WARMUP_STEPS:-0}"

K_MIN="${K_MIN:-1}"
TOPK_LOG_EVERY="${TOPK_LOG_EVERY:-100}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

DATASET="cardio"

export HPC_ROOT REPO_ROOT PYTHON_BIN TRANSFER_ROOT DATASET TOPK_MODE \
       TOPK_WARMUP_STEPS K_MIN TOPK_LOG_EVERY DEVICE SEED

# shellcheck source-path=SCRIPTDIR
# shellcheck source=../common.sh
. "${REPO_ROOT}/hpc_transfer/common.sh"

run_transfer

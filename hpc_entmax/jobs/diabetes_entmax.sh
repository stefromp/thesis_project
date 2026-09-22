#!/bin/bash -l

#SBATCH --job-name=diabetes_ablation
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/%x_%j.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=pr019022_gpu

# NOTE: the #SBATCH --output/--error paths are literal on purpose: SLURM parses
# these directives before any shell expansion, so they cannot use $HPC_ROOT.
# %x (job name) + %j (job id) make every log file unique, so no run -- of any
# mode, resubmitted or not -- can overwrite another's .out/.err, including the
# Jenks logs written by hpc/jobs/. hpc_entmax/submit_all.sh additionally
# overrides --job-name/--output/--error per mode on the sbatch command line.

# ---------------------------------------------------------------------------
# Adaptive rule: adaptive_sparsemax (alpha = 2, default) or adaptive_entmax15
# (alpha = 1.5). Nothing else is accepted -- adaptive_jenks and fixed runs
# belong to hpc/. Override from the environment, e.g.
#   TOPK_MODE=adaptive_entmax15 sbatch hpc_entmax/jobs/diabetes_entmax.sh
# (hpc_entmax/submit_all.sh passes it via --export=ALL,TOPK_MODE=<mode>.)
# ---------------------------------------------------------------------------
TOPK_MODE="${TOPK_MODE:-adaptive_sparsemax}"

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

# --- not -e: a failing combination must not kill the loop -------------------
set -uo pipefail

# ---------------------------------------------------------------------------
# Paths. Defined once, overridable from the environment.
# ---------------------------------------------------------------------------
HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabddpm310/bin/python}"

# ---------------------------------------------------------------------------
# Warm-up: training steps that keep the fixed top-k path before the entmax
# rule takes over. 0 = adaptive from step 0 -- the same value the Jenks runs
# used, so the three methods stay comparable.
# ---------------------------------------------------------------------------
TOPK_WARMUP_STEPS="${TOPK_WARMUP_STEPS:-0}"

K_MIN="${K_MIN:-1}"
TOPK_LOG_EVERY="${TOPK_LOG_EVERY:-100}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

DATASET="diabetes"

export HPC_ROOT REPO_ROOT PYTHON_BIN DATASET TOPK_MODE TOPK_WARMUP_STEPS \
       K_MIN TOPK_LOG_EVERY DEVICE SEED

# shellcheck source-path=SCRIPTDIR
# shellcheck source=../common.sh
. "${REPO_ROOT}/hpc_entmax/common.sh"

run_dataset

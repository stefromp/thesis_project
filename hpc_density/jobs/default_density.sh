#!/bin/bash -l

#SBATCH --job-name=default_density
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
# %x (job name) + %j (job id) make every log file unique. hpc_density/submit_all.sh
# overrides --job-name/--output/--error (and --mem, from MEM) on the command line.

# ---------------------------------------------------------------------------
# ONE condition per job: fixed_k<K>, fixed_dense, adaptive_jenks,
# adaptive_sparsemax or adaptive_entmax15 (valid set: density_manifest.py
# --conditions default). No default -- e.g.
#   CONDITION=fixed_k2 sbatch hpc_density/jobs/default_density.sh
# (hpc_density/submit_all.sh passes it via --export.)
# ---------------------------------------------------------------------------
CONDITION="${CONDITION:-}"

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

set -uo pipefail

# ---------------------------------------------------------------------------
# Paths. Defined once, overridable from the environment.
# ---------------------------------------------------------------------------
HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabddpm310/bin/python}"

# Same as every other adaptive run: adaptive rule from step 0, k_min 1.
TOPK_WARMUP_STEPS="${TOPK_WARMUP_STEPS:-0}"
K_MIN="${K_MIN:-1}"
TOPK_LOG_EVERY="${TOPK_LOG_EVERY:-100}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

DATASET="default"

export HPC_ROOT REPO_ROOT PYTHON_BIN DATASET CONDITION \
       TOPK_WARMUP_STEPS K_MIN TOPK_LOG_EVERY DEVICE SEED

# shellcheck source-path=SCRIPTDIR
# shellcheck source=../common.sh
. "${REPO_ROOT}/hpc_density/common.sh"

run_condition

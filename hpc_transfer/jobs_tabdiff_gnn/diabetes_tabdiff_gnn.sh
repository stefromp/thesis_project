#!/bin/bash -l

#SBATCH --job-name=diabetes_tabdiff_gnn
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/diabetes_tabdiff_gnn.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/diabetes_tabdiff_gnn.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time=48:00:00
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
# only place in this file where the root is spelled out twice. For SEED=1 / 2
# pass --job-name/--output/--error on the sbatch command line (see
# hpc_transfer/submit_tabdiff_gnn.sh) so the seeds do not share one log.

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
# TRANSFER_ROOT is shared with hpc_transfer/, but Route A writes only to
# $TRANSFER_ROOT/exp_tabdiff_gnn/ (tabdiff_gnn/config.py enforces it), so it can
# never disturb the exp/ trees of the transfer study or the TabDiff study.
#
# PYTHON_BIN is the TabDiff environment (the eval venv: torch, sdmetrics,
# category_encoders, icecream, tomli -- what TabDiff's main.py imports); it
# trains, samples and runs TabDiff's evaluator. alpha/beta run in the synthcity
# venv and wNMIS / C2ST(XGB) in the tabpc venv (tabdiff_gnn/README.md, Install).
# ---------------------------------------------------------------------------
HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabdiff_eval310/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-${PYTHON_BIN}}"
SYNTHCITY_PYTHON="${SYNTHCITY_PYTHON:-${HPC_ROOT}/venvs/tabdiff_synthcity310/bin/python}"
TABPC_PYTHON="${TABPC_PYTHON:-${HPC_ROOT}/venvs/tabpc310/bin/python}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${HPC_ROOT}/transfer_runs}"

# Run from the repo root whatever directory sbatch was called from: parts of
# the pipeline used to read repo files relative to the working directory.
cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Denoiser: AttnGNN V2 inside TabDiff's framework -- only the denoiser differs
# from TabDiff (EDM + masked diffusion, learnable per-column schedules, TabDiff's
# Trainer, sampler and evaluator @ 5ecdb33, file digests checked).
#   gcn L2 d256 top-k 5 (fixed) heads 4
#   GNN type / layers / top_k / heads from hpc_transfer/configs/adult.sh -- our choice: diabetes has no TabDDPM source
#   d_model 256 on every dataset, dim_t 128, one shared noise embedding,
#   fixed top-k rule, dynamic graph, no global attention.
# Trainer: TabDiff's Trainer, unchanged
# Results: $TRANSFER_ROOT/exp_tabdiff_gnn/tabdiff_diabetes/attngnn/v2_gcn_layers2_dim256_topk5_heads4_dimt128_noise-shared/seed<SEED>/
# ---------------------------------------------------------------------------
DENOISER="attngnn"

# ---------------------------------------------------------------------------
# Ablation: empty = the main Route-A run. A name from tabdiff_gnn/config.py
# ABLATIONS changes exactly one setting; the run gets its own path (e.g. ..._topk11_...)
# and run_meta.json records {"ablation": ...}. NOT part of the main protocol.
# ---------------------------------------------------------------------------
ABLATION="${ABLATION:-}"

# ---------------------------------------------------------------------------
# Stages this job runs (each leaves <run>/stages/<stage>.done; a resubmitted
# job skips what is done):
#   train (11.28 h) -> sample (0.48 h) -> eval_main (0.28 h) -> eval_quality (1.73 h) -> eval_tabpc (1.53 h)
# Protocol: TabDiff's tabdiff_configs.toml unchanged (8000 epochs, batch 4096,
# lr 1e-3 + ReduceLROnPlateau, EMA 0.997, best EMA loss after epoch 4000,
# 50 sampling steps), 20 tables of n_train rows (generation seeds 0..19).
# No DCR in this study.
# Walltime 48:00:00: fixed at 48 h for every job; estimate 15.30 h (x1.3 = 20 h); ref adult 0.0414 s/step (A100) x CPU ratio 5.24 = 0.217 s/step; 5.1 s/epoch
# ---------------------------------------------------------------------------
STAGES="${STAGES:-train,sample,eval_main,eval_quality,eval_tabpc}"

# ---------------------------------------------------------------------------
# Resumable training budget (news only; empty = TabDiff's Trainer, unchanged):
# after this many hours of training the run state is saved and the job exits
# cleanly (exit 0), so the next job of the chain -- this same script -- resumes.
# ---------------------------------------------------------------------------
TRAIN_BUDGET_H="${TRAIN_BUDGET_H:-}"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

DATASET="diabetes"

export HPC_ROOT REPO_ROOT PYTHON_BIN EVAL_PYTHON SYNTHCITY_PYTHON TABPC_PYTHON \
       TRANSFER_ROOT DATASET DENOISER ABLATION STAGES TRAIN_BUDGET_H DEVICE SEED

# shellcheck source-path=SCRIPTDIR
# shellcheck source=common_tabdiff_gnn.sh
. "${REPO_ROOT}/hpc_transfer/jobs_tabdiff_gnn/common_tabdiff_gnn.sh"

run_route_a

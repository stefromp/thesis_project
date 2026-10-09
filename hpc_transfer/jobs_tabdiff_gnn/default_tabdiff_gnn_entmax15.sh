#!/bin/bash -l

#SBATCH --job-name=default_tabdiff_gnn_entmax15
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/default_tabdiff_gnn_entmax15.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/default_tabdiff_gnn_entmax15.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time=17:00:00
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
#   gatv2 L3 d256 rule entmax15 (k_i in [k_min, F-1]) heads 8
#   GNN type / layers / top_k / heads from hpc_transfer/configs/default.sh
#   d_model 256 on every dataset, dim_t 128, one shared noise embedding,
#   adaptive neighbour selection adaptive_entmax15 (source top_k 5 unused), dynamic graph, no global attention.
# Trainer: TabDiff's Trainer, unchanged
# Results: $TRANSFER_ROOT/exp_tabdiff_gnn/tabdiff_default/attngnn/v2_gatv2_layers3_dim256_entmax15_heads8_dimt128_noise-shared/seed<SEED>/
# ---------------------------------------------------------------------------
DENOISER="attngnn"

# ---------------------------------------------------------------------------
# Ablation: empty = the main Route-A run. A name from tabdiff_gnn/config.py
# ABLATIONS changes exactly one setting; the run gets its own path (e.g. ..._topk11_...)
# and run_meta.json records {"ablation": ...}. NOT part of the main protocol.
# ---------------------------------------------------------------------------
ABLATION="${ABLATION:-}"

# ---------------------------------------------------------------------------
# Neighbour-selection rule (TOPK_MODE, as in hpc_transfer/jobs): adaptive_entmax15 --
# support of entmax-1.5 over the off-diagonal scores (mask only); edge weights = the same sigmoid gates as fixed.
# Settings of the thesis catfix runs, fixed in tabdiff_gnn/config.py (RULE_SETTINGS):
# TOPK_WARMUP_STEPS / K_MIN / TOPK_LOG_EVERY must keep these values (the run path
# encodes only the rule; the preflight refuses others).
# Selected-k statistics vs t in [0,1]: <run>/topk_k_stats.csv, topk_k_hist.csv.
# ---------------------------------------------------------------------------
RULE_ARM=1
TOPK_MODE="${TOPK_MODE:-adaptive_entmax15}"
TOPK_WARMUP_STEPS="${TOPK_WARMUP_STEPS:-0}"
K_MIN="${K_MIN:-1}"
TOPK_LOG_EVERY="${TOPK_LOG_EVERY:-100}"

# ---------------------------------------------------------------------------
# Stages this job runs (each leaves <run>/stages/<stage>.done; a resubmitted
# job skips what is done):
#   train (11.42 h) -> sample (0.53 h) -> eval_main (0.12 h) -> eval_quality (0.38 h) -> eval_tabpc (0.10 h)
# Protocol: TabDiff's tabdiff_configs.toml unchanged (8000 epochs, batch 4096,
# lr 1e-3 + ReduceLROnPlateau, EMA 0.997, best EMA loss after epoch 4000,
# 50 sampling steps), 20 tables of n_train rows (generation seeds 0..19).
# No DCR in this study.
# Walltime 17:00:00: estimate 12.54 h x 1.3, rounded up (min 2, cap 48); ref default 0.2710 s/step (A100) x CPU ratio 1.94 = 0.525 s/step; 5.1 s/epoch; adaptive_entmax15: train x1.00, forward x1.02 vs fixed (bench_step_ratio.py --rules)
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

DATASET="default"

export HPC_ROOT REPO_ROOT PYTHON_BIN EVAL_PYTHON SYNTHCITY_PYTHON TABPC_PYTHON \
       TRANSFER_ROOT DATASET DENOISER ABLATION STAGES TRAIN_BUDGET_H DEVICE SEED \
       RULE_ARM TOPK_MODE TOPK_WARMUP_STEPS K_MIN TOPK_LOG_EVERY

# shellcheck source-path=SCRIPTDIR
# shellcheck source=common_tabdiff_gnn.sh
. "${REPO_ROOT}/hpc_transfer/jobs_tabdiff_gnn/common_tabdiff_gnn.sh"

run_route_a

#!/bin/bash -l
#SBATCH --job-name=tdx_default
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/tdx_default.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/tdx_default.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time=14:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
##SBATCH --account=pr019022_gpu

# TABDIFF-TRANSFER STUDY -- default (binclass). ONE GPU job, part 'all':
#   tabdiff_default:train (2.31 h) -> tabdiff_default:sample (3.28 h) -> tabdiff_default:eval_main (0.07 h) -> tabdiff_default:eval_quality (0.33 h)
#   -> tabdiff_default_dcr:train (2.31 h) -> tabdiff_default_dcr:sample (1.84 h) -> tabdiff_default_dcr:eval_dcr (0.10 h)
# Every stage writes its outputs and a done-marker (<run>/stages/<stage>.done);
# resubmitting this script skips finished stages.
#
# Data      : TabDiff's own splits data/{tabdiff_default,tabdiff_default_dcr}/ (built with TabDiff's
#             process_dataset.py @ 5ecdb33 by hpc_transfer/tabdiff_study/prepare_data.py)
# Source    : default (nearest same-task TabDDPM dataset). Transferred:
#   lr                0.00046818967784044777         exp/default/ddpm_cb_best/config.toml:29 [train.main] lr
#   steps             30000                          exp/default/ddpm_cb_best/config.toml:28 [train.main] steps
#   batch_size        4096                           exp/default/ddpm_cb_best/config.toml:31 [train.main] batch_size
#   weight_decay      0.0                            exp/default/ddpm_cb_best/config.toml:30 [train.main] weight_decay
#   num_timesteps     1000                           exp/default/ddpm_cb_best/config.toml:24 [diffusion_params] num_timesteps
#   sample_batch_size 10000                          exp/default/ddpm_cb_best/config.toml:44 [sample] batch_size
#   scheduler         'cosine'                       absent from exp/default/ddpm_cb_best/config.toml; scripts/train.py default scheduler='cosine' (what TabDDPM's pipeline.py ran with)
#   normalization     'minmax'                       exp/default/ddpm_cb_best/config.toml:35 [train.T] normalization
#   is_y_cond         True                           exp/default/ddpm_cb_best/config.toml:10 [model_params] is_y_cond
#   gnn_type          'gatv2'                        hpc_transfer/configs/default.sh:51 GNN_TYPE
#   n_layers          3                              hpc_transfer/configs/default.sh:52 N_LAYERS
#   d_model           128                            hpc_transfer/configs/default.sh:53 D_MODEL
#   top_k             5                              hpc_transfer/configs/default.sh:54 TOP_K
#   n_heads           8                              hpc_transfer/configs/default.sh:55 N_HEADS
#   use_attention     False                          hpc_transfer/configs/default.sh:56 USE_ATTENTION
# Model     : GNN denoiser gatv2 L3 d128 top-k 5 (fixed) heads 8, cat_input=prob, graph dynamic, global attention off;
#             divergence check on (divergence.json + exit != 0)
# Protocol  : 20 tables of n_train rows per model (seeds 0..19); TabDiff's own
#             evaluation code (digests checked), eval venv + synthcity venv
# Results   : $TRANSFER_ROOT/exp_tabdiff/<target>/gnn/gnn_gatv2_layers3_dim128_topk5_heads8_noattn/
# Walltime 14:00:00: sum of stage estimates 10.23 h x 1.3, rounded up (measured from transfer_runs/exp/default/ablation_catfix/gnn_gatv2_layers3_dim128_topk5_heads8_noattn)

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
EVAL_PYTHON="${EVAL_PYTHON:-${HPC_ROOT}/venvs/tabdiff_eval310/bin/python}"
SYNTHCITY_PYTHON="${SYNTHCITY_PYTHON:-${HPC_ROOT}/venvs/tabdiff_synthcity310/bin/python}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${HPC_ROOT}/transfer_runs}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; exit 1; }

DS="default"
PART="all"

export HPC_ROOT REPO_ROOT PYTHON_BIN EVAL_PYTHON SYNTHCITY_PYTHON TRANSFER_ROOT DEVICE SEED DS PART

# shellcheck source=common_tabdiff.sh
. "${REPO_ROOT}/hpc_transfer/jobs_tabdiff/common_tabdiff.sh"

run_tabdiff_dataset

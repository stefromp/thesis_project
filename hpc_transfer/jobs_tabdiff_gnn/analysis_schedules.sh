#!/bin/bash -l

#SBATCH --job-name=analysis_schedules
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/analysis_schedules.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/analysis_schedules.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --partition=gpu

# ===========================================================================
# Analysis only (no training): do TabDiff's learned per-column noise schedules
# interact with the learned graph? analysis/tabdiff/schedules_vs_graph.py all:
#   extract  learned rho_i / k_j + reveal times t* from each run's sampled best-EMA checkpoint
#   sample   ~2,000 rows per run (fixed / jenks / entmax15) with TabDiff's sampler (50 steps,
#            the run's settings), logging which neighbours every node keeps per block and call;
#            logging off / on / off checks + the production entry.build path (VERIFY_ENTRY rows)
#   report   SCHEDULES.md, CSVs, figures -> analysis/tabdiff/schedules_vs_graph/
# Reads the seed-0 Route-A runs in $TRANSFER_ROOT/exp_tabdiff_gnn/ (missing runs are listed,
# not fatal); writes nothing into them.
#
# Submit with the flags on the sbatch line (the gpu partition needs the GRES; NOT submitted):
#   sbatch --gres=gpu:a100:1 --mem=24G --time=01:00:00 hpc_transfer/jobs_tabdiff_gnn/analysis_schedules.sh
# Estimate: well under 1 h (16 runs x 4 short draws on one A100 + 2,000-permutation tests on CPU).
# Hand-written: tabdiff_gnn/make_jobs.py keeps analysis_*.sh when it regenerates the jobs.
# Afterwards copy analysis/tabdiff/schedules_vs_graph/ back to the local repo.
# ===========================================================================

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

# Paths, overridable from the environment. PYTHON_BIN = the eval venv (TabDiff's
# environment, where the runs trained and sampled; has entmax, matplotlib).
HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabdiff_eval310/bin/python}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${HPC_ROOT}/transfer_runs}"
OUT="${OUT:-${REPO_ROOT}/analysis/tabdiff/schedules_vs_graph}"
DEVICE="${DEVICE:-cuda:0}"
N_ROWS="${N_ROWS:-2000}"
N_PERM="${N_PERM:-2000}"
VERIFY_ENTRY="${VERIFY_ENTRY:-256}"
DATASETS="${DATASETS:-adult,beijing,default,diabetes,magic,shoppers}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; exit 1; }

echo "===================================================================="
echo "  analysis     : schedules vs graph (Route A, seed 0) -- no training"
echo "  runs         : ${TRANSFER_ROOT}/exp_tabdiff_gnn/  datasets ${DATASETS}"
echo "  output       : ${OUT}"
echo "  python       : ${PYTHON_BIN}"
echo "  node         : $(hostname)   gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
echo "  started      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "===================================================================="

if ! "${PYTHON_BIN}" - "${REPO_ROOT}" <<'PYEOF'
import sys
repo = sys.argv[1]
sys.path.insert(0, f"{repo}/tabdiff_gnn")
import config as C
C.verify_tabdiff(repo)
import torch, numpy, pandas, scipy, matplotlib, entmax  # noqa: F401
import denoiser, entry  # noqa: F401
entry._tabdiff_imports()
assert torch.cuda.is_available(), "no CUDA in the eval venv"
print("  preflight          : TabDiff digests OK, imports OK, cuda", torch.cuda.get_device_name(0))
PYEOF
then
    echo "  ABORTED: preflight failed -- nothing was run" >&2
    exit 1
fi

srun "${PYTHON_BIN}" "${REPO_ROOT}/analysis/tabdiff/schedules_vs_graph.py" all \
    --exp_root "${TRANSFER_ROOT}" --out "${OUT}" --device "${DEVICE}" --datasets "${DATASETS}" \
    --n_rows "${N_ROWS}" --n_perm "${N_PERM}" --verify_entry "${VERIFY_ENTRY}"
status=$?
echo "  finished $(date -u +%Y-%m-%dT%H:%M:%SZ) exit ${status}"
exit ${status}

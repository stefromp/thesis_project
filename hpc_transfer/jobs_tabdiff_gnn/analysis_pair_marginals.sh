#!/bin/bash -l

#SBATCH --job-name=analysis_pair_marginals
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/analysis_pair_marginals.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/analysis_pair_marginals.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --partition=gpu

# ===========================================================================
# Analysis only (no training, no sampling, CPU only): per-pair Trend error, per-column
# Shape error and marginals of the seed-0 Route-A tables, for thesis figures 6-7.
# analysis/tabdiff/pair_and_marginals.py runs TabDiff's own TabMetrics.evaluate_density
# (sdmetrics QualityReport) on each of the 20 tables of every run and keeps the report's
# Column Shapes / Column Pair Trends details; per table it checks that their means equal
# the run's tabdiff_eval/all_results.csv shape / trend to 1e-6 and aborts otherwise.
# Datasets beijing, default, magic; arms fixed, jenks, entmax15, dense (where on disk).
# Writes <run>/analysis/{pair_trend_error,column_shape_error,marginals}.csv +
# pair_marginals_meta.json in each run; nothing else in the runs is touched.
#
# Submit with the flags on the sbatch line (the gpu partition needs the GRES; NOT submitted):
#   sbatch --gres=gpu:a100:1 --mem=24G --time=00:30:00 hpc_transfer/jobs_tabdiff_gnn/analysis_pair_marginals.sh
# Estimate: ~3 min (11 runs x 20 tables, ~0.5 s per table on a laptop CPU); the GPU is idle.
# Hand-written: tabdiff_gnn/make_jobs.py keeps analysis_*.sh when it regenerates the jobs.
# Afterwards rsync the runs' *.csv / *.json back as usual (analysis/ is inside each run).
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

# --- not -e: the failure must reach the log, not abort the shell ------------
set -uo pipefail

# Paths, overridable from the environment. PYTHON_BIN = the eval venv (TabDiff's
# evaluator: sdmetrics 0.18.0, the pin the script checks).
HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabdiff_eval310/bin/python}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${HPC_ROOT}/transfer_runs}"
DATASETS="${DATASETS:-beijing,default,magic}"
ARMS="${ARMS:-fixed,jenks,entmax15,dense}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; exit 1; }

echo "===================================================================="
echo "  analysis     : pair trends + marginals (Route A, seed 0) -- no training"
echo "  runs         : ${TRANSFER_ROOT}/exp_tabdiff_gnn/  datasets ${DATASETS}  arms ${ARMS}"
echo "  python       : ${PYTHON_BIN}"
echo "  node         : $(hostname)"
echo "  started      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "===================================================================="

if ! "${PYTHON_BIN}" - "${REPO_ROOT}" <<'PYEOF'
import os, sys
repo = sys.argv[1]
sys.path.insert(0, f"{repo}/tabdiff_gnn")
import config as C
C.verify_tabdiff(repo)
sys.path.insert(0, os.path.join(repo, C.TABDIFF_REL))
import numpy, pandas, sdmetrics  # noqa: F401
from tabdiff.metrics import TabMetrics  # noqa: F401
assert sdmetrics.__version__ == C.S.EVAL_PINS["sdmetrics"], sdmetrics.__version__
print("  preflight          : TabDiff digests OK, TabMetrics imports, sdmetrics", sdmetrics.__version__)
PYEOF
then
    echo "  ABORTED: preflight failed -- nothing was run" >&2
    exit 1
fi

srun "${PYTHON_BIN}" "${REPO_ROOT}/analysis/tabdiff/pair_and_marginals.py" \
    --exp_root "${TRANSFER_ROOT}" --datasets "${DATASETS}" --arms "${ARMS}"
status=$?
echo "  finished $(date -u +%Y-%m-%dT%H:%M:%SZ) exit ${status}"
exit ${status}

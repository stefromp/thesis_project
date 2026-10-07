#!/usr/bin/env bash
#
# Local MOCK dry run of one generated dataset job, e.g.
#
#   BASE_PY=~/anaconda3/envs/tabddpm/bin/python EVAL_PY=/path/eval-venv/bin/python \
#   SYNTH_PY=/path/synthcity-venv/bin/python \
#     hpc_transfer/tabdiff_study/mock/dryrun_mock.sh shoppers /tmp/td_mock [job_script_name]
#
# Runs the REAL hpc_transfer/jobs_tabdiff/<ds>.sh (or the given split script)
# with bash, on CPU, against a throw-away TRANSFER_ROOT. Substituted (and only
# these): `module`, `srun`, `nvidia-smi` (mock/bin), the interpreters go
# through mock_python.py (MOCK_STEPS training steps, MOCK_RUNS tables,
# gpu_hist->hist on xgboost wheels without CUDA), DEVICE=cpu, and
# TABDIFF_PREFLIGHT_ALLOW_CPU=1 (the preflight otherwise insists on a CUDA
# torch in the eval venv). Run it twice: the second run must skip every stage.
set -uo pipefail
DS="${1:?dataset}"; ROOT="${2:?mock TRANSFER_ROOT}"; JOB="${3:-${DS}.sh}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BASE_PY="${BASE_PY:-$(command -v python3)}"
EVAL_PY="${EVAL_PY:?set EVAL_PY to the eval venv python}"
SYNTH_PY="${SYNTH_PY:?set SYNTH_PY to the synthcity venv python}"

mkdir -p "${ROOT}/bin"
mkwrap() {  # $1 name  $2 real interpreter
    cat > "${ROOT}/bin/$1" <<EOF
#!/usr/bin/env bash
exec "$2" "${HERE}/mock_python.py" "\$@"
EOF
    chmod +x "${ROOT}/bin/$1"
}
mkwrap py_train "${BASE_PY}"
mkwrap py_eval "${EVAL_PY}"
mkwrap py_synth "${SYNTH_PY}"

export PATH="${HERE}/bin:${PATH}"
export HPC_ROOT="${ROOT}" REPO_ROOT="${REPO}" TRANSFER_ROOT="${ROOT}/transfer_runs"
export PYTHON_BIN="${ROOT}/bin/py_train" EVAL_PYTHON="${ROOT}/bin/py_eval"
export SYNTHCITY_PYTHON="${ROOT}/bin/py_synth"
export DEVICE=cpu TABDIFF_PREFLIGHT_ALLOW_CPU=1
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"   # the job sets OMP_NUM_THREADS from it
export MOCK_STEPS="${MOCK_STEPS:-30}" MOCK_RUNS="${MOCK_RUNS:-2}" MOCK_CPU_XGB="${MOCK_CPU_XGB:-1}"

echo "######## MOCK job: ${JOB}"
bash "${REPO}/hpc_transfer/jobs_tabdiff/${JOB}" || { echo "MOCK JOB FAILED"; exit 1; }
echo "######## MOCK DRY RUN OK -> ${TRANSFER_ROOT}/exp_tabdiff/"

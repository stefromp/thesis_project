# shellcheck shell=bash
#
# Shared function of the TabDiff-transfer jobs (hpc_transfer/jobs_tabdiff/).
# A generated job script sets the variables below and calls run_tabdiff_dataset:
#
#   REPO_ROOT PYTHON_BIN EVAL_PYTHON SYNTHCITY_PYTHON TRANSFER_ROOT DEVICE SEED
#   DS    dataset (magic, shoppers, adult, default, beijing, news)
#   PART  all (default) | main | dcr (only when a dataset is split into two jobs)
#
# It runs hpc_transfer/tabdiff_study/runner.py --dataset $DS --part $PART:
#   train -> 20 tables -> TabDiff evaluation (Shape/Trend, MLE gpu_hist, C2ST,
#   alpha/beta) -> train the _dcr model -> 20 tables -> DCR   (magic: no DCR)
# Every stage leaves <run>/stages/<stage>.done; a resubmitted job skips the
# stages already done. Results: $TRANSFER_ROOT/exp_tabdiff/<target>/gnn/<combo>/
# -- a tree no other study writes to (study.py enforces it).

TD_DIR="hpc_transfer/tabdiff_study"
# progress lines reach the .out file as they happen, not at exit
export PYTHONUNBUFFERED=1

# Proves, before any allocation is spent: TabDiff code unchanged (digests),
# data present, transferred values readable (printed), both evaluation venvs
# carry the pinned packages, the eval venv's torch sees this GPU (DCR) and
# XGBoost's tree_method='gpu_hist' (hard-coded in TabDiff's eval/mle/mle.py)
# fits here, and the uploaded denoiser has cat_input=prob (common.sh).
tabdiff_preflight() {
    if ! "${PYTHON_BIN}" "${REPO_ROOT}/${TD_DIR}/runner.py" --dataset "${DS}" --part "${PART}" \
            --exp_root "${TRANSFER_ROOT}" --repo_root "${REPO_ROOT}" --dry_run; then
        echo "PREFLIGHT FAILED: runner --dry_run" >&2
        return 1
    fi
    if ! "${EVAL_PYTHON}" - "${REPO_ROOT}" <<'PYEOF'
import os, sys, warnings
repo = sys.argv[1]
sys.path.insert(0, f"{repo}/hpc_transfer/tabdiff_study")
import study as S, tabdiff_eval as TE
S.verify_tabdiff(os.path.join(repo, S.TABDIFF_REL))
v = TE._versions(["sdmetrics", "xgboost", "pandas", "numpy", "sklearn", "torch"])
TE._check_pins(v, S.EVAL_PINS)
import torch
# TABDIFF_PREFLIGHT_ALLOW_CPU=1 is for the local mock dry run only.
assert torch.cuda.is_available() or os.environ.get("TABDIFF_PREFLIGHT_ALLOW_CPU") == "1", \
    "eval venv torch has no CUDA -- DCR is computed on this GPU"
import numpy as np
from xgboost import XGBClassifier
X = np.random.rand(200, 3); y = (X[:, 0] > .5).astype(int)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    XGBClassifier(n_estimators=5, tree_method="gpu_hist", nthread=-1,
                  objective="binary:logistic").fit(X, y)
print("  eval venv          :", v, "| cuda:", torch.cuda.is_available())
for m in w:
    print("  xgboost warning    :", str(m.message)[:160])
print("  gpu_hist fit       : OK")
PYEOF
    then
        echo "PREFLIGHT FAILED: eval venv ${EVAL_PYTHON} (see hpc_transfer/tabdiff_study/README.md)" >&2
        return 1
    fi
    if [ "${PART}" != "dcr" ]; then
        if ! "${SYNTHCITY_PYTHON}" - "${REPO_ROOT}" <<'PYEOF'
import sys
repo = sys.argv[1]
sys.path.insert(0, f"{repo}/hpc_transfer/tabdiff_study")
import study as S, tabdiff_eval as TE
v = TE._versions(["synthcity", "pandas", "numpy", "sklearn", "torch"])
TE._check_pins(v, S.QUALITY_PINS)
from synthcity.metrics import eval_statistical  # noqa: F401  (what eval_quality.py imports)
print("  synthcity venv     :", v)
PYEOF
        then
            echo "PREFLIGHT FAILED: synthcity venv ${SYNTHCITY_PYTHON}" >&2
            return 1
        fi
    fi
    # shellcheck source=../common.sh
    . "${REPO_ROOT}/hpc_transfer/common.sh"
    cat_input_preflight || return 1
    echo "  tabdiff preflight  : OK"
}

run_tabdiff_dataset() {
    cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; return 1; }
    case "${PART}" in all|main|dcr) ;; *) echo "ERROR: PART must be all|main|dcr" >&2; return 1 ;; esac
    echo "===================================================================="
    echo "  study        : tabdiff_transfer"
    echo "  dataset      : ${DS}   part: ${PART}"
    echo "  results tree : ${TRANSFER_ROOT}/exp_tabdiff/"
    echo "  python       : ${PYTHON_BIN}"
    echo "  eval / synth : ${EVAL_PYTHON} / ${SYNTHCITY_PYTHON}"
    echo "  node         : $(hostname)   gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    echo "  started      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "===================================================================="
    if ! tabdiff_preflight; then
        echo "  ABORTED: preflight failed -- nothing was run" >&2
        return 1
    fi
    local status f
    srun "${PYTHON_BIN}" "${REPO_ROOT}/${TD_DIR}/runner.py" \
        --dataset "${DS}" --part "${PART}" \
        --exp_root "${TRANSFER_ROOT}" --repo_root "${REPO_ROOT}" \
        --device "${DEVICE}" --seed "${SEED}" \
        --eval_python "${EVAL_PYTHON}" --synthcity_python "${SYNTHCITY_PYTHON}"
    status=$?
    if [ ${status} -ne 0 ]; then
        echo "  [FAIL] ${DS} (part ${PART}) exited ${status} -- finished stages keep their markers;"
        echo "         resubmit the same script to continue from the failed stage"
        for f in "${TRANSFER_ROOT}"/exp_tabdiff/tabdiff_"${DS}"*/gnn/*/divergence.json; do
            [ -f "${f}" ] && { echo "  [DIVERGED] ${f}:"; cat "${f}"; }
        done
        mkdir -p "${REPO_ROOT}/hpc_transfer/logs"
        printf '%s\t%s\t%s\texit=%d\tjob=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DS}" "${PART}" \
            "${status}" "${SLURM_JOB_ID:-none}" >> "${REPO_ROOT}/hpc_transfer/logs/tabdiff_failures.txt"
        return "${status}"
    fi
    echo "  [OK] ${DS} (part ${PART}) finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

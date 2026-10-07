# shellcheck shell=bash
#
# Shared per-job runner for the Route-A jobs (AttnGNN as the denoiser inside
# TabDiff's framework). Same shape as hpc_transfer/common.sh's run_transfer.
#
# ONE dataset, ONE configuration, ONE job: AttnGNN V2 with d_model 256, GNN type /
# layers / top_k / heads from hpc_transfer/configs/<source>.sh (tabdiff_gnn/config.py
# resolves the source; diabetes uses adult's -- our choice, it has no TabDDPM source).
#
# Sourced by hpc_transfer/jobs_tabdiff_gnn/*.sh after that script has set:
#   HPC_ROOT  REPO_ROOT  PYTHON_BIN  EVAL_PYTHON  SYNTHCITY_PYTHON  TABPC_PYTHON
#   TRANSFER_ROOT  DATASET  DENOISER  STAGES  TRAIN_BUDGET_H  DEVICE  SEED
#
#   STAGES          comma list of the stages this job runs, in order:
#                   train, sample, eval_main, eval_quality, eval_tabpc
#                   (all five, or one part of a chain)
#   TRAIN_BUDGET_H  news' resumable training only: hours after which the run state
#                   is saved and the job exits 0; the next job of the chain resumes
#
# It runs tabdiff_gnn/runner.py --stages $STAGES. Every stage leaves
# <run>/stages/<stage>.done and a resubmitted job skips what is done.
# Results: $TRANSFER_ROOT/exp_tabdiff_gnn/tabdiff_<ds>/<denoiser>/<arm>/seed<SEED>/
# No DCR in this study.

TG_DIR="tabdiff_gnn"
export PYTHONUNBUFFERED=1

# sets RA_ARGS (portable: no mapfile)
_route_a_args() {
    RA_ARGS=(--dataset "${DATASET}" --denoiser "${DENOISER}" --seed "${SEED}"
             --exp_root "${TRANSFER_ROOT}" --repo_root "${REPO_ROOT}")
    if [ -n "${TRAIN_BUDGET_H:-}" ]; then RA_ARGS+=(--train_budget_h "${TRAIN_BUDGET_H}"); fi
    # LOCAL MOCK ONLY (never set on the cluster): shortened training / fewer tables
    if [ -n "${ROUTE_A_MOCK_EPOCHS:-}" ]; then
        RA_ARGS+=(--smoke_epochs "${ROUTE_A_MOCK_EPOCHS}" --n_runs "${ROUTE_A_MOCK_RUNS:-2}")
    fi
}

# Proves before any allocation is spent: TabDiff unchanged (study + Route-A digests),
# tabpc unchanged, data present, the job's spec resolvable and its denoiser built +
# one forward pass in the train venv (runner --dry_run), the eval venv
# (= TabDiff's training env here) has the pinned evaluator, CUDA and gpu_hist, and
# imports TabDiff + our AttnGNN; the synthcity and tabpc venvs carry their pins.
route_a_preflight() {
    _route_a_args
    if ! "${PYTHON_BIN}" "${REPO_ROOT}/${TG_DIR}/runner.py" "${RA_ARGS[@]}" --dry_run; then
        echo "PREFLIGHT FAILED: runner --dry_run" >&2; return 1
    fi
    if ! "${EVAL_PYTHON}" - "${REPO_ROOT}" <<'PYEOF'
import os, sys, warnings
repo = sys.argv[1]
sys.path.insert(0, f"{repo}/tabdiff_gnn")
import config as C, tabdiff_eval as TE
C.verify_tabdiff(repo)
v = TE._versions(["sdmetrics", "xgboost", "pandas", "numpy", "sklearn", "torch"])
TE._check_pins(v, C.S.EVAL_PINS)
import torch
assert torch.cuda.is_available() or os.environ.get("TABDIFF_PREFLIGHT_ALLOW_CPU") == "1", \
    "eval venv torch has no CUDA (training and sampling run on this GPU)"
import denoiser, entry  # TabDiff modules + tab_ddpm.graph_denoiser import here
entry._tabdiff_imports()
import numpy as np
from xgboost import XGBClassifier
X = np.random.rand(200, 3); y = (X[:, 0] > .5).astype(int)
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    XGBClassifier(n_estimators=5, tree_method="gpu_hist", nthread=-1,
                  objective="binary:logistic").fit(X, y)
print("  eval/train venv    :", v, "| cuda:", torch.cuda.is_available(), "| gpu_hist OK")
PYEOF
    then
        echo "PREFLIGHT FAILED: eval venv ${EVAL_PYTHON}" >&2; return 1
    fi
    case ",${STAGES}," in
        *eval_quality*)
            if ! "${SYNTHCITY_PYTHON}" - "${REPO_ROOT}" <<'PYEOF'
import sys
sys.path.insert(0, f"{sys.argv[1]}/tabdiff_gnn")
import config as C, tabdiff_eval as TE
v = TE._versions(["synthcity", "pandas", "numpy", "sklearn", "torch"])
TE._check_pins(v, C.S.QUALITY_PINS)
from synthcity.metrics import eval_statistical  # noqa: F401
print("  synthcity venv     :", v)
PYEOF
            then echo "PREFLIGHT FAILED: synthcity venv ${SYNTHCITY_PYTHON}" >&2; return 1; fi ;;
    esac
    case ",${STAGES}," in
        *eval_tabpc*)
            if ! "${TABPC_PYTHON}" - "${REPO_ROOT}" <<'PYEOF'
import sys
sys.path.insert(0, f"{sys.argv[1]}/tabdiff_gnn")
import config as C, tabpc_eval as TP
C.verify_tabpc(sys.argv[1])
v = TP._versions(); TP.check_pins(v)
TP.import_tabpc(sys.argv[1])
print("  tabpc venv         :", v)
PYEOF
            then echo "PREFLIGHT FAILED: tabpc venv ${TABPC_PYTHON}" >&2; return 1; fi ;;
    esac
    echo "  route-A preflight  : OK"
}

run_route_a() {
    cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; return 1; }
    echo "===================================================================="
    echo "  study        : tabdiff_gnn_route_a"
    echo "  dataset      : ${DATASET}   denoiser: ${DENOISER}   seed: ${SEED}"
    echo "  stages       : ${STAGES}${TRAIN_BUDGET_H:+   (resumable training, budget ${TRAIN_BUDGET_H} h)}"
    echo "  results tree : ${TRANSFER_ROOT}/exp_tabdiff_gnn/"
    echo "  pythons      : train ${PYTHON_BIN} | eval ${EVAL_PYTHON}"
    echo "                 synthcity ${SYNTHCITY_PYTHON} | tabpc ${TABPC_PYTHON}"
    echo "  node         : $(hostname)   gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    echo "  started      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "===================================================================="
    if ! route_a_preflight; then
        echo "  ABORTED: preflight failed -- nothing was run" >&2
        return 1
    fi
    local status
    _route_a_args
    srun "${PYTHON_BIN}" "${REPO_ROOT}/${TG_DIR}/runner.py" "${RA_ARGS[@]}" \
        --stages "${STAGES}" --device "${DEVICE}" \
        --train_python "${PYTHON_BIN}" --eval_python "${EVAL_PYTHON}" \
        --synthcity_python "${SYNTHCITY_PYTHON}" --tabpc_python "${TABPC_PYTHON}"
    status=$?
    if [ ${status} -ne 0 ]; then
        echo "  [FAIL] ${DATASET}/${DENOISER}/seed${SEED} stages ${STAGES} exited ${status} -- finished"
        echo "         stages keep their markers; resubmit the same script to continue"
        mkdir -p "${REPO_ROOT}/hpc_transfer/logs"
        printf '%s\t%s\t%s\tseed%s\t%s\texit=%d\tjob=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DATASET}" \
            "${DENOISER}" "${SEED}" "${STAGES}" "${status}" "${SLURM_JOB_ID:-none}" \
            >> "${REPO_ROOT}/hpc_transfer/logs/tabdiff_gnn_failures.txt"
        return "${status}"
    fi
    echo "  [OK] ${DATASET}/${DENOISER}/seed${SEED} stages done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

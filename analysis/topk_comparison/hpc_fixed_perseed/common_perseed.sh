# shellcheck shell=bash
#
# Fixed-mode PER-SEED rerun of the 9 ablated datasets' selected configuration,
# at the SAME TRAINING SEED 0 as the original grid.
#
# Why: the original fixed-grid JSONs store only mean/std per configuration, so
# the development-set comparison has no per-generation-seed fixed values (no
# CIs, no win/tie/loss). This rerun writes per_seed_scores.csv.
#
# What it does NOT do: it removes neither the seed winner's curse (the
# configuration is the argmax over the grid at training seed 0) nor the
# test-split selection bias (that argmax was taken on the test-split score).
# The development set stays in-sample either way.
#
# How: reuses hpc_transfer/common.sh::run_transfer unchanged, with
#   TOPK_MODE=fixed, SEED=0, and TRANSFER_ROOT pointed at a staging tree.
# ablation_runner always writes fixed mode to <exp_root>/exp/<ds>/ablation/,
# so after a successful run the combo directory is MOVED to
#   ${RESULTS_ROOT}/exp/<ds>/ablation_fixed_seed0/<combo>/
# which is where analysis/topk_comparison/compare_topk.py looks for it.
# Nothing under ${RESULTS_ROOT}/exp/<ds>/ablation/ is touched.
#
# Sourced by analysis/topk_comparison/hpc_fixed_perseed/<ds>_fixed_perseed.sh
# after it has set HPC_ROOT REPO_ROOT PYTHON_BIN RESULTS_ROOT DATASET DEVICE.

run_fixed_perseed() {
    local cfg="${REPO_ROOT}/hpc_transfer/configs/${DATASET}.sh"
    if [ ! -f "${cfg}" ]; then
        echo "ERROR: no config for ${DATASET} at ${cfg}" >&2
        return 1
    fi
    # shellcheck source=/dev/null
    . "${cfg}"
    local suffix=""
    [ "${USE_ATTENTION}" = "0" ] && suffix="_noattn"
    local dir_name="gnn_${GNN_TYPE}_layers${N_LAYERS}_dim${D_MODEL}_topk${TOP_K}_heads${N_HEADS}${suffix}"

    local stage_root="${RESULTS_ROOT}/_staging_fixed_seed0"
    local staged="${stage_root}/exp/${DATASET}/ablation/${dir_name}"
    local final="${RESULTS_ROOT}/exp/${DATASET}/ablation_fixed_seed0/${dir_name}"

    if [ -f "${final}/results_full_averaged.json" ]; then
        echo "  [SKIP] ${final} already complete"
        return 0
    fi

    # run_transfer reads these; TRANSFER_ROOT is the staging tree on purpose.
    TOPK_MODE=fixed
    SEED=0
    TRANSFER_ROOT="${stage_root}"
    TOPK_WARMUP_STEPS=0
    K_MIN=1
    TOPK_LOG_EVERY=100
    export TOPK_MODE SEED TRANSFER_ROOT TOPK_WARMUP_STEPS K_MIN TOPK_LOG_EVERY
    PROVENANCE="fixed per-seed rerun, training seed 0 (same as the grid); ${PROVENANCE:-}"
    export PROVENANCE

    # shellcheck source=/dev/null
    . "${REPO_ROOT}/hpc_transfer/common.sh"
    run_transfer || return $?

    if [ ! -f "${staged}/results_full_averaged.json" ]; then
        echo "  [FAIL] ${staged}/results_full_averaged.json missing after run" >&2
        return 1
    fi
    if [ -e "${final}" ]; then
        echo "  [FAIL] ${final} exists but is incomplete -- inspect it; not overwriting" >&2
        return 1
    fi
    mkdir -p "$(dirname "${final}")"
    mv "${staged}" "${final}"
    # Mark the study in run_meta.json so the directory is never ambiguous.
    "${PYTHON_BIN}" - "${final}/run_meta.json" <<'PYEOF'
import json, sys
p = sys.argv[1]
m = json.load(open(p))
m.update(study="fixed_perseed", results_subdir="ablation_fixed_seed0", seed=0)
json.dump(m, open(p, "w"), indent=2)
PYEOF
    echo "  [OK] moved to ${final}"
}

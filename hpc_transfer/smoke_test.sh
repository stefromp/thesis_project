#!/usr/bin/env bash
#
# GPU check, then one very short end-to-end run of a target's own
# configuration. Mirrors hpc/smoke_test.sh.
#
# It goes through the same code path as a real job -- same entry point, same
# single-combination tuple, same run_ablation call -- but with 50 training
# steps, 1 x 1 seeds and the synthetic rows capped, and it writes to
# $HPC_ROOT/transfer_smoke_out, which nothing reads and which can never be
# mistaken for a real result.
#
# The row cap goes through hpc_wide/run_combo.py, which applies it to an
# in-memory COPY of DATASET_CFG. No file under ablation/ is modified.
#
#   salloc --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=26 \
#          --mem=120G --time=01:00:00 --gres=<the string sinfo reports>
#
#   ./hpc_transfer/smoke_test.sh                                  # king, fixed
#   ./hpc_transfer/smoke_test.sh king adaptive_jenks
#   ./hpc_transfer/smoke_test.sh wilt adaptive_entmax15           # cheapest entmax check
#   ./hpc_transfer/smoke_test.sh fb-comments adaptive_sparsemax   # widest graph
#   GPU_CHECK_ONLY=1 ./hpc_transfer/smoke_test.sh
#
# Run it in EVERY rule you intend to submit. The adaptive path goes through
# adaptive_adjacency.build_adjacency, which the fixed path never touches, and
# the two entmax rules additionally need the `entmax` package
# (pip install entmax==1.3) -- not in requirements.txt.
#
# Note the elapsed time it prints: that is what you scale to a real run
# (x steps/50 for training, and by num_samples/cap x 5 for sampling).

set -uo pipefail

HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabddpm310/bin/python}"
SMOKE_OUT="${SMOKE_OUT:-${HPC_ROOT}/transfer_smoke_out}"

DATASET="${1:-king}"
TOPK_MODE="${2:-${TOPK_MODE:-fixed}}"
case "${TOPK_MODE}" in
    fixed|adaptive_jenks|adaptive_sparsemax|adaptive_entmax15) ;;
    *) echo "usage: $0 [dataset] [fixed|adaptive_jenks|adaptive_sparsemax|adaptive_entmax15]" >&2
       exit 2 ;;
esac
SMOKE_STEPS="${SMOKE_STEPS:-50}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-2000}"
DEVICE="${DEVICE:-cuda:0}"

echo "== GPU =="
"${PYTHON_BIN}" - <<'PY'
import torch, sys
print("torch                :", torch.__version__)
print("cuda available       :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device               :", torch.cuda.get_device_name(0))
else:
    print("NO GPU VISIBLE -- aborting", file=sys.stderr)
    sys.exit(1)
PY
status=$?
if [ ${status} -ne 0 ]; then
    echo "GPU not visible from this node; not running anything." >&2
    exit 1
fi

if [ -n "${GPU_CHECK_ONLY:-}" ]; then
    echo "GPU_CHECK_ONLY set; stopping here."
    exit 0
fi

cfg="${REPO_ROOT}/hpc_transfer/configs/${DATASET}.sh"
if [ ! -f "${cfg}" ]; then
    echo "ERROR: no config for ${DATASET} at ${cfg}" >&2
    exit 1
fi
# shellcheck source=/dev/null
. "${cfg}"

echo
echo "== preflight (configs, shapes, baseline pairing) =="
"${PYTHON_BIN}" "${REPO_ROOT}/hpc_transfer/preflight.py" || exit 1

case "${TOPK_MODE}" in
    adaptive_sparsemax|adaptive_entmax15)
        echo
        echo "== preflight (${TOPK_MODE}: entmax package + one forward) =="
        # shellcheck source=common.sh
        . "${REPO_ROOT}/hpc_transfer/common.sh"
        entmax_preflight "${TOPK_MODE}" || exit 1
        ;;
esac

# The adaptive flags go on the command line only in adaptive mode, exactly as
# hpc_transfer/common.sh does it. Built as an array with the ${a[@]+"${a[@]}"}
# idiom: under `set -u`, bash before 4.4 aborts on a plain "${a[@]}" when the
# array is empty.
topk_flags=()
if [ "${TOPK_MODE}" != "fixed" ]; then
    topk_flags=(--topk_mode "${TOPK_MODE}"
                --k_min "${K_MIN:-1}"
                --topk_warmup_steps "${TOPK_WARMUP_STEPS:-0}"
                --topk_log_every "${TOPK_LOG_EVERY:-100}")
fi

echo
echo "== ${DATASET}: gnn_${GNN_TYPE}_layers${N_LAYERS}_dim${D_MODEL}_topk${TOP_K}_heads${N_HEADS}_noattn =="
echo "   topk_mode ${TOPK_MODE}"
echo "   ${SMOKE_STEPS} steps, 1x1 seeds, <= ${SMOKE_MAX_SAMPLES} synthetic rows"
echo "   output -> ${SMOKE_OUT}"
echo

start=$(date +%s)
"${PYTHON_BIN}" "${REPO_ROOT}/hpc_wide/run_combo.py" \
    --dataset "${DATASET}" \
    --max_num_samples "${SMOKE_MAX_SAMPLES}" \
    --gnn_type "${GNN_TYPE}" \
    --n_layers "${N_LAYERS}" \
    --d_model "${D_MODEL}" \
    --top_k "${TOP_K}" \
    --n_heads "${N_HEADS}" \
    --use_attention "${USE_ATTENTION}" \
    ${topk_flags[@]+"${topk_flags[@]}"} \
    --steps "${SMOKE_STEPS}" \
    --n_gen_seeds 1 \
    --n_clf_seeds 1 \
    --repo_root "${REPO_ROOT}" \
    --data_root "${REPO_ROOT}" \
    --exp_root "${SMOKE_OUT}" \
    --device "${DEVICE}" \
    --seed 0
status=$?
end=$(date +%s)

echo
echo "elapsed: $((end - start)) s   (exit ${status})"
exit ${status}

#!/bin/bash
#
# Interactive smoke test: prove the GPU is visible, then run ONE combination
# through the real code path with a tiny budget.
#
# Run it inside an interactive allocation, e.g.
#
#     salloc --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=26 \
#            --mem=120G --time=01:00:00 --gres=<the string sinfo reports>
#     ./hpc/smoke_test.sh
#
#   ./hpc/smoke_test.sh                  # default: wilt, first adaptive run
#   ./hpc/smoke_test.sh gesture          # a different dataset
#   STEPS=20 ./hpc/smoke_test.sh         # even smaller budget
#   GPU_CHECK_ONLY=1 ./hpc/smoke_test.sh # stop after the GPU report
#
# The GPU check runs FIRST and aborts on failure: there is no point spending
# an allocation on a run that would silently fall back to CPU.
#
# This writes into a SEPARATE output tree (exp/<ds>/ablation_adaptive_jenks/
# under SMOKE_EXP_ROOT, default $HPC_ROOT/smoke_test_out) so it can never be
# mistaken for, or collide with, a real result directory.

set -uo pipefail

HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabddpm310/bin/python}"
SMOKE_EXP_ROOT="${SMOKE_EXP_ROOT:-${HPC_ROOT}/smoke_test_out}"

DATASET="${1:-wilt}"

# Tiny budget: this is a plumbing check, not an experiment.
STEPS="${STEPS:-50}"
N_GEN_SEEDS="${N_GEN_SEEDS:-1}"
N_CLF_SEEDS="${N_CLF_SEEDS:-1}"

TOPK_WARMUP_STEPS="${TOPK_WARMUP_STEPS:-0}"
K_MIN="${K_MIN:-1}"
TOPK_LOG_EVERY="${TOPK_LOG_EVERY:-10}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
GPU_CHECK_ONLY="${GPU_CHECK_ONLY:-0}"

echo "===================================================================="
echo "  1. Environment / GPU visibility"
echo "===================================================================="
echo "  host       : $(hostname)"
echo "  python     : ${PYTHON_BIN}"
echo "  repo       : ${REPO_ROOT}"
echo "  exp root   : ${SMOKE_EXP_ROOT}   (smoke output, not real results)"
echo

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

echo "  nvidia-smi:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | sed 's/^/    /'
else
    echo "    nvidia-smi not found on this node"
fi
echo

"${PYTHON_BIN}" - <<'PYEOF'
import sys
import torch

print(f"    torch.__version__        = {torch.__version__}")
print(f"    torch.version.cuda       = {torch.version.cuda}")
print(f"    torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    torch.cuda.device_count() = {torch.cuda.device_count()}")
    print(f"    torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
    import numpy as np
    print(f"    numpy                    = {np.__version__}")
else:
    print()
    print("    ERROR: no GPU visible to torch on this node.", file=sys.stderr)
    print("    Check that the allocation actually requested a GPU "
          "(--gres) and that cuda/11.8 is loaded.", file=sys.stderr)
    sys.exit(2)
PYEOF
gpu_status=$?

if [ ${gpu_status} -ne 0 ]; then
    echo
    echo "ABORTING: GPU check failed (exit ${gpu_status}). Nothing was run." >&2
    exit ${gpu_status}
fi

echo
echo "  GPU check passed."

if [ "${GPU_CHECK_ONLY}" = "1" ]; then
    echo "  GPU_CHECK_ONLY=1 -> stopping here."
    exit 0
fi

echo
echo "===================================================================="
echo "  2. One combination, tiny budget"
echo "===================================================================="

# Take the FIRST adaptive run of the dataset, so the smoke test exercises a
# combination that is actually in the real job's list.
combo="$("${PYTHON_BIN}" "${REPO_ROOT}/hpc/adaptive_manifest.py" \
            --repo_root "${REPO_ROOT}" --dataset "${DATASET}" | head -n1)"

if [ -z "${combo}" ]; then
    echo "ERROR: no adaptive runs found for dataset ${DATASET}" >&2
    exit 1
fi

IFS=$'\t' read -r gnn_type n_layers d_model top_k n_heads use_attention baseline_top_ks <<< "${combo}"

echo "  dataset        : ${DATASET}"
echo "  gnn_type       : ${gnn_type}"
echo "  n_layers       : ${n_layers}"
echo "  d_model        : ${d_model}"
echo "  top_k          : ${top_k}   (warm-up only; collapses baseline ${baseline_top_ks})"
echo "  n_heads        : ${n_heads}"
echo "  use_attention  : ${use_attention}"
echo "  steps          : ${STEPS}   (real runs use the module default, 20000)"
echo "  seeds          : ${N_GEN_SEEDS} gen x ${N_CLF_SEEDS} clf   (real runs: 5 x 10)"
echo

mkdir -p "${SMOKE_EXP_ROOT}"

start=$(date +%s)

"${PYTHON_BIN}" "${REPO_ROOT}/ablation/${DATASET}_ablation.py" \
    --gnn_type "${gnn_type}" \
    --n_layers "${n_layers}" \
    --d_model "${d_model}" \
    --top_k "${top_k}" \
    --n_heads "${n_heads}" \
    --use_attention "${use_attention}" \
    --topk_mode adaptive_jenks \
    --k_min "${K_MIN}" \
    --topk_warmup_steps "${TOPK_WARMUP_STEPS}" \
    --topk_log_every "${TOPK_LOG_EVERY}" \
    --steps "${STEPS}" \
    --n_gen_seeds "${N_GEN_SEEDS}" \
    --n_clf_seeds "${N_CLF_SEEDS}" \
    --repo_root "${REPO_ROOT}" \
    --data_root "${REPO_ROOT}" \
    --exp_root "${SMOKE_EXP_ROOT}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --no_skip
status=$?

end=$(date +%s)
elapsed=$((end - start))

echo
echo "===================================================================="
echo "  3. Result"
echo "===================================================================="
echo "  exit status : ${status}"
echo "  elapsed     : ${elapsed}s for ${STEPS} steps + ${N_GEN_SEEDS}x${N_CLF_SEEDS} eval"

suffix=""
[ "${use_attention}" = "0" ] && suffix="_noattn"
dir_name="gnn_${gnn_type}_layers${n_layers}_dim${d_model}_topk${top_k}_heads${n_heads}${suffix}"
out_dir="${SMOKE_EXP_ROOT}/exp/${DATASET}/ablation_adaptive_jenks/${dir_name}"

echo "  output dir  : ${out_dir}"
for f in results_full_averaged.json topk_stats.csv; do
    if [ -f "${out_dir}/${f}" ]; then
        echo "    [OK]      ${f}"
    else
        echo "    [MISSING] ${f}"
    fi
done

if [ -f "${out_dir}/topk_stats.csv" ]; then
    echo
    echo "  topk_stats.csv (head):"
    head -n 6 "${out_dir}/topk_stats.csv" | sed 's/^/    /'
    echo "    ... $(wc -l < "${out_dir}/topk_stats.csv") lines total"
fi

echo
if [ ${status} -eq 0 ]; then
    echo "  SMOKE TEST PASSED."
    echo
    echo "  Scale the elapsed time above to estimate a real run:"
    echo "    real steps / ${STEPS} for training, x50 for the 5x10 seed protocol."
else
    echo "  SMOKE TEST FAILED (exit ${status})." >&2
fi
exit ${status}

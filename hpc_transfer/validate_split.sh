#!/usr/bin/env bash
#
# Point 5 of the split spec, on real numbers: a monolithic run and a split run
# of the same combination must give identical per-seed metrics.
#
# The local test (hpc_transfer/test_split_runner_backcompat.py) proves the
# orchestration -- call order, seeds, directory names, merge -- with every
# heavy function stubbed. It cannot prove that the GPU produces the same
# samples. This script does, by running both paths for real at a tiny budget.
#
#   ./hpc_transfer/validate_split.sh wilt
#   ./hpc_transfer/validate_split.sh wilt adaptive_entmax15
#   STEPS=20 MAX_SAMPLES=200 ./hpc_transfer/validate_split.sh wilt
#
# Run it inside an salloc; it needs the GPU the real jobs use. gpu:1 is
# rejected on this cluster -- use gpu:a100:1 (GRES overrides it):
#
#   salloc --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=26 \
#          --mem=120G --time=01:00:00 --gres=gpu:a100:1
#
# It writes only under $HPC_ROOT/split_validate and submits nothing.
#
# Exit 0 = identical. Exit 1 = they differ, and the diff is printed.

set -uo pipefail

HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabddpm310/bin/python}"
OUT="${OUT:-${HPC_ROOT}/split_validate}"

DATASET="${1:-wilt}"
TOPK_MODE="${2:-adaptive_entmax15}"
STEPS="${STEPS:-30}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
GEN_SEEDS="${GEN_SEEDS:-2}"      # how many generation seeds to compare
CLF_SEEDS="${CLF_SEEDS:-2}"      # classifier seeds per generated set
DEVICE="${DEVICE:-cuda:0}"
GRES="${GRES:-gpu:a100:1}"   # only printed in the hint below; salloc is yours

CFG="${REPO_ROOT}/hpc_transfer/configs/${DATASET}.sh"
[ -f "${CFG}" ] || { echo "ERROR: no config at ${CFG}" >&2; exit 2; }
# shellcheck source=/dev/null
. "${CFG}"

SPLIT="${REPO_ROOT}/hpc_transfer/split_runner.py"
COMBO="--gnn_type ${GNN_TYPE} --n_layers ${N_LAYERS} --d_model ${D_MODEL}"
COMBO="${COMBO} --top_k ${TOP_K} --n_heads ${N_HEADS} --use_attention ${USE_ATTENTION}"
if [ "${TOPK_MODE}" != "fixed" ]; then
    COMBO="${COMBO} --topk_mode ${TOPK_MODE} --k_min ${K_MIN:-1}"
    COMBO="${COMBO} --topk_warmup_steps ${TOPK_WARMUP_STEPS:-0}"
fi

MONO="${OUT}/mono"
SPLT="${OUT}/split"
rm -rf "${MONO}" "${SPLT}"
mkdir -p "${MONO}" "${SPLT}"

echo "===================================================================="
echo "  dataset     : ${DATASET}"
echo "  rule        : ${TOPK_MODE}"
echo "  budget      : ${STEPS} steps, <= ${MAX_SAMPLES} rows, "
echo "                ${GEN_SEEDS} generation seeds x ${CLF_SEEDS} clf seeds"
echo "  output      : ${OUT}"
echo "===================================================================="

echo
echo "== GPU =="
"${PYTHON_BIN}" - <<'PY'
import sys, torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    print("NO GPU -- results would not reflect the real jobs", file=sys.stderr)
    sys.exit(1)
try:
    import zero, inspect
    src = inspect.getsource(zero.improve_reproducibility)
    print("\n-- zero.improve_reproducibility --")
    print(src)
    for flag in ("deterministic", "benchmark", "use_deterministic_algorithms"):
        print(f"  sets {flag:30s}: {flag in src}")
except Exception as exc:
    print("could not inspect zero:", exc)
PY
if [ $? -ne 0 ]; then
    echo "aborting: no GPU." >&2
    echo "Get one with:" >&2
    echo "  salloc --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=26 \\" >&2
    echo "         --mem=120G --time=01:00:00 --gres=${GRES}" >&2
    exit 1
fi

# Budget: --steps / --n_gen_seeds / --n_clf_seeds go through
# ablation_runner.apply_cli_overrides, the same path run_ablation uses, so no
# module is patched behind anyone's back.
BUDGET="--steps ${STEPS} --n_gen_seeds ${GEN_SEEDS} --n_clf_seeds ${CLF_SEEDS}"

echo
echo "== 1/2 monolithic: all ${GEN_SEEDS} generation seeds in one process =="
"${PYTHON_BIN}" "${SPLIT}" --dataset "${DATASET}" ${COMBO} \
    --repo_root "${REPO_ROOT}" --data_root "${REPO_ROOT}" --exp_root "${MONO}" \
    --device "${DEVICE}" ${BUDGET} || { echo "monolithic run failed" >&2; exit 1; }

echo
echo "== 2/2 split: train + seed 0, then seeds 1.. loaded =="
"${PYTHON_BIN}" "${SPLIT}" --dataset "${DATASET}" ${COMBO} \
    --repo_root "${REPO_ROOT}" --data_root "${REPO_ROOT}" --exp_root "${SPLT}" \
    --device "${DEVICE}" ${BUDGET} --gen_seed_ids 0 \
    || { echo "split seed 0 failed" >&2; exit 1; }
k=1
while [ ${k} -lt "${GEN_SEEDS}" ]; do
    "${PYTHON_BIN}" "${SPLIT}" --dataset "${DATASET}" ${COMBO} \
        --repo_root "${REPO_ROOT}" --data_root "${REPO_ROOT}" --exp_root "${SPLT}" \
        --device "${DEVICE}" ${BUDGET} --gen_seed_ids ${k} --load_checkpoint \
        || { echo "split seed ${k} failed" >&2; exit 1; }
    k=$((k + 1))
done

echo
echo "== compare =="
"${PYTHON_BIN}" - "${MONO}" "${SPLT}" <<'PY'
import csv, json, os, sys

mono, splt = sys.argv[1], sys.argv[2]

def find(root, name):
    for d, _s, f in os.walk(root):
        if name in f:
            return os.path.join(d, name)
    return None

bad = 0
for name in ("per_seed_scores.csv", "results_full_averaged.json"):
    a, b = find(mono, name), find(splt, name)
    if not (a and b):
        print(f"  MISSING {name}: mono={bool(a)} split={bool(b)}")
        bad += 1
        continue
    if name.endswith(".csv"):
        ra = sorted(map(tuple, csv.reader(open(a))))
        rb = sorted(map(tuple, csv.reader(open(b))))
        if ra == rb:
            print(f"  OK   {name}  ({len(ra) - 1} rows identical)")
        else:
            bad += 1
            print(f"  DIFF {name}")
            for row in sorted(set(ra) - set(rb))[:6]:
                print(f"      only monolithic: {row}")
            for row in sorted(set(rb) - set(ra))[:6]:
                print(f"      only split     : {row}")
    else:
        ja, jb = json.load(open(a)), json.load(open(b))
        keys = sorted(set(ja) | set(jb))
        diff = [k for k in keys if ja.get(k) != jb.get(k)]
        if not diff:
            print(f"  OK   {name}  ({len(keys)} metrics identical)")
        else:
            bad += 1
            print(f"  DIFF {name}")
            for k in diff[:8]:
                print(f"      {k}: mono={ja.get(k)} split={jb.get(k)}")

print()
if bad:
    print("FAILED: the split run does not reproduce the monolithic run.")
    print("If only the last decimals differ, the cause is GPU nondeterminism,")
    print("not the split: check whether zero.improve_reproducibility sets")
    print("cudnn.deterministic=True and benchmark=False (printed above).")
else:
    print("PASSED: identical per-seed rows and identical aggregated metrics.")
sys.exit(1 if bad else 0)
PY
status=$?
echo
echo "artifacts kept under ${OUT} (nothing was submitted)"
exit ${status}

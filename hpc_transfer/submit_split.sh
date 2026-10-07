#!/usr/bin/env bash
#
# Split one dataset's generation seeds across SLURM jobs.
#
# PRINTS THE COMMANDS BY DEFAULT. Nothing is submitted without --submit.
#
# Job A trains and runs generation seed 0. Jobs B..E run seeds 1..4, each with
# --dependency=afterok:<A> and --load_checkpoint, so they reuse A's
# model_ema.pt instead of retraining. A final merge job (--dependency=afterok
# on all of them) rebuilds results_full_averaged.json and per_seed_scores.csv.
#
# Why this is equivalent to one monolithic job: scripts/sample.py opens with
# zero.improve_reproducibility(seed) and ablation_runner passes seed=gen_seed,
# so generation seed k depends only on k and on the saved weights -- never on
# RNG state left by seeds 0..k-1. hpc_transfer/test_split_runner_backcompat.py
# proves the orchestration matches; hpc_transfer/validate_split.sh checks the
# numbers on a GPU.
#
#   ./hpc_transfer/submit_split.sh miniboone_entmax.sh            # print
#   ./hpc_transfer/submit_split.sh miniboone_entmax.sh --submit
#   ./hpc_transfer/submit_split.sh miniboone_entmax.sh --resume    # missing seeds only
#   ./hpc_transfer/submit_split.sh miniboone_entmax.sh --seeds 2,3
#
# The job script is an explicit path, or a name under hpc_transfer/jobs/ or
# hpc_transfer/jobs_catfix/; its DATASET, TOPK_MODE and CAT_INPUT are read from
# it, and the configuration from hpc_transfer/configs/<ds>.sh -- nothing is
# duplicated here. CAT_INPUT=prob (the categorical-input fix) adds
# --cat_input prob to every split_runner call, so scan, seeds and merge all
# use the *_catfix tree (ablation_runner.results_subdir).
#
#   ./hpc_transfer/submit_split.sh jobs_catfix/default_entmax_catfix.sh
#
# --single_job (needs an existing checkpoint): ONE job per run instead of one
# per seed -- it loads the checkpoint, runs every missing generation seed in
# sequence and then merges, all in the same allocation (TIME, MEM, GRES as for
# a seed job). Meant for small/medium datasets whose 5 seeds fit in one
# walltime, e.g. the catfix reruns; the per-seed chain stays the default.
#
#   ./hpc_transfer/submit_split.sh jobs_catfix/insurance_fixed_catfix.sh --single_job
#
# Every --wrap command starts with a cd to the repo root, so the directory
# sbatch is called from does not matter.

set -uo pipefail

HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabddpm310/bin/python}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${HPC_ROOT}/transfer_runs}"
JOBS_OUT="${JOBS_OUT:-${HPC_ROOT}/jobs_out}"

MEM="${MEM:-24G}"
GRES="${GRES:-gpu:a100:1}"
PARTITION="${PARTITION:-gpu}"
CPUS="${CPUS:-26}"

# The merge job needs no GPU -- it concatenates shards and re-averages. But
# this cluster refuses a job on the gpu partition with no GRES ("No GRES
# specified for GPU partition gpu"), so it still asks for one; at 30 minutes
# and 1 CPU it holds the card only briefly. Point MERGE_PARTITION at a CPU
# queue and MERGE_GRES is dropped automatically.
MERGE_PARTITION="${MERGE_PARTITION:-${PARTITION}}"
MERGE_TIME="${MERGE_TIME:-00:30:00}"
MERGE_MEM="${MERGE_MEM:-8G}"
if [ "${MERGE_PARTITION}" = "${PARTITION}" ]; then
    MERGE_GRES="${MERGE_GRES:-${GRES}}"
else
    MERGE_GRES="${MERGE_GRES:-}"
fi

JOB_REL=""
DO_SUBMIT=0
DO_RESUME=0
FORCE_TRAIN_ONLY=0
SINGLE_JOB=0
SEEDS=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --submit) DO_SUBMIT=1 ;;
        --resume) DO_RESUME=1 ;;
        --train-only|--train_only) FORCE_TRAIN_ONLY=1 ;;
        --single-job|--single_job) SINGLE_JOB=1 ;;
        --seeds)  shift; SEEDS="${1:-}" ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) JOB_REL="$1" ;;
    esac
    shift
done

if [ -z "${JOB_REL}" ]; then
    echo "usage: $0 <job script: path, or name in hpc_transfer/jobs{,_catfix}/> [--submit] [--resume] [--seeds 1,2] [--single_job]" >&2
    echo "       e.g. $0 miniboone_entmax.sh --submit" >&2
    echo "            $0 jobs_catfix/default_entmax_catfix.sh" >&2
    exit 2
fi

# An explicit path wins; otherwise look the name up in jobs/, then jobs_catfix/.
JOB=""
for cand in "${JOB_REL}" \
            "${REPO_ROOT}/hpc_transfer/${JOB_REL}" \
            "${REPO_ROOT}/hpc_transfer/jobs/$(basename "${JOB_REL}")" \
            "${REPO_ROOT}/hpc_transfer/jobs_catfix/$(basename "${JOB_REL}")" \
            "${REPO_ROOT}/hpc_transfer/jobs_tddpm_setup/$(basename "${JOB_REL}")"; do
    if [ -f "${cand}" ]; then
        JOB="$(cd "$(dirname "${cand}")" && pwd)/$(basename "${cand}")"
        break
    fi
done
if [ -z "${JOB}" ]; then
    echo "ERROR: no such job script: ${JOB_REL} (looked in hpc_transfer/jobs/, jobs_catfix/ and jobs_tddpm_setup/)" >&2
    exit 2
fi

# DATASET and TOPK_MODE come from the job script itself.
DATASET="$(sed -n 's/^DATASET="\(.*\)"$/\1/p' "${JOB}" | head -n1)"
TOPK_MODE="$(sed -n 's/^TOPK_MODE="\${TOPK_MODE:-\(.*\)}"$/\1/p' "${JOB}" | head -n1)"
[ -z "${TOPK_MODE}" ] && TOPK_MODE="$(sed -n 's/^TOPK_MODE="\(.*\)"$/\1/p' "${JOB}" | head -n1)"
if [ -z "${DATASET}" ] || [ -z "${TOPK_MODE}" ]; then
    echo "ERROR: could not read DATASET / TOPK_MODE from ${JOB}" >&2
    exit 2
fi
# CAT_INPUT: absent in the original job scripts (= log, the old behaviour).
CAT_INPUT="$(sed -n 's/^CAT_INPUT="\${CAT_INPUT:-\(.*\)}"$/\1/p' "${JOB}" | head -n1)"
CAT_INPUT="${CAT_INPUT:-log}"
case "${CAT_INPUT}" in
    log|prob) ;;
    *) echo "ERROR: CAT_INPUT must be log or prob in ${JOB}, got '${CAT_INPUT}'" >&2; exit 2 ;;
esac
# TRAIN_SETUP: absent in every job script before jobs_tddpm_setup/ (= ablation).
TRAIN_SETUP="$(sed -n 's/^TRAIN_SETUP="\${TRAIN_SETUP:-\(.*\)}"$/\1/p' "${JOB}" | head -n1)"
TRAIN_SETUP="${TRAIN_SETUP:-ablation}"
case "${TRAIN_SETUP}" in
    ablation|tabddpm) ;;
    *) echo "ERROR: TRAIN_SETUP must be ablation or tabddpm in ${JOB}, got '${TRAIN_SETUP}'" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Per-dataset walltime and chain shape.
#
# fb-comments is the only one that needs more than 24 h: one generation seed
# alone is 14.4 - 35.3 h (1.264 M synthetic rows x 1000 diffusion steps), so it
# gets the cluster maximum of 48 h. It also gets the TRAIN-ONLY chain, because
# training (4.5 - 11.3 h) plus one seed could still overrun 48 h together --
# splitting them leaves training on its own and every seed in its own job.
#
# TIME=... and TRAIN_ONLY=1 in the environment override both.
# ---------------------------------------------------------------------------
case "${DATASET}" in
    fb-comments)
        DEFAULT_TIME="48:00:00"
        DEFAULT_TRAIN_ONLY=1
        ;;
    *)
        DEFAULT_TIME="24:00:00"
        DEFAULT_TRAIN_ONLY=0
        ;;
esac
TIME="${TIME:-${DEFAULT_TIME}}"
TRAIN_ONLY="${TRAIN_ONLY:-${DEFAULT_TRAIN_ONLY}}"
[ "${FORCE_TRAIN_ONLY}" = "1" ] && TRAIN_ONLY=1

CFG="${REPO_ROOT}/hpc_transfer/configs/${DATASET}.sh"
if [ ! -f "${CFG}" ]; then
    echo "ERROR: no config at ${CFG}" >&2
    exit 2
fi
# shellcheck source=/dev/null
. "${CFG}"

COMBO_ARGS="--gnn_type ${GNN_TYPE} --n_layers ${N_LAYERS} --d_model ${D_MODEL}"
COMBO_ARGS="${COMBO_ARGS} --top_k ${TOP_K} --n_heads ${N_HEADS}"
COMBO_ARGS="${COMBO_ARGS} --use_attention ${USE_ATTENTION}"
if [ "${TOPK_MODE}" != "fixed" ]; then
    COMBO_ARGS="${COMBO_ARGS} --topk_mode ${TOPK_MODE} --k_min ${K_MIN:-1}"
    COMBO_ARGS="${COMBO_ARGS} --topk_warmup_steps ${TOPK_WARMUP_STEPS:-0}"
    COMBO_ARGS="${COMBO_ARGS} --topk_log_every ${TOPK_LOG_EVERY:-100}"
fi
# Only for the fix, so a log-input chain keeps its exact command lines.
[ "${CAT_INPUT}" = "prob" ] && COMBO_ARGS="${COMBO_ARGS} --cat_input prob"
# Same for the budget-matched setup: scan, seeds and merge all use the
# *_tddpm_setup tree, and --load_checkpoint refuses an ablation-setup model.
[ "${TRAIN_SETUP}" = "tabddpm" ] && COMBO_ARGS="${COMBO_ARGS} --train_setup tabddpm"
ROOTS="--repo_root ${REPO_ROOT} --data_root ${REPO_ROOT} --exp_root ${TRANSFER_ROOT}"
SPLIT="${REPO_ROOT}/hpc_transfer/split_runner.py"

echo "===================================================================="
echo "  dataset    : ${DATASET}"
echo "  rule       : ${TOPK_MODE}"
echo "  cat_input  : ${CAT_INPUT}"
echo "  train_setup: ${TRAIN_SETUP}"
echo "  config     : ${GNN_TYPE} L${N_LAYERS} d${D_MODEL} k${TOP_K} h${N_HEADS}"
if [ "${SINGLE_JOB}" = "1" ]; then
    echo "  walltime   : ${TIME}   (seeds + merge in one job)"
else
    echo "  walltime   : ${TIME}   (merge ${MERGE_TIME})"
fi
echo "  job script : ${JOB}"
echo "===================================================================="
echo

# ---- which seeds, and is there a checkpoint -------------------------------
# --scan_shell gives HAS_CHECKPOINT / DONE_SEEDS / MISSING_SEEDS / RUN_DIR.
# Whether a training job is submitted at all depends on HAS_CHECKPOINT: if the
# checkpoint exists, training would overwrite it and the seeds already done
# would have come from different weights.
SCAN="$("${PYTHON_BIN}" "${SPLIT}" --dataset "${DATASET}" ${COMBO_ARGS} ${ROOTS} \
        --device cpu --scan_shell)"
if [ $? -ne 0 ] || [ -z "${SCAN}" ]; then
    echo "ERROR: --scan_shell failed; not emitting any sbatch line." >&2
    exit 1
fi
eval "${SCAN}"
echo "  results    : ${RUN_DIR}"
if [ "${SINGLE_JOB}" = "1" ]; then
    echo "  chain      : --single_job -- one job: load checkpoint, missing seeds in sequence, merge"
elif [ "${HAS_CHECKPOINT}" = "1" ]; then
    echo "  chain      : checkpoint present -- no training, one --load_checkpoint job per seed, in parallel"
else
    echo "  chain      : $([ "${TRAIN_ONLY}" = "1" ] && echo "train-only job, then all seeds in parallel" || echo "first job trains + its own seed")"
fi
echo
echo "-- current state --"
"${PYTHON_BIN}" "${SPLIT}" --dataset "${DATASET}" ${COMBO_ARGS} ${ROOTS} \
    --device cpu --scan
echo

if [ -n "${SEEDS}" ]; then
    SEED_LIST="$(echo "${SEEDS}" | tr ',' ' ')"
    echo "-- seeds given explicitly: ${SEED_LIST} --"
elif [ ${DO_RESUME} -eq 1 ] || [ "${HAS_CHECKPOINT}" = "1" ]; then
    SEED_LIST="${MISSING_SEEDS}"
else
    SEED_LIST="${DONE_SEEDS:+}${MISSING_SEEDS}"
fi

if [ -z "${SEED_LIST// /}" ]; then
    echo "Every generation seed already has metrics. Nothing to submit."
    echo "If they are not merged yet:"
    echo "  ${PYTHON_BIN} ${SPLIT} --dataset ${DATASET} ${COMBO_ARGS} ${ROOTS} --device cpu --merge_only"
    exit 0
fi

# ---------------------------------------------------------------------------
# One sbatch invocation. Prints it; with --submit, runs it and returns the id.
# ---------------------------------------------------------------------------
submit_one() {
    local name="$1" dep="$2" cmd="$3" cpu_only="${4:-0}"
    local -a args
    args=(--parsable
          --job-name="${name}"
          --output="${JOBS_OUT}/${name}_%j.out"
          --error="${JOBS_OUT}/${name}_%j.err"
          --nodes=1 --ntasks=1 --ntasks-per-node=1)
    if [ "${cpu_only}" = "1" ]; then
        # Short, single-CPU, no GPU work -- but the gpu partition rejects a job
        # without a GRES, so MERGE_GRES carries one unless MERGE_PARTITION
        # points somewhere else.
        args+=(--partition="${MERGE_PARTITION}" --cpus-per-task=1
               --time="${MERGE_TIME}" --mem="${MERGE_MEM}")
        [ -n "${MERGE_GRES}" ] && args+=(--gres="${MERGE_GRES}")
    else
        args+=(--partition="${PARTITION}" --cpus-per-task="${CPUS}"
               --time="${TIME}" --mem="${MEM}" --gres="${GRES}")
    fi
    [ -n "${dep}" ] && args+=(--dependency="afterok:${dep}")
    args+=(--wrap "cd $(printf '%q' "${REPO_ROOT}") && ${cmd}")

    if [ "${DO_SUBMIT}" -eq 1 ]; then
        local id
        id="$(sbatch "${args[@]}")"
        local st=$?
        if [ ${st} -ne 0 ] || [ -z "${id}" ]; then
            echo "ERROR: sbatch failed for ${name} (exit ${st})" >&2
            return 1
        fi
        echo "  submitted ${name}  id=${id}${dep:+  after ${dep}}" >&2
        printf '%s' "${id}"
    else
        {
          printf 'sbatch'
          for a in "${args[@]}"; do printf ' %q' "${a}"; done
          printf '\n\n'
        } >&2
        printf 'DRYRUN'
    fi
}

TAG="$(echo "${TOPK_MODE}" | sed 's/adaptive_//; s/entmax15/e15/; s/sparsemax/spm/; s/jenks/jks/')"
PY_SPLIT="${PYTHON_BIN} ${SPLIT} --dataset ${DATASET} ${COMBO_ARGS} ${ROOTS}"

if [ "${SINGLE_JOB}" = "1" ]; then
    # ------------------------------------------------------------------
    # One allocation: load the checkpoint, run the missing seeds one after
    # another (split_runner drops any partial rows of a seed before it
    # re-runs it), then merge. Never trains: without a checkpoint there is
    # nothing to load, so refuse rather than fall back to a training chain.
    # ------------------------------------------------------------------
    if [ "${HAS_CHECKPOINT}" != "1" ]; then
        echo "ERROR: --single_job needs an existing checkpoint (model_ema.pt) in ${RUN_DIR};" >&2
        echo "       none found. Drop --single_job to train with the per-seed chain." >&2
        exit 1
    fi
    SEED_CSV="$(echo ${SEED_LIST} | tr ' ' ',')"
    echo "-- --single_job: one job, checkpoint loaded, seeds ${SEED_CSV} in sequence, then merge --" >&2
    echo "   ${RUN_DIR}/model_ema.pt" >&2
    echo "   seeds already done : ${DONE_SEEDS:--}" >&2
    echo >&2
    submit_one "${DATASET}_${TAG}_all" "" \
        "${PY_SPLIT} --device cuda:0 --gen_seed_ids ${SEED_CSV} --load_checkpoint && ${PY_SPLIT} --device cpu --merge_only" \
        >/dev/null || exit 1
    echo >&2
    if [ "${DO_SUBMIT}" -eq 1 ]; then
        echo "Submitted one job. Watch with:  squeue -u \$USER -o '%.10i %.24j %.8T %.10M %R'" >&2
    else
        echo "Nothing was submitted. Add --submit to run it for real." >&2
    fi
    exit 0
fi

ALL=""
if [ "${HAS_CHECKPOINT}" = "1" ]; then
    # ------------------------------------------------------------------
    # The model already exists. No training job: every seed loads it, so
    # they are independent and run in PARALLEL -- no dependency between
    # them. Seeds already done are not in SEED_LIST.
    # ------------------------------------------------------------------
    echo "-- checkpoint present: no training job --" >&2
    echo "   ${RUN_DIR}/model_ema.pt" >&2
    echo "   seeds already done : ${DONE_SEEDS:--}" >&2
    echo "   submitting         : ${SEED_LIST}" >&2
    echo >&2
    for s in ${SEED_LIST}; do
        id="$(submit_one "${DATASET}_${TAG}_s${s}" "" \
            "${PY_SPLIT} --device cuda:0 --gen_seed_ids ${s} --load_checkpoint")" || exit 1
        ALL="${ALL:+${ALL}:}${id}"
    done
elif [ "${TRAIN_ONLY}" = "1" ]; then
    # ------------------------------------------------------------------
    # No model, and train + one seed would not fit together. Job A trains
    # and nothing else; every generation seed then runs as its own job,
    # all of them after A and all independent of each other.
    # ------------------------------------------------------------------
    echo "-- no checkpoint, TRAIN-ONLY chain --" >&2
    echo "   job A trains and stops; all ${SEED_LIST} run in parallel after it" >&2
    echo >&2
    echo "-- job A: train only --" >&2
    TRAIN_ID="$(submit_one "${DATASET}_${TAG}_trn" "" \
        "${PY_SPLIT} --device cuda:0 --train_only")" || exit 1
    echo "-- seed jobs: after A, checkpoint loaded, independent --" >&2
    for s in ${SEED_LIST}; do
        id="$(submit_one "${DATASET}_${TAG}_s${s}" "${TRAIN_ID}" \
            "${PY_SPLIT} --device cuda:0 --gen_seed_ids ${s} --load_checkpoint")" || exit 1
        ALL="${ALL:+${ALL}:}${id}"
    done
else
    # ------------------------------------------------------------------
    # No model yet. The first job trains and does its own seed; the rest
    # wait for it and load the checkpoint it leaves behind.
    # ------------------------------------------------------------------
    echo "-- no checkpoint: first job trains + its own seed --" >&2
    echo >&2
    FIRST_ID=""
    for s in ${SEED_LIST}; do
        if [ -z "${FIRST_ID}" ]; then
            echo "-- job A: train + generation seed ${s} --" >&2
            FIRST_ID="$(submit_one "${DATASET}_${TAG}_s${s}" "" \
                "${PY_SPLIT} --device cuda:0 --gen_seed_ids ${s}")" || exit 1
            ALL="${FIRST_ID}"
            echo "-- jobs B..: after A, checkpoint loaded --" >&2
        else
            id="$(submit_one "${DATASET}_${TAG}_s${s}" "${FIRST_ID}" \
                "${PY_SPLIT} --device cuda:0 --gen_seed_ids ${s} --load_checkpoint")" || exit 1
            ALL="${ALL}:${id}"
        fi
    done
fi

echo "-- merge, after every seed job --" >&2
submit_one "${DATASET}_${TAG}_mrg" "${ALL}" \
    "${PY_SPLIT} --device cpu --merge_only" 1 >/dev/null || exit 1

echo >&2
if [ "${DO_SUBMIT}" -eq 1 ]; then
    echo "Submitted. Watch with:  squeue -u \$USER -o '%.10i %.24j %.8T %.10M %R'" >&2
    if [ "${HAS_CHECKPOINT}" = "1" ]; then
        echo "All seed jobs are independent, so they queue in parallel." >&2
    else
        echo "Seeds after the first stay PENDING with (Dependency) until it finishes." >&2
    fi
else
    echo "Nothing was submitted (the ids above are placeholders)." >&2
    echo "Add --submit to run the chain for real." >&2
fi

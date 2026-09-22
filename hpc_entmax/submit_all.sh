#!/bin/bash
#
# Submit the sparsemax / entmax-1.5 jobs: one job per dataset, for ONE mode.
#
#   ./hpc_entmax/submit_all.sh adaptive_sparsemax wilt
#   ./hpc_entmax/submit_all.sh adaptive_sparsemax abalone gesture
#   ./hpc_entmax/submit_all.sh adaptive_entmax15 wilt abalone gesture
#   DRY_RUN=1 ./hpc_entmax/submit_all.sh adaptive_entmax15 wilt   # print only
#
# The mode and at least one dataset are both required: nothing is submitted by
# default, so a staged rollout (wilt first) cannot turn into a full one by
# accident. adaptive_jenks / fixed are refused -- those runs belong to hpc/.
#
# Per job, on the sbatch command line (these override the #SBATCH directives):
#   --job-name=<tag>_<dataset>   spm_wilt, e15_abalone, ... The mode tag comes
#                                first because squeue's default NAME column is
#                                8 characters wide: spm_wilt, e15_abal, spm_gest.
#   --output / --error           jobs_out/<dataset>_<mode>_%j.{out,err}
#   --export=ALL,TOPK_MODE=<mode>

set -uo pipefail

HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
JOBS_DIR="${REPO_ROOT}/hpc_entmax/jobs"
JOBS_OUT="${HPC_ROOT}/jobs_out"
DRY_RUN="${DRY_RUN:-0}"

# shellcheck source-path=SCRIPTDIR
# shellcheck source=common.sh
. "${REPO_ROOT}/hpc_entmax/common.sh"

usage() {
    local modes f names=()
    modes="$(IFS='|'; echo "${ENTMAX_MODES[*]}")"
    for f in "${JOBS_DIR}"/*_entmax.sh; do
        [ -e "${f}" ] && names+=("$(basename "${f}" _entmax.sh)")
    done
    echo "usage: $0 <${modes}> <dataset> [dataset ...]" >&2
    echo "       datasets with a job script: ${names[*]}" >&2
}

if [ "$#" -lt 2 ]; then
    usage
    exit 2
fi

MODE="$1"
shift
check_topk_mode "${MODE}" || exit 2
DATASETS=("$@")

case "${MODE}" in
    adaptive_sparsemax) TAG="spm" ;;
    adaptive_entmax15)  TAG="e15" ;;
esac

# SLURM does not create the --output directory; a missing one fails the job
# before it prints a single line.
mkdir -p "${JOBS_OUT}"
mkdir -p "${REPO_ROOT}/hpc_entmax/logs"

echo "mode      : ${MODE}  (job-name tag ${TAG})"
echo "jobs dir  : ${JOBS_DIR}"
echo "jobs_out  : ${JOBS_OUT}"
echo "datasets  : ${DATASETS[*]}"
echo

submitted=0
failed=0

for dataset in "${DATASETS[@]}"; do
    script="${JOBS_DIR}/${dataset}_entmax.sh"
    if [ ! -f "${script}" ]; then
        echo "  [SKIP] ${dataset}: no job script at ${script}" >&2
        failed=$((failed + 1))
        continue
    fi

    cmd=(sbatch
         --job-name="${TAG}_${dataset}"
         --output="${JOBS_OUT}/${dataset}_${MODE}_%j.out"
         --error="${JOBS_OUT}/${dataset}_${MODE}_%j.err"
         --export="ALL,TOPK_MODE=${MODE}"
         "${script}")

    if [ "${DRY_RUN}" = "1" ]; then
        printf '  [DRY] '
        printf '%q ' "${cmd[@]}"
        echo
        continue
    fi

    output="$("${cmd[@]}" 2>&1)"
    status=$?
    if [ ${status} -ne 0 ]; then
        echo "  [FAIL] ${dataset}: ${output}" >&2
        failed=$((failed + 1))
        continue
    fi

    # "Submitted batch job 12345"
    job_id="$(printf '%s' "${output}" | awk '{print $NF}')"
    printf '  %-10s %-20s job id %s   log %s\n' "${dataset}" "${TAG}_${dataset}" \
        "${job_id}" "${JOBS_OUT}/${dataset}_${MODE}_${job_id}.out"
    submitted=$((submitted + 1))
done

echo
echo "submitted: ${submitted}   failed: ${failed}"
[ ${failed} -gt 0 ] && exit 1
exit 0

#!/bin/bash
#
# Submit the many-feature budget study: one job per (mode, dataset).
#
#   ./hpc_wide/submit_all.sh adaptive_jenks king               # one job
#   ./hpc_wide/submit_all.sh all king                          # jenks + sparsemax + entmax15
#   ./hpc_wide/submit_all.sh adaptive_sparsemax miniboone higgs-small
#   ./hpc_wide/submit_all.sh all all                           # 3 modes x 5 datasets = 15 jobs
#   PROTOCOL=full ./hpc_wide/submit_all.sh all default         # 5 x 10 seeds instead of 1 x 1
#   DRY_RUN=1 ./hpc_wide/submit_all.sh all all                 # print only
#
# First argument: a mode, a comma-separated list of modes, or "all".
# Then: datasets, or "all" (every hpc_wide/jobs/<ds>_wide.sh). Both are
# required -- nothing is submitted by default.
#
# Per job, on the sbatch command line (these override the #SBATCH directives):
#   --job-name=<tag>_<dataset>   jnk_king, spm_miniboone, e15_fb-comments ...
#                                (squeue's 8-character NAME column: jnk_king,
#                                spm_mini, e15_fb-c)
#   --output / --error           jobs_out/wide_<dataset>_<mode>_<protocol>_%j.{out,err}
#   --export=ALL,TOPK_MODE=<mode>,PROTOCOL=<protocol>

set -uo pipefail

HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
JOBS_DIR="${REPO_ROOT}/hpc_wide/jobs"
JOBS_OUT="${HPC_ROOT}/jobs_out"
PROTOCOL="${PROTOCOL:-budget}"
DRY_RUN="${DRY_RUN:-0}"

# shellcheck source-path=SCRIPTDIR
# shellcheck source=common.sh
. "${REPO_ROOT}/hpc_wide/common.sh"

all_datasets() {
    local f
    for f in "${JOBS_DIR}"/*_wide.sh; do
        [ -e "${f}" ] && basename "${f}" _wide.sh
    done
}

usage() {
    echo "usage: $0 <mode[,mode...]|all> <dataset...|all>" >&2
    echo "       modes   : ${WIDE_MODES[*]}" >&2
    echo "       datasets: $(all_datasets | tr '\n' ' ')" >&2
    echo "       PROTOCOL=budget (default) | full" >&2
}

if [ "$#" -lt 2 ]; then
    usage
    exit 2
fi

check_protocol "${PROTOCOL}" || exit 2

if [ "$1" = "all" ]; then
    MODES=("${WIDE_MODES[@]}")
else
    IFS=',' read -r -a MODES <<< "$1"
fi
for mode in "${MODES[@]}"; do
    check_topk_mode "${mode}" || exit 2
done
shift

if [ "$1" = "all" ] && [ "$#" -eq 1 ]; then
    DATASETS=()
    while IFS= read -r ds; do DATASETS+=("${ds}"); done < <(all_datasets)
else
    DATASETS=("$@")
fi

# SLURM does not create the --output directory; a missing one fails the job
# before it prints a single line.
mkdir -p "${JOBS_OUT}"
mkdir -p "${REPO_ROOT}/hpc_wide/logs"

echo "modes     : ${MODES[*]}"
echo "datasets  : ${DATASETS[*]}"
echo "protocol  : ${PROTOCOL}"
echo "jobs_out  : ${JOBS_OUT}"
echo

submitted=0
failed=0

for mode in "${MODES[@]}"; do
    tag="$(mode_tag "${mode}")"
    for dataset in "${DATASETS[@]}"; do
        script="${JOBS_DIR}/${dataset}_wide.sh"
        if [ ! -f "${script}" ]; then
            echo "  [SKIP] ${dataset}: no job script at ${script}" >&2
            failed=$((failed + 1))
            continue
        fi
        log="${JOBS_OUT}/wide_${dataset}_${mode}_${PROTOCOL}"

        cmd=(sbatch
             --job-name="${tag}_${dataset}"
             --output="${log}_%j.out"
             --error="${log}_%j.err"
             --export="ALL,TOPK_MODE=${mode},PROTOCOL=${PROTOCOL}"
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
            echo "  [FAIL] ${mode} ${dataset}: ${output}" >&2
            failed=$((failed + 1))
            continue
        fi

        # "Submitted batch job 12345"
        job_id="$(printf '%s' "${output}" | awk '{print $NF}')"
        printf '  %-20s %-12s %-18s job id %s\n' "${mode}" "${dataset}" "${tag}_${dataset}" "${job_id}"
        submitted=$((submitted + 1))
    done
done

echo
echo "submitted: ${submitted}   failed: ${failed}"
[ ${failed} -gt 0 ] && exit 1
exit 0

#!/bin/bash
#
# Submit the density-curve study: one job per (dataset, condition).
#
#   ./hpc_density/submit_all.sh fixed_k2 wilt                 # one job
#   ./hpc_density/submit_all.sh fixed wilt abalone            # every fixed-k condition of each
#   ./hpc_density/submit_all.sh adaptive churn2               # the 3 adaptive rules
#   ./hpc_density/submit_all.sh all all                       # 37 jobs, cheapest dataset first
#   ./hpc_density/submit_all.sh fixed_k1,adaptive_jenks default gesture
#   MEM=40G ./hpc_density/submit_all.sh all wilt              # lower --mem (default 120G)
#   DRY_RUN=1 ./hpc_density/submit_all.sh all all             # print the sbatch commands only
#
# First argument: conditions (comma-separated), or all | fixed | adaptive,
# expanded per dataset from hpc_density/density_manifest.py -- the k grids
# differ, so e.g. fixed_k16 exists for gesture only. Then datasets, or "all"
# (cheapest first: wilt abalone churn2 gesture default). Both are required.
#
# Per job, on the sbatch command line (these override the #SBATCH directives):
#   --job-name=<tag>_<dataset>   tag k<K> / kD (fixed), Dj / Ds / De (adaptive):
#                                k2_wilt, k16_gest, kD_defau, Dj_abalo, De_churn
#                                in squeue's 8-character column, and no prefix
#                                shared with hpc/, hpc_entmax/ or hpc_wide/ jobs
#   --output / --error           jobs_out/density_<dataset>_<condition>_%j.{out,err}
#   --mem=${MEM}                 default 120G; lower it once sacct MaxRSS shows
#                                how much the jobs use, to fit more per node
#   --export=ALL,CONDITION=<condition>

set -uo pipefail

HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
PYTHON_BIN="${PYTHON_BIN:-${HPC_ROOT}/venvs/tabddpm310/bin/python}"
JOBS_DIR="${REPO_ROOT}/hpc_density/jobs"
JOBS_OUT="${HPC_ROOT}/jobs_out"
MEM="${MEM:-120G}"
DRY_RUN="${DRY_RUN:-0}"

# Cheapest first (GPU hours per condition, see README).
CHEAPEST_FIRST=(wilt abalone churn2 gesture default)

# shellcheck source-path=SCRIPTDIR
# shellcheck source=common.sh
. "${REPO_ROOT}/hpc_density/common.sh"

usage() {
    echo "usage: $0 <condition[,condition...]|all|fixed|adaptive> <dataset...|all>" >&2
    echo "       datasets: ${CHEAPEST_FIRST[*]}" >&2
    echo "       conditions per dataset: python3 hpc_density/density_manifest.py --repo_root . --summary" >&2
}

if [ "$#" -lt 2 ]; then
    usage
    exit 2
fi

SELECTOR="$1"
shift
if [ "$1" = "all" ] && [ "$#" -eq 1 ]; then
    DATASETS=("${CHEAPEST_FIRST[@]}")
else
    DATASETS=("$@")
fi

# Conditions of one dataset matching the selector, one per line.
select_conditions() {
    local dataset="$1" c
    case "${SELECTOR}" in
        all)      dataset_conditions "${dataset}" ;;
        fixed)    dataset_conditions "${dataset}" | grep '^fixed_' ;;
        adaptive) dataset_conditions "${dataset}" | grep '^adaptive_' ;;
        *)
            local wanted
            IFS=',' read -r -a wanted <<< "${SELECTOR}"
            for c in "${wanted[@]}"; do
                if dataset_conditions "${dataset}" | grep -qx "${c}"; then
                    echo "${c}"
                else
                    echo "  [SKIP] ${c} is not a condition of ${dataset} (valid: $(dataset_conditions "${dataset}" | tr '\n' ' '))" >&2
                fi
            done ;;
    esac
}

# SLURM does not create the --output directory; a missing one fails the job
# before it prints a single line.
mkdir -p "${JOBS_OUT}"
mkdir -p "${REPO_ROOT}/hpc_density/logs"

echo "conditions: ${SELECTOR}"
echo "datasets  : ${DATASETS[*]}"
echo "mem       : ${MEM}"
echo "jobs_out  : ${JOBS_OUT}"
echo

submitted=0
failed=0

for dataset in "${DATASETS[@]}"; do
    script="${JOBS_DIR}/${dataset}_density.sh"
    if [ ! -f "${script}" ]; then
        echo "  [FAIL] ${dataset}: no job script at ${script}" >&2
        failed=$((failed + 1))
        continue
    fi
    while IFS= read -r condition; do
        [ -z "${condition}" ] && continue
        tag="$(condition_tag "${condition}")"
        log="${JOBS_OUT}/density_${dataset}_${condition}"
        cmd=(sbatch
             --job-name="${tag}_${dataset}"
             --output="${log}_%j.out"
             --error="${log}_%j.err"
             --mem="${MEM}"
             --export="ALL,CONDITION=${condition}"
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
            echo "  [FAIL] ${dataset} ${condition}: ${output}" >&2
            failed=$((failed + 1))
            continue
        fi
        # "Submitted batch job 12345"
        job_id="$(printf '%s' "${output}" | awk '{print $NF}')"
        printf '  %-8s %-19s %-12s job id %s\n' "${dataset}" "${condition}" "${tag}_${dataset}" "${job_id}"
        submitted=$((submitted + 1))
    done < <(select_conditions "${dataset}")
done

echo
echo "submitted: ${submitted}   failed: ${failed}"
[ ${failed} -gt 0 ] && exit 1
exit 0

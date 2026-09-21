#!/bin/bash
#
# Submit one job per dataset for the adaptive-top-k re-run.
#
#   ./hpc/submit_all.sh                    # all datasets, cheapest first
#   ./hpc/submit_all.sh wilt buddy          # only these, in the order given
#   GRES=gpu:1 ./hpc/submit_all.sh          # pass --gres on the sbatch line
#   DRY_RUN=1 ./hpc/submit_all.sh           # print the sbatch commands only
#
# Datasets are submitted cheapest-first (fewest adaptive runs) so a short job
# comes back with a measured runtime before the long ones are committed. The
# gpu partition has only two nodes, so queue time can be long: this is one job
# per dataset by design, not many small ones.

set -uo pipefail

HPC_ROOT="${HPC_ROOT:-/work/pr019/atvoulo/Rompos}"
REPO_ROOT="${REPO_ROOT:-${HPC_ROOT}/thesis_project}"
JOBS_DIR="${REPO_ROOT}/hpc/jobs"

# Passed on the sbatch command line, where it overrides the #SBATCH --gres
# directive in each job script. Leave unset only if you have already filled the
# directive in; otherwise the job would run without a GPU for up to 24h.
GRES="${GRES:-}"
ALLOW_NO_GRES="${ALLOW_NO_GRES:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Cheapest first. Keep in sync with hpc/adaptive_manifest.py --list_datasets.
DEFAULT_DATASETS=(buddy gesture wilt insurance abalone churn2 diabetes)

if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

if [ -z "${GRES}" ] && [ "${ALLOW_NO_GRES}" != "1" ]; then
    cat >&2 <<'EOF'
ERROR: no GPU resource requested.

  "gpu:a100:1" is rejected on this cluster. Find the accepted string with:

      sinfo -p gpu -o "%N %G"

  then either submit with it:

      GRES=gpu:1 ./hpc/submit_all.sh

  or uncomment the #SBATCH --gres line in hpc/jobs/<dataset>_ablation.sh and
  re-run with ALLOW_NO_GRES=1 (which only suppresses this check).

  Submitting a 24h job without a GPU wastes the whole allocation, which is why
  this is an error rather than a warning.
EOF
    exit 1
fi

mkdir -p "${HPC_ROOT}/jobs_out"
mkdir -p "${REPO_ROOT}/hpc/logs"

echo "HPC_ROOT  : ${HPC_ROOT}"
echo "jobs dir  : ${JOBS_DIR}"
echo "jobs_out  : ${HPC_ROOT}/jobs_out"
echo "gres      : ${GRES:-<from #SBATCH directive>}"
echo "datasets  : ${DATASETS[*]}"
echo

submitted=0
failed=0

for dataset in "${DATASETS[@]}"; do
    script="${JOBS_DIR}/${dataset}_ablation.sh"
    if [ ! -f "${script}" ]; then
        echo "  [SKIP] ${dataset}: no job script at ${script}" >&2
        failed=$((failed + 1))
        continue
    fi

    cmd=(sbatch)
    [ -n "${GRES}" ] && cmd+=(--gres="${GRES}")
    cmd+=("${script}")

    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [DRY] ${cmd[*]}"
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
    printf '  %-12s job id %s\n' "${dataset}" "${job_id}"
    submitted=$((submitted + 1))
done

echo
echo "submitted: ${submitted}   failed: ${failed}"
[ ${failed} -gt 0 ] && exit 1
exit 0

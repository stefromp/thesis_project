#!/usr/bin/env bash
# Run one hpc_transfer job script end to end on a laptop, with every heavy step
# mocked (see mock_python.py). Exercises: the job script, common.sh (TRAIN_SETUP
# / CAT_INPUT validation, preflights, tree name, run_meta.json, flags), the
# runner CLI, training_setup() on the real config.toml, the result tree, and
# the run_meta.json / train_setup.json / denoiser_config.json records.
#
#   hpc_transfer/dryrun/dryrun_job.sh hpc_transfer/jobs_tddpm_setup/king_fixed_tddpm.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
JOB="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
TMP="$(mktemp -d)"
mkdir -p "${TMP}/bin"
printf '#!/usr/bin/env bash\nexec "$@"\n' > "${TMP}/bin/srun"
printf '#!/usr/bin/env bash\nexit 0\n' > "${TMP}/bin/module"
printf '#!/usr/bin/env bash\nexit 1\n' > "${TMP}/bin/nvidia-smi"
printf '#!/usr/bin/env bash\nexec python3 "%s" "$@"\n' "${HERE}/mock_python.py" > "${TMP}/bin/python_mock"
chmod +x "${TMP}"/bin/*
echo "dry run of $(basename "${JOB}") -- results tree: ${TMP}/transfer_runs"
PATH="${TMP}/bin:${PATH}" HPC_ROOT="${TMP}" REPO_ROOT="${REPO}" PYTHON_BIN="${TMP}/bin/python_mock" \
    TRANSFER_ROOT="${TMP}/transfer_runs" DEVICE=cpu bash "${JOB}"
status=$?
echo "exit status: ${status}"
echo "--- files written ---"
(cd "${TMP}/transfer_runs" && find . -maxdepth 4 -type d | sort | head -20)
run_dir="$(find "${TMP}/transfer_runs" -name run_meta.json -exec dirname {} \; | head -n1)"
for f in run_meta.json train_setup.json denoiser_config.json; do
    echo "--- ${f} ---"; cat "${run_dir}/${f}" 2>/dev/null || echo "(missing)"; echo
done
[ "${KEEP:-0}" = "1" ] || rm -rf "${TMP}"
exit "${status}"

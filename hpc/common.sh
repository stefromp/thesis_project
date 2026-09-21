# shellcheck shell=bash
#
# Shared per-combination runner for the adaptive-top-k jobs.
#
# Sourced by hpc/jobs/<dataset>_ablation.sh after that script has set:
#   HPC_ROOT  REPO_ROOT  PYTHON_BIN  DATASET  TOPK_WARMUP_STEPS  K_MIN
#   TOPK_LOG_EVERY  DEVICE  SEED
#
# Every command below is the literal port of what the Kaggle notebooks do per
# combination (import ablation_runner, patch nothing, call run_ablation with a
# single combo). The ONLY difference is --topk_mode adaptive_jenks (plus the
# flags that mode needs). Training budget, seed protocol and evaluator are left
# at the module defaults on purpose -- passing --steps/--n_gen_seeds/
# --n_clf_seeds would override the constants the baseline was produced with.

# ---------------------------------------------------------------------------
# One combination. Never aborts the caller: a failure is recorded and skipped.
# ---------------------------------------------------------------------------
run_one_combo() {
    local gnn_type="$1" n_layers="$2" d_model="$3" top_k="$4" n_heads="$5"
    local use_attention="$6" baseline_top_ks="$7"

    local suffix=""
    [ "$use_attention" = "0" ] && suffix="_noattn"
    local dir_name="gnn_${gnn_type}_layers${n_layers}_dim${d_model}_topk${top_k}_heads${n_heads}${suffix}"
    local out_dir="${REPO_ROOT}/exp/${DATASET}/ablation_adaptive_jenks/${dir_name}"
    local results_file="${out_dir}/results_full_averaged.json"

    echo
    echo "--------------------------------------------------------------------"
    echo "  ${DATASET} :: ${dir_name}"
    echo "--------------------------------------------------------------------"

    # Resume: a finished combination is identified by its final metrics file.
    # Training writes no intermediate checkpoints (ablation_runner passes
    # checkpoint_every=0), so a combination interrupted by the walltime is
    # restarted from step 0 -- only that one combination is lost, everything
    # already finished is skipped.
    if [ -f "${results_file}" ]; then
        echo "  [SKIP] results_full_averaged.json already present"
        return 0
    fi

    mkdir -p "${out_dir}"

    # ---- run_meta.json: resolved config + provenance -----------------------
    local gpu_name host git_commit now
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    [ -z "${gpu_name}" ] && gpu_name="unknown"
    host="$(hostname)"
    git_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # baseline_top_ks records every baseline combination this adaptive run
    # collapses (they differ only in top_k, which adaptive mode does not use),
    # so the comparison target is visible without reopening the manifest.
    cat > "${out_dir}/run_meta.json" <<EOF
{
  "dataset": "${DATASET}",
  "exp_name": "${dir_name}",
  "gnn_type": "${gnn_type}",
  "n_layers": ${n_layers},
  "d_model": ${d_model},
  "n_heads": ${n_heads},
  "use_attention": $([ "${use_attention}" = "1" ] && echo true || echo false),
  "top_k": ${top_k},
  "top_k_role": "representative: min() over the collapsed baseline top_k values; used only during warm-up",
  "baseline_top_ks": [${baseline_top_ks}],
  "topk_mode": "adaptive_jenks",
  "k_min": ${K_MIN},
  "topk_warmup_steps": ${TOPK_WARMUP_STEPS},
  "topk_log_every": ${TOPK_LOG_EVERY},
  "seed": ${SEED},
  "device": "${DEVICE}",
  "steps": "module default (ablation_runner.ABLATION_STEPS)",
  "n_gen_seeds": "module default (ablation_runner.N_GEN_SEEDS)",
  "n_clf_seeds": "module default (ablation_runner.N_CLF_SEEDS)",
  "evaluator": "module default (ablation_runner.EVALUATOR)",
  "hostname": "${host}",
  "gpu_name": "${gpu_name}",
  "git_commit": "${git_commit}",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "started_utc": "${now}"
}
EOF

    # ---- the run ------------------------------------------------------------
    # --data_root is the REPO root, not the data/ directory: ablation_runner
    # joins it with the per-dataset config's relative "data/<ds>/" path.
    # --use_attention 0 is mandatory -- the per-dataset scripts build a 5-tuple
    # single_combo, and run_ablation defaults the missing attention axis to
    # True, which would change both the architecture and the directory name.
    # Declared before the call: `local status` is itself a command and would
    # reset $? to 0 if it came between srun and the capture.
    local status

    srun "${PYTHON_BIN}" "${REPO_ROOT}/ablation/${DATASET}_ablation.py" \
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
        --repo_root "${REPO_ROOT}" \
        --data_root "${REPO_ROOT}" \
        --exp_root "${REPO_ROOT}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
    status=$?

    if [ ${status} -ne 0 ]; then
        echo "  [FAIL] ${dir_name} exited ${status}"
        mkdir -p "${REPO_ROOT}/hpc/logs"
        printf '%s\t%s\t%s\texit=%d\tjob=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DATASET}" "${dir_name}" \
            "${status}" "${SLURM_JOB_ID:-none}" \
            >> "${REPO_ROOT}/hpc/logs/${DATASET}_failures.txt"
        return 0   # keep going: one bad combination must not kill the job
    fi

    echo "  [OK] ${dir_name}"
    return 0
}

# ---------------------------------------------------------------------------
# Whole dataset: read the deduped run list, run each combination in order.
# ---------------------------------------------------------------------------
run_dataset() {
    echo "===================================================================="
    echo "  dataset           : ${DATASET}"
    echo "  repo              : ${REPO_ROOT}"
    echo "  python            : ${PYTHON_BIN}"
    echo "  topk_mode         : adaptive_jenks"
    echo "  k_min             : ${K_MIN}"
    echo "  topk_warmup_steps : ${TOPK_WARMUP_STEPS}"
    echo "  node              : $(hostname)"
    echo "  started           : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "===================================================================="

    local manifest
    manifest="$("${PYTHON_BIN}" "${REPO_ROOT}/hpc/adaptive_manifest.py" \
                    --repo_root "${REPO_ROOT}" --dataset "${DATASET}")"
    if [ -z "${manifest}" ]; then
        echo "ERROR: no adaptive runs for dataset ${DATASET}" >&2
        return 1
    fi

    local total
    total="$(printf '%s\n' "${manifest}" | wc -l | tr -d ' ')"
    echo "  ${total} adaptive run(s) for ${DATASET}"

    local i=0
    while IFS=$'\t' read -r gnn_type n_layers d_model top_k n_heads use_attention baseline_top_ks; do
        [ -z "${gnn_type}" ] && continue
        i=$((i + 1))
        echo
        echo "[${i}/${total}] ${DATASET}"
        run_one_combo "${gnn_type}" "${n_layers}" "${d_model}" "${top_k}" \
                      "${n_heads}" "${use_attention}" "${baseline_top_ks}"
    done <<< "${manifest}"

    echo
    echo "===================================================================="
    echo "  ${DATASET} finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ -f "${REPO_ROOT}/hpc/logs/${DATASET}_failures.txt" ]; then
        echo "  failures recorded in ${REPO_ROOT}/hpc/logs/${DATASET}_failures.txt:"
        cat "${REPO_ROOT}/hpc/logs/${DATASET}_failures.txt"
    else
        echo "  no failures recorded"
    fi
    echo "===================================================================="
}

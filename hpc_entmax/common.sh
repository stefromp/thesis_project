# shellcheck shell=bash
#
# Shared per-combination runner for the sparsemax / entmax-1.5 jobs.
#
# Sourced by hpc_entmax/jobs/<dataset>_entmax.sh after that script has set:
#   HPC_ROOT  REPO_ROOT  PYTHON_BIN  DATASET  TOPK_MODE  TOPK_WARMUP_STEPS
#   K_MIN  TOPK_LOG_EVERY  DEVICE  SEED
# (hpc_entmax/submit_all.sh sources it too, for check_topk_mode only.)
#
# Self-contained sibling of hpc/common.sh, which runs adaptive_jenks and is
# deliberately left untouched. Same combination list (hpc/adaptive_manifest.py,
# read-only), same per-combination command, same resume rule. Differences:
#   * TOPK_MODE is adaptive_sparsemax or adaptive_entmax15 -- nothing else;
#   * results go to exp/<ds>/ablation_<TOPK_MODE>/<combo>/;
#   * failures go to hpc_entmax/logs/<ds>_<TOPK_MODE>_failures.txt;
#   * a preflight checks the uploaded code before the first combination.
#
# Training budget, seed protocol and evaluator are left at the module defaults
# on purpose -- passing --steps/--n_gen_seeds/--n_clf_seeds would override the
# constants the baseline and the Jenks runs were produced with.

# The only modes this directory runs. adaptive_jenks and fixed belong to hpc/.
ENTMAX_MODES=(adaptive_sparsemax adaptive_entmax15)

check_topk_mode() {
    local mode="$1" m
    for m in "${ENTMAX_MODES[@]}"; do
        [ "${mode}" = "${m}" ] && return 0
    done
    echo "ERROR: TOPK_MODE='${mode}' is not run from hpc_entmax/." >&2
    echo "       Allowed here: ${ENTMAX_MODES[*]}" >&2
    case "${mode}" in
        adaptive_jenks|fixed)
            echo "       ${mode} runs belong to hpc/ (hpc/jobs/<dataset>_ablation.sh)." >&2 ;;
    esac
    return 1
}

entmax_alpha() {
    case "$1" in
        adaptive_sparsemax) echo "2.0" ;;
        adaptive_entmax15)  echo "1.5" ;;
    esac
}

# ---------------------------------------------------------------------------
# Preflight: prove the UPLOADED code can run TOPK_MODE before spending the
# allocation. Each check names the file to upload if it fails.
# ---------------------------------------------------------------------------
preflight() {
    "${PYTHON_BIN}" - "${REPO_ROOT}" "${TOPK_MODE}" <<'PYEOF'
import sys

repo, mode = sys.argv[1], sys.argv[2]
sys.path[:0] = [repo, f"{repo}/scripts", f"{repo}/ablation"]
problems = []

try:
    import entmax
    print(f"  entmax            : {getattr(entmax, '__version__', 'installed')}")
    have_entmax = True
except ImportError as exc:
    problems.append(f"entmax is not importable ({exc}): pip install entmax==1.3")
    have_entmax = False

import numpy as np
import sklearn
import torch
print(f"  torch / sklearn   : {torch.__version__} / {sklearn.__version__}")

try:
    from ablation_runner import make_parser
    choices = next(a.choices for a in make_parser()._actions if a.dest == "topk_mode")
    assert mode in choices, f"--topk_mode choices are {choices}"
except Exception as exc:
    problems.append(f"ablation/ablation_runner.py does not accept --topk_mode {mode} "
                    f"({type(exc).__name__}: {exc}): upload it")

if have_entmax:   # otherwise this fails on the import and would blame the wrong file
    try:
        from tab_ddpm.graph_builder import DynamicAdjacency
        da = DynamicAdjacency(5, 8, 3, topk_mode=mode).eval()
        with torch.no_grad():
            A = da(torch.randn(2, 5, 8), timesteps=torch.zeros(2, dtype=torch.long))
        assert A.shape == (2, 5, 5)
    except Exception as exc:
        problems.append(f"DynamicAdjacency cannot run topk_mode={mode} "
                        f"({type(exc).__name__}: {exc}): upload tab_ddpm/graph_builder.py "
                        f"and adaptive_adjacency.py")

try:
    # Without the sklearn < 1.2 fallback, every metric that one-hot encodes a
    # categorical column comes back NaN (the Jenks-run bug).
    from eval_tabddpm_protocol import _encode
    cat = np.array([["a"], ["b"]], dtype=object)
    _encode(None, cat, (None, cat))
except Exception as exc:
    problems.append(f"privacy-metric encoding fails on sklearn {sklearn.__version__} "
                    f"({type(exc).__name__}: {exc}): upload scripts/eval_tabddpm_protocol.py")

if problems:
    print("\nPREFLIGHT FAILED -- nothing was run:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)
print("  preflight         : OK")
PYEOF
}

# ---------------------------------------------------------------------------
# One combination. Never aborts the caller: a failure is recorded and skipped.
# ---------------------------------------------------------------------------
run_one_combo() {
    local gnn_type="$1" n_layers="$2" d_model="$3" top_k="$4" n_heads="$5"
    local use_attention="$6" baseline_top_ks="$7"

    local suffix=""
    [ "$use_attention" = "0" ] && suffix="_noattn"
    local dir_name="gnn_${gnn_type}_layers${n_layers}_dim${d_model}_topk${top_k}_heads${n_heads}${suffix}"
    local out_dir="${REPO_ROOT}/exp/${DATASET}/ablation_${TOPK_MODE}/${dir_name}"
    local results_file="${out_dir}/results_full_averaged.json"

    echo
    echo "--------------------------------------------------------------------"
    echo "  ${DATASET} :: ${TOPK_MODE} :: ${dir_name}"
    echo "--------------------------------------------------------------------"

    # Resume: a finished combination is identified by its final metrics file.
    # Training writes no intermediate checkpoints (checkpoint_every=0), so a
    # combination interrupted by the walltime restarts from step 0 -- only that
    # one combination is lost, everything already finished is skipped.
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
  "topk_mode": "${TOPK_MODE}",
  "entmax_alpha": $(entmax_alpha "${TOPK_MODE}"),
  "entmax_role": "mask only: the kept edges are the support of alpha-entmax over each row (computed under no_grad); edge weights are the same sigmoid gates as the fixed and Jenks paths",
  "k_min": ${K_MIN},
  "topk_warmup_steps": ${TOPK_WARMUP_STEPS},
  "topk_log_every": ${TOPK_LOG_EVERY},
  "seed": ${SEED},
  "device": "${DEVICE}",
  "steps": "module default (ablation_runner.ABLATION_STEPS)",
  "n_gen_seeds": "module default (ablation_runner.N_GEN_SEEDS)",
  "n_clf_seeds": "module default (ablation_runner.N_CLF_SEEDS)",
  "evaluator": "module default (ablation_runner.EVALUATOR)",
  "torch_version": "${TORCH_VERSION}",
  "sklearn_version": "${SKLEARN_VERSION}",
  "entmax_version": "${ENTMAX_VERSION}",
  "privacy_metrics_note": "This cluster environment uses scikit-learn ${SKLEARN_VERSION}; the Kaggle baseline (tabddpm_ablation/) used scikit-learn >= 1.2. scripts/eval_tabddpm_protocol.py _encode falls back from OneHotEncoder(sparse_output=False) to OneHotEncoder(sparse=False) on sklearn < 1.2. Both produce the same dense one-hot matrix, so dcr_tabddpm_mean, dcr_baseline and mia_auc are identical across the two environments (verified on wilt, abalone and gesture, sklearn 1.1.3 vs 1.7.2).",
  "hostname": "${host}",
  "gpu_name": "${gpu_name}",
  "git_commit": "${git_commit}",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "started_utc": "${now}"
}
EOF

    # ---- the run ------------------------------------------------------------
    # --input none: without it srun reads the job's stdin, which is the
    # here-string feeding run_dataset's loop -- it swallowed the remaining
    # combinations and only the first one ran.
    # --data_root is the REPO root, not data/: ablation_runner joins it with
    # the per-dataset config's relative "data/<ds>/" path.
    # --use_attention 0 is mandatory -- the per-dataset scripts build a 5-tuple
    # single_combo, and run_ablation defaults the missing attention axis to
    # True, which would change both the architecture and the directory name.
    # Declared before the call: `local status` is itself a command and would
    # reset $? to 0 if it came between srun and the capture.
    local status

    srun --input none "${PYTHON_BIN}" "${REPO_ROOT}/ablation/${DATASET}_ablation.py" \
        --gnn_type "${gnn_type}" \
        --n_layers "${n_layers}" \
        --d_model "${d_model}" \
        --top_k "${top_k}" \
        --n_heads "${n_heads}" \
        --use_attention "${use_attention}" \
        --topk_mode "${TOPK_MODE}" \
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
        mkdir -p "${REPO_ROOT}/hpc_entmax/logs"
        printf '%s\t%s\t%s\texit=%d\tjob=%s\tmode=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DATASET}" "${dir_name}" \
            "${status}" "${SLURM_JOB_ID:-none}" "${TOPK_MODE}" \
            >> "${FAILURES_FILE}"
        return 0   # keep going: one bad combination must not kill the job
    fi

    echo "  [OK] ${dir_name}"
    return 0
}

# ---------------------------------------------------------------------------
# Whole dataset: validate, preflight, then run the deduped list in order.
# ---------------------------------------------------------------------------
run_dataset() {
    FAILURES_FILE="${REPO_ROOT}/hpc_entmax/logs/${DATASET}_${TOPK_MODE}_failures.txt"

    echo "===================================================================="
    echo "  dataset           : ${DATASET}"
    echo "  topk_mode         : ${TOPK_MODE}"
    echo "  repo              : ${REPO_ROOT}"
    echo "  python            : ${PYTHON_BIN}"
    echo "  k_min             : ${K_MIN}"
    echo "  topk_warmup_steps : ${TOPK_WARMUP_STEPS}"
    echo "  results           : ${REPO_ROOT}/exp/${DATASET}/ablation_${TOPK_MODE}/"
    echo "  failures log      : ${FAILURES_FILE}"
    echo "  node              : $(hostname)"
    echo "  started           : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "===================================================================="

    check_topk_mode "${TOPK_MODE}" || return 1
    preflight || return 1

    TORCH_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
    SKLEARN_VERSION="$("${PYTHON_BIN}" -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null)"
    ENTMAX_VERSION="$("${PYTHON_BIN}" -c 'import entmax; print(getattr(entmax, "__version__", "installed"))' 2>/dev/null)"

    local manifest
    manifest="$("${PYTHON_BIN}" "${REPO_ROOT}/hpc/adaptive_manifest.py" \
                    --repo_root "${REPO_ROOT}" --dataset "${DATASET}")"
    if [ -z "${manifest}" ]; then
        echo "ERROR: no adaptive runs for dataset ${DATASET}" >&2
        return 1
    fi

    local total
    total="$(printf '%s\n' "${manifest}" | wc -l | tr -d ' ')"
    echo "  ${total} run(s) for ${DATASET} (${TOPK_MODE})"

    local i=0
    while IFS=$'\t' read -r gnn_type n_layers d_model top_k n_heads use_attention baseline_top_ks; do
        [ -z "${gnn_type}" ] && continue
        i=$((i + 1))
        echo
        echo "[${i}/${total}] ${DATASET} ${TOPK_MODE}"
        run_one_combo "${gnn_type}" "${n_layers}" "${d_model}" "${top_k}" \
                      "${n_heads}" "${use_attention}" "${baseline_top_ks}"
    done <<< "${manifest}"

    echo
    echo "===================================================================="
    echo "  ${DATASET} ${TOPK_MODE} finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ -f "${FAILURES_FILE}" ]; then
        echo "  failures recorded in ${FAILURES_FILE}:"
        cat "${FAILURES_FILE}"
    else
        echo "  no failures recorded"
    fi
    echo "===================================================================="
}

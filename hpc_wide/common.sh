# shellcheck shell=bash
#
# Shared per-combination runner for the many-feature budget study.
#
# Sourced by hpc_wide/jobs/<dataset>_wide.sh after that script has set:
#   HPC_ROOT  REPO_ROOT  PYTHON_BIN  DATASET  TOPK_MODE  PROTOCOL
#   TOPK_WARMUP_STEPS  K_MIN  TOPK_LOG_EVERY  DEVICE  SEED
# (hpc_wide/submit_all.sh sources it too, for the argument checks only.)
#
# Self-contained sibling of hpc/ and hpc_entmax/ (neither is touched). Runs all
# three adaptive rules -- adaptive_jenks, adaptive_sparsemax, adaptive_entmax15
# -- on datasets with 21-52 graph nodes, to see how many neighbours each rule
# keeps when there is room to choose. No ablation: one GATv2 configuration per
# dataset, from hpc_wide/wide_manifest.py (CONFIGS).
#
# PROTOCOL
#   budget (default)  full 20 000-step training, then 1 generation seed x 1
#                     classifier seed, and at most BUDGET_MAX_NUM_SAMPLES
#                     (50 000) synthetic rows: enough for the training AND
#                     sampling budgets in topk_stats.csv -- the per-t-bucket k
#                     curves stabilise long before 1.26 M rows -- and each job
#                     stays well inside the walltime. Its metrics are NOT
#                     comparable to the 5 x 10 protocol, which is why it has
#                     its own tree.
#   full              the module defaults (5 x 10) and the dataset's own
#                     num_samples, as in hpc/ and hpc_entmax/.
# The row cap is applied in memory by hpc_wide/run_combo.py; no file under
# ablation/ is changed for it.
# Results: $REPO_ROOT/wide_runs/<PROTOCOL>/exp/<ds>/ablation_<TOPK_MODE>/<combo>/
# -- never under exp/, so they cannot mix with the manifest runs.
# SMOKE_STEPS=<n> (plumbing test only) trains for n steps and writes to
# wide_runs/smoke/ instead, so a smoke run can never be resumed as a real one.

WIDE_MODES=(adaptive_jenks adaptive_sparsemax adaptive_entmax15)
WIDE_PROTOCOLS=(budget full)
BUDGET_MAX_NUM_SAMPLES=50000

check_topk_mode() {
    local mode="$1" m
    for m in "${WIDE_MODES[@]}"; do
        [ "${mode}" = "${m}" ] && return 0
    done
    echo "ERROR: TOPK_MODE='${mode}' is not run from hpc_wide/." >&2
    echo "       Allowed here: ${WIDE_MODES[*]}" >&2
    return 1
}

check_protocol() {
    local protocol="$1" p
    for p in "${WIDE_PROTOCOLS[@]}"; do
        [ "${protocol}" = "${p}" ] && return 0
    done
    echo "ERROR: PROTOCOL='${protocol}' -- allowed: ${WIDE_PROTOCOLS[*]}" >&2
    return 1
}

mode_tag() {
    case "$1" in
        adaptive_jenks)     echo "jnk" ;;
        adaptive_sparsemax) echo "spm" ;;
        adaptive_entmax15)  echo "e15" ;;
    esac
}

entmax_alpha() {   # JSON value for run_meta.json
    case "$1" in
        adaptive_sparsemax) echo "2.0" ;;
        adaptive_entmax15)  echo "1.5" ;;
        *)                  echo "null" ;;
    esac
}

# ---------------------------------------------------------------------------
# Preflight: prove the UPLOADED code and this dataset's config can run before
# spending the allocation. Each check names what to fix.
# ---------------------------------------------------------------------------
preflight() {
    "${PYTHON_BIN}" - "${REPO_ROOT}" "${TOPK_MODE}" "${DATASET}" <<'PYEOF'
import sys

repo, mode, dataset = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path[:0] = [repo, f"{repo}/scripts", f"{repo}/ablation", f"{repo}/hpc_wide"]
problems = []
needs_entmax = mode != "adaptive_jenks"

have_entmax = True
if needs_entmax:
    try:
        import entmax
        print(f"  entmax            : {getattr(entmax, '__version__', 'installed')}")
    except ImportError as exc:
        problems.append(f"entmax is not importable ({exc}): pip install entmax==1.3")
        have_entmax = False

import numpy as np
import sklearn
import torch
print(f"  torch / sklearn   : {torch.__version__} / {sklearn.__version__}")

import wide_manifest
problems += wide_manifest.check_config(repo, dataset)
try:
    N = wide_manifest.n_nodes(repo, dataset)
    print(f"  graph             : N = {N} nodes, k in 1..{N - 1}")
except Exception as exc:
    N = None
    problems.append(f"cannot compute the graph size of {dataset}: {type(exc).__name__}: {exc}")

try:
    from ablation_runner import make_parser
    choices = next(a.choices for a in make_parser()._actions if a.dest == "topk_mode")
    assert mode in choices, f"--topk_mode choices are {choices}"
except Exception as exc:
    problems.append(f"ablation/ablation_runner.py does not accept --topk_mode {mode} "
                    f"({type(exc).__name__}: {exc}): upload it")

if have_entmax and N:
    try:
        from tab_ddpm.graph_builder import DynamicAdjacency
        da = DynamicAdjacency(N, 8, 5, topk_mode=mode).eval()
        with torch.no_grad():
            A = da(torch.randn(2, N, 8), timesteps=torch.zeros(2, dtype=torch.long))
        assert A.shape == (2, N, N)
    except Exception as exc:
        problems.append(f"DynamicAdjacency cannot run topk_mode={mode} on {N} nodes "
                        f"({type(exc).__name__}: {exc}): upload tab_ddpm/graph_builder.py "
                        f"and adaptive_adjacency.py")

try:
    # Without the sklearn < 1.2 fallback, every metric that one-hot encodes a
    # categorical column comes back NaN.
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
    local gnn_type="$1" n_layers="$2" d_model="$3" top_k="$4" n_heads="$5" use_attention="$6"

    local suffix=""
    [ "$use_attention" = "0" ] && suffix="_noattn"
    local dir_name="gnn_${gnn_type}_layers${n_layers}_dim${d_model}_topk${top_k}_heads${n_heads}${suffix}"
    local out_dir="${WIDE_EXP_ROOT}/exp/${DATASET}/ablation_${TOPK_MODE}/${dir_name}"
    local results_file="${out_dir}/results_full_averaged.json"

    echo
    echo "--------------------------------------------------------------------"
    echo "  ${DATASET} :: ${TOPK_MODE} :: ${PROTOCOL_LABEL} :: ${dir_name}"
    echo "--------------------------------------------------------------------"

    # Resume: a finished combination is identified by its final metrics file.
    if [ -f "${results_file}" ]; then
        echo "  [SKIP] results_full_averaged.json already present"
        return 0
    fi

    mkdir -p "${out_dir}"

    local gpu_name host git_commit now
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    [ -z "${gpu_name}" ] && gpu_name="unknown"
    host="$(hostname)"
    git_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    cat > "${out_dir}/run_meta.json" <<EOF
{
  "study": "hpc_wide: adaptive budget on many-feature datasets",
  "dataset": "${DATASET}",
  "exp_name": "${dir_name}",
  "gnn_type": "${gnn_type}",
  "n_layers": ${n_layers},
  "d_model": ${d_model},
  "n_heads": ${n_heads},
  "use_attention": $([ "${use_attention}" = "1" ] && echo true || echo false),
  "top_k": ${top_k},
  "top_k_role": "used only during warm-up and in the directory name",
  "n_nodes": ${N_NODES},
  "k_max": $((N_NODES - 1)),
  "topk_mode": "${TOPK_MODE}",
  "entmax_alpha": $(entmax_alpha "${TOPK_MODE}"),
  "protocol": "${PROTOCOL_LABEL}",
  "protocol_note": "${PROTOCOL_NOTE}",
  "num_samples_config": ${NUM_SAMPLES_CONFIG},
  "num_samples_used": ${NUM_SAMPLES_USED},
  "num_samples_cap": ${NUM_SAMPLES_CAP_JSON},
  "sample_batch_size_used": ${SAMPLE_BATCH_USED},
  "steps": "${STEPS_LABEL}",
  "n_gen_seeds": "${GEN_SEEDS_LABEL}",
  "n_clf_seeds": "${CLF_SEEDS_LABEL}",
  "k_min": ${K_MIN},
  "topk_warmup_steps": ${TOPK_WARMUP_STEPS},
  "topk_log_every": ${TOPK_LOG_EVERY},
  "seed": ${SEED},
  "device": "${DEVICE}",
  "torch_version": "${TORCH_VERSION}",
  "sklearn_version": "${SKLEARN_VERSION}",
  "entmax_version": "${ENTMAX_VERSION}",
  "privacy_metrics_note": "This cluster environment uses scikit-learn ${SKLEARN_VERSION}; the Kaggle baseline (tabddpm_ablation/) used scikit-learn >= 1.2. scripts/eval_tabddpm_protocol.py _encode falls back from OneHotEncoder(sparse_output=False) to OneHotEncoder(sparse=False) on sklearn < 1.2. Both produce the same dense one-hot matrix, so dcr_tabddpm_mean, dcr_baseline and mia_auc are identical across the two environments.",
  "hostname": "${host}",
  "gpu_name": "${gpu_name}",
  "git_commit": "${git_commit}",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "started_utc": "${now}"
}
EOF

    # --input none: without it srun reads the job's stdin -- the here-string
    # feeding run_dataset's loop -- and only the first combination runs.
    # run_combo.py == ablation/<ds>_ablation.py, plus the in-memory row cap
    # (budget/smoke only; CAP_ARGS is empty for the full protocol).
    # ${arr[@]+"${arr[@]}"}: an empty array (PROTOCOL=full) under set -u is an
    # "unbound variable" error on bash < 4.4.
    # --exp_root is the protocol's own tree; --data_root stays the repo root.
    # --use_attention 0 is explicit: run_ablation defaults the axis to True.
    local status

    srun --input none "${PYTHON_BIN}" "${REPO_ROOT}/hpc_wide/run_combo.py" \
        --dataset "${DATASET}" \
        ${CAP_ARGS[@]+"${CAP_ARGS[@]}"} \
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
        ${PROTOCOL_ARGS[@]+"${PROTOCOL_ARGS[@]}"} \
        --repo_root "${REPO_ROOT}" \
        --data_root "${REPO_ROOT}" \
        --exp_root "${WIDE_EXP_ROOT}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
    status=$?

    if [ ${status} -ne 0 ]; then
        echo "  [FAIL] ${dir_name} exited ${status}"
        mkdir -p "${REPO_ROOT}/hpc_wide/logs"
        printf '%s\t%s\t%s\texit=%d\tjob=%s\tmode=%s\tprotocol=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DATASET}" "${dir_name}" \
            "${status}" "${SLURM_JOB_ID:-none}" "${TOPK_MODE}" "${PROTOCOL_LABEL}" \
            >> "${FAILURES_FILE}"
        return 0   # keep going: one bad combination must not kill the job
    fi

    echo "  [OK] ${dir_name}"
    return 0
}

# ---------------------------------------------------------------------------
# Whole dataset: validate, resolve the protocol, preflight, run the list.
# ---------------------------------------------------------------------------
run_dataset() {
    check_topk_mode "${TOPK_MODE}" || return 1
    # shellcheck disable=SC2153  # PROTOCOL is set by the sourcing job script
    check_protocol "${PROTOCOL}" || return 1

    ENTRY_MODULE="${DATASET//-/_}_ablation"
    if [ -n "${SMOKE_STEPS:-}" ]; then
        PROTOCOL_LABEL="smoke"
        WIDE_EXP_ROOT="${REPO_ROOT}/wide_runs/smoke"
        PROTOCOL_ARGS=(--steps "${SMOKE_STEPS}" --n_gen_seeds 1 --n_clf_seeds 1)
        NUM_SAMPLES_CAP="${BUDGET_MAX_NUM_SAMPLES}"
        STEPS_LABEL="${SMOKE_STEPS} (SMOKE_STEPS)"; GEN_SEEDS_LABEL="1"; CLF_SEEDS_LABEL="1"
        PROTOCOL_NOTE="plumbing test only: ${SMOKE_STEPS} training steps, 1 x 1 seeds; not a result"
    elif [ "${PROTOCOL}" = "budget" ]; then
        PROTOCOL_LABEL="budget"
        WIDE_EXP_ROOT="${REPO_ROOT}/wide_runs/budget"
        PROTOCOL_ARGS=(--n_gen_seeds 1 --n_clf_seeds 1)
        NUM_SAMPLES_CAP="${BUDGET_MAX_NUM_SAMPLES}"
        STEPS_LABEL="module default (ablation_runner.ABLATION_STEPS)"; GEN_SEEDS_LABEL="1"; CLF_SEEDS_LABEL="1"
        PROTOCOL_NOTE="budget protocol: full training, 1 generation seed x 1 classifier seed, synthetic rows capped at ${BUDGET_MAX_NUM_SAMPLES}. topk_stats.csv (train + sample) is complete; the metrics in results_full_averaged.json are single-seed, on the capped synthetic set, and not comparable to the 5 x 10 protocol"
    else
        PROTOCOL_LABEL="full"
        WIDE_EXP_ROOT="${REPO_ROOT}/wide_runs/full"
        PROTOCOL_ARGS=()
        NUM_SAMPLES_CAP=""
        STEPS_LABEL="module default (ablation_runner.ABLATION_STEPS)"
        GEN_SEEDS_LABEL="module default (ablation_runner.N_GEN_SEEDS)"
        CLF_SEEDS_LABEL="module default (ablation_runner.N_CLF_SEEDS)"
        PROTOCOL_NOTE="full protocol: module defaults, as in hpc/ and hpc_entmax/"
    fi
    FAILURES_FILE="${REPO_ROOT}/hpc_wide/logs/${DATASET}_${TOPK_MODE}_${PROTOCOL_LABEL}_failures.txt"
    CAP_ARGS=()
    NUM_SAMPLES_CAP_JSON="null"
    if [ -n "${NUM_SAMPLES_CAP}" ]; then
        CAP_ARGS=(--max_num_samples "${NUM_SAMPLES_CAP}")
        NUM_SAMPLES_CAP_JSON="${NUM_SAMPLES_CAP}"
    fi

    # The evaluators read tuned_models/... and write catboost_info/ relative to
    # the working directory, which for a SLURM job is wherever sbatch ran.
    cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; return 1; }

    echo "===================================================================="
    echo "  dataset           : ${DATASET}   (ablation/${ENTRY_MODULE}.py)"
    echo "  topk_mode         : ${TOPK_MODE}"
    echo "  protocol          : ${PROTOCOL_LABEL}"
    echo "  results           : ${WIDE_EXP_ROOT}/exp/${DATASET}/ablation_${TOPK_MODE}/"
    echo "  repo (cwd)        : ${REPO_ROOT}"
    echo "  python            : ${PYTHON_BIN}"
    echo "  k_min / warm-up   : ${K_MIN} / ${TOPK_WARMUP_STEPS}"
    echo "  failures log      : ${FAILURES_FILE}"
    echo "  node              : $(hostname)"
    echo "  started           : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "===================================================================="

    preflight || return 1

    N_NODES="$("${PYTHON_BIN}" "${REPO_ROOT}/hpc_wide/wide_manifest.py" \
                  --repo_root "${REPO_ROOT}" --n_nodes "${DATASET}")"
    read -r NUM_SAMPLES_CONFIG NUM_SAMPLES_USED SAMPLE_BATCH_USED <<< \
        "$("${PYTHON_BIN}" "${REPO_ROOT}/hpc_wide/wide_manifest.py" --repo_root "${REPO_ROOT}" \
              --sampling "${DATASET}" ${NUM_SAMPLES_CAP:+--cap "${NUM_SAMPLES_CAP}"})"
    echo "  synthetic rows    : ${NUM_SAMPLES_USED} (config ${NUM_SAMPLES_CONFIG}), sampling batch ${SAMPLE_BATCH_USED}"
    TORCH_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
    SKLEARN_VERSION="$("${PYTHON_BIN}" -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null)"
    ENTMAX_VERSION="$("${PYTHON_BIN}" -c 'import entmax; print(getattr(entmax, "__version__", "installed"))' 2>/dev/null)"
    [ -z "${ENTMAX_VERSION}" ] && ENTMAX_VERSION="not installed"

    local manifest
    manifest="$("${PYTHON_BIN}" "${REPO_ROOT}/hpc_wide/wide_manifest.py" \
                    --repo_root "${REPO_ROOT}" --dataset "${DATASET}")"
    if [ -z "${manifest}" ]; then
        echo "ERROR: no runs for dataset ${DATASET} in hpc_wide/wide_manifest.py" >&2
        return 1
    fi

    local total
    total="$(printf '%s\n' "${manifest}" | wc -l | tr -d ' ')"
    echo "  ${total} run(s) for ${DATASET} (${TOPK_MODE}, ${PROTOCOL_LABEL})"

    local i=0
    while IFS=$'\t' read -r gnn_type n_layers d_model top_k n_heads use_attention; do
        [ -z "${gnn_type}" ] && continue
        i=$((i + 1))
        echo
        echo "[${i}/${total}] ${DATASET} ${TOPK_MODE}"
        run_one_combo "${gnn_type}" "${n_layers}" "${d_model}" "${top_k}" \
                      "${n_heads}" "${use_attention}"
    done <<< "${manifest}"

    echo
    echo "===================================================================="
    echo "  ${DATASET} ${TOPK_MODE} ${PROTOCOL_LABEL} finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ -f "${FAILURES_FILE}" ]; then
        echo "  failures recorded in ${FAILURES_FILE}:"
        cat "${FAILURES_FILE}"
    else
        echo "  no failures recorded"
    fi
    echo "===================================================================="
}

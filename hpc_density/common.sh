# shellcheck shell=bash
# shellcheck disable=SC2153  # DATASET, CONDITION, ... are set by the sourcing job script
#
# Runner for the density-curve study: ONE condition of ONE dataset per job.
#
# Sourced by hpc_density/jobs/<dataset>_density.sh after that script has set:
#   HPC_ROOT  REPO_ROOT  PYTHON_BIN  DATASET  CONDITION
#   TOPK_WARMUP_STEPS  K_MIN  TOPK_LOG_EVERY  DEVICE  SEED
# (hpc_density/submit_all.sh sources it too, for the checks only.)
#
# Self-contained sibling of hpc/, hpc_entmax/ and hpc_wide/ (none is touched).
# CONDITION is fixed_k<K>, fixed_dense, adaptive_jenks, adaptive_sparsemax or
# adaptive_entmax15; the valid set per dataset, the architecture and the k grid
# come from hpc_density/density_manifest.py.
#
# Protocol: the module defaults, identical for every condition -- 20 000
# training steps, training seed SEED (0), generation seeds 0-4, classifier
# seeds 0-9 -- so conditions are paired by seed. Nothing here overrides them.
# Results: $REPO_ROOT/density_runs/exp/<ds>/ablation[_<mode>]/<combo>/, never
# under exp/ or wide_runs/. Each run also writes per_seed_scores.csv (one row
# per generation x classifier seed; needs the per-seed ablation_runner.py).

manifest() {
    "${PYTHON_BIN}" "${REPO_ROOT}/hpc_density/density_manifest.py" --repo_root "${REPO_ROOT}" "$@"
}

# Valid conditions of a dataset, one per line (fixed grid, dense, 3 adaptive).
dataset_conditions() {
    manifest --conditions "$1"
}

check_condition() {
    local dataset="$1" condition="$2" c
    if [ -z "${condition}" ]; then
        echo "ERROR: CONDITION is not set (e.g. CONDITION=fixed_k2 or adaptive_jenks)." >&2
        return 1
    fi
    while IFS= read -r c; do
        [ "${c}" = "${condition}" ] && return 0
    done < <(dataset_conditions "${dataset}")
    echo "ERROR: '${condition}' is not a condition of ${dataset}." >&2
    echo "       Valid: $(dataset_conditions "${dataset}" | tr '\n' ' ')" >&2
    return 1
}

# Short squeue tag: k<K> / kD for fixed, Dj / Ds / De for adaptive. None of
# these prefixes is used by hpc/, hpc_entmax/ or hpc_wide/ job names.
condition_tag() {
    case "$1" in
        fixed_dense)        echo "kD" ;;
        fixed_k*)           echo "k${1#fixed_k}" ;;
        adaptive_jenks)     echo "Dj" ;;
        adaptive_sparsemax) echo "Ds" ;;
        adaptive_entmax15)  echo "De" ;;
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
# Preflight: prove the UPLOADED code and this dataset's config can run the
# condition before spending the allocation. Each check names what to fix.
# ---------------------------------------------------------------------------
preflight() {
    "${PYTHON_BIN}" - "${REPO_ROOT}" "${DATASET}" "${TOPK_MODE}" "${TOP_K}" <<'PYEOF'
import sys

repo, dataset, mode, top_k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path[:0] = [repo, f"{repo}/scripts", f"{repo}/ablation", f"{repo}/hpc_density"]
problems = []

have_entmax = True
if mode in ("adaptive_sparsemax", "adaptive_entmax15"):
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

import density_manifest
problems += density_manifest.check_config(repo, dataset)
N = density_manifest.n_nodes(repo, dataset)

try:
    import ablation_runner
    choices = next(a.choices for a in ablation_runner.make_parser()._actions if a.dest == "topk_mode")
    assert mode in choices, f"--topk_mode choices are {choices}"
    # The paired analysis needs one record per (generation, classifier) seed.
    assert hasattr(ablation_runner, "_append_per_seed"), "no per-seed persistence"
    print(f"  seed protocol     : generation seeds 0-{ablation_runner.N_GEN_SEEDS - 1}, "
          f"classifier seeds 0-{ablation_runner.N_CLF_SEEDS - 1}, {ablation_runner.ABLATION_STEPS} steps")
except Exception as exc:
    problems.append(f"ablation/ablation_runner.py is not the density-study version "
                    f"({type(exc).__name__}: {exc}): upload it")

if have_entmax:
    try:
        from tab_ddpm.graph_builder import DynamicAdjacency
        da = DynamicAdjacency(N, 8, top_k, topk_mode=mode).eval()
        with torch.no_grad():
            A = da(torch.randn(2, N, 8), timesteps=torch.zeros(2, dtype=torch.long))
        assert A.shape == (2, N, N)
    except Exception as exc:
        problems.append(f"DynamicAdjacency cannot run topk_mode={mode} top_k={top_k} on {N} nodes "
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
# The job: resolve the condition, preflight, run it once.
# ---------------------------------------------------------------------------
run_condition() {
    check_condition "${DATASET}" "${CONDITION}" || return 1

    # The evaluators read tuned_models/... and write catboost_info/ relative to
    # the working directory, which for a SLURM job is wherever sbatch ran.
    cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; return 1; }

    local resolved
    resolved="$(manifest --resolve "${DATASET}" "${CONDITION}")" || return 1
    local gnn_type n_layers d_model n_heads use_attention
    IFS=$'\t' read -r TOPK_MODE TOP_K gnn_type n_layers d_model n_heads use_attention <<< "${resolved}"

    local n_nodes run_rel k_json k_frac_json
    n_nodes="$(manifest --n_nodes "${DATASET}")"
    run_rel="$(manifest --run_dir "${DATASET}" "${CONDITION}")"
    local exp_root="${REPO_ROOT}/density_runs"
    local out_dir="${exp_root}/exp/${DATASET}/${run_rel}"
    local results_file="${out_dir}/results_full_averaged.json"
    local failures_file="${REPO_ROOT}/hpc_density/logs/${DATASET}_${CONDITION}_failures.txt"

    case "${CONDITION}" in
        fixed_dense) k_json="$((n_nodes - 1))" ;;
        fixed_k*)    k_json="${TOP_K}" ;;
        *)           k_json="null" ;;
    esac
    if [ "${k_json}" = "null" ]; then
        k_frac_json="null"
    else
        k_frac_json="$("${PYTHON_BIN}" -c "print(${k_json} / (${n_nodes} - 1))")"
    fi

    echo "===================================================================="
    echo "  dataset           : ${DATASET}   N = ${n_nodes}"
    echo "  condition         : ${CONDITION}   (topk_mode ${TOPK_MODE}, top_k ${TOP_K})"
    echo "  architecture      : ${gnn_type} L${n_layers} d${d_model} h${n_heads} attention=${use_attention}"
    echo "  kept neighbours   : $([ "${k_json}" = "null" ] && echo "adaptive (measured from topk_stats.csv)" || echo "${k_json} of $((n_nodes - 1))")"
    echo "  results           : ${out_dir}"
    echo "  repo (cwd)        : ${REPO_ROOT}"
    echo "  python            : ${PYTHON_BIN}"
    echo "  failures log      : ${failures_file}"
    echo "  node              : $(hostname)"
    echo "  started           : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "===================================================================="

    if [ -f "${results_file}" ]; then
        echo "  [SKIP] results_full_averaged.json already present"
        return 0
    fi

    preflight || return 1

    mkdir -p "${out_dir}"
    # topk_stats.csv is appended to, so a restart after a timeout would mix the
    # killed attempt's rows with the new ones. Keep the old file, out of the way.
    # (per_seed_scores.csv is reset by ablation_runner itself.)
    if [ -f "${out_dir}/topk_stats.csv" ]; then
        local stale
        stale="${out_dir}/topk_stats.stale_$(date -u +%Y%m%dT%H%M%SZ).csv"
        mv "${out_dir}/topk_stats.csv" "${stale}"
        echo "  [RESTART] moved a previous attempt's topk_stats.csv to $(basename "${stale}")"
    fi

    local versions torch_v sklearn_v entmax_v n_gen n_clf steps gpu_name git_commit
    versions="$("${PYTHON_BIN}" - "${REPO_ROOT}" <<'PYEOF'
import sys
sys.path.insert(0, f"{sys.argv[1]}/ablation")
import sklearn, torch, ablation_runner as ar
try:
    import entmax; ev = getattr(entmax, "__version__", "installed")
except ImportError:
    ev = "not installed"
print(torch.__version__, sklearn.__version__, ev, ar.N_GEN_SEEDS, ar.N_CLF_SEEDS, ar.ABLATION_STEPS)
PYEOF
)"
    read -r torch_v sklearn_v entmax_v n_gen n_clf steps <<< "${versions}"
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    [ -z "${gpu_name}" ] && gpu_name="unknown"
    git_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"

    cat > "${out_dir}/run_meta.json" <<EOF
{
  "study": "hpc_density: density curve (fixed top-k grid vs adaptive rules)",
  "dataset": "${DATASET}",
  "condition": "${CONDITION}",
  "topk_mode": "${TOPK_MODE}",
  "top_k": ${TOP_K},
  "top_k_role": "$([ "${TOPK_MODE}" = "fixed" ] && echo "kept neighbours per node (0 = dense)" || echo "unused by the adaptive rule (warm-up 0); matches the earlier adaptive runs' directory names")",
  "k": ${k_json},
  "k_frac": ${k_frac_json},
  "n_nodes": ${n_nodes},
  "k_max": $((n_nodes - 1)),
  "entmax_alpha": $(entmax_alpha "${TOPK_MODE}"),
  "gnn_type": "${gnn_type}",
  "n_layers": ${n_layers},
  "d_model": ${d_model},
  "n_heads": ${n_heads},
  "use_attention": $([ "${use_attention}" = "1" ] && echo true || echo false),
  "train_seed": ${SEED},
  "gen_seeds": "0-$((n_gen - 1))",
  "clf_seeds": "0-$((n_clf - 1))",
  "steps": ${steps},
  "k_min": ${K_MIN},
  "topk_warmup_steps": ${TOPK_WARMUP_STEPS},
  "topk_log_every": ${TOPK_LOG_EVERY},
  "device": "${DEVICE}",
  "torch_version": "${torch_v}",
  "sklearn_version": "${sklearn_v}",
  "entmax_version": "${entmax_v}",
  "privacy_metrics_note": "This cluster environment uses scikit-learn ${sklearn_v}; the Kaggle baseline (tabddpm_ablation/) used scikit-learn >= 1.2. scripts/eval_tabddpm_protocol.py _encode falls back from OneHotEncoder(sparse_output=False) to OneHotEncoder(sparse=False) on sklearn < 1.2. Both produce the same dense one-hot matrix, so dcr_tabddpm_mean, dcr_baseline and mia_auc are identical across the two environments.",
  "hostname": "$(hostname)",
  "gpu_name": "${gpu_name}",
  "git_commit": "${git_commit}",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    # --input none: without it srun reads the job's stdin (it swallowed the
    # combination loop in hpc/; harmless here, kept for the same reason).
    # --use_attention 0 is explicit: run_ablation defaults the axis to True.
    # No --steps / --n_gen_seeds / --n_clf_seeds: every condition gets the
    # module defaults, which is what pairs them.
    local status
    srun --input none "${PYTHON_BIN}" "${REPO_ROOT}/ablation/${DATASET//-/_}_ablation.py" \
        --gnn_type "${gnn_type}" \
        --n_layers "${n_layers}" \
        --d_model "${d_model}" \
        --top_k "${TOP_K}" \
        --n_heads "${n_heads}" \
        --use_attention "${use_attention}" \
        --topk_mode "${TOPK_MODE}" \
        --k_min "${K_MIN}" \
        --topk_warmup_steps "${TOPK_WARMUP_STEPS}" \
        --topk_log_every "${TOPK_LOG_EVERY}" \
        --repo_root "${REPO_ROOT}" \
        --data_root "${REPO_ROOT}" \
        --exp_root "${exp_root}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
    status=$?

    if [ ${status} -ne 0 ]; then
        echo "  [FAIL] ${DATASET} ${CONDITION} exited ${status}"
        mkdir -p "${REPO_ROOT}/hpc_density/logs"
        printf '%s\t%s\t%s\texit=%d\tjob=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DATASET}" \
            "${CONDITION}" "${status}" "${SLURM_JOB_ID:-none}" >> "${failures_file}"
        return "${status}"
    fi

    echo
    echo "  [OK] ${DATASET} ${CONDITION} finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local expected=(results_full_averaged.json per_seed_scores.csv)
    # Only the adaptive rules log a budget; fixed top-k keeps k by definition.
    [ "${TOPK_MODE}" != "fixed" ] && expected+=(topk_stats.csv)
    for f in "${expected[@]}"; do
        if [ -f "${out_dir}/${f}" ]; then echo "    [present] ${f}"; else echo "    [MISSING] ${f}"; fi
    done
    return 0
}

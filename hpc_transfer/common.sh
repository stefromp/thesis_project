# shellcheck shell=bash
#
# Shared per-job runner for the transferred-configuration jobs.
#
# ONE dataset, ONE configuration, ONE job. There is no combination loop:
# every job here runs a single configuration, read from
# hpc_transfer/configs/<dataset>.sh.
#
# Two kinds of dataset are covered, and the config file says which:
#
#   the 7 NEW datasets      no ablation results exist, so the configuration is
#                           the one predicted in predictions.md. `fixed` is the
#                           baseline they never had; the adaptive rules are the
#                           comparison.
#   the 9 ABLATED datasets  the configuration is their own ablation winner
#                           (ablation/existing_results.load_best(), highest
#                           metrics.score.mean). No `fixed` run -- that baseline
#                           is already in tabddpm_ablation/.
#
# TOPK_MODE selects the rule, and the config file's RUN_MODES lists the ones
# that dataset is meant to be run in:
#
#   fixed               today's fixed top-k masking        -> exp/<ds>/ablation/
#   adaptive_jenks      plain 2-class Jenks split          -> exp/<ds>/ablation_adaptive_jenks/
#   adaptive_entmax15   support of entmax-1.5 (alpha 1.5)  -> exp/<ds>/ablation_adaptive_entmax15/
#
# adaptive_sparsemax (alpha 2.0) is accepted by the validation below, because
# ablation_runner accepts it, but this study does not run it -- no config lists
# it in RUN_MODES and no job script sets it.
#
# ablation_runner derives each sibling tree from --topk_mode itself, so the
# four rules of one configuration can never collide.
#
# entmax-1.5 needs the `entmax` package, which is NOT in
# requirements.txt -- hence the preflight below, ported from
# hpc_entmax/common.sh: it proves the UPLOADED code can run the mode before
# any allocation is spent.
#
# Sourced by hpc_transfer/jobs/*.sh after that script has set:
#   HPC_ROOT  REPO_ROOT  PYTHON_BIN  TRANSFER_ROOT  DATASET  TOPK_MODE
#   TOPK_WARMUP_STEPS  K_MIN  TOPK_LOG_EVERY  DEVICE  SEED
#
# ONE dataset, ONE configuration, ONE job -- the configuration predicted for
# that dataset in hpc_transfer/predictions.md, read from
# hpc_transfer/configs/<dataset>.sh. Unlike hpc/common.sh's run_dataset, there
# is no combination loop: each target has exactly one configuration, so the
# job and the run are the same thing.
#
# TOPK_MODE selects which of the two runs this job is:
#
#   fixed          the baseline. Results -> <TRANSFER_ROOT>/exp/<ds>/ablation/
#                  This is the counterpart of tabddpm_ablation/ for the seven
#                  datasets that were never ablated.
#   adaptive_jenks the adaptive-budget run, exactly what hpc/ does for the
#                  seven ablated datasets. Results ->
#                  <TRANSFER_ROOT>/exp/<ds>/ablation_adaptive_jenks/
#
# ablation_runner derives that sibling directory itself, from --topk_mode, so
# a fixed run and an adaptive run of the same combination can never collide.
#
# Two flags are load-bearing and carried over from hpc/common.sh unchanged:
#
#   * --use_attention 0 is MANDATORY. The per-dataset scripts build a
#     five-element single_combo and run_ablation defaults the missing attention
#     axis to True, which would change both the architecture and the directory
#     name away from the ablation's _noattn.
#   * --data_root is the REPO root, not the data/ directory: ablation_runner
#     joins it with the per-dataset config's relative "data/<dataset>/" path.
#
# --steps, --n_gen_seeds, --n_clf_seeds, --evaluator and --tabdiff_fidelity are
# NOT passed, so the run inherits the module constants the ablation was
# produced with: 20 000 steps, 5 generation x 10 classifier seeds, CatBoost,
# fidelity suite off. That is what makes these results directly comparable both
# to tabddpm_ablation/ and to the hpc/ adaptive re-run.

# alpha of the entmax family, as adaptive_adjacency.ENTMAX_ALPHA defines it.
# "null" for the two rules that are not entmax, so run_meta.json stays valid
# JSON either way.
entmax_alpha() {
    case "$1" in
        adaptive_sparsemax) echo "2.0" ;;
        adaptive_entmax15)  echo "1.5" ;;
        *)                  echo "null" ;;
    esac
}

# ---------------------------------------------------------------------------
# Preflight for the entmax rules: prove the UPLOADED code can run TOPK_MODE
# before spending the allocation. Ported from hpc_entmax/common.sh -- each
# check names the file to upload if it fails. Skipped for fixed and
# adaptive_jenks, which need no extra package.
# ---------------------------------------------------------------------------
entmax_preflight() {
    "${PYTHON_BIN}" - "${REPO_ROOT}" "$1" <<'PYEOF'
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

if have_entmax:   # else this fails on the import and would blame the wrong file
    try:
        from tab_ddpm.graph_builder import DynamicAdjacency
        da = DynamicAdjacency(5, 8, 3, topk_mode=mode).eval()
        with torch.no_grad():
            A = da(torch.randn(2, 5, 8), timesteps=torch.zeros(2, dtype=torch.long))
        assert A.shape == (2, 5, 5)
    except Exception as exc:
        problems.append(f"DynamicAdjacency cannot run topk_mode={mode} "
                        f"({type(exc).__name__}: {exc}): upload "
                        f"tab_ddpm/graph_builder.py and adaptive_adjacency.py")

if problems:
    print("\nPREFLIGHT FAILED -- nothing was run:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)
print("  preflight         : OK")
PYEOF
}

# ---------------------------------------------------------------------------
# Preflight for CAT_INPUT=prob (the categorical-input fix): prove the UPLOADED
# code has the option before spending the allocation. An older upload would
# reject --cat_input, or -- worse -- carry a graph_params key nobody reads.
# ---------------------------------------------------------------------------
cat_input_preflight() {
    "${PYTHON_BIN}" - "${REPO_ROOT}" <<'PYEOF'
import sys

repo = sys.argv[1]
sys.path[:0] = [repo, f"{repo}/scripts", f"{repo}/ablation"]
problems = []
try:
    from tab_ddpm.graph_denoiser import CAT_INPUTS, GraphAwareDenoiser
    m = GraphAwareDenoiser(d_in=3, num_classes=0, is_y_cond=False, d_num=1, cat_sizes=[2],
                           d_model=8, n_layers=1, n_heads=2, graph_mode="dynamic", top_k=1,
                           gnn_type="gcn", use_attention=False, cat_input="prob")
    assert "prob" in CAT_INPUTS and m.cat_input == "prob"
except Exception as exc:
    problems.append(f"tab_ddpm/graph_denoiser.py has no cat_input='prob' "
                    f"({type(exc).__name__}: {exc}): upload it")
try:
    # Read, not imported: importing pulls in the whole lib/ data stack.
    src = open(f"{repo}/scripts/utils_train.py").read()
    body = src[src.index("def get_model"):src.index("elif model_name == 'mlp'")]
    assert "cat_input=graph_params.get('cat_input'" in body, "get_model does not forward cat_input"
except Exception as exc:
    problems.append(f"scripts/utils_train.py does not pass cat_input ({exc}): upload it")
try:
    from ablation_runner import make_parser, results_subdir
    assert any(a.dest == "cat_input" for a in make_parser()._actions)
    assert results_subdir("fixed", "prob") == "ablation_catfix"
except Exception as exc:
    problems.append(f"ablation/ablation_runner.py lacks --cat_input "
                    f"({type(exc).__name__}: {exc}): upload it")
if problems:
    print("\nPREFLIGHT FAILED -- nothing was run:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)
print("  cat_input preflight: OK")
PYEOF
}

# ---------------------------------------------------------------------------
# Preflight for TRAIN_SETUP=tabddpm (budget-matched runs): prove the UPLOADED
# code has the option and that the authors' config.toml for this dataset is
# there and complete, before spending the allocation. Prints the values.
# ---------------------------------------------------------------------------
train_setup_preflight() {
    "${PYTHON_BIN}" - "${REPO_ROOT}" "$1" <<'PYEOF'
import sys

repo, ds = sys.argv[1], sys.argv[2]
sys.path[:0] = [repo, f"{repo}/scripts", f"{repo}/ablation"]
problems = []
try:
    import importlib
    from ablation_runner import make_parser, results_subdir, training_setup
    assert any(a.dest == "train_setup" for a in make_parser()._actions)
    assert results_subdir("fixed", "prob", "tabddpm") == "ablation_tddpm_setup_catfix"
    mod = importlib.import_module(ds.replace("-", "_") + "_ablation")
    values, sources = training_setup(ds, mod.DATASET_CFG, "tabddpm", repo)
    for k, v in values.items():
        print(f"  {k:17s} = {v!r:24}  <- {sources[k]}")
except Exception as exc:
    problems.append(f"ablation/ablation_runner.py / exp/{ds}/ddpm_cb_best/config.toml: "
                    f"{type(exc).__name__}: {exc} (upload both)")
try:
    src = open(f"{repo}/scripts/train.py").read()
    assert "divergence_check" in src and "class TrainingDiverged" in src
except Exception as exc:
    problems.append(f"scripts/train.py has no divergence check ({exc}): upload it")
if problems:
    print("\nPREFLIGHT FAILED -- nothing was run:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)
print("  train_setup preflight: OK")
PYEOF
}

run_transfer() {
    # Backstop for any caller that did not cd itself: the pipeline must run
    # from the repo root, whatever directory sbatch was called from.
    cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}" >&2; return 1; }
    local mode="${TOPK_MODE:-fixed}"
    local cat_input="${CAT_INPUT:-log}"
    local train_setup="${TRAIN_SETUP:-ablation}"
    case "${train_setup}" in
        ablation|tabddpm) ;;
        *) echo "ERROR: TRAIN_SETUP must be ablation or tabddpm -- got '${train_setup}'" >&2
           return 1 ;;
    esac
    case "${cat_input}" in
        log|prob) ;;
        *) echo "ERROR: CAT_INPUT must be log or prob -- got '${cat_input}'" >&2
           return 1 ;;
    esac
    case "${mode}" in
        fixed|adaptive_jenks|adaptive_sparsemax|adaptive_entmax15) ;;
        *) echo "ERROR: TOPK_MODE must be one of fixed, adaptive_jenks," >&2
           echo "       adaptive_sparsemax, adaptive_entmax15 -- got '${mode}'" >&2
           return 1 ;;
    esac

    local cfg="${REPO_ROOT}/hpc_transfer/configs/${DATASET}.sh"
    if [ ! -f "${cfg}" ]; then
        echo "ERROR: no config for ${DATASET} at ${cfg}" >&2
        return 1
    fi
    # shellcheck source=/dev/null
    . "${cfg}"

    local v
    for v in GNN_TYPE N_LAYERS D_MODEL TOP_K N_HEADS USE_ATTENTION; do
        if [ -z "${!v:-}" ]; then
            echo "ERROR: ${cfg} did not set ${v}" >&2
            return 1
        fi
    done

    local suffix=""
    [ "${USE_ATTENTION}" = "0" ] && suffix="_noattn"
    local dir_name="gnn_${GNN_TYPE}_layers${N_LAYERS}_dim${D_MODEL}_topk${TOP_K}_heads${N_HEADS}${suffix}"
    local subdir="ablation"
    [ "${mode}" != "fixed" ] && subdir="ablation_${mode}"
    # Mirrors ablation_runner.results_subdir: cat_input=prob runs live in
    # *_catfix sibling trees, never next to the log-input results.
    # ... and TRAIN_SETUP=tabddpm runs in *_tddpm_setup trees (before _catfix).
    [ "${train_setup}" = "tabddpm" ] && subdir="${subdir}_tddpm_setup"
    [ "${cat_input}" = "prob" ] && subdir="${subdir}_catfix"
    local out_dir="${TRANSFER_ROOT}/exp/${DATASET}/${subdir}/${dir_name}"
    local results_file="${out_dir}/results_full_averaged.json"

    echo "===================================================================="
    echo "  dataset      : ${DATASET}"
    echo "  configuration: ${dir_name}"
    echo "  topk_mode    : ${mode}"
    if [ "${mode}" != "fixed" ]; then
    echo "  k_min        : ${K_MIN}"
    echo "  warmup steps : ${TOPK_WARMUP_STEPS}"
    fi
    echo "  cat_input    : ${cat_input}"
    echo "  train_setup  : ${train_setup}"
    echo "  provenance   : ${PROVENANCE:-unrecorded}"
    echo "  repo         : ${REPO_ROOT}"
    echo "  results tree : ${TRANSFER_ROOT}"
    echo "  python       : ${PYTHON_BIN}"
    echo "  node         : $(hostname)"
    echo "  started      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "===================================================================="

    if [ "${train_setup}" = "tabddpm" ]; then
        echo
        echo "  -- preflight (train_setup=tabddpm) --"
        if ! train_setup_preflight "${DATASET}"; then
            echo "  ABORTED: preflight failed for train_setup=tabddpm" >&2
            return 1
        fi
    fi

    if [ "${cat_input}" = "prob" ]; then
        echo
        echo "  -- preflight (cat_input=prob) --"
        if ! cat_input_preflight; then
            echo "  ABORTED: preflight failed for cat_input=prob" >&2
            return 1
        fi
    fi

    # The entmax rules need a package that is not in requirements.txt. Check
    # before anything is created, so a missing dependency costs nothing.
    case "${mode}" in
        adaptive_sparsemax|adaptive_entmax15)
            echo
            echo "  -- preflight (${mode}) --"
            if ! entmax_preflight "${mode}"; then
                echo "  ABORTED: preflight failed for ${mode}" >&2
                return 1
            fi
            echo
            ;;
    esac

    # Resume: a finished run is identified by its final metrics file.
    # ablation_runner passes checkpoint_every=0, so training writes no
    # intermediate checkpoints. There is exactly ONE configuration per job
    # here, so unlike hpc/ -- where only the in-progress combination is lost --
    # a job killed by the walltime loses everything and restarts from step 0.
    # See the walltime table in hpc_transfer/README.md before submitting
    # fb-comments, miniboone or higgs-small.
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

    # top_k is a real masking parameter in fixed mode; in adaptive mode it is
    # unused except during warm-up (which defaults to 0) and in the directory
    # name. Recorded either way so the two runs are visibly the same combo.
    cat > "${out_dir}/run_meta.json" <<META
{
  "dataset": "${DATASET}",
  "exp_name": "${dir_name}",
  "gnn_type": "${GNN_TYPE}",
  "n_layers": ${N_LAYERS},
  "d_model": ${D_MODEL},
  "n_heads": ${N_HEADS},
  "top_k": ${TOP_K},
  "top_k_role": "$([ "${mode}" = "fixed" ] && echo "the masking budget" || echo "used only during warm-up and in the directory name")",
  "use_attention": $([ "${USE_ATTENTION}" = "1" ] && echo true || echo false),
  "topk_mode": "${mode}",
  "cat_input": "${cat_input}",
  "entmax_alpha": $(entmax_alpha "${mode}"),
  "k_min": ${K_MIN},
  "topk_warmup_steps": ${TOPK_WARMUP_STEPS},
  "topk_log_every": ${TOPK_LOG_EVERY},
  "study": "hpc_transfer",
  "provenance": "${PROVENANCE:-unrecorded}",
  "strategy": "${STRATEGY:-unrecorded}",
  "neighbours": "${NEIGHBOURS:-unrecorded}",
  "nearest_distance": "${NEAREST_DISTANCE:-unrecorded}",
  "confidence": "${CONFIDENCE:-unrecorded}",
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
META

    # The adaptive flags are passed ONLY in an adaptive mode -- the same four
    # flags for all three rules; only --topk_mode differs. Passing
    # --topk_mode fixed explicitly would be harmless, but leaving the whole
    # group off keeps the fixed run's command line identical to the one that
    # produced tabddpm_ablation/.
    # Expanded below as ${topk_flags[@]+"${topk_flags[@]}"}, not the plain
    # "${topk_flags[@]}": under `set -u`, bash before 4.4 treats an empty array
    # expanded that way as an unbound variable and aborts. That would have hit
    # every fixed-mode job, where the array is empty by design.
    local -a topk_flags=()
    if [ "${mode}" != "fixed" ]; then
        topk_flags=(--topk_mode "${mode}"
                    --k_min "${K_MIN}"
                    --topk_warmup_steps "${TOPK_WARMUP_STEPS}"
                    --topk_log_every "${TOPK_LOG_EVERY}")
    fi

    # Only for the fix: a log-input run keeps the exact command line (and so
    # still works with code uploaded before --cat_input existed).
    local -a cat_flags=()
    [ "${cat_input}" = "prob" ] && cat_flags=(--cat_input prob)
    # Same for the training setup: the default adds nothing to the command.
    local -a setup_flags=()
    [ "${train_setup}" = "tabddpm" ] && setup_flags=(--train_setup tabddpm)

    # Declared before the call: `local status` is itself a command and would
    # reset $? to 0 if it came between srun and the capture.
    local status

    srun "${PYTHON_BIN}" "${REPO_ROOT}/ablation/${DATASET//-/_}_ablation.py" \
        --gnn_type "${GNN_TYPE}" \
        --n_layers "${N_LAYERS}" \
        --d_model "${D_MODEL}" \
        --top_k "${TOP_K}" \
        --n_heads "${N_HEADS}" \
        --use_attention "${USE_ATTENTION}" \
        ${topk_flags[@]+"${topk_flags[@]}"} \
        ${cat_flags[@]+"${cat_flags[@]}"} \
        ${setup_flags[@]+"${setup_flags[@]}"} \
        --repo_root "${REPO_ROOT}" \
        --data_root "${REPO_ROOT}" \
        --exp_root "${TRANSFER_ROOT}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
    status=$?

    if [ ${status} -ne 0 ]; then
        echo "  [FAIL] ${dir_name} (${mode}) exited ${status}"
        # scripts/train.py writes this when divergence_check aborts training
        # (train_setup=tabddpm): non-finite loss or a loss explosion.
        if [ -f "${out_dir}/divergence.json" ]; then
            echo "  [DIVERGED] training aborted, no model saved -- ${out_dir}/divergence.json:"
            cat "${out_dir}/divergence.json"
        fi
        mkdir -p "${REPO_ROOT}/hpc_transfer/logs"
        printf '%s\t%s\t%s\t%s\texit=%d\tjob=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DATASET}" "${mode}" \
            "${dir_name}" "${status}" "${SLURM_JOB_ID:-none}" \
            >> "${REPO_ROOT}/hpc_transfer/logs/${DATASET}_${mode}_failures.txt"
        return "${status}"
    fi

    echo "  [OK] ${dir_name} (${mode})"
    echo "  finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    return 0
}



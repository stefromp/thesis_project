#!/usr/bin/env python3
"""
Generate hpc_transfer/jobs_tabdiff_gnn/ (job scripts, walltime_estimates.csv)
and hpc_transfer/submit_tabdiff_gnn.sh (every sbatch commented out).

    python3 tabdiff_gnn/make_jobs.py

Jobs follow hpc_transfer/jobs/ exactly (same header, the TODO(GPU) block with
the GRES line commented out, --mem=120G, the cluster template's OMP block,
paths overridable from the environment, cd to the repo root, one sourced
function): ONE job script per dataset, <ds>_tabdiff_gnn.sh, running
train -> 20 tables -> TabDiff evaluation (+ MLE-2) -> alpha/beta -> wNMIS /
C2ST(XGB). No DCR in this study. The training seed is SEED (default 0,
overridable: SEED=1 sbatch ...), as in hpc_transfer/jobs. Every job script asks
for --time=48:00:00 (JOB_WALLTIME_H); the estimate (stage estimates x 1.3) is
in its header comment and in walltime_estimates.csv. A dataset whose job would
exceed CHAIN_OVER_H (40 h) gets two scripts instead, <ds>_tabdiff_gnn_train.sh
and <ds>_tabdiff_gnn_sample_eval.sh (chained with --dependency=afterok);
training itself is never cut, except for news (resumable, below), whose train
script is submitted once per training part.

Arm:    gnn  AttnGNN V2, d_model 256 on every dataset (7 datasets). No TabDiff
             (transformer) jobs: TabDiff's numbers are the reported ones; the
             transformer code path stays tested, not run.
Seeds:  0 now; 1-2 via SEED=1 / SEED=2, double-commented "after confirmation".

Stage estimates (pessimistic; hours on one A100):
  train  epochs x (EPOCH_OVERHEAD_S + steps_per_epoch x s_step x (1 + eval/train))
         s_step = the measured A100 GNN step time of a same-type reference run
           at batch 4096 (gcn: adult L2 d128 N14; gatv2: default L3 d128 N23 h8
           -- the synced runs the TabDiff study measures,
           tabdiff_study/make_jobs.measure) x the CPU-measured cost ratio of this
           dataset's AttnGNN to that reference (bench_step_ratio.py ->
           jobs_tabdiff_gnn/step_ratio_cpu.csv, real shapes, d_model 256);
           eval/train from the same benchmark: Trainer.compute_loss runs one
           no-grad forward pass over the training set per epoch;
         the former pessimistic extrapolation (quadratic in nodes and d_model)
           is kept in walltime_estimates.csv as train_h_upper;
           EPOCH_OVERHEAD_S = 0.6: intercept of TabDiff's per-epoch time vs
           batches per epoch over GATD Table 16 (DataLoader workers re-spawned
           twice per epoch, EMA update, logging -- CPU-bound).
  sample 20 tables x n_train rows x 99 network evaluations (50 steps, 2nd-order
         correction on 49) x the reference run's measured sampling s/(row x step)
         x the CPU-measured forward-pass ratio.
news (config.RESUMABLE_DATASETS) trains with resumable.ResumableTrainer across
a chain of training jobs: each has the 48 h walltime and a training budget of
48 - TRAIN_RESERVE_H hours, after which the run state is saved and the job exits
cleanly; the next part resumes; a final job samples + evaluates. The number of
parts is ceil(train estimate x 1.3 / budget). With a measured A100 time
(timing probe) re-run:  python3 tabdiff_gnn/make_jobs.py --probe news=<s/epoch>.
For every other dataset a training estimate whose walltime exceeds 48 h cannot
run (TabDiff's Trainer, unchanged, does not resume): its job is written with the
48 h cap, flagged OVER_CAP, and blocked ('###') in the submission loop.
  eval_main / eval_quality   as tabdiff_study/make_jobs.py (local
         timings of TabDiff's evaluator; alpha/beta rows^2).
  eval_tabpc  local timing of tabpc's MI_l1 + C2ST per table (8 threads):
         magic 5.5 s, news 11.8 s, diabetes 133.2 s; others scaled from news by
         rows x columns; x 2 for the cluster.
"""
from __future__ import annotations

import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402

REPO = C.REPO
sys.path.insert(0, os.path.join(REPO, "ablation"))


def _load_study_make_jobs():
    """hpc_transfer/tabdiff_study/make_jobs.py (its measurements), loaded by path:
    this file has the same module name."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tabdiff_study_make_jobs",
                                                  os.path.join(C.STUDY_DIR, "make_jobs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MJ = _load_study_make_jobs()

OUT = os.path.join(REPO, "hpc_transfer", "jobs_tabdiff_gnn")
SUBMIT = os.path.join(REPO, "hpc_transfer", "submit_tabdiff_gnn.sh")
MARGIN, CAP_H, MIN_H = 1.3, 48, 2
CHAIN_OVER_H = 40                     # split train | rest above this walltime
JOB_WALLTIME_H = 48                   # every job script asks for 48:00:00 (user's choice); the
                                      # estimate is kept in the header comment and the CSV
TRAIN_RESERVE_H = 1.5                 # resumable parts: preflight, data, model build, final save, slack
PROBE_S_PER_EPOCH = {}                # dataset -> measured A100 s/epoch (--probe), overrides the estimate
EPOCHS = 8000
BATCH = 4096
NFE = 99
EMA_PASS = 1 / 3
EPOCH_OVERHEAD_S = 0.6
STAGE_OVERHEAD_H = 0.05               # process start, data loading, imports
REF_RUN = {"gcn": "adult", "gatv2": "default"}      # same-type measured reference (tabdiff_study SOURCE_RUNS)
REF_GNN = {"adult": {"n_layers": 2, "d_model": 128}, "default": {"n_layers": 3, "d_model": 128}}
# GATD (arXiv 2606.02607v1) Table 16, TabDiff on one NVIDIA T4: (train h, sample s per table)
GATD_TABDIFF_T4 = {"adult": (3.62, 34.6), "beijing": (4.21, 33.0), "default": (3.74, 29.5),
                   "diabetes": (13.51, 299.5), "magic": (2.24, 14.6), "news": (5.34, 51.4),
                   "shoppers": (2.19, 19.2)}
TABPC_S_MEASURED = {"magic": 5.5, "news": 11.8, "diabetes": 133.2}
TABPC_CLUSTER_FACTOR = 2.0
ARMS = {  # name -> (denoiser, datasets, training seeds)
    "gnn": ("attngnn", list(C.ORDER), list(C.TRAIN_SEEDS)),
    "gnn_jenks": ("attngnn", list(C.ORDER), list(C.TRAIN_SEEDS)),
    "gnn_entmax15": ("attngnn", list(C.ORDER), list(C.TRAIN_SEEDS)),
}
# neighbour-selection rule of each arm (config.TOPK_MODES); "gnn" = the fixed main arm
ARM_RULE = {"gnn": "fixed", "gnn_jenks": "adaptive_jenks", "gnn_entmax15": "adaptive_entmax15"}
RULE_RATIO_CSV = os.path.join(OUT, "step_ratio_rules_cpu.csv")
# Every sbatch line carries what the gpu partition requires (plain sbatch fails without --gres);
# command-line flags override the script's #SBATCH directives.
SBATCH_FLAGS = "--gres=gpu:a100:1 --mem=24G"
RULE_ORDER = ["beijing", "magic", "shoppers", "adult", "default", "diabetes", "news"]
POST = ["sample", "eval_main", "eval_quality", "eval_tabpc"]
# Ablation jobs (NOT the main protocol): (arm, dataset, config.ABLATIONS key), seed 0 only.
ABLATION_JOBS = [("gnn", "beijing", "topk_dense")]
# Measured cost of the ablation relative to the main run (CPU, beijing, 3 alternating rounds,
# median): GATv2 scores every node pair and then masks, so top_k = N-1 costs the same.
ABLATION_COST = {"topk_dense": 1.00}
RATIO_CSV = os.path.join(OUT, "step_ratio_cpu.csv")
# Submission blocks (seed 0), in order; seeds 1-2 follow, double-commented.
SUBMIT_BLOCKS = [
    ("AttnGNN d256 seed 0: magic, shoppers, adult, beijing",
     [("gnn", "magic"), ("gnn", "shoppers"), ("gnn", "adult"), ("gnn", "beijing")]),
    ("AttnGNN d256 seed 0: default, news (resumable 3-job chain), diabetes",
     [("gnn", "default"), ("gnn", "news"), ("gnn", "diabetes")]),
    ("AttnGNN d256 + adaptive_jenks, seed 0",
     [("gnn_jenks", ds) for ds in RULE_ORDER]),
    ("AttnGNN d256 + adaptive_entmax15, seed 0",
     [("gnn_entmax15", ds) for ds in RULE_ORDER]),
]


def rule_ratios() -> dict:
    """(dataset, topk_mode) -> {train_ratio_vs_fixed, eval_ratio_vs_fixed} (bench_step_ratio.py --rules)."""
    if not os.path.exists(RULE_RATIO_CSV):
        raise SystemExit(f"{RULE_RATIO_CSV} missing: run  python tabdiff_gnn/bench_step_ratio.py --rules")
    with open(RULE_RATIO_CSV) as fh:
        rows = list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))
    return {(r["dataset"], r["topk_mode"]): {k: float(r[k]) for k in ("train_ratio_vs_fixed",
                                                                       "eval_ratio_vs_fixed")} for r in rows}


def ratios() -> dict:
    with open(RATIO_CSV) as fh:
        rows = list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))
    return {r["dataset"]: {k: float(r[k]) for k in ("train_ratio", "eval_ratio", "eval_over_train")}
            for r in rows}


def _info(target):
    import json
    return json.load(open(C.data_files(target)["info"]))


def n_nodes(target) -> int:
    i = _info(target)
    return len(i["num_col_idx"]) + len(i["cat_col_idx"]) + len(i["target_col_idx"])


def steps_per_epoch(target) -> int:
    return math.ceil(_info(target)["train_num"] / BATCH)


def _scale(r: float) -> float:
    return r * r if r > 1 else r                       # pessimistic: quadratic up, linear down


_MEAS = {}


def reference(gnn_type: str) -> dict:
    src = REF_RUN[gnn_type]
    if src not in _MEAS:
        m = MJ.measure(src)
        m["nodes"] = MJ.nodes_source(src)
        m.update(REF_GNN[src])
        _MEAS[src] = m
    return _MEAS[src]


def gnn_scale(target: str, g: dict) -> tuple:
    ref = reference(g["gnn_type"])
    f = (g["n_layers"] / ref["n_layers"]) * _scale(n_nodes(target) / ref["nodes"]) \
        * _scale(g["d_model"] / ref["d_model"])
    return f, ref


def tabpc_s(ds: str, target: str) -> float:
    if ds in TABPC_S_MEASURED:
        return TABPC_S_MEASURED[ds] * TABPC_CLUSTER_FACTOR
    i = _info(target)
    ni = _info(C.target_name("news"))
    cells = i["train_num"] * len(i["column_names"])
    ncells = ni["train_num"] * len(ni["column_names"])
    return max(1.0, TABPC_S_MEASURED["news"] * cells / ncells) * TABPC_CLUSTER_FACTOR


def stage_hours(arm: str, ds: str) -> dict:
    tgt = C.target_name(ds)
    info = _info(tgt)
    n, spe = info["train_num"], steps_per_epoch(tgt)
    g, _ = C.gnn_config(ds)
    ref = reference(g["gnn_type"])
    rt = ratios()[ds]
    rule = ARM_RULE.get(arm, "fixed")
    rr = {"train_ratio_vs_fixed": 1.0, "eval_ratio_vs_fixed": 1.0} if rule == "fixed" else rule_ratios()[(ds, rule)]
    s_step = ref["train_s_per_step"] * rt["train_ratio"]
    s_epoch = PROBE_S_PER_EPOCH.get(ds, EPOCH_OVERHEAD_S + spe * s_step * (1 + rt["eval_over_train"]))
    if rule != "fixed":           # training step and the per-epoch EMA forward pass scale separately
        s_epoch = (PROBE_S_PER_EPOCH[ds] * rr["train_ratio_vs_fixed"] if ds in PROBE_S_PER_EPOCH else
                   EPOCH_OVERHEAD_S + spe * s_step * (rr["train_ratio_vs_fixed"]
                                                       + rt["eval_over_train"] * rr["eval_ratio_vs_fixed"]))
    st = {"train": EPOCHS * s_epoch / 3600 + STAGE_OVERHEAD_H,
          "sample": C.N_TABLES * n * NFE * ref["sample_s_per_row_step"] * rt["eval_ratio"]
          * rr["eval_ratio_vs_fixed"] / 3600 + STAGE_OVERHEAD_H}
    f_up, _ = gnn_scale(tgt, g)
    upper = EPOCHS * (EPOCH_OVERHEAD_S + spe * ref["train_s_per_step"] * f_up * (1 + EMA_PASS)) / 3600
    cols = len(info["column_names"])
    st["eval_main"] = C.N_TABLES * MJ.EVAL_MAIN_S_PER_CELL * n * cols / 3600 + STAGE_OVERHEAD_H
    st["eval_quality"] = C.N_TABLES * MJ.EVAL_QUALITY_S * (n / 11097) ** 2 / 3600 + STAGE_OVERHEAD_H
    st["eval_tabpc"] = C.N_TABLES * tabpc_s(ds, tgt) / 3600 + STAGE_OVERHEAD_H
    basis = (f"ref {REF_RUN[g['gnn_type']]} {ref['train_s_per_step']:.4f} s/step (A100) x CPU ratio "
             f"{rt['train_ratio']:.2f} = {s_step:.3f} s/step; {s_epoch:.1f} s/epoch")
    if ds in PROBE_S_PER_EPOCH:
        basis = f"A100 timing probe: {s_epoch:.1f} s/epoch"
    if rule != "fixed":
        basis += (f"; {rule}: train x{rr['train_ratio_vs_fixed']:.2f}, forward x{rr['eval_ratio_vs_fixed']:.2f} "
                  f"vs fixed (bench_step_ratio.py --rules)")
    return {"target": tgt, "n_train": n, "steps_per_epoch": spe, "nodes": n_nodes(tgt), "stages_h": st,
            "train_h_upper": upper, "basis": basis}


def _hours(h: float) -> int:
    return min(CAP_H, max(MIN_H, math.ceil(h * MARGIN)))


def over_cap(arm: str, ds: str) -> bool:
    """Training alone would need more than the 48 h cap and the dataset trains with
    TabDiff's own (non-resumable) Trainer."""
    return (ds not in C.RESUMABLE_DATASETS
            and math.ceil(stage_hours(arm, ds)["stages_h"]["train"] * MARGIN) > CAP_H)


def train_parts(arm: str, ds: str) -> int:
    """Resumable training: number of 48 h training jobs."""
    budget = CAP_H - TRAIN_RESERVE_H
    return max(1, math.ceil(stage_hours(arm, ds)["stages_h"]["train"] * MARGIN / budget))


def pack(arm: str, ds: str) -> list:
    """[(stages, hours)] per job: one job, or train | rest when one job would
    exceed CHAIN_OVER_H walltime (and always when training is over the cap)."""
    st = stage_hours(arm, ds)["stages_h"]
    tr, post = st["train"], sum(st[s] for s in POST)
    if ds in C.RESUMABLE_DATASETS and _hours(tr) > CHAIN_OVER_H:
        n = train_parts(arm, ds)
        return [(["train"], tr / n)] * n + [(POST, post)]
    if not over_cap(arm, ds) and _hours(tr + post) <= CHAIN_OVER_H:
        return [(["train"] + POST, tr + post)]
    return [(["train"], tr), (POST, post)]


HEADER = """#!/bin/bash -l

#SBATCH --job-name={job}
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/{job}.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/{job}.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time={time}
#SBATCH --mem=120G
#SBATCH --partition=gpu

# ===========================================================================
# TODO(GPU) -- FILL THIS IN BEFORE SUBMITTING. This is the ONLY line to edit.
# "gpu:a100:1" is rejected on this cluster ("invalid GRES format or GPU type").
# Find the accepted string with:      sinfo -p gpu -o "%N %G"
# then uncomment the line below with that value, e.g. #SBATCH --gres=gpu:1
#
##SBATCH --gres=gpu:1
#
# Alternatively leave it commented and pass it on the sbatch command line
# (GRES=... ) -- a command-line --gres overrides this directive, so you only
# have to get it right in one place.
# ===========================================================================

# Enable if submission is rejected for a missing account association
# (association: account=pr019022_gpu, partitions gpu and mig).
##SBATCH --account=pr019022_gpu

# NOTE: the #SBATCH --output/--error paths above are literal on purpose.
# SLURM parses these directives before any shell expansion, so they cannot use
# $HPC_ROOT. If you move HPC_ROOT, update those two lines too -- they are the
# only place in this file where the root is spelled out twice. For SEED=1 / 2
# pass --job-name/--output/--error on the sbatch command line (see
# hpc_transfer/submit_tabdiff_gnn.sh) so the seeds do not share one log.

# Verbatim from the cluster's job template -- kept byte-for-byte on purpose,
# so the x-prefix comparison and unquoted expansion are intentional here.
# shellcheck disable=SC2268,SC2086
if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

module purge
module load hdf5
module load python/3.10
module load cuda/11.8
module load cmake

export TORCH_CUDA_ARCH_LIST="8.0"

# --- not -e: the failure must reach the log, not abort the shell ------------
set -uo pipefail
"""

BODY = r"""
# ---------------------------------------------------------------------------
# Paths. Defined once, overridable from the environment.
#
# TRANSFER_ROOT is shared with hpc_transfer/, but Route A writes only to
# $TRANSFER_ROOT/exp_tabdiff_gnn/ (tabdiff_gnn/config.py enforces it), so it can
# never disturb the exp/ trees of the transfer study or the TabDiff study.
#
# PYTHON_BIN is the TabDiff environment (the eval venv: torch, sdmetrics,
# category_encoders, icecream, tomli -- what TabDiff's main.py imports); it
# trains, samples and runs TabDiff's evaluator. alpha/beta run in the synthcity
# venv and wNMIS / C2ST(XGB) in the tabpc venv (tabdiff_gnn/README.md, Install).
# ---------------------------------------------------------------------------
HPC_ROOT="${{HPC_ROOT:-/work/pr019/atvoulo/Rompos}}"
REPO_ROOT="${{REPO_ROOT:-${{HPC_ROOT}}/thesis_project}}"
PYTHON_BIN="${{PYTHON_BIN:-${{HPC_ROOT}}/venvs/tabdiff_eval310/bin/python}}"
EVAL_PYTHON="${{EVAL_PYTHON:-${{PYTHON_BIN}}}}"
SYNTHCITY_PYTHON="${{SYNTHCITY_PYTHON:-${{HPC_ROOT}}/venvs/tabdiff_synthcity310/bin/python}}"
TABPC_PYTHON="${{TABPC_PYTHON:-${{HPC_ROOT}}/venvs/tabpc310/bin/python}}"
TRANSFER_ROOT="${{TRANSFER_ROOT:-${{HPC_ROOT}}/transfer_runs}}"

# Run from the repo root whatever directory sbatch was called from: parts of
# the pipeline used to read repo files relative to the working directory.
cd "${{REPO_ROOT}}" || {{ echo "ERROR: cannot cd to ${{REPO_ROOT}}" >&2; exit 1; }}

# ---------------------------------------------------------------------------
# Denoiser: AttnGNN V2 inside TabDiff's framework -- only the denoiser differs
# from TabDiff (EDM + masked diffusion, learnable per-column schedules, TabDiff's
# Trainer, sampler and evaluator @ {commit_short}, file digests checked).
#   {arch}
#   GNN type / layers / top_k / heads from hpc_transfer/configs/{src}.sh{src_note}
#   d_model {d_model} on every dataset, dim_t {dim_t}, one shared noise embedding,
#   {rule_line}, dynamic graph, no global attention.
# Trainer: {trainer}
# Results: $TRANSFER_ROOT/exp_tabdiff_gnn/{target}/attngnn/{arm_dir}/seed<SEED>/
# ---------------------------------------------------------------------------
DENOISER="attngnn"

# ---------------------------------------------------------------------------
# Ablation: empty = the main Route-A run. A name from tabdiff_gnn/config.py
# ABLATIONS changes exactly one setting; the run gets its own path (e.g. ..._topk11_...)
# and run_meta.json records {{"ablation": ...}}. NOT part of the main protocol.
# ---------------------------------------------------------------------------
ABLATION="${{ABLATION:-{ablation}}}"

{rule_block}# ---------------------------------------------------------------------------
# Stages this job runs (each leaves <run>/stages/<stage>.done; a resubmitted
# job skips what is done):
#   {stage_line}
# Protocol: TabDiff's tabdiff_configs.toml unchanged (8000 epochs, batch 4096,
# lr 1e-3 + ReduceLROnPlateau, EMA 0.997, best EMA loss after epoch 4000,
# 50 sampling steps), 20 tables of n_train rows (generation seeds 0..19).
# No DCR in this study.
# Walltime {time}: {why}{chain_note}
# ---------------------------------------------------------------------------
STAGES="${{STAGES:-{stages}}}"

# ---------------------------------------------------------------------------
# Resumable training budget (news only; empty = TabDiff's Trainer, unchanged):
# after this many hours of training the run state is saved and the job exits
# cleanly (exit 0), so the next job of the chain -- this same script -- resumes.
# ---------------------------------------------------------------------------
TRAIN_BUDGET_H="${{TRAIN_BUDGET_H:-{budget}}}"

DEVICE="${{DEVICE:-cuda:0}}"
SEED="${{SEED:-0}}"

DATASET="{ds}"

export HPC_ROOT REPO_ROOT PYTHON_BIN EVAL_PYTHON SYNTHCITY_PYTHON TABPC_PYTHON \
       TRANSFER_ROOT DATASET DENOISER ABLATION STAGES TRAIN_BUDGET_H DEVICE SEED{rule_exports}

# shellcheck source-path=SCRIPTDIR
# shellcheck source=common_tabdiff_gnn.sh
. "${{REPO_ROOT}}/hpc_transfer/jobs_tabdiff_gnn/common_tabdiff_gnn.sh"

run_route_a
"""


def jobs_for(arm: str, ds: str, ablation: str | None = None) -> list:
    """Job scripts of one dataset: [{name, stages, hours, est, budget, parts, blocked}].
    parts = how many times the script is submitted in a row (news' resumable training)."""
    e = dict(stage_hours(arm, ds))
    chain = pack(arm, ds)
    blocked = over_cap(arm, ds)
    rule = ARM_RULE.get(arm, "fixed")
    base = (f"{ds}_tabdiff_gnn" + (f"_{C.RULE_TAG[rule]}" if rule != "fixed" else "")
            + (f"_ablation_{ablation}" if ablation else ""))
    # the fixed arm keeps its 48 h scripts (unchanged); the rule arms get estimate x 1.3 (cap 48 h)
    wall = (lambda est: JOB_WALLTIME_H) if rule == "fixed" else _hours
    if ablation:
        f = ABLATION_COST[ablation]
        e["stages_h"] = {k: (v * f if k in ("train", "sample") else v) for k, v in e["stages_h"].items()}
        e["basis"] += f"; ablation {ablation}: x{f:.2f} measured"
        chain = [(st, sum(e["stages_h"][x] for x in st)) for st, _ in chain]
        if len(chain) > 1 or ds in C.RESUMABLE_DATASETS:
            raise SystemExit(f"ablation jobs are single-job datasets only ({ds})")
    if len(chain) == 1:
        (stages, est), = chain
        return [dict(name=base, stages=stages, hours=wall(est), est=est, budget="", parts=1,
                     blocked=blocked, e=e, rule=rule)]
    trains = [c for c in chain if c[0] == ["train"]]
    post = chain[-1]
    resumable = ds in C.RESUMABLE_DATASETS
    tr_est = sum(c[1] for c in trains)
    return [dict(name=f"{base}_train", stages=["train"], hours=CAP_H if resumable else wall(tr_est),
                 est=tr_est, budget=f"{CAP_H - TRAIN_BUDGET_RESERVE:g}" if resumable else "",
                 parts=len(trains), blocked=blocked, e=e, rule=rule),
            dict(name=f"{base}_sample_eval", stages=post[0], hours=wall(post[1]), est=post[1], budget="",
                 parts=1, blocked=blocked, e=e, rule=rule)]


TRAIN_BUDGET_RESERVE = TRAIN_RESERVE_H


def write_jobs() -> dict:
    import glob
    os.makedirs(OUT, exist_ok=True)
    for old in glob.glob(os.path.join(OUT, "*.sh")):
        if os.path.basename(old) != "common_tabdiff_gnn.sh":
            os.remove(old)
    plan = {}
    todo = [(arm, ds, None) for arm, (_, dsets, _) in ARMS.items() for ds in dsets] + ABLATION_JOBS
    for arm, ds, ablation in todo:
            denoiser = ARMS[arm][0]
            js = jobs_for(arm, ds, ablation)
            g, src = C.gnn_config(ds, ablation=ablation)
            for i, j in enumerate(js):
                e = j["e"]
                stage_line = " -> ".join(f"{st} ({e['stages_h'][st]:.2f} h)" for st in j["stages"])
                chain_note = ""
                if len(js) > 1:
                    if j["stages"] == ["train"] and j["budget"]:
                        chain_note = (f"\n# CHAIN: submit this script {j['parts']} times in a row "
                                      f"(--dependency=afterok), then {js[1]['name']}.sh; each part trains "
                                      f"until the {j['budget']} h budget, saves and exits 0, the next resumes.")
                    elif j["stages"] == ["train"]:
                        chain_note = f"\n# CHAIN part 1/2: then {js[1]['name']}.sh with --dependency=afterok."
                    else:
                        chain_note = (f"\n# CHAIN last part: submit with --dependency=afterok on the last "
                                      f"{js[0]['name']}.sh job.")
                if j["blocked"] and j["stages"] == ["train"]:
                    chain_note += (f"\n# OVER_CAP -- DO NOT SUBMIT YET: training is estimated at "
                                   f"{e['stages_h']['train']:.1f} h (x{MARGIN} > {CAP_H} h) and TabDiff's Trainer "
                                   f"cannot resume; run the timing probe (tabdiff_gnn/README.md) first.")
                rule = j["rule"]
                if j["budget"]:
                    why = f"fixed at {JOB_WALLTIME_H} h; resumable training budget {j['budget']} h + reserve"
                elif rule == "fixed":
                    why = (f"fixed at {JOB_WALLTIME_H} h for every job; estimate {j['est']:.2f} h "
                           f"(x{MARGIN} = {math.ceil(j['est'] * MARGIN)} h)")
                else:
                    why = (f"estimate {j['est']:.2f} h x {MARGIN}, rounded up (min {MIN_H}, cap {CAP_H})")
                if rule == "fixed":
                    rule_line, rule_block, rule_exports = "fixed top-k rule", "", ""
                    sel = f"top-k {g['top_k']} (fixed)"
                else:
                    rd = C.RULE_DETAILS[rule]
                    rule_line = f"adaptive neighbour selection {rule} (source top_k {g['top_k']} unused)"
                    sel = f"rule {C.RULE_TAG[rule]} (k_i in {rd['k_range']})"
                    rule_block = (
                        "# ---------------------------------------------------------------------------\n"
                        f"# Neighbour-selection rule (TOPK_MODE, as in hpc_transfer/jobs): {rule} --\n"
                        f"# {rd['rule']}; edge weights = the same sigmoid gates as fixed.\n"
                        "# Settings of the thesis catfix runs, fixed in tabdiff_gnn/config.py (RULE_SETTINGS):\n"
                        "# TOPK_WARMUP_STEPS / K_MIN / TOPK_LOG_EVERY must keep these values (the run path\n"
                        "# encodes only the rule; the preflight refuses others).\n"
                        "# Selected-k statistics vs t in [0,1]: <run>/topk_k_stats.csv, topk_k_hist.csv.\n"
                        "# ---------------------------------------------------------------------------\n"
                        "RULE_ARM=1\n"
                        f'TOPK_MODE="${{TOPK_MODE:-{rule}}}"\n'
                        f'TOPK_WARMUP_STEPS="${{TOPK_WARMUP_STEPS:-{C.RULE_SETTINGS["topk_warmup_steps"]}}}"\n'
                        f'K_MIN="${{K_MIN:-{C.RULE_SETTINGS["k_min"]}}}"\n'
                        f'TOPK_LOG_EVERY="${{TOPK_LOG_EVERY:-{C.RULE_SETTINGS["topk_log_every"]}}}"\n\n')
                    rule_exports = " \\\n       RULE_ARM TOPK_MODE TOPK_WARMUP_STEPS K_MIN TOPK_LOG_EVERY"
                body = BODY.format(
                    commit_short=C.TABDIFF_COMMIT[:7],
                    arch=(f"{g['gnn_type']} L{g['n_layers']} d{g['d_model']} {sel} "
                          f"heads {g['n_heads']}"
                          + (f"\n#   ABLATION {ablation} (NOT the main protocol): {C.ABLATIONS[ablation]['change']}"
                             f"\n#   question: {C.ABLATIONS[ablation]['question']}" if ablation else "")),
                    src=C.SOURCES[ds],
                    src_note=(" -- our choice: diabetes has no TabDDPM source" if ds in C.SOURCE_NOTES else ""),
                    d_model=C.D_MODEL, dim_t=C.DIM_T,
                    trainer=("resumable copy of TabDiff's Trainer.run_loop (news only; verbatim + "
                             "save/restore hooks; evidence tests/test_tabdiff_gnn.py)"
                             if ds in C.RESUMABLE_DATASETS else "TabDiff's Trainer, unchanged"),
                    target=C.target_name(ds), arm_dir=C.arm_name(denoiser, ds, ablation=ablation, topk_mode=rule),
                    rule_line=rule_line, rule_block=rule_block, rule_exports=rule_exports,
                    ablation=ablation or "",
                    stage_line=stage_line, time=f"{j['hours']}:00:00",
                    why=f"{why}; {e['basis']}", chain_note=chain_note,
                    stages=",".join(j["stages"]), budget=j["budget"], ds=ds)
                path = os.path.join(OUT, f"{j['name']}.sh")
                with open(path, "w") as fh:
                    fh.write(HEADER.format(job=j["name"], time=f"{j['hours']}:00:00") + body)
                os.chmod(path, 0o755)
            plan[(arm, ds) if ablation is None else (arm, ds, ablation)] = js
    return plan


def sbatch_hours(j: dict) -> int:
    """--time on the sbatch line: the estimate x 1.3 (min 2, cap 48); 48 for a resumable training part."""
    return CAP_H if j["budget"] else _hours(j["est"])


def _submit_lines(js, seed: int, c: str) -> tuple:
    """Commented sbatch lines for one dataset and seed."""
    gpu_h = sum(j["est"] for j in js)
    if js[0]["blocked"]:
        c = "### "
    out, prev = [], None
    for j in js:
        for k in range(1, j["parts"] + 1):
            # each submission gets its own job name / log: SLURM truncates a reused --output
            name = (j["name"] + (f"_p{k}" if j["parts"] > 1 and k > 1 else "")
                    + ("" if seed == 0 else f"_s{seed}"))
            env = "" if seed == 0 else f"SEED={seed} "
            over = ("" if name == j["name"] else
                    f"--job-name={name} --output=${{OUT}}/{name}.out --error=${{OUT}}/{name}.err ")
            dep = "--dependency=afterok:${jid} " if prev else ""
            # realistic walltime from the estimate (resumable training parts: the 48 h cap)
            flags = f"{SBATCH_FLAGS} --time={sbatch_hours(j)}:00:00 "
            if len(js) > 1 or j["parts"] > 1:
                out.append(f'{c}jid=$({env}sbatch --parsable {flags}{dep}{over}'
                           f'hpc_transfer/jobs_tabdiff_gnn/{j["name"]}.sh)')
            else:
                out.append(f'{c}{env}sbatch {flags}{over}hpc_transfer/jobs_tabdiff_gnn/{j["name"]}.sh')
            prev = True
    return out, gpu_h


def write_submit(plan: dict) -> dict:
    rows = []
    for arm, ds in [x for _, items in SUBMIT_BLOCKS for x in items]:
        js = plan[(arm, ds)]
        e = js[0]["e"]
        g, _ = C.gnn_config(ds)
        cfg = f"{g['gnn_type']}_L{g['n_layers']}_d{g['d_model']}"
        rule = C.RULE_TAG.get(ARM_RULE[arm], "fixed")
        jobs_txt = " + ".join((f"{j['parts']}x " if j["parts"] > 1 else "") + f"{sbatch_hours(j)} h" for j in js)
        rows.append(f"#   {rule:8s} {ds:9s} {cfg:16s} {e['nodes']:3d}  {e['n_train']:7,d}  "
                    f"{sum(j['est'] for j in js):6.1f} h   {jobs_txt}"
                    + ("   OVER_CAP (blocked)" if js[0]["blocked"] else ""))
    lines = ["#!/usr/bin/env bash", "#",
             "# Submission lines for Route A (hpc_transfer/jobs_tabdiff_gnn/): AttnGNN as the",
             "# denoiser inside TabDiff's framework.",
             "#",
             "# NOTHING HERE IS EXECUTED. Every sbatch line is commented out on purpose.",
             "#",
             "# One job script per dataset, <ds>_tabdiff_gnn.sh: train -> 20 tables -> TabDiff's",
             "# evaluation (+ MLE-2) -> alpha/beta -> wNMIS / C2ST(XGB). One configuration: AttnGNN V2,",
             "# d_model 256, GNN type / layers / top_k / heads from hpc_transfer/configs/<source>.sh.",
             "# No TabDiff (transformer) runs -- TabDiff's numbers are the reported ones -- and no DCR.",
             "#",
             "# news trains with the resumable copy of TabDiff's Trainer.run_loop (its 8000 epochs do",
             "# not fit one 48 h job): news_tabdiff_gnn_train.sh is submitted once per training part,",
             "# each part saves its run state before its budget and exits 0, the next resumes; then",
             "# news_tabdiff_gnn_sample_eval.sh. If training needs one more part than planned, the",
             "# sample_eval job refuses to start (\"stage 'train' is not done\"): submit the train",
             "# script once more, then sample_eval again.",
             "#",
             "# The training seed is SEED (default 0). Seeds 1-2 are double-commented (##): submit",
             "# them only after confirmation; they get their own job name and logs.",
             "#",
             "# Rule arms (adaptive_jenks, adaptive_entmax15): same as the fixed arm except the",
             "# neighbour selection; scripts <ds>_tabdiff_gnn_<jenks|entmax15>.sh, results ..._<rule>_...",
             "#",
             "# Every sbatch line carries what the gpu partition requires: --gres=gpu:a100:1 --mem=24G",
             "# and a realistic --time (estimate x1.3, min 2 h, cap 48 h; 48 h for a resumable",
             "# training part); command-line flags override the scripts' #SBATCH directives.",
             "#",
             "# Estimates per seed (hours on one A100, before the x1.3 walltime margin):",
             "#",
             "#   rule     dataset   config            N  n_train   estimate   --time",
             *rows,
             "#",
             "# Re-size from a measured A100 time (timing probe, tabdiff_gnn/README.md):",
             "#     python3 tabdiff_gnn/make_jobs.py --probe news=<s/epoch>",
             "#",
             "# Before anything (seed 1-2 lines use it for their own logs):",
             "#     OUT=/work/pr019/atvoulo/Rompos/jobs_out",
             "# ---------------------------------------------------------------------------",
             "",
             "set -euo pipefail",
             'echo "This script does not submit anything. Open it and uncomment what you want." >&2',
             "exit 0", ""]
    totals = {}
    for i, (title, items) in enumerate(SUBMIT_BLOCKS, 1):
        lines.append(f"# === block {i}: {title} " + "=" * max(3, 60 - len(title)))
        tot = 0.0
        for arm, ds in items:
            ls, g = _submit_lines(plan[(arm, ds)], 0, "# ")
            lines += [f"# --- {ds} (~{g:.1f} GPU-h)"] + ls
            tot += g
        totals[f"block {i}: {title}"] = tot
        lines.append("")
    lines.append("# === ABLATION (NOT the main protocol; seed 0; compare with the main run) " + "=" * 8)
    for arm, ds, abl in ABLATION_JOBS:
        ls, g = _submit_lines(plan[(arm, ds, abl)], 0, "# ")
        lines += [f"# --- {ds} {abl}: {C.ABLATIONS[abl]['change'].split('(')[0].strip()} (~{g:.1f} GPU-h)"] + ls
        totals[f"ablation {ds} {abl}"] = g
    lines.append("")
    for seed in [x for x in C.TRAIN_SEEDS if x != 0]:
        lines.append(f"# === training seed {seed} -- ONLY AFTER CONFIRMATION " + "=" * 30)
        tot = 0.0
        for arm, ds in [x for _, items in SUBMIT_BLOCKS for x in items]:
            ls, g = _submit_lines(plan[(arm, ds)], seed, "## ")
            tag = C.RULE_TAG.get(ARM_RULE.get(arm, "fixed"), "fixed")
            lines += [f"# --- {ds} {tag} seed {seed} (~{g:.1f} GPU-h)"] + ls
            tot += g
        totals[f"seed {seed}"] = tot
        lines.append("")
    s0 = sum(v for k, v in totals.items() if k.startswith("block"))
    s0_fixed = sum(v for k, v in totals.items() if k.startswith("block") and "adaptive" not in k)
    ab = sum(v for k, v in totals.items() if k.startswith("ablation"))
    lines += [f"# GPU-h (estimate, before the x{MARGIN} margin): seed 0 {s0:.1f} (fixed {s0_fixed:.1f}, "
              f"rules {s0 - s0_fixed:.1f}); ablation {ab:.1f}; "
              f"seeds 1-2 {totals.get('seed 1', 0) + totals.get('seed 2', 0):.1f} more",
              "# Analysis afterwards: python3 analysis/tabdiff/compare.py --route_a", ""]
    with open(SUBMIT, "w") as fh:
        fh.write("\n".join(lines))
    os.chmod(SUBMIT, 0o755)
    return totals


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", nargs="*", default=[], metavar="DS=S_PER_EPOCH",
                    help="measured A100 seconds per epoch from the timing probe, e.g. news=31.5")
    for item in ap.parse_args(argv).probe:
        ds, v = item.split("=")
        C.parse_target(C.target_name(ds))
        PROBE_S_PER_EPOCH[ds] = float(v)
    rows = []
    for arm, (denoiser, dsets, seeds) in ARMS.items():
        for ds in dsets:
            e = stage_hours(arm, ds)
            for stage, h in e["stages_h"].items():
                rows.append({"arm": arm, "dataset": ds, "target": e["target"], "stage": stage,
                             "hours_per_seed": round(h, 3), "n_train": e["n_train"],
                             "steps_per_epoch": e["steps_per_epoch"], "nodes": e["nodes"],
                             "train_h_upper": round(e["train_h_upper"], 2) if stage == "train" else "",
                             "basis": e["basis"]})
    plan = write_jobs()
    for key, js in plan.items():
        arm, ds = key[0], key[1]
        for j in js:
            rows.append({"arm": arm, "dataset": ds, "target": f"JOB {j['name']}.sh"
                         + (f" x{j['parts']}" if j["parts"] > 1 else ""), "stage": "walltime",
                         "hours_per_seed": j["hours"], "basis": f"estimate {j['est']:.2f} h x {MARGIN}"
                         + (" -- OVER_CAP (capped)" if j["blocked"] else "")})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, "walltime_estimates.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    totals = write_submit(plan)
    for _, items in SUBMIT_BLOCKS:
        for arm, ds in items:
            js = plan[(arm, ds)]
            print(f"  {ds:9s} {sum(j['est'] for j in js):6.1f} GPU-h  -> "
                  + ", ".join(f"{j['name']}.sh {j['hours']} h" + (f" x{j['parts']}" if j["parts"] > 1 else "")
                              for j in js)
                  + ("   OVER_CAP (blocked)" if js[0]["blocked"] else ""))
    for k, v in totals.items():
        print(f"  {k:80s} {v:6.1f} GPU-h")
    print(f"  wrote {OUT}, {SUBMIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

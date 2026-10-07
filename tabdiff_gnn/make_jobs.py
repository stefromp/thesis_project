#!/usr/bin/env python3
"""
Generate hpc_transfer/jobs_tabdiff_gnn/ (job scripts, walltime_estimates.csv)
and hpc_transfer/submit_tabdiff_gnn.sh (every sbatch commented out).

    python3 tabdiff_gnn/make_jobs.py

Jobs: ONE resumable job per (arm, dataset, training seed): train -> 20 tables
-> TabDiff evaluation (+ MLE-2) -> alpha/beta -> wNMIS / C2ST(XGB). No DCR in
this study. gpu partition, --gres=gpu:a100:1, --mem=24G, cd to the repo root.
Walltime = stage estimates x 1.3, rounded up, min 2 h, cap 48 h. A job whose
walltime would exceed CHAIN_OVER_H (40 h) is cut into a chain at the
train | sample+evaluation boundary (<job>_p1.sh train, <job>_p2.sh rest, with
--dependency=afterok), keeping headroom under the 48 h cap; training itself is
never cut (TabDiff's Trainer cannot resume faithfully).

Arm:    gnn  AttnGNN V2, d_model 256 on every dataset (7 datasets). No TabDiff
             (transformer) jobs: TabDiff's numbers are the reported ones; the
             transformer code path stays tested, not run.
Seeds:  0 now; 1-2 generated, double-commented "after confirmation".

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
}
POST = ["sample", "eval_main", "eval_quality", "eval_tabpc"]
RATIO_CSV = os.path.join(OUT, "step_ratio_cpu.csv")
# Submission blocks (seed 0), in order; seeds 1-2 follow, double-commented.
SUBMIT_BLOCKS = [
    ("AttnGNN d256 seed 0: magic, shoppers, adult, beijing",
     [("gnn", "magic"), ("gnn", "shoppers"), ("gnn", "adult"), ("gnn", "beijing")]),
    ("AttnGNN d256 seed 0: default, news (resumable 3-job chain), diabetes",
     [("gnn", "default"), ("gnn", "news"), ("gnn", "diabetes")]),
]


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
    s_step = ref["train_s_per_step"] * rt["train_ratio"]
    s_epoch = PROBE_S_PER_EPOCH.get(ds, EPOCH_OVERHEAD_S + spe * s_step * (1 + rt["eval_over_train"]))
    st = {"train": EPOCHS * s_epoch / 3600 + STAGE_OVERHEAD_H,
          "sample": C.N_TABLES * n * NFE * ref["sample_s_per_row_step"] * rt["eval_ratio"] / 3600
          + STAGE_OVERHEAD_H}
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
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
##SBATCH --account=pr019022_gpu
"""

BODY = """
# ROUTE A (TabDiff framework, only the denoiser differs) -- {ds} ({task}), arm {arm}:
#   denoiser {denoiser}{arch}
#   training seed {seed}; {n_tables} tables per model (generation seeds 0..{last})
# Stages of this job (each leaves <run>/stages/<stage>.done; resubmitting skips them):
#   {unit_lines}{chain_note}
# Protocol  : TabDiff tabdiff_configs.toml unchanged (8000 epochs, batch 4096, lr 1e-3 +
#             ReduceLROnPlateau, EMA 0.997, best EMA loss after epoch 4000, 50 sampling steps);
#             routine eval every 2000 epochs off (no-effect deviation)
# Trainer   : {trainer}
# Evaluation: TabDiff's evaluator (digests checked) + MLE-2 + alpha/beta + tabpc wNMIS / C2ST(XGB);
#             no DCR in this study
# Data      : data/tabdiff_gnn/{target}/ (TabDiff's process_dataset.py @ {commit_short})
# Results   : $TRANSFER_ROOT/exp_tabdiff_gnn/{target}/{denoiser}/{arm_dir}/seed{seed}/
# Walltime {time}: {why}{overcap}

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

set -uo pipefail

HPC_ROOT="${{HPC_ROOT:-/work/pr019/atvoulo/Rompos}}"
REPO_ROOT="${{REPO_ROOT:-${{HPC_ROOT}}/thesis_project}}"
EVAL_PYTHON="${{EVAL_PYTHON:-${{HPC_ROOT}}/venvs/tabdiff_eval310/bin/python}}"
SYNTHCITY_PYTHON="${{SYNTHCITY_PYTHON:-${{HPC_ROOT}}/venvs/tabdiff_synthcity310/bin/python}}"
TABPC_PYTHON="${{TABPC_PYTHON:-${{HPC_ROOT}}/venvs/tabpc310/bin/python}}"
TRAIN_PYTHON="${{TRAIN_PYTHON:-${{EVAL_PYTHON}}}}"
TRANSFER_ROOT="${{TRANSFER_ROOT:-${{HPC_ROOT}}/transfer_runs}}"
DEVICE="${{DEVICE:-cuda:0}}"

cd "${{REPO_ROOT}}" || {{ echo "ERROR: cannot cd to ${{REPO_ROOT}}" >&2; exit 1; }}

DS="{ds}"
DENOISER="{denoiser}"
SEED="{seed}"
STAGE_GROUPS="{units}"
TRAIN_BUDGET_H="{budget}"

export HPC_ROOT REPO_ROOT EVAL_PYTHON SYNTHCITY_PYTHON TABPC_PYTHON TRAIN_PYTHON TRANSFER_ROOT DEVICE
export DS DENOISER SEED STAGE_GROUPS TRAIN_BUDGET_H

# shellcheck source=common_tabdiff_gnn.sh
. "${{REPO_ROOT}}/hpc_transfer/jobs_tabdiff_gnn/common_tabdiff_gnn.sh"

run_route_a
"""


def job_base(arm: str, ds: str, seed: int) -> str:
    return f"{arm}_{ds}_s{seed}"


def write_jobs() -> dict:
    import glob
    os.makedirs(OUT, exist_ok=True)
    for old in glob.glob(os.path.join(OUT, "*.sh")):
        if os.path.basename(old) != "common_tabdiff_gnn.sh":
            os.remove(old)
    plan = {}
    for arm, (denoiser, dsets, seeds) in ARMS.items():
        for ds in dsets:
            e = stage_hours(arm, ds)
            chain = pack(arm, ds)
            blocked = over_cap(arm, ds)
            for seed in seeds:
                base = job_base(arm, ds, seed)
                names = [base] if len(chain) == 1 else [f"{base}_p{i + 1}" for i in range(len(chain))]
                for i, (name, (stages, est)) in enumerate(zip(names, chain)):
                    resumable_train = ds in C.RESUMABLE_DATASETS and stages == ["train"]
                    hours = CAP_H if resumable_train else _hours(est)
                    budget = f"{CAP_H - TRAIN_RESERVE_H:g}" if resumable_train else ""
                    unit_lines = " -> ".join(f"{s} ({e['stages_h'][s]:.2f} h)" for s in stages)
                    chain_note = ""
                    if len(chain) > 1:
                        what = ("resumable training: pauses + saves its run state before "
                                f"{CAP_H - TRAIN_RESERVE_H:g} h, the next part resumes" if resumable_train
                                else "train | sample+evaluation")
                        chain_note = (f"\n# CHAIN part {i + 1}/{len(chain)} ({what}): submit as "
                                      + (f"sbatch {name}.sh" if i == 0 else
                                         f"sbatch --dependency=afterok:<{names[i - 1]} id> {name}.sh"))
                    g, _ = C.gnn_config(ds)
                    arch = (f" V2: {g['gnn_type']} L{g['n_layers']} d{g['d_model']} top-k {g['top_k']} "
                            f"(fixed) heads {g['n_heads']}, dynamic graph, no global attention, "
                            f"dim_t {C.DIM_T}, shared noise embedding; GNN source {C.SOURCES[ds]}"
                            + (" (our choice: diabetes has no TabDDPM source)" if ds in C.SOURCE_NOTES else ""))
                    overcap = ""
                    if blocked and "train" in stages:
                        overcap = (f"\n# OVER_CAP -- DO NOT SUBMIT YET: training is estimated at "
                                   f"{e['stages_h']['train']:.1f} h (x{MARGIN} = "
                                   f"{e['stages_h']['train'] * MARGIN:.0f} h > {CAP_H} h) and TabDiff's Trainer "
                                   f"cannot resume; the walltime is only the cap. Run the timing probe "
                                   f"(README) first.")
                    body = BODY.format(
                        ds=ds, task=C.TASK[ds], arm=arm, denoiser=denoiser, arch=arch, seed=seed,
                        n_tables=C.N_TABLES, last=C.N_TABLES - 1, unit_lines=unit_lines,
                        chain_note=chain_note, target=C.target_name(ds),
                        commit_short=C.TABDIFF_COMMIT[:7],
                        arm_dir=C.arm_name(denoiser, ds), time=f"{hours}:00:00", overcap=overcap,
                        why=(f"stage estimates {est:.2f} h x {MARGIN}, rounded up (min {MIN_H}, cap {CAP_H}); "
                             + e["basis"]),
                        units=",".join(stages), budget=budget,
                        trainer=("resumable copy of TabDiff's Trainer.run_loop (news only; "
                                 "evidence tests/test_tabdiff_gnn.py)" if ds in C.RESUMABLE_DATASETS
                                 else "TabDiff's Trainer, unchanged"))
                    path = os.path.join(OUT, f"{name}.sh")
                    with open(path, "w") as fh:
                        fh.write(HEADER.format(job=f"tdg_{name}", time=f"{hours}:00:00") + body)
                    os.chmod(path, 0o755)
                    plan.setdefault((arm, ds, seed), []).append((name, hours, est, blocked))
    return plan


def _submit_lines(plan, arm, ds, seed, c):
    jobs = plan[(arm, ds, seed)]
    gpu_h = sum(j[2] for j in jobs)
    if jobs[0][3]:
        c = "### "                                      # OVER_CAP: blocked until the probe
    if len(jobs) == 1:
        (name, h, _, _), = jobs
        return [f"# {arm} {ds}: {h} h walltime, ~{gpu_h:.1f} GPU-h", f"{c}sbatch {name}.sh"], gpu_h
    out = [f"# {arm} {ds}: CHAIN of {len(jobs)} ({' + '.join(f'{j[1]} h' for j in jobs)}), ~{gpu_h:.1f} GPU-h"
           + ("  [OVER_CAP: training estimate > 48 h, BLOCKED until the timing probe]" if jobs[0][3] else "")]
    prev = None
    for name, _, _, _ in jobs:
        v = f"j_{arm}_{ds}_s{seed}_" + name.split("_p")[-1]
        out.append(f"{c}{v}=$(sbatch --parsable " + (f"--dependency=afterok:${{{prev}}} " if prev else "")
                   + f"{name}.sh)")
        prev = v
    return out, gpu_h


def write_submit(plan: dict) -> dict:
    lines = ["#!/usr/bin/env bash", "#",
             "# Submission loop for Route A (AttnGNN d_model 256 inside TabDiff). NOTHING HERE IS EXECUTED:",
             "# every sbatch line is commented out on purpose; uncomment one block at a time.",
             "# No TabDiff (transformer) runs: TabDiff's numbers are the reported ones. No DCR.",
             "# '###' = OVER_CAP (training estimate > 48 h, TabDiff's Trainer cannot resume): blocked",
             "# until the timing probe (README) shows it fits. Regenerate: python3 tabdiff_gnn/make_jobs.py.",
             "",
             'cd "$(dirname "$0")/jobs_tabdiff_gnn" || exit 1', ""]
    totals = {}
    for i, (title, items) in enumerate(SUBMIT_BLOCKS, 1):
        lines.append(f"# ==== block {i}: {title}")
        tot = 0.0
        for arm, ds in items:
            ls, g = _submit_lines(plan, arm, ds, 0, "# ")
            lines += ls
            tot += g
        totals[f"block {i}: {title}"] = tot
        lines += [f"# block {i}: ~{tot:.1f} GPU-h", ""]
    for seed in [s for s in C.TRAIN_SEEDS if s != 0]:
        lines.append(f"# ==== training seed {seed} (ONLY AFTER CONFIRMATION)")
        tot = 0.0
        for arm, ds in [x for _, items in SUBMIT_BLOCKS for x in items]:
            ls, g = _submit_lines(plan, arm, ds, seed, "## ")
            lines += ls
            tot += g
        totals[f"seed {seed}"] = tot
        lines += [f"# seed {seed}: ~{tot:.1f} GPU-h", ""]
    s0 = sum(v for k, v in totals.items() if not k.startswith("seed"))
    lines += [f"# GPU-h (estimate, before the x{MARGIN} margin): seed 0 {s0:.1f}; seeds 1-2 "
              f"{totals.get('seed 1', 0) + totals.get('seed 2', 0):.1f} more", "",
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
    for (arm, ds, seed), jobs in plan.items():
        if seed == 0:
            for name, h, est, blocked in jobs:
                rows.append({"arm": arm, "dataset": ds, "target": f"JOB {name}.sh", "stage": "walltime",
                             "hours_per_seed": h, "basis": f"estimate {est:.2f} h x {MARGIN}"
                             + (" -- OVER_CAP (capped)" if blocked else "")})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, "walltime_estimates.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    totals = write_submit(plan)
    for _, items in SUBMIT_BLOCKS:
        for arm, ds in items:
            jobs = plan[(arm, ds, 0)]
            print(f"  {arm:7s} {ds:9s} {sum(j[2] for j in jobs):6.1f} GPU-h  -> "
                  + ", ".join(f"{n}.sh {h} h" for n, h, _, _ in jobs)
                  + ("   OVER_CAP (blocked)" if jobs[0][3] else ""))
    for k, v in totals.items():
        print(f"  {k:80s} {v:6.1f} GPU-h")
    print(f"  wrote {OUT}, {SUBMIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Write hpc_transfer/jobs_tddpm_setup/: budget-matched fixed-rule jobs.

    python3 hpc_transfer/make_tddpm_setup_jobs.py

One job per dataset (train + 5 gen seeds + aggregate, through common.sh /
run_transfer) with TRAIN_SETUP=tabddpm: lr, steps, batch size, weight decay,
num_timesteps, scheduler and sampling batch from the authors'
exp/<ds>/ddpm_cb_best/config.toml (ablation_runner.training_setup). Everything
else is the run the current analysis uses for that dataset: same configuration
(hpc_transfer/configs/<ds>.sh), cat_input=prob on categorical datasets (as the
catfix runs), training seed 0, 5 x 10 seeds.

Walltime: measured from the synced run used by the current analysis
(transfer_runs/exp, rsync -a keeps mtimes): training = loss.csv mtime -
run_meta.started_utc; one generation seed = median gap between consecutive
gen_seed_<k> directories (sampling of k+1 + CatBoost of k). Estimate =
train x (new steps / 20000) + 5 x seed x (new num_timesteps / 1000 for the
sampling share, NOT applied: the sample/eval split is not observable, so the
estimate stays conservative), x 1.5 margin, rounded up to whole hours, at least
2 h, capped at 48 h. A dataset whose estimate exceeds 48 h gets the split chain
(submit_split.sh, train-only job then one job per seed) instead.

Also writes walltime_estimates.csv next to the jobs.
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "jobs_tddpm_setup")
RESULTS = os.path.join(REPO, "transfer_runs", "exp")
sys.path[:0] = [os.path.join(REPO, "ablation")]

# dataset -> (tree of the fixed run the current analysis uses, CAT_INPUT, group)
PLAN = {
    "cardio":      ("ablation_catfix",      "prob", "main"),
    "house":       ("ablation",             "log",  "main"),
    "king":        ("ablation_catfix",      "prob", "main"),
    "gesture":     ("ablation_fixed_seed0", "log",  "main"),
    "california":  ("ablation_fixed_seed0", "log",  "main"),
    "higgs-small": ("ablation",             "log",  "expensive"),
    "miniboone":   ("ablation",             "log",  "expensive"),
}
OLD_STEPS = 20_000
MARGIN = 1.5
CAP_H = 48

HEADER = """#!/bin/bash -l

#SBATCH --job-name={short}
#SBATCH --output=/work/pr019/atvoulo/Rompos/jobs_out/{name}.out
#SBATCH --error=/work/pr019/atvoulo/Rompos/jobs_out/{name}.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --time={time}
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
##SBATCH --account=pr019022_gpu

# BUDGET-MATCHED RUN -- {ds}, topk_mode fixed, TRAIN_SETUP=tabddpm.
#
# Same configuration as the run the current analysis uses for {ds}
# (hpc_transfer/configs/{ds}.sh, cat_input={cat}, training seed 0, 5 gen x 10
# clf seeds, EMA, same synthetic size, CatBoost) EXCEPT the training setup,
# which is read from the TabDDPM authors' exp/{ds}/ddpm_cb_best/config.toml:
{values}
# (ablation_runner.training_setup; the values and their file:line also land in
# run_meta.json / train_setup.json.) Training aborts with divergence.json if
# the loss becomes non-finite or explodes; nothing else about training changes.
#
# Results: $TRANSFER_ROOT/exp/{ds}/{tree}/<combo>/
# -- a *_tddpm_setup sibling tree; no existing result is touched.
#
# Walltime {time}: {why}
{chain_note}
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
PYTHON_BIN="${{PYTHON_BIN:-${{HPC_ROOT}}/venvs/tabddpm310/bin/python}}"
TRANSFER_ROOT="${{TRANSFER_ROOT:-${{HPC_ROOT}}/transfer_runs}}"

cd "${{REPO_ROOT}}" || {{ echo "ERROR: cannot cd to ${{REPO_ROOT}}" >&2; exit 1; }}

TOPK_MODE="${{TOPK_MODE:-fixed}}"
CAT_INPUT="${{CAT_INPUT:-{cat}}}"
TRAIN_SETUP="${{TRAIN_SETUP:-tabddpm}}"
TOPK_WARMUP_STEPS="${{TOPK_WARMUP_STEPS:-0}}"
K_MIN="${{K_MIN:-1}}"
TOPK_LOG_EVERY="${{TOPK_LOG_EVERY:-100}}"
DEVICE="${{DEVICE:-cuda:0}}"
SEED="${{SEED:-0}}"

DATASET="{ds}"

export HPC_ROOT REPO_ROOT PYTHON_BIN TRANSFER_ROOT DATASET TOPK_MODE CAT_INPUT \\
       TRAIN_SETUP TOPK_WARMUP_STEPS K_MIN TOPK_LOG_EVERY DEVICE SEED

# shellcheck source=../common.sh
. "${{REPO_ROOT}}/hpc_transfer/common.sh"

run_transfer
"""


def _ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def measure(ds: str, tree: str) -> dict:
    run = sorted(glob.glob(os.path.join(RESULTS, ds, tree, "gnn_*")))[0]
    meta = json.load(open(os.path.join(run, "run_meta.json")))
    start = _ts(meta["started_utc"])
    train_h = (os.path.getmtime(os.path.join(run, "loss.csv")) - start) / 3600
    firsts = []
    for k in range(5):
        files = glob.glob(os.path.join(run, f"gen_seed_{k}", "*"))
        if files:
            firsts.append(min(map(os.path.getmtime, files)))
    gaps = [(b - a) / 3600 for a, b in zip(firsts, firsts[1:]) if b > a]
    # Seeds run in sequence give consistent gaps. Split runs (seeds as parallel
    # jobs, queued at different times) do not: their gaps are queue artefacts
    # (miniboone: 104 h, 0.4 h, 3.9 h, 9.6 h). Then fall back to the first seed
    # of the training job: end of training -> seed 0's metrics.
    if len(gaps) >= 2 and min(gaps) > 0.01 and max(gaps) / min(gaps) <= 3:
        seed_h, how = statistics.median(gaps), f"median of {len(gaps)} consecutive seed gaps"
    else:
        # sampling of seed 0 = end of training -> first file written in
        # gen_seed_0 (the training job sampled it right after training);
        # evaluation = median span between first and last file of a gen dir.
        loss_t = os.path.getmtime(os.path.join(run, "loss.csv"))
        sample_h = (firsts[0] - loss_t) / 3600
        spans = []
        for k in range(5):
            files = glob.glob(os.path.join(run, f"gen_seed_{k}", "*"))
            if files:
                t = list(map(os.path.getmtime, files))
                spans.append((max(t) - min(t)) / 3600)
        seed_h = sample_h + statistics.median(spans)
        how = (f"split run: seed-0 sampling {sample_h:.2f} h (end of training -> gen_seed_0) "
               f"+ median eval {statistics.median(spans):.2f} h")
    return {"run": os.path.relpath(run, REPO), "train_h": train_h, "seed_h": seed_h, "seed_how": how}


def main() -> int:
    from ablation_runner import results_subdir, training_setup
    import importlib
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ds, (tree, cat, group) in PLAN.items():
        mod = importlib.import_module(ds.replace("-", "_") + "_ablation")
        values, sources = training_setup(ds, mod.DATASET_CFG, "tabddpm", REPO)
        m = measure(ds, tree)
        est = m["train_h"] * values["steps"] / OLD_STEPS + 5 * m["seed_h"]
        hours = max(2, math.ceil(est * MARGIN))
        chain = hours > CAP_H
        if chain:
            train_t = max(2, math.ceil(m["train_h"] * values["steps"] / OLD_STEPS * MARGIN))
            seed_t = max(2, math.ceil(m["seed_h"] * MARGIN))
            job_h = min(CAP_H, max(train_t, seed_t))
            why = (f"monolithic estimate {est:.1f} h x {MARGIN} = {est * MARGIN:.0f} h > {CAP_H} h, so this "
                   f"dataset uses the SPLIT CHAIN: train-only (~{train_t} h) then one --load_checkpoint job "
                   f"per seed (~{seed_t} h each), all at TIME={job_h}:00:00. Submit with submit_split.sh, "
                   f"not sbatch.")
        else:
            job_h = hours
            why = (f"measured {m['train_h']:.2f} h training ({m['run']}) x {values['steps']}/{OLD_STEPS} steps "
                   f"+ 5 x {m['seed_h']:.2f} h per seed ({m['seed_how']}) = {est:.1f} h, x {MARGIN} margin, "
                   f"rounded up.")
        time = f"{job_h}:00:00"
        name = f"{ds}_fixed_tddpm"
        vals = "\n".join(f"#   {k:17s} {v!r:24} {sources[k]}" for k, v in values.items())
        chain_note = ("#\n# DO NOT sbatch this file directly for the full run: use\n"
                      f"#   TIME={time} TRAIN_ONLY=1 ./hpc_transfer/submit_split.sh {name}.sh --submit\n"
                      if chain else "")
        text = HEADER.format(short=f"{ds[:6]}_tddpm", name=name, time=time, ds=ds, cat=cat,
                             tree=results_subdir("fixed", cat, "tabddpm"), values=vals, why=why,
                             chain_note=chain_note)
        path = os.path.join(OUT, f"{name}.sh")
        with open(path, "w") as fh:
            fh.write(text)
        os.chmod(path, 0o755)
        rows.append({"dataset": ds, "group": group, "cat_input": cat, "current_run": m["run"],
                     "train_h_measured": round(m["train_h"], 2), "seed_h_measured": round(m["seed_h"], 2),
                     "seed_h_how": m["seed_how"], "new_steps": values["steps"],
                     "new_num_timesteps": values["num_timesteps"], "estimate_h": round(est, 1),
                     "walltime": time, "mode": "split chain" if chain else "single job"})
        print(f"{name}.sh  {time}  {'SPLIT CHAIN' if chain else 'single job'}  (est {est:.1f} h)")
    with open(os.path.join(OUT, "walltime_estimates.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

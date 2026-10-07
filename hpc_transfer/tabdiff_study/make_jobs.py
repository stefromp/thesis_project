#!/usr/bin/env python3
"""
Generate hpc_transfer/jobs_tabdiff/ (job scripts + walltime_estimates.csv) and
hpc_transfer/submit_tabdiff.sh (submission loop, every sbatch commented out).

    python3 hpc_transfer/tabdiff_study/make_jobs.py

ONE job per dataset (gpu partition, --gres=gpu:a100:1, --mem=24G), running
runner.py --dataset <ds>: train -> 20 tables -> TabDiff evaluation (Shape/Trend,
MLE gpu_hist, C2ST, alpha/beta) -> train the _dcr model -> 20 tables -> DCR
(magic: no DCR part). Each stage leaves a done-marker; a resubmitted job skips
finished stages.

Walltime = sum of the stage estimates x 1.3, rounded up, min 2 h. If that
exceeds 48 h the dataset is split into exactly two jobs, <ds>_main.sh and
<ds>_dcr.sh (--part main / --part dcr), chained with --dependency=afterok.

Stage estimates, measured not guessed:
  train, sample  per-step training time and per-(row x diffusion step) sampling
                 time of the SOURCE configuration, read from the synced A100 run
                 the current analysis uses (run_meta.json start -> loss.csv mtime
                 = 20 000 steps; consecutive gen_seed gaps = sampling of
                 num_samples rows at T=1000), scaled by steps, T, rows and graph
                 nodes -- linear (optimistic) and quadratic (pessimistic: the
                 dynamic graph scores all node pairs) in the node ratio; the
                 pessimistic value is used. magic's batch 256 is costed as 4096
                 (upper bound).
  eval_main      local real-vs-real timing of TabDiff's density+MLE+C2ST: 31 s per
                 news table (35 679 rows x 48 columns, 10 CPU cores, CPU XGBoost),
                 scaled by rows x columns; on the A100 gpu_hist only gets faster.
  eval_quality   local timing of TabDiff's alpha/beta (synthcity): ~10 s per shoppers
                 table (11 097 rows), scaled by rows^2 (nearest-neighbour based).
  eval_dcr       TabDiff's DCR on the GPU: seconds per table; 0.1 h flat.
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
sys.path.insert(0, HERE)
import study as S  # noqa: E402

REPO = S.REPO
OUT = os.path.join(REPO, "hpc_transfer", "jobs_tabdiff")
SUBMIT = os.path.join(REPO, "hpc_transfer", "submit_tabdiff.sh")
MARGIN = 1.3
CAP_H = 48
MIN_H = 2
OVERHEAD_H = 0.1                      # per target: data loading per table, process start-up
EVAL_MAIN_S_PER_CELL = 31.0 / (35679 * 48)
EVAL_QUALITY_S = 10.0                 # per 11 097-row table, x (n / 11 097)^2
EVAL_DCR_H = 0.1

# The synced runs the walltime is measured from (fixed rule, the analysis'
# cells; steps / T / batch = ablation_runner constants at the time).
SOURCE_RUNS = {
    "adult":   "transfer_runs/exp/adult/ablation_catfix/gnn_gcn_layers2_dim128_topk5_heads4_noattn",
    "default": "transfer_runs/exp/default/ablation_catfix/gnn_gatv2_layers3_dim128_topk5_heads8_noattn",
    "king":    "transfer_runs/exp/king/ablation_catfix/gnn_gatv2_layers3_dim128_topk5_heads8_noattn",
    "wilt":    "transfer_runs/exp/wilt/ablation_fixed_seed0/gnn_gatv2_layers2_dim64_topk3_heads8_noattn",
}
MEASURED_STEPS, MEASURED_T = 20_000, 1000


def _ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def measure(src: str) -> dict:
    run = os.path.join(REPO, SOURCE_RUNS[src])
    meta = json.load(open(os.path.join(run, "run_meta.json")))
    mod_cfg = S._ablation_runner(REPO) and __import__(f"{src}_ablation").DATASET_CFG
    t0 = _ts(meta["started_utc"])
    t_loss = os.path.getmtime(os.path.join(run, "loss.csv"))
    first, last = [], []
    for k in range(5):
        ts = [os.path.getmtime(f) for f in glob.glob(os.path.join(run, f"gen_seed_{k}", "*"))]
        first.append(min(ts))
        last.append(max(ts))
    # seed 0's first file can be rewritten later (metric backfill), so use
    # seeds 1..4: sampling = first file of seed k - last file of seed k-1
    gaps = [first[k] - last[k - 1] for k in range(1, 5)]
    sample_s = statistics.median(gaps)
    n_samples = mod_cfg["num_samples"]
    return {"run": SOURCE_RUNS[src], "train_s_per_step": (t_loss - t0) / MEASURED_STEPS,
            "sample_s_per_row_step": sample_s / (n_samples * MEASURED_T),
            "measured_num_samples": n_samples}


def nodes_source(src: str) -> int:
    import pandas as pd
    m = pd.read_csv(os.path.join(REPO, "hpc_transfer", "meta_features.csv")).set_index("dataset")
    return int(m.loc[src, "n_num"] + m.loc[src, "n_cat"])     # n_num includes the y node


def _info(target: str) -> dict:
    return json.load(open(os.path.join(S.data_dir(target), "info.json")))


def nodes_target(target: str) -> int:
    info = _info(target)
    return info["n_num_features"] + info["n_cat_features"] + (info["task_type"] == "regression")


def stage_estimates(ds: str, dcr: bool, m: dict, v: dict, ns: int) -> dict:
    """Hours per stage of one target (pessimistic) + the linear total."""
    tgt = S.target_name(ds, dcr)
    info = _info(tgt)
    n, nt = info["train_size"], nodes_target(tgt)
    r = nt / ns
    scale = max(r, r * r)                                     # pessimistic node scaling
    train_lin = v["steps"] * m["train_s_per_step"] * r / 3600
    samp_lin = S.N_SAMPLE_RUNS * n * v["num_timesteps"] * m["sample_s_per_row_step"] * r / 3600
    st = {"train": train_lin * scale / r + OVERHEAD_H / 2,
          "sample": samp_lin * scale / r + OVERHEAD_H / 2}
    if dcr:
        st["eval_dcr"] = EVAL_DCR_H
    else:
        n_cols = len(info["column_names"])
        st["eval_main"] = S.N_SAMPLE_RUNS * EVAL_MAIN_S_PER_CELL * n * n_cols / 3600
        st["eval_quality"] = S.N_SAMPLE_RUNS * EVAL_QUALITY_S * (n / 11097) ** 2 / 3600
    lin = train_lin + samp_lin + OVERHEAD_H + sum(h for k, h in st.items() if k.startswith("eval"))
    return {"target": tgt, "n_train": n, "nodes": nt, "stages_h": st, "linear_h": lin,
            "pessimistic_h": sum(st.values())}


def _hours(h: float) -> int:
    return max(MIN_H, math.ceil(h * MARGIN))


def estimate(ds: str) -> dict:
    src = S.SOURCES[ds]
    m = measure(src)
    v, _ = S.train_config(src)
    ns = nodes_source(src)
    parts = {"main": stage_estimates(ds, False, m, v, ns)}
    if S.HAS_DCR[ds]:
        parts["dcr"] = stage_estimates(ds, True, m, v, ns)
    total = sum(p["pessimistic_h"] for p in parts.values())
    split = _hours(total) > CAP_H
    jobs = ({f"{ds}_{k}": (k, _hours(p["pessimistic_h"])) for k, p in parts.items()} if split
            else {ds: ("all", _hours(total))})
    for name, (_, h) in jobs.items():
        if h > CAP_H:
            raise SystemExit(f"{name}: {h} h > {CAP_H} h even after splitting -- not generated")
    return {"dataset": ds, "source": src, "measured_run": m["run"], "steps": v["steps"],
            "T": v["num_timesteps"], "batch": v["batch_size"], "source_nodes": ns,
            "parts": parts, "gpu_h_linear": sum(p["linear_h"] for p in parts.values()),
            "gpu_h_pessimistic": total, "split": split, "jobs": jobs}


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
# TABDIFF-TRANSFER STUDY -- {ds} ({task}). ONE GPU job, part '{part}':
#   {stages}
# Every stage writes its outputs and a done-marker (<run>/stages/<stage>.done);
# resubmitting this script skips finished stages.{split_note}
#
# Data      : TabDiff's own splits data/{targets_data} (built with TabDiff's
#             process_dataset.py @ {commit_short} by hpc_transfer/tabdiff_study/prepare_data.py)
# Source    : {src} (nearest same-task TabDDPM dataset). Transferred:
{values}
# Model     : GNN denoiser {gnn}, cat_input={cat_input}, graph {graph}, global attention off;
#             divergence check on (divergence.json + exit != 0)
# Protocol  : {n_runs} tables of n_train rows per model (seeds 0..{last}); TabDiff's own
#             evaluation code (digests checked), eval venv + synthcity venv
# Results   : $TRANSFER_ROOT/exp_tabdiff/<target>/gnn/{combo}/
# Walltime {time}: {why}

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
EVAL_PYTHON="${{EVAL_PYTHON:-${{HPC_ROOT}}/venvs/tabdiff_eval310/bin/python}}"
SYNTHCITY_PYTHON="${{SYNTHCITY_PYTHON:-${{HPC_ROOT}}/venvs/tabdiff_synthcity310/bin/python}}"
TRANSFER_ROOT="${{TRANSFER_ROOT:-${{HPC_ROOT}}/transfer_runs}}"
DEVICE="${{DEVICE:-cuda:0}}"
SEED="${{SEED:-0}}"

cd "${{REPO_ROOT}}" || {{ echo "ERROR: cannot cd to ${{REPO_ROOT}}" >&2; exit 1; }}

DS="{ds}"
PART="{part}"

export HPC_ROOT REPO_ROOT PYTHON_BIN EVAL_PYTHON SYNTHCITY_PYTHON TRANSFER_ROOT DEVICE SEED DS PART

# shellcheck source=common_tabdiff.sh
. "${{REPO_ROOT}}/hpc_transfer/jobs_tabdiff/common_tabdiff.sh"

run_tabdiff_dataset
"""


def _fmt_values(ds: str) -> str:
    v, s = S.transferred(ds)
    return "\n".join(f"#   {k:17s} {v[k]!r:30} {s[k]}" for k in s)


def _stage_line(e: dict, part: str) -> str:
    keys = ["main", "dcr"] if part == "all" else [part]
    out = []
    for k in keys:
        if k in e["parts"]:
            p = e["parts"][k]
            out.append(" -> ".join(f"{p['target']}:{st} ({h:.2f} h)" for st, h in p["stages_h"].items()))
    return "\n#   -> ".join(out)


def write_jobs(est: dict) -> list:
    os.makedirs(OUT, exist_ok=True)
    for old in glob.glob(os.path.join(OUT, "*.sh")):
        if os.path.basename(old) != "common_tabdiff.sh":
            os.remove(old)
    written = []
    for ds in S.ORDER:
        e = est[ds]
        v, _ = S.transferred(ds)
        for job, (part, hours) in e["jobs"].items():
            time = f"{hours}:00:00"
            keys = ["main", "dcr"] if part == "all" else [part]
            est_h = sum(e["parts"][k]["pessimistic_h"] for k in keys if k in e["parts"])
            why = (f"sum of stage estimates {est_h:.2f} h x {MARGIN}, rounded up "
                   f"(measured from {e['measured_run']})")
            split_note = ""
            if e["split"]:
                split_note = ("\n# SPLIT: this dataset does not fit 48 h as one job; "
                              f"{ds}_dcr.sh must be submitted with --dependency=afterok:<{ds}_main.sh id>.")
            targets_data = "{" + ",".join(e["parts"][k]["target"] for k in keys if k in e["parts"]) + "}/"
            body = BODY.format(ds=ds, task=S.TASK[ds], part=part, stages=_stage_line(e, part),
                               split_note=split_note, targets_data=targets_data,
                               commit_short=S.TABDIFF_COMMIT[:7], src=S.SOURCES[ds],
                               values=_fmt_values(ds),
                               gnn=(f"{v['gnn_type']} L{v['n_layers']} d{v['d_model']} "
                                    f"top-k {v['top_k']} (fixed) heads {v['n_heads']}"),
                               cat_input=S.CAT_INPUT, graph=S.GRAPH_MODE,
                               n_runs=S.N_SAMPLE_RUNS, last=S.N_SAMPLE_RUNS - 1,
                               combo=S.combo_name(ds), time=time, why=why)
            path = os.path.join(OUT, f"{job}.sh")
            with open(path, "w") as fh:
                fh.write(HEADER.format(job=f"tdx_{job}", time=time) + body)
            os.chmod(path, 0o755)
            written.append(f"{job}.sh")
    return written


def write_submit(est: dict) -> None:
    lines = ["#!/usr/bin/env bash", "#",
             "# Submission loop for the TabDiff-transfer study. NOTHING HERE IS EXECUTED:",
             "# every sbatch line is commented out on purpose. One job per dataset (train ->",
             "# 20 tables -> TabDiff evaluation -> _dcr model -> 20 tables -> DCR), cheapest",
             "# first so a problem shows up early. A resubmitted job skips finished stages.",
             "# Regenerate jobs + this file: python3 hpc_transfer/tabdiff_study/make_jobs.py",
             "# Before the first submission run the cluster tests in",
             "# hpc_transfer/tabdiff_study/README.md (venvs, preflight, dry run).",
             "",
             'cd "$(dirname "$0")/jobs_tabdiff" || exit 1',
             ""]
    total = 0.0
    for ds in S.ORDER:
        e = est[ds]
        total += e["gpu_h_pessimistic"]
        if e["split"]:
            (m, (_, hm)), (d, (_, hd)) = e["jobs"].items()
            lines += [f"# {ds}: SPLIT into two chained jobs ({hm} h + {hd} h; estimate "
                      f"{e['gpu_h_linear']:.1f}-{e['gpu_h_pessimistic']:.1f} GPU-h)",
                      f"# jid=$(sbatch --parsable {m}.sh) && sbatch --dependency=afterok:${{jid}} {d}.sh"]
        else:
            (job, (_, h)), = e["jobs"].items()
            lines += [f"# {ds}: {h} h walltime (estimate {e['gpu_h_linear']:.1f}-"
                      f"{e['gpu_h_pessimistic']:.1f} GPU-h)",
                      f"# sbatch {job}.sh"]
    lines += ["", f"# total: ~{total:.1f} GPU-h (pessimistic, before the x{MARGIN} walltime margin)",
              "", "# Analysis afterwards (rerunnable): python3 analysis/tabdiff/compare.py", ""]
    with open(SUBMIT, "w") as fh:
        fh.write("\n".join(lines))
    os.chmod(SUBMIT, 0o755)


def main() -> int:
    sys.path.insert(0, os.path.join(REPO, "ablation"))
    est = {ds: estimate(ds) for ds in S.ORDER}
    rows = []
    for ds in S.ORDER:
        e = est[ds]
        for part, p in e["parts"].items():
            for stage, h in p["stages_h"].items():
                rows.append({"dataset": ds, "source": e["source"], "target": p["target"],
                             "stage": stage, "hours_pessimistic": round(h, 3),
                             "n_train": p["n_train"], "nodes": p["nodes"],
                             "source_nodes": e["source_nodes"], "steps": e["steps"], "T": e["T"],
                             "batch": e["batch"], "measured_run": e["measured_run"]})
        for job, (part, hours) in e["jobs"].items():
            rows.append({"dataset": ds, "source": e["source"], "target": f"JOB {job}.sh (part {part})",
                         "stage": "walltime", "hours_pessimistic": hours})
    os.makedirs(OUT, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, "walltime_estimates.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    written = write_jobs(est)
    write_submit(est)
    for ds in S.ORDER:
        e = est[ds]
        jobs = ", ".join(f"{j}.sh {h} h" for j, (_, h) in e["jobs"].items())
        print(f"  {ds:9s} <- {e['source']:8s} T={e['T']:<5d} {e['gpu_h_linear']:5.1f}-"
              f"{e['gpu_h_pessimistic']:5.1f} GPU-h -> {jobs}{'  (SPLIT)' if e['split'] else ''}")
    print(f"  total GPU-h: {sum(e['gpu_h_linear'] for e in est.values()):.1f} linear / "
          f"{sum(e['gpu_h_pessimistic'] for e in est.values()):.1f} pessimistic")
    print(f"  wrote {len(written)} job scripts to {OUT}, {SUBMIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

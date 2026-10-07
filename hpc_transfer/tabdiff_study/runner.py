#!/usr/bin/env python3
"""
The whole TabDiff-transfer job for ONE dataset, on one GPU, as a sequence of
stages with done-markers:

    python hpc_transfer/tabdiff_study/runner.py --dataset adult \\
        --exp_root $TRANSFER_ROOT --device cuda:0 \\
        --eval_python $EVAL_PYTHON --synthcity_python $SYNTHCITY_PYTHON [--part all|main|dcr]

  part main  (target tabdiff_<ds>)
    1 train         scripts/train.train with the source's TabDDPM training setup
                    (study.transferred), GNN denoiser (fixed top-k, cat_input=prob,
                    dynamic graph, no global attention), divergence check on
    2 sample        scripts/sample.sample, seeds 0..19, n_train rows each,
                    exported to samples/samples_<k>.csv (export.py)
    3 eval_main     TabDiff's Shape/Trend, MLE (XGBoost gpu_hist on this GPU), C2ST
                    (tabdiff_eval.py main, eval venv)
    4 eval_quality  TabDiff's alpha / beta (tabdiff_eval.py quality, synthcity venv)
  part dcr   (target tabdiff_<ds>_dcr; not magic)
    5 train, 6 sample, 7 eval_dcr  TabDiff's DCR (tabdiff_eval.py dcr, this GPU)

Each stage writes its outputs, then <run>/stages/<stage>.done. On resubmission
a stage whose marker is present AND whose outputs still check out is skipped,
so a crash or timeout never repeats finished work. A stage that does run
invalidates the markers of the later stages of the same target (and a
retrained model discards the old tables and evaluation). A model on disk that
was trained differently is refused, never silently reused.

tabdiff_eval.py stays usable on its own to re-run an evaluation on existing tables.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
from copy import deepcopy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study as S  # noqa: E402
import export as E  # noqa: E402

RUN_RECORD = "denoiser_config.json"
EVAL_DIR = "tabdiff_eval"
STAGE_DIR = "stages"
STAGES = {False: ["train", "sample", "eval_main", "eval_quality"],
          True:  ["train", "sample", "eval_dcr"]}
PARTS = ("all", "main", "dcr")
UPLOAD_COMMIT = os.path.join(HERE, "UPLOAD_COMMIT")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit(repo: str) -> str:
    """HEAD (+ '-dirty') from git; on the cluster, which has no git history,
    the UPLOAD_COMMIT file written locally at upload time (README)."""
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        pass
    if os.path.exists(UPLOAD_COMMIT):
        with open(UPLOAD_COMMIT) as fh:
            return fh.read().strip() + " (UPLOAD_COMMIT)"
    return os.environ.get("GIT_COMMIT", "unknown (no git history and no UPLOAD_COMMIT file)")


# ---------------------------------------------------------------------------
# Checkpoint record / refusal
# ---------------------------------------------------------------------------
def run_record(target: str, values: dict) -> dict:
    """What the checkpoint was trained with. A load must match it exactly."""
    ds, _ = S.parse_target(target)
    return {"study": S.STUDY, "target": target, "model": S.MODEL_DIR, "source": S.SOURCES[ds],
            "train_setup": S.TRAIN_SETUP,
            "lr": values["lr"], "steps": values["steps"], "batch_size": values["batch_size"],
            "num_timesteps": values["num_timesteps"], "scheduler": values["scheduler"],
            "weight_decay": values["weight_decay"], "normalization": values["normalization"],
            "cat_input": S.CAT_INPUT, "topk_mode": S.TOPK_MODE, "graph_mode": S.GRAPH_MODE,
            "use_attention": S.USE_ATTENTION}


class CheckpointMismatch(RuntimeError):
    pass


def check_checkpoint(run: str, want: dict) -> bool:
    """True = a matching trained model is there; False = none.
    Raises on a checkpoint trained differently, or a model without a record."""
    model = os.path.join(run, "model_ema.pt")
    rec_p = os.path.join(run, RUN_RECORD)
    if not os.path.exists(model):
        return False
    if not os.path.exists(rec_p):
        raise CheckpointMismatch(f"refusing to load {model}: no {RUN_RECORD} records what it was "
                                 f"trained with. Move it away to retrain.")
    with open(rec_p) as fh:
        have = json.load(fh)
    diff = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
    if diff:
        lines = "\n".join(f"  {k}: checkpoint {a!r} vs this run {b!r}" for k, (a, b) in diff.items())
        raise CheckpointMismatch(f"refusing to load {model}: it was trained differently:\n{lines}")
    return True


def model_and_graph(ds: str, values: dict, data_info: dict):
    task = data_info["task_type"]
    num_classes = data_info.get("n_classes", 0) if task != "regression" else 0
    if values["is_y_cond"] != (task != "regression"):
        raise ValueError(f"source {S.SOURCES[ds]} is_y_cond={values['is_y_cond']} does not fit "
                         f"a {task} target (TabDDPM convention: y-conditioning for classification)")
    model_params = dict(num_classes=num_classes, is_y_cond=values["is_y_cond"],
                        rtdl_params=dict(d_layers=[], dropout=0.0))
    graph_params = dict(enabled=True, mode=S.GRAPH_MODE, gnn_type=values["gnn_type"],
                        d_model=values["d_model"], n_layers=values["n_layers"],
                        n_heads=values["n_heads"], sparsity_top_k=values["top_k"],
                        dropout=0.0, use_attention=S.USE_ATTENTION, topk_mode=S.TOPK_MODE,
                        k_min=1, topk_warmup_steps=0,
                        num_timesteps=values["num_timesteps"], cat_input=S.CAT_INPUT)
    return model_params, graph_params


def T_dict(values: dict, seed: int) -> dict:
    # As ablation_runner._run_one's T_dict_train; normalization from the source.
    return dict(seed=seed, normalization=values["normalization"], num_nan_policy=None,
                cat_nan_policy=None, cat_min_frequency=None, cat_encoding=None,
                y_policy="default")


# ---------------------------------------------------------------------------
# Evaluation outputs (resume checks)
# ---------------------------------------------------------------------------
EVAL_STEP = {"eval_main": "main", "eval_quality": "quality", "eval_dcr": "dcr"}


def eval_step_done(run: str, step: str, n_runs: int) -> bool:
    """Outputs of tabdiff_eval.py `step` are complete for n_runs tables."""
    d = os.path.join(run, EVAL_DIR)
    meta_p = os.path.join(d, "eval_meta.json")
    if not os.path.exists(meta_p):
        return False
    with open(meta_p) as fh:
        meta = json.load(fh)
    if step not in meta or meta[step].get("n_runs") != n_runs:
        return False
    name, cols = {"main": ("all_results.csv", {"shape", "trend", "mle", "c2st"}),
                  "dcr": ("all_results.csv", {"dcr"}),
                  "quality": ("quality.csv", {"alpha", "beta"})}[step]
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return False
    with open(p) as fh:
        rows = list(csv.DictReader(fh))
    return len(rows) == n_runs and cols <= set(rows[0] if rows else {})


def samples_done(run: str, data: str, n_runs: int) -> bool:
    paths = [os.path.join(run, "samples", f"samples_{k}.csv") for k in range(n_runs)]
    return all(os.path.exists(p) for p in paths)


# ---------------------------------------------------------------------------
# Stage markers
# ---------------------------------------------------------------------------
def marker(run: str, stage: str) -> str:
    return os.path.join(run, STAGE_DIR, f"{stage}.done")


def mark_done(run: str, stage: str, info: dict) -> None:
    os.makedirs(os.path.join(run, STAGE_DIR), exist_ok=True)
    tmp = marker(run, stage) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"stage": stage, "finished_utc": _now(), **info}, fh, indent=2)
    os.replace(tmp, marker(run, stage))


def clear_after(run: str, stages: list, stage: str) -> None:
    for later in stages[stages.index(stage) + 1:]:
        if os.path.exists(marker(run, later)):
            os.remove(marker(run, later))


# ---------------------------------------------------------------------------
# One target
# ---------------------------------------------------------------------------
class Target:
    def __init__(self, ds: str, dcr: bool, args, repo: str, tabdiff_info: dict):
        self.ds, self.dcr = ds, dcr
        self.name = S.target_name(ds, dcr)
        self.args, self.repo, self.tabdiff_info = args, repo, tabdiff_info
        self.data = S.data_dir(self.name, repo)
        self.info = E.load_info(self.data)
        self.values, self.sources = S.transferred(ds, repo)
        self.run = S.run_dir(args.exp_root, self.name, repo)
        self.n_train = int(self.info["train_size"])
        self.want = run_record(self.name, self.values)
        self.stages = STAGES[dcr]

    def meta(self, extra=None) -> None:
        meta_p = os.path.join(self.run, "run_meta.json")
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        manifest_p = os.path.join(self.repo, "data", "tabdiff_manifest.json")
        manifest = json.load(open(manifest_p)) if os.path.exists(manifest_p) else {}
        meta.update({
            "study": S.STUDY,
            "dataset": self.ds,
            "target": self.name,
            "tabdiff_dataset": S.tabdiff_dataname(self.name),
            "dcr_model": self.dcr,
            "model": "GNN denoiser",
            "source_dataset": S.SOURCES[self.ds],
            "source_config_paths": sorted({v.split(":")[0].split(" ")[0] for v in self.sources.values()
                                           if not v.startswith("absent")}),
            "train_setup": S.TRAIN_SETUP,
            "transferred": {k: {"value": self.values[k], "source": self.sources[k]} for k in self.sources},
            "cat_input": S.CAT_INPUT, "topk_mode": S.TOPK_MODE, "graph_mode": S.GRAPH_MODE,
            "use_attention": S.USE_ATTENTION, "divergence_check": True,
            "checkpoint_used": "model_ema.pt (final EMA)",
            "n_sample_runs": self.args.n_runs,
            "num_samples_per_run": self.n_train,
            "sample_seeds": list(range(self.args.n_runs)),
            "train_seed": self.args.seed,
            "stages": self.stages,
            "data_dir": os.path.relpath(self.data, self.repo),
            "data_files_sha256_16": manifest.get("datasets", {}).get(self.name, {}).get("files_sha256_16"),
            "tabdiff": self.tabdiff_info,
            "deviations_from_tabdiff": S.DEVIATIONS,
            "git_commit": _git_commit(self.repo),
            "hostname": socket.gethostname(),
            "device": self.args.device,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
        })
        meta.setdefault("started_utc", _now())
        if extra:
            meta.update(extra)
        os.makedirs(self.run, exist_ok=True)
        with open(meta_p, "w") as fh:
            json.dump(meta, fh, indent=2)

    def outputs_ok(self, stage: str) -> bool:
        """A marker counts only while its outputs still check out."""
        if stage == "train":
            return check_checkpoint(self.run, self.want)          # raises on mismatch
        if stage == "sample":
            return samples_done(self.run, self.data, self.args.n_runs)
        return eval_step_done(self.run, EVAL_STEP[stage], self.args.n_runs)

    def banner(self) -> None:
        print("=" * 72)
        print(f"  target       : {self.name}  (TabDiff '{S.tabdiff_dataname(self.name)}', "
              f"{self.info['task_type']}, n_train={self.n_train})")
        print(f"  source       : {S.SOURCES[self.ds]}")
        for k in self.sources:
            print(f"    {k:17s} = {self.values[k]!r:28}  <- {self.sources[k]}")
        print(f"  run dir      : {self.run}")
        print(f"  stages       : {' -> '.join(self.stages)}")
        print(f"  sampling     : {self.args.n_runs} tables x {self.n_train} rows")
        print("=" * 72, flush=True)


class Heavy:
    """The expensive operations, separated so tests can replace them."""

    def __init__(self, repo: str):
        self.repo = repo
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        for p in (self.repo, os.path.join(self.repo, "scripts"), os.path.join(self.repo, "ablation")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import ablation_runner as ar
        ar._setup_zero_rtdl_stubs()
        import torch
        from train import train
        from sample import sample
        self.torch, self.train_fn, self.sample_fn = torch, train, sample
        self._loaded = True

    def _common(self, t: Target) -> dict:
        model_params, graph_params = model_and_graph(t.ds, t.values, t.info)
        self.model_params = model_params
        return dict(real_data_path=t.data, model_type="mlp", T_dict=T_dict(t.values, t.args.seed),
                    num_numerical_features=t.info["n_num_features"],
                    device=self.torch.device(t.args.device), change_val=False,
                    graph_params=graph_params, num_timesteps=t.values["num_timesteps"],
                    gaussian_loss_type="mse", scheduler=t.values["scheduler"])

    def train(self, t: Target) -> None:
        self._load()
        common = self._common(t)
        self.train_fn(parent_dir=t.run, model_params=deepcopy(self.model_params), seed=t.args.seed,
                      lr=t.values["lr"], weight_decay=t.values["weight_decay"],
                      batch_size=t.values["batch_size"], steps=t.values["steps"],
                      checkpoint_every=0, divergence_check=True, **common)

    def sample(self, t: Target, k: int, gen_dir: str) -> None:
        self._load()
        common = self._common(t)
        self.sample_fn(parent_dir=gen_dir, model_path=os.path.join(t.run, "model_ema.pt"),
                       model_params=deepcopy(self.model_params), seed=k, num_samples=t.n_train,
                       batch_size=t.values["sample_batch_size"], **common)
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def evaluate(self, cmd: list) -> int:
        return subprocess.run(cmd).returncode


def _discard(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def run_stage(t: Target, stage: str, heavy: Heavy) -> None:
    a = t.args
    clear_after(t.run, t.stages, stage)
    if stage == "train":
        if check_checkpoint(t.run, t.want):
            print(f"  [LOAD] matching model_ema.pt found (crash after training) -- not retrained")
        else:
            # a new model makes every table and evaluation of the old one stale
            for p in [os.path.join(t.run, "samples"), os.path.join(t.run, EVAL_DIR)] + \
                     glob.glob(os.path.join(t.run, "gen_seed_*")):
                _discard(p)
            with open(os.path.join(t.run, "train_setup.json"), "w") as fh:
                json.dump({"train_setup": S.TRAIN_SETUP, "values": t.values, "sources": t.sources},
                          fh, indent=2)
            t0 = _now()
            heavy.train(t)
            with open(os.path.join(t.run, RUN_RECORD), "w") as fh:
                json.dump(t.want, fh, indent=2)
            t.meta({"train_started_utc": t0, "train_finished_utc": _now()})
        mark_done(t.run, stage, {"model": "model_ema.pt"})
        return

    if stage == "sample":
        regenerated = False
        t0 = _now()
        for k in range(a.n_runs):
            out_csv = os.path.join(t.run, "samples", f"samples_{k}.csv")
            if os.path.exists(out_csv):
                try:
                    E.check_against_real(out_csv, t.data)
                    print(f"  [KEEP] samples_{k}.csv")
                    continue
                except Exception as exc:
                    print(f"  [REDO] samples_{k}.csv: {exc}")
            gen = os.path.join(t.run, f"gen_seed_{k}")
            os.makedirs(gen, exist_ok=True)
            heavy.sample(t, k, gen)
            E.export_seed(gen, t.data, out_csv)
            regenerated = True
            print(f"  [OK] samples_{k}.csv ({t.n_train} rows)", flush=True)
        if regenerated:
            _discard(os.path.join(t.run, EVAL_DIR))
        t.meta({"sampling_started_utc": t0, "sampling_finished_utc": _now()})
        mark_done(t.run, stage, {"n_runs": a.n_runs, "rows_per_table": t.n_train})
        return

    step = EVAL_STEP[stage]
    py = a.synthcity_python if step == "quality" else a.eval_python
    cmd = [py, os.path.join(HERE, "tabdiff_eval.py"), step, "--run_dir", t.run,
           "--target", t.name, "--repo_root", t.repo, "--exp_root", a.exp_root,
           "--n_runs", str(a.n_runs)]
    if step != "quality":
        cmd += ["--device", a.device]
    print(f"  [EVAL {step}] " + " ".join(cmd), flush=True)
    t0 = _now()
    rc = heavy.evaluate(cmd)
    if rc != 0:
        raise RuntimeError(f"evaluation step '{step}' exited {rc}")
    if not eval_step_done(t.run, step, a.n_runs):
        raise RuntimeError(f"evaluation step '{step}' finished but its outputs are incomplete")
    t.meta({f"{stage}_started_utc": t0, f"{stage}_finished_utc": _now()})
    mark_done(t.run, stage, {"step": step, "n_runs": a.n_runs})


def plan(ds: str, part: str):
    parts = {"all": [False, True], "main": [False], "dcr": [True]}[part]
    return [d for d in parts if not d or S.HAS_DCR[ds]]


def run_dataset(args, heavy: Heavy = None) -> int:
    repo = os.path.abspath(args.repo_root)
    tabdiff_info = S.verify_tabdiff(os.path.join(repo, S.TABDIFF_REL))
    if args.part == "dcr" and not S.HAS_DCR[args.dataset]:
        raise SystemExit(f"{args.dataset} has no DCR part")
    heavy = heavy or Heavy(repo)
    print(f"  study {S.STUDY} -- dataset {args.dataset}, part {args.part}, "
          f"TabDiff {tabdiff_info['commit']} (digests verified)", flush=True)
    for dcr in plan(args.dataset, args.part):
        t = Target(args.dataset, dcr, args, repo, tabdiff_info)
        t.banner()
        if args.dry_run:
            state = {s: os.path.exists(marker(t.run, s)) for s in t.stages}
            have = check_checkpoint(t.run, t.want) if os.path.isdir(t.run) else False
            print(f"  [DRY RUN] stage markers {state}; matching model on disk: {have}")
            continue
        t.meta()
        for stage in t.stages:
            if os.path.exists(marker(t.run, stage)) and t.outputs_ok(stage):
                print(f"  [SKIP] {t.name}: {stage} done")
                continue
            print(f"  [STAGE] {t.name}: {stage}  ({_now()})", flush=True)
            run_stage(t, stage, heavy)
        print(f"  [DONE] {t.name}  {_now()}", flush=True)
    return 0


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=S.ORDER)
    ap.add_argument("--part", choices=PARTS, default="all",
                    help="all (default); main / dcr only when a dataset is split into two jobs")
    ap.add_argument("--exp_root", required=True, help="results go to <exp_root>/exp_tabdiff/...")
    ap.add_argument("--repo_root", default=S.REPO)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0, help="training seed")
    ap.add_argument("--eval_python", default=None, help="python of the eval venv")
    ap.add_argument("--synthcity_python", default=None, help="python of the synthcity venv")
    ap.add_argument("--n_runs", type=int, default=S.N_SAMPLE_RUNS, help=argparse.SUPPRESS)
    ap.add_argument("--dry_run", action="store_true",
                    help="print the plan, values and stage state; run nothing")
    return ap


def main(argv=None) -> int:
    ap = make_parser()
    args = ap.parse_args(argv)
    if not args.dry_run:
        if not args.eval_python:
            ap.error("--eval_python is required (TabDiff's evaluation runs in this job)")
        if args.part in ("all", "main") and not args.synthcity_python:
            ap.error("--synthcity_python is required for the main part (alpha/beta)")
    return run_dataset(args)


if __name__ == "__main__":
    sys.exit(main())

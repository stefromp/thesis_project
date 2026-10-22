#!/usr/bin/env python3
"""
One Route-A job: ONE dataset x ONE denoiser arm x ONE training seed, on one
GPU, as stages with done-markers (same conventions as the TabDiff study's
hpc_transfer/tabdiff_study/runner.py):

    python tabdiff_gnn/runner.py --dataset magic --denoiser attngnn --seed 0 \\
        --exp_root $TRANSFER_ROOT --device cuda:0 --train_python $EVAL_PYTHON \\
        --eval_python $EVAL_PYTHON --synthcity_python $SYNTHCITY_PYTHON \\
        --tabpc_python $TABPC_PYTHON [--stages ...]

  target tabdiff_<ds> (no DCR in this study: no _dcr model)
    1 train         entry.py train  -- TabDiff's Trainer.run_loop, 8000 epochs,
                                       best EMA after epoch 4000 (train python)
    2 sample        entry.py sample -- 20 tables of n_train rows, generation seeds 0..19
    3 eval_main     evaluate.py main    TabDiff Shape/Trend, MLE (gpu_hist) + MLE-2, C2ST
    4 eval_quality  evaluate.py quality TabDiff alpha / beta (synthcity venv)
    5 eval_tabpc    tabpc_eval.py       wNMIS, C2ST (XGBoost) (tabpc venv)

Every stage runs as its own process (fresh CUDA context), writes its outputs,
then <run>/stages/<stage>.done. A resubmitted job skips a stage whose marker
is present AND whose outputs still check out; a stage that runs clears the
later markers; a retrained model discards the old tables and
evaluations. A model on disk trained with a different spec is refused.
TabDiff's Trainer cannot resume a run faithfully (optimizer, EMA, LR
scheduler and best-loss state are not checkpointed), so an interrupted
training restarts from epoch 0 -- each walltime covers a full training run.

--dry_run prints the plan and the stage state, and imports TabDiff + AttnGNN and
instantiates the arm's denoiser on the dataset's real shapes with one forward
pass (CPU) -- run it with the train python (eval venv) so a missing package
fails on the login node, not in a GPU job.

Results: <exp_root>/exp_tabdiff_gnn/tabdiff_<ds>/<denoiser>/<arm>/seed<s>/
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402

STAGES = ["train", "sample", "eval_main", "eval_quality", "eval_tabpc"]
SPEC_KEYS = ("study", "target", "denoiser", "seed", "variant", "gnn", "dim_t", "node_noise",
             "geo_loss", "geo_edge_bias", "smoke_epochs", "trainer", "ablation")
PAUSED_EXIT = 75          # entry.py: resumable training paused on its time budget


class Paused(Exception):
    """Resumable training saved its state and paused; the next chain part resumes it."""
UPLOAD_COMMIT = os.path.join(HERE, "UPLOAD_COMMIT")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(repo: str) -> str:
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        pass
    if os.path.exists(UPLOAD_COMMIT):
        return open(UPLOAD_COMMIT).read().strip() + " (UPLOAD_COMMIT)"
    return os.environ.get("GIT_COMMIT", "unknown (no git history and no UPLOAD_COMMIT file)")


class SpecMismatch(RuntimeError):
    pass


def _json(p):
    with open(p) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Output checks
# ---------------------------------------------------------------------------
def trained_ok(run: str, want: dict) -> bool:
    """True = a finished model with this spec is there; False = none.
    Raises on a model trained with another spec."""
    spec_p, rec_p = os.path.join(run, "route_a_spec.json"), os.path.join(run, "train_record.json")
    if not os.path.exists(spec_p):
        return False
    have = _json(spec_p)
    diff = {k: (have.get(k), want.get(k)) for k in SPEC_KEYS if have.get(k) != want.get(k)}
    if diff:
        lines = "\n".join(f"  {k}: on disk {a!r} vs this job {b!r}" for k, (a, b) in diff.items())
        raise SpecMismatch(f"refusing to reuse {run}: trained with another spec:\n{lines}\n"
                           f"move the directory away to retrain")
    if not os.path.exists(rec_p):
        return False                                    # training never finished
    rec = _json(rec_p)
    return bool(rec.get("completed")) and os.path.exists(os.path.join(run, "ckpt", rec.get("checkpoint", "-")))


def samples_ok(run: str, n_runs: int) -> bool:
    rec_p = os.path.join(run, "sample_record.json")
    if not os.path.exists(rec_p):
        return False
    rec, train = _json(rec_p), _json(os.path.join(run, "train_record.json"))
    if rec.get("checkpoint") != train.get("checkpoint") or rec.get("n_tables") != n_runs:
        return False
    return all(os.path.exists(os.path.join(run, "samples", f"samples_{k}.csv")) for k in range(n_runs))


def eval_ok(run: str, step: str, n_runs: int) -> bool:
    if step == "tabpc":
        import tabpc_eval
        return tabpc_eval.done(run, n_runs)
    d = os.path.join(run, "tabdiff_eval")
    meta_p = os.path.join(d, "eval_meta.json")
    if not os.path.exists(meta_p):
        return False
    meta = _json(meta_p)
    if step not in meta or meta[step].get("n_runs") != n_runs:
        return False
    need = {"main": [("all_results.csv", {"shape", "trend", "mle", "c2st"}), ("mle2.csv", {"run"})],
            "quality": [("quality.csv", {"alpha", "beta"})]}[step]
    for name, cols in need:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            return False
        with open(p) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != n_runs or not cols <= set(rows[0]):
            return False
    return True


EVAL_STEP = {"eval_main": "main", "eval_quality": "quality", "eval_tabpc": "tabpc"}


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def marker(run: str, stage: str) -> str:
    return os.path.join(run, "stages", f"{stage}.done")


def mark_done(run: str, stage: str, info: dict) -> None:
    os.makedirs(os.path.join(run, "stages"), exist_ok=True)
    tmp = marker(run, stage) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"stage": stage, "finished_utc": _now(), **info}, fh, indent=2)
    os.replace(tmp, marker(run, stage))


def clear_after(run: str, stages: list, stage: str) -> None:
    for later in stages[stages.index(stage) + 1:]:
        if os.path.exists(marker(run, later)):
            os.remove(marker(run, later))


def _discard(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# One target
# ---------------------------------------------------------------------------
class Target:
    def __init__(self, ds: str, args, repo: str):
        import entry
        self.ds, self.args, self.repo = ds, args, repo
        self.name = C.target_name(ds)
        self.files = C.data_files(self.name, repo)
        self.info = _json(self.files["info"])
        self.run = C.run_dir(args.exp_root, self.name, args.denoiser, args.seed,
                             node_noise=args.node_noise, repo_root=repo, ablation=args.ablation,
                             topk_mode=args.topk_mode)
        self.want = entry.spec_for(self.name, args.denoiser, args.seed, node_noise=args.node_noise,
                                   geo_loss=args.geo_loss, geo_edge_bias=args.geo_edge_bias,
                                   ablation=args.ablation, topk_mode=args.topk_mode)
        self.want["smoke_epochs"] = args.smoke_epochs
        self.stages = list(STAGES)

    def meta(self, extra=None) -> None:
        p = os.path.join(self.run, "run_meta.json")
        meta = _json(p) if os.path.exists(p) else {}
        man_p = os.path.join(self.repo, C.DATA_REL, "manifest.json")
        man = _json(man_p) if os.path.exists(man_p) else {}
        spec_p = os.path.join(self.run, "route_a_spec.json")
        spec = _json(spec_p) if os.path.exists(spec_p) else {}
        a = self.args
        meta.update({
            "study": C.STUDY,
            "dataset": self.ds, "target": self.name, "tabdiff_dataset": C.tabdiff_dataname(self.name),
            "dcr": "not part of this study", "task_type": self.info["task_type"],
            "denoiser": a.denoiser,
            "variant": C.VARIANT if a.denoiser == "attngnn" else None,
            "arm": os.path.basename(os.path.dirname(self.run)),
            "d_model_rule": self.want.get("d_model_rule"),
            "topk_mode": a.topk_mode,
            "rule_settings": self.want.get("rule_settings") or {
                "topk_mode": "fixed", "top_k": (self.want.get("gnn") or {}).get("top_k"),
                "rule": "fixed top-k: each node keeps its top_k highest-scoring neighbours"},
            "k_logging": (None if a.topk_mode == "fixed" else
                          "topk_k_stats.csv / topk_k_hist.csv in the run dir (tabdiff_gnn/topk_log.py): "
                          "selected k per (phase, block, t bin, node), t in [0,1] from c_noise"),
            "ablation": (None if a.ablation is None else {
                "name": a.ablation, **C.ABLATIONS[a.ablation],
                "main_run_dir": C.run_dir(a.exp_root, self.name, a.denoiser, a.seed,
                                          node_noise=a.node_noise, repo_root=self.repo,
                                          topk_mode=a.topk_mode),
                "note": "ABLATION -- not part of the main Route-A protocol; compare only with the "
                        "main run of the same dataset and seed"}),
            "train_seed": a.seed,
            "gnn": self.want.get("gnn"), "gnn_sources": self.want.get("gnn_sources"),
            "gnn_source_dataset": self.want.get("gnn_source_dataset"),
            "gnn_source_note": C.SOURCE_NOTES.get(self.ds),
            "gnn_source_is_our_choice": self.ds in C.SOURCE_NOTES if a.denoiser == "attngnn" else None,
            "dim_t": self.want.get("dim_t"), "node_noise": self.want.get("node_noise"),
            "trainer": ("resumable copy of TabDiff's Trainer.run_loop (verbatim + save/restore hooks)"
                        if self.want.get("trainer") == "resumable" else "TabDiff's Trainer, unchanged"),
            "trainer_evidence": C.TRAINER_EVIDENCE if self.want.get("trainer") == "resumable" else None,
            "geo_loss": a.geo_loss, "geo_edge_bias": a.geo_edge_bias,
            "tabdiff_config": {k: spec.get(k) for k in ("raw_config_train", "diffusion_params",
                                                        "unimodmlp_params")},
            "param_counts": spec.get("param_counts"),
            "protocol": {"n_tables": a.n_runs, "rows_per_table": self.info.get("train_num"),
                         "generation_seeds": list(range(a.n_runs)),
                         "checkpoint": "best EMA training loss after epoch 4000 (TabDiff Trainer)",
                         "sampling_steps": 50, "train_seeds_planned": list(C.TRAIN_SEEDS)},
            "stages": self.stages,
            "smoke_epochs": a.smoke_epochs,
            "data_dir": os.path.relpath(self.files["dir"], self.repo),
            "data_files_sha256_16": man.get("datasets", {}).get(self.name, {}).get("files_sha256_16"),
            "tabdiff": C.verify_tabdiff(self.repo),
            "tabpc": {"commit": C.TABPC_COMMIT, "url": C.TABPC_URL},
            "deviations_from_tabdiff": C.DEVIATIONS + (C.DEVIATIONS_ATTNGNN if a.denoiser == "attngnn" else [])
                                       + (C.DEVIATIONS_RESUMABLE if self.want.get("trainer") == "resumable" else []),
            "git_commit": git_commit(self.repo),
            "hostname": socket.gethostname(), "device": a.device,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
        })
        for rec in ("train_record.json", "sample_record.json"):
            rp = os.path.join(self.run, rec)
            if os.path.exists(rp):
                meta[rec[:-5]] = _json(rp)
        meta.setdefault("started_utc", _now())
        if extra:
            meta.update(extra)
        os.makedirs(self.run, exist_ok=True)
        with open(p, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

    def outputs_ok(self, stage: str) -> bool:
        if stage == "train":
            return trained_ok(self.run, self.want)
        if stage == "sample":
            return samples_ok(self.run, self.args.n_runs)
        return eval_ok(self.run, EVAL_STEP[stage], self.args.n_runs)

    def banner(self) -> None:
        print("=" * 72)
        print(f"  target   : {self.name} ({self.info['task_type']}, n_train={self.info.get('train_num')}, "
              f"val.csv={'yes' if self.files['val'] else 'no'})")
        print(f"  denoiser : {self.args.denoiser}  seed {self.args.seed}"
)
        if self.want.get("gnn"):
            for k, v in self.want["gnn"].items():
                print(f"    {k:14s} = {v!r:10}  <- {self.want['gnn_sources'].get(k, '--topk_mode (thesis catfix settings)' if k in ('topk_mode', 'k_min', 'topk_warmup_steps') and self.args.topk_mode != 'fixed' else 'route-A fixed')}")
            print(f"    dim_t={self.want['dim_t']}  node_noise={self.want['node_noise']}")
            if self.ds in C.SOURCE_NOTES:
                print(f"    GNN source {C.SOURCES[self.ds]}: {C.SOURCE_NOTES[self.ds]}")
        print(f"  run dir  : {self.run}")
        print(f"  stages   : {' -> '.join(self.stages)}")
        print("=" * 72, flush=True)


class Exec:
    """Runs stage commands (tests replace it)."""

    def run(self, cmd: list) -> int:
        print("  $ " + " ".join(cmd), flush=True)
        return subprocess.run(cmd).returncode


def stage_cmd(t: Target, stage: str) -> list:
    a = t.args
    if stage == "train":
        cmd = [a.train_python, os.path.join(HERE, "entry.py"), "train", "--target", t.name,
               "--denoiser", a.denoiser, "--seed", str(a.seed), "--run_dir", t.run, "--device", a.device,
               "--node_noise", a.node_noise] + (["--ablation", a.ablation] if a.ablation else []) \
              + (["--topk_mode", a.topk_mode] if a.topk_mode != "fixed" else [])
        if a.geo_loss:
            cmd += ["--geo_loss"]
        if a.geo_edge_bias:
            cmd += ["--geo_edge_bias"]
        if a.smoke_epochs is not None:
            cmd += ["--smoke_epochs", str(a.smoke_epochs)]
        if a.train_budget_h is not None and t.want.get("trainer") == "resumable":
            cmd += ["--train_budget_h", str(a.train_budget_h)]
        return cmd
    if stage == "sample":
        return [a.train_python, os.path.join(HERE, "entry.py"), "sample", "--target", t.name,
                "--run_dir", t.run, "--device", a.device, "--n_tables", str(a.n_runs)]
    step = EVAL_STEP[stage]
    if step == "tabpc":
        return [a.tabpc_python, os.path.join(HERE, "tabpc_eval.py"), "--run_dir", t.run,
                "--target", t.name, "--n_runs", str(a.n_runs)] + (["--no_pin_check"] if a.no_pin_check else [])
    py = a.synthcity_python if step == "quality" else a.eval_python
    cmd = [py, os.path.join(HERE, "evaluate.py"), step, "--run_dir", t.run, "--target", t.name,
           "--exp_root", a.exp_root, "--n_runs", str(a.n_runs)]
    if step != "quality":
        cmd += ["--device", a.device]
    return cmd + (["--no_pin_check"] if a.no_pin_check else [])


def run_stage(t: Target, stage: str, ex: Exec) -> None:
    clear_after(t.run, t.stages, stage)
    if stage == "train":
        if trained_ok(t.run, t.want):
            print("  [LOAD] finished model with this spec found (crash after training) -- not retrained")
            mark_done(t.run, stage, {"n_runs": t.args.n_runs, "reused": True})
            return
        if os.path.exists(os.path.join(t.run, "ckpt", "resume_state.pt")):
            # resumable training (news) part-way through: keep it, entry.py resumes it
            # (trained_ok above has already refused a run state with another spec)
            print("  [RESUME] saved run state found -- training continues from it")
        else:
            for p in ("samples", "tabdiff_eval", "tabpc_eval", "ckpt", "result", "sample_record.json",
                      "train_record.json", "route_a_spec.json", "train_log.csv"):
                _discard(os.path.join(t.run, p))
    if stage == "sample":
        for p in ("tabdiff_eval", "tabpc_eval"):          # new tables -> stale evaluations
            _discard(os.path.join(t.run, p))
    t0 = _now()
    rc = ex.run(stage_cmd(t, stage))
    if stage == "train" and rc == PAUSED_EXIT:
        t.meta({"train_paused_utc": _now()})
        raise Paused(f"{t.name}: training paused on its time budget (state saved); "
                     f"the next part of the chain resumes it")
    if rc != 0:
        raise RuntimeError(f"{t.name}: stage '{stage}' exited {rc}")
    if not t.outputs_ok(stage):
        raise RuntimeError(f"{t.name}: stage '{stage}' finished but its outputs are incomplete")
    t.meta({f"{stage}_started_utc": t0, f"{stage}_finished_utc": _now()})
    mark_done(t.run, stage, {"n_runs": t.args.n_runs})


def instantiate_check(t: "Target", topk_mode: str | None = None) -> dict:
    """Import TabDiff + AttnGNN, build this job's denoiser on the dataset's real
    shapes (TabDiff's own TabDiffDataset) inside Model/Precond, one CPU forward
    pass. Proves the venv can build the model before a GPU is allocated."""
    import torch
    import entry
    import params
    from denoiser import build_denoiser, count_params
    _, _, _, Model, _, _ = entry._tabdiff_imports()
    d_num, cats, _ = params.shapes(t.name)
    raw = entry.load_raw_config()
    up = dict(raw["unimodmlp_params"], d_numerical=d_num, categories=[k + 1 for k in cats])
    gnn = t.want.get("gnn")
    if topk_mode is not None and gnn is not None:      # every rule on this dataset's shapes
        gnn = {**{k: v for k, v in gnn.items() if k not in ("k_min", "topk_warmup_steps")},
               "topk_mode": "fixed", **C.gnn_rule_fields(topk_mode)}
        if topk_mode != "fixed":
            import entmax  # noqa: F401  (adaptive_entmax15 needs it; fail here, not in a GPU job)
    m = build_denoiser(t.want["denoiser"], up, gnn, dim_t=t.want.get("dim_t") or C.DIM_T,
                       node_noise=t.want.get("node_noise") or "shared")
    model = Model(m, **raw["diffusion_params"]["edm_params"])
    B = 4
    x_num = torch.randn(B, d_num)
    x_cat = torch.cat([torch.nn.functional.one_hot(torch.full((B,), k), k + 1).float() for k in cats], 1) \
        if cats else torch.zeros(B, 0)                       # every column in its [MASK] state
    if t.want.get("node_noise") == "per_column":
        return {"denoiser": type(m).__name__, "params": count_params(m), "forward": "skipped (per_column needs a diffusion)"}
    with torch.no_grad():
        D_x, logits = model(x_num, x_cat, torch.rand(B), torch.rand(B, d_num) + 0.1)
    assert D_x.shape == (B, d_num) and logits.shape == x_cat.shape and torch.isfinite(D_x).all()
    return {"denoiser": type(m).__name__, "params": count_params(m), "forward": "ok"}


def run_job(args, ex: Exec = None) -> int:
    repo = os.path.abspath(args.repo_root)
    td = C.verify_tabdiff(repo)
    ex = ex or Exec()
    print(f"  study {C.STUDY} -- {args.dataset} / {args.denoiser} / seed {args.seed}; "
          f"TabDiff {td['commit'][:7]} (digests verified)", flush=True)
    t = Target(args.dataset, args, repo)
    t.banner()
    if args.dry_run:
        state = {s: os.path.exists(marker(t.run, s)) for s in t.stages}
        have = trained_ok(t.run, t.want) if os.path.isdir(t.run) else False
        print(f"  [DRY RUN] markers {state}; finished model with this spec on disk: {have}")
        if not args.no_instantiate:
            r = instantiate_check(t)
            print(f"  [DRY RUN] built {r['denoiser']} ({r['params']:,} parameters) on the real shapes "
                  f"inside Model/Precond; forward: {r['forward']}")
            if args.check_all_rules and args.denoiser == "attngnn":
                for mode in C.TOPK_MODES:
                    r = instantiate_check(t, topk_mode=mode)
                    print(f"  [DRY RUN] rule {mode:17s}: built + forward {r['forward']}")
        return 0
    t.meta()
    only = set(args.stages.split(",")) if args.stages else None
    if only and not only <= set(t.stages):
        raise SystemExit(f"--stages {sorted(only - set(t.stages))} not stages of {t.name} {t.stages}")
    for stage in t.stages:
        if only is not None and all(os.path.exists(marker(t.run, s)) for s in only):
            break                                      # this job's share is done
        if os.path.exists(marker(t.run, stage)) and t.outputs_ok(stage):
            print(f"  [SKIP] {t.name}: {stage} done")
            continue
        if only is not None and stage not in only:
            # a chained job's earlier part must have finished this stage
            raise SystemExit(f"{t.name}: stage '{stage}' is not done and not in --stages "
                             f"{args.stages} -- submit the earlier job of this chain first")
        print(f"  [STAGE] {t.name}: {stage}  ({_now()})", flush=True)
        try:
            run_stage(t, stage, ex)
        except Paused as p:
            print(f"  [PAUSED] {p}", flush=True)
            return 0                                    # clean exit: afterok lets the next part run
    print(f"  [DONE] {t.name}  {_now()}", flush=True)
    return 0


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.ORDER)
    ap.add_argument("--denoiser", required=True, choices=C.DENOISERS)
    ap.add_argument("--seed", type=int, default=0, choices=C.TRAIN_SEEDS)
    ap.add_argument("--node_noise", choices=C.NODE_NOISE_CHOICES, default=C.NODE_NOISE)
    ap.add_argument("--ablation", choices=sorted(C.ABLATIONS), default=None,
                    help="NOT the main protocol: a one-change ablation (config.ABLATIONS), own run path")
    ap.add_argument("--topk_mode", choices=C.TOPK_MODES, default=C.TOPK_MODE,
                    help="neighbour-selection rule: fixed (main arm) | adaptive_jenks | adaptive_entmax15")
    ap.add_argument("--check_all_rules", action="store_true",
                    help="with --dry_run: also build + run the denoiser with every rule (imports entmax)")
    ap.add_argument("--geo_loss", action="store_true", help="reserved hook, default off (not implemented)")
    ap.add_argument("--geo_edge_bias", action="store_true", help="reserved hook, default off (not implemented)")
    ap.add_argument("--stages", default=None,
                    help="comma list: run only these stages (chained jobs); every earlier stage "
                         "must already be done")
    ap.add_argument("--exp_root", required=True, help="results go to <exp_root>/exp_tabdiff_gnn/...")
    ap.add_argument("--repo_root", default=C.REPO)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train_python", default=sys.executable, help="TabDiff env (the eval venv)")
    ap.add_argument("--eval_python", default=None)
    ap.add_argument("--synthcity_python", default=None)
    ap.add_argument("--tabpc_python", default=None)
    ap.add_argument("--n_runs", type=int, default=C.N_TABLES, help=argparse.SUPPRESS)
    ap.add_argument("--smoke_epochs", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--no_pin_check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--train_budget_h", type=float, default=None,
                    help="resumable trainer only (news): pause + save before this many hours of training")
    ap.add_argument("--dry_run", action="store_true",
                    help="plan + stage state + import TabDiff/AttnGNN and instantiate the denoiser")
    ap.add_argument("--no_instantiate", action="store_true", help=argparse.SUPPRESS)
    return ap


def main(argv=None) -> int:
    ap = make_parser()
    a = ap.parse_args(argv)
    if not a.dry_run:
        need = ["eval_python", "synthcity_python", "tabpc_python"]
        missing = [n for n in need if not getattr(a, n)]
        if missing:
            ap.error("required: " + ", ".join("--" + n for n in missing))
    return run_job(a)


if __name__ == "__main__":
    sys.exit(main())

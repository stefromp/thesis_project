#!/usr/bin/env python3
"""
Thin entry point: TabDiff's own train / sample pipeline with the selected
denoiser. Runs in TabDiff's environment (the eval venv: torch, sdmetrics,
category_encoders, icecream, tomli -- what tabdiff/main.py imports).

    python tabdiff_gnn/entry.py train  --target tabdiff_magic --denoiser attngnn --seed 0 \\
        --run_dir RUN --device cuda:0 [--node_noise shared]
    python tabdiff_gnn/entry.py sample --target tabdiff_magic --run_dir RUN --device cuda:0 \\
        [--n_tables 20]

It MIRRORS tabdiff/main.py (lines 36-283, train and `--mode test` branches):
the same classes, called in the same order with the same arguments --
TabDiffDataset + DataLoader(batch, shuffle, 4 workers), TabMetrics, Model
(EDM Precond) around the denoiser, the learnable per-column schedules
(main.py:212-214), UnifiedCtimeDiffusion, Trainer.run_loop / sample_synthetic,
config.pkl cached at train time and reloaded for sampling, best_ema_model_*
selected for sampling (main.py:66-69). It does not call main() because
main.py:185 hard-codes UniModMLP, writes checkpoints inside the submodule and
needs wandb. tests/test_tabdiff_gnn.py proves the transformer path builds the
model main.py builds (same raw_config, parameters, state_dict).

Differences (config.DEVIATIONS): routine evaluation every 2000 epochs off;
explicit training seed; table k drawn after seeding with k; no-op logger
(per-epoch losses go to train_log.csv instead of wandb).

Outputs in RUN: ckpt/{config.pkl, best_ema_model_<loss>_<epoch>.pt,
best_model_*}, route_a_spec.json, train_log.csv, train_record.json,
samples/samples_<k>.csv.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import pickle
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402

TABDIFF_ROOT = os.path.join(C.REPO, C.TABDIFF_REL)
CONFIG_TOML = os.path.join(TABDIFF_ROOT, "tabdiff", "configs", "tabdiff_configs.toml")
SPEC = "route_a_spec.json"
PAUSED_EXIT = 75          # resumable training paused on its time budget: the next chain part resumes
SPEC_KEYS = ("study", "target", "denoiser", "seed", "variant", "gnn", "dim_t", "node_noise",
             "geo_loss", "geo_edge_bias", "smoke_epochs", "trainer", "ablation")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tabdiff_imports():
    for p in (TABDIFF_ROOT, C.REPO):
        if p not in sys.path:
            sys.path.insert(0, p)
    import src  # TabDiff's src package
    from tabdiff.metrics import TabMetrics
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model
    from tabdiff.trainer import Trainer
    from utils_train import TabDiffDataset
    return src, TabMetrics, UnifiedCtimeDiffusion, Model, Trainer, TabDiffDataset


class CSVLogger:
    """Stands in for the wandb run: Trainer calls define_metric() and log()."""

    def __init__(self, path, append=False):
        self.path, self.rows, self._t = path, 0, time.time()
        self.fh, self.writer, self.append = None, None, append

    def define_metric(self, *a, **k):
        return None

    def log(self, d, *a, **k):
        flat = {k2: v for k2, v in d.items() if isinstance(v, (int, float))}
        if "epoch" not in flat:
            return
        flat["wall_s"] = round(time.time() - self._t, 2)
        if self.writer is None:
            if self.append and os.path.exists(self.path):          # resumed segment: same file
                with open(self.path) as fh:
                    fields = next(csv.reader(fh))
                self.fh = open(self.path, "a", newline="")
                self.writer = csv.DictWriter(self.fh, fieldnames=fields, extrasaction="ignore")
            else:
                self.fh = open(self.path, "w", newline="")
                self.writer = csv.DictWriter(self.fh, fieldnames=list(flat), extrasaction="ignore")
                self.writer.writeheader()
        self.writer.writerow(flat)
        self.rows += 1
        if self.rows % 50 == 0:
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_config():
    src = _tabdiff_imports()[0]
    return src.load_config(CONFIG_TOML)            # main.py:53-55


def spec_for(target: str, denoiser: str, seed: int, node_noise=C.NODE_NOISE,
             geo_loss=False, geo_edge_bias=False, ablation=None, topk_mode=C.TOPK_MODE) -> dict:
    """Everything that defines the model besides TabDiff's toml (route_a_spec.json)."""
    ds = C.parse_target(target)
    spec = {"study": C.STUDY, "target": target, "tabdiff_dataname": C.tabdiff_dataname(target),
            "denoiser": denoiser, "seed": seed, "geo_loss": geo_loss, "geo_edge_bias": geo_edge_bias,
            "trainer": C.TRAINER[ds], "ablation": ablation}
    if ablation is not None and denoiser != "attngnn":
        raise ValueError("ablations are defined for the attngnn denoiser only")
    if topk_mode != "fixed" and denoiser != "attngnn":
        raise ValueError("--topk_mode is an attngnn option")
    C.check_ablation(ablation, topk_mode)
    if denoiser == "attngnn":
        g, src = C.gnn_config(ds, ablation=ablation)
        rule = C.gnn_rule_fields(topk_mode)             # {} for fixed: spec unchanged
        if rule:
            src = {**src, "top_k": f"unused by {topk_mode} (topk_warmup_steps 0); source value "
                                   f"{g['top_k']} from {src['top_k']}"}
            spec["rule_settings"] = {**C.RULE_SETTINGS, **C.RULE_DETAILS[topk_mode], "topk_mode": topk_mode}
        spec.update({"variant": C.VARIANT, "gnn": {**g, **C.GNN_FIXED, **rule}, "gnn_sources": src,
                     "gnn_source_dataset": C.SOURCES[ds], "dim_t": C.DIM_T, "node_noise": node_noise,
                     "d_model_rule": f"{C.D_MODEL} on every dataset"})
    else:
        spec.update({"variant": None, "gnn": None, "dim_t": None, "node_noise": None})
    return spec


def build(spec: dict, raw_config: dict, device: str, mode: str, run_dir: str,
          ckpt_path: str | None = None, trainer_cls=None, trainer_kwargs=None, files=None,
          num_workers: int = 4, log_append: bool = False, k_logging: bool = True):
    """main.py:116-259 for one target; returns (trainer, diffusion, backbone, raw_config).
    trainer_cls: None = TabDiff's Trainer (every dataset but news); resumable.ResumableTrainer
    for news. files / num_workers: tests only (a data subset, no worker processes)."""
    from torch.utils.data import DataLoader
    from denoiser import build_denoiser
    _, TabMetrics, UnifiedCtimeDiffusion, Model, Trainer, TabDiffDataset = _tabdiff_imports()

    files = files or C.data_files(spec["target"])
    dataname = spec["tabdiff_dataname"]
    with open(files["info"]) as fh:
        info = json.load(fh)

    model_save_path = os.path.join(run_dir, "ckpt") if mode == "train" else None
    result_save_path = os.path.join(run_dir, "result")
    raw_config["model_save_path"] = model_save_path
    raw_config["result_save_path"] = result_save_path
    for p in (model_save_path, result_save_path):
        if p:
            os.makedirs(p, exist_ok=True)
    raw_config["deterministic"] = False
    if mode == "train":
        # deviation: routine sample+evaluate every check_val_every epochs off
        raw_config["train"]["main"]["check_val_every"] = raw_config["train"]["main"]["steps"] + 1

    batch_size = raw_config["train"]["main"]["batch_size"]
    train_data = TabDiffDataset(dataname, files["dir"], info, y_only=False, isTrain=True,
                                dequant_dist=raw_config["data"]["dequant_dist"],
                                int_dequant_factor=raw_config["data"]["int_dequant_factor"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    d_numerical, categories = train_data.d_numerical, train_data.categories
    val_data = TabDiffDataset(dataname, files["dir"], info, y_only=False, isTrain=False,
                              dequant_dist=raw_config["data"]["dequant_dist"],
                              int_dequant_factor=raw_config["data"]["int_dequant_factor"])
    # train: ["density"] as main.py:145; sample: no metric is computed here
    # (evaluation is its own stage), TabMetrics only provides info / real_data_size.
    metric_list = ["density"] if mode == "train" else []
    metrics = TabMetrics(files["real"], files["test"], files["val"], info, device, metric_list=metric_list)

    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (categories + 1).tolist()
    backbone = build_denoiser(spec["denoiser"], raw_config["unimodmlp_params"], spec.get("gnn"),
                              dim_t=spec.get("dim_t") or C.DIM_T,
                              node_noise=spec.get("node_noise") or "shared",
                              geo_loss=spec["geo_loss"], geo_edge_bias=spec["geo_edge_bias"])
    gnn = spec.get("gnn") or {}
    if k_logging and gnn.get("topk_mode", "fixed") != "fixed":
        # adaptive arms: per-node selected-k statistics vs t (observation only, topk_log.py)
        from topk_log import KStatsLogger
        cols = C.model_node_columns(C.parse_target(spec["target"]))
        KStatsLogger(len(cols), len(backbone.blocks), cols,
                     log_every=(spec.get("rule_settings") or {}).get("topk_log_every", 100),
                     eval_phase=None,
                     # sampling: every forward is "sample" (cmd_sample, like before, does not
                     # call diffusion.eval(); outputs are mode-independent, see topk_log.py)
                     force_phase=None if mode == "train" else "sample").attach(backbone)
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)

    raw_config["diffusion_params"]["scheduler"] = "power_mean_per_column"       # main.py:212-214
    raw_config["diffusion_params"]["cat_scheduler"] = "log_linear_per_column"
    diffusion = UnifiedCtimeDiffusion(num_classes=categories, num_numerical_features=d_numerical,
                                      denoise_fn=model, y_only_model=None,
                                      **raw_config["diffusion_params"], device=device)
    if spec.get("node_noise") == "per_column":
        backbone.attach_diffusion(diffusion)
    diffusion.to(device)
    diffusion.train()

    logger = (CSVLogger(os.path.join(run_dir, "train_log.csv"), append=log_append) if mode == "train"
              else CSVLogger(os.devnull))
    trainer = (trainer_cls or Trainer)(diffusion, train_loader, train_data, val_data, metrics, logger,
                      **raw_config["train"]["main"],
                      sample_batch_size=raw_config["sample"]["batch_size"],
                      num_samples_to_generate=None,
                      model_save_path=raw_config["model_save_path"],
                      result_save_path=raw_config["result_save_path"],
                      device=device, ckpt_path=ckpt_path, y_only=False, **(trainer_kwargs or {}))
    return trainer, diffusion, backbone, raw_config


def find_ckpt(run_dir: str, allow_smoke: bool = False) -> str:
    """main.py:66-69: the best EMA checkpoint (TabDiff keeps exactly one)."""
    found = glob.glob(os.path.join(run_dir, "ckpt", "best_ema_model*"))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError(f"{len(found)} best_ema_model files in {run_dir}/ckpt")
    if allow_smoke:
        smoke = glob.glob(os.path.join(run_dir, "ckpt", "smoke_final_ema_model_*"))
        if smoke:
            return smoke[0]
    raise FileNotFoundError(f"no best_ema_model* in {run_dir}/ckpt (TabDiff writes it only after epoch 4000)")


def param_counts(diffusion, backbone) -> dict:
    from denoiser import param_breakdown, count_params
    return {"denoiser": param_breakdown(backbone),
            "denoiser_total": count_params(backbone),
            "schedules": count_params(diffusion.num_schedule) + count_params(diffusion.cat_schedule),
            "diffusion_total_as_tabdiff_prints": count_params(diffusion)}


def cmd_train(a) -> int:
    import torch
    job_start = time.time()
    C.verify_tabdiff()
    spec = spec_for(a.target, a.denoiser, a.seed, node_noise=a.node_noise, geo_loss=a.geo_loss,
                    geo_edge_bias=a.geo_edge_bias, ablation=a.ablation, topk_mode=a.topk_mode)
    spec["smoke_epochs"] = a.smoke_epochs
    os.makedirs(a.run_dir, exist_ok=True)
    resumable = spec["trainer"] == "resumable"
    trainer_cls, trainer_kwargs, resuming = None, None, False
    if resumable:
        import resumable as RS
        resume_path = os.path.join(a.run_dir, "ckpt", RS.RESUME_FILE)
        resuming = os.path.exists(resume_path)
        trainer_cls = RS.ResumableTrainer
        trainer_kwargs = {"resume_path": resume_path, "save_every": a.save_every,
                          "budget_deadline": job_start + a.train_budget_h * 3600 if a.train_budget_h else None,
                          "stop_after_epoch": a.stop_after_epoch}
    if resuming:
        have = json.load(open(os.path.join(a.run_dir, SPEC)))
        diff = {k: (have.get(k), spec.get(k)) for k in SPEC_KEYS if have.get(k) != spec.get(k)}
        if diff:
            raise RuntimeError(f"refusing to resume {a.run_dir}: run state has another spec {diff}")
        with open(os.path.join(a.run_dir, "ckpt", "config.pkl"), "rb") as fh:   # the run's own config
            raw_config = pickle.load(fh)
        spec = have
        epochs = raw_config["train"]["main"]["steps"]
    else:
        raw_config = load_raw_config()
        epochs = raw_config["train"]["main"]["steps"]
        if a.smoke_epochs is not None:
            epochs = raw_config["train"]["main"]["steps"] = int(a.smoke_epochs)
    smoke = a.smoke_epochs is not None
    seed_everything(a.seed)                                   # deviation: explicit seed
    trainer, diffusion, backbone, raw_config = build(spec, raw_config, a.device, "train", a.run_dir,
                                                     trainer_cls=trainer_cls, trainer_kwargs=trainer_kwargs,
                                                     log_append=resuming)
    counts = param_counts(diffusion, backbone)
    if not resuming:
        spec.update({"param_counts": counts,
                     "raw_config_train": raw_config["train"]["main"],
                     "diffusion_params": raw_config["diffusion_params"],
                     "unimodmlp_params": raw_config["unimodmlp_params"]})
        with open(os.path.join(a.run_dir, SPEC), "w") as fh:
            json.dump(spec, fh, indent=2, default=str)
        with open(os.path.join(raw_config["model_save_path"], "config.pkl"), "wb") as fh:  # main.py:280-282
            pickle.dump(raw_config, fh)
    print(f"  [route A] {spec['target']} denoiser={spec['denoiser']} trainer={spec['trainer']} "
          f"params={counts['denoiser_total']:,} epochs={epochs} batch={raw_config['train']['main']['batch_size']} "
          f"device={a.device}" + (f" -- RESUMING at epoch {trainer.resumed_from_epoch}" if resuming else ""),
          flush=True)
    rec_p = os.path.join(a.run_dir, "train_record.json")
    prev = json.load(open(rec_p)) if (resuming and os.path.exists(rec_p)) else {}
    t0, w0 = _now(), time.time()
    trainer.run_loop()
    trainer.logger.close()
    if getattr(backbone, "k_logger", None) is not None:
        backbone.k_logger.write(a.run_dir)               # merged: safe across resumed segments
    segment = {"start_epoch": getattr(trainer, "resumed_from_epoch", 0), "end_epoch": trainer.curr_epoch,
               "started_utc": t0, "finished_utc": _now(), "wall_s": round(time.time() - w0, 1),
               "paused": bool(getattr(trainer, "paused_at", None)), "n_state_saves": getattr(trainer, "n_saves", 0),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none")}
    segments = prev.get("segments", []) + [segment]
    rec = {"train_started_utc": prev.get("train_started_utc", t0), "train_finished_utc": _now(),
           "train_wall_s": round(sum(x["wall_s"] for x in segments), 1),
           "epochs_configured": epochs, "epochs_completed": trainer.curr_epoch,
           "steps_per_epoch": len(trainer.train_iter), "smoke": smoke, "trainer": spec["trainer"],
           "segments": segments}
    if resumable and trainer.paused_at:
        rec.update({"completed": False, "paused": True})
        with open(rec_p, "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  [PAUSED] epoch {trainer.paused_at}/{epochs}; the next job resumes from "
              f"{trainer_kwargs['resume_path']}")
        return PAUSED_EXIT
    completed = trainer.curr_epoch >= epochs
    rec.update({"completed": completed, "paused": False})
    if smoke and not glob.glob(os.path.join(a.run_dir, "ckpt", "best_ema_model*")):
        # TabDiff saves best EMA only after epoch 4000; a smoke run stores its final EMA instead
        torch.save({"denoise_fn": trainer.ema_model.state_dict(),
                    "num_schedule": trainer.ema_num_schedule.state_dict(),
                    "cat_schedule": trainer.ema_cat_schedule.state_dict()},
                   os.path.join(a.run_dir, "ckpt", f"smoke_final_ema_model_{epochs}.pt"))
    rec["checkpoint"] = os.path.basename(find_ckpt(a.run_dir, allow_smoke=smoke))
    if completed and resumable and os.path.exists(trainer_kwargs["resume_path"]):
        os.remove(trainer_kwargs["resume_path"])          # finished: nothing left to resume
    with open(rec_p, "w") as fh:
        json.dump(rec, fh, indent=2)
    if not completed:
        print(f"  [FAIL] training stopped at epoch {trainer.curr_epoch}/{epochs} (NaN loss, see train_log.csv)")
        return 3
    return 0


def cmd_sample(a) -> int:
    import torch
    C.verify_tabdiff()
    with open(os.path.join(a.run_dir, SPEC)) as fh:
        spec = json.load(fh)
    if spec["target"] != a.target:
        raise ValueError(f"{a.run_dir} holds {spec['target']}, not {a.target}")
    ckpt = find_ckpt(a.run_dir, allow_smoke=bool(spec.get("smoke_epochs")))
    with open(os.path.join(os.path.dirname(ckpt), "config.pkl"), "rb") as fh:     # main.py:70-75
        raw_config = pickle.load(fh)
    trainer, diffusion, backbone, raw_config = build(spec, raw_config, a.device, "test", a.run_dir,
                                                     ckpt_path=ckpt)
    out_dir = os.path.join(a.run_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    n = trainer.metrics.real_data_size
    times = {}
    for k in range(a.n_tables):
        path = os.path.join(out_dir, f"samples_{k}.csv")
        if os.path.exists(path) and not a.overwrite:
            print(f"  [KEEP] samples_{k}.csv")
            continue
        seed_everything(k)                                     # deviation: generation seed k
        w0 = time.time()
        syn_df = trainer.sample_synthetic(n, ema=False)        # as Trainer.evaluate_generation
        syn_df.to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        times[k] = round(time.time() - w0, 1)
        print(f"  [OK] samples_{k}.csv ({len(syn_df)} rows, {times[k]} s)", flush=True)
        del syn_df
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if getattr(backbone, "k_logger", None) is not None:
        backbone.k_logger.write(a.run_dir)
    with open(os.path.join(a.run_dir, "sample_record.json"), "w") as fh:
        json.dump({"checkpoint": os.path.basename(ckpt), "rows_per_table": n, "n_tables": a.n_tables,
                   "generation_seeds": list(range(a.n_tables)), "seconds_per_table": times,
                   "num_timesteps": raw_config["diffusion_params"]["num_timesteps"],
                   "finished_utc": _now()}, fh, indent=2)
    return 0


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--target", required=True)
    t.add_argument("--denoiser", required=True, choices=C.DENOISERS)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--run_dir", required=True)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--node_noise", choices=C.NODE_NOISE_CHOICES, default=C.NODE_NOISE)
    t.add_argument("--topk_mode", choices=C.TOPK_MODES, default=C.TOPK_MODE,
                   help="neighbour-selection rule: fixed (main arm) | adaptive_jenks | adaptive_entmax15")
    t.add_argument("--ablation", choices=sorted(C.ABLATIONS), default=None,
                   help="NOT the main protocol: one-change ablation (config.ABLATIONS)")
    t.add_argument("--geo_loss", action="store_true", help="reserved hook (not implemented)")
    t.add_argument("--geo_edge_bias", action="store_true", help="reserved hook (not implemented)")
    t.add_argument("--smoke_epochs", type=int, default=None, help=argparse.SUPPRESS)
    t.add_argument("--train_budget_h", type=float, default=None,
                   help="resumable trainer (news): save the run state and pause before this many hours")
    t.add_argument("--save_every", type=int, default=100, help="resumable trainer: save the run state every N epochs")
    t.add_argument("--stop_after_epoch", type=int, default=None, help=argparse.SUPPRESS)
    s = sub.add_parser("sample")
    s.add_argument("--target", required=True)
    s.add_argument("--run_dir", required=True)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--n_tables", type=int, default=C.N_TABLES)
    s.add_argument("--overwrite", action="store_true")
    return ap


def main(argv=None) -> int:
    a = make_parser().parse_args(argv)
    C.parse_target(a.target)
    if sys.platform == "darwin":
        # local runs only: macOS 'spawn' cannot pickle TabDiffDataset's lambdas for
        # the 4 DataLoader workers; Linux (the cluster) forks by default, as for TabDiff.
        import multiprocessing as mp
        mp.set_start_method("fork", force=True)
    return cmd_train(a) if a.cmd == "train" else cmd_sample(a)


if __name__ == "__main__":
    sys.exit(main())

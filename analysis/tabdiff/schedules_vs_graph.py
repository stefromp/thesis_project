#!/usr/bin/env python3
"""
Do TabDiff's learned per-column noise schedules interact with our graph?
Analysis of the Route-A runs (AttnGNN inside TabDiff); nothing is trained.

    python analysis/tabdiff/schedules_vs_graph.py all --exp_root $TRANSFER_ROOT --device cuda:0
    python analysis/tabdiff/schedules_vs_graph.py extract | sample | report   (one stage)

Runs read: <exp_root>/exp_tabdiff_gnn/tabdiff_<ds>/attngnn/<arm dir>/seed0, the arm dir
resolved by tabdiff_gnn/config.run_dir for the arms
    fixed  adaptive_jenks  adaptive_entmax15  topk_dense (ablation; beijing, magic)
on adult, beijing, default, diabetes, magic, shoppers. A missing run is reported, not
an error.

Stage 1 -- extract (checkpoints only; torch, no TabDiff import)
  From the checkpoint each run sampled with (sample_record.json "checkpoint" = the
  best EMA checkpoint after epoch 4000, cross-checked with train_record.json and the
  one best_ema_model_* on disk), the learned schedule parameters, in TabDiff's own
  parameterisation (SCHEDULE_CODE below, line numbers checked by the tests):
    numerical column i   rho_i = softplus(rho_raw_i) + rho_offset
                         sigma_i(t) = (s_min^(1/rho_i) + t (s_max^(1/rho_i) - s_min^(1/rho_i)))^rho_i
    categorical column j k_j = softplus(k_raw_j) + k_offset
                         mask_j(t) = 1 - exp(-total_noise_j(t)) = (1 - e_max - e_min) t^k_j + e_min
  with the run's own config.pkl values (TabDiff's toml: s_min 0.002, s_max 80,
  rho_init 7 -> rho 12.0009, rho_offset 5, k_init -6 -> k 1.00248, k_offset 1,
  e_max 1e-3, e_min 1e-5, sigma_data 1). Reveal time t*: numerical sigma_i(t*) =
  sigma_data, categorical mask_j(t*) = 0.5, both closed form. Sampling runs t from 1
  to 0, so a larger t* = informative earlier.
  -> schedules.csv, curves.csv.gz

Stage 2 -- sample (GPU; fixed, adaptive_jenks, adaptive_entmax15)
  The run's model is rebuilt from its own route_a_spec.json + ckpt/config.pkl
  (build_denoiser -> TabDiff Model/Precond -> UnifiedCtimeDiffusion with the
  per-column schedules, as tabdiff_gnn/entry.build) and the sampled checkpoint is
  loaded as TabDiff's Trainer loads it (trainer.py:77-81). TabDiff's sampler
  (UnifiedCtimeDiffusion.sample_all, 50 steps, stochastic + 2nd-order correction from
  the config, module mode as entry.cmd_sample) draws --n_rows rows after
  entry.seed_everything(--seed). EdgeLogger records, for every denoiser call
  (t recovered from c_noise) and block, the N x N count of rows in which node i keeps
  node j (A[b, i, j] > 0, i != j, DynamicAdjacency's output; row i aggregates from j).
  Observation only: forward hooks, no_grad, no randomness, outputs untouched.
  Checks per run (sampling_checks.csv): logging off / on / off again (bit-identity),
  and -- with --verify_entry -- the same draw through tabdiff_gnn/entry.build +
  TabDiff's Trainer (the production sampling path).
  -> raw/edge_counts_<ds>_<arm>.csv.gz, raw/sample_meta_<ds>_<arm>.json

Stage 3 -- report (numpy / pandas / scipy / matplotlib)
  Spearman of t* between arms, movers fixed -> entmax15, edge frequencies per block
  and t bin, and the interaction test: per (call, node i), the share of i's kept
  neighbours j that are already clean (t*_j > t) against the share expected if
  neighbours were drawn at random among the other N - 1 columns; aggregated over the
  "informative" (call, i) pairs (some but not all other columns clean); two-sided
  permutation test over column labels (t* permuted across columns).
  -> SCHEDULES.md, *.csv, figures/, provenance.csv

Output: analysis/tabdiff/schedules_vs_graph/ (--out). Provenance as analysis/summary:
every value row carries source, source_commit (git commit, or "untracked
sha256/16=..." for run outputs) and producer_commit (the run's run_meta git_commit);
provenance.csv lists every file read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TG = os.path.join(REPO, "tabdiff_gnn")
TABDIFF_ROOT = os.path.join(REPO, "third_party", "TabDiff")
for _p in (TG,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config as C  # noqa: E402  (tabdiff_gnn/config.py: run layout, node order)

OUT_DEFAULT = os.path.join(HERE, "schedules_vs_graph")
EXP_ROOT_DEFAULT = os.path.join(REPO, "transfer_runs")
DATASETS = ["adult", "beijing", "default", "diabetes", "magic", "shoppers"]
FOCUS = ["beijing", "magic"]                     # reported first where an order matters
ARMS = {  # arm -> (topk_mode, ablation)
    "fixed": ("fixed", None),
    "adaptive_jenks": ("adaptive_jenks", None),
    "adaptive_entmax15": ("adaptive_entmax15", None),
    "topk_dense": ("fixed", "topk_dense"),
}
SHORT = {"fixed": "fixed", "adaptive_jenks": "jenks", "adaptive_entmax15": "entmax15",
         "topk_dense": "dense", "init": "init"}
SAMPLE_ARMS = ["fixed", "adaptive_jenks", "adaptive_entmax15"]
SEED = 0                                         # training seed of the runs analysed
T_GRID = np.linspace(0.0, 1.0, 101)
N_TBINS = 10
LOW_NOISE, HIGH_NOISE = (0.0, 0.2), (0.8, 1.0)   # heatmap t ranges
MASK_LEVEL = 0.5

# Where TabDiff defines what is extracted (third_party/TabDiff @ 5ecdb33; the tests
# check that each quoted line holds the quoted code).
SCHEDULE_CODE = [
    ("tabdiff/models/noise_schedule.py", 80,
     "self.rho_raw = nn.Parameter(torch.tensor([rho_init] * self.num_numerical, dtype=torch.float32))",
     "numerical: raw parameter rho_raw, initialised to rho_init"),
    ("tabdiff/models/noise_schedule.py", 84, "return nn.functional.softplus(self.rho_raw) + self.rho_offset",
     "rho_i = softplus(rho_raw_i) + rho_offset"),
    ("tabdiff/models/noise_schedule.py", 99,
     "sigma = (sigma_min_pow + t * (sigma_max_pow - sigma_min_pow)).pow(rho)",
     "sigma_i(t) = (s_min^(1/rho_i) + t (s_max^(1/rho_i) - s_min^(1/rho_i)))^rho_i"),
    ("tabdiff/models/noise_schedule.py", 133,
     "self.k_raw = nn.Parameter(torch.tensor([k_init] * self.num_categories, dtype=torch.float32))",
     "categorical: raw parameter k_raw, initialised to k_init"),
    ("tabdiff/models/noise_schedule.py", 136, "return torch.nn.functional.softplus(self.k_raw) + self.k_offset",
     "k_j = softplus(k_raw_j) + k_offset"),
    ("tabdiff/models/noise_schedule.py", 155,
     "total_noise = -torch.log1p(-((1 - self.eps_max - self.eps_min) * t.pow(k) + self.eps_min))",
     "total_noise_j(t) = -log(1 - ((1 - e_max - e_min) t^k_j + e_min))"),
    ("tabdiff/models/unified_ctime_diffusion.py", 121, "move_chance = -torch.expm1(-sigma_cat)",
     "mask probability = 1 - exp(-total_noise_j(t)) = (1 - e_max - e_min) t^k_j + e_min"),
    ("tabdiff/main.py", 213, "raw_config['diffusion_params']['scheduler'] = 'power_mean_per_column'",
     "TabDiff's main selects the learnable per-column schedules (Route A: entry.build, same)"),
    ("tabdiff/trainer.py", 260, "'denoise_fn': self.ema_model.state_dict(),",
     "best EMA checkpoint holds the EMA denoiser ..."),
    ("tabdiff/trainer.py", 261, "'num_schedule':self.ema_num_schedule.state_dict(),",
     "... and the EMA schedules (what sampling uses)"),
]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===========================================================================
# Provenance (analysis/summary/build_summary.py conventions)
# ===========================================================================
_PROV: dict = {}


def rel(p) -> str:
    a = os.path.realpath(p)
    return os.path.relpath(a, REPO) if a.startswith(REPO + os.sep) else a


def _git(*args) -> str:
    r = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def file_info(p) -> dict:
    r = rel(p)
    if r not in _PROV:
        tracked = (not os.path.isabs(r)) and subprocess.run(
            ["git", "-C", REPO, "ls-files", "--error-unmatch", r], capture_output=True).returncode == 0
        commit = _git("log", "-1", "--format=%h", "--", r) if tracked else ""
        if tracked and _git("status", "--porcelain", "--", r):
            commit += " (modified in working tree)"
        _PROV[r] = {"path": r, "git_commit": commit if tracked else "untracked", "sha256_16": sha16(p)}
    return _PROV[r]


def src(paths, producer: str = "") -> dict:
    paths = [paths] if isinstance(paths, str) else list(paths)
    infos = [file_info(p) for p in paths]
    commits = []
    for i in infos:
        c = i["git_commit"] if i["git_commit"] != "untracked" else f"untracked sha256/16={i['sha256_16']}"
        if c not in commits:
            commits.append(c)
    return {"source": "; ".join(i["path"] for i in infos), "source_commit": "; ".join(commits),
            "producer_commit": producer}


def analysis_commit() -> str:
    """tabdiff_gnn/runner.git_commit: HEAD[-dirty], else tabdiff_gnn/UPLOAD_COMMIT (cluster).
    Loaded under its own name: hpc_transfer/tabdiff_study has a runner.py too."""
    import importlib.util
    key = "route_a_runner"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, os.path.join(TG, "runner.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key].git_commit(REPO)


def _json(p):
    with open(p) as fh:
        return json.load(fh)


# ===========================================================================
# Part 1 -- TabDiff's schedule functions in numpy (float64)
# ===========================================================================
def softplus(x):
    """torch.nn.functional.softplus (beta 1, threshold 20)."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def rho_from_raw(rho_raw, rho_offset):
    return softplus(rho_raw) + rho_offset


def k_from_raw(k_raw, k_offset):
    return softplus(k_raw) + k_offset


def sigma_curve(t, rho, s_min, s_max):
    """(len(t), n_num): PowerMeanNoise_PerColumn.total_noise."""
    t = np.asarray(t, dtype=np.float64)[:, None]
    rho = np.asarray(rho, dtype=np.float64)[None, :]
    a, b = s_min ** (1 / rho), s_max ** (1 / rho)
    return (a + t * (b - a)) ** rho


def mask_curve(t, k, eps_max, eps_min):
    """(len(t), n_cat): 1 - exp(-LogLinearNoise_PerColumn.total_noise) = move chance."""
    t = np.asarray(t, dtype=np.float64)[:, None]
    k = np.asarray(k, dtype=np.float64)[None, :]
    return (1 - eps_max - eps_min) * t ** k + eps_min


def tstar_num(rho, s_min, s_max, sigma_data):
    """t with sigma_i(t) = sigma_data (= PowerMeanNoise_PerColumn.inverse_to_t(sigma_data))."""
    rho = np.asarray(rho, dtype=np.float64)
    a, b = s_min ** (1 / rho), s_max ** (1 / rho)
    return (sigma_data ** (1 / rho) - a) / (b - a)


def tstar_cat(k, eps_max, eps_min, level=MASK_LEVEL):
    """t with mask_j(t) = level."""
    k = np.asarray(k, dtype=np.float64)
    return ((level - eps_min) / (1 - eps_max - eps_min)) ** (1 / k)


def schedule_params(raw_config: dict) -> dict:
    nsp = raw_config["diffusion_params"]["noise_schedule_params"]
    return {"sigma_min": float(nsp["sigma_min"]), "sigma_max": float(nsp["sigma_max"]),
            "rho_init": float(nsp["rho_init"]), "rho_offset": float(nsp["rho_offset"]),
            "k_init": float(nsp["k_init"]), "k_offset": float(nsp["k_offset"]),
            "eps_max": float(nsp["eps_max"]), "eps_min": float(nsp["eps_min"]),
            "sigma_data": float(raw_config["diffusion_params"]["edm_params"]["sigma_data"]),
            "scheduler": raw_config["diffusion_params"].get("scheduler"),
            "cat_scheduler": raw_config["diffusion_params"].get("cat_scheduler")}


def column_schedules(rho_raw, k_raw, p: dict) -> dict:
    """Everything Part 1 reports for one run, from the raw parameters."""
    rho_raw, k_raw = np.asarray(rho_raw, np.float64), np.asarray(k_raw, np.float64)
    rho, k = rho_from_raw(rho_raw, p["rho_offset"]), k_from_raw(k_raw, p["k_offset"])
    rho0 = rho_from_raw(np.full_like(rho_raw, p["rho_init"]), p["rho_offset"])
    k0 = k_from_raw(np.full_like(k_raw, p["k_init"]), p["k_offset"])
    return {"rho_raw": rho_raw, "k_raw": k_raw, "rho": rho, "k": k, "rho0": rho0, "k0": k0,
            "tstar_num": tstar_num(rho, p["sigma_min"], p["sigma_max"], p["sigma_data"]),
            "tstar_cat": tstar_cat(k, p["eps_max"], p["eps_min"]),
            "tstar_num0": tstar_num(rho0, p["sigma_min"], p["sigma_max"], p["sigma_data"]),
            "tstar_cat0": tstar_cat(k0, p["eps_max"], p["eps_min"])}


# ===========================================================================
# Runs
# ===========================================================================
def arms_for(ds: str) -> list:
    return [a for a in ARMS if ARMS[a][1] is None or ds in C.ABLATIONS[ARMS[a][1]]["datasets"]]


def resolve_run(ds: str, arm: str, exp_root: str) -> dict:
    """{'status': 'ok' | reason, 'run', 'ckpt', ...} for one dataset x arm (seed 0)."""
    mode, abl = ARMS[arm]
    run = C.run_dir(exp_root, C.target_name(ds), "attngnn", SEED, topk_mode=mode, ablation=abl)
    out = {"dataset": ds, "arm": arm, "run": run, "ckpt": None}
    if not os.path.isdir(run):
        return {**out, "status": "no run on disk"}
    sr_p, tr_p = os.path.join(run, "sample_record.json"), os.path.join(run, "train_record.json")
    if not os.path.exists(sr_p):
        return {**out, "status": "not sampled (no sample_record.json)"}
    sr, tr = _json(sr_p), (_json(tr_p) if os.path.exists(tr_p) else {})
    found = glob.glob(os.path.join(run, "ckpt", "best_ema_model*"))
    ck = os.path.join(run, "ckpt", sr["checkpoint"])
    if not os.path.exists(ck):
        return {**out, "status": f"sampled checkpoint {sr['checkpoint']} not on disk"}
    if len(found) != 1 or os.path.basename(found[0]) != sr["checkpoint"] or tr.get("checkpoint") != sr["checkpoint"]:
        return {**out, "status": f"checkpoint mismatch: sample_record {sr['checkpoint']}, train_record "
                                 f"{tr.get('checkpoint')}, on disk {[os.path.basename(f) for f in found]}"}
    meta_p = os.path.join(run, "run_meta.json")
    meta = _json(meta_p) if os.path.exists(meta_p) else {}
    return {**out, "status": "ok", "ckpt": ck, "config": os.path.join(run, "ckpt", "config.pkl"),
            "spec": os.path.join(run, "route_a_spec.json"), "run_meta": meta_p if meta else None,
            "producer_commit": meta.get("git_commit", ""), "checkpoint": sr["checkpoint"],
            "epoch": int(sr["checkpoint"].rsplit("_", 1)[-1].split(".")[0])}


def load_ckpt(path: str, device="cpu"):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:                                    # torch < 1.13
        return torch.load(path, map_location=device)


def load_config(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def node_layout(ds: str, raw_config: dict) -> tuple:
    cols = C.model_node_columns(ds)
    d_num = int(raw_config["unimodmlp_params"]["d_numerical"])
    n_cat = len(raw_config["unimodmlp_params"]["categories"])
    if d_num + n_cat != len(cols):
        raise RuntimeError(f"{ds}: {d_num} + {n_cat} model columns != {len(cols)} node names")
    return cols, d_num, n_cat


# ===========================================================================
# Reference arm: TabDiff's own trained checkpoints, if any are on disk
# ===========================================================================
CKPT_EXT = (".pt", ".pth", ".ckpt")
TABDIFF_CKPT_PREFIXES = ("best_ema_model_", "best_model_", "model_", "ema_model_")
SKIP_DIRS = {".git", "transfer_runs", "tddpm_setup_runs", "__pycache__", "node_modules", "wandb"}


def reference_search(exp_root: str) -> dict:
    """Every checkpoint file in the repo (incl. third_party/) outside the run trees;
    the ones named like TabDiff's saves (trainer.py:237/264/273/281) are opened and
    counted as TabDiff checkpoints when they hold TabDiff's {denoise_fn, num_schedule,
    cat_schedule} with a UniModMLP denoiser. Nothing is downloaded."""
    skip_abs = {os.path.realpath(os.path.join(exp_root, C.RESULTS_DIRNAME))}
    found, tabdiff = [], []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and os.path.realpath(os.path.join(root, d)) not in skip_abs]
        for f in files:
            if not f.endswith(CKPT_EXT):
                continue
            p = os.path.join(root, f)
            found.append(rel(p))
            if not f.startswith(TABDIFF_CKPT_PREFIXES):
                continue
            try:
                sd = load_ckpt(p)
            except Exception:
                continue
            if isinstance(sd, dict) and {"denoise_fn", "num_schedule", "cat_schedule"} <= set(sd) and \
                    not any(".blocks." in k for k in sd["denoise_fn"]):         # not an AttnGNN (Route A)
                tabdiff.append(rel(p))
    return {"searched": "repository incl. third_party/, excluding " + ", ".join(sorted(SKIP_DIRS))
                        + " and the Route-A run tree", "checkpoint_files": sorted(found),
            "tabdiff_checkpoints": sorted(tabdiff), "searched_utc": _now()}


# ===========================================================================
# Stage 1 -- extract
# ===========================================================================
def extract(exp_root: str, out: str, datasets=DATASETS) -> list:
    import pandas as pd
    rows, curves, status, init_done = [], [], [], set()
    for ds in datasets:
        for arm in arms_for(ds):
            r = resolve_run(ds, arm, exp_root)
            status.append({k: r.get(k) for k in ("dataset", "arm", "status", "checkpoint", "epoch",
                                                  "producer_commit")} | {"run": rel(r["run"])})
            if r["status"] != "ok":
                print(f"  [skip] {ds}/{arm}: {r['status']}")
                continue
            raw = load_config(r["config"])
            sd = load_ckpt(r["ckpt"])
            p = schedule_params(raw)
            if (p["scheduler"], p["cat_scheduler"]) != ("power_mean_per_column", "log_linear_per_column"):
                raise RuntimeError(f"{ds}/{arm}: schedules {p['scheduler']}/{p['cat_scheduler']} are not learnable")
            cols, d_num, n_cat = node_layout(ds, raw)
            rho_raw = sd["num_schedule"]["rho_raw"].double().cpu().numpy()
            k_raw = sd["cat_schedule"]["k_raw"].double().cpu().numpy()
            if (len(rho_raw), len(k_raw)) != (d_num, n_cat):
                raise RuntimeError(f"{ds}/{arm}: {len(rho_raw)} rho / {len(k_raw)} k vs {d_num} / {n_cat} columns")
            s = column_schedules(rho_raw, k_raw, p)
            prov = src([r["ckpt"], r["config"]] + ([r["run_meta"]] if r["run_meta"] else []), r["producer_commit"])
            tstar = np.concatenate([s["tstar_num"], s["tstar_cat"]])
            order = sorted(range(len(cols)), key=lambda i: (-tstar[i], i))
            rank = {i: n + 1 for n, i in enumerate(order)}
            for i, col in enumerate(cols):
                num = i < d_num
                j = i if num else i - d_num
                rows.append({"dataset": ds, "arm": arm, "node": i, "column": col, "type": "num" if num else "cat",
                             "param": "rho" if num else "k",
                             "raw": float(s["rho_raw" if num else "k_raw"][j]),
                             "value": float(s["rho" if num else "k"][j]),
                             "raw_init": p["rho_init"] if num else p["k_init"],
                             "value_init": float(s["rho0" if num else "k0"][j]),
                             "t_star": float(s["tstar_num" if num else "tstar_cat"][j]),
                             "t_star_init": float(s["tstar_num0" if num else "tstar_cat0"][j]),
                             "reveal_rank": rank[i], "n_nodes": len(cols), "checkpoint": r["checkpoint"],
                             "epoch": r["epoch"], **p, **prov})
            sig = sigma_curve(T_GRID, s["rho"], p["sigma_min"], p["sigma_max"])
            msk = mask_curve(T_GRID, s["k"], p["eps_max"], p["eps_min"])
            sig0 = sigma_curve(T_GRID, s["rho0"][:1], p["sigma_min"], p["sigma_max"])[:, 0]
            msk0 = mask_curve(T_GRID, s["k0"][:1], p["eps_max"], p["eps_min"])[:, 0]
            for i, col in enumerate(cols):
                num = i < d_num
                vals = sig[:, i] if num else msk[:, i - d_num]
                for t, v in zip(T_GRID, vals):
                    curves.append((ds, arm, i, col, "num" if num else "cat", "sigma" if num else "mask", round(t, 4), v))
            if ds not in init_done:                      # TabDiff's initial schedule, once per dataset
                init_done.add(ds)
                for t, a, b in zip(T_GRID, sig0, msk0):
                    curves.append((ds, "init", -1, "(init)", "num", "sigma", round(t, 4), a))
                    curves.append((ds, "init", -1, "(init)", "cat", "mask", round(t, 4), b))
            print(f"  [ok] {ds}/{arm}: {d_num} rho in [{s['rho'].min():.3f}, {s['rho'].max():.3f}], "
                  f"{n_cat} k in [{s['k'].min():.3f}, {s['k'].max():.3f}] ({r['checkpoint']})")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "schedules.csv"), index=False)
    cv = pd.DataFrame(curves, columns=["dataset", "arm", "node", "column", "type", "quantity", "t", "value"])
    cv.to_csv(os.path.join(out, "curves.csv.gz"), index=False, float_format="%.6g")
    pd.DataFrame(status).to_csv(os.path.join(out, "runs.csv"), index=False)
    with open(os.path.join(out, "reference_search.json"), "w") as fh:
        json.dump(reference_search(exp_root), fh, indent=2)
    _write_prov(out, merge=True)
    return rows


def _write_prov(out: str, merge: bool) -> None:
    import pandas as pd
    p = os.path.join(out, "provenance.csv")
    have = pd.read_csv(p) if (merge and os.path.exists(p)) else pd.DataFrame(columns=["path", "git_commit", "sha256_16"])
    new = pd.DataFrame(list(_PROV.values()), columns=["path", "git_commit", "sha256_16"])
    allp = pd.concat([have, new]).drop_duplicates("path", keep="last").sort_values("path")
    allp.to_csv(p, index=False)


# ===========================================================================
# Stage 2 -- sample with edge logging
# ===========================================================================
class EdgeLogger:
    """Per denoiser call and block: counts[i, j] = rows in which node i keeps node j.

    A forward pre-hook on the AttnGNN denoiser reads its `timesteps` argument
    (c_noise; t recovered with denoiser.t_from_c_noise), forward hooks on each block's
    DynamicAdjacency read its output A (B, N, N). Both run under no_grad, use no
    randomness, return None and never write to a model tensor, so the computation is
    unchanged (tests/test_schedules_vs_graph.py)."""

    def __init__(self):
        self.records = []              # (call, t, block, counts (N, N) int64, rows)
        self.calls = []                # (call, t, t_spread)
        self._handles, self._call, self._t = [], -1, None

    def attach(self, backbone) -> "EdgeLogger":
        self._handles.append(backbone.register_forward_pre_hook(self._pre))
        for b, block in enumerate(backbone.blocks):
            if block.dynamic_adj is None:
                raise RuntimeError("EdgeLogger needs the dynamic adjacency")
            self._handles.append(block.dynamic_adj.register_forward_hook(self._hook(b)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _pre(self, module, args):
        import torch
        from denoiser import t_from_c_noise
        with torch.no_grad():
            t = t_from_c_noise(args[2].detach().float().reshape(-1))
            lo, hi = float(t.min()), float(t.max())
        if hi - lo > 1e-5:
            raise RuntimeError(f"rows of one call at different t ({lo}..{hi}): not TabDiff's sampler")
        self._call += 1
        self._t = 0.5 * (lo + hi)
        self.calls.append((self._call, self._t, hi - lo))
        return None

    def _hook(self, block: int):
        def hook(module, inputs, A):
            import torch
            with torch.no_grad():
                N = A.shape[-1]
                eye = torch.eye(N, dtype=torch.bool, device=A.device)
                kept = (A.detach() > 0) & ~eye
                counts = kept.sum(0).to("cpu", torch.int64).numpy()
            self.records.append((self._call, self._t, block, counts, int(A.shape[0])))
            return None
        return hook

    def to_rows(self):
        rows = []
        for call, t, block, counts, n in self.records:
            ii, jj = np.nonzero(counts)
            for i, j in zip(ii.tolist(), jj.tolist()):
                rows.append((call, t, block, i, j, int(counts[i, j]), n))
        return rows


def build_lean(run: dict, device: str):
    """The run's sampling model from its own spec + config.pkl, the checkpoint loaded
    as TabDiff's Trainer loads it (trainer.py:77-81). Same objects and arguments as
    tabdiff_gnn/entry.build (checked against it with --verify_entry)."""
    import torch
    sys.path.insert(0, TABDIFF_ROOT)
    from denoiser import build_denoiser
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model
    spec, raw = _json(run["spec"]), load_config(run["config"])
    up = raw["unimodmlp_params"]
    backbone = build_denoiser(spec["denoiser"], up, spec.get("gnn"), dim_t=spec.get("dim_t") or C.DIM_T,
                              node_noise=spec.get("node_noise") or "shared",
                              geo_loss=spec.get("geo_loss", False), geo_edge_bias=spec.get("geo_edge_bias", False))
    model = Model(backbone, **raw["diffusion_params"]["edm_params"])
    model.to(device)
    dp = dict(raw["diffusion_params"], scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    categories = np.array(up["categories"], dtype=np.int64) - 1          # TabDiffDataset.categories
    diffusion = UnifiedCtimeDiffusion(num_classes=categories, num_numerical_features=int(up["d_numerical"]),
                                      denoise_fn=model, y_only_model=None, **dp, device=device)
    if spec.get("node_noise") == "per_column":
        backbone.attach_diffusion(diffusion)
    diffusion.to(device)
    diffusion.train()                                   # entry.build; cmd_sample keeps it
    sd = load_ckpt(run["ckpt"], device)
    diffusion._denoise_fn.load_state_dict(sd["denoise_fn"])
    diffusion.num_schedule.load_state_dict(sd["num_schedule"])
    diffusion.cat_schedule.load_state_dict(sd["cat_schedule"])
    return diffusion, backbone, raw


def draw(diffusion, n: int, seed: int):
    """entry.cmd_sample's draw (seed, then TabDiff's sampler), before decoding, as ONE
    batch of exactly n rows: sample_all always draws a full batch_size batch and keeps
    the first n (unified_ctime_diffusion.py:215-234), so the config's sample batch
    (10000) would run -- and log -- 10000 rows for n = 2000."""
    import entry
    entry.seed_everything(seed)
    return diffusion.sample_all(n, n, keep_nan_samples=True)


def compare(a, b, d_num: int) -> dict:
    import torch
    same = bool(torch.equal(a, b))
    dn = float((a[:, :d_num] - b[:, :d_num]).abs().max()) if d_num else 0.0
    dc = float((a[:, d_num:] != b[:, d_num:]).float().mean()) if a.shape[1] > d_num else 0.0
    return {"identical": same, "max_abs_diff_num": dn, "frac_cat_differ": dc}


def sample_run(run: dict, out: str, device: str, n_rows: int, seed: int, verify_entry: int) -> dict:
    import pandas as pd
    import torch
    t0 = time.time()
    diffusion, backbone, raw = build_lean(run, device)
    d_num = int(raw["unimodmlp_params"]["d_numerical"])
    a = draw(diffusion, n_rows, seed)                              # logging off
    log = EdgeLogger().attach(backbone)
    b = draw(diffusion, n_rows, seed)                              # logging on
    log.detach()
    c = draw(diffusion, n_rows, seed)                              # off again (run-to-run reference)
    if any(r[4] != n_rows for r in log.records):
        raise RuntimeError(f"logged batches of {sorted({r[4] for r in log.records})} rows, expected {n_rows}")
    chk = {"on_vs_off": compare(a, b, d_num), "off_vs_off_repeat": compare(a, c, d_num)}
    if verify_entry:
        chk["lean_vs_entry"] = verify_against_entry(run, diffusion, device, min(verify_entry, n_rows), seed, d_num)
    rows = log.to_rows()
    tag = f"{run['dataset']}_{run['arm']}"
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    df = pd.DataFrame(rows, columns=["call", "t", "block", "i", "j", "count", "n_rows"])
    df.to_csv(os.path.join(out, "raw", f"edge_counts_{tag}.csv.gz"), index=False, float_format="%.7g")
    n_blocks = len(backbone.blocks)
    meta = {"dataset": run["dataset"], "arm": run["arm"], "run": rel(run["run"]), "checkpoint": run["checkpoint"],
            "checkpoint_sha256_16": file_info(run["ckpt"])["sha256_16"], "producer_commit": run["producer_commit"],
            "n_rows": n_rows, "sample_batch_size": n_rows, "config_sample_batch_size": raw["sample"]["batch_size"],
            "generation_seed": seed, "num_timesteps": raw["diffusion_params"]["num_timesteps"],
            "sampler_params": raw["diffusion_params"]["sampler_params"], "n_calls": len(log.calls),
            "calls": [{"call": k, "t": t, "t_spread": s} for k, t, s in log.calls], "n_blocks": n_blocks,
            "n_nodes": int(backbone.n_nodes), "gnn": _json(run["spec"]).get("gnn"),
            "checks": chk, "device": device, "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
            "analysis_commit": analysis_commit(), "finished_utc": _now()}
    with open(os.path.join(out, "raw", f"sample_meta_{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"  [ok] {tag}: {len(log.calls)} calls x {n_blocks} blocks, on==off {chk['on_vs_off']['identical']}, "
          f"repeat {chk['off_vs_off_repeat']['identical']}"
          + (f", entry {chk['lean_vs_entry'].get('identical')}" if verify_entry else "")
          + f" ({meta['seconds']} s)", flush=True)
    del diffusion, backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def verify_against_entry(run: dict, lean, device: str, n: int, seed: int, d_num: int) -> dict:
    """Same draw through the production path: entry.build(..., 'test', ckpt_path=...) ->
    TabDiff's Trainer loads the checkpoint (k logging off, nothing written)."""
    try:
        import entry
        spec, raw = _json(run["spec"]), load_config(run["config"])
        trainer, diffusion, _, _ = entry.build(spec, raw, device, "test", run["run"], ckpt_path=run["ckpt"],
                                               k_logging=False)
        e = draw(diffusion, n, seed)
        l1 = draw(lean, n, seed)
        out = compare(l1, e, d_num)
        out["n_rows"] = n
        del trainer, diffusion
        return out
    except Exception as exc:                            # recorded, reported in SCHEDULES.md
        return {"identical": None, "error": f"{type(exc).__name__}: {exc}"}


def sample(exp_root: str, out: str, device: str, n_rows: int, seed: int, verify_entry: int,
           datasets=DATASETS, arms=SAMPLE_ARMS) -> list:
    metas = []
    for ds in datasets:
        for arm in arms:
            r = resolve_run(ds, arm, exp_root)
            if r["status"] != "ok":
                print(f"  [skip] {ds}/{arm}: {r['status']}")
                continue
            metas.append(sample_run(r, out, device, n_rows, seed, verify_entry))
    _write_prov(out, merge=True)
    return metas


# ===========================================================================
# Stage 3 -- interaction statistics
# ===========================================================================
def call_tensor(df, n_nodes: int, blocks=None):
    """(t (M,), counts (M, N, N) float64, rows (M,)) for the chosen blocks, one entry
    per (call, block)."""
    sub = df if blocks is None else df[df["block"].isin(blocks)]
    keys = sub[["call", "block"]].drop_duplicates().sort_values(["call", "block"]).to_numpy()
    idx = {(int(c), int(b)): m for m, (c, b) in enumerate(keys)}
    M = len(keys)
    counts = np.zeros((M, n_nodes, n_nodes))
    t = np.zeros(M)
    rows = np.zeros(M)
    m = np.array([idx[(int(c), int(b))] for c, b in zip(sub["call"], sub["block"])], dtype=np.int64)
    np.add.at(counts, (m, sub["i"].to_numpy(), sub["j"].to_numpy()), sub["count"].to_numpy(float))
    t[m] = sub["t"].to_numpy()
    rows[m] = sub["n_rows"].to_numpy()
    return t, counts, rows


def clean_share(t, counts, tstar):
    """Observed / expected kept-edge share into clean columns (t*_j > t) over the
    informative (call, i) pairs. Returns (obs, exp, n_pairs, n_edges)."""
    N = len(tstar)
    clean = tstar[None, :] > t[:, None]                               # (M, N)
    K = counts.sum(2)                                                 # (M, N) kept edges of i
    obs = np.einsum("mij,mj->mi", counts, clean)
    others = clean.sum(1, keepdims=True) - clean                      # clean columns other than i
    exp = K * others / (N - 1)
    inf = (others > 0) & (others < N - 1) & (K > 0)
    n_edges = K[inf].sum()
    if n_edges == 0:
        return np.nan, np.nan, 0, 0.0
    return obs[inf].sum() / n_edges, exp[inf].sum() / n_edges, int(inf.sum()), float(n_edges)


def earlier_share(counts, tstar):
    """Secondary definition: share of i's kept edges into columns revealed before i
    (t*_j > t*_i), time-independent; informative i = some but not all others earlier."""
    N = len(tstar)
    tot = counts.sum(0)                                               # (N, N)
    E = tstar[None, :] > tstar[:, None]
    np.fill_diagonal(E, False)
    K = tot.sum(1)
    n_earlier = E.sum(1)
    inf = (n_earlier > 0) & (n_earlier < N - 1) & (K > 0)
    if K[inf].sum() == 0:
        return np.nan, np.nan
    return (tot * E)[inf].sum() / K[inf].sum(), (K * n_earlier / (N - 1))[inf].sum() / K[inf].sum()


def permutation_test(t, counts, tstar, n_perm: int, seed: int = 0) -> dict:
    obs, exp, n_pairs, n_edges = clean_share(t, counts, tstar)
    eo, ee = earlier_share(counts, tstar)
    res = {"obs_clean": obs, "exp_clean": exp, "excess_clean": obs - exp,
           "ratio_clean": obs / exp if exp else np.nan, "n_informative_pairs": n_pairs,
           "n_kept_edges_informative": n_edges, "obs_earlier": eo, "exp_earlier": ee,
           "excess_earlier": eo - ee, "n_perm": n_perm}
    if not np.isfinite(obs):
        return {**res, "p_two_sided": np.nan, "p_greater": np.nan, "p_earlier_two_sided": np.nan}
    rng = np.random.default_rng(seed)
    d0, e0 = obs - exp, eo - ee
    ge = gt = ge_e = 0
    for _ in range(n_perm):
        tp = tstar[rng.permutation(len(tstar))]
        o, x, _, _ = clean_share(t, counts, tp)
        d = (o - x) if np.isfinite(o) else 0.0
        ge += abs(d) >= abs(d0) - 1e-12
        gt += d >= d0 - 1e-12
        if np.isfinite(e0):
            a, b = earlier_share(counts, tp)
            ge_e += abs((a - b) if np.isfinite(a) else 0.0) >= abs(e0) - 1e-12
    return {**res, "p_two_sided": (1 + ge) / (1 + n_perm), "p_greater": (1 + gt) / (1 + n_perm),
            "p_earlier_two_sided": (1 + ge_e) / (1 + n_perm) if np.isfinite(e0) else np.nan}


def holm(p):
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m, run = len(idx), 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out


# ===========================================================================
# Stage 3 -- report
# ===========================================================================
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:d}"
    return f"{x:.{nd}f}"


def _pfmt(p):
    if p is None or not np.isfinite(p):
        return "—"
    return f"{p:.4f}" if p >= 1e-3 else f"{p:.1e}"


def _sgn(x, nd=3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    r = round(float(x), nd)
    return f"{r + 0.0:+.{nd}f}"


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def arm_style():
    # categorical slots 1, 2, 3, 7 of the reference palette + distinct markers
    return {"fixed": ("#2a78d6", "o"), "adaptive_jenks": ("#eb6834", "s"),
            "adaptive_entmax15": ("#1baf7a", "D"), "topk_dense": ("#4a3aa7", "^")}


def report(out: str, n_perm: int, perm_seed: int = 0) -> str:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    sch = pd.read_csv(os.path.join(out, "schedules.csv"))
    runs = pd.read_csv(os.path.join(out, "runs.csv"))
    curves = pd.read_csv(os.path.join(out, "curves.csv.gz"))
    a_commit = analysis_commit()
    file_info(os.path.join(out, "schedules.csv"))
    md = []
    readings = {}
    present = sch[["dataset", "arm"]].drop_duplicates()
    datasets = [d for d in FOCUS + DATASETS if d in set(present["dataset"])]
    datasets = list(dict.fromkeys(datasets))

    def arms_of(ds):
        return [a for a in ARMS if ((present["dataset"] == ds) & (present["arm"] == a)).any()]

    def ts(ds, arm):
        s = sch[(sch["dataset"] == ds) & (sch["arm"] == arm)].sort_values("node")
        return s

    # ---- Table 1: learned parameters per run -------------------------------
    t1 = []
    for ds in datasets:
        for arm in arms_of(ds):
            s = ts(ds, arm)
            n, c = s[s["type"] == "num"], s[s["type"] == "cat"]
            t1.append({"dataset": ds, "arm": arm, "checkpoint": s["checkpoint"].iloc[0],
                       "n_num": len(n), "n_cat": len(c),
                       "rho_min": n["value"].min(), "rho_max": n["value"].max(), "rho_init": n["value_init"].iloc[0] if len(n) else np.nan,
                       "k_min": c["value"].min(), "k_max": c["value"].max(), "k_init": c["value_init"].iloc[0] if len(c) else np.nan,
                       "tstar_num_min": n["t_star"].min(), "tstar_num_max": n["t_star"].max(),
                       "tstar_cat_min": c["t_star"].min(), "tstar_cat_max": c["t_star"].max(),
                       "tstar_num_mean": n["t_star"].mean(), "tstar_cat_mean": c["t_star"].mean(),
                       "mean_abs_dtstar_vs_init": (s["t_star"] - s["t_star_init"]).abs().mean(),
                       "source": s["source"].iloc[0], "source_commit": s["source_commit"].iloc[0],
                       "producer_commit": s["producer_commit"].iloc[0], "analysis_commit": a_commit})
    t1 = pd.DataFrame(t1)
    t1.to_csv(os.path.join(out, "table1_parameters.csv"), index=False)

    # ---- Table 2: reveal order per dataset ----------------------------------
    order_rows = []
    for ds in datasets:
        for arm in arms_of(ds):
            s = ts(ds, arm).sort_values("reveal_rank")
            for _, r in s.iterrows():
                order_rows.append({"dataset": ds, "arm": arm, "reveal_rank": r["reveal_rank"], "column": r["column"],
                                   "type": r["type"], "t_star": r["t_star"], "source": r["source"],
                                   "source_commit": r["source_commit"], "producer_commit": r["producer_commit"]})
    pd.DataFrame(order_rows).to_csv(os.path.join(out, "table2_reveal_order.csv"), index=False)

    # ---- Table 3: Spearman between arms --------------------------------------
    t3 = []
    for ds in datasets:
        arms = arms_of(ds)
        for x in range(len(arms)):
            for y in range(x + 1, len(arms)):
                a, b = ts(ds, arms[x]), ts(ds, arms[y])
                m = a.merge(b, on="node", suffixes=("_a", "_b"))
                rho, p = stats.spearmanr(m["t_star_a"], m["t_star_b"])
                within = {}
                for typ in ("num", "cat"):
                    mm = m[m["type_a"] == typ]
                    within[typ] = (stats.spearmanr(mm["t_star_a"], mm["t_star_b"])[0]
                                   if len(mm) >= 3 and mm["t_star_a"].nunique() > 1 and mm["t_star_b"].nunique() > 1 else np.nan)
                t3.append({"dataset": ds, "arm_a": arms[x], "arm_b": arms[y], "n_columns": len(m),
                           "spearman": rho, "p": p, "spearman_num_only": within["num"],
                           "spearman_cat_only": within["cat"],
                           "same_rank": int((m["reveal_rank_a"] == m["reveal_rank_b"]).sum()),
                           "source": f"{a['source'].iloc[0]}; {b['source'].iloc[0]}",
                           "source_commit": f"{a['source_commit'].iloc[0]}; {b['source_commit'].iloc[0]}",
                           "producer_commit": f"{a['producer_commit'].iloc[0]}; {b['producer_commit'].iloc[0]}",
                           "analysis_commit": a_commit})
    t3 = pd.DataFrame(t3)
    t3.to_csv(os.path.join(out, "table3_spearman.csv"), index=False)

    # ---- Table 4: movers fixed -> entmax15 ----------------------------------
    t4 = []
    for ds in datasets:
        if not {"fixed", "adaptive_entmax15"} <= set(arms_of(ds)):
            continue
        m = ts(ds, "fixed").merge(ts(ds, "adaptive_entmax15"), on="node", suffixes=("_fixed", "_entmax15"))
        for _, r in m.iterrows():
            t4.append({"dataset": ds, "node": r["node"], "column": r["column_fixed"], "type": r["type_fixed"],
                       "param": r["param_fixed"], "value_fixed": r["value_fixed"], "value_entmax15": r["value_entmax15"],
                       "t_star_fixed": r["t_star_fixed"], "t_star_entmax15": r["t_star_entmax15"],
                       "delta_t_star": r["t_star_entmax15"] - r["t_star_fixed"],
                       "rank_fixed": r["reveal_rank_fixed"], "rank_entmax15": r["reveal_rank_entmax15"],
                       "source": f"{r['source_fixed']}; {r['source_entmax15']}",
                       "source_commit": f"{r['source_commit_fixed']}; {r['source_commit_entmax15']}",
                       "producer_commit": f"{r['producer_commit_fixed']}; {r['producer_commit_entmax15']}"})
    t4 = pd.DataFrame(t4)
    if len(t4):
        t4 = t4.assign(abs_delta=t4["delta_t_star"].abs()).sort_values(
            ["dataset", "abs_delta"], ascending=[True, False],
            key=lambda c: c.map({d: i for i, d in enumerate(datasets)}) if c.name == "dataset" else c)
    t4.to_csv(os.path.join(out, "table4_movers_fixed_vs_entmax15.csv"), index=False)

    # ---- Part 2 inputs --------------------------------------------------------
    metas = {}
    for p in sorted(glob.glob(os.path.join(out, "raw", "sample_meta_*.json"))):
        m = _json(p)
        metas[(m["dataset"], m["arm"])] = (m, p)
    rank_ds = {d: i for i, d in enumerate(dict.fromkeys(FOCUS + DATASETS))}
    metas = dict(sorted(metas.items(), key=lambda kv: (rank_ds.get(kv[0][0], 99), list(ARMS).index(kv[0][1]))))
    edges = {}
    for (ds, arm), (m, p) in metas.items():
        ep = os.path.join(out, "raw", f"edge_counts_{ds}_{arm}.csv.gz")
        edges[(ds, arm)] = (pd.read_csv(ep), ep)
        file_info(ep)
        file_info(p)

    # Table 6: sampling checks
    t6 = []
    for (ds, arm), (m, p) in metas.items():
        ch = m["checks"]
        le = ch.get("lean_vs_entry") or {}
        t6.append({"dataset": ds, "arm": arm, "n_rows": m["n_rows"], "n_calls": m["n_calls"], "n_blocks": m["n_blocks"],
                   "on_vs_off_identical": ch["on_vs_off"]["identical"],
                   "on_vs_off_max_abs_num": ch["on_vs_off"]["max_abs_diff_num"],
                   "on_vs_off_frac_cat": ch["on_vs_off"]["frac_cat_differ"],
                   "repeat_identical": ch["off_vs_off_repeat"]["identical"],
                   "lean_vs_entry_identical": le.get("identical"),
                   "lean_vs_entry_max_abs_num": le.get("max_abs_diff_num"), "lean_vs_entry_error": le.get("error"),
                   "device": m["device"], "torch": m["torch"], "source": rel(p), "source_commit": src(p)["source_commit"],
                   "producer_commit": m["producer_commit"], "analysis_commit": m.get("analysis_commit")})
    t6 = pd.DataFrame(t6)
    t6.to_csv(os.path.join(out, "table5_sampling_checks.csv"), index=False)

    # Edge frequencies per block and t bin
    ef, t7, t8 = [], [], []
    for (ds, arm), (df, ep) in edges.items():
        s = ts(ds, arm)
        if not len(s):
            continue
        tstar = s["t_star"].to_numpy()
        cols = s["column"].tolist()
        N = len(tstar)
        meta = metas[(ds, arm)][0]
        prov = src(ep, meta["producer_commit"])
        nb = meta["n_blocks"]
        for b in range(nb):
            t, cnt, rows = call_tensor(df, N, [b])
            tb = np.minimum((t * N_TBINS).astype(int), N_TBINS - 1)
            for k in range(N_TBINS):
                sel = tb == k
                if not sel.any():
                    continue
                freq = cnt[sel].sum(0) / rows[sel].sum()
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            ef.append((ds, arm, b, k, k / N_TBINS, (k + 1) / N_TBINS, i, j, cols[i], cols[j],
                                       freq[i, j], int(rows[sel].sum())))
                o, x, npairs, ned = clean_share(t[sel], cnt[sel], tstar)
                t8.append({"dataset": ds, "arm": arm, "block": b, "t_bin": k, "t_lo": k / N_TBINS,
                           "t_hi": (k + 1) / N_TBINS, "obs_clean": o, "exp_clean": x, "excess_clean": o - x,
                           "n_informative_pairs": npairs, "n_kept_edges_informative": ned,
                           "mean_k": cnt[sel].sum() / rows[sel].sum() / N, **prov})
        for blocks in [[b] for b in range(nb)] + [list(range(nb))]:
            t, cnt, _ = call_tensor(df, N, blocks)
            res = permutation_test(t, cnt, tstar, n_perm, perm_seed)
            t7.append({"dataset": ds, "arm": arm, "block": blocks[0] if len(blocks) == 1 else "all",
                       "n_nodes": N, **res, **prov, "analysis_commit": a_commit})
    pd.DataFrame(ef, columns=["dataset", "arm", "block", "t_bin", "t_lo", "t_hi", "i", "j", "column_i", "column_j",
                              "freq", "n_rows_x_calls"]).to_csv(os.path.join(out, "edge_freq.csv.gz"), index=False,
                                                               float_format="%.5g")
    t7 = pd.DataFrame(t7)
    if len(t7):
        blk = t7["block"] != "all"
        t7["p_holm"] = np.nan
        t7.loc[blk, "p_holm"] = holm(t7.loc[blk, "p_two_sided"].to_numpy())
    t7.to_csv(os.path.join(out, "table6_interaction.csv"), index=False)
    t8 = pd.DataFrame(t8)
    t8.to_csv(os.path.join(out, "table7_interaction_by_tbin.csv"), index=False)

    # ---- figures --------------------------------------------------------------
    figs = []
    for ds in datasets:
        figs += fig_curves(plt, curves, sch, ds, arms_of(ds), fig_dir)
    figs += fig_tstar(plt, sch, datasets, arms_of, fig_dir)
    for ds in datasets:
        figs += fig_heatmaps(plt, edges, metas, sch, ds, fig_dir)

    # ---- SCHEDULES.md -----------------------------------------------------------
    ref_p = os.path.join(out, "reference_search.json")
    md += header_md(runs, a_commit, n_perm, metas, _json(ref_p) if os.path.exists(ref_p) else None)
    md += section_part1(t1, sch, datasets, arms_of, t3, t4)
    md += section_part2(t6, t7, t8, metas)
    md += ["## Figures", ""] + [f"- `{os.path.relpath(f, out)}`" for f in figs] + [""]
    md += ["## Files", "",
           "`runs.csv` (run status), `reference_search.json` (checkpoint search for a TabDiff reference arm), `schedules.csv` (per column: raw and transformed parameter, init, t*, rank; "
           "provenance), `curves.csv.gz` (sigma_i(t) / mask_j(t) on 101 t points; arm `init` = TabDiff's "
           "initial schedule), `table1..7_*.csv` (the tables above, with source / source_commit / "
           "producer_commit), `edge_freq.csv.gz` (per dataset, arm, block, t bin: fraction of rows in which "
           "node i keeps node j), `raw/edge_counts_<ds>_<arm>.csv.gz` (per denoiser call and block: rows in "
           "which i keeps j, nonzero only), `raw/sample_meta_<ds>_<arm>.json` (sampling settings, call t's, "
           "checks), `provenance.csv` (every file read: git commit or untracked + sha256/16).", ""]
    _write_prov(out, merge=True)
    text = "\n".join(md)
    with open(os.path.join(out, "SCHEDULES.md"), "w") as fh:
        fh.write(text)
    print(f"  wrote {rel(os.path.join(out, 'SCHEDULES.md'))}: {len(sch)} column rows, {len(t7)} interaction rows")
    return text


# ---------------------------------------------------------------------------
# Markdown sections (every reading line is computed from the table it follows)
# ---------------------------------------------------------------------------
def reference_md(ref) -> str:
    if ref is None:
        return "**Reference arm (TabDiff's own model).** Not searched (rerun `extract`)."
    files = ref["checkpoint_files"]
    if ref["tabdiff_checkpoints"]:
        return ("**Reference arm (TabDiff's own model).** TabDiff checkpoints FOUND on disk but NOT included "
                "(their dataset / column order is not verified by this script): "
                + ", ".join(f"`{f}`" for f in ref["tabdiff_checkpoints"]) + ". TabDiff's *initial* schedule is "
                "shown as arm `init`.")
    return ("**Reference arm (TabDiff's own model).** None available: no TabDiff-trained checkpoint in the "
            f"repository or third_party/ (searched {ref['searched_utc']}: {ref['searched']}; checkpoint files "
            f"found: {', '.join(f'`{f}`' for f in files) if files else 'none'} -- none in TabDiff's format with "
            "its own denoiser). The TabDiff submodule contains no trained model, Route A has no transformer runs by design, and "
            "nothing was downloaded. TabDiff's *initial* schedule is shown instead (arm `init`).")


def header_md(runs, a_commit, n_perm, metas, ref=None) -> list:
    st = runs.copy()
    rows = [(r["dataset"], r["arm"], r["status"], r["checkpoint"] if isinstance(r["checkpoint"], str) else "—",
             (r["producer_commit"] or "—")[:12] if isinstance(r["producer_commit"], str) else "—")
            for _, r in st.iterrows()]
    lines = ["# Learned noise schedules vs the learned graph (Route A)", "",
             f"Generated {_now()} by `analysis/tabdiff/schedules_vs_graph.py` at commit `{a_commit}`; "
             f"TabDiff @ `{C.TABDIFF_COMMIT[:7]}` (file digests verified by tabdiff_gnn/config.verify_tabdiff "
             "in the training runs). Seed-0 runs only; nothing is trained. Rerun the tables and figures from the "
             "files here with `python analysis/tabdiff/schedules_vs_graph.py report`.", "",
             reference_md(ref), "",
             "### Runs", "", md_table(["dataset", "arm", "status", "checkpoint (sampled)", "run commit"], rows), ""]
    return lines


def section_part1(t1, sch, datasets, arms_of, t3, t4) -> list:
    L = ["## Part 1 — learned schedules (checkpoints only)", "",
         "**Where TabDiff defines them** (`third_party/TabDiff`, tests check each line):", ""]
    L += [f"- `{f}:{n}` `{code}` — {what}" for f, n, code, what in SCHEDULE_CODE]
    p = sch.iloc[0] if len(sch) else None
    if p is not None:
        L += ["", f"Run values (config.pkl): sigma_min {p['sigma_min']:g}, sigma_max {p['sigma_max']:g}, "
                  f"rho_offset {p['rho_offset']:g}, rho_init (raw) {p['rho_init']:g} -> rho "
                  f"{rho_from_raw(p['rho_init'], p['rho_offset']):.4f}, k_offset {p['k_offset']:g}, k_init (raw) "
                  f"{p['k_init']:g} -> k {k_from_raw(p['k_init'], p['k_offset']):.5f}, eps_max {p['eps_max']:g}, "
                  f"eps_min {p['eps_min']:g}, sigma_data {p['sigma_data']:g}. Reveal time: numerical "
                  f"t* = (sigma_data^(1/rho) - s_min^(1/rho)) / (s_max^(1/rho) - s_min^(1/rho)) "
                  f"(init {tstar_num(rho_from_raw(p['rho_init'], p['rho_offset']), p['sigma_min'], p['sigma_max'], p['sigma_data']):.4f}); "
                  f"categorical t* = ((0.5 - e_min) / (1 - e_max - e_min))^(1/k) "
                  f"(init {tstar_cat(k_from_raw(p['k_init'], p['k_offset']), p['eps_max'], p['eps_min']):.4f}). "
                  "Larger t* = informative earlier in sampling (t runs 1 -> 0)."]
    if not len(t1):
        return L + ["", "_No run with a checkpoint on disk — nothing to report._", ""]
    # Table 1
    L += ["", "### Table 1 — learned parameters per run", "",
          md_table(["dataset", "arm", "n num / cat", "rho [min, max]", "k [min, max]", "t* num [min, max]",
                    "t* cat [min, max]", "mean abs(t* − t*init)"],
                   [(r.dataset, SHORT[r.arm], f"{r.n_num} / {r.n_cat}", f"[{_fmt(r.rho_min)}, {_fmt(r.rho_max)}]",
                     f"[{_fmt(r.k_min)}, {_fmt(r.k_max)}]", f"[{_fmt(r.tstar_num_min)}, {_fmt(r.tstar_num_max)}]",
                     f"[{_fmt(r.tstar_cat_min)}, {_fmt(r.tstar_cat_max)}]", _fmt(r.mean_abs_dtstar_vs_init))
                    for r in t1.itertuples()]), ""]
    num, cat = sch[sch["type"] == "num"], sch[sch["type"] == "cat"]
    dev = t1.sort_values("mean_abs_dtstar_vs_init")
    earlier = int((t1["tstar_num_mean"] > t1["tstar_cat_mean"]).sum())
    both = int(t1[["tstar_num_mean", "tstar_cat_mean"]].notna().all(1).sum())
    L += ["Reading:",
          f"1. Numerical t* spans [{_fmt(num['t_star'].min())}, {_fmt(num['t_star'].max())}] over all runs "
          f"(init {_fmt(num['t_star_init'].iloc[0]) if len(num) else '—'}); categorical t* spans "
          f"[{_fmt(cat['t_star'].min())}, {_fmt(cat['t_star'].max())}] (init "
          f"{_fmt(cat['t_star_init'].iloc[0]) if len(cat) else '—'}).",
          f"2. Mean |t* − t*init| is largest for {dev.iloc[-1]['dataset']}/{SHORT[dev.iloc[-1]['arm']]} "
          f"({_fmt(dev.iloc[-1]['mean_abs_dtstar_vs_init'])}) and smallest for {dev.iloc[0]['dataset']}/"
          f"{SHORT[dev.iloc[0]['arm']]} ({_fmt(dev.iloc[0]['mean_abs_dtstar_vs_init'])}).",
          f"3. The mean numerical t* exceeds the mean categorical t* (numerical columns informative earlier "
          f"on average) in {earlier} of {both} runs that have both types.", ""]
    # Table 2: one table per dataset
    L += ["### Table 2 — columns ordered by reveal time t* (rank 1 = informative first)", ""]
    for ds in datasets:
        arms = arms_of(ds)
        by = {a: sch[(sch["dataset"] == ds) & (sch["arm"] == a)].sort_values("reveal_rank") for a in arms}
        n = max(len(v) for v in by.values())
        rows = []
        for k in range(n):
            rows.append([k + 1] + [f"{by[a].iloc[k]['column']} ({by[a].iloc[k]['type'][0]}) {by[a].iloc[k]['t_star']:.3f}"
                                   if k < len(by[a]) else "" for a in arms])
        L += [f"**{ds}** (n = numerical, c = categorical; value = t*)", "",
              md_table(["rank"] + [SHORT[a] for a in arms], rows), ""]
        ref = by.get("fixed")
        same = []
        if ref is not None:
            for a in arms:
                if a != "fixed":
                    m = ref.merge(by[a], on="node")
                    same.append(f"{SHORT[a]} {int((m['reveal_rank_x'] == m['reveal_rank_y']).sum())}/{len(m)}")
        means = [f"{SHORT[a]} {by[a][by[a]['type'] == 'num']['t_star'].mean():.3f} / "
                 f"{by[a][by[a]['type'] == 'cat']['t_star'].mean():.3f}" for a in arms]
        L += ["Reading:",
              "1. First revealed: " + "; ".join(f"{SHORT[a]} {by[a].iloc[0]['column']}" for a in arms)
              + ". Last: " + "; ".join(f"{SHORT[a]} {by[a].iloc[-1]['column']}" for a in arms) + ".",
              "2. Columns at the same rank as in fixed: " + (", ".join(same) if same else "no fixed arm") + ".",
              "3. Mean t* numerical / categorical: " + "; ".join(means) + ".", ""]
    # Table 3
    L += ["### Table 3 — Spearman correlation of t* between arms", "",
          md_table(["dataset", "arms", "n", "Spearman (all)", "num only", "cat only", "same rank"],
                   [(r.dataset, f"{SHORT[r.arm_a]} vs {SHORT[r.arm_b]}", r.n_columns, _fmt(r.spearman),
                     _fmt(r.spearman_num_only), _fmt(r.spearman_cat_only), r.same_rank) for r in t3.itertuples()]), ""]
    if len(t3):
        lo, hi = t3.loc[t3["spearman"].idxmin()], t3.loc[t3["spearman"].idxmax()]
        fe = t3[(t3["arm_a"] == "fixed") & (t3["arm_b"] == "adaptive_entmax15")]
        L += ["Reading:",
              f"1. Median Spearman over {len(t3)} arm pairs: {_fmt(t3['spearman'].median())} "
              f"(range [{_fmt(t3['spearman'].min())}, {_fmt(t3['spearman'].max())}]).",
              f"2. Lowest: {lo['dataset']} {SHORT[lo['arm_a']]} vs {SHORT[lo['arm_b']]} ({_fmt(lo['spearman'])}); "
              f"highest: {hi['dataset']} {SHORT[hi['arm_a']]} vs {SHORT[hi['arm_b']]} ({_fmt(hi['spearman'])}).",
              "3. fixed vs entmax15: " + ("; ".join(f"{r.dataset} {_fmt(r.spearman)}" for r in fe.itertuples())
                                          or "no dataset has both") + ".", ""]
    else:
        L += ["_Fewer than two arms on every dataset: no correlation._", ""]
    # Table 4
    L += ["### Table 4 — columns that move most between fixed and entmax15 (top 5 per dataset, |Δt*|)", ""]
    if len(t4):
        rows = []
        for ds in dict.fromkeys(t4["dataset"]):
            for r in t4[t4["dataset"] == ds].head(5).itertuples():
                rows.append((ds, r.column, r.type, f"{_fmt(r.value_fixed)} → {_fmt(r.value_entmax15)}",
                             f"{_fmt(r.t_star_fixed)} → {_fmt(r.t_star_entmax15)}", _sgn(r.delta_t_star),
                             f"{r.rank_fixed} → {r.rank_entmax15}"))
        L += [md_table(["dataset", "column", "type", "rho / k (fixed → entmax15)", "t*", "Δt*", "rank"], rows), ""]
        top = t4.loc[t4.groupby("dataset")["abs_delta"].idxmax()]
        med = t4.groupby("dataset")["abs_delta"].median()
        L += ["Reading:",
              "1. Largest |Δt*| per dataset: " + "; ".join(
                  f"{r.dataset} {r.column} ({_sgn(r.delta_t_star)})" for r in
                  top.set_index("dataset").loc[[d for d in dict.fromkeys(t4["dataset"])]].reset_index().itertuples()) + ".",
              "2. Median |Δt*|: " + "; ".join(f"{d} {med[d]:.3f}" for d in dict.fromkeys(t4["dataset"])) + ".",
              "3. Columns revealed earlier / later under entmax15 than fixed (Δt* > 0 / < 0): " + "; ".join(
                  f"{d} {int((t4[t4['dataset'] == d]['delta_t_star'] > 0).sum())} / "
                  f"{int((t4[t4['dataset'] == d]['delta_t_star'] < 0).sum())}" for d in dict.fromkeys(t4["dataset"])) + ".",
              ""]
    else:
        L += ["_No dataset has both a fixed and an entmax15 run._", ""]
    return L


def section_part2(t6, t7, t8, metas) -> list:
    L = ["## Part 2 — where the edges point (sampling with edge logging)", ""]
    if not len(t6):
        return L + ["_No sampling output in raw/ — run the `sample` stage (GPU job "
                    "hpc_transfer/jobs_tabdiff_gnn/analysis_schedules.sh)._", ""]
    m0 = next(iter(metas.values()))[0]
    L += [f"TabDiff's sampler (UnifiedCtimeDiffusion.sample_all; {m0['num_timesteps']} steps, sampler "
          f"{m0['sampler_params']}), {m0['n_rows']} rows as one batch (the config's sample batch "
          f"{m0.get('config_sample_batch_size')} would draw a full batch and keep the first {m0['n_rows']}), "
          f"generation seed {m0['generation_seed']}, the sampled "
          f"checkpoint; {m0['n_calls']} denoiser calls per draw (t_hat per step plus the 2nd-order correction "
          "at t_next). Edge = node i keeps node j in a block's DynamicAdjacency (A[i, j] > 0, i != j).", "",
          "### Table 5 — logging does not change the computation", "",
          md_table(["dataset", "arm", "on = off", "repeat off = off", "lean = entry.build path", "device"],
                   [(r.dataset, SHORT[r.arm], r.on_vs_off_identical, r.repeat_identical,
                     f"error: {r.lean_vs_entry_error}" if isinstance(r.lean_vs_entry_error, str)
                     else ("not checked" if r.lean_vs_entry_identical is None else r.lean_vs_entry_identical),
                     r.device) for r in t6.itertuples()]), ""]
    le = [x for x in t6["lean_vs_entry_identical"] if x is not None and not (isinstance(x, float) and np.isnan(x))]
    n_err = int(sum(isinstance(e, str) for e in t6["lean_vs_entry_error"]))
    L += ["Reading:",
          f"1. Logging on vs off: bit-identical samples in {int(t6['on_vs_off_identical'].sum())} of {len(t6)} runs "
          f"(max |Δ| numerical {t6['on_vs_off_max_abs_num'].max():.2e}, categorical mismatch "
          f"{t6['on_vs_off_frac_cat'].max():.2e}).",
          f"2. Same draw repeated without logging: identical in {int(t6['repeat_identical'].sum())} of {len(t6)} "
          "runs (run-to-run reference for item 1).",
          f"3. Rebuilt model vs the production entry.build + Trainer path: identical in "
          f"{int(sum(bool(x) for x in le))} of {len(le)} checked runs; {n_err} errored, "
          f"{len(t6) - len(le) - n_err} not checked (--verify_entry 0).", ""]
    if not len(t7):
        return L
    L += ["### Table 6 — do kept neighbours point to already-clean columns?", "",
          "Per (denoiser call, node i): observed = share of i's kept neighbours j with t*_j > t (clean at that "
          "t); expected = share of the other N − 1 columns that are clean (random choice). Pooled over the "
          "informative (call, i) pairs (some, not all, other columns clean), weighted by kept edges. "
          "Excess = observed − expected. p = two-sided permutation test over column labels (t* permuted across "
          f"columns, {int(t7['n_perm'].iloc[0])} permutations); Holm = Holm-adjusted over the per-block rows. "
          "'earlier' = secondary definition, share of kept neighbours revealed before i (t*_j > t*_i).", "",
          md_table(["dataset", "arm", "block", "informative pairs", "observed", "expected", "excess", "p", "p Holm",
                    "excess (earlier)", "p (earlier)"],
                   [(r.dataset, SHORT[r.arm], r.block, r.n_informative_pairs, _fmt(r.obs_clean), _fmt(r.exp_clean),
                     _sgn(r.excess_clean), _pfmt(r.p_two_sided),
                     _pfmt(r.p_holm), _sgn(r.excess_earlier),
                     _pfmt(r.p_earlier_two_sided)) for r in t7.itertuples()]), ""]
    blk = t7[t7["block"].astype(str) != "all"]
    fin = blk[np.isfinite(blk["excess_clean"])]
    sig = fin[fin["p_two_sided"] < 0.05]
    by_arm = t7[t7["block"].astype(str) == "all"].groupby("arm")["excess_clean"].mean()
    L += ["Reading:",
          f"1. Per-block excess ranges [{_sgn(fin['excess_clean'].min())}, {_sgn(fin['excess_clean'].max())}]; "
          f"positive (towards clean columns) in {int((fin['excess_clean'] > 0).sum())} of {len(fin)} blocks."
          if len(fin) else "1. No block has informative pairs.",
          f"2. p < 0.05 in {len(sig)} of {len(fin)} blocks"
          + (" (" + "; ".join(f"{r.dataset}/{SHORT[r.arm]}/b{r.block} {_sgn(r.excess_clean)}, p {_pfmt(r.p_two_sided)}"
                              for r in sig.itertuples()) + ")" if len(sig) else "")
          + f"; after Holm: {int((fin['p_holm'] < 0.05).sum())}.",
          "3. Mean excess over datasets (all blocks pooled): " + "; ".join(
              f"{SHORT[a]} {_sgn(by_arm[a])}" for a in SAMPLE_ARMS if a in by_arm.index) + ".", ""]
    if len(t8):
        pooled = (t8.assign(w_obs=t8["obs_clean"] * t8["n_kept_edges_informative"],
                            w_exp=t8["exp_clean"] * t8["n_kept_edges_informative"])
                  .groupby(["arm", "t_bin"])[["w_obs", "w_exp", "n_kept_edges_informative"]].sum())
        pooled = pooled[pooled["n_kept_edges_informative"] > 0]
        pooled["excess"] = (pooled["w_obs"] - pooled["w_exp"]) / pooled["n_kept_edges_informative"]
        bins = sorted({b for _, b in pooled.index})
        rows = [[SHORT[a]] + [_sgn(pooled.loc[(a, b), 'excess']) if (a, b) in pooled.index else "—" for b in bins]
                for a in SAMPLE_ARMS if a in {x for x, _ in pooled.index}]
        L += ["### Table 7 — excess by t bin (all datasets and blocks pooled, kept-edge weighted)", "",
              md_table(["arm"] + [f"t∈[{b / N_TBINS:.1f},{(b + 1) / N_TBINS:.1f})" for b in bins], rows), "",
              "Per dataset / block: `table7_interaction_by_tbin.csv` (also mean k per node and bin).", ""]
        best = pooled["excess"].idxmax() if len(pooled) else None
        worst = pooled["excess"].idxmin() if len(pooled) else None
        L += ["Reading:",
              f"1. Informative t bins (some columns clean, some not): {', '.join(f'{b / N_TBINS:.1f}' for b in bins)} "
              "(lower edge); outside them every column is on the same side of t* and the test has no content.",
              (f"2. Largest pooled excess: {SHORT[best[0]]} at t∈[{best[1] / N_TBINS:.1f},{(best[1] + 1) / N_TBINS:.1f}) "
               f"({_sgn(pooled.loc[best, 'excess'])}); smallest: {SHORT[worst[0]]} at t∈[{worst[1] / N_TBINS:.1f},"
               f"{(worst[1] + 1) / N_TBINS:.1f}) ({_sgn(pooled.loc[worst, 'excess'])}).") if best else "2. —",
              "3. Pooled bins mix datasets with different N and t* sets; the per-dataset rows are in the CSV.", ""]
    L += ["**Caveats.** t* is a deterministic threshold: a categorical column at t is unmasked in a random "
          "(1 − mask_j(t)) share of the rows, and a numerical column's noise falls continuously; 'clean' "
          "summarises the schedule, not the row's state. The permutation test conditions on the observed "
          "graphs and permutes t* across all columns (numerical and categorical alike). One training seed, one "
          f"draw of {m0['n_rows']} rows per run; p-values are not corrected across datasets except the Holm column.", ""]
    return L


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(plt, fig, path_noext) -> list:
    fig.savefig(path_noext + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(path_noext + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return [path_noext + ".png", path_noext + ".pdf"]


def fig_curves(plt, curves, sch, ds, arms, fig_dir) -> list:
    from matplotlib import cm, colors
    ref = "fixed" if "fixed" in arms else arms[0]
    rank = sch[(sch["dataset"] == ds) & (sch["arm"] == ref)].set_index("node")["reveal_rank"]
    N = len(rank)
    cmap = colors.LinearSegmentedColormap.from_list("blues", ["#b7d3f2", "#0b3d7a"])
    norm = colors.Normalize(1, max(N, 2))
    fig, ax = plt.subplots(2, len(arms), figsize=(3.2 * len(arms), 5.6), sharex=True, sharey="row", squeeze=False)
    init = curves[(curves["dataset"] == ds) & (curves["arm"] == "init")]
    for c, arm in enumerate(arms):
        cv = curves[(curves["dataset"] == ds) & (curves["arm"] == arm)]
        for row, (q, level) in enumerate((("sigma", 1.0), ("mask", MASK_LEVEL))):
            a = ax[row, c]
            sub = cv[cv["quantity"] == q]
            for node, g in sub.groupby("node"):
                a.plot(g["t"], g["value"], lw=1.2, color=cmap(norm(rank.get(node, 1))))
            ig = init[init["quantity"] == q]
            if len(sub):
                a.plot(ig["t"], ig["value"], lw=1.5, ls="--", color="#8a8984")
            a.axhline(level, lw=0.8, color="#52514e", ls=":")
            if q == "sigma":
                a.set_yscale("log")
            a.grid(alpha=0.25, lw=0.5)
            a.spines[["top", "right"]].set_visible(False)
            if row == 0:
                a.set_title(f"{SHORT[arm]}", fontsize=10)
            if row == 1:
                a.set_xlabel("t (sampling runs 1 → 0)")
            if not len(sub):
                a.text(0.5, 0.5, "no column of this type", transform=a.transAxes, ha="center", fontsize=8, color="#52514e")
        ax[0, 0].set_ylabel("numerical σ_i(t)")
        ax[1, 0].set_ylabel("categorical mask_j(t)")
    from matplotlib.lines import Line2D
    ax[0, 0].legend(handles=[Line2D([], [], ls="--", color="#8a8984", lw=1.5, label="TabDiff init"),
                             Line2D([], [], ls=":", color="#52514e", lw=0.8, label="σ_data = 1 (t*)")],
                    fontsize=7, frameon=False, loc="upper left")
    ax[1, 0].legend(handles=[Line2D([], [], ls="--", color="#8a8984", lw=1.5, label="TabDiff init"),
                             Line2D([], [], ls=":", color="#52514e", lw=0.8, label="mask 0.5 (t*)")],
                    fontsize=7, frameon=False, loc="upper left")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(f"reveal rank in {SHORT[ref]} (1 = first)", fontsize=8)
    fig.suptitle(f"{ds}: learned per-column schedules", fontsize=11)
    return _save(plt, fig, os.path.join(fig_dir, f"curves_{ds}"))


def fig_tstar(plt, sch, datasets, arms_of, fig_dir) -> list:
    if not datasets:
        return []
    style = arm_style()
    n = len(datasets)
    ncol = 2 if n > 1 else 1
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2 * ncol, 3.0 * nrow), squeeze=False)
    for k, ds in enumerate(datasets):
        a = axes[k // ncol, k % ncol]
        arms = arms_of(ds)
        ref = "fixed" if "fixed" in arms else arms[0]
        r = sch[(sch["dataset"] == ds) & (sch["arm"] == ref)].sort_values("reveal_rank")
        pos = {node: x for x, node in enumerate(r["node"])}
        for off, arm in zip(np.linspace(-0.2, 0.2, len(arms)), arms):
            s = sch[(sch["dataset"] == ds) & (sch["arm"] == arm)]
            col, mk = style[arm]
            a.scatter([pos[n_] + off for n_ in s["node"]], s["t_star"], s=22, color=col, marker=mk,
                      edgecolor="white", linewidth=0.6, label=SHORT[arm], zorder=3)
        init = sch[sch["dataset"] == ds].drop_duplicates("type")
        for _, ir in init.iterrows():
            a.axhline(ir["t_star_init"], color="#8a8984", lw=0.8, ls="--" if ir["type"] == "num" else ":")
        a.set_xticks(range(len(r)))
        a.set_xticklabels([f"{c} ({t[0]})" for c, t in zip(r["column"], r["type"])], rotation=70, ha="right", fontsize=6)
        a.set_ylabel("t*")
        a.set_title(f"{ds} (columns in {SHORT[ref]} reveal order; init: -- num, ··· cat)", fontsize=9)
        a.grid(axis="y", alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(fontsize=7, frameon=False, ncol=len(arms), loc="best")
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.tight_layout()
    return _save(plt, fig, os.path.join(fig_dir, "tstar_by_arm"))


def fig_heatmaps(plt, edges, metas, sch, ds, fig_dir) -> list:
    from matplotlib import colors
    arms = [a for a in SAMPLE_ARMS if (ds, a) in edges]
    if not arms:
        return []
    nb = max(metas[(ds, a)][0]["n_blocks"] for a in arms)
    cmap = colors.LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#0b3d7a"])
    fig, ax = plt.subplots(len(arms), 2 * nb, figsize=(2.6 * 2 * nb + 1, 2.7 * len(arms) + 0.6), squeeze=False)
    im = None
    for r_, arm in enumerate(arms):
        df = edges[(ds, arm)][0]
        s = sch[(sch["dataset"] == ds) & (sch["arm"] == arm)].sort_values("node")
        N = len(s)
        order = s.sort_values("reveal_rank")["node"].to_numpy()
        labels = [f"{c} ({t[0]})" for c, t in zip(s.set_index("node").loc[order, "column"],
                                                  s.set_index("node").loc[order, "type"])]
        for b in range(nb):
            t, cnt, rows = call_tensor(df, N, [b])
            for h, (name, (lo, hi)) in enumerate((("high noise", HIGH_NOISE), ("low noise", LOW_NOISE))):
                a = ax[r_, 2 * b + h]
                sel = (t >= lo) & (t <= hi) if hi >= 1.0 else (t >= lo) & (t < hi)
                if b >= metas[(ds, arm)][0]["n_blocks"] or not sel.any():
                    a.axis("off")
                    continue
                freq = cnt[sel].sum(0) / rows[sel].sum()
                np.fill_diagonal(freq, np.nan)
                im = a.imshow(freq[np.ix_(order, order)], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                a.set_title(f"{SHORT[arm]} · block {b} · {name} t∈[{lo:.1f},{hi:.1f}]", fontsize=7)
                fs = 5 if N > 20 else 6
                a.set_xticks(range(N))
                a.set_yticks(range(N))
                a.set_xticklabels(labels, rotation=90, fontsize=fs)
                a.set_yticklabels(labels if 2 * b + h == 0 else [], fontsize=fs)
                if 2 * b + h == 0:
                    a.set_ylabel("node i (keeps →)", fontsize=7)
                if r_ == len(arms) - 1:
                    a.set_xlabel("neighbour j", fontsize=7)
    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label("share of rows in which i keeps j", fontsize=8)
    fig.suptitle(f"{ds}: kept-edge frequency during sampling (axes ordered by each arm's t*, informative first)",
                 fontsize=10)
    return _save(plt, fig, os.path.join(fig_dir, f"edges_{ds}"))


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["extract", "sample", "report", "all"])
    ap.add_argument("--exp_root", default=os.environ.get("TRANSFER_ROOT", EXP_ROOT_DEFAULT))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="generation seed of the logged draw")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--verify_entry", type=int, default=0,
                    help="rows of an extra draw through entry.build + TabDiff's Trainer (0 = off)")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args(argv)
    ds = [d for d in a.datasets.split(",") if d]
    if sys.platform == "darwin":
        import multiprocessing as mp                    # as entry.main (TabDiffDataset lambdas)
        mp.set_start_method("fork", force=True)
    print(f"  schedules_vs_graph {a.stage}: exp_root {a.exp_root} -> {a.out}", flush=True)
    if a.stage in ("extract", "all"):
        extract(a.exp_root, a.out, ds)
    if a.stage in ("sample", "all"):
        sample(a.exp_root, a.out, a.device, a.n_rows, a.seed, a.verify_entry, ds)
    if a.stage in ("report", "all"):
        report(a.out, a.n_perm)
    return 0


if __name__ == "__main__":
    sys.exit(main())

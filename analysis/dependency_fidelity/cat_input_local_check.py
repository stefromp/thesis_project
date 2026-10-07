#!/usr/bin/env python3
"""Local CPU check of the categorical-input fix on real data.

For each dataset in DATASETS and each cat_input in ("log", "prob"): train the
GNN denoiser through the pipeline's own scripts/train.py + scripts/sample.py
(same dataset config, diffusion and transforms as ablation_runner), on a SMALL
model for a SHORT run, then report
  * final multinomial loss vs the always-uniform / real-marginal baselines
  * per-categorical-column TVD(synthetic marginal, real train marginal).

It is a sanity check of the direction of the effect, not a substitute for the
cluster runs (smaller model, 2k instead of 20k steps, one seed).

    python3 analysis/dependency_fidelity/cat_input_local_check.py [work_dir]

Training directories go to work_dir (default: a temp dir); only the JSON and
Markdown summaries are written next to this file.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATASETS = ("cardio", "churn2")
CAT_INPUTS = ("log", "prob")
STEPS, BATCH, N_SAMPLES, SEED = 2000, 1024, 10_000, 0
SMALL = dict(gnn_type="gatv2", d_model=64, n_layers=2, n_heads=4, sparsity_top_k=5)


def _setup():
    for p in (str(REPO), str(REPO / "scripts"), str(REPO / "ablation")):
        if p not in sys.path:
            sys.path.insert(0, p)
    # lib/data.py imports category_encoders, which imports statsmodels; locally
    # statsmodels 0.14.0 is incompatible with scipy 1.16 (_lazywhere removed).
    # The pipeline only uses LeaveOneOutEncoder for cat_encoding="counter",
    # never here (cat_encoding=None), so a stub that fails loudly if used is
    # enough. In-process only; the environment is not modified.
    try:
        import category_encoders  # noqa: F401
    except ImportError:
        import types

        class LeaveOneOutEncoder:  # pragma: no cover
            def __init__(self, *a, **k):
                raise RuntimeError("category_encoders stub: LeaveOneOutEncoder is not available locally")

        sys.modules["category_encoders"] = types.SimpleNamespace(LeaveOneOutEncoder=LeaveOneOutEncoder)
    import ablation_runner as ar
    ar._setup_zero_rtdl_stubs()
    return ar


def run_one(args):
    ds, cat_input, work = args
    import torch
    torch.set_num_threads(2)
    ar = _setup()
    from train import train as tabddpm_train
    from sample import sample as tabddpm_sample
    cfg = importlib.import_module(f"{ds}_ablation").DATASET_CFG
    out = Path(work) / f"{ds}_{cat_input}"
    out.mkdir(parents=True, exist_ok=True)
    model_params = dict(num_classes=cfg["num_classes"], is_y_cond=cfg["is_y_cond"],
                        rtdl_params=dict(d_layers=list(cfg["rtdl_d_layers"]), dropout=0.0))
    T = dict(seed=SEED, normalization=cfg["train_normalization"], num_nan_policy=None, cat_nan_policy=None,
             cat_min_frequency=None, cat_encoding=None, y_policy="default")
    graph = dict(enabled=True, mode="dynamic", dropout=0.0, use_attention=False, topk_mode="fixed",
                 k_min=1, topk_warmup_steps=0, num_timesteps=1000, cat_input=cat_input, **SMALL)
    common = dict(real_data_path=str(REPO / cfg["real_data_path"]), model_type="mlp",
                  T_dict=T, num_numerical_features=cfg["num_numerical_features"], device=torch.device("cpu"),
                  change_val=False, graph_params=graph, num_timesteps=1000, gaussian_loss_type="mse",
                  scheduler=cfg["scheduler"])
    tabddpm_train(parent_dir=str(out), model_params=deepcopy(model_params), seed=SEED, lr=cfg["lr"],
                  weight_decay=cfg["weight_decay"], batch_size=BATCH, steps=STEPS, checkpoint_every=0, **common)
    tabddpm_sample(parent_dir=str(out), model_path=str(out / "model_ema.pt"), model_params=deepcopy(model_params),
                   seed=SEED, num_samples=N_SAMPLES, batch_size=5000, **common)
    return ds, cat_input, str(out)


def baselines(ds):
    import torch
    _setup()
    from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
    Xc = np.load(REPO / "data" / ds / "X_cat_train.npy", allow_pickle=True).astype(str)
    codes = np.stack([np.searchsorted(np.array(sorted(set(Xc[:, j]))), Xc[:, j]) for j in range(Xc.shape[1])], 1)
    K = np.array([len(set(Xc[:, j])) for j in range(Xc.shape[1])])
    marg = np.concatenate([np.bincount(codes[:, j], minlength=K[j]) / len(codes) for j in range(len(K))])

    class Const(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, x, t, **kw):
            return self.v.repeat(x.shape[0], 1)

    out = {}
    for name, v in (("uniform", torch.zeros(int(K.sum()))), ("marginal", torch.log(torch.tensor(marg).float() + 1e-12))):
        torch.manual_seed(0)
        d = GaussianMultinomialDiffusion(num_classes=K, num_numerical_features=0, denoise_fn=Const(v),
                                         num_timesteps=1000, scheduler="cosine", device=torch.device("cpu"))
        rng = np.random.default_rng(0)
        out[name] = float(np.mean([d.mixed_loss(torch.tensor(codes[rng.choice(len(codes), 4096, replace=False)]).float(),
                                                {})[0].item() for _ in range(20)]))
    return out


def summarise(ds, cat_input, out):
    import pandas as pd
    out = Path(out)
    loss = pd.read_csv(out / "loss.csv")
    real = np.load(REPO / "data" / ds / "X_cat_train.npy", allow_pickle=True).astype(str)
    syn = np.load(out / "X_cat_train.npy", allow_pickle=True).astype(str)
    tvd = []
    for j in range(real.shape[1]):
        vals = sorted(set(real[:, j]))
        pr = np.array([(real[:, j] == v).mean() for v in vals])
        ps = np.array([(syn[:, j] == v).mean() for v in vals])
        tvd.append(round(float(0.5 * np.abs(pr - ps).sum()), 4))
    return {"dataset": ds, "cat_input": cat_input, "final_mloss": float(loss.mloss.tail(5).mean()),
            "final_gloss": float(loss.gloss.tail(5).mean()), "tvd_per_column": tvd, "tvd_max": max(tvd)}


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="cat_input_check_")
    jobs = [(ds, ci, work) for ds in DATASETS for ci in CAT_INPUTS]
    with ProcessPoolExecutor(len(jobs)) as ex:
        done = list(ex.map(run_one, jobs))
    res = {"settings": {"steps": STEPS, "batch": BATCH, "n_samples": N_SAMPLES, "seed": SEED, **SMALL,
                        "topk_mode": "fixed", "use_attention": False, "work_dir": work},
           "baselines": {ds: baselines(ds) for ds in DATASETS},
           "runs": [summarise(*d) for d in done]}
    (HERE / "cat_input_local_check.json").write_text(json.dumps(res, indent=2))
    L = ["| dataset | cat_input | final MLoss | uniform | marginal | TVD per categorical column | max |",
         "|---|---|---|---|---|---|---|"]
    for r in res["runs"]:
        b = res["baselines"][r["dataset"]]
        L.append(f"| {r['dataset']} | {r['cat_input']} | {r['final_mloss']:.3f} | {b['uniform']:.3f} | "
                 f"{b['marginal']:.3f} | {', '.join(f'{v:.3f}' for v in r['tvd_per_column'])} | {r['tvd_max']:.3f} |")
    (HERE / "cat_input_local_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

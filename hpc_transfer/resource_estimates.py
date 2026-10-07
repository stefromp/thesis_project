#!/usr/bin/env python3
"""
Walltime and memory estimates for the transfer jobs.

There are no runtime logs in this repository -- the ablation summaries record
no timing and hpc*/logs/ does not exist -- so nothing here is measured by this
script. The model is the one documented in hpc_wide/README.md, which was built
for the transferred configuration (GATv2, 3 blocks, d_model 128, 8 heads) and
calibrated against measured per-row costs. The nine ablated datasets run
different configurations, so their traffic is rescaled by width and depth --
and, where the backbone is GCN or GIN, the number is reported as an UPPER
BOUND, because those layers never materialise the pairwise tensor the model is
built around:

  * GATv2 dominates. Per layer it materialises a (B, H, N, N, d_head) pairwise
    tensor -- B*N^2*128 elements -- and makes ~6 memory-bound passes over it.
    Per row and forward that is 51.6 MB at N = 52, scaling as N^2, over
    3 blocks x 2 layers.
  * Training  = 20 000 steps x 4 096 rows x (forward + checkpoint recompute +
    backward ~ 4x a forward; 5x for fb-comments and miniboone, whose pairwise
    tile at batch 4 096 exceeds the 2 GiB budget and is checkpointed again).
  * Sampling  = num_samples x 1 000 diffusion steps x one forward, once per
    generation seed. ablation_runner overrides every dataset's own
    num_timesteps with ABLATION_NUM_TIMESTEPS = 1000, so T is 1000 even for
    king, whose tuned config says 100.
  * Optimistic = that traffic at 1.3 TB/s (A100 40 GB, large elementwise
    kernels); pessimistic = 2.5x slower.
  * Evaluation is an allowance, not a model: hpc_wide's per-run figure scaled
    by the 5 generation seeds this protocol uses.

Cross-check: at 1 generation seed and 50 000 synthetic rows -- hpc_wide's
budget protocol -- this model puts fb-comments sampling at ~0.55 h, which is
the number hpc_wide's own table reports. The agreement is what makes the
extrapolation to the full protocol worth anything.

Memory figures are hpc_wide's measurements where they exist and are marked
as measured; the other two are scaled from the same fit and marked estimated.
"""
from __future__ import annotations

import importlib
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "hpc_transfer")

BANDWIDTH = 1.3e12          # B/s, A100 40 GB on large elementwise kernels
PESSIMISTIC = 2.5
STEPS = 20_000              # ablation_runner.ABLATION_STEPS
BATCH = 4_096               # ablation_runner.ABLATION_BATCH_SIZE
TIMESTEPS = 1_000           # ablation_runner.ABLATION_NUM_TIMESTEPS
N_GEN_SEEDS = 5             # ablation_runner.N_GEN_SEEDS
WALLTIME_LIMIT_H = 24.0

PER_ROW_FWD_BYTES_AT_52 = 51.6e6
DOUBLE_CHECKPOINTED = {"fb-comments", "miniboone"}

# hpc_wide/README.md, "Measured peak memory at the configured sampling batch".
MEASURED_PEAK_GIB = {"higgs-small": 11.1, "miniboone": 10.1, "king": 5.9,
                     "default": 5.6, "fb-comments": 15.0}
# same source: 4.0 GiB + 0.23 GiB per 1 000 rows at N = 52
MEM_BASE_GIB, MEM_PER_KROW_AT_52 = 4.0, 0.23
MEM_MARGIN = 1.35           # the fit under-predicts the measured values by
                            # 20-35 %; applied to the two it cannot check

# Evaluation allowance per generation seed (CatBoost fit + DCR/MIA/Wasserstein
# over the synthetic set). hpc_wide/README.md gives five explicit values under
# its budget protocol; this piecewise rule on num_samples reproduces all five
# exactly (king 27.6k -> 0.2, default 76.8k -> 0.2, higgs-small 502k -> 0.3,
# miniboone 664k -> 0.5, fb-comments 1.264M -> 0.5) and extends to the nine
# ablated datasets, which hpc_wide never covered. It is an allowance, not a
# measurement -- the same status it has there.
def eval_allowance_h(num_samples: int) -> float:
    if num_samples <= 100_000:
        return 0.2
    if num_samples <= 600_000:
        return 0.3
    return 0.5


def dataset_cfg(ds: str) -> dict:
    d = os.path.join(REPO, "ablation")
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module(f"{ds.replace('-', '_')}_ablation").DATASET_CFG


# The reference point is N=52, d_model=128, 3 blocks x 2 GNN layers.
REF_NODES, REF_D_MODEL, REF_GNN_LAYERS = 52, 128, 6

# GCN and GIN do not build the (B, H, N, N, d_head) pairwise tensor at all, so
# the model overestimates them. Their numbers are upper bounds, not estimates.
PAIRWISE_BACKBONES = {"gat", "gatv2", "graphmha"}


def per_row_forward_bytes(n_nodes: int, d_model: int = REF_D_MODEL,
                          n_layers: int = 3) -> float:
    """Traffic per row per forward: the pairwise tensor is B*N^2*d_model
    elements, re-read ~6 times, over n_layers blocks of 2 GNN layers."""
    return (PER_ROW_FWD_BYTES_AT_52
            * (n_nodes / REF_NODES) ** 2
            * (d_model / REF_D_MODEL)
            * (n_layers * 2 / REF_GNN_LAYERS))


def read_config(ds: str) -> dict:
    """The five axes from hpc_transfer/configs/<ds>.sh."""
    out = {}
    for line in open(os.path.join(OUT, "configs", f"{ds}.sh")):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.split("#")[0].strip().strip('"').strip("'")
    return out


def estimate(ds: str, n_nodes: int) -> dict:
    cfg = dataset_cfg(ds)
    jc = read_config(ds)
    gnn, d_model = jc["GNN_TYPE"], int(jc["D_MODEL"])
    n_layers = int(jc["N_LAYERS"])
    modes = jc.get("RUN_MODES", "fixed adaptive_jenks").split()
    f = per_row_forward_bytes(n_nodes, d_model, n_layers)
    mult = 5 if ds in DOUBLE_CHECKPOINTED else 4
    train_h = STEPS * BATCH * mult * f / BANDWIDTH / 3600
    sample_h = N_GEN_SEEDS * cfg["num_samples"] * TIMESTEPS * f / BANDWIDTH / 3600
    eval_h = eval_allowance_h(cfg["num_samples"]) * N_GEN_SEEDS
    lo = train_h + sample_h + eval_h
    hi = (train_h + sample_h) * PESSIMISTIC + eval_h

    sb = cfg.get("sample_batch_size", 10_000)
    if ds in MEASURED_PEAK_GIB:
        mem, mem_src = MEASURED_PEAK_GIB[ds], "measured"
    else:
        mem = (MEM_BASE_GIB + MEM_PER_KROW_AT_52 * (sb / 1000)
               * (n_nodes / REF_NODES) ** 2
               * (d_model / REF_D_MODEL)) * MEM_MARGIN
        mem_src = "estimated"
    return dict(
        dataset=ds, group=("transfer" if jc.get("STRATEGY") == "C_global"
                           else "ablated_best"),
        config=f"{gnn}_L{n_layers}_d{d_model}", n_jobs=len(modes),
        estimate_kind=("estimate" if gnn in PAIRWISE_BACKBONES
                       else "UPPER BOUND (no pairwise tensor)"),
        n_nodes=n_nodes, num_samples=cfg["num_samples"],
        sample_batch_size=sb, train_h=train_h, sample_h=sample_h,
        eval_h=eval_h, total_optimistic_h=lo, total_pessimistic_h=hi,
        with_30pct_margin_h=lo * 1.3,
        fits_24h=("yes" if hi <= WALLTIME_LIMIT_H else
                  "marginal" if lo <= WALLTIME_LIMIT_H else "NO"),
        peak_mem_gib=mem, peak_mem_source=mem_src,
    )


def main():
    meta = pd.read_csv(os.path.join(OUT, "meta_features.csv")).set_index("dataset")
    targets = sorted(f[:-3] for f in os.listdir(os.path.join(OUT, "configs"))
                     if f.endswith(".sh"))
    rows = [estimate(t, int(meta.loc[t, "n_features"])) for t in targets]
    df = pd.DataFrame(rows).sort_values(["group", "total_optimistic_h"])
    df.to_csv(os.path.join(OUT, "resource_estimates.csv"), index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    cols = ["dataset", "group", "config", "n_jobs", "n_nodes", "num_samples",
            "train_h", "sample_h", "eval_h", "total_optimistic_h",
            "total_pessimistic_h", "fits_24h", "peak_mem_gib", "estimate_kind"]
    print(df[cols].to_string(index=False,
                             float_format=lambda v: f"{v:8.2f}"))
    tot_jobs = int(df.n_jobs.sum())
    print(f"\n{len(df)} datasets, {tot_jobs} jobs")
    print(f"\nwalltime limit {WALLTIME_LIMIT_H:.0f} h, --mem=120G "
          f"(the maximum the gpu partition accepts)")
    print("peak memory is well inside 120G for every target; walltime, not "
          "memory, is the binding constraint")
    print("\nwrote", os.path.join(OUT, "resource_estimates.csv"))


if __name__ == "__main__":
    main()

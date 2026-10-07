#!/usr/bin/env python3
"""
Parameter counts of every Route-A arm on every dataset, from the real data
shapes (TabDiff's own TabDiffDataset -> d_numerical, categories; +1 mask
class per column as main.py:159), without training anything.

    python tabdiff_gnn/params.py            # table to stdout, params.csv next to the jobs

Counts the denoiser (what replaces UniModMLP) and, separately, the learnable
schedules (d_num + n_cat, identical across arms); TabDiff's own print
(main.py:223) is their sum.
"""
from __future__ import annotations

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402


def shapes(target: str):
    """(d_numerical, categories without mask, n_train) exactly as TabDiff derives them."""
    from entry import _tabdiff_imports
    TabDiffDataset = _tabdiff_imports()[-1]
    files = C.data_files(target)
    info = json.load(open(files["info"]))
    ds = TabDiffDataset(C.tabdiff_dataname(target), files["dir"], info, isTrain=True,
                        dequant_dist="none", int_dequant_factor=0)
    return int(ds.d_numerical), [int(k) for k in ds.categories], int(info["train_num"])


def counts(target: str) -> dict:
    import torch
    from denoiser import build_denoiser, count_params, param_breakdown
    from entry import load_raw_config
    ds = C.parse_target(target)
    d_num, cats, n = shapes(target)
    raw = load_raw_config()
    up = dict(raw["unimodmlp_params"], d_numerical=d_num, categories=[k + 1 for k in cats])
    out = {"target": target, "d_numerical": d_num, "n_cat": len(cats), "categories": cats,
           "max_cardinality": max(cats) if cats else 0, "nodes": d_num + len(cats), "n_train": n,
           "schedules": d_num + len(cats)}
    torch.manual_seed(0)
    tf = build_denoiser("transformer", up)
    out["transformer"] = count_params(tf)
    out["transformer_blocks"] = param_breakdown(tf)
    g, _ = C.gnn_config(ds)
    m = build_denoiser("attngnn", up, {**g, **C.GNN_FIXED}, dim_t=C.DIM_T)
    out["attngnn"] = count_params(m)
    out["attngnn_blocks"] = param_breakdown(m)
    out["attngnn_config"] = g
    out["ratio"] = out["attngnn"] / out["transformer"]
    return out


def main() -> int:
    rows = [counts(C.target_name(ds)) for ds in C.ORDER]
    print(f"{'dataset':10s} {'N':>3s} {'d_num':>5s} {'n_cat':>5s} {'maxK':>5s} {'n_train':>8s} "
          f"{'config (d256)':28s} {'AttnGNN':>10s} {'TabDiff':>11s} {'ratio':>6s}")
    for r in rows:
        ds = C.parse_target(r["target"])
        g = r["attngnn_config"]
        cfg = f"{g['gnn_type']} L{g['n_layers']} k{g['top_k']} h{g['n_heads']}"
        print(f"{ds:10s} {r['nodes']:3d} {r['d_numerical']:5d} {r['n_cat']:5d} {r['max_cardinality']:5d} "
              f"{r['n_train']:8d} {cfg:28s} {r['attngnn']:10,d} {r['transformer']:11,d} {r['ratio']:6.3f}")
    out = os.path.join(C.REPO, "hpc_transfer", "jobs_tabdiff_gnn", "params.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "nodes", "d_numerical", "n_cat", "max_cardinality", "n_train", "schedules",
                    "transformer", "attngnn", "ratio"])
        for r in rows:
            w.writerow([C.parse_target(r["target"]), r["nodes"], r["d_numerical"], r["n_cat"],
                        r["max_cardinality"], r["n_train"], r["schedules"], r["transformer"], r["attngnn"],
                        round(r["ratio"], 4)])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

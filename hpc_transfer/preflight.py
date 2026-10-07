#!/usr/bin/env python3
"""
Preflight for the transfer jobs -- everything that can be checked without a GPU
and without the cluster environment.

Per target dataset:
  1. hpc_transfer/configs/<ds>.sh parses and sets all six variables.
  2. The entry point ablation/<ds>_ablation.py exists and its DATASET_CFG loads
     through the repo's own convention.
  3. DATASET_CFG's num_numerical_features matches the actual width of
     data/<ds>/X_num_train.npy. This is the check that matters most: a mismatch
     here is what made every `default` run crash at sampling, AFTER the full
     20 000 training steps, because sample.py splits the generated matrix at
     that index (see hpc_wide/README.md).
  4. d_model is divisible by n_heads.
  5. The node count the denoiser will build matches the meta-features.
  6. A real GraphAwareDenoiser is constructed at the configured size with the
     dataset's true cat_sizes, and a forward + backward pass is run on a small
     batch. This is what catches shape and dtype errors from the new column
     counts and cardinalities before a job is ever submitted.
  7. The output directory name the job will use matches exp_dir_name().
  8. Both job scripts exist for the dataset -- the fixed baseline and the
     adaptive-Jenks run -- and each sets the TOPK_MODE it claims.

Exits non-zero if any check fails.
"""
from __future__ import annotations

import importlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "hpc_transfer")
BATCH = 8   # a real forward/backward, small enough to run anywhere


def parse_config(path: str) -> dict:
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.split("#")[0].strip().strip('"').strip("'")
    return out


def main() -> int:
    sys.path.insert(0, os.path.join(REPO, "ablation"))
    sys.path.insert(0, REPO)
    from ablation_runner import _setup_zero_rtdl_stubs, exp_dir_name
    _setup_zero_rtdl_stubs()
    import torch
    from tab_ddpm.graph_denoiser import GraphAwareDenoiser

    targets = sorted(f[:-3] for f in os.listdir(os.path.join(OUT, "configs"))
                     if f.endswith(".sh"))
    meta = {}
    mf = os.path.join(OUT, "meta_features.csv")
    if os.path.exists(mf):
        import csv
        for r in csv.DictReader(open(mf)):
            meta[r["dataset"]] = r

    failures = []
    print(f"{'dataset':12s} {'config':8s} {'entry':6s} {'width':6s} "
          f"{'heads':6s} {'nodes':6s} {'fwd/bwd':8s} {'dir':8s} {'jobs':6s}")
    for ds in targets:
        marks, fail = [], []

        cfg_path = os.path.join(OUT, "configs", f"{ds}.sh")
        c = parse_config(cfg_path)
        need = ["GNN_TYPE", "N_LAYERS", "D_MODEL", "TOP_K", "N_HEADS",
                "USE_ATTENTION"]
        ok_cfg = all(k in c for k in need)
        marks.append("ok" if ok_cfg else "FAIL")
        if not ok_cfg:
            fail.append(f"{ds}: config missing {[k for k in need if k not in c]}")
            print(f"{ds:12s} FAIL"); failures += fail; continue

        gnn, nl, dm = c["GNN_TYPE"], int(c["N_LAYERS"]), int(c["D_MODEL"])
        tk, nh, ua = int(c["TOP_K"]), int(c["N_HEADS"]), int(c["USE_ATTENTION"])

        entry = os.path.join(REPO, "ablation", f"{ds.replace('-', '_')}_ablation.py")
        ok_entry = os.path.exists(entry)
        marks.append("ok" if ok_entry else "FAIL")
        if not ok_entry:
            fail.append(f"{ds}: no entry point {entry}")
            print(f"{ds:12s} " + " ".join(marks)); failures += fail; continue

        dcfg = importlib.import_module(
            f"{ds.replace('-', '_')}_ablation").DATASET_CFG
        info = json.load(open(os.path.join(REPO, "data", ds, "info.json")))
        task = info["task_type"]

        Xn = np.load(os.path.join(REPO, "data", ds, "X_num_train.npy"),
                     allow_pickle=True)
        ok_width = int(dcfg["num_numerical_features"]) == Xn.shape[1]
        marks.append("ok" if ok_width else "FAIL")
        if not ok_width:
            fail.append(f"{ds}: num_numerical_features="
                        f"{dcfg['num_numerical_features']} but "
                        f"X_num_train has {Xn.shape[1]} columns")

        ok_heads = (dm % nh == 0)
        marks.append("ok" if ok_heads else "FAIL")
        if not ok_heads:
            fail.append(f"{ds}: d_model {dm} not divisible by n_heads {nh}")

        d_num = int(dcfg["num_numerical_features"])
        if task == "regression" and not dcfg["is_y_cond"]:
            d_num += 1
        cat_path = os.path.join(REPO, "data", ds, "X_cat_train.npy")
        if os.path.exists(cat_path):
            Xc = np.load(cat_path, allow_pickle=True)
            cat_sizes = [len(set(Xc[:, i].tolist())) for i in range(Xc.shape[1])]
        else:
            cat_sizes = []
        cat_sizes = [k for k in cat_sizes if k > 0]
        n_nodes = d_num + len(cat_sizes)
        ok_nodes = (ds not in meta) or (n_nodes == int(meta[ds]["n_features"]))
        marks.append("ok" if ok_nodes else "FAIL")
        if not ok_nodes:
            fail.append(f"{ds}: node count {n_nodes} != meta_features "
                        f"{meta[ds]['n_features']}")

        d_in = int(sum(cat_sizes)) + d_num
        try:
            den = GraphAwareDenoiser(
                d_in=d_in, num_classes=dcfg["num_classes"],
                is_y_cond=dcfg["is_y_cond"], d_num=d_num, cat_sizes=cat_sizes,
                d_model=dm, n_layers=nl, n_heads=nh, graph_mode="dynamic",
                top_k=tk, gnn_type=gnn, dropout=0.0, use_attention=bool(ua))
            x = torch.randn(BATCH, d_in)
            t = torch.randint(0, 1000, (BATCH,))
            y = (torch.randint(0, max(int(dcfg["num_classes"]), 1), (BATCH,))
                 if dcfg["is_y_cond"] else None)
            out = den(x, t, y) if y is not None else den(x, t)
            assert out.shape == (BATCH, d_in), f"{out.shape} != {(BATCH, d_in)}"
            out.sum().backward()
            assert any(p.grad is not None for p in den.parameters()), "no grads"
            ok_fwd = True
        except Exception as exc:                       # noqa: BLE001
            ok_fwd = False
            fail.append(f"{ds}: forward/backward failed: {type(exc).__name__}: {exc}")
        marks.append("ok" if ok_fwd else "FAIL")

        # one job script per mode this config declares in RUN_MODES
        modes = c.get("RUN_MODES", "fixed adaptive_jenks").split()
        is_target = c.get("STRATEGY") == "C_global"
        want_jobs = ({"fixed": f"{ds}_transfer.sh",
                      "adaptive_jenks": f"{ds}_transfer_jenks.sh",
                      "adaptive_entmax15": f"{ds}_entmax.sh"} if is_target
                     else {"adaptive_jenks": f"{ds}_jenks.sh",
                           "adaptive_entmax15": f"{ds}_entmax.sh"})
        ok_jobs = True
        for mode in modes:
            if mode not in want_jobs:
                ok_jobs = False
                fail.append(f"{ds}: RUN_MODES lists '{mode}' with no job script")
                continue
            jp = os.path.join(OUT, "jobs", want_jobs[mode])
            if not os.path.exists(jp):
                ok_jobs = False
                fail.append(f"{ds}: missing job script {jp}")
                continue
            body = open(jp).read()
            if f'TOPK_MODE="${{TOPK_MODE:-{mode}}}"' not in body:
                ok_jobs = False
                fail.append(f"{ds}: {os.path.basename(jp)} does not set "
                            f"TOPK_MODE default to {mode}")

        want = exp_dir_name(gnn, nl, dm, tk, nh, bool(ua))
        ok_dir = want == (f"gnn_{gnn}_layers{nl}_dim{dm}_topk{tk}_heads{nh}"
                          + ("_noattn" if ua == 0 else ""))
        marks.append("ok" if ok_dir else "FAIL")
        if not ok_dir:
            fail.append(f"{ds}: dir name mismatch, exp_dir_name gives {want}")

        marks.append("ok" if ok_jobs else "FAIL")
        line = f"{ds:12s} " + " ".join(f"{m:<7s}" for m in marks)
        if ok_fwd:
            line += (f"  N={n_nodes} d_in={d_in} "
                     f"params={sum(p.numel() for p in den.parameters()):,}")
        print(line)
        failures += fail

    # --- the 9 ablated datasets must pair 1:1 with a fixed baseline --------
    import glob as _glob
    baselines = {}
    for f in _glob.glob(os.path.join(REPO, "tabddpm_ablation", "*.json")):
        for r in json.load(open(f)):
            baselines.setdefault(r["dataset"], set()).add(r["exp_name"])
    print()
    for ds in sorted(targets):
        c = parse_config(os.path.join(OUT, "configs", f"{ds}.sh"))
        if c.get("STRATEGY") == "C_global":
            continue   # a new dataset: no baseline to pair with, by definition
        name = (f"gnn_{c['GNN_TYPE']}_layers{c['N_LAYERS']}_dim{c['D_MODEL']}"
                f"_topk{c['TOP_K']}_heads{c['N_HEADS']}"
                + ("_noattn" if int(c["USE_ATTENTION"]) == 0 else ""))
        paired = name in baselines.get(ds, ())
        if not paired:
            failures.append(f"{ds}: adaptive run {name} has no fixed baseline "
                            f"in tabddpm_ablation/ to pair with")
        print(f"pairing {ds:12s} {name:46s} "
              f"{'ok' if paired else 'FAIL'}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"all {len(targets)} datasets passed preflight")
    return 0


if __name__ == "__main__":
    sys.exit(main())

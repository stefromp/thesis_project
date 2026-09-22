#!/usr/bin/env python3
"""Run list for the many-feature adaptive top-k budget study.

Question: how many neighbours does each adaptive rule keep per node when the
feature graph is large? The manifest datasets have 5-11 nodes (gesture 32),
which leaves little room between k = 1 and k = N - 1. The datasets here have
21-52 nodes.

No ablation is run on these datasets: each gets ONE configuration, GATv2 only,
taken from the GATv2 best scenario of the swept datasets at a comparable size
(see CONFIGS below). None of them has a fixed-top-k baseline sweep, so the run
list is defined here rather than replayed from tabddpm_ablation/.

The graph size N is computed exactly as scripts/train.py builds it: one node
per numerical column (plus y for regression without y conditioning) and one
per categorical column -- read from the .npy headers, not info.json.

    python3 hpc_wide/wide_manifest.py --repo_root R --dataset miniboone
        -> gnn_type  n_layers  d_model  top_k  n_heads  use_attention   (tab-separated)
    python3 hpc_wide/wide_manifest.py --repo_root R --summary
    python3 hpc_wide/wide_manifest.py --repo_root R --n_nodes miniboone
    python3 hpc_wide/wide_manifest.py --repo_root R --check miniboone

Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import sys

# Datasets with >= 20 graph nodes and an ablation/<ds>_ablation.py entry point.
# gesture (32 nodes) is already covered by hpc/ and hpc_entmax/; news (58) has
# no TabDDPM entry point.
DATASETS = ("fb-comments", "miniboone", "higgs-small", "default", "king")

# One GATv2 configuration per dataset -- no ablation. Chosen from the best
# GATv2 run of each swept dataset (tabddpm_ablation/, best score.mean), by
# training-set size:
#
#     rows    dataset      GATv2 runs  best GATv2
#      491    diabetes          8      L3 d64  h4
#      856    insurance         8      L2 d64  h4
#     2672    abalone           8      L3 d64  h4
#     3096    wilt              4      L2 d64  h8
#     6400    churn2            6      L2 d128 h4
#    13209    california        2      L3 d128 h8   (L3 beat L2 by 0.014, ~7x the seed std)
#
# Width grows with size (d64 up to ~3k rows, d128 from ~6k), and at the largest
# swept size (california, 13.2k rows) 3 blocks with 8 heads won. Every dataset
# here has >= 13.8k rows, so each takes that best scenario. Nothing wider is
# used for the 60k+ row datasets: no GATv2 run supports it, and it would cost
# ~2x. (gesture's single GATv2 run is not a sweep and is left out.)
# top_k only matters during warm-up (off) and in the directory name.
GATV2_LARGE = ("gatv2", 3, 128, 5, 8, False)   # gnn_type, n_layers, d_model, top_k, n_heads, use_attention
CONFIGS = {
    # dataset       train rows  N
    "fb-comments": GATV2_LARGE,   # 157 638   52
    "miniboone":   GATV2_LARGE,   #  83 240   50
    "higgs-small": GATV2_LARGE,   #  62 751   28
    "default":     GATV2_LARGE,   #  19 200   23
    "king":        GATV2_LARGE,   #  13 832   21
}
assert set(CONFIGS) == set(DATASETS)


def entry_module(dataset: str) -> str:
    """ablation/<module>.py for a dataset: fb-comments -> fb_comments_ablation."""
    return f"{dataset.replace('-', '_')}_ablation"


def _dataset_cfg(repo_root: str, dataset: str):
    ablation_dir = os.path.join(repo_root, "ablation")
    if ablation_dir not in sys.path:
        sys.path.insert(0, ablation_dir)
    module = importlib.import_module(entry_module(dataset))
    return module.DATASET_NAME, module.DATASET_CFG


def _npy_shape(path: str) -> tuple:
    """Shape from a .npy header, without loading (object arrays included)."""
    with open(path, "rb") as fh:
        if fh.read(6) != b"\x93NUMPY":
            raise ValueError(f"not a .npy file: {path}")
        major = fh.read(2)[0]
        header_len = int.from_bytes(fh.read(2 if major == 1 else 4), "little")
        header = ast.literal_eval(fh.read(header_len).decode("latin1"))
    return tuple(header["shape"])


def _widths(repo_root: str, dataset: str):
    _, cfg = _dataset_cfg(repo_root, dataset)
    data = os.path.join(repo_root, cfg["real_data_path"])
    n_num = _npy_shape(os.path.join(data, "X_num_train.npy"))[1]
    cat = os.path.join(data, "X_cat_train.npy")
    n_cat = _npy_shape(cat)[1] if os.path.exists(cat) else 0
    with open(os.path.join(data, "info.json")) as fh:
        task_type = json.load(fh)["task_type"]
    y_node = int(task_type == "regression" and not cfg["is_y_cond"])
    return n_num, n_cat, y_node, cfg


def n_nodes(repo_root: str, dataset: str) -> int:
    n_num, n_cat, y_node, _ = _widths(repo_root, dataset)
    return n_num + y_node + n_cat


def sampling_config(repo_root: str, dataset: str, cap=None):
    """(num_samples_config, num_samples_used, sample_batch_size_used): the
    dataset's own values, or those hpc_wide/run_combo.py uses under a cap."""
    _, cfg = _dataset_cfg(repo_root, dataset)
    configured = int(cfg["num_samples"])
    used = configured if cap is None else min(configured, int(cap))
    return configured, used, min(int(cfg["sample_batch_size"]), used)


def check_config(repo_root: str, dataset: str) -> list:
    """Problems that would only surface after training (empty list = OK)."""
    problems = []
    try:
        name, _ = _dataset_cfg(repo_root, dataset)
        if name != dataset:
            problems.append(f"{entry_module(dataset)}.DATASET_NAME is {name!r}, expected {dataset!r}")
        n_num, _, _, cfg = _widths(repo_root, dataset)
        if int(cfg["num_numerical_features"]) != n_num:
            # sample.py splits the generated matrix at this index: a mismatch
            # crashes (or silently garbles) sampling after the whole training run.
            problems.append(f"ablation/{entry_module(dataset)}.py has num_numerical_features="
                            f"{cfg['num_numerical_features']} but X_num_train has {n_num} columns")
    except Exception as exc:
        problems.append(f"cannot read the config/data of {dataset}: {type(exc).__name__}: {exc}")
    return problems


def exp_dir_name(combo) -> str:
    """Same as ablation_runner.exp_dir_name()."""
    gnn_type, n_layers, d_model, top_k, n_heads, use_attention = combo
    suffix = "" if use_attention else "_noattn"
    return f"gnn_{gnn_type}_layers{n_layers}_dim{d_model}_topk{top_k}_heads{n_heads}{suffix}"


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", required=True)
    p.add_argument("--dataset", help="Emit the tab-separated run list for this dataset")
    p.add_argument("--n_nodes", metavar="DATASET", help="Print the graph size N")
    p.add_argument("--sampling", metavar="DATASET",
                   help="Print num_samples_config num_samples_used sample_batch_size_used")
    p.add_argument("--cap", type=int, default=None, help="with --sampling: the row cap")
    p.add_argument("--check", metavar="DATASET", help="Config/data consistency; exit 1 on a problem")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()
    repo_root = os.path.abspath(args.repo_root)

    if args.n_nodes:
        print(n_nodes(repo_root, args.n_nodes))
        return 0

    if args.sampling:
        print(*sampling_config(repo_root, args.sampling, args.cap))
        return 0

    if args.check:
        problems = check_config(repo_root, args.check)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0

    if args.summary:
        print(f"{'dataset':12s} {'N':>3s}  {'num+y+cat':12s} {'k range':9s} {'run (one per mode)':44s} config")
        for dataset in DATASETS:
            n_num, n_cat, y_node, _ = _widths(repo_root, dataset)
            N = n_num + y_node + n_cat
            status = "; ".join(check_config(repo_root, dataset)) or "OK"
            print(f"{dataset:12s} {N:3d}  {f'{n_num}+{y_node}+{n_cat}':12s} {f'1..{N - 1}':9s} "
                  f"{exp_dir_name(CONFIGS[dataset]):44s} {status}")
        return 0

    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"error: {args.dataset!r} is not in the wide study {DATASETS}", file=sys.stderr)
            return 1
        gnn_type, n_layers, d_model, top_k, n_heads, use_attention = CONFIGS[args.dataset]
        print("\t".join([gnn_type, str(n_layers), str(d_model), str(top_k),
                         str(n_heads), "1" if use_attention else "0"]))
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(_main())

#!/usr/bin/env python3
"""Run list for the density-curve study (Stage 1).

For each dataset, ONE fixed GATv2 architecture is run under

    fixed top-k at a grid of densities   -> the curve "metric vs k"
    adaptive_jenks / _sparsemax / _entmax15 -> three points at their achieved mean k

so that an adaptive rule can be judged against fixed top-k AT THE SAME DENSITY:
below the curve (Wasserstein) = adaptivity adds something beyond density; on the
curve = the rule is just choosing a density.

adaptive_jenks is PLAIN Jenks: the 2-class split, strong class kept, every
split accepted (no acceptance test on the strong-class mean) and no
hyperparameters beyond the 2 classes -- k_min stays 1 and never binds.

Every condition uses the same seed protocol (training seed 0, generation seeds
0-4, classifier seeds 0-9: ablation_runner's module defaults), which is what
makes the comparison paired. All 3 adaptive modes are rerun on all 5 datasets:
no existing run is reused, so paired and unpaired points are never mixed.

N is computed exactly as scripts/train.py builds the graph (one node per
numerical column, plus y for regression without y conditioning, plus one per
categorical column), from the .npy headers -- the same logic as
hpc_wide/wide_manifest.py, repeated here so this directory is self-contained.

    python3 hpc_density/density_manifest.py --repo_root R --summary
    python3 hpc_density/density_manifest.py --repo_root R --check wilt
    python3 hpc_density/density_manifest.py --repo_root R --conditions wilt
    python3 hpc_density/density_manifest.py --repo_root R --resolve wilt fixed_k2
        -> topk_mode  top_k  gnn_type  n_layers  d_model  n_heads  use_attention  (tab-separated)

Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import sys

DATASETS = ("wilt", "abalone", "churn2", "default", "gesture")

# ---------------------------------------------------------------------------
# Architectures: GATv2 only, chosen by the rule fixed BEFORE the study --
# the best GATv2 run of each dataset by score.mean in tabddpm_ablation/.
# Not chosen by Wasserstein: that is the metric that judges the study.
#
#   wilt     4 GATv2 runs. L2 d64 h8 0.9809 +/- 0.0019 (best). All four widths
#            are within one std on score (d32 0.9808, d512 0.9798, d128 0.9790),
#            and d512 was clearly better on Wasserstein (0.0125 vs 0.0233 for
#            d64). Kept d64: the rule, not the study metric, decides.
#   abalone  8 GATv2 runs. L3 d64 h4 -2.0098 +/- 0.0185 (best, at k=3) is TIED
#            with L2 d64 h4 -2.0120 +/- 0.0108 (within one std). Kept the
#            rule's winner, L3 d64.
#   churn2   6 GATv2 runs. L2 d128 h4 0.8583 +/- 0.0028 (best); L3 d128 h4
#            0.8581 +/- 0.0024 is tied. Kept L2 d128.
#   default  no baseline sweep: the hpc_wide configuration (GATv2 best
#            scenario of the largest swept dataset, california).
#   gesture  1 GATv2 run only: L2 d32 h4 0.5311 +/- 0.0050. Kept as the only
#            GATv2 evidence (the overall best was GAT L3 d128, 0.6019).
# ---------------------------------------------------------------------------
ARCHS = {
    #            gnn_type n_layers d_model n_heads use_attention
    "wilt":    ("gatv2", 2,  64, 8, False),
    "abalone": ("gatv2", 3,  64, 4, False),
    "churn2":  ("gatv2", 2, 128, 4, False),
    "default": ("gatv2", 3, 128, 8, False),
    "gesture": ("gatv2", 2,  32, 4, False),
}

# top_k written into the ADAPTIVE runs' directory names. Unused by the adaptive
# modes (warm-up is 0); it is the representative top_k of the earlier adaptive
# runs of the same architecture, so the names line up with them.
ADAPTIVE_TOP_K = {"wilt": 3, "abalone": 3, "churn2": 5, "default": 5, "gesture": 5}

# ---------------------------------------------------------------------------
# Fixed-k grid. Targets k/(N-1) = 10 %, 25 %, 50 %, plus dense:
#   * round half up (floor(x + 0.5)) -- churn2's 25 % target is 2.5 -> 3
#     (banker's rounding would give 2);
#   * minimum 1; a value >= N-1 is dense;
#   * "dense" is --top_k 0 (DynamicAdjacency treats top_k <= 0 or >= N-1 as
#     dense), plotted at k = N-1;
#   * deduplicate; if fewer than 4 distinct values result (wilt), use every k
#     in 1..N-2 plus dense instead;
#   * EXTRA_K adds values on top of the rule: k=1 for default and gesture,
#     whose adaptive points (Jenks ~1.1-2.7) would otherwise fall below the
#     smallest grid value, i.e. outside the curve.
# ---------------------------------------------------------------------------
TARGET_FRACTIONS = (0.10, 0.25, 0.50)
EXTRA_K = {"default": (1,), "gesture": (1,)}

ADAPTIVE_MODES = ("adaptive_jenks", "adaptive_sparsemax", "adaptive_entmax15")


def entry_module(dataset: str) -> str:
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


def check_config(repo_root: str, dataset: str) -> list:
    """Problems that would only surface after training (empty list = OK)."""
    problems = []
    try:
        name, _ = _dataset_cfg(repo_root, dataset)
        if name != dataset:
            problems.append(f"{entry_module(dataset)}.DATASET_NAME is {name!r}, expected {dataset!r}")
        n_num, _, _, cfg = _widths(repo_root, dataset)
        if int(cfg["num_numerical_features"]) != n_num:
            problems.append(f"ablation/{entry_module(dataset)}.py has num_numerical_features="
                            f"{cfg['num_numerical_features']} but X_num_train has {n_num} columns")
    except Exception as exc:
        problems.append(f"cannot read the config/data of {dataset}: {type(exc).__name__}: {exc}")
    return problems


def k_grid(N: int, dataset: str) -> list:
    """Sparse fixed-k values (sorted ints, 1 <= k <= N-2), dense excluded."""
    half_up = lambda x: int(math.floor(x + 0.5))
    ks = {max(1, half_up(f * (N - 1))) for f in TARGET_FRACTIONS}
    ks = {k for k in ks if k < N - 1}                  # k >= N-1 is dense
    if len(ks) + 1 < 4:                                # + dense
        ks = set(range(1, N - 1))
    ks |= {k for k in EXTRA_K.get(dataset, ()) if 1 <= k < N - 1}
    return sorted(ks)


def conditions(repo_root: str, dataset: str) -> list:
    N = n_nodes(repo_root, dataset)
    return ([f"fixed_k{k}" for k in k_grid(N, dataset)] + ["fixed_dense"]
            + list(ADAPTIVE_MODES))


def resolve(repo_root: str, dataset: str, condition: str) -> dict:
    """What one condition runs: topk_mode, top_k, architecture, and the k the
    fixed conditions keep (None for adaptive: measured from topk_stats.csv)."""
    if condition not in conditions(repo_root, dataset):
        raise ValueError(f"{condition!r} is not a condition of {dataset}: "
                         f"{conditions(repo_root, dataset)}")
    N = n_nodes(repo_root, dataset)
    gnn_type, n_layers, d_model, n_heads, use_attention = ARCHS[dataset]
    if condition == "fixed_dense":
        mode, top_k, k = "fixed", 0, N - 1
    elif condition.startswith("fixed_k"):
        k = int(condition[len("fixed_k"):])
        mode, top_k = "fixed", k
    else:
        mode, top_k, k = condition, ADAPTIVE_TOP_K[dataset], None
    return {"dataset": dataset, "condition": condition, "topk_mode": mode, "top_k": top_k,
            "k": k, "n_nodes": N, "gnn_type": gnn_type, "n_layers": n_layers,
            "d_model": d_model, "n_heads": n_heads, "use_attention": use_attention}


def exp_dir_name(r: dict) -> str:
    """Same as ablation_runner.exp_dir_name()."""
    suffix = "" if r["use_attention"] else "_noattn"
    return (f"gnn_{r['gnn_type']}_layers{r['n_layers']}_dim{r['d_model']}"
            f"_topk{r['top_k']}_heads{r['n_heads']}{suffix}")


def run_subdir(r: dict) -> str:
    """ablation_runner's subtree: 'ablation' for fixed, 'ablation_<mode>' otherwise."""
    return "ablation" if r["topk_mode"] == "fixed" else f"ablation_{r['topk_mode']}"


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", required=True)
    p.add_argument("--summary", action="store_true")
    p.add_argument("--check", metavar="DATASET", help="config/data consistency; exit 1 on a problem")
    p.add_argument("--n_nodes", metavar="DATASET")
    p.add_argument("--conditions", metavar="DATASET", help="one condition per line")
    p.add_argument("--resolve", nargs=2, metavar=("DATASET", "CONDITION"),
                   help="tab-separated: topk_mode top_k gnn_type n_layers d_model n_heads use_attention")
    p.add_argument("--run_dir", nargs=2, metavar=("DATASET", "CONDITION"),
                   help="print <subtree>/<exp dir name> of a condition")
    args = p.parse_args()
    repo_root = os.path.abspath(args.repo_root)

    if args.check:
        problems = check_config(repo_root, args.check)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    if args.n_nodes:
        print(n_nodes(repo_root, args.n_nodes))
        return 0
    if args.conditions:
        print("\n".join(conditions(repo_root, args.conditions)))
        return 0
    if args.resolve:
        try:
            r = resolve(repo_root, *args.resolve)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("\t".join([r["topk_mode"], str(r["top_k"]), r["gnn_type"], str(r["n_layers"]),
                         str(r["d_model"]), str(r["n_heads"]), "1" if r["use_attention"] else "0"]))
        return 0
    if args.run_dir:
        r = resolve(repo_root, *args.run_dir)
        print(f"{run_subdir(r)}/{exp_dir_name(r)}")
        return 0
    if args.summary:
        total = 0
        print(f"{'dataset':8s} {'N':>3s}  {'architecture':24s} {'fixed k (dense = N-1)':26s} runs  config")
        for ds in DATASETS:
            N = n_nodes(repo_root, ds)
            g, L, d, h, att = ARCHS[ds]
            grid = ", ".join(str(k) for k in k_grid(N, ds)) + f", dense({N - 1})"
            n = len(conditions(repo_root, ds))
            total += n
            status = "; ".join(check_config(repo_root, ds)) or "OK"
            print(f"{ds:8s} {N:3d}  {f'{g} L{L} d{d} h{h}':24s} {grid:26s} {n:4d}  {status}")
        print(f"{'TOTAL':8s} {'':54s} {total:4d}   (fixed grid + {len(ADAPTIVE_MODES)} adaptive per dataset)")
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(_main())

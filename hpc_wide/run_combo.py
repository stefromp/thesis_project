#!/usr/bin/env python3
"""Run ONE hpc_wide combination through the dataset's own entry point, with an
optional in-memory cap on the number of synthetic rows.

Equivalent to `python ablation/<ds>_ablation.py <args>`: same parser
(ablation_runner.make_parser), same single-combination tuple, same
run_ablation call. The only difference, and only when --max_num_samples is
given, is that this process uses a COPY of DATASET_CFG with

    num_samples       = min(num_samples, max_num_samples)
    sample_batch_size = min(sample_batch_size, num_samples)   # one batch, no rows generated to be discarded

No file under ablation/ is read differently or modified, and without
--max_num_samples the dataset's own config is used unchanged (full protocol).

    python hpc_wide/run_combo.py --dataset fb-comments --max_num_samples 50000 \\
        --gnn_type gatv2 --n_layers 3 ... (every other flag goes to make_parser)
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--dataset", required=True)
    own.add_argument("--max_num_samples", type=int, default=None)
    ours, rest = own.parse_known_args()

    for sub in ("ablation", "hpc_wide"):
        path = os.path.join(REPO_ROOT, sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import wide_manifest
    module = importlib.import_module(wide_manifest.entry_module(ours.dataset))
    from ablation_runner import make_parser, run_ablation

    cfg = dict(module.DATASET_CFG)
    if ours.max_num_samples is not None:
        cfg["num_samples"] = min(int(cfg["num_samples"]), ours.max_num_samples)
        cfg["sample_batch_size"] = min(int(cfg["sample_batch_size"]), cfg["num_samples"])
    print(f"[hpc_wide] {module.DATASET_NAME}: num_samples {module.DATASET_CFG['num_samples']} -> "
          f"{cfg['num_samples']}, sample_batch_size {module.DATASET_CFG['sample_batch_size']} -> "
          f"{cfg['sample_batch_size']}", flush=True)

    args = make_parser(module.DATASET_NAME).parse_args(rest)
    combo = (args.gnn_type, args.n_layers, args.d_model, args.top_k, args.n_heads)
    if any(v is None for v in combo):
        print("error: run_combo.py runs exactly one combination -- pass --gnn_type "
              "--n_layers --d_model --top_k --n_heads", file=sys.stderr)
        return 2
    run_ablation(module.DATASET_NAME, cfg, args, single_combo=combo)
    return 0


if __name__ == "__main__":
    sys.exit(main())

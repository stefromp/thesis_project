#!/usr/bin/env python3
"""Read the finished ablation results in ``tabddpm_ablation/`` as a job manifest.

The HPC re-run exists to add the TabDiff-paper fidelity metrics (Shape, Trend,
C2ST, DCR, alpha/beta) to combinations that were already evaluated with the
TabDDPM protocol. Those combinations are *not* a clean cartesian product:

  * each dataset was swept over a different grid (abalone over d_model 32/64,
    wilt over 32/64/128/512, ...),
  * some products are only partially covered (abalone has 19 of 24),
  * ``gesture`` contains a ``gat`` run that is no longer in ``GNN_TYPES``.

Rebuilding a grid from the union of observed values would therefore invent
combinations that were never run. Instead this module replays the exact
combination list found in each summary file, so the array job reproduces the
existing study one-for-one.

Note the filename is not the dataset name — ``churn_ablation.json`` holds
dataset ``churn2`` — so the ``dataset`` field inside each entry is what counts.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

DEFAULT_RESULTS_DIR = "tabddpm_ablation"

# One combination as the rest of the pipeline spells it.
COMBO_FIELDS = ("gnn_type", "n_layers", "d_model", "top_k", "n_heads",
                "use_attention")


def _entry_to_combo(entry: dict) -> Optional[tuple]:
    if any(field not in entry for field in COMBO_FIELDS[:5]):
        return None
    return (
        str(entry["gnn_type"]),
        int(entry["n_layers"]),
        int(entry["d_model"]),
        int(entry["top_k"]),
        int(entry["n_heads"]),
        # Older summaries predate the attention axis; those runs had it on.
        bool(entry.get("use_attention", True)),
    )


def load_manifest(results_dir: str = DEFAULT_RESULTS_DIR,
                  status: Optional[str] = "done") -> dict:
    """Map dataset name -> ordered list of combination tuples.

    ``status`` filters the entries to replay ("done" by default); pass None to
    take every entry regardless of how the original run ended.
    """
    manifest: dict = {}

    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            try:
                entries = json.load(fh)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] {path}: not valid JSON ({exc}) — skipped")
                continue

        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict) or "dataset" not in entry:
                continue
            if status is not None and entry.get("status") != status:
                continue
            combo = _entry_to_combo(entry)
            if combo is None:
                continue
            combos = manifest.setdefault(entry["dataset"], [])
            if combo not in combos:          # same combo may appear in two files
                combos.append(combo)

    return manifest


def load_best(results_dir: str = DEFAULT_RESULTS_DIR) -> dict:
    """Map dataset name -> the single best-scoring combination.

    Ranked by ``metrics.score.mean``, which the TabDDPM evaluator writes for
    every task type and always orients higher-is-better: accuracy / F1 for
    classification, and NEGATIVE RMSE for regression (abalone's best scores
    -2.01, not its R2 of 0.556). Since R2 = 1 - MSE/var with var fixed by the
    test split, ranking on -RMSE and on R2 give the same winner.

    Useful as a smoke test: nine jobs that exercise every code path — both task
    types, all four GNN types, the TabDiff fidelity suite — before committing
    the full manifest to the queue.
    """
    best: dict = {}
    scores: dict = {}

    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            try:
                entries = json.load(fh)
            except json.JSONDecodeError:
                continue
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict) or entry.get("status") != "done":
                continue
            score = (entry.get("metrics") or {}).get("score")
            if not isinstance(score, dict) or "mean" not in score:
                continue
            combo = _entry_to_combo(entry)
            if combo is None:
                continue
            dataset = entry["dataset"]
            if dataset not in scores or score["mean"] > scores[dataset]:
                scores[dataset] = score["mean"]
                best[dataset] = combo

    return {dataset: [combo] for dataset, combo in best.items()}


def load_datasets(results_dir: str = DEFAULT_RESULTS_DIR) -> list:
    """Just the dataset names that have results, ignoring their combinations.

    Used when the run should cover these datasets but sweep the notebook grid
    (the ablation_runner constants) rather than replay exactly what was run
    before — the recorded combinations are a subset of that grid.
    """
    return sorted(load_manifest(results_dir, status=None))


def describe_manifest(manifest: dict) -> str:
    lines = [f"Manifest: {len(manifest)} datasets, "
             f"{sum(len(c) for c in manifest.values())} combinations"]
    for dataset, combos in sorted(manifest.items()):
        axes = {field: sorted({c[i] for c in combos}, key=str)
                for i, field in enumerate(COMBO_FIELDS)}
        lines.append(f"  {dataset:12s} {len(combos):3d} combos  "
                     + "  ".join(f"{k}={v}" for k, v in axes.items()))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    print(describe_manifest(load_manifest(directory)))

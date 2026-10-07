#!/usr/bin/env python3
"""Full-load integrity check of transfer_runs/exp and a coverage table.

    python3 analysis/integrity/validate_runs.py [--skip higgs-small,miniboone]

Every *.csv, *.json, *.npy (and *.pt, if torch is importable) under each
run directory is loaded in full. A CSV whose last byte is not a newline is
flagged as truncated even when pandas parses it. Per generation seed the
X_num / X_cat / y row counts must agree and the numerical arrays be finite.

Writes integrity.csv (one row per file problem; empty = clean) and
coverage.csv (dataset x tree x config) next to this file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "transfer_runs" / "exp"
EXPECTED_GEN, EXPECTED_CLF = 5, 10

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def check_file(p: Path) -> str | None:
    try:
        if p.suffix == ".csv":
            pd.read_csv(p)
            if p.stat().st_size and p.read_bytes()[-1:] not in (b"\n", b"\r"):
                return "csv does not end with newline (truncated?)"
        elif p.suffix == ".json":
            json.load(open(p))
        elif p.suffix == ".npy":
            a = np.load(p, allow_pickle=True)
            _ = a.copy()  # touch every byte
            if a.dtype.kind in "fc" and not np.isfinite(a).all():
                return f"non-finite values ({(~np.isfinite(a)).sum()})"
        elif p.suffix == ".pt" and torch is not None:
            torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"[:200]
    return None


def per_seed_counts(run: Path) -> tuple[int, int, list[int]]:
    files = sorted(run.glob("per_seed_scores.csv")) + sorted(run.glob("per_seed_scores.gen*.csv"))
    if not files:
        return 0, 0, []
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw["clf_key"] = raw["clf_seed"].fillna(-1)
    raw = raw.drop_duplicates(["gen_seed", "clf_key", "metric"], keep="last")
    util = raw[raw.clf_seed.notna() & (raw.metric == "score")]
    gen = raw[raw.clf_seed.isna()]
    n_clf = util.groupby("gen_seed").size()
    has_gen = set(gen[gen.metric == "mia_auc"].gen_seed) & set(gen[gen.metric == "wasserstein_mean"].gen_seed)
    ok = sorted(int(s) for s, n in n_clf.items() if n >= EXPECTED_CLF and s in has_gen)
    return len(ok), int(n_clf[n_clf.index.isin(ok)].min()) if ok else 0, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="higgs-small,miniboone")
    args = ap.parse_args()
    skip = {s for s in args.skip.split(",") if s}

    problems, cov = [], []
    n_files = 0
    for ds_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if ds_dir.name in skip:
            continue
        for tree in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            for f in sorted(tree.glob("*_ablation.json")):
                n_files += 1
                if (e := check_file(f)):
                    problems.append({"file": str(f.relative_to(ROOT)), "problem": e})
            for run in sorted(p for p in tree.glob("gnn_*") if p.is_dir()):
                for f in sorted(run.rglob("*")):
                    if f.is_file() and f.suffix in (".csv", ".json", ".npy", ".pt"):
                        n_files += 1
                        if (e := check_file(f)):
                            problems.append({"file": str(f.relative_to(ROOT)), "problem": e})
                # per-gen-seed sample consistency
                gens = sorted(run.glob("gen_seed_*"))
                rows = []
                for g in gens:
                    n = {}
                    for name in ("X_num_train", "X_cat_train", "y_train"):
                        q = g / f"{name}.npy"
                        if q.exists():
                            try:
                                n[name] = len(np.load(q, allow_pickle=True))
                            except Exception:  # noqa: BLE001
                                n[name] = -1
                    if len(set(n.values())) > 1:
                        problems.append({"file": str(g.relative_to(ROOT)), "problem": f"row mismatch {n}"})
                    if "y_train" not in n:
                        problems.append({"file": str(g.relative_to(ROOT)), "problem": "y_train.npy missing"})
                    rows.append(n.get("y_train", 0))
                meta = json.load(open(run / "run_meta.json")) if (run / "run_meta.json").exists() else {}
                dc = json.load(open(run / "denoiser_config.json")) if (run / "denoiser_config.json").exists() else {}
                n_gen, n_clf, seeds = per_seed_counts(run)
                cat_in = dc.get("cat_input") or meta.get("cat_input")
                cov.append({
                    "dataset": ds_dir.name, "tree": tree.name, "config": run.name,
                    "n_gen_seeds": n_gen, "n_clf_seeds": n_clf, "gen_seeds": ";".join(map(str, seeds)),
                    "n_sample_dirs": len(gens), "sample_rows": ";".join(map(str, rows)),
                    "cat_input": cat_in or "log (inferred: predates option)",
                    "cat_input_source": "denoiser_config.json" if "cat_input" in dc else
                                        ("run_meta.json" if "cat_input" in meta else "default"),
                    "topk_mode": dc.get("topk_mode") or meta.get("topk_mode") or "fixed",
                    "training_seed": meta.get("seed"), "slurm_job": meta.get("slurm_job_id"),
                    "started_utc": meta.get("started_utc"),
                    "has_topk_stats": (run / "topk_stats.csv").exists(),
                })
                # denoiser_config vs run_meta consistency
                for k in ("cat_input", "topk_mode"):
                    if k in dc and k in meta and dc[k] != meta[k]:
                        problems.append({"file": str(run.relative_to(ROOT)),
                                         "problem": f"{k}: denoiser_config={dc[k]} run_meta={meta[k]}"})
    pd.DataFrame(problems, columns=["file", "problem"]).to_csv(HERE / "integrity.csv", index=False)
    pd.DataFrame(cov).to_csv(HERE / "coverage.csv", index=False)
    print(f"files loaded: {n_files}; problems: {len(problems)}")
    for p in problems:
        print("  ", p["file"], "->", p["problem"])
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

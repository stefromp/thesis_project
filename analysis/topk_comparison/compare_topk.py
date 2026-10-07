#!/usr/bin/env python3
"""Compare the three top-k rules (fixed, adaptive_jenks, adaptive_entmax15).

One command rebuilds every output in this directory:

    python3 analysis/topk_comparison/compare_topk.py

Read-only on results. Everything is read from RESULTS_ROOT
(transfer_runs/exp/<dataset>/...) plus three provenance files in hpc_transfer/.
Nothing is imputed: a cell that is missing or incomplete is reported as such.

Where each rule's numbers come from
-----------------------------------
  jenks      <ds>/ablation_adaptive_jenks/<config>/per_seed_scores*.csv
  entmax15   <ds>/ablation_adaptive_entmax15/<config>/per_seed_scores*.csv
  fixed      first match of
               1. <ds>/ablation_fixed_seed0/<config>/per_seed_scores*.csv
                  (ablated datasets: per-seed rerun of the same config at the
                  same TRAINING SEED 0 -- see hpc_fixed_perseed/. It only adds
                  per-seed values; it removes neither the seed winner's curse
                  nor the test-split selection bias)
               2. <ds>/ablation/<config>/per_seed_scores*.csv
                  (transferred datasets)
               3. <ds>/ablation/<ds>_ablation.json, the row for <config>
                  (ablated datasets, training seed 0, SUMMARY ONLY: mean/std,
                  no per-seed values). Also kept as a reference column
                  ("fixed_seed0_ref") once 1. exists.

Two subsets, never mixed silently
---------------------------------
  held-out evaluation  the transferred datasets: configuration chosen without
                       seeing any of that dataset's results. Primary claims.
  development set      the ablated datasets: configuration = argmax of the
                       fixed-mode TEST-split score over the dataset's own grid.
                       In-sample with respect to selection, optimistic
                       (especially for fixed).
  pooled               both, reported for completeness only.

Unit of replication = generation seed. Utility metrics are averaged over the
classifier seeds inside each generation seed first; the per-generation-seed
metrics (Wasserstein, DCR, MIA) are used as they are.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Overridable only for testing; the defaults are the thesis results.
RESULTS_ROOT = Path(os.environ.get("COMPARE_TOPK_RESULTS_ROOT", REPO / "transfer_runs" / "exp"))
OUT = Path(os.environ.get("COMPARE_TOPK_OUT", HERE))
PREDICTIONS_MD = REPO / "hpc_transfer" / "predictions.md"
BEST_CSV = REPO / "hpc_transfer" / "best_per_dataset.csv"
META_CSV = REPO / "hpc_transfer" / "meta_features.csv"
BEST_MANIFEST_PY = REPO / "hpc_transfer" / "best_manifest.py"

EXPECTED_GEN = 5
EXPECTED_CLF = 10
N_BOOT = 10_000
ALPHA = 0.05
RNG_SEED = 0

RULES = ("fixed", "jenks", "entmax15")
RULE_LABEL = {"fixed": "fixed", "jenks": "Jenks", "entmax15": "entmax-1.5"}
RULE_DIR = {"jenks": "ablation_adaptive_jenks",
            "entmax15": "ablation_adaptive_entmax15"}
FIXED_PERSEED_DIR = "ablation_fixed_seed0"
PAIRS = (("jenks", "fixed"), ("entmax15", "fixed"), ("entmax15", "jenks"))

# Categorical-input fix (GraphAwareDenoiser cat_input="prob"): its runs live in
# sibling trees with this suffix. A dataset uses them only when ALL three rules
# are complete there; otherwise it stays entirely on the original trees. Old
# (cat_input="log") and fixed runs are never mixed within a dataset.
CATFIX_SUFFIX = "_catfix"
# "auto": catfix where complete (above); "legacy": force the original trees for
# every dataset (the before-the-fix comparison, written to a separate OUT).
VARIANT_MODE = os.environ.get("COMPARE_TOPK_VARIANT", "auto")
assert VARIANT_MODE in ("auto", "legacy"), VARIANT_MODE
# Datasets whose jobs are still running are left out entirely (not reported as
# partial). Override with COMPARE_TOPK_EXCLUDE="" once they finish.
EXCLUDE = {d for d in os.environ.get("COMPARE_TOPK_EXCLUDE", "higgs-small,miniboone").split(",") if d}
# wasserstein_mean / mia_auc recomputed post hoc for split-runner runs, whose
# shards lack them (split_runner passed one argument too few; the TypeError was
# swallowed). See analysis/integrity/backfill_gen_metrics.py.
POSTHOC_CSV = REPO / "analysis" / "integrity" / "posthoc_gen_metrics.csv"
_POSTHOC = None


def posthoc_rows(run_dir: Path) -> pd.DataFrame:
    global _POSTHOC
    if _POSTHOC is None:
        _POSTHOC = pd.read_csv(POSTHOC_CSV) if POSTHOC_CSV.exists() else pd.DataFrame(columns=["run_dir"])
    try:
        key = str(run_dir.resolve().relative_to(RESULTS_ROOT.resolve()))
    except ValueError:
        return pd.DataFrame()
    return _POSTHOC[_POSTHOC.run_dir == key].drop(columns="run_dir")

# (key, provenance filter, label). Order = reporting order: held-out first.
SUBSETS = (("held_out", "transfer", "held-out evaluation (transferred)"),
           ("development", "own ablation", "development set (ablated)"),
           ("pooled", None, "pooled (held-out + development)"))
SUBSET_LABEL = {k: lab for k, _, lab in SUBSETS}
# Table / figure group headings, held-out first.
GROUPS = (("Held-out evaluation (transferred): configuration chosen without seeing any of its results",
           "transfer"),
          ("Development set (ablated): configuration selected on the test split by fixed-mode score; "
           "in-sample and optimistic", "own ablation"))

CONFIG_KEYS = ("gnn_type", "n_layers", "d_model", "top_k", "n_heads", "use_attention")

# metric -> (+1 higher is better | -1 lower is better, family, label)
# "tabddpm_util" is a derived key: macro-F1 for classification, R2 for
# regression (the TabDDPM paper's convention). "mia_dev" is |mean AUC - 0.5|.
METRICS = {
    "score":            (+1, "utility",  "score (acc / -RMSE)"),
    "tabddpm_util":     (+1, "utility",  "macro-F1 / R2"),
    "roc_auc":          (+1, "utility",  "ROC AUC"),
    "wasserstein_mean": (-1, "fidelity", "Wasserstein"),
    "dcr_tabddpm_mean": (+1, "privacy",  "DCR (mean L2)"),
    "dcr_baseline":     (+1, "privacy",  "DCR (median+label)"),
    "mia_dev":          (-1, "privacy",  "|MIA AUC - 0.5|"),
}
GEN_LEVEL = {"wasserstein_mean", "dcr_tabddpm_mean", "dcr_baseline", "mia_auc"}
PRIVACY = ("dcr_tabddpm_mean", "dcr_baseline", "mia_dev")
ACROSS_METRICS = ("score", "tabddpm_util")

# Figure palette (dataviz reference palette, light mode).
C_RULE = {"fixed": "#1baf7a", "jenks": "#2a78d6", "entmax15": "#eb6834"}
C_TEXT, C_TEXT2, C_GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_DIV_NEG, C_DIV_MID, C_DIV_POS = "#e34948", "#f0efec", "#2a78d6"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def config_id(c: dict) -> str:
    return (f"{c['gnn_type']}_L{c['n_layers']}_d{c['d_model']}_k{c['top_k']}"
            f"_h{c['n_heads']}{'' if not c['use_attention'] else '_attn'}")


def config_from_exp_name(name: str) -> dict | None:
    m = re.fullmatch(r"gnn_(\w+?)_layers(\d+)_dim(\d+)_topk(\d+)_heads(\d+)(_noattn)?", name)
    if not m:
        return None
    return {"gnn_type": m.group(1), "n_layers": int(m.group(2)), "d_model": int(m.group(3)),
            "top_k": int(m.group(4)), "n_heads": int(m.group(5)),
            "use_attention": m.group(6) is None}


def task_type(ds: str) -> str:
    return json.load(open(REPO / "data" / ds / "info.json"))["task_type"]


def rel(p: Path | str) -> str:
    p = Path(p)
    return os.path.relpath(p, REPO) if str(p).startswith(str(REPO)) else str(p)


def fmt_num(x: float, ref: float | None = None) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    mag = abs(ref if ref is not None and np.isfinite(ref) else x)
    if mag >= 1000:
        return f"{x:.1f}"
    if mag >= 10:
        return f"{x:.2f}"
    return f"{x:.4f}"


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return out.tolist()
    order = ok[np.argsort(p[ok])]
    m, running = len(ok), 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        out[idx] = running
    return out.tolist()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_run_dir(run_dir: Path) -> dict:
    """Per-generation-seed values of one run directory.

    Reads per_seed_scores.csv and any split-run shards per_seed_scores.gen<k>.csv
    (same format), de-duplicated. A generation seed counts only when its
    utility has all EXPECTED_CLF classifier rows AND its per-generation-seed
    metrics are present -- a shard still being written is not counted.
    """
    files = sorted(run_dir.glob("per_seed_scores.csv")) + sorted(run_dir.glob("per_seed_scores.gen*.csv"))
    meta_path = run_dir / "run_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    if not files:
        return {"per_gen": pd.DataFrame(), "n_gen": 0, "n_clf": 0, "meta": meta,
                "files": [], "gen_seeds": [], "n_posthoc": 0}
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw["clf_key"] = raw["clf_seed"].fillna(-1)
    raw = raw.drop_duplicates(["gen_seed", "clf_key", "metric"], keep="last")
    # post-hoc gen-level metrics only where the run itself has none
    ph = posthoc_rows(run_dir)
    n_posthoc = 0
    if len(ph):
        have = set(zip(raw[raw.clf_seed.isna()].gen_seed, raw[raw.clf_seed.isna()].metric))
        ph = ph[[(g, m) not in have for g, m in zip(ph.gen_seed, ph.metric)]].assign(clf_key=-1)
        n_posthoc = len(ph)
        raw = pd.concat([raw, ph], ignore_index=True)
    util = raw[raw["clf_seed"].notna()]
    gen = raw[raw["clf_seed"].isna()]

    n_clf = util[util.metric == "score"].groupby("gen_seed").size()
    has_gen = set(gen[gen.metric == "mia_auc"].gen_seed) & set(gen[gen.metric == "wasserstein_mean"].gen_seed)
    complete = sorted(int(s) for s, n in n_clf.items() if n >= EXPECTED_CLF and s in has_gen)

    u = (util[util.gen_seed.isin(complete)]
         .groupby(["gen_seed", "metric"])["value"].mean().unstack())
    g = gen[gen.gen_seed.isin(complete)].pivot_table(index="gen_seed", columns="metric", values="value")
    per_gen = pd.concat([u, g], axis=1).sort_index()
    return {"per_gen": per_gen, "n_gen": len(complete),
            "n_clf": int(n_clf[n_clf.index.isin(complete)].min()) if complete else 0,
            "meta": meta, "files": [rel(f) for f in files], "gen_seeds": complete,
            "n_posthoc": n_posthoc}


def derive(per_gen: pd.DataFrame, task: str) -> pd.DataFrame:
    """Add tabddpm_util; keep only the metrics this study reports."""
    df = per_gen.copy()
    src = "r2" if task == "regression" else "macro avg/f1-score"
    if src in df:
        df["tabddpm_util"] = df[src]
    keep = [m for m in list(METRICS) + ["mia_auc", "rmse", "r2", "accuracy", "macro avg/f1-score"]
            if m in df and m != "mia_dev"]
    return df[keep]


def load_summary_json(ds: str, cfg: dict) -> dict | None:
    """The seed-0 fixed-mode row of <ds>/ablation/<ds>_ablation.json."""
    paths = sorted((RESULTS_ROOT / ds / "ablation").glob("*_ablation.json"))
    for p in paths:
        for rec in json.load(open(p)):
            if all(rec.get(k) == cfg[k] for k in CONFIG_KEYS) and rec.get("status") == "done":
                return {"rec": rec, "path": p}
    return None


def summary_stats(rec: dict, task: str) -> dict:
    """metric -> dict(mean, sd, sd_kind). sd is a sample (ddof=1) std over
    generation seeds for per-generation-seed metrics; for utility metrics the
    stored std spans all gen x clf runs and is NOT a between-seed std."""
    n_gen, n_clf = rec["n_gen_seeds"], rec["n_clf_seeds"]
    out = {}
    m = dict(rec["metrics"])
    src = "r2" if task == "regression" else "macro avg/f1-score"
    if src in m:
        m["tabddpm_util"] = m[src]
    for key, v in m.items():
        if key in GEN_LEVEL:
            sd = v["std"] * math.sqrt(n_gen / (n_gen - 1)) if n_gen > 1 else np.nan
            out[key] = {"mean": v["mean"], "sd": sd, "sd_kind": f"gen seeds (n={n_gen}, ddof=1)"}
        else:
            out[key] = {"mean": v["mean"], "sd": v["std"],
                        "sd_kind": f"all {n_gen}x{n_clf} gen x clf runs (ddof=0)"}
    return out


def result_trees(variant: str) -> dict:
    """Tree names per rule for one variant ("legacy" = cat_input log, "catfix")."""
    suf = CATFIX_SUFFIX if variant == "catfix" else ""
    return {"jenks": RULE_DIR["jenks"] + suf, "entmax15": RULE_DIR["entmax15"] + suf,
            # the per-seed fixed reruns exist only for the legacy variant; the
            # catfix fixed runs are per-seed from the start
            "fixed_perseed": None if variant == "catfix" else FIXED_PERSEED_DIR,
            "fixed_dir": "ablation" + suf}


def n_categorical(ds: str) -> int:
    return int(json.load(open(REPO / "data" / ds / "info.json")).get("n_cat_features", 0) or 0)


def choose_variant(ds: str) -> tuple[str, str]:
    """("catfix" | "legacy", note). catfix only when every rule is complete in
    the *_catfix trees (the same combo in all three)."""
    if VARIANT_MODE == "legacy":
        return "legacy", ""
    tr = result_trees("catfix")
    per_rule = {}
    for r, key in (("fixed", "fixed_dir"), ("jenks", "jenks"), ("entmax15", "entmax15")):
        combos = {d.name: load_run_dir(d)["n_gen"]
                  for d in sorted((RESULTS_ROOT / ds / tr[key]).glob("gnn_*")) if d.is_dir()}
        per_rule[r] = combos
    if not any(per_rule.values()):
        return "legacy", ""
    shared = set.intersection(*(set(c) for c in per_rule.values())) if all(per_rule.values()) else set()
    if any(all(per_rule[r][c] >= EXPECTED_GEN for r in per_rule) for c in shared):
        return "catfix", ""
    note = "catfix in progress: " + ", ".join(
        f"{r} " + (f"{max(c.values())}/{EXPECTED_GEN}" if c else "none") for r, c in per_rule.items())
    return "legacy", note


def discover() -> tuple[pd.DataFrame, dict]:
    meta = pd.read_csv(META_CSV)
    best = pd.read_csv(BEST_CSV)
    ablated = set(best.dataset)
    datasets = sorted((set(meta.dataset) | {p.name for p in RESULTS_ROOT.iterdir() if p.is_dir()}) - EXCLUDE)

    cov_rows, cells = [], {}
    for ds in datasets:
        task = task_type(ds)
        prov = "own ablation" if ds in ablated else "transfer"
        variant, catfix_note = choose_variant(ds)
        tr = result_trees(variant)
        n_cat = n_categorical(ds)
        # Which configuration directories exist per rule
        dirs = {k: (sorted(p for p in (RESULTS_ROOT / ds / t).glob("gnn_*") if p.is_dir()) if t else [])
                for k, t in tr.items()}
        names = {k: {p.name for p in v} for k, v in dirs.items()}
        candidate = set().union(*names.values())
        # A configuration is usable when every rule that has any run has it.
        # (The seed-0 summary JSON contains the whole grid, so it never restricts.)
        present = [names["jenks"], names["entmax15"], names["fixed_perseed"] | names["fixed_dir"]]
        present = [s for s in present if s]
        common = set.intersection(*present) if present else set()
        mismatch = bool(present) and not common
        if len(common) > 1:
            # prefer the configuration with the most complete cells
            common = {max(common, key=lambda n: sum(n in s for s in present))}
        exp_name = next(iter(common)) if common else (next(iter(candidate)) if candidate else None)
        cfg = config_from_exp_name(exp_name) if exp_name else None

        ds_cells = {}
        for r in RULES:
            cell = {"status": "missing", "n_gen": 0, "n_clf": 0, "per_gen": None,
                    "summary": None, "source": "", "kind": "", "config": "", "config_ok": None,
                    "seed": None, "n_posthoc": 0, "cat_input": "", "topk_mode": ""}
            run_dir = None
            if r == "fixed":
                if exp_name and exp_name in names["fixed_perseed"]:
                    run_dir, cell["kind"] = (RESULTS_ROOT / ds / tr["fixed_perseed"] / exp_name,
                                             "per-seed (training-seed-0 rerun)")
                elif exp_name and exp_name in names["fixed_dir"]:
                    run_dir, cell["kind"] = RESULTS_ROOT / ds / tr["fixed_dir"] / exp_name, "per-seed"
            elif exp_name and exp_name in names[r]:
                run_dir, cell["kind"] = RESULTS_ROOT / ds / tr[r] / exp_name, "per-seed"

            loaded = load_run_dir(run_dir) if run_dir else None
            if loaded and loaded["n_gen"] > 0:
                m = loaded["meta"]
                cell.update(per_gen=derive(loaded["per_gen"], task), n_gen=loaded["n_gen"],
                            n_clf=loaded["n_clf"], source=rel(run_dir),
                            status="complete" if loaded["n_gen"] >= EXPECTED_GEN else "partial",
                            seed=m.get("seed"), n_posthoc=loaded["n_posthoc"])
                dc_path = run_dir / "denoiser_config.json"
                dc = json.load(open(dc_path)) if dc_path.exists() else {}
                cell["cat_input"] = dc.get("cat_input") or m.get("cat_input") or "log (default)"
                cell["topk_mode"] = dc.get("topk_mode") or m.get("topk_mode") or "fixed"
                if m:
                    mc = {k: m.get(k) for k in CONFIG_KEYS}
                    cell["config"] = config_id(mc)
                    cell["config_ok"] = cfg is not None and all(mc[k] == cfg[k] for k in CONFIG_KEYS)
                else:
                    cell["config"], cell["config_ok"] = config_id(cfg), None
            elif run_dir:
                cell.update(status="partial" if (run_dir / "run_meta.json").exists() else "missing",
                            source=rel(run_dir), config=config_id(cfg) if cfg else "")

            # The seed-0 grid summary is a cat_input="log" run: legacy only.
            if r == "fixed" and cfg is not None and variant == "legacy":
                sj = load_summary_json(ds, cfg)
                if sj:
                    cell["seed0_ref"] = {"stats": summary_stats(sj["rec"], task), "path": rel(sj["path"]),
                                         "n_gen": sj["rec"]["n_gen_seeds"], "n_clf": sj["rec"]["n_clf_seeds"]}
                    if cell["status"] != "complete":
                        rec = sj["rec"]
                        cell.update(status="complete (summary only)", n_gen=rec["n_gen_seeds"],
                                    n_clf=rec["n_clf_seeds"], summary=summary_stats(rec, task),
                                    source=f"{rel(sj['path'])} [{rec['exp_name']}]",
                                    kind="summary only (training seed 0)", config=config_id(cfg),
                                    config_ok=True, per_gen=None, seed=0)
            ds_cells[r] = cell
            cov_rows.append({
                "dataset": ds, "task_type": task, "provenance": prov, "rule": r,
                "config": cell["config"], "config_matches_dataset": cell["config_ok"],
                "status": cell["status"], "data_kind": cell["kind"],
                "n_gen_seeds": cell["n_gen"], "n_clf_seeds": cell["n_clf"],
                "training_seed": cell["seed"], "source": cell["source"],
                "cat_input": cell["cat_input"], "topk_mode": cell["topk_mode"],
                "posthoc_gen_metric_rows": cell["n_posthoc"],
                "variant": variant, "n_categorical": n_cat,
                "cat_input_affected": bool(n_cat and variant == "legacy"), "catfix_note": catfix_note})
        cells[ds] = {"task": task, "prov": prov, "cfg": cfg, "exp_name": exp_name,
                     "mismatch": mismatch, "rules": ds_cells, "variant": variant,
                     "catfix_note": catfix_note, "n_cat": n_cat,
                     "cat_affected": bool(n_cat and variant == "legacy")}
    return pd.DataFrame(cov_rows), cells


# ---------------------------------------------------------------------------
# Per-rule descriptives
# ---------------------------------------------------------------------------
def rule_values(cell: dict, metric: str) -> dict | None:
    """Return {'vals': array|None, 'mean', 'sd', 'sd_kind', 'n'} for one cell."""
    if cell["status"] == "missing":
        return None
    if cell["per_gen"] is not None:
        df = cell["per_gen"]
        key = "mia_auc" if metric == "mia_dev" else metric
        if key not in df or df[key].isna().all():
            return None
        v = df[key].dropna().to_numpy(float)
        if metric == "mia_dev":
            return {"vals": v, "mean": abs(v.mean() - 0.5), "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                    "sd_kind": "sd of MIA AUC over gen seeds", "n": len(v), "raw_mean": v.mean()}
        return {"vals": v, "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                "sd_kind": "gen seeds (ddof=1)", "n": len(v)}
    if cell["summary"] is not None:
        key = "mia_auc" if metric == "mia_dev" else metric
        s = cell["summary"].get(key)
        if s is None:
            return None
        out = {"vals": None, "mean": s["mean"], "sd": s["sd"], "sd_kind": s["sd_kind"], "n": cell["n_gen"]}
        if metric == "mia_dev":
            out.update(mean=abs(s["mean"] - 0.5), raw_mean=s["mean"])
        return out
    return None


def per_dataset_table(cells: dict) -> pd.DataFrame:
    rows = []
    for ds, info in cells.items():
        for r in RULES:
            cell = info["rules"][r]
            for metric in list(METRICS) + ["mia_auc"]:
                if metric == "roc_auc" and info["task"] != "binclass":
                    continue
                rv = rule_values(cell, metric)
                if rv is None:
                    continue
                rows.append({"dataset": ds, "provenance": info["prov"], "task_type": info["task"],
                             "rule": r, "metric": metric, "mean": rv["mean"], "sd": rv["sd"],
                             "sd_kind": rv["sd_kind"], "n_gen_seeds": rv["n"], "status": cell["status"],
                             "source": cell["source"]})
            ref = cell.get("seed0_ref")
            if r == "fixed" and ref and cell["summary"] is None:
                for metric, s in ref["stats"].items():
                    if metric not in METRICS and metric != "mia_auc":
                        continue
                    rows.append({"dataset": ds, "provenance": info["prov"], "task_type": info["task"],
                                 "rule": "fixed_seed0_ref", "metric": metric, "mean": s["mean"],
                                 "sd": s["sd"], "sd_kind": s["sd_kind"], "n_gen_seeds": ref["n_gen"],
                                 "status": "reference", "source": ref["path"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 1: pairwise per-dataset comparison
# ---------------------------------------------------------------------------
def compare(a: dict, b: dict, metric: str, rng: np.random.Generator) -> dict:
    """Delta of rule a vs rule b, signed so that positive = a is better."""
    sign = METRICS[metric][0]
    out = {"delta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_welch": np.nan,
           "ci_method": "", "rel_delta_pct": np.nan, "delta_over_sd_b": np.nan, "outcome": "n/a"}
    if a is None or b is None:
        out["outcome"] = "missing"
        return out
    if metric == "mia_dev":
        out["delta"] = b["mean"] - a["mean"]          # smaller deviation is better
    else:
        out["delta"] = sign * (a["mean"] - b["mean"])
        if b["mean"] != 0:
            out["rel_delta_pct"] = 100 * out["delta"] / abs(b["mean"])
    if np.isfinite(b["sd"]) and b["sd"] > 0 and not b["sd_kind"].startswith("all "):
        out["delta_over_sd_b"] = out["delta"] / b["sd"]

    partial = a["n"] < EXPECTED_GEN or b["n"] < EXPECTED_GEN
    if a["vals"] is not None and b["vals"] is not None:
        va, vb = a["vals"], b["vals"]
        ia = rng.integers(0, len(va), size=(N_BOOT, len(va)))
        ib = rng.integers(0, len(vb), size=(N_BOOT, len(vb)))
        ma, mb = va[ia].mean(1), vb[ib].mean(1)
        boot = (np.abs(mb - 0.5) - np.abs(ma - 0.5)) if metric == "mia_dev" else sign * (ma - mb)
        out["ci_lo"], out["ci_hi"] = np.percentile(boot, [2.5, 97.5])
        out["ci_method"] = "bootstrap (gen seeds, independent resampling)"
        if len(va) > 1 and len(vb) > 1 and (va.std() > 0 or vb.std() > 0):
            out["p_welch"] = stats.ttest_ind(va, vb, equal_var=False).pvalue
    elif metric in GEN_LEVEL or metric == "mia_dev":
        # b is the seed-0 summary: exact Welch test from mean/sd over gen seeds.
        if a["n"] > 1 and b["n"] > 1 and np.isfinite(a["sd"]) and np.isfinite(b["sd"]):
            ma = a.get("raw_mean", a["mean"])
            mb = b.get("raw_mean", b["mean"])
            res = stats.ttest_ind_from_stats(ma, a["sd"], a["n"], mb, b["sd"], b["n"], equal_var=False)
            out["p_welch"] = res.pvalue
            se2a, se2b = a["sd"] ** 2 / a["n"], b["sd"] ** 2 / b["n"]
            se = math.sqrt(se2a + se2b)
            dof = (se2a + se2b) ** 2 / (se2a ** 2 / (a["n"] - 1) + se2b ** 2 / (b["n"] - 1)) if se > 0 else np.nan
            if se > 0:
                half = stats.t.ppf(0.975, dof) * se
                raw = ma - mb
                if metric == "mia_dev":
                    # the CI is on the raw AUC difference; a win/loss is read
                    # from whether that difference is resolved, and its sign
                    # from the change in |AUC - 0.5|.
                    resolved = (raw - half) > 0 or (raw + half) < 0
                    out["ci_lo"], out["ci_hi"] = (np.nan, np.nan)
                    out["ci_method"] = "Welch t from summary stats (raw AUC)"
                    out["_resolved"] = resolved
                else:
                    lo, hi = sign * (raw - half), sign * (raw + half)
                    out["ci_lo"], out["ci_hi"] = min(lo, hi), max(lo, hi)
                    out["ci_method"] = "Welch t from summary stats"
    else:
        out["outcome"] = "n/a (fixed has no per-seed utility)"
        return out

    if partial:
        out["outcome"] = "partial"
    elif "_resolved" in out:
        out["outcome"] = ("win" if out["delta"] > 0 else "loss") if out.pop("_resolved") else "tie"
    elif np.isfinite(out["ci_lo"]):
        out["outcome"] = "win" if out["ci_lo"] > 0 else ("loss" if out["ci_hi"] < 0 else "tie")
    out.pop("_resolved", None)
    return out


def deltas_table(cells: dict) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for ds, info in cells.items():
        if info["mismatch"]:
            continue
        for a_rule, b_rule in PAIRS:
            ca, cb = info["rules"][a_rule], info["rules"][b_rule]
            for metric in METRICS:
                if metric == "roc_auc" and info["task"] != "binclass":
                    continue
                a, b = rule_values(ca, metric), rule_values(cb, metric)
                if a is None and b is None:
                    continue
                res = compare(a, b, metric, rng)
                privacy_flag = metric in PRIVACY and res["outcome"] == "loss"
                rows.append({"dataset": ds, "provenance": info["prov"], "task_type": info["task"],
                             "rule": a_rule, "vs": b_rule, "metric": metric,
                             "family": METRICS[metric][1],
                             "mean_rule": a["mean"] if a else np.nan, "mean_vs": b["mean"] if b else np.nan,
                             "n_rule": a["n"] if a else 0, "n_vs": b["n"] if b else 0,
                             "vs_kind": cb["kind"], **res, "privacy_flag": privacy_flag,
                             "subset": "development" if info["prov"] == "own ablation" else "held_out",
                             "fixed_summary_only": "fixed" in (a_rule, b_rule)
                                                   and cells[ds]["rules"]["fixed"]["summary"] is not None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: across datasets
# ---------------------------------------------------------------------------
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}


def across(cells: dict, deltas: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, cd_data = [], {}
    for metric in ACROSS_METRICS:
        sign = METRICS[metric][0]
        for sub, prov, _ in SUBSETS:
            use = [ds for ds, info in cells.items()
                   if not info["mismatch"] and (prov is None or info["prov"] == prov)
                   and all(info["rules"][r]["status"].startswith("complete") for r in RULES)
                   and all(rule_values(info["rules"][r], metric) is not None for r in RULES)]
            includes_dev = any(cells[d]["prov"] == "own ablation" for d in use)
            n_summary = sum(cells[d]["rules"]["fixed"]["summary"] is not None for d in use)
            base = {"metric": metric, "subset": sub, "n_datasets": len(use),
                    "datasets": ";".join(use), "includes_development": includes_dev,
                    "n_fixed_summary_only": n_summary}
            if not use:
                rows.append({**base, "kind": "none", "note": "no dataset complete for all 3 rules"})
                continue
            M = np.array([[sign * rule_values(cells[d]["rules"][r], metric)["mean"] for r in RULES]
                          for d in use])
            ranks = np.vstack([stats.rankdata(-row) for row in M])
            avg = ranks.mean(0)
            for r, ar in zip(RULES, avg):
                rows.append({**base, "kind": "avg_rank", "a": r, "value": ar})
            fried_p = np.nan
            if len(use) >= 3:
                try:
                    st, fried_p = stats.friedmanchisquare(*M.T)
                    rows.append({**base, "kind": "friedman", "statistic": st, "p_value": fried_p})
                except ValueError as exc:
                    rows.append({**base, "kind": "friedman", "note": str(exc)})
            else:
                rows.append({**base, "kind": "friedman", "note": "n < 3 datasets"})
            k, n = len(RULES), len(use)
            cd = NEMENYI_Q05[k] * math.sqrt(k * (k + 1) / (6 * n))
            nemenyi = []
            if np.isfinite(fried_p) and fried_p < ALPHA:
                se = math.sqrt(k * (k + 1) / (6 * n))
                for i in range(k):
                    for j in range(i + 1, k):
                        q = abs(avg[i] - avg[j]) / se * math.sqrt(2)
                        p = float(stats.studentized_range.sf(q, k, np.inf))
                        nemenyi.append((RULES[i], RULES[j], p))
                        rows.append({**base, "kind": "nemenyi", "a": RULES[i], "b": RULES[j],
                                     "value": abs(avg[i] - avg[j]), "p_value": p, "note": f"CD={cd:.3f}"})
            cd_data[(metric, sub)] = {"avg": dict(zip(RULES, avg)), "n": n, "cd": cd,
                                      "fried_p": fried_p, "nemenyi": nemenyi,
                                      "includes_dev": includes_dev, "n_summary": n_summary}

            # Wilcoxon on the RELATIVE delta (%), so datasets measured in
            # different units (accuracy vs RMSE in the thousands) are comparable.
            wl = []
            for a_rule, b_rule in PAIRS:
                d = deltas[(deltas.metric == metric) & (deltas.rule == a_rule) & (deltas.vs == b_rule)
                           & deltas.dataset.isin(use)]
                x = d.rel_delta_pct.dropna().to_numpy()
                p = st = np.nan
                note = ""
                if len(x) >= 1 and np.any(x != 0):
                    try:
                        res = stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided")
                        st, p = res.statistic, res.pvalue
                    except ValueError as exc:
                        note = str(exc)
                oc = d.outcome.value_counts()
                wl.append({**base, "kind": "wilcoxon", "a": a_rule, "b": b_rule, "statistic": st,
                           "p_value": p, "value": float(np.median(x)) if len(x) else np.nan,
                           "note": (note + " value = median relative delta %").strip(),
                           "wins": int(oc.get("win", 0)), "ties": int(oc.get("tie", 0)),
                           "losses": int(oc.get("loss", 0)),
                           "not_testable": int(len(d) - oc.get("win", 0) - oc.get("tie", 0) - oc.get("loss", 0))})
            for w, ph in zip(wl, holm([w["p_value"] for w in wl])):
                w["p_holm"] = ph
            rows.extend(wl)
    return pd.DataFrame(rows), cd_data


# ---------------------------------------------------------------------------
# Step 3: mechanism
# ---------------------------------------------------------------------------
SAMPLE_KEY = ["step", "block", "t_bucket", "tag"]


def read_topk_log(path: Path) -> tuple[pd.DataFrame, bool]:
    """A topk_stats.csv / jenks_gate_stats.csv, cleaned for analysis.

    * Truncated copies (the file ends mid-line, e.g. an incomplete transfer)
      are flagged; the cut-off last row is dropped by keeping rows with a
      finite n_samples.
    * Sampling rows are unique on (step, block, t_bucket, tag): the counter
      restarts per sample() call and tag is the gen seed. A gen seed that was
      re-run (split_runner --load_checkpoint after a failed evaluation)
      appends the same keys again; keep the last copy so no seed counts twice.
      Training rows are left alone.
    """
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        truncated = False
        if size:
            fh.seek(-1, 2)
            truncated = fh.read(1) not in (b"\n", b"\r")
    t = pd.read_csv(path)
    t = t[pd.to_numeric(t.n_samples, errors="coerce").notna()]
    sam = t[t.phase == "sample"].drop_duplicates(SAMPLE_KEY, keep="last")
    return pd.concat([t[t.phase != "sample"], sam], ignore_index=True), truncated


def effective_k(cells: dict, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, info in cells.items():
        if info["cfg"] is None:
            continue
        n_nodes = int(meta.set_index("dataset").loc[ds, "mg_n_nodes"]) if ds in set(meta.dataset) else np.nan
        k_fixed = min(info["cfg"]["top_k"], n_nodes - 1) if np.isfinite(n_nodes) else info["cfg"]["top_k"]
        for r in ("jenks", "entmax15"):
            if not info["rules"][r]["status"].startswith("complete"):
                continue  # partial runs (e.g. fb-comments) are reported as missing, not analysed
            src = info["rules"][r]["source"]
            path = (Path(src) if os.path.isabs(src) else REPO / src) / "topk_stats.csv" if src else None
            if not path or not path.exists():
                continue
            t, truncated = read_topk_log(path)
            row = {"dataset": ds, "provenance": info["prov"], "rule": r, "n_nodes": n_nodes,
                   "top_k_config": info["cfg"]["top_k"], "k_fixed_effective": k_fixed,
                   "status": info["rules"][r]["status"], "source": rel(path),
                   "topk_log_truncated": truncated}
            s = t[t.phase == "sample"]
            row["sample_gen_seeds_logged"] = ";".join(
                str(int(x)) for x in sorted(s.tag.dropna().unique())) if len(s) else ""
            if len(s):
                row["k_eff_sample"] = np.average(s.k_mean, weights=s.n_samples)
                per_block = pd.Series({b: np.average(g.k_mean, weights=g.n_samples)
                                       for b, g in s.groupby("block")})
                row["k_eff_sample_block_min"] = per_block.min()
                row["k_eff_sample_block_max"] = per_block.max()
            tr = t[t.phase == "train"]
            if len(tr):
                last = tr[tr.step >= 0.8 * tr.step.max()]
                row["k_eff_train_last20pct"] = np.average(last.k_mean, weights=last.n_samples)
            if "k_eff_sample" in row:
                row["ratio_to_fixed"] = row["k_eff_sample"] / k_fixed
                row["vs_fixed"] = "fewer" if row["k_eff_sample"] < k_fixed else "more"
                row["density_eff"] = row["k_eff_sample"] / (n_nodes - 1)
            # Jenks only: jenks_gate_stats.csv is a COUNTERFACTUAL log. The runs
            # use plain Jenks (use_gate_rule=False); frac_gate_off is the share
            # of (sample, node) rows whose strong-class mean is <= 0, i.e. where
            # the dropped acceptance test WOULD have fallen back to k_min. The
            # k of those rows is not logged separately.
            gpath = path.parent / "jenks_gate_stats.csv"
            if r == "jenks" and gpath.exists():
                gs, gtrunc = read_topk_log(gpath)
                row["topk_log_truncated"] = row["topk_log_truncated"] or gtrunc
                gsam = gs[gs.phase == "sample"]
                if len(gsam):
                    row["jenks_would_gate_frac_sample"] = np.average(gsam.frac_gate_off, weights=gsam.n_samples)
                gtr = gs[gs.phase == "train"]
                if len(gtr):
                    last = gtr[gtr.step >= 0.8 * gtr.step.max()]
                    row["jenks_would_gate_frac_train_last20pct"] = np.average(last.frac_gate_off,
                                                                              weights=last.n_samples)
            rows.append(row)
    return pd.DataFrame(rows)


PREDICTORS = ("assoc_density_0.3", "assoc_assoc_mean", "n_features", "num_cat_ratio")


def correlations(deltas: pd.DataFrame, meta: pd.DataFrame, cells: dict) -> pd.DataFrame:
    m = meta.set_index("dataset")
    rows = []
    for metric in ACROSS_METRICS:
        for a_rule in ("jenks", "entmax15"):
            d = deltas[(deltas.metric == metric) & (deltas.rule == a_rule) & (deltas.vs == "fixed")
                       & deltas.rel_delta_pct.notna() & (deltas.n_rule >= EXPECTED_GEN)
                       & (deltas.n_vs >= EXPECTED_GEN)]
            d = d[d.dataset.isin(m.index)]
            for pred in PREDICTORS:
                x, y = m.loc[d.dataset, pred].to_numpy(float), d.rel_delta_pct.to_numpy(float)
                rho = p = np.nan
                if len(d) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                    rho, p = stats.spearmanr(x, y)
                rows.append({"metric": metric, "rule": a_rule, "predictor": pred, "n": len(d),
                             "spearman_rho": rho, "p_value": p,
                             "includes_development": bool((d.subset == "development").any()),
                             "datasets": ";".join(d.dataset)})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for metric, g in out.groupby("metric"):
        out.loc[g.index, "p_holm"] = holm(g.p_value.tolist())
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def display_value(metric: str, task: str, mean: float) -> float:
    """score is -RMSE for regression; show RMSE (positive, lower is better)."""
    return -mean if (metric == "score" and task == "regression") else mean


def metric_header(metric: str, task: str) -> str:
    if metric == "score":
        return "RMSE ↓" if task == "regression" else "Acc ↑"
    if metric == "tabddpm_util":
        return "R² ↑" if task == "regression" else "macro-F1 ↑"
    return METRICS[metric][2]


def utility_table(cells: dict, metric: str, meta: pd.DataFrame) -> tuple[str, str]:
    m = meta.set_index("dataset")
    show_ref = any(info["rules"]["fixed"].get("seed0_ref") and info["rules"]["fixed"]["summary"] is None
                   for info in cells.values())
    cols = list(RULES) + (["seed0_ref"] if show_ref else [])
    head = ["dataset", "task", "imbal.", "metric"] + [RULE_LABEL.get(c, "fixed grid summary (ref)") for c in cols]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = [r"\begin{table}[t]", r"\centering\small",
           r"\begin{tabular}{llrl" + "c" * len(cols) + "}", r"\toprule",
           " & ".join(["Dataset", "Task", "Imbal.", "Metric"]
                      + [RULE_LABEL.get(c, "fixed grid summary (ref.)") for c in cols]) + r" \\", r"\midrule"]
    for group, prov in GROUPS:
        md.append(f"| **{group}** |" + " |" * (len(head) - 1))
        tex.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(head), group))
        for ds, info in cells.items():
            if info["prov"] != prov:
                continue
            task = info["task"]
            imb = m.loc[ds, "class_imbalance"] if ds in m.index else np.nan
            imb_s = f"{imb:.2f}" if np.isfinite(imb) else "—"
            vals = {}
            for c in RULES:
                vals[c] = rule_values(info["rules"][c], metric)
            best = None
            full = [c for c in RULES if vals[c] is not None]
            if full:
                sign = METRICS[metric][0]
                best = max(full, key=lambda c: sign * vals[c]["mean"])
            md_cells, tex_cells = [], []
            for c in cols:
                if c == "seed0_ref":
                    ref = info["rules"]["fixed"].get("seed0_ref")
                    s = ref["stats"].get(metric) if ref and info["rules"]["fixed"]["summary"] is None else None
                    v = {"mean": s["mean"], "sd": s["sd"]} if s else None
                    status, sdk = "reference", "all"
                else:
                    v = vals[c]
                    status = info["rules"][c]["status"]
                    sdk = v["sd_kind"] if v else ""
                if v is None:
                    md_cells.append(status if status in ("missing", "partial") else "—")
                    tex_cells.append("---" if status != "partial" else r"partial$^\dagger$")
                    continue
                mean = display_value(metric, task, v["mean"])
                txt = f"{fmt_num(mean)} ± {fmt_num(v['sd'], mean)}"
                mark_md = mark_tex = ""
                if status == "partial":
                    mark_md, mark_tex = " †", r"$^\dagger$"
                if sdk.startswith("all "):
                    mark_md, mark_tex = mark_md + " ‡", mark_tex + r"$^\ddagger$"
                bold = (c == best)
                md_cells.append((f"**{txt}**" if bold else txt) + mark_md)
                t = txt.replace("±", r"$\pm$")
                tex_cells.append((r"\textbf{" + t + "}" if bold else t) + mark_tex)
            hdr = metric_header(metric, task)
            lab = ds + (" §" if info["cat_affected"] else "")
            md.append(f"| {lab} | {task} | {imb_s} | {hdr} | " + " | ".join(md_cells) + " |")
            tex.append(" & ".join([ds + (r"$^\S$" if info["cat_affected"] else ""), task, imb_s, hdr.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")
                                   .replace("²", "$^2$")] + tex_cells) + r" \\")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    note = ("Primary claims rest on the held-out (transferred) rows; development-set (ablated) rows are "
            "in-sample with respect to configuration selection, which was made on the test split by "
            "fixed-mode score, so they are optimistic, most of all for fixed. "
            "mean ± sd over generation seeds (n=5, each averaged over 10 CatBoost seeds). "
            "Bold = best mean in the row. † partial (<5 gen seeds). § trained with the categorical-input bug "
            "(cat_input=log; see caveats). ‡ fixed grid summary: "
            "sd spans all 5×10 gen×clf runs, not generation seeds. Imbal. = majority/minority "
            "class ratio (— for regression).")
    tex += [r"\end{tabular}",
            r"\caption{%s. %s}" % (("Primary utility (accuracy / RMSE)" if metric == "score"
                                    else "TabDDPM-convention utility (macro-F1 / $R^2$)"),
                                   note.replace("±", r"$\pm$").replace("†", r"$^\dagger$")
                                   .replace("‡", r"$^\ddagger$").replace("×", r"$\times$")),
            r"\end{table}"]
    return "\n".join(md) + "\n\n" + note + "\n", "\n".join(tex) + "\n"


def privacy_table(cells: dict) -> tuple[str, str]:
    mets = ("wasserstein_mean", "dcr_tabddpm_mean", "dcr_baseline", "mia_auc")
    hdr = ["dataset", "rule", "Wasserstein ↓", "DCR mean-L2 ↑", "DCR median+label ↑", "MIA AUC (→0.5)"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    tex = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{llcccc}", r"\toprule",
           r"Dataset & Rule & Wasserstein $\downarrow$ & DCR mean-L2 $\uparrow$ & DCR median+label $\uparrow$ "
           r"& MIA AUC ($\to 0.5$) \\", r"\midrule"]
    for group, prov in GROUPS:
        md.append(f"| **{group}** |" + " |" * (len(hdr) - 1))
        tex.append(r"\multicolumn{6}{l}{\textit{%s}} \\" % group)
        for ds, info in cells.items():
            if info["prov"] != prov:
                continue
            vals = {r: {m: rule_values(info["rules"][r], "mia_dev" if m == "mia_auc" else m) for m in mets}
                    for r in RULES}
            best = {}
            for m in mets:
                have = [r for r in RULES if vals[r][m] is not None]
                if have:
                    key = "mia_dev" if m == "mia_auc" else m
                    best[m] = max(have, key=lambda r: METRICS[key][0] * vals[r][m]["mean"])
            for i, r in enumerate(RULES):
                st = info["rules"][r]["status"]
                mc, tc = [], []
                for m in mets:
                    v = vals[r][m]
                    if v is None:
                        mc.append(st if st in ("missing", "partial") else "—")
                        tc.append("---")
                        continue
                    mean = v.get("raw_mean", v["mean"])
                    txt = f"{fmt_num(mean)} ± {fmt_num(v['sd'], mean)}"
                    dag = st == "partial"
                    mc.append((f"**{txt}**" if best.get(m) == r else txt) + (" †" if dag else ""))
                    t = txt.replace("±", r"$\pm$")
                    tc.append((r"\textbf{" + t + "}" if best.get(m) == r else t) + (r"$^\dagger$" if dag else ""))
                name = (ds + (" §" if info["cat_affected"] else "")) if i == 0 else ""
                md.append(f"| {name} | {RULE_LABEL[r]} | " + " | ".join(mc) + " |")
                tex.append(" & ".join([name, RULE_LABEL[r]] + tc) + r" \\")
            tex.append(r"\addlinespace")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    note = ("Held-out (transferred) rows carry the primary claims; development-set (ablated) rows are "
            "in-sample with respect to configuration selection. "
            "mean ± sd over 5 generation seeds (ddof=1; the seed-0 fixed summaries were converted from "
            "the stored ddof=0 std). Bold = best per dataset and metric (MIA: closest to 0.5). † partial. "
            "Higher DCR = synthetic rows further from training rows (more private). § trained with the "
            "categorical-input bug (cat_input=log). "
            "— = metric not recorded for that run.")
    tex += [r"\end{tabular}", r"\caption{Fidelity and privacy (TabDDPM protocol). %s}"
            % note.replace("±", r"$\pm$").replace("†", r"$^\dagger$"), r"\end{table}"]
    return "\n".join(md) + "\n\n" + note + "\n", "\n".join(tex) + "\n"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_TEXT2)
    ax.tick_params(colors=C_TEXT2, labelcolor=C_TEXT)
    ax.grid(axis="x", color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_delta_dotplot(deltas: pd.DataFrame, cells: dict):
    order = ([d for d, i in cells.items() if i["prov"] == "transfer"]
             + [d for d, i in cells.items() if i["prov"] == "own ablation"])
    panels = (("score", "Primary utility: accuracy / RMSE"), ("wasserstein_mean", "Fidelity: Wasserstein"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * len(order) + 1.6), sharey=True)
    ypos = {d: len(order) - 1 - i for i, d in enumerate(order)}
    for ax, (metric, title) in zip(axes, panels):
        _style(ax)
        ax.axvline(0, color=C_TEXT2, lw=1)
        for off, rule in ((0.15, "jenks"), (-0.15, "entmax15")):
            d = deltas[(deltas.metric == metric) & (deltas.rule == rule) & (deltas.vs == "fixed")]
            for _, r in d.iterrows():
                if not np.isfinite(r.rel_delta_pct):
                    continue
                y = ypos[r.dataset] + off
                scale = 100 / abs(r.mean_vs)
                has_ci = np.isfinite(r.ci_lo)
                if has_ci:
                    ax.plot([r.ci_lo * scale, r.ci_hi * scale], [y, y], color=C_RULE[rule], lw=2,
                            solid_capstyle="round")
                ax.plot(r.rel_delta_pct, y, "o", ms=7, mec=C_RULE[rule], mew=2,
                        mfc=C_RULE[rule] if has_ci and r.outcome != "partial" else "white")
        n_held = sum(cells[d]["prov"] == "transfer" for d in order)
        ax.axhline(len(order) - n_held - 0.5, color=C_GRID, lw=1, ls="--")
        ax.set_title(title, color=C_TEXT, fontsize=11, loc="left")
        xlabel = "Δ vs fixed, % of fixed mean (positive = better)"
        if metric == "wasserstein_mean":
            ax.set_xscale("symlog", linthresh=10)   # one rule/dataset reaches several hundred %
            xlabel += "\nsymmetric-log axis, linear within ±10%"
        ax.set_xlabel(xlabel, color=C_TEXT2)
    labels = []
    for d in order:
        st = {r: cells[d]["rules"][r]["status"] for r in RULES}
        lab = d
        if any(s in ("missing",) for s in st.values()):
            lab += " (missing)"
        elif any(s == "partial" for s in st.values()):
            lab += " †"
        labels.append(lab)
    axes[0].set_yticks([ypos[d] for d in order], labels)
    axes[0].set_ylim(-0.7, len(order) - 0.3)
    handles = [plt.Line2D([], [], color=C_RULE[r], marker="o", lw=2, ms=7, label=RULE_LABEL[r])
               for r in ("jenks", "entmax15")]
    handles.append(plt.Line2D([], [], color=C_TEXT2, marker="o", mfc="white", lw=0, ms=7,
                              label="no CI (fixed: grid summary only) or partial"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, 0.99, "Above dashed line: HELD-OUT evaluation (transferred; primary claims).  Below: DEVELOPMENT set "
             "(ablated; config selected on the test split by fixed score — in-sample, optimistic).",
             color=C_TEXT2, fontsize=8, va="top")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    _save(fig, "delta_vs_fixed_dotplot")


def fig_cd(cd_data: dict):
    subs = [k for k, _, _ in SUBSETS]
    fig, axes = plt.subplots(len(subs), 1, figsize=(6.5, 1.5 * len(subs) + 0.4))
    for ax, sub in zip(axes, subs):
        ax.set_xlim(0.8, 3.2)
        ax.set_ylim(0, 1.2)
        ax.axis("off")
        d = cd_data.get(("score", sub))
        if not d:
            ax.text(0.5, 0.5, f"{SUBSET_LABEL[sub]}: no complete datasets", transform=ax.transAxes, ha="center",
                    color=C_TEXT2)
            continue
        ax.plot([1, 3], [0.7, 0.7], color=C_TEXT2, lw=1)
        for t in (1, 1.5, 2, 2.5, 3):
            ax.plot([t, t], [0.7, 0.75 if t % 1 == 0 else 0.73], color=C_TEXT2, lw=1)
            if t % 1 == 0:
                ax.text(t, 0.8, f"{int(t)}", ha="center", color=C_TEXT2, fontsize=8)
        items = sorted(d["avg"].items(), key=lambda kv: kv[1])
        for i, (r, v) in enumerate(items):
            y = 0.45 - 0.17 * i
            ax.plot([v, v], [0.7, y], color=C_RULE[r], lw=1.5)
            ax.plot(v, 0.7, "o", color=C_RULE[r], ms=6)
            ax.text(v + 0.03, y, f"{RULE_LABEL[r]} ({v:.2f})", va="center", fontsize=9, color=C_TEXT)
        sig = np.isfinite(d["fried_p"]) and d["fried_p"] < ALPHA
        fp = f"{d['fried_p']:.4f}" if np.isfinite(d["fried_p"]) else "n/a"
        role = {"held_out": "primary", "development": "in-sample, optimistic",
                "pooled": "includes development set"}[sub]
        title = (f"{SUBSET_LABEL[sub]}, n={d['n']} ({role}): Friedman p={fp}"
                 + ("" if sig else "; not significant, no post-hoc"))
        ax.text(0.8, 1.2, title, fontsize=9, color=C_TEXT, va="top")
        if sig:
            ax.plot([1, 1 + d["cd"]], [0.95, 0.95], color=C_TEXT, lw=2)
            ax.text(1 + d["cd"] + 0.03, 0.95, f"CD = {d['cd']:.2f} (Nemenyi, α=0.05)", va="center",
                    fontsize=8, color=C_TEXT)
    fig.suptitle("Average rank on primary utility (1 = best)", color=C_TEXT, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, "cd_diagram")


def fig_wtl_heatmap(deltas: pd.DataFrame):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [C_DIV_NEG, C_DIV_MID, C_DIV_POS])
    mets = [m for m in METRICS if m in set(deltas.metric)]
    pairs = [f"{RULE_LABEL[a]}\nvs {RULE_LABEL[b]}" for a, b in PAIRS]
    panels = (("held_out", "Held-out evaluation (transferred)\nprimary claims"),
              ("development", "Development set (ablated)\nin-sample, optimistic"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(mets) + 1.8), sharey=True)
    for ax, (sub, title) in zip(axes, panels):
        Z = np.full((len(mets), len(PAIRS)), np.nan)
        txt = [["" for _ in PAIRS] for _ in mets]
        dsub = deltas[deltas.subset == sub]
        for i, m in enumerate(mets):
            for j, (a, b) in enumerate(PAIRS):
                d = dsub[(dsub.metric == m) & (dsub.rule == a) & (dsub.vs == b)]
                oc = d.outcome.value_counts()
                w, t, l = (int(oc.get(k, 0)) for k in ("win", "tie", "loss"))
                na = len(d) - w - t - l
                if w + t + l:
                    Z[i, j] = (w - l) / (w + t + l)
                txt[i][j] = (f"{w}/{t}/{l}" if w + t + l else "n/a") + (f"\n(+{na} n/a)" if na and w + t + l else "")
        im = ax.imshow(Z, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(len(mets)):
            for j in range(len(PAIRS)):
                ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=8, color=C_TEXT)
        ax.set_xticks(range(len(PAIRS)), pairs, fontsize=8)
        ax.set_yticks(range(len(mets)), [METRICS[m][2] for m in mets], fontsize=9)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(PAIRS)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(mets)), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        n = dsub.dataset.nunique()
        ax.set_title(f"{title} (n={n})", color=C_TEXT, fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.03)
    cb.set_label("(wins − losses) / tested", color=C_TEXT2)
    fig.suptitle("Win / tie / loss per dataset (95% CI excludes 0); cells W/T/L", color=C_TEXT,
                 fontsize=11, x=0.02, ha="left")
    _save(fig, "wtl_heatmap")


def fig_density_scatter(deltas: pd.DataFrame, meta: pd.DataFrame, corr: pd.DataFrame):
    m = meta.set_index("dataset")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    _style(ax)
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.axhline(0, color=C_TEXT2, lw=1)
    for rule, mk in (("jenks", "o"), ("entmax15", "s")):
        d = deltas[(deltas.metric == "score") & (deltas.rule == rule) & (deltas.vs == "fixed")
                   & deltas.rel_delta_pct.notna() & (deltas.n_rule >= EXPECTED_GEN) & (deltas.n_vs >= EXPECTED_GEN)]
        c = corr[(corr.metric == "score") & (corr.rule == rule) & (corr.predictor == "assoc_density_0.3")]
        rho = c.spearman_rho.iloc[0] if len(c) else np.nan
        n = int(c.n.iloc[0]) if len(c) else 0
        for _, r in d.iterrows():
            x = m.loc[r.dataset, "assoc_density_0.3"]
            filled = r.provenance == "transfer"
            ax.plot(x, r.rel_delta_pct, mk, ms=8, mec=C_RULE[rule], mew=2,
                    mfc=C_RULE[rule] if filled else "white")
            if abs(r.rel_delta_pct) > 2:
                ax.annotate(r.dataset, (x, r.rel_delta_pct), xytext=(6, 4), textcoords="offset points",
                            fontsize=7, color=C_TEXT2)
        ax.plot([], [], mk, ms=8, mec=C_RULE[rule], mfc=C_RULE[rule], mew=2,
                label=f"{RULE_LABEL[rule]} (Spearman ρ={rho:.2f}, n={n})" if np.isfinite(rho)
                else f"{RULE_LABEL[rule]} (n={n})")
    ax.plot([], [], "o", ms=8, mec=C_TEXT2, mfc=C_TEXT2, mew=2, label="filled = held-out (transferred)")
    ax.plot([], [], "o", ms=8, mec=C_TEXT2, mfc="white", mew=2, label="hollow = development set (in-sample)")
    ax.set_xlabel("dependency-graph density at |assoc| ≥ 0.3 (assoc_density_0.3)", color=C_TEXT2)
    ax.set_ylabel("Δ primary utility vs fixed (% of fixed)", color=C_TEXT2)
    ax.set_title("Δ utility vs graph density (exploratory)", color=C_TEXT, fontsize=11, loc="left")
    ax.text(0.99, 0.02, "points within ±2% unlabelled", transform=ax.transAxes, ha="right",
            fontsize=7, color=C_TEXT2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "delta_vs_density")


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------
QUOTE_SCALED = ("`d64` is therefore a ceiling of the shared abalone/diabetes/insurance sub-grid, not a "
                "measured optimum — and all seven targets are larger than every donor. Each target therefore "
                "gets **two jobs**: the LOO-validated configuration, and one size-scaled alternative.")
QUOTE_SCALED_ROW = ("| scaled alternative | `gnn_gatv2_layers3_dim128_topk5_heads8_noattn` | measured best on "
                    "churn2 and inside california's 1σ tie set — the two largest donors; the `d_model`-ceiling "
                    "result says the targets, all larger still, should not be run at `d64` alone |")
QUOTE_MANIFEST = ("\"Best\" here is the repository's own definition --\n"
                  "``ablation/existing_results.load_best()``, which ranks by ``metrics.score.mean``")


def _quote_ok(path: Path, text: str) -> bool:
    return text in path.read_text()


# (file, line, text that must be on that line, what it shows). Checked at
# report time so a moved line is flagged instead of silently mis-cited.
EVIDENCE = (
    ("hpc_transfer/best_manifest.py", 58, "best = existing_results.load_best(results_dir)",
     "the manifest takes `load_best`'s winner"),
    ("ablation/existing_results.py", 29, 'DEFAULT_RESULTS_DIR = "tabddpm_ablation"',
     "`load_best` reads the fixed-grid JSONs"),
    ("ablation/existing_results.py", 113, 'score = (entry.get("metrics") or {}).get("score")',
     "the ranked quantity is `metrics.score`"),
    ("ablation/existing_results.py", 120, 'if dataset not in scores or score["mean"] > scores[dataset]:',
     "argmax of `score.mean`"),
    ("ablation/ablation_runner.py", 714, 'for metric, value in run_report["metrics"]["test"].items():',
     "every CatBoost metric, `score` included, is read from the **test** block"),
    ("ablation/ablation_runner.py", 741, "for metric, values in all_test_metrics.items():",
     "those test values are averaged into `results_full_averaged.json`"),
    ("ablation/ablation_runner.py", 828, 'entry["metrics"] = json.load(fh)',
     "the grid JSON entry copies that file verbatim"),
)


def _line_ok(path: Path, line: int, text: str) -> bool:
    try:
        return text in path.read_text().splitlines()[line - 1]
    except (OSError, IndexError):
        return False


def cite(path: str, line: int, text: str) -> str:
    """`path:N` for the line that currently holds `text` (the pinned `line` is
    only a hint), so edits elsewhere in the file never break a citation. If
    the text is gone the citation is flagged instead of silently wrong."""
    try:
        lines = (REPO / path).read_text().splitlines()
    except OSError:
        return f"`{path}:{line}` **[file not found — re-check]**"
    hits = [i + 1 for i, l in enumerate(lines) if text in l]
    if not hits:
        return f"`{path}:{line}` **[cited text not found — re-check]**"
    best = min(hits, key=lambda n: abs(n - line))
    return f"`{path}:{best}`"


def _val_key_check() -> str:
    """Does any fixed-grid JSON carry a validation metric?"""
    files = sorted(RESULTS_ROOT.glob("*/ablation/*_ablation.json"))
    hits = set()
    for f in files:
        for e in json.load(open(f)):
            keys = set(e) | set((e.get("metrics") or {}))
            hits |= {k for k in keys if "val" in k.lower()}
    return (f"None of the {len(files)} fixed-grid JSONs stores a validation metric."
            if not hits else f"Validation-like keys found in the grid JSONs: {sorted(hits)} — re-check.")


def write_report(cov, cells, deltas, acr, cd_data, keff, corr, meta):
    L = []
    today = _dt.date.today().isoformat()
    n_ds = len(cells)
    complete_all = [d for d, i in cells.items()
                    if all(i["rules"][r]["status"].startswith("complete") for r in RULES)]
    missing = [d for d, i in cells.items() if d not in complete_all]
    abl_summary = [d for d, i in cells.items() if i["rules"]["fixed"]["summary"] is not None]
    L.append("# Top-k rule comparison: fixed vs Jenks vs entmax-1.5\n")
    L.append(f"_Generated {today} by `compare_topk.py` from `{rel(RESULTS_ROOT)}`. Every number below "
             "is computed by the script; rerun it to refresh._\n")

    n_held = sum(i["prov"] == "transfer" for i in cells.values() if not i["mismatch"]
                 and all(i["rules"][r]["status"].startswith("complete") for r in RULES))
    n_dev = sum(i["prov"] == "own ablation" for i in cells.values() if not i["mismatch"]
                and all(i["rules"][r]["status"].startswith("complete") for r in RULES))
    L.append("## How to read the two subsets\n")
    L.append(f"- **Held-out evaluation — transferred datasets (complete: {n_held}).** The configuration was chosen "
             "by transfer from other datasets, without seeing any of this dataset's results. **The primary claims "
             "rest on this subset.**")
    L.append(f"- **Development set — ablated datasets (complete: {n_dev}).** The configuration is the argmax of the "
             "fixed-mode score on the **test split** over the dataset's own grid (caveats 1–2). Results are "
             "in-sample with respect to that selection and therefore optimistic: most of all for fixed, the rule "
             "the selection was made with, and also relative to the TabDDPM paper, which tuned on the validation "
             "split.")
    L.append("- **Pooled** numbers mix the two and inherit the development-set bias; they are reported for "
             "completeness only.\n")
    L.append("## Coverage\n")
    L.append(f"- Datasets known: {n_ds}. Complete for all three rules: **{len(complete_all)}** "
             f"({', '.join(complete_all)}).")
    L.append("- Not complete: " + (", ".join(f"{d} ({_status_str(cells[d])})" for d in missing) or "none") + "."
             + (" fb-comments has no results (decision pending) and is reported as missing, not analysed."
                if "fb-comments" in missing else ""))
    if EXCLUDE:
        L.append(f"- Excluded entirely (jobs still running): {', '.join(sorted(EXCLUDE))}.")
    n_ph = sum(i["rules"][r].get("n_posthoc", 0) > 0 for i in cells.values() for r in RULES)
    if n_ph:
        L.append(f"- **Wasserstein and MIA AUC of {n_ph} cells are recomputed post hoc** "
                 "(`analysis/integrity/posthoc_gen_metrics.csv`): `hpc_transfer/split_runner.py` called both "
                 "functions without `dataset_name`; the TypeError was caught and only logged, so the `*_catfix` "
                 "shards hold utility and DCR only. The functions are deterministic and the recomputation "
                 "reproduces stored values of non-split runs to 1e-10 (`backfill_gen_metrics.py`).")
    if abl_summary:
        L.append(f"- **Fixed on the development set is summary-only** for {len(abl_summary)} datasets "
                 f"({', '.join(abl_summary)}): the grid JSON stores mean/std only. Their fixed utility has no "
                 "per-generation-seed values, so no utility CI or win/tie/loss exists for Jenks/entmax vs fixed on "
                 "them; fidelity/privacy metrics get an exact Welch test from the summary. The per-seed rerun in "
                 "`hpc_fixed_perseed/` (same configuration, same training seed 0) is picked up automatically once "
                 "its results exist. It only adds per-seed values: it removes neither the seed winner's curse nor "
                 "the test-split selection bias (caveats 2 and 5).")
    mism = [d for d, i in cells.items() if i["mismatch"]]
    cfg_bad = cov[cov.config_matches_dataset == False]  # noqa: E712
    L.append(f"- Configuration check: {'no mismatches' if not mism and cfg_bad.empty else 'MISMATCH: ' + ', '.join(mism + list(cfg_bad.dataset))} "
             "(gnn type, layers, d_model, top_k, heads, attention identical across the three rules).\n")

    aff = [d for d, i in cells.items() if i["cat_affected"]]
    fixed_ = [d for d, i in cells.items() if i["variant"] == "catfix"]
    prog = [f"{d} ({i['catfix_note']})" for d, i in cells.items() if i["catfix_note"]]
    L.append(f"- **Categorical-input fix:** datasets on the fixed (`*_catfix`) runs: {', '.join(fixed_) or 'none yet'}. "
             f"Datasets with categorical columns still on the original runs, marked § in the tables: "
             f"{', '.join(aff) or 'none'}." + (f" In progress: {'; '.join(prog)}." if prog else "")
             + " A dataset switches only when all three rules are complete in `*_catfix`; the two are never "
             "mixed. See caveat 10.\n")
    L.extend(key_findings(cells, deltas, acr, cd_data, keff, corr))
    L.append("## Primary utility across datasets (score = accuracy / −RMSE)\n")
    L.append("| subset | role | n | avg rank fixed | avg rank Jenks | avg rank entmax | Friedman p |")
    L.append("|---|---|---|---|---|---|---|")
    roles = {"held_out": "**primary**", "development": "in-sample, optimistic", "pooled": "completeness only"}
    for sub, _, lab in SUBSETS:
        d = cd_data.get(("score", sub))
        if not d:
            L.append(f"| {lab} | {roles[sub]} | 0 | | | | |")
            continue
        fp = f"{d['fried_p']:.4f}" if np.isfinite(d["fried_p"]) else "n/a (n<3)"
        L.append(f"| {lab} | {roles[sub]} | {d['n']} | "
                 + " | ".join(f"{d['avg'][r]:.2f}" for r in RULES) + f" | {fp} |")
    L.append("")
    L.append("Pairwise Wilcoxon signed-rank on the relative Δ (% of the reference mean), Holm-corrected within "
             "each subset; W/T/L from per-dataset 95% CIs:\n")
    L.append("| subset | pair | median Δ% | Wilcoxon p | Holm p | W/T/L | not testable |")
    L.append("|---|---|---|---|---|---|---|")
    w = acr[(acr.kind == "wilcoxon") & (acr.metric == "score")]
    w = w.assign(_o=w.subset.map({k: i for i, (k, _, _) in enumerate(SUBSETS)})).sort_values("_o", kind="stable")
    for _, r in w.iterrows():
        p = f"{r.p_value:.3f}" if np.isfinite(r.p_value) else "n/a"
        ph = f"{r.p_holm:.3f}" if np.isfinite(r.p_holm) else "n/a"
        wtl = f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}" if r.wins + r.ties + r.losses else "n/a"
        L.append(f"| {SUBSET_LABEL[r.subset]} | {RULE_LABEL[r.a]} vs {RULE_LABEL[r.b]} | {r.value:+.2f} | {p} | {ph} | "
                 f"{wtl} | {int(r.not_testable)} |")
    L.append("\nOnly the **held-out** rows support claims. On the development set the configuration was selected "
             "on the test split by fixed-mode score, which favours fixed.\n")

    L.append("## Per-dataset primary utility (Δ vs fixed, positive = better)\n")
    L.append("| dataset | subset | Jenks Δ% | outcome | entmax Δ% | outcome | entmax vs Jenks |")
    L.append("|---|---|---|---|---|---|---|")
    s = deltas[deltas.metric == "score"]
    ordered = ([d for d, i in cells.items() if i["prov"] == "transfer"]
               + [d for d, i in cells.items() if i["prov"] == "own ablation"])
    for ds in ordered:
        info = cells[ds]
        g = s[s.dataset == ds]
        if g.empty:
            continue

        def cell(a, b):
            r = g[(g.rule == a) & (g.vs == b)]
            if r.empty:
                return "—", "—"
            r = r.iloc[0]
            v = f"{r.rel_delta_pct:+.2f}" if np.isfinite(r.rel_delta_pct) else "—"
            return v, r.outcome
        j, e, ej = cell("jenks", "fixed"), cell("entmax15", "fixed"), cell("entmax15", "jenks")
        L.append(f"| {ds} | {'development' if info['prov'] == 'own ablation' else 'held-out'} | {j[0]} | {j[1]} | "
                 f"{e[0]} | {e[1]} | {ej[0]} ({ej[1]}) |")
    L.append("")

    L.append("## Privacy as a constraint\n")
    flags = deltas[deltas.privacy_flag]
    if flags.empty:
        L.append("No rule shows a DCR drop or an MIA AUC move away from 0.5 that exceeds seed noise "
                 "(95% CI) relative to the comparison rule.\n")
    else:
        L.append("Cases where the 95% CI shows a privacy regression (DCR lower, or MIA AUC further from 0.5). "
                 "\"Beyond seed noise\" is statistical, not practical: with n=5 and very small seed sd, changes "
                 "well under 1% of the reference are resolved, so read the relative size too.\n")
        for _, r in flags.iterrows():
            L.append(f"- {r.dataset}: {RULE_LABEL[r.rule]} vs {RULE_LABEL[r.vs]}, {METRICS[r.metric][2]}: "
                     f"{fmt_num(r.mean_vs)} → {fmt_num(r.mean_rule)} (Δ={r.delta:+.4g}"
                     + (f", {r.rel_delta_pct:+.1f}% of reference" if np.isfinite(r.rel_delta_pct) else "")
                     + (f", {r.delta_over_sd_b:+.1f} sd of reference" if np.isfinite(r.delta_over_sd_b) else "")
                     + f"; {r.ci_method})")
        L.append("")
    fam = deltas[deltas.vs == "fixed"].copy()
    L.append("Fidelity (Wasserstein) W/T/L vs fixed: " + "; ".join(
        f"{RULE_LABEL[a]} {_wtl(fam[(fam.rule == a) & (fam.metric == 'wasserstein_mean')])}"
        for a in ("jenks", "entmax15")) + ".\n")

    L.append("## Mechanism: effective neighbours per node (generation phase)\n")
    if keff.empty:
        L.append("No `topk_stats.csv` found.\n")
    else:
        L.append("| dataset | N nodes | fixed k | Jenks k_eff | entmax k_eff |")
        L.append("|---|---|---|---|---|")
        for ds in keff.dataset.unique():
            g = keff[keff.dataset == ds].set_index("rule")
            kf = g.k_fixed_effective.iloc[0]
            vals = [f"{g.loc[r, 'k_eff_sample']:.2f}" if r in g.index and np.isfinite(g.loc[r, "k_eff_sample"])
                    else "—" for r in ("jenks", "entmax15")]
            L.append(f"| {ds} | {int(g.n_nodes.iloc[0])} | {int(kf)} | {vals[0]} | {vals[1]} |")
        for r in ("jenks", "entmax15"):
            g = keff[keff.rule == r]
            L.append(f"\n- {RULE_LABEL[r]}: fewer neighbours than fixed on {int((g.vs_fixed == 'fewer').sum())}/"
                     f"{len(g)} datasets; median k_eff/k_fixed = {g.ratio_to_fixed.median():.2f}.")
        L.append("\nk_eff = mean over generation-time calls, blocks and timestep buckets of the off-diagonal "
                 "neighbours kept per node, weighted by sample count. Fixed k = min(top_k, N−1).\n")
        tr = keff[keff.get("topk_log_truncated", False) == True]  # noqa: E712
        if len(tr):
            seeds = tr.sample_gen_seeds_logged.replace("", "none").value_counts().to_dict()
            L.append(f"**Truncated logs:** {len(tr)}/{len(keff)} `topk_stats.csv` / `jenks_gate_stats.csv` "
                     "files in the results tree end mid-line (an incomplete copy). Their training rows are "
                     "intact, but the generation-phase rows cover only the gen seeds listed in "
                     f"`effective_k.csv` (`sample_gen_seeds_logged`; here: {seeds}), so k_eff and the Jenks "
                     "would-reject shares above rest on those seeds only. Re-copy the files and rerun.\n")
        L.extend(jenks_gate_section(cells, deltas, keff))

    L.append("## Δ utility vs dependency-graph meta-features (exploratory)\n")
    c = corr[corr.metric == "score"]
    L.append("| rule | predictor | n | Spearman ρ | p | Holm p (8 tests) |")
    L.append("|---|---|---|---|---|---|")
    for _, r in c.iterrows():
        rho = f"{r.spearman_rho:+.2f}" if np.isfinite(r.spearman_rho) else "n/a"
        p = f"{r.p_value:.3f}" if np.isfinite(r.p_value) else "n/a"
        ph = f"{r.p_holm:.3f}" if np.isfinite(r.p_holm) else "n/a"
        L.append(f"| {RULE_LABEL[r.rule]} | {r.predictor} | {r.n} | {rho} | {p} | {ph} |")
    L.append("\nExploratory only: n is the number of datasets (small), no causal reading; development-set "
             "datasets enter with a fixed mean that is in-sample with respect to selection (optimistic), and "
             "held-out alone is too small (n=4) for a correlation. `assoc_*` (rank-based specification matrix: |Spearman|, Cramér's V, "
             "correlation ratio) is the predictor. `mg_*` (the repo's static `build_static_adjacency` graph: "
             "|Pearson| / Cramér's V / normalised MI, pre-thresholded at 0.1) is reported only descriptively "
             "below: it is a second encoding of the same data dependencies, so testing it too would double the "
             "number of exploratory tests without adding an independent variable. Note that it is *not* "
             "produced by the top-k rule — the runs use `graph_mode=\"dynamic\"`, whose learned adjacency is "
             "what the rules sparsify, and neither meta-feature reads it.\n")
    mm = meta.set_index("dataset")
    L.append("| dataset | assoc_density_0.3 | mg_density_0.3 | mg_n_nodes |")
    L.append("|---|---|---|---|")
    for ds in cells:
        if ds in mm.index:
            L.append(f"| {ds} | {mm.loc[ds, 'assoc_density_0.3']:.3f} | {mm.loc[ds, 'mg_density_0.3']:.3f} | "
                     f"{int(mm.loc[ds, 'mg_n_nodes'])} |")
    L.append("")

    L.append("## Secondary utility (TabDDPM convention: macro-F1 / R²)\n")
    w2 = acr[(acr.kind == "wilcoxon") & (acr.metric == "tabddpm_util")]
    for sub, _, lab in SUBSETS:
        d = cd_data.get(("tabddpm_util", sub))
        if not d:
            continue
        fp = f"{d['fried_p']:.4f}" if np.isfinite(d["fried_p"]) else "n/a"
        parts = "; ".join(f"{RULE_LABEL[r.a]} vs {RULE_LABEL[r.b]} W/T/L "
                          + (f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}" if r.wins + r.ties + r.losses
                             else "n/a")
                          for _, r in w2[w2.subset == sub].iterrows())
        L.append(f"- {lab} (n={d['n']}): avg ranks " + ", ".join(f"{RULE_LABEL[r]} {d['avg'][r]:.2f}" for r in RULES)
                 + f"; Friedman p={fp}; {parts}.")
    L.append("\nClass imbalance (majority/minority ratio) is shown per dataset in `tables/utility_secondary.md`; "
             "accuracy is least informative where it is large (wilt 17.5, buddy 6.1, churn2 3.9, default 3.5, "
             "adult 3.2).\n")

    L.append("## Caveats for the thesis text\n")
    best = pd.read_csv(BEST_CSV).set_index("dataset")
    diff = [d for d, i in cells.items() if i["prov"] == "own ablation" and i["exp_name"]
            and d in best.index and best.loc[d, "exp_name"] != i["exp_name"]]
    n_abl = sum(i["prov"] == "own ablation" for i in cells.values())
    L.append(f"1. **Actual configuration-selection rule.** The ablated datasets' Jenks/entmax (and hence fixed) "
             f"configuration is the argmax of `score.mean` over the fixed grid, via `best_manifest.py`:")
    L.append("")
    L.append("   > " + QUOTE_MANIFEST.replace("\n", "\n   > ")
             + ("" if _quote_ok(BEST_MANIFEST_PY, QUOTE_MANIFEST) else "  \n   > **[quote not found in file — re-check]**"))
    L.append("")
    L.append(f"   It differs from the `config_id` in `hpc_transfer/best_per_dataset.csv` / `predictions.md` "
             f"(composite rank, ties broken on parameter count) in **{len(diff)}/{n_abl}** ablated datasets "
             f"({', '.join(diff)}). Describe the argmax-score rule in the text.")
    L.append("2. **The selection score is the test-split score.** `load_best` ranks on `metrics.score.mean`, and "
             "that value is the mean of the CatBoost metrics read from the *test* part of each evaluation report:")
    L.append("")
    for path, line, text, what in EVIDENCE:
        L.append(f"   - {cite(path, line, text)} — {what}")
    L.append("")
    L.append(f"   {_val_key_check()} So the development-set configuration was chosen on the same split its "
             "results are reported on: those results are in-sample and optimistic (most for fixed), and are not "
             "comparable with the TabDDPM paper, which tuned on the validation split. A validation-based "
             "re-selection needs per-configuration validation metrics (e.g. per-run `results_catboost.json` on "
             "the cluster), if they still exist.")
    trf_cfgs = sorted({i["exp_name"] for i in cells.values() if i["prov"] == "transfer" and i["exp_name"]})
    L.append(f"3. **Transferred datasets ran the scaled alternative, not the LOO-validated primary.** Configurations "
             f"present in the results: {', '.join(f'`{c}`' for c in trf_cfgs) or 'none'}. The LOO-validated global "
             "best is `gnn_gatv2_layers3_dim64_topk3_heads4_noattn`. `predictions.md` gives the reason for the "
             "scaled job:")
    L.append("")
    for q in (QUOTE_SCALED, QUOTE_SCALED_ROW):
        L.append("   > " + q + ("" if _quote_ok(PREDICTIONS_MD, q) else "  **[quote not found — re-check]**"))
        L.append("")
    L.append("   `predictions.md` plans *both* jobs per target; only the scaled one exists in the results tree, "
             "so no transferred result speaks for the LOO-validated configuration.")
    L.append("4. **n = 5 generation seeds of one trained model per cell.** Training-seed variance is not in any CI; "
             "bootstrap CIs with n=5 are narrow and the Welch tests have low power. Treat per-dataset W/T/L as "
             "descriptive.")
    L.append("5. **Seed winner's curse on the development set.** The fixed number is the maximum over the "
             "dataset's grid (up to 24 configurations) at training seed 0, so fixed is favoured. The per-seed "
             "rerun uses the same configuration and the same training seed 0: it adds per-generation-seed values "
             "(CIs, win/tie/loss) but removes neither this bias nor the test-split selection bias (caveat 2). The "
             "development set stays in-sample regardless.")
    L.append(f"6. **Missing datasets:** {', '.join(missing) or 'none'} — excluded from every across-dataset test.")
    L.append("7. **Units.** Across-dataset Wilcoxon and the meta-feature correlations use Δ as % of the reference "
             "mean, because regression RMSE (e.g. insurance ≈ 4.7k) and accuracy are not on one scale. MIA is "
             "scored as |mean AUC − 0.5|.")
    L.append("8. **Two DCR variants.** `dcr_tabddpm_mean` (mean L2, no label) is the only one recorded for every "
             "run; `dcr_baseline` (median, label included) is missing from some seed-0 fixed grids. Only "
             "`dcr_baseline` uses the same recipe as the TabDDPM baseline `privacy.json`, so only it may be "
             "compared with those numbers.")
    L.append("9. **The hpc_density README's Jenks densities describe the GATED variant.** Its \"Jenks ~1–2.4 "
             "neighbours\" line was written in de700e6 (2026-09-22), when the call site still passed "
             "`use_gate_rule=True` (also in 3394bd8, the code shipped for the earlier ARIS Jenks runs). Under the gate, "
             "every rejected row falls back to k_min = 1, which pulls the mean budget towards 1. All Jenks runs "
             "here are plain Jenks (see 'Jenks acceptance test'), so their k_eff "
             + (f"({keff[keff.rule == 'jenks'].k_eff_sample.min():.1f}–{keff[keff.rule == 'jenks'].k_eff_sample.max():.1f}) "
                if not keff.empty else "")
             + "is not comparable with the README figure; do not quote the README for these runs.")
    L.append("10. **Categorical-input bug (all runs before the fix).** The GNN denoiser embedded each categorical "
             "column from the diffusion's log one-hot (entries ≈ −69) through its own `Linear(K, d_model)`, which "
             "makes categorical node embeddings ~200× the timestep embedding at initialization; some categorical "
             "columns then learn to copy their noisy input instead of denoising it (multinomial loss at or above "
             "an always-uniform predictor on cardio, churn2, abalone). It affects every rule the same way on a "
             "given dataset, but all absolute numbers on datasets with categorical columns, marked §. The fix "
             "(`cat_input=\"prob\"`) is rerun into `*_catfix` trees; evidence and the list of affected runs are "
             "in `analysis/dependency_fidelity/REPORT.md` ('Categorical-input bug').")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")


JENKS_EVIDENCE = (
    ("adaptive_adjacency.py", 38, "sdcm = (a2 - a1 ** 2 / m) + (b2 - b1 ** 2 / (n - m))",
     "the 2-class split minimises the within-class sum of squared deviations of the sorted raw scores"),
    ("adaptive_adjacency.py", 43, "k = best.squeeze(-1) + 1", "k = size of the strong class"),
    ("adaptive_adjacency.py", 47, "accept = mu_s > 0.0",
     "the acceptance test (\"gate\"): accept only if the strong-class mean raw score is > 0, "
     "i.e. its sigmoid edge weight is > 0.5"),
    ("adaptive_adjacency.py", 53, "k = torch.where(accept, k, torch.full_like(k, k_min))",
     "on rejection the row falls back to **k_min** (not to the fixed top_k)"),
    ("tab_ddpm/graph_builder.py", 305, "use_gate_rule=False,",
     "the model's call site switches the test off: every split is accepted and k_min never binds"),
    ("tab_ddpm/graph_builder.py", 333, "(mu_strong <= 0).float().mean(dim=-1).detach()",
     "`jenks_gate_stats.csv` logs the share of rows where the test WOULD have fired"),
)


def jenks_gate_section(cells, deltas, keff) -> list[str]:
    L = ["### Jenks acceptance test (counterfactual only)\n"]
    L.append("What the code does:\n")
    for path, line, text, what in JENKS_EVIDENCE:
        L.append(f"- {cite(path, line, text)} — {what}")
    starts = []
    for info in cells.values():
        src = info["rules"]["jenks"]["source"]
        mp = (Path(src) if os.path.isabs(src) else REPO / src) / "run_meta.json" if src else None
        if mp and mp.exists():
            starts.append(json.load(open(mp)).get("started_utc", ""))
    starts = sorted(x for x in starts if x)
    L.append("")
    L.append(f"**These runs used plain Jenks: no split was ever rejected.** The acceptance test was switched off "
             f"in commit 3b8b8b3 (2026-09-23 21:40 UTC), which also introduced `jenks_gate_stats.csv`. All "
             f"{len(starts)} local Jenks runs started after it (earliest {starts[0] if starts else 'n/a'}), and "
             "every one has that file, which only the post-switch code writes. So `jenks_gate_stats.csv` does "
             "not record accept/reject decisions. It records how often the dropped test *would* have "
             "rejected the split. The k of those rows is not logged separately, so \"k when accepted vs "
             "rejected\" cannot be computed from these files.\n")
    j = keff[keff.rule == "jenks"].copy() if not keff.empty else pd.DataFrame()
    if j.empty or "jenks_would_gate_frac_sample" not in j:
        L.append("No `jenks_gate_stats.csv` found.\n")
        return L
    s = deltas[(deltas.metric == "score") & (deltas.rule == "jenks") & (deltas.vs == "fixed")].set_index("dataset")
    L.append("| dataset | subset | fixed k | Jenks k_eff | would-reject share, generation | would-reject share, "
             "last 20% of training | Jenks vs fixed (score) |")
    L.append("|---|---|---|---|---|---|---|")
    j["_o"] = j.provenance.map({"transfer": 0, "own ablation": 1})
    for _, r in j.sort_values(["_o", "dataset"]).iterrows():
        sub = "held-out" if r.provenance == "transfer" else "development"
        if r.dataset in s.index:
            d = s.loc[r.dataset]
            res = (f"{d.rel_delta_pct:+.2f}% ({d.outcome})" if r.provenance == "transfer"
                   else f"{d.rel_delta_pct:+.2f}% (no per-seed fixed)" if d.fixed_summary_only
                   else f"{d.rel_delta_pct:+.2f}% ({d.outcome})")
        else:
            res = "—"
        g1 = f"{r.jenks_would_gate_frac_sample:.1%}" if np.isfinite(r.jenks_would_gate_frac_sample) else "—"
        g2 = (f"{r.jenks_would_gate_frac_train_last20pct:.1%}"
              if np.isfinite(r.get("jenks_would_gate_frac_train_last20pct", np.nan)) else "—")
        L.append(f"| {r.dataset} | {sub} | {int(r.k_fixed_effective)} | {r.k_eff_sample:.2f} | {g1} | {g2} | {res} |")
    L.append("")
    held = j[j.provenance == "transfer"].set_index("dataset")
    if len(held):
        byout = {}
        for ds in held.index:
            if ds in s.index:
                byout.setdefault(s.loc[ds].outcome, []).append(
                    f"{ds} {held.loc[ds, 'jenks_would_gate_frac_sample']:.1%}")
        L.append("Held-out, descriptive: would-reject share at generation by Jenks-vs-fixed outcome — "
                 + "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(byout.items())) + ".")
    rho = stats.spearmanr(j.jenks_would_gate_frac_sample, j.k_eff_sample / j.k_fixed_effective)
    L.append(f"Across all {len(j)} Jenks runs, the would-reject share vs k_eff/k_fixed: Spearman ρ={rho.statistic:+.2f} "
             f"(p={rho.pvalue:.3f}), descriptive only.")
    L.append("\nThe question \"do Jenks ties coincide with the gate rejecting?\" has no gate to test here. What can be "
             "said is whether the rows that a gate would have rejected (weak strong class) are common where "
             "Jenks ties or gains, and the table above gives that. The share is a property of the learned "
             "scores under plain Jenks. It does not predict what a gated run would have done, because gating "
             "changes the adjacency and therefore training.\n")
    return L


def key_findings(cells, deltas, acr, cd_data, keff, corr) -> list[str]:
    """Numbers-only headline bullets; every sentence is derived from the data."""
    L = ["## Key findings\n"]
    w = acr[(acr.kind == "wilcoxon") & (acr.metric == "score")]

    def pair(sub, a, b):
        r = w[(w.subset == sub) & (w.a == a) & (w.b == b)]
        return r.iloc[0] if len(r) else None

    t = cd_data.get(("score", "held_out"))
    if t:
        parts = []
        for a, b in PAIRS:
            r = pair("held_out", a, b)
            if r is not None:
                parts.append(f"{RULE_LABEL[a]} vs {RULE_LABEL[b]} {int(r.wins)}/{int(r.ties)}/{int(r.losses)} "
                             f"(median Δ {r.value:+.2f}%)")
        fp = f"{t['fried_p']:.3f}" if np.isfinite(t["fried_p"]) else "n/a"
        L.append(f"- **Held-out evaluation (transferred, n={t['n']}) — primary:** average rank "
                 + ", ".join(f"{RULE_LABEL[r]} {t['avg'][r]:.2f}" for r in RULES)
                 + f"; Friedman p={fp}. W/T/L: " + "; ".join(parts) + ". With n=4 no across-dataset test "
                 "can reach significance (smallest exact two-sided Wilcoxon p is 0.125).")
    dv = cd_data.get(("score", "development"))
    if dv:
        fp = dv["fried_p"]
        ej = pair("development", "entmax15", "jenks")
        L.append(f"- **Development set (ablated, n={dv['n']}) — in-sample, optimistic for fixed:** average rank "
                 + ", ".join(f"{RULE_LABEL[r]} {dv['avg'][r]:.2f}" for r in RULES)
                 + (f"; Friedman p={fp:.4f}" if np.isfinite(fp) else "") + "."
                 + (f" entmax-1.5 vs Jenks W/T/L {int(ej.wins)}/{int(ej.ties)}/{int(ej.losses)} "
                    f"(median Δ {ej.value:+.2f}%)." if ej is not None else ""))
    a = cd_data.get(("score", "pooled"))
    if a:
        fp = a["fried_p"]
        L.append(f"- **Pooled (n={a['n']}; inherits the development-set bias, completeness only):** average rank "
                 + ", ".join(f"{RULE_LABEL[r]} {a['avg'][r]:.2f}" for r in RULES)
                 + f"; Friedman p={fp:.4f}"
                 + (" (borderline)" if np.isfinite(fp) and 0.04 <= fp <= 0.06 else "") + "."
                 + "".join(f" Nemenyi {RULE_LABEL[x]} vs {RULE_LABEL[y]} p={p:.3f}." for x, y, p in a["nemenyi"]))
    s = deltas[(deltas.metric == "score") & (deltas.vs == "fixed") & (deltas.rel_delta_pct.abs() > 5)]
    if len(s):
        s = s.sort_values("subset", key=lambda c: c.map({"held_out": 0, "development": 1}), kind="stable")
        L.append("- **Large utility changes vs fixed (|Δ| > 5%):** " + "; ".join(
            f"{r.dataset} {RULE_LABEL[r.rule]} {r.rel_delta_pct:+.1f}% "
            + ("[held-out, " + r.outcome + "]" if r.subset == "held_out"
               else "[development" + (", no per-seed fixed" if r.fixed_summary_only else ", " + r.outcome) + "]")
            for _, r in s.iterrows()) + ".")
    f = deltas[deltas.privacy_flag & (deltas.vs == "fixed")]
    if len(f):
        fh = f[f.subset == "held_out"]
        L.append(f"- **Privacy vs fixed:** {len(f)} resolved regressions across {f.dataset.nunique()} datasets "
                 f"({len(fh)} on held-out: {', '.join(sorted(fh.dataset.unique())) or 'none'}); "
                 "the largest relative DCR drops are "
                 + "; ".join(f"{r.dataset} {RULE_LABEL[r.rule]} {r.rel_delta_pct:+.1f}%"
                             for _, r in f[f.rel_delta_pct.notna()].nsmallest(3, "rel_delta_pct").iterrows())
                 + ". See 'Privacy as a constraint'.")
    if not keff.empty:
        L.append("- **Mechanism:** median k_eff/k_fixed is "
                 + ", ".join(f"{RULE_LABEL[r]} {keff[keff.rule == r].ratio_to_fixed.median():.2f}"
                             for r in ("jenks", "entmax15"))
                 + "; both rules keep fewer neighbours than fixed on some datasets and more on others "
                 "(table below), so neither is uniformly sparser.")
    c = corr[(corr.metric == "score")]
    if len(c):
        best = c.loc[c.p_value.idxmin()]
        L.append(f"- **Meta-features (exploratory):** strongest association is {RULE_LABEL[best.rule]} Δ vs "
                 f"{best.predictor} (ρ={best.spearman_rho:+.2f}, p={best.p_value:.3f}, Holm p={best.p_holm:.3f}, "
                 f"n={best.n}). assoc_density_0.3: "
                 + ", ".join(f"{RULE_LABEL[r.rule]} ρ={r.spearman_rho:+.2f} (p={r.p_value:.3f})"
                             for _, r in c[c.predictor == "assoc_density_0.3"].iterrows()) + ".")
    L.append("")
    return L


def _status_str(info) -> str:
    return ", ".join(f"{r} {info['rules'][r]['status']}" for r in RULES
                     if not info["rules"][r]["status"].startswith("complete"))


def _wtl(d: pd.DataFrame) -> str:
    oc = d.outcome.value_counts()
    return f"{int(oc.get('win', 0))}/{int(oc.get('tie', 0))}/{int(oc.get('loss', 0))}"


# ---------------------------------------------------------------------------
def main():
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    meta = pd.read_csv(META_CSV)

    cov, cells = discover()
    cov.to_csv(OUT / "coverage.csv", index=False)
    pdm = per_dataset_table(cells)
    pdm.to_csv(OUT / "per_dataset_metrics.csv", index=False)
    deltas = deltas_table(cells)
    deltas.to_csv(OUT / "deltas.csv", index=False)
    acr, cd_data = across(cells, deltas)
    acr.to_csv(OUT / "across_datasets.csv", index=False)
    keff = effective_k(cells, meta)
    keff.to_csv(OUT / "effective_k.csv", index=False)
    corr = correlations(deltas, meta, cells)
    corr.to_csv(OUT / "correlations.csv", index=False)

    for metric, name in (("score", "utility_primary"), ("tabddpm_util", "utility_secondary")):
        md, tex = utility_table(cells, metric, meta)
        (OUT / "tables" / f"{name}.md").write_text(md)
        (OUT / "tables" / f"{name}.tex").write_text(tex)
    md, tex = privacy_table(cells)
    (OUT / "tables" / "privacy_fidelity.md").write_text(md)
    (OUT / "tables" / "privacy_fidelity.tex").write_text(tex)

    fig_delta_dotplot(deltas, cells)
    fig_cd(cd_data)
    fig_wtl_heatmap(deltas)
    fig_density_scatter(deltas, meta, corr)
    write_report(cov, cells, deltas, acr, cd_data, keff, corr, meta)

    # ---- terminal summary --------------------------------------------------
    print(f"Results root: {RESULTS_ROOT}")
    st = cov.pivot(index="dataset", columns="rule", values="status")[list(RULES)]
    print("\nCoverage:\n" + st.to_string())
    print("\nPrimary utility (score), average rank (1 = best):")
    for sub, _, lab in SUBSETS:
        d = cd_data.get(("score", sub))
        if d:
            fp = f"{d['fried_p']:.4f}" if np.isfinite(d["fried_p"]) else "n/a"
            print(f"  {lab:34s} n={d['n']:2d}  "
                  + "  ".join(f"{RULE_LABEL[r]}={d['avg'][r]:.2f}" for r in RULES) + f"  Friedman p={fp}")
    w = acr[(acr.kind == "wilcoxon") & (acr.metric == "score")]
    w = w.assign(_o=w.subset.map({k: i for i, (k, _, _) in enumerate(SUBSETS)})).sort_values("_o", kind="stable")
    print("\nPairwise (score): median Δ%, Holm p, W/T/L [not testable]")
    for _, r in w.iterrows():
        ph = f"{r.p_holm:.3f}" if np.isfinite(r.p_holm) else "n/a"
        print(f"  {r.subset:11s} {RULE_LABEL[r.a]:>10s} vs {RULE_LABEL[r.b]:<10s} {r.value:+7.2f}%  p_holm={ph:>5s}  "
              f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)} [{int(r.not_testable)}]")
    flags = deltas[deltas.privacy_flag]
    print(f"\nPrivacy regressions beyond seed noise: {len(flags)}")
    for _, r in flags.iterrows():
        print(f"  {r.dataset}: {RULE_LABEL[r.rule]} vs {RULE_LABEL[r.vs]} {r.metric} Δ={r.delta:+.4g}")
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()

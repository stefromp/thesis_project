#!/usr/bin/env python3
"""
TabDiff-transfer study: our runs vs the numbers reported in the TabDiff paper.
Rerunnable; reads only finished evaluation outputs.

    python3 analysis/tabdiff/compare.py [--exp_root transfer_runs]

Computed (ours, marked "computed"): for each dataset, the per-table metrics
TabDiff's own code wrote into <exp_root>/exp_tabdiff/tabdiff_<ds>/gnn/<combo>/tabdiff_eval/
  all_results.csv  shape, trend (error %), mle, c2st (x100)   Trainer.report_test
  quality.csv      alpha, beta (%)                             eval/eval_quality.py
and DCR (%) from the tabdiff_<ds>_dcr run's all_results.csv (report_test_dcr).
mean +/- std over the 20 tables, std with ddof=1 -- exactly TabDiff's
aggregation (pandas .std() in trainer.report_test). C2ST is divided by 100 to
match the paper's Table 9 (fractions).
Reported (marked "reported"): analysis/tabdiff/reported.py, transcribed from
the paper's Tables 1-3 and 7-10.

Writes analysis/tabdiff/: RESULTS.md, results_long.csv, provenance.json.

    python3 analysis/tabdiff/compare.py --route_a [--exp_root transfer_runs]

Route A (tabdiff_gnn/: AttnGNN as the denoiser inside TabDiff) instead.
TabDiff is NOT re-run: its numbers are reported -- primary reference GATD's
TabDiff (3 training seeds x 20 tables, reported_gatd.py), secondary the TabDiff
paper, and tabpc's TabDiff for wNMIS / C2ST-XGB. Columns "reported: ..." vs
"computed: TabDiff + AttnGNN (d_model 256)", plus parameter counts ->
ROUTE_A.md, route_a_long.csv, route_a_provenance.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "hpc_transfer", "tabdiff_study"))
import reported as RP  # noqa: E402
import study as S  # noqa: E402

DS = S.ORDER                       # magic, shoppers, adult, default, beijing, news
PAPER_DS = ["adult", "default", "shoppers", "magic", "beijing", "news"]   # the paper's column order
METRICS = [  # key, title, unit, better, decimals
    ("shape", "Shape error rate (%) -- Table 1", "%", "lower", 2),
    ("trend", "Trend error rate (%) -- Table 2", "%", "lower", 2),
    ("mle",   "MLE: AUC (classification) / RMSE (beijing, news) -- Table 3", "", "AUC higher / RMSE lower", 3),
    ("alpha", "alpha-precision (%) -- Table 7", "%", "higher", 2),
    ("beta",  "beta-recall (%) -- Table 8", "%", "higher", 2),
    ("c2st",  "C2ST detection score -- Table 9", "", "higher", 4),
    ("dcr",   "DCR (%) -- Table 10 (closer to 50 is better)", "%", "~50", 2),
]
OURS = "GNN (ours, transferred)"


def _read(path, cols):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks {missing}")
    return df[cols]


def computed(exp_root: str) -> tuple[pd.DataFrame, dict]:
    rows, prov = [], {}
    for ds in DS:
        run = S.run_dir(exp_root, S.target_name(ds))
        ev = os.path.join(run, "tabdiff_eval")
        series = {}
        main = _read(os.path.join(ev, "all_results.csv"), ["shape", "trend", "mle", "c2st"])
        if main is not None:
            for k in main.columns:
                series[k] = main[k] / (100.0 if k == "c2st" else 1.0)
        q = _read(os.path.join(ev, "quality.csv"), ["alpha", "beta"])
        if q is not None:
            series.update({k: q[k] for k in q.columns})
        if S.HAS_DCR[ds]:
            drun = S.run_dir(exp_root, S.target_name(ds, True))
            d = _read(os.path.join(drun, "tabdiff_eval", "all_results.csv"), ["dcr"])
            if d is not None:
                series["dcr"] = d["dcr"]
                prov[f"{ds}/dcr"] = _run_prov(drun)
        if series:
            prov[ds] = _run_prov(run)
        for k, s in series.items():
            v = s.to_numpy(dtype=float)
            rows.append(dict(dataset=ds, metric=k, method=OURS, kind="computed",
                             mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
                             n=len(v), source=os.path.relpath(run, REPO)))
    return pd.DataFrame(rows), prov


def _run_prov(run: str) -> dict:
    out = {"run": os.path.relpath(run, REPO)}
    for name in ("run_meta.json", os.path.join("tabdiff_eval", "eval_meta.json")):
        p = os.path.join(run, name)
        if os.path.exists(p):
            out[name] = json.load(open(p))
    meta = out.get("run_meta.json", {})
    out["summary"] = {k: meta.get(k) for k in ("source_dataset", "train_setup", "git_commit",
                                                "n_sample_runs", "num_samples_per_run", "checkpoint_used")}
    return out


def reported() -> pd.DataFrame:
    rows = []
    for metric, table in RP.REPORTED.items():
        for method, cells in table.items():
            for ds, c in zip(RP.DATASETS, cells):
                if ds not in DS or c is None:
                    continue
                if c == "OOM":
                    rows.append(dict(dataset=ds, metric=metric, method=method, kind="reported",
                                     mean=np.nan, std=np.nan, n=np.nan, source="OOM in paper"))
                    continue
                rows.append(dict(dataset=ds, metric=metric, method=method, kind="reported",
                                 mean=c[0], std=np.nan if c[1] is None else c[1], n=np.nan,
                                 source=f"{RP.SOURCE}, {RP.TABLE_OF[metric]}"))
    return pd.DataFrame(rows)


def _fmt(r, dec):
    if r is None:
        return "—"
    if r["source"] == "OOM in paper":
        return "OOM"
    if np.isnan(r["mean"]):
        return "—"
    s = f"{r['mean']:.{dec}f}"
    if not np.isnan(r["std"]):
        s += f" ± {r['std']:.{dec}f}"
    return s + (" †" if r["kind"] == "computed" else "")


def markdown(long: pd.DataFrame, prov: dict, exp_root: str) -> str:
    out = ["# TabDiff benchmark: transferred GNN vs the TabDiff paper", "",
           f"_Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by `analysis/tabdiff/compare.py` "
           f"from `{os.path.relpath(S.results_root(exp_root), REPO)}`._", "",
           "**†** = computed by us with TabDiff's own evaluation code "
           f"([{S.TABDIFF_URL}]({S.TABDIFF_URL}) @ `{S.TABDIFF_COMMIT}`), mean ± std (ddof=1) over "
           f"{S.N_SAMPLE_RUNS} sampled tables of n_train rows each, on TabDiff's own splits. "
           "No mark = reported in the TabDiff paper (Shi et al., ICLR 2025), "
           "transcribed in `analysis/tabdiff/reported.py`; their TabSyn row is the authors' "
           "reproduction, the other baselines come from Zhang et al. (2024). "
           "— = not run yet / not reported. Diabetes is not part of this study.", "",
           "Source dataset per TabDiff dataset (TabDDPM-tuned training setup + GNN configuration "
           "transferred from it): " + ", ".join(f"{d} ← {S.SOURCES[d]}" for d in PAPER_DS) + ".", ""]
    methods = ["Real"] + RP.METHODS + [OURS]
    for key, title, _, better, dec in METRICS:
        sub = long[long.metric == key]
        out += [f"## {title}", "", f"Better: {better}.", "",
                "| Method | " + " | ".join(PAPER_DS) + " |",
                "|---|" + "---|" * len(PAPER_DS)]
        for m in methods:
            msub = sub[sub.method == m]
            if msub.empty:
                if m == OURS:
                    out.append(f"| {m} | " + " | ".join("—" for _ in PAPER_DS) + " |")
                continue
            cells = []
            for ds in PAPER_DS:
                r = msub[msub.dataset == ds]
                cells.append(_fmt(r.iloc[0] if len(r) else None, dec))
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## Deviations from TabDiff's own protocol", ""] + [f"- {d}" for d in S.DEVIATIONS] + [""]
    vers = {k: v.get("tabdiff_eval/eval_meta.json", {}) for k, v in prov.items()}
    used = sorted({json.dumps(step.get("versions", {}), sort_keys=True)
                   for m in vers.values() for step in m.values() if isinstance(step, dict)})
    out += ["## Evaluator environment", "",
            f"Pins: {S.EVAL_PINS} (eval venv), {S.QUALITY_PINS} (synthcity venv).", ""]
    out += [f"- `{u}`" for u in used] or ["- (no evaluated runs yet)"]
    out += ["", "Per-run provenance (run_meta.json, eval_meta.json): `analysis/tabdiff/provenance.json`.", ""]
    return "\n".join(out)


# ===========================================================================
# Route A: AttnGNN as the denoiser inside TabDiff (tabdiff_gnn/)
# ===========================================================================
# TabDiff is not re-run: reported references (GATD's 3x20 re-run = primary,
# TabDiff paper = secondary, tabpc's 5x1 for wNMIS / C2ST-XGB) vs the one
# computed arm, TabDiff + AttnGNN with d_model 256.
RA_ARM = ("attngnn", "computed: TabDiff + AttnGNN (d_model 256)")
RA_REFS = [("gatd", "reported: TabDiff, GATD re-run (3 seeds × 20 tables) — primary"),
           ("paper", "reported: TabDiff paper (1 model × 20 tables) — secondary"),
           ("tabpc", "reported: TabDiff, tabpc re-run (5 models × 1 table)")]
RA_METRICS = [  # key, title, decimals
    ("shape", "Shape error rate (%) ↓", 2),
    ("trend", "Trend error rate (%) ↓", 2),
    ("mle", "MLE-1: AUROC ↑ (classification) / RMSE ↓ (regression) -- TabDiff's 'mle'", 3),
    ("mle_alt", "MLE-2: F1 ↑ (classification) / R2 ↑ (regression)", 3),
    ("c2st", "C2ST (logistic regression, TabDiff's) ↑", 4),
    ("alpha", "alpha-precision (%) ↑", 2),
    ("beta", "beta-recall (%) ↑", 2),
    ("wnmis", "wNMIS (tabpc) ↑", 4),
    ("c2st_xgb", "C2ST (XGBoost, tabpc) ↑", 4),
]   # no DCR in Route A
RA_DS = ["magic", "shoppers", "adult", "default", "beijing", "news", "diabetes"]


def _ra_imports():
    sys.path.insert(0, os.path.join(REPO, "tabdiff_gnn"))
    import config as RC  # noqa: E402
    import reported_gatd as RG  # noqa: E402
    return RC, RG


def ra_reference(metric: str, ds: str, task: str) -> dict:
    """{'gatd': (m, s), 'paper': (m, s|None), 'tabpc': (m, s)} where reported."""
    RC, RG = _ra_imports()
    out = {}
    paper_key = {"shape": "shape", "trend": "trend", "mle": "mle", "c2st": "c2st", "alpha": "alpha",
                 "beta": "beta"}.get(metric)
    if paper_key and ds in RP.DATASETS:
        c = RP.cell(paper_key, "TabDiff", ds)
        if isinstance(c, tuple):
            out["paper"] = c
    reg = task == "regression"
    gatd_key = {"shape": "shape", "trend": "trend",
                "mle": "mle2" if reg else "mle1", "mle_alt": "mle1" if reg else "mle2"}.get(metric)
    if gatd_key and RG.cell(gatd_key, ds):
        out["gatd"] = RG.cell(gatd_key, ds)
    tabpc_file = {"wnmis": "nmi_l1_weighted_complement.csv", "c2st_xgb": "xgb_detection.csv"}.get(metric)
    if tabpc_file:
        p = os.path.join(REPO, RC.TABPC_REL, "article_material", "our_results", tabpc_file)
        if os.path.exists(p):
            t = pd.read_csv(p)
            r = t[(t.method == "TabDiff") & (t.dataset == ds)]
            if len(r):
                out["tabpc"] = (float(r["mean"].iloc[0]), float(r["std"].iloc[0]))
    return out


def ra_tables(run: str) -> dict:
    """metric -> per-table values of one seed's run."""
    vals = {}
    ev = os.path.join(run, "tabdiff_eval")
    p = os.path.join(ev, "all_results.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, index_col=0)
        for k in ("shape", "trend", "mle", "c2st"):
            if k in df:
                vals[k] = df[k].to_numpy(float) / (100.0 if k == "c2st" else 1.0)
    p = os.path.join(ev, "mle2.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        vals["mle_alt"] = df["r2" if "r2" in df else "f1"].to_numpy(float)
    p = os.path.join(ev, "quality.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, index_col=0)
        for k in ("alpha", "beta"):
            if k in df:
                vals[k] = df[k].to_numpy(float)
    p = os.path.join(run, "tabpc_eval", "tabpc_results.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        for k in ("wnmis", "c2st_xgb"):
            vals[k] = df[k].to_numpy(float)
    return vals


def ra_computed(exp_root: str):
    """Pooled over seeds: mean and std (ddof=1) over all tables of all seeds found
    (GATD: 3 seeds x 20 tables), plus the std of the per-seed means."""
    RC, _ = _ra_imports()
    denoiser, label = RA_ARM
    rows, prov, params = [], {}, {}
    for ds in RA_DS:
        per_metric = {}
        for seed in RC.TRAIN_SEEDS:
            run = RC.run_dir(exp_root, RC.target_name(ds), denoiser, seed)
            if not os.path.isdir(run):
                continue
            meta_p = os.path.join(run, "run_meta.json")
            meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
            if meta.get("smoke_epochs"):
                continue                                       # never report smoke runs
            v = ra_tables(run)
            if v:
                prov[f"{ds}/seed{seed}"] = {"run": os.path.relpath(run, REPO),
                                            **{k: meta.get(k) for k in ("git_commit", "tabdiff", "param_counts",
                                                                        "denoiser", "variant", "gnn",
                                                                        "train_record")}}
                if meta.get("param_counts"):
                    params[ds] = meta["param_counts"]["denoiser_total"]
            for k, arr in v.items():
                per_metric.setdefault(k, []).append(arr)
        for k, arrs in per_metric.items():
            allv = np.concatenate(arrs)
            seed_means = [float(np.mean(a)) for a in arrs]
            rows.append(dict(dataset=ds, metric=k, method=label, kind="computed", mean=float(np.mean(allv)),
                             std=float(np.std(allv, ddof=1)) if len(allv) > 1 else np.nan,
                             std_seed_means=float(np.std(seed_means, ddof=1)) if len(arrs) > 1 else np.nan,
                             n=len(allv), n_seeds=len(arrs), source=label))
    return pd.DataFrame(rows), prov, params


def ra_static_params() -> dict:
    """dataset -> (AttnGNN params, TabDiff params) counted from the data shapes by
    tabdiff_gnn/params.py (hpc_transfer/jobs_tabdiff_gnn/params.csv)."""
    p = os.path.join(REPO, "hpc_transfer", "jobs_tabdiff_gnn", "params.csv")
    if not os.path.exists(p):
        return {}
    t = pd.read_csv(p).set_index("dataset")
    return {ds: (int(r["attngnn"]), int(r["transformer"])) for ds, r in t.iterrows()}


def _cell(m, s, dec, mark=""):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "—"
    out = f"{m:.{dec}f}"
    if s is not None and not (isinstance(s, float) and np.isnan(s)):
        out += f" ± {s:.{dec}f}"
    return out + mark


def ra_long_reported() -> pd.DataFrame:
    RC, _ = _ra_imports()
    rows = []
    for key, _, _ in RA_METRICS:
        for ds in RA_DS:
            for name, ref in ra_reference(key, ds, RC.TASK[ds]).items():
                rows.append(dict(dataset=ds, metric=key, method=dict(RA_REFS)[name], kind="reported",
                                 mean=ref[0], std=ref[1], n=np.nan, n_seeds=np.nan, source=name))
    return pd.DataFrame(rows)


def ra_markdown(comp: pd.DataFrame, params: dict, static: dict, exp_root: str) -> str:
    RC, RG = _ra_imports()
    label = RA_ARM[1]
    out = ["# Route A: AttnGNN as the denoiser inside TabDiff", "",
           f"_Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by `analysis/tabdiff/compare.py --route_a` "
           f"from `{os.path.relpath(RC.results_root(exp_root), REPO)}`._", "",
           "## Reference", "",
           "- **TabDiff numbers are reported, not reproduced.** Primary reference: GATD's re-run of TabDiff "
           "with the official code and protocol, 3 training seeds × 20 tables (its std includes retraining "
           "variance; `analysis/tabdiff/reported_gatd.py`). Secondary: the TabDiff paper (one trained model × "
           "20 tables). wNMIS and C2ST (XGB): tabpc's TabDiff (5 trained models × 1 table).",
           "- **The framework is TabDiff's.** Only the denoiser differs: with `denoiser=transformer` our "
           "wrapper gives bit-identical parameters and outputs to TabDiff's `UniModMLP`, and our entry point "
           "builds bit-for-bit the model `tabdiff/main.py` builds (`tests/test_tabdiff_gnn.py`).",
           f"- **Ours (computed):** TabDiff + AttnGNN V2, d_model {RC.D_MODEL} on every dataset; mean ± std "
           "(ddof=1) pooled over all tables of all training seeds found (target 3 × 20); `n` = tables, `s` = "
           "seeds. No DCR in this study.", ""]
    out += ["## Results", "", "Column prefix **reported:** = published TabDiff numbers; **computed:** = ours.", ""]
    for key, title, dec in RA_METRICS:
        refs_here = [r for r in RA_REFS if any(r[0] in ra_reference(key, ds, RC.TASK[ds]) for ds in RA_DS)]
        out += [f"### {title}", "",
                "| dataset | " + " | ".join(r[1] for r in refs_here) + f" | {label} |",
                "|---|" + "---|" * (len(refs_here) + 1)]
        for ds in RA_DS:
            refs = ra_reference(key, ds, RC.TASK[ds])
            cells = [_cell(*refs[r[0]], dec) if r[0] in refs else "—" for r in refs_here]
            r = comp[(comp.dataset == ds) & (comp.metric == key)] if len(comp) else comp
            if len(r):
                r = r.iloc[0]
                cells.append(_cell(r["mean"], r["std"], dec, f" (n={int(r['n'])}, s={int(r['n_seeds'])})"))
            else:
                cells.append("—")
            out.append(f"| {ds} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## Parameter counts (denoiser)", "",
            f"| dataset | AttnGNN (d_model {RC.D_MODEL}) | TabDiff UniModMLP | ratio | reported: TabDiff (GATD Table 16) |",
            "|---|---|---|---|---|"]
    for ds in RA_DS:
        a, t = static.get(ds, (None, None))
        a = params.get(ds, a)
        out.append(f"| {ds} | {f'{a:,}' if a else '—'} | {f'{t:,}' if t else '—'} | "
                   f"{f'{a / t:.3f}' if a and t else '—'} | {RG.TABDIFF_COST.get(ds, ('—',))[0]} |")
    out += ["", "Both counted on the dataset's real shapes with the same code (`tabdiff_gnn/params.py`; "
            "the learnable schedules add d_num + n_cat in both). GNN configuration from the source mapping "
            "(gnn type, layers, top_k, heads); diabetes uses adult's -- our choice, diabetes has no TabDDPM "
            "source.", "", "## Deviations from TabDiff's protocol", ""]
    out += [f"- {d}" for d in RC.DEVIATIONS + RC.DEVIATIONS_ATTNGNN] + [""]
    return "\n".join(out)


def route_a_main(exp_root: str, out_dir: str) -> int:
    comp, prov, params = ra_computed(exp_root)
    md = ra_markdown(comp, params, ra_static_params(), exp_root)
    pd.concat([ra_long_reported(), comp], ignore_index=True).to_csv(
        os.path.join(out_dir, "route_a_long.csv"), index=False)
    with open(os.path.join(out_dir, "route_a_provenance.json"), "w") as fh:
        json.dump({"exp_root": exp_root, "runs": prov}, fh, indent=2, default=str)
    with open(os.path.join(out_dir, "ROUTE_A.md"), "w") as fh:
        fh.write(md)
    print(f"route A: computed cells {len(comp)} from {len(prov)} seed runs; "
          f"wrote {out_dir}/ROUTE_A.md, route_a_long.csv, route_a_provenance.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", default=os.path.join(REPO, "transfer_runs"))
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--route_a", action="store_true",
                    help="Route A report (tabdiff_gnn/, <exp_root>/exp_tabdiff_gnn/) -> ROUTE_A.md")
    args = ap.parse_args()
    if args.route_a:
        return route_a_main(args.exp_root, args.out)
    ours, prov = computed(args.exp_root)
    long = pd.concat([reported(), ours], ignore_index=True)
    long.to_csv(os.path.join(args.out, "results_long.csv"), index=False)
    with open(os.path.join(args.out, "provenance.json"), "w") as fh:
        json.dump({"tabdiff_commit": S.TABDIFF_COMMIT, "exp_root": args.exp_root,
                   "sources": S.SOURCES, "deviations": S.DEVIATIONS, "runs": prov}, fh, indent=2, default=str)
    with open(os.path.join(args.out, "RESULTS.md"), "w") as fh:
        fh.write(markdown(long, prov, args.exp_root))
    n_runs = len([k for k in prov if not k.endswith("/dcr")])
    print(f"computed cells: {len(ours)} from {n_runs} evaluated runs; reported cells: {(long.kind == 'reported').sum()}")
    print(f"wrote {args.out}/RESULTS.md, results_long.csv, provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

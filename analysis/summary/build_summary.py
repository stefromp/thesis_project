#!/usr/bin/env python3
"""Supervisor summary: every comparison table built from results already on disk.

    python3 analysis/summary/build_summary.py

Writes analysis/summary/:
  summary.md            every table (Markdown), how to read each, protocols, caveats
  tables/<id>.tex       a LaTeX (booktabs) version of each table
  csv/<id>.csv          the values behind each table, one row per value, with its
                        kind (computed / reported), source file(s) and commit
  csv/all_values.csv    all of the above in one file
  provenance.csv        every file read: path, last git commit touching it (or
                        "untracked" + sha256/16 for gitignored run outputs)
  figures/topk_k_vs_t.{pdf,png}   B7 figure (topk_frac_vs_t: the same, y = k / (N − 1))

    python3 analysis/summary/build_summary.py --figures-only   rewrites only the two B7 figures

Nothing is trained or sampled and nothing is imputed: each number is read from a
results file (or derived from such numbers: ranks, ratios, win/tie/loss counts,
which are labelled "derived"). Reuses, never modifies:
  analysis/topk_comparison/compare_topk.py   run-cell selection (discover), loaders
  analysis/topk_comparison/paper_comparison.py   TabDDPM Table 5 + 50-run statistic
  analysis/tabdiff/compare.py                Route A per-table readers (ra_tables)
  analysis/tabdiff/reported.py, reported_gatd.py   transcribed TabDiff / GATD numbers
  tabdiff_gnn/config.py                      Route A run-directory layout

SUMMARY_PYTEST=0 skips running tests/test_selection_differentiability.py (part D);
by default it is run with the current interpreter and PYTHONPATH and its outcome is
quoted in summary.md.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if __name__ == "__main__":
    sys.modules.setdefault("build_summary", sys.modules["__main__"])
OUT, TAB, CSV, FIG = HERE, HERE / "tables", HERE / "csv", HERE / "figures"

# compare_topk reads these at import time
A_EXCLUDED = ["higgs-small", "miniboone", "fb-comments"]
os.environ["COMPARE_TOPK_EXCLUDE"] = ",".join(A_EXCLUDED)
os.environ.pop("COMPARE_TOPK_VARIANT", None)
os.environ.pop("COMPARE_TOPK_RESULTS_ROOT", None)
for p in ("analysis/topk_comparison", "analysis/tabdiff", "tabdiff_gnn", "analysis/summary"):
    sys.path.insert(0, str(REPO / p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import thesis_style as TS  # noqa: E402  (analysis/summary/thesis_style.py: house style of the figures)
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
plt = TS.plt

import compare_topk as ct  # noqa: E402
import paper_comparison as pc  # noqa: E402
import reported as RP  # noqa: E402
import reported_gatd as RG  # noqa: E402
import compare as CP  # noqa: E402  (analysis/tabdiff/compare.py)
import config as RC  # noqa: E402  (tabdiff_gnn/config.py)

EXP_ROOT = REPO / "transfer_runs"

# ===========================================================================
# Provenance
# ===========================================================================
_PROV: dict[str, dict] = {}


def jload(p):
    with open(p) as fh:
        return json.load(fh)


def rel(p) -> str:
    return os.path.relpath(Path(p).resolve(), REPO) if str(Path(p).resolve()).startswith(str(REPO)) else str(p)


def _git(*args) -> str:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def _file_info(p) -> dict:
    r = rel(p)
    if r not in _PROV:
        tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", r],
                                 capture_output=True).returncode == 0
        commit = _git("log", "-1", "--format=%h", "--", r) if tracked else ""
        if tracked and _git("status", "--porcelain", "--", r):
            commit += " (modified in working tree)"
        sha = hashlib.sha256(Path(REPO / r).read_bytes()).hexdigest()[:16]
        _PROV[r] = {"path": r, "git_commit": commit if tracked else "untracked", "sha256_16": sha}
    return _PROV[r]


def src(paths, producer: str = "") -> dict:
    """source / source_commit fields for a value read from `paths`."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    infos = [_file_info(p) for p in paths]
    commits = []
    for i in infos:
        c = i["git_commit"] if i["git_commit"] != "untracked" else f"untracked sha256/16={i['sha256_16']}"
        if c not in commits:
            commits.append(c)
    return {"source": "; ".join(i["path"] for i in infos), "source_commit": "; ".join(commits),
            "producer_commit": producer}


def code_src(path: str, what: str) -> dict:
    """A value transcribed into code (reported numbers): the transcription file + commit."""
    s = src(REPO / path)
    s["source"] = f"{what} (transcribed in {s['source']})"
    return s


# ===========================================================================
# Value records and formatting
# ===========================================================================
ROWS: list[dict] = []
NA = {"not reported": ("ᵖ", "p"), "not computed": ("ⁿ", "n"), "not applicable": ("ᵃ", "a"),
      "no run on disk": ("ᵈ", "d")}
LEGEND_MD = ("— = no value: ᵖ not reported, ⁿ not computed, ᵃ not applicable, ᵈ no run on disk; "
             "**pending** = run still in progress. † = reported (paper, or the authors' released "
             "evaluation files); unmarked values are computed by us.")
LEGEND_TEX = (r"---: no value ($^{p}$ not reported, $^{n}$ not computed, $^{a}$ not applicable, "
              r"$^{d}$ no run on disk); \emph{pending}: run in progress. $^\dagger$reported (paper or the "
              r"authors' released evaluation files); unmarked: computed by us.")


def V(table, dataset, model, metric, mean=np.nan, std=np.nan, *, n=None, std_kind="", kind="computed",
      status="ok", note="", **source) -> dict:
    rec = {"table": table, "dataset": dataset, "model": model, "metric": metric,
           "mean": float(mean) if mean is not None else np.nan,
           "std": float(std) if std is not None else np.nan, "n": n, "std_kind": std_kind,
           "kind": kind, "status": status, "source": source.get("source", ""),
           "source_commit": source.get("source_commit", ""),
           "producer_commit": source.get("producer_commit", ""), "note": note}
    ROWS.append(rec)
    return rec


def missing(table, dataset, model, metric, status, note="", kind="computed", **source) -> dict:
    assert status in NA or status == "pending", status
    return V(table, dataset, model, metric, status=status, note=note, kind=kind, **source)


def fmt(rec: dict | None, nd: int | str, tex: bool = False) -> str:
    if rec is None:
        rec = {"status": "not computed"}
    st = rec["status"]
    if st == "pending":
        return r"\emph{pending}" if tex else "**pending**"
    if st in NA:
        return ("---$^{%s}$" % NA[st][1]) if tex else ("—" + NA[st][0])
    m, s = rec["mean"], rec["std"]
    dag = (r"$^\dagger$" if tex else "†") if rec["kind"] == "reported" else ""
    if nd == "int":
        t = f"{int(round(m)):,}"
        return (t.replace(",", "{,}") if tex else t) + dag
    if nd == "auto":                                # by magnitude (RMSE in target units)
        nd = 1 if abs(m) >= 1000 else 2 if abs(m) >= 10 else 4
    t = f"{m:.{nd}f}"
    if tex:
        t = t.replace("-", "$-$")
        if np.isfinite(s):
            t += r"{\scriptsize$\pm$" + f"{s:.{nd}f}" + "}"
    elif np.isfinite(s):
        t += f" ± {s:.{nd}f}"
    return t + dag


def md_table(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("#", r"\#").replace("_", r"\_").replace("²", "$^2$").replace("−", "$-$")
            .replace("→", r"$\to$").replace("±", r"$\pm$").replace("—", "--").replace("◊", r"$^\diamond$"))


def tex_table(label: str, caption: str, head: list[str], rows: list[list[str]], note: str,
              groups: list[int] | None = None, colspec: str | None = None) -> str:
    """booktabs table; `groups` = row indices before which a \\midrule goes."""
    colspec = colspec or ("l" + "c" * (len(head) - 1))
    L = [r"% requires \usepackage{booktabs,graphicx}", r"\begin{table}[t]", r"\centering",
         r"\caption{%s}" % caption, r"\label{%s}" % label, r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{%s}" % colspec, r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if groups and i in groups and i:
            L.append(r"\midrule")
        L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\par\smallskip\footnotesize " + note + " " + LEGEND_TEX,
          r"\end{table}"]
    return "\n".join(L) + "\n"


SECTIONS: list[tuple[str, str]] = []      # (table id, markdown block) in summary order
TEX_FILES: list[str] = []


def emit(tid: str, title: str, how: str, protocol: str, head_md, rows_md, note_md,
         head_tex, rows_tex, note_tex, caption, groups=None, colspec=None, select=None):
    """select: which value records go to csv/<tid>.csv (default: those created for table tid)."""
    md = [f"### {tid}. {title}", "", f"**How to read.** {how}", "", f"**Protocol.** {protocol}", "",
          md_table(head_md, rows_md), "", note_md + " " + LEGEND_MD, ""]
    SECTIONS.append((tid, "\n".join(md)))
    tex = tex_table(f"tab:summary_{tid.lower()}", caption, head_tex, rows_tex, note_tex, groups, colspec)
    name = f"{tid}.tex"
    (TAB / name).write_text(tex)
    TEX_FILES.append(name)
    recs = [r for r in ROWS if (select(r) if select else r["table"] == tid)]
    assert recs, f"{tid}: no value records"
    pd.DataFrame(recs).to_csv(CSV / f"{tid}.csv", index=False)


def text_section(title: str, body: str):
    SECTIONS.append(("", f"### {title}\n\n{body}\n"))


# ===========================================================================
# A. TabDDPM framework
# ===========================================================================
A_DS = ["abalone", "adult", "buddy", "california", "cardio", "churn2", "default", "diabetes",
        "gesture", "house", "insurance", "king", "wilt"]
AB = {"abalone": "AB", "adult": "AD", "buddy": "BU", "california": "CA", "cardio": "CAR", "churn2": "CH",
      "default": "DE", "diabetes": "DI", "gesture": "GE", "house": "HO", "insurance": "IN", "king": "KI",
      "wilt": "WI"}
NLFL = {"default"}                            # local default != the authors' default
RULES = ["fixed", "jenks", "entmax15"]
RULE_LABEL = {"fixed": "fixed top-k", "jenks": "adaptive_jenks", "entmax15": "adaptive_entmax15"}
TDDPM_PAPER = "Kotelnikov et al., TabDDPM, arXiv:2209.15421v2, Table 5"
BUDGET_DS = ["california", "house", "king", "gesture", "cardio"]

# metric key -> (raw name: classification, raw name: regression, md header, tex header, decimals)
A_UTIL = {
    "mle": ("macro avg/f1-score", "r2", "MLE: F1 / R² ↑", r"MLE: $F_1$ / $R^2$ $\uparrow$", 4),
    "accuracy": ("accuracy", None, "Accuracy ↑", r"Acc.\ $\uparrow$", 4),
    "rmse": (None, "rmse", "RMSE ↓", r"RMSE $\downarrow$", "auto"),
    "roc_auc": ("roc_auc", None, "ROC AUC ↑", r"ROC AUC $\uparrow$", 4),
    "macro_precision": ("macro avg/precision", None, "macro-P ↑", r"macro-P $\uparrow$", 4),
    "macro_recall": ("macro avg/recall", None, "macro-R ↑", r"macro-R $\uparrow$", 4),
    "weighted_f1": ("weighted avg/f1-score", None, "weighted-F1 ↑", r"w-$F_1$ $\uparrow$", 4),
    "weighted_precision": ("weighted avg/precision", None, "weighted-P ↑", r"w-P $\uparrow$", 4),
    "weighted_recall": ("weighted avg/recall", None, "weighted-R ↑", r"w-R $\uparrow$", 4),
}
A_GEN = {   # per-generation-seed metrics
    "dcr": ("dcr_baseline", "DCR ↑", r"DCR $\uparrow$", 4),
    "dcr_mean_nolabel": ("dcr_tabddpm_mean", "DCR mean-L2, no label", r"DCR mean-L2, no label", 4),
    "mia_auc": ("mia_auc", "MIA AUC (→ 0.5)", r"MIA AUC ($\to0.5$)", 3),
    "wasserstein": ("wasserstein_mean", "Wasserstein ↓", r"Wasserstein $\downarrow$", 4),
}
A_FID = {   # dependency_fidelity/per_dataset.csv
    "shape_err": ("column_shapes", "Shape err (%) ↓", r"Shape err.\ (\%) $\downarrow$", 2),
    "trend_err": ("column_pair_trends", "Trend err (%) ↓", r"Trend err.\ (\%) $\downarrow$", 2),
    "assoc_mean_abs": ("assoc_mean_abs", "assoc. dist. (mean abs) ↓", r"assoc.\ (mean abs) $\downarrow$", 4),
    "assoc_frobenius": ("assoc_frobenius", "assoc. dist. (Frobenius) ↓", r"assoc.\ (Frob.) $\downarrow$", 3),
}
A_MAIN_COLS = ["mle", "shape_err", "trend_err", "dcr", "dcr_ratio", "mia_auc", "wasserstein", "params"]
A_EXT_COLS = ["accuracy", "rmse", "roc_auc", "macro_precision", "macro_recall", "weighted_f1",
              "weighted_precision", "weighted_recall", "dcr_mean_nolabel", "assoc_mean_abs", "assoc_frobenius"]
HEAD = {**{k: (v[2], v[3]) for k, v in A_UTIL.items()}, **{k: (v[1], v[2]) for k, v in A_GEN.items()},
        **{k: (v[1], v[2]) for k, v in A_FID.items()},
        "dcr_ratio": ("DCR / holdout", r"DCR/holdout"), "params": ("Params", "Params")}
ND = {**{k: v[4] for k, v in A_UTIL.items()}, **{k: v[3] for k, v in A_GEN.items()},
      **{k: v[3] for k, v in A_FID.items()}, "dcr_ratio": 3, "params": "int"}


def task_of(ds: str) -> str:
    return jload(REPO / "data" / ds / "info.json")["task_type"]


def is_reg(ds):
    return task_of(ds) == "regression"


def a_ds_label(ds: str, tex: bool) -> str:
    dev = ds in DEV
    if tex:
        return tex_escape(ds) + (r"$^{*}$" if dev else "") + (r"$^\ddagger$" if ds in NLFL else "")
    return ds + (" \\*" if dev else "") + (" ‡" if ds in NLFL else "")


def a_unit(ds, tex):
    return ("$R^2$" if tex else "R²") if is_reg(ds) else ("$F_1$" if tex else "F1")


def _cfg_tuple_from_id(cid: str):          # "gatv2_L3_d128_k5_h8"
    g, L, d, k, h = cid.split("_")[:5]
    return g, int(L[1:]), int(d[1:]), int(k[1:]), int(h[1:])


def _cfg_tuple_from_arch(s: str):          # "gatv2 L3 d128 h8 k5"
    g, L, d, h, k = s.split()
    return g, int(L[1:]), int(d[1:]), int(k[1:]), int(h[1:])


def load_a_cells():
    _, cells = ct.discover()
    assert sorted(cells) == sorted(A_DS), sorted(cells)
    for ds in A_DS:
        for r in RULES:
            c = cells[ds]["rules"][r]
            assert c["status"] == "complete" and c["per_gen"] is not None, (ds, r, c["status"])
            assert not cells[ds]["cat_affected"], ds
    return cells


def gnn_a_values(cells) -> dict:
    """(ds, rule, metric) -> record; tables A1-A3 (+ the extended block)."""
    per = pd.read_csv(REPO / "analysis/dependency_fidelity/per_dataset.csv")
    cov = pd.read_csv(REPO / "analysis/dependency_fidelity/coverage.csv")
    fid_src = src([REPO / "analysis/dependency_fidelity/per_dataset.csv",
                   REPO / "analysis/dependency_fidelity/coverage.csv"])
    hold = pd.read_csv(REPO / "analysis/paper_tables/holdout_dcr.csv").set_index("dataset").dcr_holdout
    hold_src = src(REPO / "analysis/paper_tables/holdout_dcr.csv")
    pcnt = pd.read_csv(REPO / "analysis/config_comparison/param_counts.csv").set_index("dataset")
    p_src = src(REPO / "analysis/config_comparison/param_counts.csv")
    posthoc = REPO / "analysis/integrity/posthoc_gen_metrics.csv"
    out, xcheck = {}, []
    for ds in A_DS:
        reg = is_reg(ds)
        for r in RULES:
            tid = {"fixed": "A1", "jenks": "A2", "entmax15": "A3"}[r]
            model = f"GNN ({RULE_LABEL[r]})"
            c = cells[ds]["rules"][r]
            run = REPO / c["source"]
            files = sorted(run.glob("per_seed_scores.csv")) + sorted(run.glob("per_seed_scores.gen*.csv"))
            meta = jload(run / "run_meta.json") if (run / "run_meta.json").exists() else {}
            producer = f"run_meta git_commit={meta.get('git_commit', 'not recorded')}"
            rs = src(files + [run / "run_meta.json"], producer)
            raw = pd.concat([pd.read_csv(f) for f in files])
            own_metrics = set(raw.metric)
            # utility: all 5 gen x 10 CatBoost runs, std ddof=0 (TabDDPM's lib/metrics convention)
            for key, (cname, rname, *_ ) in A_UTIL.items():
                name = rname if reg else cname
                if name is None:
                    out[(ds, r, key)] = missing(tid, ds, model, key, "not applicable",
                                                "regression dataset" if not reg else "classification metric")
                    continue
                v = pc.raw_runs(run, name)
                if len(v) == 0:
                    note = "multiclass: the evaluator records no ROC AUC" if key == "roc_auc" else ""
                    out[(ds, r, key)] = missing(tid, ds, model, key, "not computed", note, **rs)
                    continue
                assert len(v) == ct.EXPECTED_GEN * ct.EXPECTED_CLF, (ds, r, key, len(v))
                out[(ds, r, key)] = V(tid, ds, model, key, v.mean(), v.std(), n=len(v),
                                      std_kind="std over 5 gen x 10 CatBoost runs (ddof=0)",
                                      note=f"raw metric '{name}'", **rs)
            # per generation seed: mean, sd over 5 gen seeds (ddof=1)
            for key, (name, *_ ) in A_GEN.items():
                rv = ct.rule_values(c, name)
                s = rs if name in own_metrics else src(files + [run / "run_meta.json", posthoc], producer)
                if rv is None:
                    out[(ds, r, key)] = missing(tid, ds, model, key, "not computed", **rs)
                    continue
                out[(ds, r, key)] = V(tid, ds, model, key, rv["mean"], rv["sd"], n=rv["n"],
                                      std_kind="sd over 5 generation seeds (ddof=1)",
                                      note=("" if name in own_metrics else
                                            "post-hoc backfill (split_runner shards lacked it)"), **s)
            # DCR / holdout (derived from two computed files)
            d = out[(ds, r, "dcr")]
            h = hold[ds]
            out[(ds, r, "dcr_ratio")] = V(tid, ds, model, "dcr_ratio", d["mean"] / h, d["std"] / h,
                                          n=d["n"], std_kind="sd of DCR / holdout DCR",
                                          note=f"derived: DCR / holdout DCR (holdout = {h:.6f})",
                                          source=d["source"] + "; " + hold_src["source"],
                                          source_commit=d["source_commit"] + "; " + hold_src["source_commit"])
            # fidelity (Shape / Trend / association), from the same run cell
            cv = cov[(cov.dataset == ds) & (cov.rule == r)]
            tree, combo = c["source"].rsplit("/", 1)
            if cv.empty or cv.tree.iloc[0] != tree or cv.combo.iloc[0] != combo:
                raise RuntimeError(f"dependency_fidelity is stale for {ds}/{r}")
            for key, (name, *_ ) in A_FID.items():
                x = per[(per.dataset == ds) & (per.rule == r) & (per.metric == name)]
                assert len(x) == 1 and x.status.iloc[0] == "complete" and x.n_gen_seeds.iloc[0] == 5, (ds, r, name)
                m, s_ = float(x["mean"].iloc[0]), float(x["sd"].iloc[0])
                if key in ("shape_err", "trend_err"):
                    m, s_ = 100 * (1 - m), 100 * s_
                out[(ds, r, key)] = V(tid, ds, model, key, m, s_, n=5,
                                      std_kind="sd over 5 generation seeds (ddof=1)",
                                      note="error = 100 x (1 - score)" if key.endswith("_err") else "",
                                      **fid_src)
            # parameters: instantiated on CPU with the dataset's shapes (config_comparison.py)
            pr = pcnt.loc[ds]
            assert _cfg_tuple_from_arch(pr.my_architecture) == _cfg_tuple_from_id(c["config"]), (ds, r)
            out[(ds, r, "params")] = V(tid, ds, model, "params", pr[f"params_{r}"], n=1,
                                       note=f"architecture {pr.my_architecture}", **p_src)
            # cross-check against the run's own results_full_averaged.json
            rfa = run / "results_full_averaged.json"
            if rfa.exists():
                j = jload(rfa)
                for key, (cname, rname, *_ ) in A_UTIL.items():
                    name = rname if reg else cname
                    if name and name in j and (ds, r, key) in out and out[(ds, r, key)]["status"] == "ok":
                        xcheck.append(abs(j[name]["mean"] - out[(ds, r, key)]["mean"]))
                for key, (name, *_ ) in A_GEN.items():
                    if name in j and out[(ds, r, key)]["status"] == "ok":
                        xcheck.append(abs(j[name]["mean"] - out[(ds, r, key)]["mean"]))
    out["_xcheck"] = xcheck
    return out


def table_a_rule(vals, r):
    tid = {"fixed": "A1", "jenks": "A2", "entmax15": "A3"}[r]
    rows_md, rows_tex = [], []
    for ds in A_DS:
        rows_md.append([a_ds_label(ds, False), a_unit(ds, False)] + [fmt(vals[(ds, r, k)], ND[k]) for k in A_MAIN_COLS])
        rows_tex.append([a_ds_label(ds, True), a_unit(ds, True)] + [fmt(vals[(ds, r, k)], ND[k], True)
                                                                  for k in A_MAIN_COLS])
    ext_md = [[a_ds_label(ds, False)] + [fmt(vals[(ds, r, k)], ND[k]) for k in A_EXT_COLS] for ds in A_DS]
    ext_tex = [[a_ds_label(ds, True)] + [fmt(vals[(ds, r, k)], ND[k], True) for k in A_EXT_COLS] for ds in A_DS]
    head_md = ["Dataset", "MLE unit"] + [HEAD[k][0] for k in A_MAIN_COLS]
    head_tex = ["Dataset", "unit"] + [HEAD[k][1] for k in A_MAIN_COLS]
    how = ("Rows = datasets, one GNN-denoiser run cell each (the GNN configuration used for that dataset, "
           f"neighbour-selection rule **{RULE_LABEL[r]}**). MLE: macro-F1 (\"macro avg/f1-score\", not "
           "\"score\") for classification, R² for regression, higher is better. Shape / Trend: error = "
           "100 × (1 − score), lower is better. DCR / holdout ≈ 1 means synthetic rows are as far from "
           "the training rows as unseen real rows. MIA AUC 0.5 = no membership leakage. The second "
           "block lists every other metric the runs recorded.")
    note = ("\\* development dataset (GNN configuration selected on the test split, in-sample); ‡ local "
            "default data differ from the TabDDPM authors'. MLE and the other CatBoost metrics: mean ± std "
            "over 5 generation × 10 CatBoost seeds (50 runs, ddof=0). Shape, Trend, DCR, MIA, Wasserstein, "
            "association distances: mean ± sd over 5 generation seeds (ddof=1). DCR = median L2 distance "
            "of each synthetic row to its closest real training row (target included); holdout DCR = the "
            "same statistic for the real test split (table A0). Params = trainable parameters of the "
            "denoiser, instantiated on the dataset's shapes.")
    note_tex = (r"$^{*}$Development dataset (configuration selected on the test split); $^\ddagger$local "
                r"\textit{default} data differ from the authors'. CatBoost metrics: mean$\pm$std over 50 runs "
                r"(5 generation $\times$ 10 CatBoost seeds, ddof=0); all others mean$\pm$sd over 5 "
                r"generation seeds (ddof=1). Shape/Trend: error $=100(1-\mathrm{score})$.")
    emit(tid, f"GNN denoiser, {RULE_LABEL[r]} (TabDDPM framework)", how, PROTO_A,
         head_md, rows_md, note, head_tex, rows_tex, note_tex,
         f"GNN denoiser in the TabDDPM framework, {tex_escape(RULE_LABEL[r])} rule: main metrics.",
         colspec="ll" + "c" * len(A_MAIN_COLS),
         select=lambda x: x["table"] == tid and x["metric"] in A_MAIN_COLS)
    # extended block: same cells, every other recorded metric
    tid2 = tid + "x"
    head2_md = ["Dataset"] + [HEAD[k][0] for k in A_EXT_COLS]
    head2_tex = ["Dataset"] + [HEAD[k][1] for k in A_EXT_COLS]
    note2 = ("Every other metric recorded for the same run cells: the remaining CatBoost metrics of "
             "results_full_averaged.json (\"score\" = accuracy / −RMSE is a duplicate and not shown; "
             "\"support\" is a row count), the DCR variant without the target and with the mean instead "
             "of the median (not comparable with the TabDDPM authors' DCR), and the association-matrix "
             "distances of the dependency-fidelity analysis (real vs synthetic pairwise association, "
             "lower is better).")
    emit(tid2, f"GNN denoiser, {RULE_LABEL[r]}: all other recorded metrics",
         "Continuation of " + tid + ": same run cells, same seeds; rows = datasets.", PROTO_A,
         head2_md, ext_md, note2, head2_tex, ext_tex,
         r"Same cells as Table~\ref{tab:summary_%s}: CatBoost metrics (50 runs) and per-generation-seed "
         r"metrics (5 seeds)." % tid.lower(),
         f"GNN denoiser, {tex_escape(RULE_LABEL[r])} rule: all other recorded metrics.",
         select=lambda x: x["table"] == tid and x["metric"] in A_EXT_COLS)


def tabddpm_values() -> dict:
    """A4 (TabDDPM) and A0 (Real) records."""
    hold = pd.read_csv(REPO / "analysis/paper_tables/holdout_dcr.csv").set_index("dataset").dcr_holdout
    hold_src = src(REPO / "analysis/paper_tables/holdout_dcr.csv")
    pcnt = pd.read_csv(REPO / "analysis/config_comparison/param_counts.csv").set_index("dataset")
    p_src = src(REPO / "analysis/config_comparison/param_counts.csv")
    floor = pd.read_csv(REPO / "analysis/dependency_fidelity/floor.csv").set_index("dataset")
    f_src = src(REPO / "analysis/dependency_fidelity/floor.csv")
    paper_src = code_src("analysis/topk_comparison/paper_comparison.py", TDDPM_PAPER)
    out = {}
    model, real = "TabDDPM (MLP denoiser)", "Real"
    for ds in A_DS:
        reg = is_reg(ds)
        ab, unit, pm, ps, rm, rsd = pc.PAPER[ds]
        assert ab == AB[ds] and unit == ("R2" if reg else "F1")
        out[(ds, "tddpm", "mle_paper")] = V("A4", ds, model, "mle_paper", pm, ps, n=50, kind="reported",
                                            std_kind="std over 50 runs (paper)", **paper_src)
        out[(ds, "real", "mle_paper")] = V("A0", ds, real, "mle_paper", rm, rsd, n=50, kind="reported",
                                           std_kind="std over runs (paper)", **paper_src)
        ev = REPO / "exp" / ds / "ddpm_cb_best" / "eval_catboost.json"
        e = jload(ev)
        es = src(ev)
        for who, block, tid in (("tddpm", "synthetic", "A4"), ("real", "real", "A0")):
            t = e[block]["test"]
            for key, raw_key in (("mle", "r2" if reg else "f1"), ("accuracy", None if reg else "acc"),
                                 ("rmse", "rmse" if reg else None), ("roc_auc", None if reg else "roc_auc")):
                mname = model if who == "tddpm" else real
                if raw_key is None:
                    out[(ds, who, key)] = missing(tid, ds, mname, key, "not applicable", kind="reported")
                elif f"{raw_key}-mean" not in t:
                    out[(ds, who, key)] = missing(tid, ds, mname, key, "not reported", kind="reported",
                                                  note="not in the authors' eval_catboost.json", **es)
                else:
                    out[(ds, who, key)] = V(tid, ds, mname, key, t[f"{raw_key}-mean"], t[f"{raw_key}-std"],
                                            n=int(t[f"{raw_key}-count"]), kind="reported",
                                            std_kind="authors' std over their runs",
                                            note="authors' released evaluation file, test split", **es)
        for key in ("macro_precision", "macro_recall", "weighted_f1", "weighted_precision", "weighted_recall"):
            out[(ds, "tddpm", key)] = (missing("A4", ds, model, key, "not applicable") if reg else
                                       missing("A4", ds, model, key, "not reported", kind="reported",
                                               note="not in the paper or the authors' evaluation files"))
        no_samples = "TabDDPM's synthetic samples are not on disk (only its evaluation files are)"
        for key in ("shape_err", "trend_err", "mia_auc", "wasserstein", "dcr_mean_nolabel",
                    "assoc_mean_abs", "assoc_frobenius"):
            out[(ds, "tddpm", key)] = missing("A4", ds, model, key, "not computed", no_samples)
        # DCR: the authors' privacy.json (same statistic as our dcr_baseline, verified)
        pj = REPO / "exp" / ds / "ddpm_cb_best" / "privacy.json"
        n_train = len(np.load(REPO / "data" / ds / "y_train.npy", allow_pickle=True))
        sub = n_train > 50_000
        if pj.exists():
            d = float(jload(pj)["privacy"])
            out[(ds, "tddpm", "dcr")] = V("A4", ds, model, "dcr", d, n=1, kind="reported",
                                          std_kind="one generated sample",
                                          note=("authors' privacy.json; same recipe as dcr_baseline "
                                                "(analysis/paper_tables/DCR_RECIPE_CHECK.md)"
                                                + ("; real train subsampled to 50k" if sub else "")), **src(pj))
            if ds in NLFL or sub:
                out[(ds, "tddpm", "dcr_ratio")] = missing(
                    "A4", ds, model, "dcr_ratio", "not computed",
                    "authors' data differ from ours (default)" if ds in NLFL else "authors' DCR subsampled")
            else:
                out[(ds, "tddpm", "dcr_ratio")] = V(
                    "A4", ds, model, "dcr_ratio", d / hold[ds], n=1, note="derived: authors' DCR / our holdout DCR",
                    source=src(pj)["source"] + "; " + hold_src["source"],
                    source_commit=src(pj)["source_commit"] + "; " + hold_src["source_commit"])
        else:
            out[(ds, "tddpm", "dcr")] = missing("A4", ds, model, "dcr", "not reported", kind="reported")
            out[(ds, "tddpm", "dcr_ratio")] = missing("A4", ds, model, "dcr_ratio", "not computed")
        out[(ds, "tddpm", "params")] = V("A4", ds, model, "params", pcnt.loc[ds].params_tabddpm, n=1,
                                         note=f"MLP d_layers {pcnt.loc[ds].tabddpm_d_layers} from the authors' "
                                              "tuned config.toml, instantiated on CPU", **p_src)
        # Real reference (A0): real-test Shape/Trend floor, holdout DCR
        for key, col in (("shape_err", "column_shapes"), ("trend_err", "column_pair_trends")):
            out[(ds, "real", key)] = V("A0", ds, real, key, 100 * (1 - floor.loc[ds, col]), n=1,
                                       note="real test split vs real train split", **f_src)
        out[(ds, "real", "dcr_holdout")] = V("A0", ds, real, "dcr_holdout", hold[ds], n=1,
                                             note="real test split vs real train split, dcr_baseline recipe",
                                             **hold_src)
    return out


def table_a0(tv):
    cols = [("mle_paper", "MLE (paper)", "MLE (paper)", 3), ("mle", "MLE (authors' release)",
            "MLE (release)", 4), ("shape_err", "Shape err (%), real test", r"Shape err.\ (\%)", 2),
            ("trend_err", "Trend err (%), real test", r"Trend err.\ (\%)", 2),
            ("dcr_holdout", "holdout DCR", "holdout DCR", 4)]
    rows_md = [[a_ds_label(ds, False), a_unit(ds, False)] + [fmt(tv[(ds, "real", k)], nd) for k, _, _, nd in cols]
               for ds in A_DS]
    rows_tex = [[a_ds_label(ds, True), a_unit(ds, True)] + [fmt(tv[(ds, "real", k)], nd, True)
                                                           for k, _, _, nd in cols] for ds in A_DS]
    emit("A0", "Real-data reference (TabDDPM framework)",
         "What real data achieve on each metric: the ceiling for MLE and the sampling-noise floor for "
         "Shape / Trend / DCR. Read every GNN or TabDDPM value against its row here.",
         "Real MLE: CatBoost trained on the real train split, test split (paper Table 5 and the authors' "
         "released eval_catboost.json). Shape / Trend / holdout DCR: the real test split scored against "
         "the real train split with exactly the code used for the synthetic samples.",
         ["Dataset", "unit"] + [c[1] for c in cols], rows_md,
         "\\* development dataset; ‡ default: the paper's and the authors' values are on the authors' "
         "version of the data.", ["Dataset", "unit"] + [c[2] for c in cols], rows_tex,
         r"$^{*}$Development dataset; $^\ddagger$authors' version of \textit{default}.",
         "Real-data reference values (TabDDPM framework datasets).", colspec="ll" + "c" * len(cols))


def table_a4(tv):
    cols = [("mle_paper", "MLE (paper) ↑", "MLE (paper) $\\uparrow$", 3),
            ("mle", "MLE (authors' release) ↑", "MLE (release) $\\uparrow$", 4)] + \
           [(k, HEAD[k][0], HEAD[k][1], ND[k]) for k in A_MAIN_COLS if k != "mle"]
    rows_md = [[a_ds_label(ds, False), a_unit(ds, False)] + [fmt(tv[(ds, "tddpm", k)], nd) for k, _, _, nd in cols]
               for ds in A_DS]
    rows_tex = [[a_ds_label(ds, True), a_unit(ds, True)] + [fmt(tv[(ds, "tddpm", k)], nd, True)
                                                           for k, _, _, nd in cols] for ds in A_DS]
    emit("A4", "TabDDPM (original MLP denoiser)",
         "The original model on the same 13 datasets and the same columns as A1–A3 (MLE given twice: the "
         "paper's Table 5 value and the authors' released evaluation file it was rounded from). Every value "
         "is reported except DCR / holdout (our ratio) and the parameter count (we instantiated the "
         "authors' tuned MLP). Fidelity and attack metrics need synthetic samples, which are not on disk.",
         PROTO_TDDPM + " DCR from one generated sample.",
         ["Dataset", "unit"] + [c[1] for c in cols], rows_md,
         "MLE (paper): mean ± std as printed (3 decimals). DCR: the authors' exp/<ds>/ddpm_cb_best/"
         "privacy.json — the same statistic as our DCR, computed by the same code (verified to 4.6e-13 "
         "on our samples, analysis/paper_tables/DCR_RECIPE_CHECK.md), one sample, so no sd. "
         "‡ default: authors' version of the data, not like-for-like.",
         ["Dataset", "unit"] + [c[2] for c in cols], rows_tex,
         r"MLE (paper): as printed in Table~5 of the TabDDPM paper. DCR: the authors' released value, same "
         r"statistic and code as ours, one sample. $^\ddagger$Authors' version of \textit{default}.",
         "TabDDPM (original MLP denoiser) on the same datasets.", colspec="ll" + "c" * len(cols),
         select=lambda x: x["table"] == "A4" and x["metric"] not in A_EXT_COLS)
    ext = ["accuracy", "rmse", "roc_auc", "macro_precision", "macro_recall", "weighted_f1",
           "weighted_precision", "weighted_recall", "dcr_mean_nolabel", "assoc_mean_abs", "assoc_frobenius"]
    rows_md = [[a_ds_label(ds, False)] + [fmt(tv[(ds, "tddpm", k)], ND[k]) for k in ext] for ds in A_DS]
    rows_tex = [[a_ds_label(ds, True)] + [fmt(tv[(ds, "tddpm", k)], ND[k], True) for k in ext] for ds in A_DS]
    emit("A4x", "TabDDPM: the other columns of A1x–A3x",
         "Same columns as A1x–A3x. Accuracy / RMSE / ROC AUC come from the authors' released evaluation "
         "files; the rest were neither reported nor computable without samples.", PROTO_TDDPM,
         ["Dataset"] + [HEAD[k][0] for k in ext], rows_md, "", ["Dataset"] + [HEAD[k][1] for k in ext],
         rows_tex, "", "TabDDPM: the other recorded metrics.",
         select=lambda x: x["table"] == "A4" and x["metric"] in A_EXT_COLS)


def avg_ranks(M: pd.DataFrame, higher_better: bool, nd: int):
    M = M.dropna()
    if M.empty:
        return pd.Series(np.nan, index=M.columns), pd.Series(0, index=M.columns), []
    R = M.round(nd).apply(lambda row: pd.Series(stats.rankdata(-row if higher_better else row),
                                                index=M.columns), axis=1)
    return R.mean(), R.eq(R.min(axis=1), axis=0).sum(), list(M.index)


def table_a5(vals, tv):
    meths = RULES + ["tddpm"]
    lab = {**{r: f"GNN-{r}" for r in RULES}, "tddpm": "TabDDPM"}

    def mat(key, methods, drop=()):
        M = pd.DataFrame(index=[d for d in A_DS if d not in drop], columns=methods, dtype=float)
        for ds in M.index:
            for m in methods:
                rec = vals.get((ds, m, key)) if m != "tddpm" else tv.get((ds, "tddpm", key))
                M.loc[ds, m] = rec["mean"] if rec and rec["status"] == "ok" else np.nan
        return M

    # MLE: ours vs TabDDPM as REPORTED in the paper (3 decimals); DE excluded
    M = mat("mle", RULES)
    M["tddpm"] = [tv[(d, "tddpm", "mle_paper")]["mean"] for d in M.index]
    M = M.drop(index=list(NLFL))
    specs = [("MLE (F1 / R²)", "mle", M, True, 3, "ours vs paper Table 5 (rounded to its 3 decimals); DE excluded"),
             ("Shape err", "shape_err", mat("shape_err", RULES), False, 2, "GNN rules only (TabDDPM: no samples)"),
             ("Trend err", "trend_err", mat("trend_err", RULES), False, 2, "GNN rules only"),
             ("DCR", "dcr", mat("dcr", meths, drop=NLFL), True, 4,
              "larger ranked better (thesis convention); TabDDPM = one sample; DE excluded"),
             ("MIA: distance of AUC from 0.5", "mia", (mat("mia_auc", RULES) - 0.5).abs(), False, 3, "GNN rules only"),
             ("Wasserstein", "wasserstein", mat("wasserstein", RULES), False, 4, "GNN rules only")]
    res = {}
    for name, key, Mx, hb, nd, how in specs:
        rk, best, used = avg_ranks(Mx, hb, nd)
        res[key] = (rk, best, used, how, name)
        for m in Mx.columns:
            V("A5", "; ".join(used), lab[m], f"avg_rank_{key}", rk.get(m, np.nan), n=len(used), kind="computed",
              std_kind="", note=f"derived from A1-A4; {how}; first place on {int(best.get(m, 0))} datasets",
              source="analysis/summary/csv/A1-A4.csv (derived)")
    # Wilcoxon vs TabDDPM (paper) on MLE
    wil = {}
    for r in RULES:
        d = (M[r] - M["tddpm"]).dropna()
        w = stats.wilcoxon(d.to_numpy())
        wil[r] = {"n": len(d), "median": d.median(), "above": int((d > 0).sum()), "below": int((d < 0).sum()),
                  "W": w.statistic, "p": w.pvalue}
    ph = ct.holm([wil[r]["p"] for r in RULES])
    for r, p in zip(RULES, ph):
        wil[r]["p_holm"] = p
        V("A5", "; ".join(M.index), lab[r], "wilcoxon_mle_vs_tabddpm_p", wil[r]["p"], n=wil[r]["n"],
          note=(f"derived; median delta {wil[r]['median']:+.4f}, higher on {wil[r]['above']}, lower on "
                f"{wil[r]['below']}, W={wil[r]['W']:.1f}, Holm p={p:.3f}; TabDDPM = paper Table 5"),
          source="analysis/summary/csv/A1-A4.csv (derived)")
    # sensitivity: authors' release (4 decimals) instead of the paper
    rel_ = mat("mle", RULES + ["tddpm"]).drop(index=list(NLFL))
    sens = {r: stats.wilcoxon((rel_[r] - rel_["tddpm"]).dropna().to_numpy()).pvalue for r in RULES}
    # parameter ratio
    ratio = {r: np.array([vals[(d, r, "params")]["mean"] / tv[(d, "tddpm", "params")]["mean"] for d in A_DS])
             for r in RULES}
    for r in RULES:
        V("A5", "; ".join(A_DS), lab[r], "median_param_ratio_vs_tabddpm", np.median(ratio[r]), n=len(A_DS),
          note=f"derived; min {ratio[r].min():.3f}, max {ratio[r].max():.3f}",
          source="analysis/summary/csv/A1-A4.csv (derived)")

    keys = [s[1] for s in specs]
    head_md = ["Method"] + [f"{res[k][4]} (n={len(res[k][2])})" for k in keys] + \
              ["Wilcoxon MLE vs TabDDPM: median Δ, p (Holm)", "median params GNN / TabDDPM"]
    head_tex = ["Method"] + [tex_escape(res[k][4]) + f" ($n={len(res[k][2])}$)"
                             for k in keys] + [r"Wilcoxon MLE: med.\ $\Delta$, $p$ (Holm)", r"med.\ params ratio"]
    rows_md, rows_tex = [], []
    for m in meths:
        cells_md, cells_tex = [], []
        for k in keys:
            rk, best, used, _, _ = res[k]
            if m not in rk.index or not np.isfinite(rk.get(m, np.nan)):
                cells_md.append("—ⁿ")
                cells_tex.append("---$^{n}$")
            else:
                t = f"{rk[m]:.2f} ({int(best[m])}/{len(used)})"
                cells_md.append(t)
                cells_tex.append(t)
        if m in wil:
            w = wil[m]
            t = f"{w['median']:+.4f}, p = {w['p']:.3f} ({w['p_holm']:.3f})"
            cells_md += [t, f"{np.median(ratio[m]):.3f}"]
            cells_tex += [t.replace("-", "$-$").replace("p =", "$p=$"), f"{np.median(ratio[m]):.3f}"]
        else:
            cells_md += ["(reference)", "1"]
            cells_tex += ["(reference)", "1"]
        rows_md.append([lab[m]] + cells_md)
        rows_tex.append([lab[m]] + cells_tex)
    sens_txt = ", ".join(f"{r} p = {sens[r]:.3f}" for r in RULES)
    emit("A5", "Summary of the TabDDPM framework: average ranks, Wilcoxon, parameter ratio",
         "Each metric cell: average rank (1 = best) over the n datasets where every ranked method has a "
         "value, and in brackets on how many of them the method ranks first (ties count for each). "
         "MLE and DCR rank all four methods; Shape, Trend, MIA and Wasserstein exist for the GNN rules only, "
         "so those ranks are among three and not on the MLE scale.",
         "As A1–A4. Ranks on the displayed precision (MLE 3 decimals = the paper's), ties share the average "
         "rank.",
         head_md, rows_md,
         ("MLE: TabDDPM = the paper's Table 5 value; default (‡) excluded from every column that involves "
          "TabDDPM. DCR: larger is ranked better (the thesis convention; a larger DCR can also reflect lower "
          "fidelity). Wilcoxon: two-sided signed-rank test of the per-dataset MLE difference GNN − TabDDPM "
          "(n = 12, default excluded), Holm-corrected over the three rules. With the authors' released "
          f"values (4 decimals) instead of the paper's: {sens_txt}. Params ratio: median over the 13 "
          "datasets of GNN / TabDDPM trainable parameters (the three rules have identical counts)."),
         head_tex, rows_tex,
         r"MLE vs.\ the paper's TabDDPM values; \textit{default} excluded where TabDDPM is involved. "
         r"DCR: larger ranked better. Wilcoxon: two-sided, $n=12$, Holm over the three rules.",
         "Summary: average rank per metric (first places / $n$), Wilcoxon vs.\\ TabDDPM, parameter ratio.")
    return res, wil, ratio, sens


def table_a6(vals, tv):
    cc = pd.read_csv(REPO / "analysis/config_comparison/config_comparison.csv")
    cc_src = src(REPO / "analysis/config_comparison/config_comparison.csv")
    trees = sorted(rel(p) for p in list((EXP_ROOT / "exp").glob("*/*tddpm_setup*"))
                   + list((REPO / "exp").glob("*/*tddpm_setup*")))
    rows_md, rows_tex = [], []
    for ds in BUDGET_DS:
        g = vals[(ds, "fixed", "mle")]
        V("A6", ds, "GNN fixed (original setup)", "mle", g["mean"], g["std"], n=g["n"], std_kind=g["std_kind"],
          source=g["source"], source_commit=g["source_commit"], producer_commit=g["producer_commit"])
        b = missing("A6", ds, "GNN fixed (TabDDPM setup)", "mle", "no run on disk",
                    "budget-matched run (--train_setup tabddpm, hpc_transfer/jobs_tddpm_setup/) has no results "
                    "on disk")
        for key, model in (("mle_paper", "TabDDPM"), ("mle", "TabDDPM")):
            r_ = tv[(ds, "tddpm", key)]
            V("A6", ds, model + (" (paper)" if key == "mle_paper" else " (authors' release)"), "mle",
              r_["mean"], r_["std"], n=r_["n"], kind="reported", source=r_["source"],
              source_commit=r_["source_commit"])
        setup = {}
        for key in ("steps", "lr", "batch_size", "num_timesteps"):
            x = cc[(cc.dataset == ds) & (cc.key == key)].iloc[0]
            setup[key] = (x.tabddpm_value, x.my_value)
            V("A6", ds, "training setup", key, np.nan, note=f"TabDDPM {x.tabddpm_value} vs ours {x.my_value}",
              kind="computed", status="ok", **cc_src)
        rr, rp = tv[(ds, "real", "mle")], tv[(ds, "real", "mle_paper")]
        row = [g, b, tv[(ds, "tddpm", "mle_paper")], tv[(ds, "tddpm", "mle")], rp, rr]
        nds = [4, 4, 3, 4, 3, 4]
        st = (f"{setup['steps'][1]} / {setup['lr'][1]} / {setup['batch_size'][1]} / {setup['num_timesteps'][1]}",
              f"{setup['steps'][0]} / {setup['lr'][0]} / {setup['batch_size'][0]} / {setup['num_timesteps'][0]}")
        sub = "held-out" if ds not in DEV else "development \\*"
        rows_md.append([ds, sub, a_unit(ds, False)] + [fmt(x, nd) for x, nd in zip(row, nds)] + list(st))
        rows_tex.append([tex_escape(ds), sub.replace("\\*", "$^{*}$"), a_unit(ds, True)]
                        + [fmt(x, nd, True) for x, nd in zip(row, nds)] + list(st))
    head = ["Dataset", "subset", "unit", "GNN (original setup)", "GNN (TabDDPM setup)", "TabDDPM (paper)",
            "TabDDPM (authors' release)", "Real (paper)", "Real (authors' release)",
            "ours: steps / lr / batch / T", "TabDDPM: steps / lr / batch / T"]
    head_tex = ["Dataset", "subset", "unit", "GNN (orig.)", "GNN (TabDDPM setup)", "TabDDPM (paper)",
                "TabDDPM (release)", "Real (paper)", "Real (release)", "ours: steps/lr/batch/$T$",
                "TabDDPM: steps/lr/batch/$T$"]
    emit("A6", "Budget-matched experiment (train_setup = tabddpm)",
         "Does training the same GNN with TabDDPM's own budget (steps, lr, batch, T) close the MLE gap? "
         "Compare column 2 with column 1 (same GNN, same seeds, different training setup) and with TabDDPM.",
         "GNN fixed rule, the configuration of A1, 5 generation × 10 CatBoost seeds; the budget-matched run "
         "changes only lr / steps / batch / num_timesteps / scheduler to the authors' config.toml values.",
         head, rows_md,
         ("Metric: macro-F1 / R², the same rule as A1. The budget-matched runs are prepared "
          "(hpc_transfer/jobs_tddpm_setup/) but no results are on disk: no *_tddpm_setup tree exists under "
          f"transfer_runs/exp/ or exp/ ({'found: ' + ', '.join(trees) if trees else 'none found'}), so "
          "column 2 is —ᵈ (whether the jobs ran on the cluster cannot be seen from here). Setup columns from analysis/config_comparison/config_comparison.csv."),
         head_tex, rows_tex, r"Budget-matched runs prepared; no results on disk.",
         "Budget-matched experiment: GNN with its own vs.\\ TabDDPM's training setup.",
         colspec="lll" + "c" * 8)


# ===========================================================================
# B. TabDiff framework (Route A)
# ===========================================================================
B_DS = ["adult", "beijing", "default", "diabetes", "magic", "shoppers", "news"]
B_ARMS = [("fixed", "fixed"), ("jenks", "adaptive_jenks"), ("entmax15", "adaptive_entmax15")]
# ablation runs (NOT the main protocol): tag -> (rule, config.ABLATIONS key, table)
B_ABL = {"dense": ("fixed", "topk_dense", "B4"),
         **{f"d3_{tag}": (mode, "depth3", "B4b") for tag, mode in B_ARMS}}
B_COLS = [  # key, md header, tex header, decimals, better (+1 / -1)
    ("shape", "Shape err (%) ↓", r"Shape err.\ (\%) $\downarrow$", 2, -1),
    ("trend", "Trend err (%) ↓", r"Trend err.\ (\%) $\downarrow$", 2, -1),
    ("mle1", "MLE-1: AUC / R² ↑", r"MLE-1: AUC / $R^2$ $\uparrow$", 3, +1),
    ("mle2", "MLE-2: F1 ↑ / RMSE ↓", r"MLE-2: $F_1\uparrow$ / RMSE$\downarrow$", 3, 0),
    ("alpha", "α-precision (%) ↑", r"$\alpha$-prec.\ (\%) $\uparrow$", 2, +1),
    ("beta", "β-recall (%) ↑", r"$\beta$-rec.\ (\%) $\uparrow$", 2, +1),
    ("c2st", "C2ST (LR) ↑", r"C2ST (LR) $\uparrow$", 4, +1),
    ("wnmis", "wNMIS ↑", r"wNMIS $\uparrow$", 4, +1),
    ("c2st_xgb", "C2ST (XGBoost) ↑", r"C2ST (XGB) $\uparrow$", 4, +1),
]
B_ND = {k: nd for k, _, _, nd, _ in B_COLS}
PAPER_METRICS = ["shape", "trend", "mle", "alpha", "beta", "c2st"]   # compared with TabDiff (B6)
TDIFF_PAPER = "Shi et al., TabDiff, ICLR 2025"


STEP0 = pd.read_csv(REPO / "hpc_transfer/jobs_tabdiff_gnn/params.csv").set_index("dataset")
STEP0_SRC: dict = {}


def ra_run(ds, mode, ablation=None) -> Path:
    return Path(RC.run_dir(str(EXP_ROOT), RC.target_name(ds), "attngnn", 0, topk_mode=mode, ablation=ablation))


def b_values() -> dict:
    STEP0_SRC.update(src(REPO / "hpc_transfer/jobs_tabdiff_gnn/params.csv"))
    out = {}
    for ds in B_DS:
        reg = RC.TASK[ds] == "regression"
        arms = list(B_ARMS) + [(t, m) for t, (m, abl, _) in B_ABL.items() if ds in RC.ABLATIONS[abl]["datasets"]]
        for tag, mode in arms:
            abl = B_ABL[tag][1] if tag in B_ABL else None
            tid = {"fixed": "B1", "jenks": "B2", "entmax15": "B3"}.get(tag) or B_ABL[tag][2]
            run = ra_run(ds, mode, abl)
            model = "GNN in TabDiff (" + ({"topk_dense": "topk_dense ablation"}.get(abl) or
                                          (f"depth3 ablation, {mode}" if abl else mode)) + ")"
            ev = run / "tabdiff_eval"
            if not run.is_dir():
                st = "no run on disk"
            elif not (ev / "all_results.csv").exists():
                st = "pending"
            else:
                st = "ok"
            if st != "ok":
                for key, *_ in B_COLS:
                    out[(ds, tag, key)] = missing(tid, ds, model, key, st, rel(run) if run.is_dir() else "")
                out[(ds, tag, "train_h")] = missing(tid, ds, model, "train_h", st)
                mp = run / "run_meta.json"
                pc_ = jload(mp).get("param_counts") if mp.exists() else None
                if pc_:
                    out[(ds, tag, "params")] = V(tid, ds, model, "params", pc_["denoiser_total"], n=1,
                                                 note="denoiser parameters, counted at the start of the run",
                                                 **src(mp, f"run_meta git_commit={jload(mp).get('git_commit')}"))
                elif run.is_dir() and abl is None:
                    # unfinished run without counts: the Step-0 count from the same code (params.py),
                    # checked below to equal run_meta wherever both exist
                    out[(ds, tag, "params")] = V(tid, ds, model, "params", STEP0.loc[ds].attngnn, n=1,
                                                 note="run_meta has no count yet; Step-0 count "
                                                      "(tabdiff_gnn/params.py)", **STEP0_SRC)
                else:
                    out[(ds, tag, "params")] = missing(tid, ds, model, "params", st)
                continue
            meta = jload(run / "run_meta.json")
            assert not meta.get("smoke_epochs"), run
            assert (meta.get("ablation") or {}).get("name") == abl, run
            producer = f"run_meta git_commit={meta.get('git_commit')}"
            v = CP.ra_tables(str(run))
            files = [ev / "all_results.csv", ev / "mle2.csv", ev / "quality.csv",
                     run / "tabpc_eval" / "tabpc_results.csv", run / "run_meta.json"]
            # mle2.csv holds both MLE numbers of the same MLE call; its first one must equal 'mle'
            m2 = pd.read_csv(ev / "mle2.csv")
            first = m2["rmse" if reg else "auroc"].to_numpy(float)
            assert np.allclose(first, v["mle"]), run
            v["mle1"], v["mle2"] = (v["mle_alt"], v["mle"]) if reg else (v["mle"], v["mle_alt"])
            per_file = {"shape": 0, "trend": 0, "mle1": 1, "mle2": 1, "c2st": 0, "alpha": 2, "beta": 2,
                        "wnmis": 3, "c2st_xgb": 3}
            for key, *_ in B_COLS:
                a = np.asarray(v[key], float)
                assert len(a) == RC.N_TABLES, (run, key, len(a))
                note = {"mle1": "R2" if reg else "AUROC", "mle2": "RMSE" if reg else "F1",
                        "c2st": "TabDiff's C2ST (logistic regression), /100", "wnmis": "tabpc nmi_l1_weighted_complement",
                        "c2st_xgb": "tabpc xgboost (C2ST)"}.get(key, "")
                out[(ds, tag, key)] = V(tid, ds, model, key, a.mean(), a.std(ddof=1), n=len(a),
                                        std_kind="std over the 20 sampled tables of one trained model (ddof=1)",
                                        note=note, **src([files[per_file[key]], run / "run_meta.json"], producer))
            if abl is None:
                assert meta["param_counts"]["denoiser_total"] == STEP0.loc[ds].attngnn, (ds, tag)
            out[(ds, tag, "params")] = V(tid, ds, model, "params", meta["param_counts"]["denoiser_total"], n=1,
                                         note="denoiser parameters, counted at the start of the run "
                                              f"(gnn {meta['gnn']['gnn_type']} L{meta['gnn']['n_layers']} "
                                              f"d{meta['gnn']['d_model']} h{meta['gnn']['n_heads']} "
                                              f"k{meta['gnn']['top_k']})",
                                         **src(run / "run_meta.json", producer))
            out[(ds, tag, "train_h")] = V(tid, ds, model, "train_h", meta["train_record"]["train_wall_s"] / 3600,
                                          n=1, note=f"train_wall_s on {meta.get('device')} host {meta.get('hostname')}",
                                          **src(run / "run_meta.json", producer))
            if tag in ("jenks", "entmax15"):
                out[(ds, tag, "_kstats")] = run / "topk_k_stats.csv"
    return out


def b_tabdiff_values() -> dict:
    out = {}
    psrc = src(REPO / "hpc_transfer/jobs_tabdiff_gnn/params.csv")
    prm = pd.read_csv(REPO / "hpc_transfer/jobs_tabdiff_gnn/params.csv").set_index("dataset")
    model = "TabDiff (UniModMLP)"
    for ds in B_DS:
        reg = RC.TASK[ds] == "regression"
        for key, *_ in B_COLS:
            if key in ("wnmis", "c2st_xgb"):
                out[(ds, key)] = missing("B5", ds, model, key, "not reported", kind="reported",
                                         note="not in the TabDiff paper")
                continue
            pkey = "mle" if key in ("mle1", "mle2") else key
            if (key == "mle1" and reg) or (key == "mle2" and not reg):
                out[(ds, key)] = missing("B5", ds, model, key, "not reported", kind="reported",
                                         note="paper Table 3 gives AUC (classification) / RMSE (regression) only")
                continue
            c = RP.cell(pkey, "TabDiff", ds)
            s = code_src("analysis/tabdiff/reported.py", f"{TDIFF_PAPER}, {RP.TABLE_OF[pkey]}")
            if not isinstance(c, tuple):
                out[(ds, key)] = missing("B5", ds, model, key, "not reported", kind="reported", **s)
                continue
            out[(ds, key)] = V("B5", ds, model, key, c[0], np.nan if c[1] is None else c[1], n=20,
                               kind="reported", std_kind="std over 20 sampled tables (paper)", **s)
        out[(ds, "params")] = V("B5", ds, model, "params", prm.loc[ds].transformer, n=1,
                                note="TabDiff's UniModMLP from its own toml, instantiated on the dataset's shapes "
                                     "(tabdiff_gnn/params.py)", **psrc)
    return out


def _b_table(tid, title, how, vals, tags, label_of, caption, note="", note_tex="", datasets=None):
    main_tids = {{"fixed": "B1", "jenks": "B2", "entmax15": "B3"}[t] for t in tags if t in dict(B_ARMS)}
    head_md = ["Dataset", "task"] + [c[1] for c in B_COLS] + ["Params"]
    head_tex = ["Dataset", "task"] + [c[2] for c in B_COLS] + ["Params"]
    rows_md, rows_tex, groups = [], [], []
    for ds in datasets or B_DS:
        task = "reg." if RC.TASK[ds] == "regression" else "clf."
        for tag in tags:
            if (ds, tag, "shape") not in vals:
                continue
            if len(tags) > 1 and tag == tags[0]:
                groups.append(len(rows_md))
            lab = ds + (f" — {label_of[(ds, tag)]}" if len(tags) > 1 else "")
            cells = [vals[(ds, tag, k)] for k, *_ in B_COLS] + [vals[(ds, tag, "params")]]
            nds = [B_ND[k] for k, *_ in B_COLS] + ["int"]
            rows_md.append([lab, task] + [fmt(c, nd) for c, nd in zip(cells, nds)])
            rows_tex.append([tex_escape(lab), task] + [fmt(c, nd, True) for c, nd in zip(cells, nds)])
    emit(tid, title, how, PROTO_B, head_md, rows_md,
         ("Mean ± std (ddof=1, TabDiff's aggregation) over the 20 tables sampled from ONE trained model "
          "(training seed 0). MLE-1 = AUROC (classification) or R² (regression); MLE-2 = F1 or RMSE, from "
          "the same MLE call. C2ST (LR) = TabDiff's detection score as a fraction; wNMIS and C2ST (XGBoost) "
          "from tabpc. Params = denoiser parameters. " + note),
         head_tex, rows_tex,
         r"Mean$\pm$std (ddof=1) over 20 tables from one trained model (seed 0). MLE-1: AUROC/$R^2$; "
         r"MLE-2: $F_1$/RMSE. wNMIS, C2ST (XGB): tabpc. " + note_tex, caption,
         groups=groups or None, colspec="ll" + "c" * (len(B_COLS) + 1),
         select=None if len(tags) == 1 else
         (lambda x: x["table"] == tid or (x["table"] in main_tids and x["dataset"] in (datasets or B_DS))))


def b_coverage(bv, tag: str) -> str:
    """Which datasets have results for a rule, read from the run statuses (never hand-written)."""
    st = {d: bv[(d, tag, "shape")]["status"] for d in B_DS}
    ok = [d for d in B_DS if st[d] == "ok"]
    pend = [d for d in B_DS if st[d] == "pending"]
    none = [d for d in B_DS if st[d] == "no run on disk"]
    return ("Results on disk for " + (", ".join(ok) or "no dataset")
            + (f"; still running (pending): {', '.join(pend)}" if pend else "")
            + (f"; no run on disk: {', '.join(none)}" if none else "") + ".")


def table_b(bv, tdv):
    how_rule = ("Rows = the seven TabDiff benchmark datasets, one trained model each, rule **{r}**. "
                "Shape / Trend: TabDiff's error rates (lower better); MLE-1 / MLE-2, α, β, C2ST, wNMIS: higher "
                "better (except RMSE). Compare with B5 (TabDiff as reported) only on the paper's metrics.")
    for tag, mode in B_ARMS:
        tid = {"fixed": "B1", "jenks": "B2", "entmax15": "B3"}[tag]
        note = b_coverage(bv, tag)
        _b_table(tid, f"GNN in TabDiff, {mode} (Route A, d_model 256)", how_rule.format(r=mode), bv, [tag],
                 {}, f"AttnGNN (d\\_model 256) as TabDiff's denoiser, {tex_escape(mode)} rule.",
                 note, tex_escape(note))
    _b_table("B4", "ABLATION: topk_dense (top_k = N − 1) on beijing and magic",
             "**Ablation, not part of the main protocol.** Each dataset's main fixed-rule run next to the same "
             "model with a dense graph (every node keeps all N − 1 others; edge weights still sigmoid scores). "
             "Compare only within a dataset: it asks whether the sparse top-k graph causes the gap.",
             bv, ["fixed", "dense"],
             {**{(d, "fixed"): f"main fixed (k = {jload(ra_run(d, 'fixed') / 'run_meta.json')['gnn']['top_k']})"
                 for d in RC.ABLATIONS["topk_dense"]["datasets"]},
              **{(d, "dense"): f"topk_dense (k = N − 1 = {RC.n_nodes(d) - 1})"
                 for d in RC.ABLATIONS["topk_dense"]["datasets"]}},
             "Ablation: dense graph ($k=N-1$) vs.\\ the main fixed-$k$ run, beijing and magic.",
             "beijing: N − 1 = 11 (main k = 5); magic: N − 1 = 10 (main k = 3).",
             "beijing: $N-1=11$ (main $k=5$); magic: $N-1=10$ (main $k=3$).",
             datasets=list(RC.ABLATIONS["topk_dense"]["datasets"]))
    d3 = list(RC.ABLATIONS["depth3"]["datasets"])
    _b_table("B4b", "ABLATION: depth3 (3 GNN blocks instead of 2) on magic, every rule",
             "**Ablation, not part of the main protocol.** Each rule's main run (2 AttnGNN blocks) next to the "
             "same model with 3 blocks; gnn type, heads, d_model, dim_t, noise mode, TabDiff's protocol and the "
             "seed are unchanged. Compare only within a rule: it asks whether more message passing closes "
             "magic's gap to TabDiff.",
             bv, [x for t, _ in B_ARMS for x in (t, f"d3_{t}")],
             {**{(d, t): f"main {m} (2 blocks)" for d in d3 for t, m in B_ARMS},
              **{(d, f"d3_{t}"): f"depth3 {m} (3 blocks)" for d in d3 for t, m in B_ARMS}},
             "Ablation: three vs.\\ two AttnGNN blocks on magic, each neighbour-selection rule.",
             "Only n_layers differs (2 → 3); run dirs ..._layers3_..., run_meta.json records the ablation.",
             "Only $n_\\text{layers}$ differs ($2\\to3$).", datasets=d3)
    # B5 TabDiff as reported
    rows_md, rows_tex = [], []
    for ds in B_DS:
        task = "reg." if RC.TASK[ds] == "regression" else "clf."
        cells = [tdv[(ds, k)] for k, *_ in B_COLS] + [tdv[(ds, "params")]]
        nds = [B_ND[k] for k, *_ in B_COLS] + ["int"]
        rows_md.append([ds, task] + [fmt(c, nd) for c, nd in zip(cells, nds)])
        rows_tex.append([ds, task] + [fmt(c, nd, True) for c, nd in zip(cells, nds)])
    emit("B5", "TabDiff as reported in its paper (1 model × 20 tables)",
         "The original model's published numbers in the same columns as B1–B3. The paper reports one MLE "
         "number per dataset (AUC for classification → MLE-1, RMSE for regression → MLE-2), no wNMIS and no "
         "XGBoost C2ST.", "TabDiff paper: best-validated model, 20 sampled tables, 8000 epochs (its "
         "protocol). Params: computed by us by instantiating TabDiff's UniModMLP (its own toml) on each "
         "dataset's shapes (Route A Step 0).",
         ["Dataset", "task"] + [c[1] for c in B_COLS] + ["Params"], rows_md,
         "Tables 1, 2 (Shape, Trend), 3 (MLE), 7, 8 (α, β), 9 (C2ST; no std printed) of the TabDiff paper, "
         "transcribed in analysis/tabdiff/reported.py. Params = computed (unmarked).",
         ["Dataset", "task"] + [c[2] for c in B_COLS] + ["Params"], rows_tex,
         r"TabDiff paper Tables 1--3, 7--9 (C2ST printed without std). Params computed by us.",
         "TabDiff as reported in its paper.", colspec="ll" + "c" * (len(B_COLS) + 1))


def paper_key(ds, key):
    """Column holding a paper metric: 'mle' = MLE-1 (classification, AUC) or MLE-2 (regression, RMSE)."""
    if key != "mle":
        return key
    return "mle2" if RC.TASK[ds] == "regression" else "mle1"


def table_b6(bv, tdv):
    better = {"shape": -1, "trend": -1, "alpha": +1, "beta": +1, "c2st": +1}
    detail = []
    for tag, mode in B_ARMS:
        for ds in B_DS:
            reg = RC.TASK[ds] == "regression"
            for key in PAPER_METRICS:
                k = paper_key(ds, key)
                o, t = bv.get((ds, tag, k)), tdv.get((ds, k))
                if o is None or o["status"] != "ok" or t is None or t["status"] != "ok":
                    continue
                sgn = (-1 if reg else +1) if key == "mle" else better[key]
                diff = o["mean"] - t["mean"]
                res = "T" if abs(diff) <= o["std"] else ("W" if sgn * diff > 0 else "L")
                detail.append({"rule": tag, "dataset": ds, "metric": key, "ours": o["mean"], "ours_std": o["std"],
                               "tabdiff": t["mean"], "diff": diff, "result": res})
    D = pd.DataFrame(detail)
    # best arm per dataset (optimistic): max (W - L), then W, then the arm order
    best_rows = []
    for ds in B_DS:
        cand = []
        for i, (tag, _) in enumerate(B_ARMS):
            x = D[(D.dataset == ds) & (D.rule == tag)]
            if len(x):
                w, l = (x.result == "W").sum(), (x.result == "L").sum()
                cand.append((w - l, w, -i, tag))
        if cand:
            tag = max(cand)[3]
            x = D[(D.dataset == ds) & (D.rule == tag)].copy()
            x["rule"] = f"best arm ({tag})"
            best_rows.append(x)
    B = pd.concat(best_rows) if best_rows else pd.DataFrame(columns=D.columns)
    D.assign(table="B6").to_csv(CSV / "B6_detail.csv", index=False)
    B.assign(table="B6").to_csv(CSV / "B6_best_arm_detail.csv", index=False)
    rows_md, rows_tex = [], []
    for name, sub in [(f"GNN {m}", D[D.rule == t]) for t, m in B_ARMS] + \
                     [("best arm per dataset (selected on test, optimistic)", B)]:
        cells = []
        for key in PAPER_METRICS:
            s = sub[sub.metric == key]
            w, t_, l = (s.result == "W").sum(), (s.result == "T").sum(), (s.result == "L").sum()
            cells.append(f"{w} / {t_} / {l}")
            V("B6", "; ".join(s.dataset), name, f"wtl_{key}", np.nan, n=len(s),
              note=f"derived: wins {w}, ties {t_}, losses {l}", source="analysis/summary/csv/B6_detail.csv (derived)")
        w, t_, l = (sub.result == "W").sum(), (sub.result == "T").sum(), (sub.result == "L").sum()
        cells.append(f"{w} / {t_} / {l}")
        nds = sub.dataset.nunique()
        rows_md.append([name, str(nds)] + cells)
        rows_tex.append([tex_escape(name), str(nds)] + cells)
    head = ["GNN arm vs TabDiff (paper)", "datasets"] + \
           ["Shape err", "Trend err", "MLE (AUC / RMSE)", "α-precision", "β-recall", "C2ST (LR)", "all six"]
    head_tex = ["GNN arm vs.\\ TabDiff (paper)", "datasets"] + \
               ["Shape", "Trend", "MLE (AUC/RMSE)", r"$\alpha$", r"$\beta$", "C2ST (LR)", "all six"]
    chosen = ", ".join(f"{ds}: {B[B.dataset == ds].rule.iloc[0][10:-1]}" for ds in B_DS if (B.dataset == ds).any())
    emit("B6", "Wins / ties / losses vs TabDiff (paper metrics only)",
         "For each rule separately: on how many datasets our model beats (W), ties (T) or loses to (L) "
         "TabDiff's reported value, per metric. Only the six metrics the TabDiff paper reports are compared; "
         "wNMIS and C2ST (XGBoost) are never compared with TabDiff.",
         "Ours: B1–B3 (1 model × 20 tables). TabDiff: B5 (paper, 1 model × 20 tables).", head, rows_md,
         ("Tie rule: |ours − TabDiff| ≤ our std over the 20 tables; otherwise W/L by the metric's direction "
          "(Shape, Trend, RMSE lower is better; AUC, α, β, C2ST higher). MLE = AUC on adult, default, diabetes, "
          "magic, shoppers and RMSE on beijing (news pending). Per-dataset outcomes: csv/B6_detail.csv. "
          "**The last row is not a fair comparison**: for each dataset it keeps whichever of our arms has "
          f"the best (W − L) against TabDiff on these same test tables ({chosen}). Choosing among three arms "
          "after seeing their test scores rewards noise (the maximum of several noisy estimates is biased "
          "upward), while TabDiff contributes one pre-specified model; it is an optimistic upper bound, not "
          "a result."),
         head_tex, rows_tex,
         r"Tie: $|\text{ours}-\text{TabDiff}|\le$ our std over 20 tables. Last row: arm chosen per dataset "
         r"on the same test tables---optimistic, not a fair comparison.",
         "Wins / ties / losses against TabDiff's reported values (paper metrics only).",
         colspec="lc" + "c" * 7)
    return D, B


def k_vs_t_groups(bv):
    """(dataset, rule, N-1, kstats path, per-(block, t_bin) sums) of every adaptive run with
    k statistics on disk; sampling phase only."""
    for ds in B_DS:
        n_minus_1 = RC.n_nodes(ds) - 1
        for tag, mode in B_ARMS[1:]:
            p = bv.get((ds, tag, "_kstats"))
            if p is None or not Path(p).exists():
                continue
            t = pd.read_csv(p)
            t = t[t.phase == "sample"]
            # weighted mean over nodes = sum(k_sum) / sum(n): aggregate sums, then divide (no groupby.apply)
            g = t.groupby(["block", "t_bin"], as_index=False)[["n", "k_sum"]].sum()
            g["k_mean"] = g.k_sum / g.n
            yield ds, mode, n_minus_1, p, g


def k_vs_t_records(bv) -> pd.DataFrame:
    """Per (dataset, rule, block, t_bin): the values behind csv/B7_k_vs_t.csv and the figures."""
    recs = []
    for ds, mode, n_minus_1, _, g in k_vs_t_groups(bv):
        for _, x in g.iterrows():
            recs.append({"dataset": ds, "rule": mode, "block": int(x.block), "t_bin": int(x.t_bin),
                         "t_mid": (x.t_bin + 0.5) / 10, "k_mean": x.k_mean, "n": x.n, "N-1": n_minus_1})
    return pd.DataFrame(recs)


def table_b7(bv):
    rows = []
    for ds, mode, n_minus_1, p, g in k_vs_t_groups(bv):
        s = src(p)
        for block, gb in g.groupby("block"):
            lo = gb.loc[gb.t_bin == 0, "k_mean"].iloc[0]
            hi = gb.loc[gb.t_bin == 9, "k_mean"].iloc[0]
            allt = gb.k_sum.sum() / gb.n.sum()
            rows.append({"dataset": ds, "rule": mode, "block": int(block), "N-1": n_minus_1,
                         "k_low": lo, "k_high": hi, "k_all": allt})
            for name, val in (("mean_k_t0.0-0.1", lo), ("mean_k_t0.9-1.0", hi), ("mean_k_all_t", allt)):
                V("B7", ds, f"GNN in TabDiff ({mode}), block {int(block)}", name, val, n=int(gb.n.sum()),
                  note=f"sampling phase; N-1 = {n_minus_1}", **s)
    recs = k_vs_t_records(bv)
    K = pd.DataFrame(rows)
    recs.to_csv(CSV / "B7_k_vs_t.csv", index=False)
    rows_md = [[r.dataset, r.rule, str(r.block), str(r["N-1"]), f"{r.k_low:.2f}", f"{r.k_high:.2f}",
                f"{r.k_all:.2f}", f"{r.k_low / r['N-1']:.2f} → {r.k_high / r['N-1']:.2f}"] for _, r in K.iterrows()]
    rows_tex = [[x[0], tex_escape(x[1]), *x[2:7], x[7].replace("→", r"$\to$")] for x in rows_md]
    emit("B7", "Selected neighbours per node at sampling (adaptive rules)",
         "Mean number of neighbours k each node keeps, during sampling, per block, at low noise (t ∈ [0, 0.1], "
         "nearly clean data) and high noise (t ∈ [0.9, 1], nearly pure noise); N − 1 = every other column. "
         "The fixed rule keeps k = top_k at every t.",
         "Seed-0 runs of B2/B3; statistics logged by tabdiff_gnn/topk_log.py over all sampling forwards of "
         "the 20 tables (t recovered from c_noise), averaged over nodes weighted by count.",
         ["Dataset", "rule", "block", "N − 1", "mean k, t ∈ [0, 0.1]", "mean k, t ∈ [0.9, 1]", "mean k, all t",
          "fraction of N − 1 (low → high t)"], rows_md,
         "Source: each run's topk_k_stats.csv (phase = sample). Per t-bin values behind the figure: "
         "csv/B7_k_vs_t.csv. Figures: figures/topk_k_vs_t.png, normalised by N − 1: "
         "figures/topk_frac_vs_t.png.",
         ["Dataset", "rule", "block", "$N-1$", r"$\bar k$, $t\in[0,.1]$", r"$\bar k$, $t\in[.9,1]$",
          r"$\bar k$, all $t$", r"frac.\ of $N-1$"], rows_tex,
         r"Sampling phase, mean over nodes (count-weighted).", "Mean selected $k$ per block at low and high noise.",
         colspec="llcccccc")
    return K


RULE_TAG = {"adaptive_jenks": "jenks", "adaptive_entmax15": "entmax15"}


def k_vs_t_caption(R: pd.DataFrame, normalise: bool) -> str:
    """Figure caption (LaTeX); which rule has k statistics on which dataset is read from the records."""
    have = {m: [TS.ds_label(d) for d in B_DS if ((R.dataset == d) & (R.rule == m)).any()] for m in RULE_TAG}
    none = [TS.ds_label(d) for d in B_DS if not (R.dataset == d).any()]
    what = (r"fraction of the $N-1$ candidate neighbours that a node keeps, $k/(N-1)$" if normalise
            else r"number of neighbours $k$ that a node keeps")
    avail = "; ".join(f"{TS.rule_label(RULE_TAG[m])}: {', '.join(have[m]) or 'none'}" for m in RULE_TAG)
    return (rf"Mean {what}, during sampling, as a function of the diffusion time $t$ ($t=0$: clean data, "
            r"$t=1$: pure noise), for the two adaptive rules (colour) in each GNN block (line style). Each "
            r"point averages the selected $k$ over all nodes and all denoiser calls whose $t$ falls in a bin "
            r"of width 0.1 (weighted by count), over the 20 sampled tables. The dotted grey line is the "
            r"fixed top-$k$ rule's constant " + ("$k/(N-1)$" if normalise else "$k$") +
            r"; $N-1$ is the number of other columns. Route A, training seed 0, 20 sampled tables. "
            rf"Runs with $k$ statistics: {avail}."
            + (f" No adaptive-rule run of {', '.join(none)} has $k$ statistics yet." if none else ""))


def _place_fixed_label(ax, curves, y, text, **kw):
    """Put the fixed-k label next to its reference line at y without touching any plotted curve:
    try the right end then the left end, just above / below the line, then step away from it
    (a thin leader then joins label and line). Extents are measured, not estimated."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tt = np.linspace(0, 1, 400)
    pts = (ax.transData.transform(np.concatenate([np.c_[tt, np.interp(tt, cx, cy)] for cx, cy in curves]))
           if curves else np.empty((0, 2)))
    t = ax.text(0, y, text, transform=ax.get_yaxis_transform(), **kw)
    ybox = ax.get_window_extent(r)
    line_px = ax.transData.transform((0, y))[1]
    gap = 1.5 * fig.dpi / 72
    pad = 2 * fig.dpi / 72

    def hits(x0, x1, y0, y1):
        return bool(((pts[:, 0] >= x0 - pad) & (pts[:, 0] <= x1 + pad) & (pts[:, 1] >= y0 - pad)
                     & (pts[:, 1] <= y1 + pad)).any())

    for step in range(0, 14):
        for ha, x in (("right", 0.985), ("left", 0.015), ("center", 0.5)):
            for va in ("bottom", "top"):
                off = (gap + step * 3 * fig.dpi / 72) * (1 if va == "bottom" else -1)
                ty = ax.transData.inverted().transform((0, line_px + off))[1]
                t.set_position((x, ty))
                t.set_ha(ha)
                t.set_va(va)
                bb = t.get_window_extent(r)
                if hits(bb.x0, bb.x1, bb.y0, bb.y1) or bb.y0 < ybox.y0 or bb.y1 > ybox.y1:
                    continue
                if step:                       # leader from the label to its line, only if it crosses no data
                    xl = bb.x1 - 4 if ha == "right" else bb.x0 + 4 if ha == "left" else (bb.x0 + bb.x1) / 2
                    ye = bb.y0 if va == "bottom" else bb.y1
                    if not hits(xl, xl, min(line_px, ye), max(line_px, ye)):
                        xa = ax.transAxes.inverted().transform((xl, 0))[0]
                        ya = ax.transData.inverted().transform((0, ye))[1]
                        ax.plot([xa, xa], [y, ya], transform=ax.get_yaxis_transform(), color=kw.get("color"),
                                lw=0.5, zorder=1)
                return t
    t.set_position((0.985, y))
    t.set_ha("right")
    t.set_va("bottom")
    return t


def fig_k_vs_t(R: pd.DataFrame, normalise: bool = False):
    """Mean selected k vs t per dataset and block (topk_k_vs_t, appendix); normalise=True divides by
    N - 1, the number of candidate neighbours (topk_frac_vs_t, main text)."""
    dss = [d for d in B_DS if (R.dataset == d).any()]
    ncol = 3
    nrow = int(np.ceil(len(dss) / ncol))
    fig = TS.figure(TS.FULL, 2.1 * nrow)
    axes = fig.subplots(nrow, ncol, squeeze=False, sharex=True, sharey=normalise)
    fixed_labels = []
    for i, (ax, ds) in enumerate(zip(axes.flat, dss)):
        TS.style_axes(ax)
        sub = R[R.dataset == ds]
        n1 = int(sub["N-1"].iloc[0])
        div = n1 if normalise else 1
        curves = []
        for mode, tag in RULE_TAG.items():
            for b, gb in sub[sub.rule == mode].groupby("block"):
                y = (gb.k_mean / div).to_numpy()
                ax.plot(gb.t_mid, y, ls=TS.BLOCK_LS[int(b) % 3], color=TS.RULE_COLOR[tag], lw=TS.LW_DATA,
                        marker=TS.RULE_MARKER[tag], ms=3, mew=0.4)
                curves.append((gb.t_mid.to_numpy(), y))
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        if normalise:
            ax.set_ylim(0, 1)
        else:
            top = max(max(c[1].max() for c in curves), 0)
        fixed_k = jload(ra_run(ds, "fixed") / "run_meta.json")["gnn"]["top_k"] \
            if (ra_run(ds, "fixed") / "run_meta.json").exists() else None
        if not normalise:
            ax.set_ylim(0, 1.12 * max(top, fixed_k or 0))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
        if fixed_k:
            yf = fixed_k / div
            ax.axhline(yf, color=TS.RULE_COLOR["fixed"], lw=0.9, ls=(0, (1.2, 1.6)), zorder=1)
            lab = (rf"Fixed $k/(N-1) = {fixed_k}/{n1}$" if normalise else rf"Fixed $k = {fixed_k}$")
            fixed_labels.append((ax, curves, yf, lab))
        TS.title2(ax, rf"{TS.ds_label(ds)} ($N-1 = {n1}$)")
        TS.panel_label(ax, "abcdefgh"[i])
        if i % ncol == 0:
            ax.set_ylabel(r"Mean $k/(N-1)$" if normalise else r"Mean selected $k$")
        if i // ncol == nrow - 1 or i + ncol >= len(dss):
            ax.set_xlabel(r"Diffusion time $t$")
            ax.xaxis.set_tick_params(labelbottom=True)
    for ax in list(axes.flat)[len(dss):]:
        ax.axis("off")
    blank = plt.Line2D([], [], lw=0, alpha=0)
    handles = ([blank] + [plt.Line2D([], [], color=TS.RULE_COLOR[t], lw=TS.LW_DATA, marker=TS.RULE_MARKER[t], ms=3,
                                     mew=0.4, mec="white") for t in ("jenks", "entmax15")]
               + [plt.Line2D([], [], color=TS.RULE_COLOR["fixed"], lw=0.9, ls=(0, (1.2, 1.6)))]
               + [blank] + [plt.Line2D([], [], color=TS.INK2, lw=TS.LW_DATA, ls=ls) for ls in TS.BLOCK_LS])
    labels = (["Rule:", TS.rule_label("jenks"), TS.rule_label("entmax15"), TS.rule_label("fixed")]
              + ["Block:"] + [f"Block {i}" for i in range(3)])
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), handlelength=2.2,
               columnspacing=1.0, handletextpad=0.4)
    for ax, curves, yf, lab in fixed_labels:          # after the layout is final: measured placement
        _place_fixed_label(ax, curves, yf, lab, fontsize=TS.FS_CELL + 0.5, color=TS.RULE_COLOR["fixed"])
    name = "topk_frac_vs_t" if normalise else "topk_k_vs_t"
    TS.save(fig, name, k_vs_t_caption(R, normalise))


# ===========================================================================
# C. Model size across frameworks
# ===========================================================================
def table_c(vals, tv, bv, tdv):
    union = A_DS + [d for d in B_DS if d not in A_DS]
    rows_md, rows_tex = [], []
    for ds in union:
        a = vals.get((ds, "fixed", "params")) if ds in A_DS else None
        if a is not None:
            same_a = all(vals[(ds, r, "params")]["mean"] == a["mean"] for r in RULES)
            assert same_a, ds
        t = tv.get((ds, "tddpm", "params")) if ds in A_DS else None
        b_avail = [bv[(ds, tag, "params")] for tag, _ in B_ARMS if ds in B_DS and bv[(ds, tag, "params")]["status"] == "ok"]
        if b_avail:
            assert len({x["mean"] for x in b_avail}) == 1, ds
        b = b_avail[0] if b_avail else (None if ds not in B_DS else
                                         missing("C", ds, "GNN in TabDiff", "params", "pending"))
        d = tdv.get((ds, "params")) if ds in B_DS else None
        g = RG.TABDIFF_COST.get(ds) if ds in B_DS else None
        na = lambda: {"status": "not applicable"}  # noqa: E731
        r1 = (a["mean"] / t["mean"]) if a and t else None
        r2 = (b["mean"] / d["mean"]) if b and b.get("status") == "ok" and d else None
        for nm, x in (("GNN, TabDDPM framework (all rules)", a), ("TabDDPM", t), ("GNN in TabDiff d256 (all rules)", b),
                      ("TabDiff", d)):
            if x is not None and x.get("status") == "ok":
                V("C", ds, nm, "params", x["mean"], n=1, source=x["source"], source_commit=x["source_commit"],
                  producer_commit=x.get("producer_commit", ""))
        if r1:
            V("C", ds, "ratio GNN / TabDDPM", "params_ratio", r1, note="derived")
        if r2:
            V("C", ds, "ratio GNN / TabDiff", "params_ratio", r2, note="derived")
        if g:
            V("C", ds, "TabDiff (GATD Table 16)", "params", np.nan, kind="reported",
              note=f"{g[0]} as printed", **code_src("analysis/tabdiff/reported_gatd.py", f"{RG.SOURCE}, Table 16"))
        name = ds + (" ◊" if ds == "diabetes" else "")
        cells = [a or na(), t or na(), b or na(), d or na()]
        ratio_cells = [f"{r1:.3f}" if r1 else "—ᵃ",
                       f"{r2:.3f}" if r2 else ("**pending**" if b and b.get("status") == "pending" else "—ᵃ")]
        gatd = (g[0] + "†") if g else "—ᵃ"
        rows_md.append([name] + [fmt(c, "int") for c in cells] + ratio_cells + [gatd])
        rows_tex.append([tex_escape(name)] + [fmt(c, "int", True) for c in cells]
                        + [x.replace("—ᵃ", "---$^{a}$").replace("**pending**", r"\emph{pending}") for x in ratio_cells]
                        + [(g[0] + r"$^\dagger$") if g else "---$^{a}$"])
    head = ["Dataset", "GNN, TabDDPM framework (fixed = jenks = entmax15)", "TabDDPM (MLP)",
            "GNN in TabDiff, d_model 256 (fixed = jenks = entmax15)", "TabDiff (UniModMLP)",
            "GNN / TabDDPM", "GNN / TabDiff", "TabDiff, GATD Table 16"]
    head_tex = ["Dataset", "GNN (TabDDPM fw.)", "TabDDPM", r"GNN in TabDiff ($d=256$)", "TabDiff",
                "GNN/TabDDPM", "GNN/TabDiff", "TabDiff (GATD)"]
    emit("C", "Model size across frameworks (trainable denoiser parameters)",
         "One row per dataset name; a model has a value only where it was run on that dataset (—ᵃ "
         "otherwise). Ratios < 1 mean the GNN is smaller than the original denoiser it replaces.",
         "All counts are of the denoiser alone, instantiated on the dataset's real shapes (TabDDPM framework: "
         "analysis/config_comparison/param_counts.csv; Route A: run_meta.json for the GNN, "
         "hpc_transfer/jobs_tabdiff_gnn/params.csv for TabDiff).", head, rows_md,
         ("The three rules have identical parameter counts in both frameworks (checked here: the rules choose "
          "a mask and add no parameters). TabDiff's learnable noise schedules (d_num + n_cat scalars) are not "
          "included. ◊ diabetes is two different datasets: TabDDPM's is Pima (8 columns), TabDiff's is the "
          "CDC diabetes set (37 columns); the TabDDPM-framework columns refer to the first, the TabDiff ones "
          "to the second. adult and default also differ in splits / typing between the frameworks."),
         head_tex, rows_tex,
         r"Denoiser parameters on each dataset's shapes; rules add none. $^\diamond$Different datasets in "
         r"the two frameworks (Pima vs.\ CDC).", "Trainable denoiser parameters across frameworks.",
         select=lambda x: x["table"] == "C" and x["metric"] != "train_h")
    # wall time
    rows_md, rows_tex = [], []
    for ds in B_DS:
        cells = [bv[(ds, tag, "train_h")] for tag, _ in B_ARMS]
        cells.append(bv[(ds, "dense", "train_h")] if (ds, "dense", "train_h") in bv else {"status": "not applicable"})
        g = RG.TABDIFF_COST.get(ds)
        if g:
            V("C", ds, "TabDiff (GATD Table 16, T4)", "train_h", g[1], n=1, kind="reported",
              **code_src("analysis/tabdiff/reported_gatd.py", f"{RG.SOURCE}, Table 16"))
        gat = {"status": "ok", "mean": g[1], "std": np.nan, "kind": "reported"} if g else {"status": "not reported"}
        rows_md.append([ds] + [fmt(c, 1) for c in cells] + [fmt(gat, 2)])
        rows_tex.append([ds] + [fmt(c, 1, True) for c in cells] + [fmt(gat, 2, True)])
    emit("C2", "Training wall time per run (hours)",
         "Hours of training (not sampling or evaluation) per run. Only Route A records it; the TabDDPM-"
         "framework runs and the authors' TabDDPM runs record no wall time (—ⁿ, so not shown).",
         "Route A: train_record.train_wall_s in run_meta.json, one NVIDIA A100 (gpu partition), 8000 epochs. "
         "TabDiff: GATD Table 16, one NVIDIA T4, 8000 epochs — different GPU, so only an order of magnitude.",
         ["Dataset", "GNN fixed", "GNN adaptive_jenks", "GNN adaptive_entmax15", "GNN topk_dense (ablation)",
          "TabDiff (GATD, T4)"], rows_md, "news: the entmax15 run is still training (pending).",
         ["Dataset", "GNN fixed", "GNN Jenks", "GNN entmax-1.5", "GNN dense (abl.)", "TabDiff (GATD, T4)"],
         rows_tex, "", "Training wall time per run (hours).",
         select=lambda x: x["metric"] == "train_h")


def table_appendix_gatd():
    rows_md, rows_tex = [], []
    for ds in B_DS:
        reg = RC.TASK[ds] == "regression"
        cells = []
        for key in ("shape", "trend", "mle1", "mle2"):
            c = RG.cell(key, ds)
            s = code_src("analysis/tabdiff/reported_gatd.py", f"{RG.SOURCE}, Table 9")
            cells.append(V("GATD", ds, "TabDiff (GATD re-run, 3 seeds x 20 tables)", key, c[0], c[1], n=60,
                           kind="reported", std_kind="std over 3 training seeds x 20 tables", **s))
        nds = [2, 2, 3, 3]
        rows_md.append([ds, "reg." if reg else "clf."] + [fmt(c, nd) for c, nd in zip(cells, nds)])
        rows_tex.append([ds, "reg." if reg else "clf."] + [fmt(c, nd, True) for c, nd in zip(cells, nds)])
    emit("GATD", "APPENDIX: TabDiff re-run by GATD (3 training seeds × 20 tables, other protocol)",
         "Not part of the comparison tables: a different protocol (three trained models, so its std includes "
         "retraining variance, which ours does not). Shown only as context for B5.",
         "GATD (Zagardo, ICML 2026, arXiv:2606.02607v1), Table 9: TabDiff from the official code, 8000 epochs, "
         "3 training seeds × 20 tables.",
         ["Dataset", "task", "Shape err (%)", "Trend err (%)", "MLE-1: AUROC / R²", "MLE-2: F1 / RMSE"], rows_md,
         "Transcribed in analysis/tabdiff/reported_gatd.py.",
         ["Dataset", "task", r"Shape err.\ (\%)", r"Trend err.\ (\%)", r"MLE-1: AUROC/$R^2$", r"MLE-2: $F_1$/RMSE"],
         rows_tex, "", "Appendix: TabDiff as re-run by GATD (3 training seeds $\\times$ 20 tables).")


# ===========================================================================
# D. Differentiability check (static facts + the test's outcome)
# ===========================================================================
def line_of(path: str, pattern: str) -> str:
    for i, ln in enumerate((REPO / path).read_text().splitlines(), 1):
        if pattern in ln:
            return f"{path}:{i}"
    raise RuntimeError(f"{pattern!r} not found in {path}")


def section_d() -> str:
    test = "tests/test_selection_differentiability.py"
    outcome = "not run (SUMMARY_PYTEST=0)"
    if os.environ.get("SUMMARY_PYTEST", "1") != "0":
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", test], cwd=REPO, capture_output=True,
                           text=True)
        last = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        outcome = (last[-1] if last else "no output") + f" (exit code {r.returncode})"
        try:
            import entmax  # noqa: F401
            em = f"entmax {entmax.__version__} available"
        except ImportError:
            em = "entmax NOT installed here, so the entmax-1.5 cases were skipped"
        outcome += f"; {em}; run {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} on CPU, torch " + \
                   __import__("torch").__version__
    L = [
        "**Where the selection is computed.** Each AttnGNN block owns a `DynamicAdjacency` "
        f"(`{line_of('tab_ddpm/graph_builder.py', 'class DynamicAdjacency')}`). Its forward computes the raw "
        f"scores φ = W_Q h (W_K h)ᵀ / √d (`{line_of('tab_ddpm/graph_builder.py', 'scores = torch.bmm(Q, K.transpose(1, 2))')}`), "
        "then one of three rules picks the kept edges:",
        f"- **fixed**: `torch.topk` on the off-diagonal scores, the rest set to −1e9 before the sigmoid "
        f"(`{line_of('tab_ddpm/graph_builder.py', '_, topk_idx = torch.topk(scores_no_self')}`).",
        f"- **adaptive_jenks**: `build_adjacency` (`{line_of('adaptive_adjacency.py', 'def build_adjacency(')}`); "
        f"the per-node budget k comes from `jenks_budget` (`{line_of('adaptive_adjacency.py', 'def jenks_budget')}`), "
        "a 2-class split of the sorted scores, run under `@torch.no_grad()` on `s.detach()`; the mask is "
        "`phi_off >= thresh` (boolean).",
        f"- **adaptive_entmax15**: `build_adjacency_entmax` (`{line_of('adaptive_adjacency.py', 'def build_adjacency_entmax')}`); "
        f"the mask is the support `p > 0` of entmax-1.5 computed by `entmax_support` "
        f"(`{line_of('adaptive_adjacency.py', 'def entmax_support')}`) under `@torch.no_grad()` on `phi.detach()`.",
        "- In all three, the edge weights are `sigmoid(φ)` on kept edges and exactly 0 elsewhere (masked to −1e9 "
        "before the sigmoid); self-loops are set to 1. The GNN layers then use these weights: GCN / GIN as "
        "edge weights, GAT / GATv2 as an additive log-gate `log(adj.clamp(min=1e-12))` on the attention "
        f"logits (`{line_of('tab_ddpm/gnn_layers.py', 'adj_gate = torch.log(')}`).",
        "",
        "**Are the entmax probabilities used in the forward pass?** No — only their support. `entmax_support` "
        "returns the boolean mask and the support size; the probabilities never reach the adjacency. The test "
        "rescales them (×7, same support) and the model output stays bit-identical.",
        "",
        "**Is the selection a discrete mask and do gradients reach W_Q / W_K and the rest of the denoiser?** "
        "Yes, for all three rules. "
        f"`{test}` builds AttnGNN (d_model 256) with the GNN configuration of magic (GATv2, k = 3), shoppers "
        "(GCN, k = 5) and beijing (GATv2, k = 5) inside TabDiff's `UnifiedCtimeDiffusion` (learnable per-column "
        "schedules, EDM preconditioning) on a small synthetic batch (64 rows), runs TabDiff's own loss "
        "`mixed_loss` weighted as its Trainer does, and backpropagates. Per rule and per block it checks: "
        "off-diagonal weights exactly 0 off the mask and exactly sigmoid(φ) on it, diagonal exactly 1; the mask "
        "equals the rule's own (top-k indices / an independent brute-force Jenks split / the entmax-1.5 "
        "support); ∂L/∂φ is exactly 0 on every dropped edge and on the diagonal and non-zero on kept edges; "
        "W_Q and W_K receive finite, non-zero gradients in every block; every other denoiser parameter receives "
        "a finite gradient and every sub-module (tokenizer, noise embedding, GNN blocks, reconstructor) a "
        "non-zero one.",
        "",
        "**One expected exception, found by the test.** Some kept edges get exactly zero gradient: all of them "
        "point into categorical nodes that are *not* masked at the sampled t. TabDiff's masked (absorbing-"
        "state) diffusion loss has no term for an unmasked categorical cell, so nothing flows back into that "
        "node's incoming edges (in the last block; earlier blocks only through other loss-free nodes). This is "
        "a property of the loss, not a blocked path; the test asserts that every kept edge into a node with "
        "a loss term carries gradient.",
        "",
        "The TabDDPM-framework denoiser (`GraphAwareDenoiser`) uses the same `AttnGNNBlock` / "
        "`DynamicAdjacency`, so the mask and edge-weight construction are identical there; the gradient test "
        "runs the TabDiff loss only, as asked.",
        "",
        "**No unintended .detach() / no_grad.** The test also audits every `.detach()` / `torch.no_grad()` on "
        "the forward path (adaptive_adjacency.py, tab_ddpm/graph_builder.py, graph_denoiser.py, gnn_layers.py, "
        "tabdiff_gnn/denoiser.py, topk_log.py) against a list with the reason each is harmless: the Jenks "
        "budget and the entmax support (integer / boolean outputs, not differentiable anyway), the k-logging, "
        "and the `node_noise='per_column'` path that no run uses. Any new one fails the test.",
        "",
        "**Wording to fix in the code's own docs.** `DynamicAdjacency`'s docstring says the top-k result is "
        "\"differentiable via a straight-through mask\". It is not straight-through: dropped edges get exactly "
        "zero gradient (no surrogate gradient is passed through the selection); kept edges are differentiable "
        "through the sigmoid. The selection itself is not differentiated in any of the three rules.",
        "",
        f"**Test outcome:** `{outcome}`.",
    ]
    return "\n".join(L)


# ===========================================================================
# Protocol texts, summary.md
# ===========================================================================
PROTO_A = ("TabDDPM framework (Gaussian + multinomial diffusion, TabDDPM's code); one trained model per cell "
           "(training seed 0), 20k steps, lr 1e-4, batch 4096, T = 1000 for every dataset (ablation_runner "
           "constants; TabDDPM's tuned budgets differ, see A6), EMA weights sampled; 5 generation seeds; MLE "
           "with the authors' tuned CatBoost, 10 CatBoost seeds per sample, test split.")
PROTO_TDDPM = ("Authors' protocol: tuned hyper-parameters per dataset (30k steps, tuned lr / batch / T), "
               "5 sampling seeds × 10 CatBoost seeds, test split; values from their released "
               "exp/<ds>/ddpm_cb_best/eval_catboost.json.")
PROTO_B = ("Route A: TabDiff's framework (EDM + masked diffusion, learnable schedules, Trainer, sampler, "
           "evaluation code @5ecdb33) with AttnGNN V2 (d_model 256) as the denoiser; TabDiff's own splits and "
           "budget (8000 epochs, lr 1e-3, batch 4096, 50 sampling steps); training seed 0 (1 trained model), "
           "checkpoint = best EMA training loss after epoch 4000, 20 tables of n_train rows (generation seeds "
           "0–19); wNMIS and C2ST (XGBoost) from tabpc @b821a47.")


def caveats(bv) -> str:
    dev = ", ".join(sorted(DEV))
    return "\n".join([
        f"- **Configuration selection on test (A, 9 development datasets: {dev}).** Their GNN configuration "
        "was chosen by the fixed rule's test-split score in our ablation: in-sample and optimistic, most of "
        "all for the fixed rule. cardio, default, house and king are held-out (configuration transferred "
        "without seeing their results). TabDDPM's authors tuned on validation.",
        "- **default (‡) is not like-for-like with TabDDPM.** Our local data are 14 numerical + 9 categorical "
        "columns with val/test 6000/4800; the authors' are 20 + 3 with 4800/6000. Every paper / release value "
        "on default is on their version; default is excluded from every rank and test involving TabDDPM.",
        "- **Training budget (A).** Our GNN runs use 20k steps, lr 1e-4, batch 4096, T = 1000 everywhere; "
        "TabDDPM used 30k steps and tuned lr (1.15e-5 … 2.8e-3), T = 100 on six datasets, batch 256 on wilt, "
        "and samples raw (not EMA) weights. The budget-matched runs (A6) that would isolate this have no "
        "results on disk.",
        "- **One training seed everywhere.** All std in A are over generation × CatBoost seeds of one trained "
        "model, all std in B over 20 tables of one trained model: training-seed variance is not measured.",
        "- **Excluded (A).** higgs-small, miniboone and fb-comments are not in any table.",
        "- **Shape / Trend in A are our implementation** (SDMetrics column shapes; pair trends on shared real "
        "bins over all pairs) and are **not comparable** with the Shape / Trend of B, which are TabDiff's own "
        "code on TabDiff's splits.",
        "- **TabDDPM DCR** is the authors' value from one generated sample; it is the same statistic, by the "
        "same code, as ours (verified). The ranks treat a larger DCR as better, which is a convention: a large "
        "DCR can also mean low fidelity; DCR / holdout ≈ 1 is the meaningful reference.",
        "- **cat_input.** Datasets with categorical columns use the *_catfix trees (cat_input = prob); the "
        "others (california, diabetes, gesture, house, wilt) use the original trees, where the option has no "
        "effect. MIA and Wasserstein of the catfix runs are a post-hoc recomputation from the saved samples "
        "(analysis/integrity/posthoc_gen_metrics.csv, exact reproduction 1e-10).",
        "- **B is compared with TabDiff as reported, not reproduced.** The paper's model is the "
        "best-validated one; ours is the best-EMA-loss checkpoint of a single run. The paper prints one MLE "
        "number per dataset (AUC / RMSE). wNMIS and C2ST (XGBoost) are never compared with TabDiff.",
        "- **B code versions.** The fixed-arm runs of adult, beijing, magic and shoppers were made at 7856cf5, "
        "those of default and diabetes at 9f86515, the rule arms at de83865, the dense ablation at 008ffe0 "
        "(beijing) / 9f86515 (magic); the fixed-arm denoiser is pinned bit-identical across these commits by "
        "tests/test_tabdiff_gnn.py.",
        "- **B6 best-arm line** is selected on the test tables and is optimistic by construction.",
        "- **news** (B): " + "; ".join(f"{m}: {bv[('news', t, 'shape')]['status']}" for t, m in B_ARMS) + ".",
        "- **diabetes** is Pima in A and the CDC dataset in B: two different datasets under one name.",
        "- **Figure style** (analysis/summary/thesis_style.py): one palette in every figure — Fixed top-k grey "
        "#6b6b6b (square), Jenks blue #1f6fd1 (circle), entmax-1.5 orange #e8662e (triangle), Dense black "
        "#111111 (diamond); datasets (Figure 3 only) Okabe–Ito without black and yellow; sequential maps "
        "Purples, diverging RdBu (blue = better). Text in Computer Modern (CMU Serif converted to TrueType, "
        "analysis/summary/fonts/, OFL) because no LaTeX engine is installed here; with a `latex` binary on the "
        "PATH the figures switch to text.usetex. The palette validator was not run (no Node.js here).",
    ])


def write_summary(xcheck, a5, b6, files, bv):
    head = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--", "analysis", "tests"))
    secs = {tid: md for tid, md in SECTIONS if tid}
    other = [md for tid, md in SECTIONS if not tid]
    L = ["# Comparison tables (supervisor summary)", "",
         f"_Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by `analysis/summary/build_summary.py` at "
         f"commit `{head}`{' (+ uncommitted changes)' if dirty else ''}. Rerun it to refresh; nothing is "
         "trained or sampled._", "",
         "## How values are marked", "",
         "- Every value comes from a results file; `csv/<table>.csv` lists, per value, its **kind** "
         "(`computed` by us / `reported` from a paper or the authors' release), its **source** file(s) and "
         "the **commit** (the last git commit touching a tracked file; for gitignored run outputs "
         "`untracked` + the file's sha256/16, plus the producing run's own `git_commit` from its "
         "run_meta.json). `provenance.csv` lists every file read. Ranks, ratios and win/tie/loss counts are "
         "marked `derived`.",
         "- In the tables: " + LEGEND_MD,
         f"- Cross-check: every A-table mean that also appears in a run's own results_full_averaged.json "
         f"matches it (max |difference| = {max(xcheck):.1e} over {len(xcheck)} values).", "",
         "## A. TabDDPM framework (thesis core)", "",
         "13 datasets (higgs-small, miniboone, fb-comments excluded). Layout: our GNN denoiser, one table per "
         "neighbour-selection rule (A1–A3, each followed by its block of all other recorded metrics), then "
         "TabDDPM on the same datasets and columns (A4), the summary (A5) and the budget-matched experiment "
         "(A6). A0 gives the real-data reference.", ""]
    for t in ("A0", "A1", "A1x", "A2", "A2x", "A3", "A3x", "A4", "A4x", "A5", "A6"):
        L.append(secs[t])
    L += ["## B. TabDiff framework (Route A, seed 0 = 1 trained model × 20 tables)", "",
          "Datasets: adult, beijing, default, diabetes, magic, shoppers; news pending. GNN tables per rule "
          "(B1–B3), the dense-graph ablation (B4), the depth ablation (B4b), then TabDiff as reported (B5), the win/tie/loss summary "
          "(B6) and the selected-neighbour statistics (B7).", ""]
    for t in ("B1", "B2", "B3", "B4", "B4b", "B5", "B6", "B7"):
        L.append(secs[t])
    L += ["Figures of the selected k: Figure 4 (`topk_frac_vs_t`) and appendix `topk_k_vs_t`, see "
          "*Thesis figures* below.", "",
          "## C. Model size across frameworks", "", secs["C"], secs["C2"],
          "## D. Differentiability of the selection rules", "", section_d(), "",
          "## Thesis figures (Route A)", "", "All figures are drawn at their final size for an A4 thesis "
          "(text width 6.27 in; `\\includegraphics[width=\\textwidth]` is 1:1) in the house style of "
          "analysis/summary/thesis_style.py; each has a LaTeX caption in figures/<name>.caption.tex (shown here "
          "as plain text). One command rebuilds them all: `python3 analysis/summary/build_summary.py "
          "--figures-only`. Contact sheet: figures/_contact_sheet.pdf; LaTeX preview: figures/_preview.tex.", "",
          figures_section(), "",
          "## Appendix", "", secs["GATD"], "## Caveats", "", caveats(bv), "",
          "## Files", "", *[f"- `{f}`" for f in files], ""]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


DEV: set = set()


# ===========================================================================
# Thesis figures (all of them; analysis/summary/thesis_style.py is the house style)
# ===========================================================================
THESIS_FIGURES = [  # name, place in the thesis, script
    ("selection_example", "Figure 2", "fig_selection_example.py"),
    ("density_vs_fidelity", "Figure 3", "fig_density_vs_fidelity.py"),
    ("topk_frac_vs_t", "Figure 4", "build_summary.py"),
    ("vs_tabdiff_reldiff", "Figure 5", "fig_vs_tabdiff_heatmap.py"),
    ("pair_trend_error_beijing", "Figure 6", "fig_pairs_marginals.py"),
    ("marginals", "Figure 7", "fig_pairs_marginals.py"),
    ("paper_style_marginals", "Paper-style (TabDiff Fig. 3)", "fig_paper_style.py"),
    ("paper_style_pair_heatmaps", "Paper-style (TabDiff Fig. 4)", "fig_paper_style.py"),
    ("topk_k_vs_t", "Appendix", "build_summary.py"),
    ("pair_trend_error_default", "Appendix", "fig_pairs_marginals.py"),
    ("pair_trend_error_magic", "Appendix", "fig_pairs_marginals.py"),
    ("marginals_appendix", "Appendix", "fig_pairs_marginals.py"),
]
FIG_STATUS: dict[str, str] = {}


def build_figures(bv, tdv) -> dict[str, str]:
    """Every thesis figure + figures/_contact_sheet.pdf + figures/_preview.tex. A figure whose
    inputs are missing here (entmax for Figure 2, <run>/analysis/ for Figures 6-7) is left as it is
    on disk and reported as not rebuilt."""
    import route_a_figs as RF
    RF._BV, RF._TDV = bv, tdv
    recs = k_vs_t_records(bv)
    fig_k_vs_t(recs)
    fig_k_vs_t(recs, normalise=True)
    FIG_STATUS.update({"topk_k_vs_t": "rebuilt", "topk_frac_vs_t": "rebuilt"})
    try:
        import entmax  # noqa: F401
        import fig_selection_example as FSE
        FSE.main()
        FIG_STATUS["selection_example"] = "rebuilt"
    except ImportError as e:
        FIG_STATUS["selection_example"] = f"NOT rebuilt ({e}; needs entmax==1.3, as the model)"
    import fig_density_vs_fidelity as FDV
    FDV.main()
    FIG_STATUS["density_vs_fidelity"] = "rebuilt"
    import fig_vs_tabdiff_heatmap as FVT
    FVT.main([])
    FIG_STATUS["vs_tabdiff_reldiff"] = "rebuilt"
    import fig_pairs_marginals as FPM
    pm = [n for n, _, scr in THESIS_FIGURES if scr == "fig_pairs_marginals.py"]
    try:
        FPM.main()
        FIG_STATUS.update({n: "rebuilt" for n in pm})
    except SystemExit as e:
        FIG_STATUS.update({n: f"NOT rebuilt ({e})" for n in pm})
    import fig_paper_style as FPS
    ps = [n for n, _, scr in THESIS_FIGURES if scr == "fig_paper_style.py"]
    try:
        FPS.main()
        FIG_STATUS.update({n: "rebuilt" for n in ps})
    except SystemExit as e:
        FIG_STATUS.update({n: f"NOT rebuilt ({e})" for n in ps})
    names = [n for n, *_ in THESIS_FIGURES]
    TS.contact_sheet(names)
    TS.latex_preview(names)
    for n, where, _ in THESIS_FIGURES:
        size = TS.SAVED.get(n)
        print(f"  {where:28s} {n:26s} {FIG_STATUS.get(n, '?'):10s} "
              + (f"{size[0]:.2f} x {size[1]:.2f} in" if size else ""))
    return FIG_STATUS


def figures_section() -> str:
    """summary.md: every thesis figure with its caption (from figures/<name>.caption.tex)."""
    L = []
    for n, where, scr in THESIS_FIGURES:
        cap = (FIG / f"{n}.caption.tex")
        text = TS._caption_plain(cap.read_text()) if cap.exists() else "(no caption file)"
        size = TS.SAVED.get(n)
        L += [f"### {where}: `{n}`", "",
              f"![{n}](figures/{n}.png)", "",
              text, "",
              f"_Script: analysis/summary/{scr}; files: figures/{n}.pdf / .png / .caption.tex"
              + (f"; {size[0]:.2f} × {size[1]:.2f} in" if size else "")
              + (f"; {FIG_STATUS[n]}" if n in FIG_STATUS and FIG_STATUS[n] != "rebuilt" else "") + "._", ""]
    return "\n".join(L)


def figures_only():
    """Every thesis figure, from the same records as the tables; no table, csv or summary.md is
    rewritten."""
    FIG.mkdir(parents=True, exist_ok=True)
    bv = b_values()
    tdv = b_tabdiff_values()
    st = build_figures(bv, tdv)
    bad = [n for n, v in st.items() if v != "rebuilt"]
    print(f"wrote {rel(FIG)}/: {len(st) - len(bad)} figures rebuilt" + (f"; NOT rebuilt: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


def main():
    if "--figures-only" in sys.argv[1:]:
        sys.exit(figures_only())
    for d in (TAB, CSV, FIG):
        d.mkdir(parents=True, exist_ok=True)
    global DEV
    DEV = set(pd.read_csv(ct.BEST_CSV).dataset)
    _file_info(ct.BEST_CSV)
    cells = load_a_cells()
    vals = gnn_a_values(cells)
    xcheck = vals.pop("_xcheck")
    tv = tabddpm_values()
    for ds in A_DS:
        _file_info(REPO / "data" / ds / "info.json")
    table_a0(tv)
    for r in RULES:
        table_a_rule(vals, r)
    table_a4(tv)
    a5 = table_a5(vals, tv)
    table_a6(vals, tv)
    bv = b_values()
    tdv = b_tabdiff_values()
    table_b(bv, tdv)
    b6 = table_b6(bv, tdv)
    table_b7(bv)
    table_c(vals, tv, bv, tdv)
    table_appendix_gatd()
    allv = pd.DataFrame(ROWS)
    allv.to_csv(CSV / "all_values.csv", index=False)
    pd.DataFrame(list(_PROV.values())).sort_values("path").to_csv(OUT / "provenance.csv", index=False)
    build_figures(bv, tdv)
    files = (["analysis/summary/build_summary.py", "analysis/summary/summary.md", "analysis/summary/provenance.csv"]
             + [f"analysis/summary/tables/{f}" for f in TEX_FILES]
             + sorted(f"analysis/summary/csv/{p.name}" for p in CSV.glob("*.csv"))
             + [f"analysis/summary/figures/{n}.{e}" for n, *_ in THESIS_FIGURES for e in ("pdf", "png", "caption.tex")]
             + ["analysis/summary/figures/_contact_sheet.pdf", "analysis/summary/figures/_preview.tex",
                "analysis/summary/thesis_style.py", "analysis/summary/route_a_figs.py"]
             + [f"analysis/summary/{scr}" for scr in sorted({s_ for *_, s_ in THESIS_FIGURES} - {"build_summary.py"})]
             + ["tests/test_selection_differentiability.py"])
    write_summary(xcheck, a5, b6, files, bv)
    print(f"wrote {rel(OUT / 'summary.md')}: {len(TEX_FILES)} tables, {len(allv)} values, "
          f"{len(_PROV)} source files; A cross-check max |diff| {max(xcheck):.1e}")


if __name__ == "__main__":
    main()

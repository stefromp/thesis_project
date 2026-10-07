#!/usr/bin/env python3
"""Paper-style results tables and figures (TabDDPM / TabDiff layout) for the thesis.

    python3 analysis/paper_tables/paper_tables.py

A presentation layer only: it reads existing results and writes everything
under analysis/paper_tables/. Nothing is trained or sampled, nothing is
imputed, and the other analyses (topk_comparison, dependency_fidelity, ...)
are imported, never modified.

Sources
-------
  Real, TabDDPM   exp/<ds>/ddpm_cb_best/eval_catboost.json (authors' release):
                  "real" and "synthetic" blocks, test split.
  CTGAN, TVAE,    Kotelnikov et al., arXiv:2209.15421v2, Table 5 (tuned CatBoost),
  CTABGAN(+),     transcribed below ("reported").
  SMOTE
  GNN-fixed /     the run cells chosen by analysis/topk_comparison/compare_topk.py
  -Jenks /        (discover()): *_catfix trees where every rule is complete,
  -entmax         otherwise ablation_fixed_seed0 / ablation / adaptive trees.
                  Only cells with 5 complete generation seeds are shown.
  Shape, Trend    analysis/dependency_fidelity/per_dataset.csv + floor.csv
                  (shared real bins, all column pairs), checked against the
                  same run cells.
  DCR, MIA        per-generation-seed dcr_baseline / mia_auc of those cells;
                  the holdout reference DCR (real test -> real train) is computed
                  here with the same recipe (scripts/eval_tabddpm_protocol.py).

The budget-matched runs (*_tddpm_setup* trees) are still running and are never
read: every path this script opens is checked against FORBIDDEN, and the run
aborts if one matches.

Environment: PAPER_TABLES_EXCLUDE (default "higgs-small,miniboone") lists
datasets whose GNN runs have not been delivered; their GNN cells are "—".
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# compare_topk reads COMPARE_TOPK_EXCLUDE at import time; drive it from ours.
EXCLUDE = os.environ.get("PAPER_TABLES_EXCLUDE", "higgs-small,miniboone")
os.environ["COMPARE_TOPK_EXCLUDE"] = EXCLUDE
os.environ.pop("COMPARE_TOPK_VARIANT", None)        # always the current (auto) selection
os.environ.pop("COMPARE_TOPK_RESULTS_ROOT", None)

sys.path.insert(0, str(REPO / "analysis" / "topk_comparison"))
sys.path.insert(0, str(REPO / "analysis" / "dependency_fidelity"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import compare_topk as ct  # noqa: E402
import paper_comparison as pc  # noqa: E402

OUT = HERE
TAB = OUT / "tables"
FIG = OUT / "figures"
DEPFID = REPO / "analysis" / "dependency_fidelity"
FORBIDDEN = "tddpm_setup"

# ---------------------------------------------------------------------------
# Datasets, methods
# ---------------------------------------------------------------------------
DATASETS = [("abalone", "AB"), ("adult", "AD"), ("buddy", "BU"), ("california", "CA"),
            ("cardio", "CAR"), ("churn2", "CH"), ("default", "DE"), ("diabetes", "DI"),
            ("fb-comments", "FB"), ("gesture", "GE"), ("higgs-small", "HI"), ("house", "HO"),
            ("insurance", "IN"), ("king", "KI"), ("miniboone", "MI"), ("wilt", "WI")]
DS = [d for d, _ in DATASETS]
AB = dict(DATASETS)
# Local default split / column typing differs from the authors' (14 num + 9 cat
# vs 20 num + 3 cat): no value from the paper or the authors' release is
# like-for-like with ours there. Not ranked, not highlighted.
NOT_LIKE_FOR_LIKE = {"default"}
HIGHLIGHT_NLFL = False

BASELINES = ["CTGAN", "TVAE", "CTABGAN", "CTABGAN+", "SMOTE"]
OURS = ["fixed", "jenks", "entmax15"]
OURS_LABEL = {"fixed": "GNN-fixed", "jenks": "GNN-Jenks", "entmax15": "GNN-entmax"}
SYNTH = BASELINES + ["TabDDPM"] + OURS
LABEL = {**{b: b for b in BASELINES}, "TabDDPM": "TabDDPM", **OURS_LABEL, "Real": "Real"}

# Kotelnikov et al., "TabDDPM: Modelling Tabular Data with Diffusion Models",
# arXiv:2209.15421v2, Table 5 (tuned CatBoost), dataset order of DATASETS.
# "–" in the paper (method not run) -> None.
_T5 = {
    "CTGAN": "0.420±.004 0.789±.001 0.867±.003 0.686±.003 0.730±.001 0.723±.006 0.699±.002 0.459±.096 "
             "0.443±.005 0.333±.013 0.575±.006 0.433±.005 0.745±.009 0.772±.005 0.783±.005 0.749±.015",
    "TVAE": "0.433±.008 0.781±.002 0.864±.005 0.752±.001 0.717±.001 0.732±.006 0.656±.007 0.714±.039 "
            "0.685±.003 0.434±.006 0.638±.003 0.493±.006 0.784±.010 0.824±.003 0.912±.001 0.501±.012",
    "CTABGAN": "– 0.783±.002 0.855±.005 – 0.717±.001 0.688±.006 0.644±.011 0.731±.022 "
               "– 0.392±.006 0.575±.004 – – – 0.889±.002 0.906±.019",
    "CTABGAN+": "0.467±.004 0.772±.003 0.884±.005 0.525±.004 0.733±.001 0.702±.012 0.686±.004 0.734±.020 "
                "0.509±.011 0.406±.009 0.664±.002 0.504±.005 0.797±.005 0.444±.014 0.892±.002 0.798±.021",
    "SMOTE": "0.549±.005 0.791±.002 0.891±.003 0.840±.001 0.732±.001 0.743±.005 0.693±.003 0.683±.037 "
             "0.803±.002 0.658±.007 0.722±.001 0.662±.004 0.812±.002 0.842±.004 0.932±.001 0.913±.007",
}


def _parse_t5() -> dict:
    out = {}
    for m, s in _T5.items():
        toks = s.split()
        assert len(toks) == len(DS), m
        for ds, t in zip(DS, toks):
            if t == "–":
                out[(m, ds)] = None
            else:
                a, b = t.split("±")
                out[(m, ds)] = (float(a), float(b))
    return out


T5 = _parse_t5()

# Colours (dataviz reference palette; same rule colours as compare_topk figures).
C_OURS = ct.C_RULE
C_TDDPM, C_BASE = "#52514e", "#b5b3ad"
C_TEXT, C_TEXT2, C_GRID = ct.C_TEXT, ct.C_TEXT2, ct.C_GRID

# ---------------------------------------------------------------------------
# Provenance guard
# ---------------------------------------------------------------------------
_READ: list[dict] = []


def track(path: Path | str, what: str) -> Path:
    p = Path(path)
    if FORBIDDEN in str(p):
        raise RuntimeError(f"refusing to read a budget-matched ({FORBIDDEN}) result: {p}")
    _READ.append({"what": what, "path": ct.rel(p)})
    return p


def task_of(ds: str) -> str:
    return json.load(open(track(REPO / "data" / ds / "info.json", "data info")))["task_type"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_authors() -> pd.DataFrame:
    rows = []
    for ds in DS:
        p = track(REPO / "exp" / ds / "ddpm_cb_best" / "eval_catboost.json", "authors TabDDPM/Real")
        d = json.load(open(p))
        key = "r2" if task_of(ds) == "regression" else "f1"
        for method, block in (("TabDDPM", "synthetic"), ("Real", "real")):
            t = d[block]["test"]
            rows.append({"dataset": ds, "method": method, "mean": t[f"{key}-mean"], "std": t[f"{key}-std"],
                         "n_runs": int(t[f"{key}-count"]), "source": ct.rel(p)})
    return pd.DataFrame(rows)


def load_ours() -> tuple[pd.DataFrame, dict]:
    """compare_topk's cell selection; utility = mean/std over all 50 runs."""
    _, cells = ct.discover()
    rows = []
    for ds, info in cells.items():
        for r in OURS:
            c = info["rules"][r]
            if c["source"]:
                track(c["source"].split(" [")[0], f"GNN run ({r})")
            ok = c["status"] == "complete" and c["per_gen"] is not None
            if ok and info["cat_affected"]:
                print(f"WARNING {ds}/{r}: run trained with cat_input=log on a categorical dataset; shown as —")
                ok = False
            row = {"dataset": ds, "method": r, "status": c["status"], "source": c["source"],
                   "config": c["config"], "n_gen": c["n_gen"]}
            if ok:
                metric = "r2" if info["task"] == "regression" else "macro avg/f1-score"
                v = pc.raw_runs(REPO / c["source"], metric)
                assert len(v) == ct.EXPECTED_GEN * ct.EXPECTED_CLF, (ds, r, len(v))
                row.update(mean=v.mean(), std=v.std(), n_runs=len(v))   # ddof=0, as the authors
                for key in ("dcr_baseline", "mia_auc"):
                    rv = ct.rule_values(c, key)
                    row[f"{key}_mean"] = rv["mean"] if rv else np.nan
                    row[f"{key}_sd"] = rv["sd"] if rv else np.nan
            rows.append(row)
    return pd.DataFrame(rows), cells


def load_fidelity(ours: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per = pd.read_csv(track(DEPFID / "per_dataset.csv", "Shape/Trend per dataset"))
    floor = pd.read_csv(track(DEPFID / "floor.csv", "Shape/Trend real-test reference"))
    cov = pd.read_csv(track(DEPFID / "coverage.csv", "Shape/Trend run trees"))
    # The fidelity numbers must come from the same run cells as everything else.
    ok = ours[ours["mean"].notna()]
    for _, r in ok.iterrows():
        c = cov[(cov.dataset == r.dataset) & (cov.rule == r.method)]
        tree = r.source.rsplit("/", 1)[0]
        if c.empty or c.tree.iloc[0] != tree or c.combo.iloc[0] != r.source.rsplit("/", 1)[1]:
            raise RuntimeError(f"dependency_fidelity is stale for {r.dataset}/{r.method}: rerun "
                               f"analysis/dependency_fidelity/dependency_fidelity.py")
    per = per[(per.status == "complete") & (per.n_gen_seeds >= ct.EXPECTED_GEN)
              & per.metric.isin(["column_shapes", "column_pair_trends"])]
    per = per.merge(ok[["dataset", "method"]], left_on=["dataset", "rule"], right_on=["dataset", "method"])
    return per[["dataset", "rule", "metric", "mean", "sd", "n_gen_seeds"]], floor


def holdout_dcr(datasets: list[str], ours: pd.DataFrame) -> pd.DataFrame:
    """DCR of the real test split against the real train split, with the exact
    recipe of dcr_baseline (median NN-L2, label included, minmax num +
    onehot/sqrt(2) cat, encoder fit on real train). Self-check: the same code
    reproduces one stored synthetic dcr_baseline."""
    from cat_input_local_check import _setup   # stubs the heavy optional imports of lib
    _setup()
    import eval_tabddpm_protocol as ep

    def dcr(real_dir: Path, query_dir: Path, query_split: str, task: str) -> float:
        rn, rc, ry = ep._read(str(track(real_dir, "real data")), "train")
        qn, qc, qy = ep._read(str(track(query_dir, "DCR query rows")), query_split)
        rn, rc = ep._append_label(rn, rc, ry, task)
        qn, qc = ep._append_label(qn, qc, qy, task)
        Xr, Xq = ep._encode(rn, rc, (rn, rc), (qn, qc))
        return ep._dcr_from_encoded(Xr, Xq, aggregation="median")

    rows = []
    for ds in datasets:
        rows.append({"dataset": ds, "dcr_holdout": dcr(REPO / "data" / ds, REPO / "data" / ds, "test", task_of(ds))})
    # self-check on the first available cell, gen seed 0
    chk = ours[ours["mean"].notna()].iloc[0]
    run = REPO / chk.source
    stored = pd.concat([pd.read_csv(f) for f in sorted(run.glob("per_seed_scores*.csv"))])
    stored = stored[(stored.metric == "dcr_baseline") & (stored.gen_seed == 0)].value.iloc[-1]
    again = dcr(REPO / "data" / chk.dataset, run / "gen_seed_0", "train", task_of(chk.dataset))
    if not np.isclose(stored, again, rtol=1e-6, atol=1e-9):
        raise RuntimeError(f"DCR recipe mismatch on {chk.source}: stored {stored} vs recomputed {again}")
    out = pd.DataFrame(rows)
    out.attrs["selfcheck"] = f"{chk.source}/gen_seed_0: stored {stored:.6f}, recomputed {again:.6f}"
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def sd_str(s: float, nd: int) -> str:
    t = f"{s:.{nd}f}"
    return t[1:] if t.startswith("0.") else t


def cell(mean, sd, nd, mark="", tex=False):
    """mark: '' | 'b' (bold) | 'u' (underline)."""
    if mean is None or not np.isfinite(mean):
        return "---" if tex else "—"
    m = f"{mean:.{nd}f}"
    s = sd_str(sd, nd) if sd is not None and np.isfinite(sd) else None
    if tex:
        m = {"b": r"\textbf{%s}", "u": r"\underline{%s}"}.get(mark, "%s") % m
        return m + (r"{\scriptsize$\pm$%s}" % s if s else "")
    m = {"b": "**%s**", "u": "<u>%s</u>"}.get(mark, "%s") % m
    return m + (f" ±{s}" if s else "")


def marks(vals: dict, higher_better: bool, nd: int) -> dict:
    """Bold best / underline second best on the displayed (rounded) values."""
    v = {k: round(x, nd) for k, x in vals.items() if x is not None and np.isfinite(x)}
    if not v:
        return {}
    sgn = 1 if higher_better else -1
    order = sorted(set(v.values()), key=lambda x: -sgn * x)
    out = {k: "b" for k, x in v.items() if x == order[0]}
    if len(order) > 1:
        out.update({k: "u" for k, x in v.items() if x == order[1]})
    return out


def avg_ranks(M: pd.DataFrame, higher_better: bool, nd: int) -> tuple[pd.Series, pd.Series, list]:
    """M: datasets x methods (complete rows only). Ranks on rounded values,
    ties share the average rank. Returns (avg rank, #datasets best, datasets)."""
    M = M.dropna()
    if M.empty:
        return pd.Series(np.nan, index=M.columns), pd.Series(0, index=M.columns), []
    R = M.round(nd).apply(lambda row: pd.Series(stats.rankdata(-row if higher_better else row), index=M.columns),
                          axis=1)
    best = (R.eq(R.min(axis=1), axis=0)).sum()
    return R.mean(), best, list(M.index)


def tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")


def md_table(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def tex_table(colspec: str, head: list[str], blocks: list[list[list[str]]], caption: str, label: str,
              subhead: list[str] | None = None) -> str:
    L = [r"% requires \usepackage{booktabs,graphicx}", r"\begin{table}[t]", r"\centering",
         r"\caption{%s}" % caption, r"\label{%s}" % label, r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{%s}" % colspec, r"\toprule", " & ".join(head) + r" \\"]
    if subhead:
        L.append(" & ".join(subhead) + r" \\")
    L.append(r"\midrule")
    for i, b in enumerate(blocks):
        L += [" & ".join(r) + r" \\" for r in b]
        if i < len(blocks) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L) + "\n"


def ds_head(ds: str, tex: bool) -> str:
    unit = "R2" if TASK[ds] == "regression" else "F1"
    if tex:
        unit = "$R^2$" if unit == "R2" else "$F_1$"
        return AB[ds] + (r"$^\ddagger$" if ds in NOT_LIKE_FOR_LIKE else "") + r" \tiny(" + unit + ")"
    return AB[ds] + (" ‡" if ds in NOT_LIKE_FOR_LIKE else "") + f" ({'R²' if unit == 'R2' else 'F1'})"


# ---------------------------------------------------------------------------
# Table 1: ML efficiency
# ---------------------------------------------------------------------------
def mle_long(auth: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, ds), v in T5.items():
        rows.append({"dataset": ds, "method": m, "mean": v[0] if v else np.nan, "std": v[1] if v else np.nan,
                     "n_runs": 50 if v else 0, "source": "reported: arXiv:2209.15421v2 Table 5"})
    rows += auth.to_dict("records")
    o = ours.copy()
    o["source"] = o.source.where(o["mean"].notna(), "")
    rows += o[["dataset", "method", "mean", "std", "n_runs", "source"]].to_dict("records")
    df = pd.DataFrame(rows)
    df["n_runs"] = df.n_runs.fillna(0).astype(int)
    return df


def mle_ranks(W: pd.DataFrame, methods: list[str]) -> tuple[pd.Series, pd.Series, list]:
    M = W.loc[[d for d in DS if d not in NOT_LIKE_FOR_LIKE], methods]
    return avg_ranks(M, True, 3)


def table_mle(mle: pd.DataFrame) -> tuple[str, str, dict]:
    W = mle.pivot(index="dataset", columns="method", values="mean").reindex(DS)
    S = mle.pivot(index="dataset", columns="method", values="std").reindex(DS)
    rank_all, best_all, ds_all = mle_ranks(W, SYNTH)
    no_ctab = [m for m in SYNTH if m != "CTABGAN"]
    rank_nc, best_nc, ds_nc = mle_ranks(W, no_ctab)
    mk = {ds: ({} if (ds in NOT_LIKE_FOR_LIKE and not HIGHLIGHT_NLFL)
               else marks({m: W.loc[ds, m] for m in SYNTH}, True, 3)) for ds in DS}

    def row(m, tex):
        lab = LABEL[m]
        if tex:
            lab = tex_escape(lab) + (r"$^\dagger$" if m in BASELINES else "")
            if m == "Real":
                lab = r"\textit{Real}"
        else:
            lab = lab + (" †" if m in BASELINES else "")
            lab = "*Real*" if m == "Real" else lab
        cells = [cell(W.loc[ds, m], S.loc[ds, m], 3, mk[ds].get(m, ""), tex) for ds in DS]
        rk = ["" if m == "Real" else (f"{rank_all[m]:.2f}" if m in rank_all and np.isfinite(rank_all[m]) else
                                      ("---" if tex else "—")),
              "" if m == "Real" else (f"{rank_nc[m]:.2f}" if m in rank_nc and np.isfinite(rank_nc[m]) else
                                      ("---" if tex else "—"))]
        return [lab] + cells + rk

    groups = [["Real"], BASELINES, ["TabDDPM"], OURS]
    rk_h = [f"Rank (n={len(ds_all)})", f"Rank w/o CTABGAN (n={len(ds_nc)})"]
    md = md_table(["Method"] + [ds_head(d, False) for d in DS] + rk_h,
                  [row(m, False) for g in groups for m in g])
    note = (f"Mean ± std over 5 generation seeds × 10 CatBoost seeds (50 runs; std with ddof=0) of macro-F1 "
            f"(classification) or R² (regression) on the test split, CatBoost tuned per dataset on real data. "
            f"**Bold** = best and <u>underline</u> = second best synthetic method per dataset (on the values as "
            f"displayed; Real is not ranked). † reported: values from Kotelnikov et al. (TabDDPM, "
            f"arXiv:2209.15421v2, Table 5), same protocol. Real and TabDDPM: the authors' released evaluation "
            f"files (exp/<ds>/ddpm_cb_best/eval_catboost.json). — = not available. "
            f"Rank = average rank (1 = best) over the {len(ds_all)} datasets where every synthetic method has a "
            f"value ({', '.join(AB[d] for d in ds_all)}); the second rank column drops CTABGAN, which has no "
            f"regression results, and covers {len(ds_nc)} datasets ({', '.join(AB[d] for d in ds_nc)}). "
            f"‡ DE: our local default split and column typing differ from the authors', so the reported, Real "
            f"and TabDDPM values on DE are not like-for-like with ours; DE is not highlighted and is excluded "
            f"from both rank columns.")
    cap = (r"Machine learning efficiency w.r.t.\ tuned CatBoost: macro-$F_1$ (classification) or $R^2$ "
           r"(regression), mean$\pm$std over 50 runs (5 generation $\times$ 10 CatBoost seeds). "
           r"\textbf{Bold}: best, \underline{underline}: second-best synthetic method (Real not ranked). "
           r"$^\dagger$Reported in the TabDDPM paper~\cite{kotelnikov2023tabddpm}, Table~5 (same protocol). Real and TabDDPM from "
           r"the authors' released evaluation files. Rank: average rank over the %d datasets where all synthetic "
           r"methods have a value; ``w/o CTABGAN'': over %d datasets, CTABGAN excluded. $^\ddagger$Our local "
           r"\textit{default} split and column typing differ from the authors', so values on DE are not "
           r"like-for-like with ours; DE is not highlighted and is excluded from the ranks.") % (len(ds_all), len(ds_nc))
    tex = tex_table("l" + "c" * len(DS) + "cc",
                    ["Method"] + [ds_head(d, True) for d in DS] + [r"Rank$\downarrow$", r"Rank$\downarrow$"],
                    [[row(m, True) for m in g] for g in groups], cap, "tab:mle",
                    subhead=[""] * (len(DS) + 1) + [r"\tiny (all)", r"\tiny (w/o CTABGAN)"])
    info = {"rank_all": rank_all, "best_all": best_all, "ds_all": ds_all,
            "rank_nc": rank_nc, "best_nc": best_nc, "ds_nc": ds_nc, "W": W, "S": S}
    return md + "\n\n" + note + "\n", tex, info


# ---------------------------------------------------------------------------
# Table 2: Shape / Trend (ours only)
# ---------------------------------------------------------------------------
FID = (("column_shapes", "Column Shapes"), ("column_pair_trends", "Column Pair Trends"))


def table_fidelity(fid: pd.DataFrame, floor: pd.DataFrame) -> tuple[str, str, dict]:
    md_parts, tex_blocks, info = [], [], {}
    fl = floor.set_index("dataset")
    for key, name in FID:
        f = fid[fid.metric == key]
        E = f.pivot(index="dataset", columns="rule", values="mean").reindex(index=DS, columns=OURS)
        E = 100 * (1 - E)
        SD = 100 * f.pivot(index="dataset", columns="rule", values="sd").reindex(index=DS, columns=OURS)
        rank, best, used = avg_ranks(E, False, 2)
        info[key] = {"rank": rank, "best": best, "ds": used, "E": E}
        mk = {ds: marks({r: E.loc[ds, r] for r in OURS}, False, 2) for ds in DS}

        def ref(tex):
            vals = [cell(100 * (1 - fl.loc[d, key]) if d in fl.index and d in used else np.nan, None, 2, "", tex)
                    for d in DS]
            return [r"\textit{Real test}" if tex else "*Real test*"] + vals + [""]

        def row(r, tex):
            return ([OURS_LABEL[r]] + [cell(E.loc[d, r], SD.loc[d, r], 2, mk[d].get(r, ""), tex) for d in DS]
                    + [f"{rank[r]:.2f}" if np.isfinite(rank[r]) else "—"])

        head = ["Method"] + [AB[d] for d in DS] + [f"Rank (n={len(used)})"]
        md_parts.append(f"**{name} error (%) ↓**\n\n" + md_table(head, [ref(False)] + [row(r, False) for r in OURS]))
        tex_blocks.append([[r"\multicolumn{%d}{l}{\textit{%s error (\%%)} $\downarrow$} " % (len(DS) + 2, name)]]
                          + [ref(True)] + [row(r, True) for r in OURS])
    n = len(info["column_shapes"]["ds"])
    note = ("Error = 100 × (1 − score), lower is better; mean ± sd over 5 generation seeds (sd with ddof=1). "
            "Column Shapes: SDMetrics per-column similarity (KS complement for numerical, TV complement for "
            "categorical columns) averaged over columns. Column Pair Trends: correlation / contingency similarity "
            "averaged over ALL column pairs, each pair scored on bins shared between real and synthetic data "
            "(not the SDMetrics QualityReport aggregate). Each synthetic sample (and the Real-test reference) is "
            "compared with the real train split on the same number of rows. *Real test* = the real test split "
            "scored against the real train split (sampling-noise reference). **Bold** best, <u>underline</u> "
            f"second best among the three GNN variants; Rank = average over the {n} datasets with all three. "
            "**These values are computed with our own implementation and splits and are not comparable to the "
            "Shape/Trend numbers reported in the TabDiff paper.**")
    cap = (r"Fidelity of our three GNN variants: Column Shapes and Column Pair Trends error rates (\%%, "
           r"$100\times(1-\mathrm{score})$, lower is better), mean$\pm$sd over 5 generation seeds. Column Pair Trends "
           r"is averaged over all column pairs with bins shared between real and synthetic data. \textit{Real test}: "
           r"real test split vs.\ real train split (sampling-noise reference). \textbf{Bold}: best, "
           r"\underline{underline}: second best; Rank: average over the %d datasets with all three variants. "
           r"Computed with our own implementation and splits; \emph{not comparable} to the Shape/Trend values "
           r"reported by TabDiff.") % n
    head_tex = ["Method"] + [AB[d] for d in DS] + [r"Rank$\downarrow$"]
    tex = tex_table("l" + "c" * len(DS) + "c", head_tex,
                    tex_blocks, cap, "tab:fidelity")
    return "\n\n".join(md_parts) + "\n\n" + note + "\n", tex, info


# ---------------------------------------------------------------------------
# Table 3: privacy
# ---------------------------------------------------------------------------
# TabDDPM's DCR = np.median(resample_privacy.privacy_metrics(...)) written by
# calc_privacy to exp/<ds>/ddpm_cb_best/privacy.json: one generated sample
# (sampling seed 0). dcr_recipe_check.py runs that function verbatim on our
# samples and reproduces our dcr_baseline to ~1e-12, so the two are the same
# statistic. privacies.json is NOT used: its first entry is the same median on
# whatever sample was on disk, the others follow DCR-filtered resampling.
# privacy_metrics subsamples the real train set to 50k rows when it is larger
# (resample_privacy.py:56-59); our dcr_baseline never does.
REF_CAP = 50_000
PRIV_METHODS = ["TabDDPM"] + OURS


def load_tabddpm_dcr() -> pd.Series:
    out = {}
    for ds in DS:
        p = REPO / "exp" / ds / "ddpm_cb_best" / "privacy.json"
        if p.exists():
            out[ds] = float(json.load(open(track(p, "authors TabDDPM DCR")))["privacy"])
    return pd.Series(out, dtype=float)


def n_real_train(ds: str) -> int:
    return len(np.load(track(REPO / "data" / ds / "y_train.npy", "real train size"), allow_pickle=True))


def table_privacy(ours: pd.DataFrame, hold: pd.DataFrame, tdcr: pd.Series) -> tuple[str, str, dict, pd.DataFrame]:
    o = ours[ours["mean"].notna()].set_index(["dataset", "method"])
    h = hold.set_index("dataset").dcr_holdout
    ref_sub = {ds for ds in DS if n_real_train(ds) > REF_CAP}
    rows = []
    for (ds, r), x in o.iterrows():
        rows.append({"dataset": ds, "method": r, "dcr": x.dcr_baseline_mean, "dcr_sd": x.dcr_baseline_sd,
                     "n_samples": ct.EXPECTED_GEN, "dcr_holdout": h.get(ds, np.nan),
                     "ratio": x.dcr_baseline_mean / h.get(ds, np.nan), "ratio_sd": x.dcr_baseline_sd / h.get(ds, np.nan),
                     "mia_auc": x.mia_auc_mean, "mia_auc_sd": x.mia_auc_sd})
    for ds, v in tdcr.items():
        comparable = ds not in NOT_LIKE_FOR_LIKE and ds not in ref_sub
        hv = h.get(ds, np.nan) if comparable else np.nan
        rows.append({"dataset": ds, "method": "TabDDPM", "dcr": v, "dcr_sd": np.nan, "n_samples": 1,
                     "dcr_holdout": hv, "ratio": v / hv, "ratio_sd": np.nan, "mia_auc": np.nan,
                     "mia_auc_sd": np.nan, "real_ref_subsampled": ds in ref_sub,
                     "not_like_for_like": ds in NOT_LIKE_FOR_LIKE})
    P = pd.DataFrame(rows)
    piv = lambda col, cols: P.pivot(index="dataset", columns="method", values=col).reindex(index=DS, columns=cols)  # noqa
    D, Dsd, R, Rsd = (piv(c, PRIV_METHODS) for c in ("dcr", "dcr_sd", "ratio", "ratio_sd"))
    A, Asd = piv("mia_auc", OURS), piv("mia_auc_sd", OURS)
    # rank: TabDDPM + ours, where all four exist; DE and reference-subsampled datasets excluded
    rank, best, used = avg_ranks(D.drop(index=list(NOT_LIKE_FOR_LIKE | ref_sub), errors="ignore"), True, 4)
    mia_rank, mia_best, mia_used = avg_ranks((A - 0.5).abs(), False, 3)
    info = {"rank": rank, "best": best, "ds": used, "mia_rank": mia_rank, "mia_ds": mia_used, "ref_sub": ref_sub}

    def tab_mark(d, tex):
        m = ""
        if d in NOT_LIKE_FOR_LIKE:
            m += r"$^\ddagger$" if tex else " ‡"
        if d in ref_sub:
            m += r"$^\diamond$" if tex else " ◇"
        return m

    def rows_for(tex):
        dash = lambda v, nd: cell(v, None, nd, "", tex)  # noqa
        lab_t = (r"TabDDPM$^\S$" if tex else "TabDDPM §")
        B1 = [[r"\textit{Holdout}" if tex else "*Holdout*"] + [dash(h.get(d, np.nan), 4) for d in DS]]
        B1 += [[lab_t] + [dash(D.loc[d, "TabDDPM"], 4) + (tab_mark(d, tex) if np.isfinite(D.loc[d, "TabDDPM"]) else "")
                          for d in DS]]
        B1 += [[OURS_LABEL[r]] + [cell(D.loc[d, r], Dsd.loc[d, r], 4, "", tex) for d in DS] for r in OURS]
        B2 = [[lab_t] + [dash(R.loc[d, "TabDDPM"], 3) for d in DS]]
        B2 += [[OURS_LABEL[r]] + [cell(R.loc[d, r], Rsd.loc[d, r], 3, "", tex) for d in DS] for r in OURS]
        B3 = [[OURS_LABEL[r]] + [cell(A.loc[d, r], Asd.loc[d, r], 3, "", tex) for d in DS] for r in OURS]
        return B1, B2, B3

    head = ["Method"] + [AB[d] for d in DS]
    B1, B2, B3 = rows_for(False)
    md = ("**DCR (median nearest-neighbour L2 to real train) ↑**\n\n" + md_table(head, B1)
          + "\n\n**DCR / holdout DCR**\n\n" + md_table(head, B2)
          + "\n\n**MIA AUC (→ 0.5)**\n\n" + md_table(head, B3))
    sub_ab = ", ".join(AB[d] for d in DS if d in ref_sub)
    note = ("DCR: median, over synthetic rows, of the L2 distance to the closest real training row (numerical "
            "columns min-max scaled and categorical columns one-hot / √2, both fitted on real train; target "
            "included). GNN rows: mean ± sd over 5 generation seeds. § TabDDPM: the authors' "
            "exp/<ds>/ddpm_cb_best/privacy.json, the same statistic computed by the same code "
            "(scripts/resample_privacy.py `privacy_metrics`, median), on one generated sample, so no sd; that code "
            "reproduces our stored GNN DCR to ~1e-12 when run on our samples (`DCR_RECIPE_CHECK.md`). It subsamples "
            "the synthetic rows to 50,000 when more were generated (≤0.4% effect on the median, measured on our "
            "samples) and the real train rows to 50,000 when more exist: ◇ = real train set subsampled "
            f"({sub_ab}), not comparable to an all-rows DCR. ‡ DE: computed on the authors' version of the dataset, "
            "not like-for-like. *Holdout*: the same statistic for the real test split, i.e. the distance a fresh "
            "sample from the data distribution has to the training set; DCR / holdout ≈ 1 means synthetic rows "
            "are as far from the training rows as unseen real rows, < 1 closer (not computed where ‡ or ◇). "
            "MIA AUC: distance-based membership-inference attack (train vs test rows scored by proximity to the "
            "synthetic set), 0.5 = no leakage; not available for TabDDPM. No value is highlighted: a larger DCR "
            "is not unconditionally better (it can also reflect lower fidelity).")
    cap = (r"Privacy. DCR: median L2 distance from each synthetic row to its closest real training row "
           r"(min-max numerical, one-hot$/\sqrt2$ categorical, target included); GNN rows: mean$\pm$sd over 5 "
           r"generation seeds. $^\S$TabDDPM: the authors' released DCR, computed with the same code on one "
           r"generated sample (no sd); it subsamples synthetic rows to 50k (${\le}0.4\%%$ effect) and real "
           r"training rows to 50k where larger ($^\diamond$, %s; not comparable). $^\ddagger$Authors' version of "
           r"the \textit{default} dataset, not like-for-like. \textit{Holdout}: the same statistic for the real "
           r"test split; DCR/holdout $\approx1$: synthetic rows are as far from the training data as unseen real "
           r"rows. MIA AUC: distance-based membership inference (0.5 = no leakage), not available for TabDDPM.") % sub_ab
    T1, T2, T3 = rows_for(True)
    sec = lambda t: [[r"\multicolumn{%d}{l}{\textit{%s}}" % (len(DS) + 1, t)]]  # noqa
    tex = tex_table("l" + "c" * len(DS), head,
                    [sec(r"DCR $\uparrow$") + T1, sec("DCR / holdout DCR") + T2, sec(r"MIA AUC ($\to 0.5$)") + T3],
                    cap, "tab:privacy")
    return md + "\n\n" + note + "\n", tex, info, P


# ---------------------------------------------------------------------------
# Table 4: summary + Wilcoxon vs TabDDPM
# ---------------------------------------------------------------------------
def wilcoxon_vs_tabddpm(W: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in OURS:
        d = (W[r] - W["TabDDPM"]).drop(index=list(NOT_LIKE_FOR_LIKE), errors="ignore").dropna()
        res = stats.wilcoxon(d.to_numpy()) if len(d) and np.any(d != 0) else None
        rows.append({"method": r, "n": len(d), "datasets": ";".join(AB[x] for x in d.index),
                     "median_delta": d.median(), "mean_delta": d.mean(),
                     "n_above": int((d > 0).sum()), "n_below": int((d < 0).sum()),
                     "W": res.statistic if res else np.nan, "p": res.pvalue if res else np.nan})
    out = pd.DataFrame(rows)
    out["p_holm"] = ct.holm(out.p.tolist())
    return out


def table_summary(mle_i: dict, fid_i: dict, prv_i: dict, wil: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    rows = []
    wl = wil.set_index("method")
    for m in SYNTH:
        rec = {"method": m,
               "mle_rank": mle_i["rank_all"].get(m, np.nan), "mle_best": mle_i["best_all"].get(m, np.nan),
               "mle_rank_noctab": mle_i["rank_nc"].get(m, np.nan) if m != "CTABGAN" else np.nan,
               "mle_best_noctab": mle_i["best_nc"].get(m, np.nan) if m != "CTABGAN" else np.nan}
        for key, _ in FID:
            rec[f"{key}_rank"] = fid_i[key]["rank"].get(m, np.nan)
            rec[f"{key}_best"] = fid_i[key]["best"].get(m, np.nan)
        rec["dcr_rank"] = prv_i["rank"].get(m, np.nan)
        rec["dcr_best"] = prv_i["best"].get(m, np.nan)
        if m in wl.index:
            rec.update({f"wilcoxon_{k}": wl.loc[m, k] for k in ("n", "median_delta", "W", "p", "p_holm")})
        rows.append(rec)
    S = pd.DataFrame(rows)

    def rb(rank, best, n, tex):
        if not np.isfinite(rank):
            return "---" if tex else "—"
        return f"{rank:.2f} ({int(best)}/{n})"

    def wtxt(m, tex):
        if m not in wl.index:
            return ["---" if tex else "—"] * 2
        x = wl.loc[m]
        delta = f"{x.median_delta:+.4f}"
        p = f"{x.p:.3f} / {x.p_holm:.3f}"
        return [delta.replace("-", "$-$") if tex else delta, p]

    n_all, n_nc = len(mle_i["ds_all"]), len(mle_i["ds_nc"])
    n_sh, n_tr, n_dcr = (len(fid_i["column_shapes"]["ds"]), len(fid_i["column_pair_trends"]["ds"]), len(prv_i["ds"]))
    n_w = int(wil.n.iloc[0]) if len(wil) else 0

    def row(m, tex):
        s = S.set_index("method").loc[m]
        lab = tex_escape(LABEL[m]) + (r"$^\dagger$" if tex and m in BASELINES else (" †" if m in BASELINES else ""))
        return [lab, rb(s.mle_rank, s.mle_best, n_all, tex), rb(s.mle_rank_noctab, s.mle_best_noctab, n_nc, tex),
                rb(s.column_shapes_rank, s.column_shapes_best, n_sh, tex),
                rb(s.column_pair_trends_rank, s.column_pair_trends_best, n_tr, tex),
                rb(s.dcr_rank, s.dcr_best, n_dcr, tex)] + wtxt(m, tex)

    head = ["Method", f"MLE (n={n_all})", f"MLE w/o CTABGAN (n={n_nc})", f"Shape (n={n_sh})",
            f"Trend (n={n_tr})", f"DCR (n={n_dcr})", "median Δ MLE vs TabDDPM", f"Wilcoxon p / Holm (n={n_w})"]
    groups = [BASELINES, ["TabDDPM"], OURS]
    md = md_table(head, [row(m, False) for g in groups for m in g])
    note = ("Each metric cell: average rank (1 = best) over the n datasets where every method ranked in that "
            "column has a value, and in brackets the number of those datasets on which the method ranks first "
            "(ties count for every tied method). MLE ranks are over all synthetic methods of Table 1 (DE excluded); "
            "Shape and Trend are available for our three variants only (ranks among three); DCR for TabDDPM and "
            "our three variants (ranks among four; DE and the datasets whose real train set the TabDDPM DCR "
            "subsamples excluded; TabDDPM's DCR is from one generated sample). These ranks are therefore not on "
            "the same scale as the MLE ranks. Shape/Trend: lower error is better; DCR: larger is ranked better. Last two columns: per-dataset difference in ML efficiency, GNN variant "
            f"minus TabDDPM (authors' values), over the n = {n_w} datasets where both exist (DE excluded); "
            "two-sided Wilcoxon signed-rank test, p-values raw and Holm-corrected over the three variants. "
            "† reported values (TabDDPM paper, Table 5).")
    cap = (r"Summary. Each cell: average rank (number of datasets ranked first / n). MLE ranks are over all "
           r"synthetic methods of Table~\ref{tab:mle} (DE excluded); Shape and Trend exist for our variants only "
           r"(ranks among three), DCR for TabDDPM and our variants (ranks among four, larger DCR ranked better; "
           r"TabDDPM from one generated sample); none of these is on the MLE scale. Last columns: median per-dataset difference in ML "
           r"efficiency vs.\ TabDDPM and two-sided Wilcoxon signed-rank $p$ (raw / Holm-corrected over the three "
           r"variants), $n=%d$ datasets (DE excluded). $^\dagger$Reported values.") % n_w
    tex = tex_table("l" + "c" * 7, [tex_escape(h).replace("Δ", r"$\Delta$") for h in head],
                    [[row(m, True) for m in g] for g in groups], cap, "tab:summary")
    return md + "\n\n" + note + "\n", tex, S


def table_configs(ours: pd.DataFrame) -> tuple[str, str]:
    c = ours[ours["mean"].notna()].groupby("dataset").config.first().reindex(DS)
    rows = []
    for ds in DS:
        cfg = c.get(ds)
        if not isinstance(cfg, str) or not cfg:
            rows.append([f"{ds} ({AB[ds]})", "—", "—", "—", "—", "—"])
            continue
        g, L, d, k, h = cfg.split("_")[:5]
        rows.append([f"{ds} ({AB[ds]})", g.upper().replace("GATV2", "GATv2"), L[1:], d[1:], k[1:], h[1:]])
    head = ["Dataset", "GNN", "layers", "d_model", "top-k (fixed)", "heads"]
    md = md_table(head, rows)
    tex = tex_table("lccccc", [tex_escape(x) for x in head],
                    [[[tex_escape(x).replace("—", "---") for x in r] for r in rows]],
                    "GNN configuration used for each dataset (identical for the three top-$k$ rules; "
                    "top-$k$ applies to GNN-fixed).", "tab:gnn_configs")
    return md + "\n", tex.replace(r"\resizebox{\textwidth}{!}{%", r"{%")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_TEXT2)
    ax.tick_params(colors=C_TEXT2, labelcolor=C_TEXT, labelsize=9)
    ax.grid(axis=grid, color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def colour(m):
    return C_OURS.get(m, C_TDDPM if m == "TabDDPM" else C_BASE)


def fig_ranks(summary: pd.DataFrame, mle_i, fid_i, prv_i):
    S = summary.set_index("method")
    panels = [("mle_rank", f"ML efficiency (n={len(mle_i['ds_all'])})", SYNTH),
              ("column_shapes_rank", f"Column Shapes (n={len(fid_i['column_shapes']['ds'])})", OURS),
              ("column_pair_trends_rank", f"Column Pair Trends (n={len(fid_i['column_pair_trends']['ds'])})", OURS),
              ("dcr_rank", f"DCR (n={len(prv_i['ds'])})", PRIV_METHODS)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    for ax, (col, title, methods) in zip(axes, panels):
        _style(ax)
        y = np.arange(len(methods))[::-1]
        v = [S.loc[m, col] for m in methods]
        ax.barh(y, v, height=0.62, color=[colour(m) for m in methods], edgecolor="white", linewidth=2)
        for yi, vi in zip(y, v):
            if np.isfinite(vi):
                ax.text(vi + 0.05, yi, f"{vi:.2f}", va="center", fontsize=8, color=C_TEXT)
        ax.set_yticks(y, [LABEL[m] for m in methods])
        ax.set_xlim(0, len(methods) + 0.6)
        ax.set_xlabel("average rank (1 = best)", color=C_TEXT2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=C_TEXT)
    fig.text(0.01, -0.04, "ML efficiency: rank among all synthetic methods, DE excluded. Shape, Trend: the three "
             "GNN variants only (rank among three). DCR: TabDDPM (one generated sample) and the three GNN variants, "
             "DE excluded; larger DCR ranked better.", fontsize=8, color=C_TEXT2)
    fig.tight_layout()
    _save(fig, "average_rank")


def fig_delta(mle_i: dict):
    W, S = mle_i["W"], mle_i["S"]
    dss = [d for d in DS if W.loc[d, OURS].notna().any() and np.isfinite(W.loc[d, "TabDDPM"])]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(dss) + 1.4))
    _style(ax)
    ax.axvline(0, color=C_TEXT2, lw=1)
    ypos = {d: len(dss) - 1 - i for i, d in enumerate(dss)}
    for off, r in zip((0.22, 0.0, -0.22), OURS):
        for d in dss:
            if not np.isfinite(W.loc[d, r]):
                continue
            delta = W.loc[d, r] - W.loc[d, "TabDDPM"]
            err = np.hypot(S.loc[d, r], S.loc[d, "TabDDPM"])
            y = ypos[d] + off
            nl = d in NOT_LIKE_FOR_LIKE
            ax.errorbar(delta, y, xerr=err, fmt="o", ms=6, color=C_OURS[r], mfc="white" if nl else C_OURS[r],
                        mew=1.6, elinewidth=1.6, capsize=0)
    if any(d in NOT_LIKE_FOR_LIKE for d in dss):
        for d in NOT_LIKE_FOR_LIKE & set(dss):
            ax.axhspan(ypos[d] - 0.45, ypos[d] + 0.45, color=C_GRID, alpha=0.5, lw=0, zorder=0)
    ax.set_yticks([ypos[d] for d in dss],
                  [f"{AB[d]} ({'R²' if TASK[d] == 'regression' else 'F1'})"
                   + (" ‡ not like-for-like" if d in NOT_LIKE_FOR_LIKE else "") for d in dss])
    ax.set_ylim(-0.7, len(dss) - 0.3)
    ax.set_xlabel("Δ ML efficiency vs TabDDPM (macro-F1 or R²; positive = GNN higher)", color=C_TEXT2, fontsize=9)
    handles = [plt.Line2D([], [], color=C_OURS[r], marker="o", lw=0, ms=6, label=OURS_LABEL[r]) for r in OURS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    fig.text(0.01, -0.02, "Error bars: ±√(sd²_GNN + sd²_TabDDPM), sds over the 50 runs. Hollow markers on a "
             "shaded row: DE, local split and column typing differ from the authors'.", fontsize=7.5, color=C_TEXT2)
    fig.tight_layout()
    _save(fig, "delta_mle_vs_tabddpm")


# ---------------------------------------------------------------------------
# RESULTS.md
# ---------------------------------------------------------------------------
def summary_text(mle_i, fid_i, prv_i, wil, P) -> list[str]:
    L = []
    ra, rn = mle_i["rank_all"], mle_i["rank_nc"]
    order = ra.sort_values()
    L.append(f"- **ML efficiency.** Over the {len(mle_i['ds_all'])} datasets where every synthetic method has a "
             f"value, the average ranks are " + ", ".join(f"{LABEL[m]} {v:.2f}" for m, v in order.items())
             + f". Without CTABGAN ({len(mle_i['ds_nc'])} datasets): "
             + ", ".join(f"{LABEL[m]} {v:.2f}" for m, v in rn.sort_values().items()) + ".")
    for _, w in wil.iterrows():
        L.append(f"- **{OURS_LABEL[w.method]} vs TabDDPM** (n = {w.n}, DE excluded): higher on {w.n_above}, "
                 f"lower on {w.n_below} datasets; median Δ = {w.median_delta:+.4f}, mean Δ = {w.mean_delta:+.4f}; "
                 f"Wilcoxon p = {w.p:.3f} (Holm {w.p_holm:.3f}).")
    for key, name in FID:
        r = fid_i[key]["rank"].sort_values()
        L.append(f"- **{name}** (n = {len(fid_i[key]['ds'])}): average rank "
                 + ", ".join(f"{OURS_LABEL[m]} {v:.2f}" for m, v in r.items()) + ".")
    ratio = P.dropna(subset=["ratio"]).groupby("method").ratio.agg(["min", "median", "max", "count"])
    L.append(f"- **DCR** (n = {len(prv_i['ds'])}): average rank "
             + ", ".join(f"{LABEL[m]} {v:.2f}" for m, v in prv_i["rank"].sort_values().items())
             + " (larger DCR ranked better; TabDDPM from one generated sample).")
    L.append("- **DCR / holdout DCR** across datasets (min / median / max): "
             + "; ".join(f"{LABEL[m]} {ratio.loc[m, 'min']:.2f} / {ratio.loc[m, 'median']:.2f} / "
                         f"{ratio.loc[m, 'max']:.2f} (n = {int(ratio.loc[m, 'count'])})"
                         for m in PRIV_METHODS if m in ratio.index) + ".")
    mia = P.groupby("method").mia_auc.agg(["min", "max"])
    L.append("- **MIA AUC** range across datasets: "
             + "; ".join(f"{OURS_LABEL[m]} {mia.loc[m, 'min']:.3f}–{mia.loc[m, 'max']:.3f}" for m in OURS
                         if m in mia.index) + ".")
    return L


def write_results(parts: dict, mle_i, fid_i, prv_i, wil, P, hold, missing: list[str]):
    L = ["# Results (paper-style presentation)\n",
         "_Generated by `analysis/paper_tables/paper_tables.py`; rerun it to refresh. LaTeX versions in "
         "`tables/`, figures (PDF + PNG) in `figures/`, numbers behind every cell in the CSV files._\n",
         "Methods: Real (CatBoost on the real train split); CTGAN, TVAE, CTABGAN, CTABGAN+, SMOTE (reported in "
         "the TabDDPM paper, Table 5, same tuned-CatBoost protocol); TabDDPM (authors' released evaluation); "
         "GNN-fixed, GNN-Jenks, GNN-entmax (ours; the three top-k rules on the GNN configuration used for each "
         "dataset, listed at the end). "
         + (f"No GNN results are available for {', '.join(missing)}." if missing else "") + "\n",
         "## Summary of results\n", *summary_text(mle_i, fid_i, prv_i, wil, P), "",
         "## Table 1 — Machine learning efficiency\n", parts["mle"],
         "## Table 2 — Fidelity: Column Shapes and Column Pair Trends (ours only)\n", parts["fid"],
         "## Table 3 — Privacy\n", parts["prv"],
         "## Table 4 — Summary\n", parts["sum"],
         "## Figures\n",
         "![Average rank per metric](figures/average_rank.png)\n",
         "![Δ ML efficiency vs TabDDPM](figures/delta_mle_vs_tabddpm.png)\n",
         "## GNN configuration used for each dataset\n", parts["cfg"],
         "## Data sources\n",
         f"- Every file read is listed in `provenance.csv` ({len(set(r['path'] for r in _READ))} distinct paths). "
         f"The guard aborts on any path containing `{FORBIDDEN}`: the budget-matched runs are not read. "
         f"Such trees present on disk (not opened): {parts['tddpm_trees'] or 'none'}.",
         "- TabDDPM's DCR (`privacy.json`) is the same statistic as our DCR, computed by the same code: "
         "`dcr_recipe_check.py` runs the authors' `privacy_metrics` verbatim on our samples (see "
         "`DCR_RECIPE_CHECK.md`).",
         f"- Holdout DCR recomputed from `data/<ds>/` with the protocol's own DCR code; self-check that the same "
         f"code reproduces a stored synthetic DCR: {hold.attrs.get('selfcheck', 'n/a')}.",
         ""]
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
def main():
    for d in (TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    global TASK
    TASK = {ds: task_of(ds) for ds in DS}

    auth = load_authors()
    ours, cells = load_ours()
    have = ours[ours["mean"].notna()]
    missing = [AB[d] for d in DS if d not in set(have.dataset)]
    fid, floor = load_fidelity(ours)
    hold = holdout_dcr(sorted(set(have.dataset), key=DS.index), ours)

    mle = mle_long(auth, ours)
    md1, tex1, mle_i = table_mle(mle)
    md2, tex2, fid_i = table_fidelity(fid, floor)
    md3, tex3, prv_i, P = table_privacy(ours, hold, load_tabddpm_dcr())
    wil = wilcoxon_vs_tabddpm(mle_i["W"])
    md4, tex4, summ = table_summary(mle_i, fid_i, prv_i, wil)
    md5, tex5 = table_configs(ours)

    for name, md, tex in (("table1_mle", md1, tex1), ("table2_fidelity", md2, tex2),
                          ("table3_privacy", md3, tex3), ("table4_summary", md4, tex4),
                          ("gnn_configs", md5, tex5)):
        (TAB / f"{name}.md").write_text(md)
        (TAB / f"{name}.tex").write_text(tex)
    mle.to_csv(OUT / "mle_values.csv", index=False)
    fid.assign(error_pct=100 * (1 - fid["mean"])).to_csv(OUT / "fidelity_values.csv", index=False)
    P.to_csv(OUT / "privacy_values.csv", index=False)
    hold.to_csv(OUT / "holdout_dcr.csv", index=False)
    wil.to_csv(OUT / "wilcoxon_vs_tabddpm.csv", index=False)
    summ.to_csv(OUT / "summary.csv", index=False)

    fig_ranks(summ, mle_i, fid_i, prv_i)
    fig_delta(mle_i)

    # names only: proves the guard had something to guard against, opens nothing inside
    tddpm_trees = sorted(ct.rel(p) for p in ct.RESULTS_ROOT.glob(f"*/*{FORBIDDEN}*"))
    tddpm_trees += sorted(ct.rel(p) for p in (REPO / "exp").glob(f"*/*{FORBIDDEN}*"))
    pd.DataFrame(_READ).drop_duplicates().to_csv(OUT / "provenance.csv", index=False)
    write_results({"mle": md1, "fid": md2, "prv": md3, "sum": md4, "cfg": md5,
                   "tddpm_trees": ", ".join(tddpm_trees)}, mle_i, fid_i, prv_i, wil, P, hold, missing)
    print(f"wrote {ct.rel(OUT / 'RESULTS.md')}, tables/, figures/  (GNN results missing for: {', '.join(missing)})")


TASK: dict = {}

if __name__ == "__main__":
    main()

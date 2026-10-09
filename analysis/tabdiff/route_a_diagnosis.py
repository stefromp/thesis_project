#!/usr/bin/env python3
"""
Route A diagnosis: where AttnGNN (d256) inside TabDiff loses fidelity on beijing
and magic, and how training went (beijing, magic vs shoppers, adult). CPU only,
reads finished runs, trains nothing.

Two steps (two venvs, as the evaluation itself):

    # 1. tabpc venv: per-pair NMI with tabpc's own binning + MI (the pieces of wNMIS)
    $TABPC_PYTHON analysis/tabdiff/route_a_diagnosis.py nmi --exp_root $TRANSFER_ROOT
    # 2. eval venv: per-column Shape / per-pair Trend with TabDiff's own QualityReport,
    #    beijing specifics, training diagnostics, figures, the report
    $EVAL_PYTHON  analysis/tabdiff/route_a_diagnosis.py report --exp_root $TRANSFER_ROOT

Runs: <exp_root>/exp_tabdiff_gnn/tabdiff_<ds>/attngnn/<arm>/seed<seed>/ (config.run_dir);
override per dataset with --run_dir ds=PATH (e.g. after an rsync of only
samples/ or tabdiff_eval/all_samples/, train_log.csv, train_record.json, ckpt/).
Tables: <run>/samples/samples_<k>.csv, else <run>/tabdiff_eval/all_samples/samples_<k>.csv.

Outputs: analysis/tabdiff/route_a_diagnosis.md + analysis/tabdiff/route_a_diagnosis/
(*.csv per-column / per-pair tables, figures/*.png).

Definitions (all on the tables the evaluation scored):
  Shape error  (1 - sdmetrics Column Shapes score) x 100, per column, mean over tables
               -- TabDiff's QualityReport on TabDiff's reordered frame (tabdiff.metrics.reorder)
  Trend error  (1 - Column Pair Trends score) x 100, per pair, mean over tables
  NMI pair     tabpc: 10 equal-width bins on the real range (df_common_binning), NMI matrix
               (df_mutual_information); wNMIS = 1 - sum_p w_p |NMI_r - NMI_s| / sum_p w_p,
               w_p = NMI_r + NMI_s; a pair's share of the loss = w_p |d| / sum w
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tabdiff_gnn"))
import config as C  # noqa: E402

OUT_DIR = os.path.join(HERE, "route_a_diagnosis")
FIG_DIR = os.path.join(OUT_DIR, "figures")
REPORT = os.path.join(HERE, "route_a_diagnosis.md")


def set_out(base):
    """--out: write the report and its folder under `base` instead of analysis/tabdiff/
    (tests and stand-in data must never overwrite the real report)."""
    global OUT_DIR, FIG_DIR, REPORT
    OUT_DIR = os.path.join(base, "route_a_diagnosis")
    FIG_DIR = os.path.join(OUT_DIR, "figures")
    REPORT = os.path.join(base, "route_a_diagnosis.md")
FIDELITY_DS = ["beijing", "magic"]
TRAINING_DS = ["beijing", "magic", "shoppers", "adult"]
WEATHER = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
TIME = ["year", "month", "day", "hour"]


# ---------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------
def run_dirs(exp_root, overrides, seed=0):
    out = {}
    for ds in sorted(set(FIDELITY_DS + TRAINING_DS)):
        out[ds] = overrides.get(ds) or (C.run_dir(exp_root, C.target_name(ds), "attngnn", seed)
                                        if exp_root else None)
    return out


def table_paths(run):
    for sub in ("samples", os.path.join("tabdiff_eval", "all_samples")):
        ps = sorted(glob.glob(os.path.join(run, sub, "samples_*.csv")),
                    key=lambda p: int(re.search(r"samples_(\d+)", p).group(1)))
        if ps:
            return ps
    return []


def info_and_real(ds):
    import pandas as pd
    f = C.data_files(C.target_name(ds))
    return json.load(open(f["info"])), pd.read_csv(f["real"])


def column_types(info):
    """TabDiff's typing: numerical = num_col_idx (+ target for regression), categorical =
    cat_col_idx (+ target for classification); and the order of reorder()."""
    cols = info["column_names"]
    num = list(info["num_col_idx"]) + (list(info["target_col_idx"]) if info["task_type"] == "regression" else [])
    cat = list(info["cat_col_idx"]) + (list(info["target_col_idx"]) if info["task_type"] != "regression" else [])
    return [cols[i] for i in num], [cols[i] for i in cat]


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# step 1 (tabpc venv): per-pair NMI
# ---------------------------------------------------------------------------
def cmd_nmi(a):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, os.path.join(REPO, "tabdiff_gnn"))
    import tabpc_eval
    M = tabpc_eval.import_tabpc()                    # tabpc's src (pinned, digests checked)
    C.verify_tabpc()
    from src.util import df_mutual_information
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = run_dirs(a.exp_root, parse_overrides(a.run_dir), a.seed)
    for ds in FIDELITY_DS:
        info, real = info_and_real(ds)
        paths = table_paths(runs[ds]) if runs[ds] else []
        if not paths:
            print(f"  [nmi] {ds}: no tables under {runs[ds]} -- skipped")
            continue
        rows = []
        for k, p in enumerate(paths):
            syn = pd.read_csv(p)
            br, bs = M.df_common_binning(real_data=real, synthetic_data=syn, info=info, bins=10)
            mr = df_mutual_information(br, encode=False, device="cpu")["nmi"]
            ms = df_mutual_information(bs, encode=False, device="cpu")["nmi"]
            cols = list(real.columns)
            ii, jj = np.tril_indices(len(cols), k=-1)
            err = np.abs(mr[ii, jj] - ms[ii, jj])
            w = mr[ii, jj] + ms[ii, jj]
            wnmis = 1 - float((w * err).sum() / w.sum())
            # cross-check against tabpc's own MI_l1 on the same table
            ref = M.MI_l1()(real, syn, None, info, device="cpu")["nmi_l1_weighted_complement"]
            if abs(ref - wnmis) > 1e-6:
                raise RuntimeError(f"{ds} table {k}: wNMIS {wnmis} != tabpc MI_l1 {ref}")
            for a_, b_, r, s, e, ww in zip(ii, jj, mr[ii, jj], ms[ii, jj], err, w):
                rows.append({"table": k, "col_a": cols[b_], "col_b": cols[a_], "nmi_real": float(r),
                             "nmi_syn": float(s), "abs_err": float(e), "weight": float(ww),
                             "loss_share": float(ww * e / w.sum()), "wnmis_table": wnmis})
            print(f"  [nmi] {ds} table {k}: wNMIS {wnmis:.4f} (= tabpc MI_l1)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"{ds}_nmi_pairs_per_table.csv"), index=False)
    return 0


# ---------------------------------------------------------------------------
# step 2 (eval venv): fidelity
# ---------------------------------------------------------------------------
def quality_details(real, syn, info):
    """TabDiff's density evaluation (tabdiff.metrics: reorder + sdmetrics QualityReport),
    returning the per-column and per-pair details with real column names."""
    from copy import deepcopy
    from sdmetrics.reports.single_table import QualityReport
    sys.path.insert(0, os.path.join(REPO, C.TABDIFF_REL))
    from tabdiff.metrics import reorder
    inf = deepcopy(info)
    r, s = real.copy(), syn.copy()
    r.columns = range(len(r.columns))
    s.columns = range(len(s.columns))
    inf["metadata"]["columns"] = {int(k): v for k, v in inf["metadata"]["columns"].items()}
    nr, ns, meta = reorder(r, s, inf)
    num, cat = column_types(info)
    names = num + cat
    q = QualityReport()
    q.generate(nr, ns, meta, verbose=False)
    shapes = q.get_details(property_name="Column Shapes").copy()
    trends = q.get_details(property_name="Column Pair Trends").copy()
    shapes["Column"] = [names[int(c)] for c in shapes["Column"]]
    trends["Column 1"] = [names[int(c)] for c in trends["Column 1"]]
    trends["Column 2"] = [names[int(c)] for c in trends["Column 2"]]
    return shapes, trends, q.get_properties()


def eta(cat, y):
    """Correlation ratio: share of y's variance explained by the categories."""
    import numpy as np
    y = np.asarray(y, float)
    tot = ((y - y.mean()) ** 2).sum()
    if tot == 0:
        return float("nan")
    import pandas as pd
    g = pd.DataFrame({"c": cat, "y": y}).groupby("c")["y"]
    return float((g.count() * (g.mean() - y.mean()) ** 2).sum() / tot)


def fidelity(ds, run, out):
    import numpy as np
    import pandas as pd
    info, real = info_and_real(ds)
    paths = table_paths(run)
    if not paths:
        return None
    sh_all, tr_all, props = [], [], []
    tables = []
    for k, p in enumerate(paths):
        syn = pd.read_csv(p)
        tables.append(syn)
        sh, tr, pr = quality_details(real, syn, info)
        sh["table"], tr["table"] = k, k
        sh_all.append(sh)
        tr_all.append(tr)
        props.append(dict(zip(pr["Property"], pr["Score"])))
        print(f"  [fidelity] {ds} table {k}: Shape err {100 * (1 - props[-1]['Column Shapes']):.2f}  "
              f"Trend err {100 * (1 - props[-1]['Column Pair Trends']):.2f}", flush=True)
    sh = pd.concat(sh_all)
    tr = pd.concat(tr_all)
    sh["error"] = 100 * (1 - sh["Score"])
    tr["error"] = 100 * (1 - tr["Score"])
    num, cat = column_types(info)
    col_tab = (sh.groupby(["Column", "Metric"])["error"].agg(["mean", "std"]).reset_index()
               .sort_values("mean", ascending=False))
    col_tab["type"] = ["numerical" if c in num else "categorical" for c in col_tab["Column"]]
    agg = {"error": ["mean", "std"]}
    for extra in ("Real Correlation", "Synthetic Correlation"):
        if extra in tr:
            agg[extra] = "mean"
    pair_tab = tr.groupby(["Column 1", "Column 2", "Metric"]).agg(agg).reset_index()
    pair_tab.columns = [" ".join(c).strip() if isinstance(c, tuple) else c for c in pair_tab.columns]
    pair_tab = pair_tab.rename(columns={"error mean": "mean", "error std": "std",
                                        "Real Correlation mean": "real_corr",
                                        "Synthetic Correlation mean": "syn_corr"})
    pair_tab = pair_tab.sort_values("mean", ascending=False)
    col_tab.to_csv(os.path.join(out, f"{ds}_shape_per_column.csv"), index=False)
    pair_tab.to_csv(os.path.join(out, f"{ds}_trend_per_pair.csv"), index=False)
    res = {"ds": ds, "n_tables": len(paths), "info": info, "real": real, "tables": tables,
           "num": num, "cat": cat, "col_tab": col_tab, "pair_tab": pair_tab,
           "shape_err": 100 * (1 - np.mean([p["Column Shapes"] for p in props])),
           "trend_err": 100 * (1 - np.mean([p["Column Pair Trends"] for p in props]))}
    nmi_p = os.path.join(out, f"{ds}_nmi_pairs_per_table.csv")
    if os.path.exists(nmi_p):
        n = pd.read_csv(nmi_p)
        res["wnmis"] = n.groupby("table")["wnmis_table"].first().mean()
        nt = (n.groupby(["col_a", "col_b"])[["nmi_real", "nmi_syn", "abs_err", "loss_share"]].mean()
              .reset_index().sort_values("loss_share", ascending=False))
        nt.to_csv(os.path.join(out, f"{ds}_nmi_per_pair.csv"), index=False)
        res["nmi_tab"] = nt
    return res


def beijing_specifics(res, out):
    """pm2.5 marginal and dependencies; time x weather structure."""
    import numpy as np
    import pandas as pd
    from scipy import stats
    real, tables = res["real"], res["tables"]
    syn = pd.concat(tables, ignore_index=True)
    tgt = "pm2.5"
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    marg = pd.DataFrame({"quantile": [f"{100 * q:g}%" for q in qs],
                         "real": [real[tgt].quantile(q) for q in qs],
                         "synthetic (pooled)": [syn[tgt].quantile(q) for q in qs]})
    extra = {"mean": (real[tgt].mean(), syn[tgt].mean()), "std": (real[tgt].std(), syn[tgt].std()),
             "min": (real[tgt].min(), syn[tgt].min()), "max": (real[tgt].max(), syn[tgt].max()),
             "share <= 10": ((real[tgt] <= 10).mean(), (syn[tgt] <= 10).mean())}
    ks = [stats.ks_2samp(real[tgt], t[tgt]).statistic for t in tables]
    num, cat = res["num"], res["cat"]
    dep = []
    for col in [c for c in real.columns if c != tgt]:
        if col in num:
            r = stats.spearmanr(real[tgt], real[col]).correlation
            s_ = [stats.spearmanr(t[tgt], t[col]).correlation for t in tables]
            kind = "Spearman rho"
        else:
            r = eta(real[col].astype(str), real[tgt])
            s_ = [eta(t[col].astype(str), t[tgt]) for t in tables]
            kind = "eta^2 (pm2.5 | col)"
        dep.append({"column": col, "type": "numerical" if col in num else "categorical", "measure": kind,
                    "real": r, "synthetic": float(np.nanmean(s_)), "abs diff": abs(r - float(np.nanmean(s_)))})
    dep = pd.DataFrame(dep)
    pt = res["pair_tab"]
    tr = pt[(pt["Column 1"] == tgt) | (pt["Column 2"] == tgt)].copy()
    tr["other"] = [b if a == tgt else a for a, b in zip(tr["Column 1"], tr["Column 2"])]
    dep = dep.merge(tr[["other", "mean"]].rename(columns={"other": "column", "mean": "Trend err"}),
                    on="column", how="left")
    if "nmi_tab" in res:
        nt = res["nmi_tab"]
        m = nt[(nt.col_a == tgt) | (nt.col_b == tgt)].copy()
        m["column"] = [b if a == tgt else a for a, b in zip(m.col_a, m.col_b)]
        dep = dep.merge(m[["column", "nmi_real", "nmi_syn"]], on="column", how="left")
    dep = dep.sort_values("abs diff", ascending=False)
    # time x weather: eta^2(numeric | time/cbwd column), real vs synthetic
    tw = []
    for c_ in [c for c in TIME + ["cbwd"] if c in real.columns]:
        for n_ in [n for n in WEATHER + [tgt] if n in real.columns]:
            r = eta(real[c_].astype(str), real[n_])
            s_ = float(np.nanmean([eta(t[c_].astype(str), t[n_]) for t in tables]))
            tw.append({"categorical": c_, "numerical": n_, "eta2 real": r, "eta2 syn": s_,
                       "abs diff": abs(r - s_)})
    tw = pd.DataFrame(tw).sort_values("abs diff", ascending=False)
    # weather block: Spearman among numericals
    nums = [n for n in [tgt] + WEATHER if n in real.columns]
    rr = real[nums].corr(method="spearman")
    ss = sum(t[nums].corr(method="spearman") for t in tables) / len(tables)
    # category coverage: unseen / missing categories in the synthetic tables
    cov = []
    for c_ in cat:
        rv, sv = set(real[c_].astype(str)), set(syn[c_].astype(str))
        cov.append({"column": c_, "real levels": len(rv), "synthetic levels": len(sv),
                    "levels never generated": len(rv - sv), "invented levels": len(sv - rv)})
    out_tabs = {"pm25_marginal": marg, "pm25_dependencies": dep, "time_weather_eta2": tw,
                "spearman_real": rr, "spearman_syn": ss, "category_coverage": pd.DataFrame(cov)}
    for k, v in out_tabs.items():
        v.to_csv(os.path.join(out, f"beijing_{k}.csv"), index=k.startswith("spearman"))
    return {"marg": marg, "extra": extra, "ks": ks, "dep": dep, "tw": tw, "rr": rr, "ss": ss,
            "cov": pd.DataFrame(cov), "syn": syn}


# ---------------------------------------------------------------------------
# step 2: training diagnostics
# ---------------------------------------------------------------------------
def training(ds, run):
    import numpy as np
    import pandas as pd
    if not run or not os.path.exists(os.path.join(run, "train_log.csv")):
        return None
    log = pd.read_csv(os.path.join(run, "train_log.csv"))
    rec = json.load(open(os.path.join(run, "train_record.json"))) if os.path.exists(
        os.path.join(run, "train_record.json")) else {}
    best = sorted(glob.glob(os.path.join(run, "ckpt", "best_ema_model_*.pt")))
    sel_epoch = sel_loss = None
    if best:
        m = re.search(r"best_ema_model_([-\d.eE]+)_(\d+)\.pt$", os.path.basename(best[0]))
        sel_loss, sel_epoch = float(m.group(1)), int(m.group(2))
    elif rec.get("checkpoint"):
        m = re.search(r"_([-\d.eE]+)_(\d+)\.pt$", rec["checkpoint"])
        if m:
            sel_loss, sel_epoch = float(m.group(1)), int(m.group(2))
    total = log["loss/total_loss"].to_numpy()
    ep = log["epoch"].to_numpy()
    last = ep.max()

    def slope(lo, hi):
        sel = (ep > lo) & (ep <= hi)
        if sel.sum() < 10:
            return float("nan")
        b = np.polyfit(ep[sel], total[sel], 1)[0]
        return float(b * 1000)                          # change per 1000 epochs

    w = min(1000, max(10, int(last // 8)))
    s_last = slope(last - w, last)
    s_prev = slope(last - 2 * w, last - w)
    m_last = float(total[ep > last - w // 2].mean())
    m_prev = float(total[(ep > last - w) & (ep <= last - w // 2)].mean())
    info, _ = info_and_real(ds)
    num, cat = column_types(info)
    # TabDiff's column order inside the model: y first (make_dataset concat_y_to_X)
    if info["task_type"] == "regression":
        num_order = num[-1:] + num[:-1]
        cat_order = cat
    else:
        num_order = num
        cat_order = cat[-1:] + cat[:-1]
    rho_cols = sorted([c for c in log.columns if c.startswith("num_noise/rho_col_")],
                      key=lambda c: int(c.rsplit("_", 1)[1]))
    k_cols = sorted([c for c in log.columns if c.startswith("cat_noise/k_col_")],
                    key=lambda c: int(c.rsplit("_", 1)[1]))
    rho_final = {num_order[i]: float(log[c].iloc[-1]) for i, c in enumerate(rho_cols)}
    k_final = {cat_order[i]: float(log[c].iloc[-1]) for i, c in enumerate(k_cols)}
    rho_sel, k_sel = {}, {}
    if best:
        try:
            import torch
            st = torch.load(best[0], map_location="cpu", weights_only=False)
            sp = torch.nn.functional.softplus
            if "rho_raw" in st.get("num_schedule", {}):
                v = (sp(st["num_schedule"]["rho_raw"]) + 5.0).tolist()      # rho_offset = 5 (toml)
                rho_sel = dict(zip(num_order, v))
            if "k_raw" in st.get("cat_schedule", {}):
                v = (sp(st["cat_schedule"]["k_raw"]) + 1.0).tolist()        # k_offset = 1 (toml)
                k_sel = dict(zip(cat_order, v))
        except Exception as exc:                                             # recorded, not fatal
            rho_sel = {"error": str(exc)}
    lrs = log["lr"].to_numpy()
    return {"ds": ds, "log": log, "epochs": int(last), "steps_per_epoch": rec.get("steps_per_epoch"),
            "sel_epoch": sel_epoch, "sel_loss": sel_loss, "slope_last": s_last, "slope_prev": s_prev,
            "window": w, "mean_last": m_last, "mean_prev": m_prev,
            "min_total": float(total.min()), "min_epoch": int(ep[total.argmin()]),
            "final_total": float(total[-1]), "lr_final": float(lrs[-1]),
            "lr_reductions": int((np.diff(lrs) < 0).sum()),
            "rho_final": rho_final, "k_final": k_final, "rho_sel": rho_sel, "k_sel": k_sel,
            "train_wall_h": (rec.get("train_wall_s") or 0) / 3600, "completed": rec.get("completed")}


# ---------------------------------------------------------------------------
# figures + report
# ---------------------------------------------------------------------------
def figures(fid, bj, trn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(FIG_DIR, exist_ok=True)
    made = []
    present = [r for r in fid if r]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 4.2), squeeze=False)
        for ax, r in zip(axes[0], present):
            t = r["col_tab"].groupby("Column")["mean"].mean().sort_values(ascending=False)
            colors = ["#c0392b" if c in r["num"] else "#2e86c1" for c in t.index]
            ax.bar(range(len(t)), t.values, color=colors)
            ax.set_xticks(range(len(t)))
            ax.set_xticklabels(t.index, rotation=60, ha="right", fontsize=8)
            ax.set_ylabel("Shape error (%)")
            ax.set_title(f"{r['ds']}: per-column Shape error (red = numerical, blue = categorical)",
                         fontsize=9)
        fig.tight_layout()
        p = os.path.join(FIG_DIR, "shape_per_column.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    rb = next((r for r in fid if r and r["ds"] == "beijing"), None)
    if rb is not None:
        cols = rb["num"] + rb["cat"]
        M = np.full((len(cols), len(cols)), np.nan)
        for _, row in rb["pair_tab"].iterrows():
            i, j = cols.index(row["Column 1"]), cols.index(row["Column 2"])
            M[i, j] = M[j, i] = row["mean"]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(M, cmap="Reds")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(cols, fontsize=8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="Trend error (%)")
        ax.set_title("beijing: per-pair Trend error (mean over tables)", fontsize=9)
        fig.tight_layout()
        p = os.path.join(FIG_DIR, "beijing_trend_heatmap.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    if bj is not None:
        real, syn = rb["real"], bj["syn"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        bins = np.linspace(0, np.log1p(max(real["pm2.5"].max(), syn["pm2.5"].max())), 60)
        axes[0].hist(np.log1p(real["pm2.5"]), bins=bins, density=True, alpha=.55, label="real")
        axes[0].hist(np.log1p(syn["pm2.5"].clip(lower=0)), bins=bins, density=True, alpha=.55,
                     label="synthetic (pooled)")
        axes[0].set_xlabel("log(1 + pm2.5)")
        axes[0].set_title("pm2.5 marginal")
        axes[0].legend(fontsize=8)
        for ax, (c_, n_) in zip(axes[1:], [("hour", "pm2.5"), ("month", "TEMP")]):
            r_ = real.groupby(c_)[n_].mean()
            s_ = syn.groupby(c_)[n_].mean()
            ax.plot(r_.index, r_.values, "o-", label="real")
            ax.plot(s_.index, s_.values, "s--", label="synthetic")
            ax.set_xlabel(c_)
            ax.set_ylabel(f"mean {n_}")
            ax.set_title(f"mean {n_} by {c_}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(FIG_DIR, "beijing_pm25_and_time_weather.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    tp = [t for t in trn if t]
    if tp:
        fig, axes = plt.subplots(1, len(tp), figsize=(4.5 * len(tp), 3.8), squeeze=False)
        for ax, t in zip(axes[0], tp):
            lg = t["log"]
            ax.plot(lg["epoch"], lg["loss/total_loss"], lw=.6, label="total")
            ax.plot(lg["epoch"], lg["loss/c_loss"], lw=.6, label="CLoss (numerical)")
            ax.plot(lg["epoch"], lg["loss/d_loss"], lw=.6, label="DLoss (categorical)")
            if t["sel_epoch"]:
                ax.axvline(t["sel_epoch"], color="k", ls=":", lw=1, label=f"best EMA @ {t['sel_epoch']}")
            ax.axvline(4000, color="grey", ls="--", lw=.6)
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_title(t["ds"])
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(FIG_DIR, "training_loss_curves.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    return made


def md_table(df, cols=None, fmt="{:.3f}", n=None):
    import pandas as pd
    d = df if cols is None else df[cols]
    if n:
        d = d.head(n)
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "---|" * len(d.columns)
    rows = []
    for _, r in d.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, float):
                cells.append("—" if (v != v) else fmt.format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def report_md(fid, bj, trn, figs, runs):
    import pandas as pd
    L = ["# Route A diagnosis: AttnGNN (d256) inside TabDiff on beijing and magic", "",
         "_Generated by `analysis/tabdiff/route_a_diagnosis.py`; CPU only, no new runs. Per-table "
         "numbers are on the 20 tables the evaluation scored, then averaged._", ""]
    L += ["## Column typing in TabDiff's preprocessing", ""]
    for ds in FIDELITY_DS:
        info, _ = info_and_real(ds)
        num, cat = column_types(info)
        L.append(f"- **{ds}** ({info['task_type']}): numerical {num}; categorical {cat or '—'}; target "
                 f"`{info['column_names'][info['target_col_idx'][0]]}`.")
    L += ["", "beijing's time columns (year, month, day, hour) are **categorical** in TabDiff's typing "
          "(cardinalities 5 / 12 / 31 / 24), so each is one categorical node; the weather columns and "
          "pm2.5 are numerical (Is, Ir, DEWP integer-typed; Is, Ir zero-inflated).", ""]
    for r in fid:
        if not r:
            continue
        L += [f"## {r['ds']}: per-column Shape and per-pair Trend ({r['n_tables']} tables)", "",
              f"Overall: Shape error {r['shape_err']:.2f} %, Trend error {r['trend_err']:.2f} %"
              + (f", wNMIS {r['wnmis']:.4f}" if "wnmis" in r else "") + ".", "",
              "**10 worst columns (Shape error %)**", "",
              md_table(r["col_tab"], ["Column", "type", "Metric", "mean", "std"], "{:.2f}", 10), "",
              "**10 worst pairs (Trend error %)**", ""]
        cols = ["Column 1", "Column 2", "Metric", "mean", "std"] + [c for c in ("real_corr", "syn_corr")
                                                                     if c in r["pair_tab"]]
        L += [md_table(r["pair_tab"], cols, "{:.2f}", 10), ""]
        if "nmi_tab" in r:
            L += ["**10 worst pairs for wNMIS** (tabpc NMI; `loss_share` = the pair's share of 1 − wNMIS)", "",
                  md_table(r["nmi_tab"], ["col_a", "col_b", "nmi_real", "nmi_syn", "abs_err", "loss_share"],
                           "{:.4f}", 10), ""]
        else:
            L += ["_wNMIS per pair not computed: run the `nmi` step in the tabpc venv first._", ""]
    if bj:
        e = bj["extra"]
        L += ["## beijing: the target pm2.5", "", "**Marginal** (real vs pooled synthetic)", "",
              md_table(bj["marg"], fmt="{:.1f}"), "",
              "| statistic | real | synthetic |", "|---|---|---|"]
        L += [f"| {k} | {a:.3f} | {b:.3f} |" for k, (a, b) in e.items()]
        L += [f"| KS statistic (mean over tables) | — | {sum(bj['ks']) / len(bj['ks']):.3f} |", "",
              "**pm2.5 against every other column** (numerical partner: Spearman ρ; categorical partner: "
              "η² = share of pm2.5's variance explained by the category; Trend err and NMI where available)", "",
              md_table(bj["dep"], fmt="{:.3f}"), "",
              "**Time × weather structure**: η² of each numerical column given each time / wind-direction "
              "column, real vs synthetic (largest differences first)", "",
              md_table(bj["tw"], fmt="{:.3f}", n=15), "",
              "**Spearman among the numerical columns** (real / synthetic mean over tables)", "",
              md_table(bj["rr"].reset_index().rename(columns={"index": "real"}), fmt="{:.2f}"), "",
              md_table(bj["ss"].reset_index().rename(columns={"index": "synthetic"}), fmt="{:.2f}"), "",
              "**Category coverage**", "", md_table(bj["cov"], fmt="{:.0f}"), ""]
    tp = [t for t in trn if t]
    if tp:
        L += ["## Training diagnostics", "",
              "TabDiff's Trainer computes the EMA loss every epoch (`trainer.py:243-251`) but **never logs it**, "
              "so no EMA-loss curve exists; what survives is the selected checkpoint "
              "(`best_ema_model_<EMA loss>_<epoch>.pt`: lowest EMA total loss after epoch 4000). Curves below "
              "are the per-epoch training losses (unweighted CLoss + DLoss; the CLoss weight in the update "
              "anneals 1 → 0 over the run).", "",
              "| dataset | epochs | selected best-EMA epoch | its EMA loss | min train total (epoch) | "
              "final total | slope last window (/1000 ep) | slope window before | LR reductions | final lr | "
              "train h |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in tp:
            L.append(f"| {t['ds']} | {t['epochs']} | {t['sel_epoch']} | {t['sel_loss']} | "
                     f"{t['min_total']:.4f} ({t['min_epoch']}) | {t['final_total']:.4f} | "
                     f"{t['slope_last']:+.4f} | {t['slope_prev']:+.4f} | {t['lr_reductions']} | "
                     f"{t['lr_final']:.2e} | {t['train_wall_h']:.1f} |")
        L += ["", f"Windows: the last {tp[0]['window']} epochs and the {tp[0]['window']} before; a slope "
              "near 0 (relative to the loss) = plateaued, clearly negative = still decreasing.", "",
              "**Learned noise schedules** (selected EMA checkpoint; final training value in brackets): "
              "ρ per numerical column (power-mean schedule, offset 5), k per categorical column "
              "(log-linear masking schedule, offset 1).", ""]
        for t in tp:
            rs = ", ".join(f"{c} {t['rho_sel'].get(c, float('nan')):.2f} ({v:.2f})"
                           for c, v in t["rho_final"].items())
            ks = ", ".join(f"{c} {t['k_sel'].get(c, float('nan')):.2f} ({v:.2f})"
                           for c, v in t["k_final"].items())
            L.append(f"- **{t['ds']}** ρ: {rs or '—'}; k: {ks or '—'}")
        L.append("")
    if figs:
        L += ["## Figures", ""] + [f"![{os.path.basename(f)}](route_a_diagnosis/figures/{os.path.basename(f)})"
                                   for f in figs] + [""]
    L += ["## Runs read", ""] + [f"- {ds}: `{p}`" for ds, p in runs.items()] + [""]
    return "\n".join(L)


def cmd_report(a):
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = run_dirs(a.exp_root, parse_overrides(a.run_dir), a.seed)
    fid = [fidelity(ds, runs[ds], OUT_DIR) if runs[ds] else None for ds in FIDELITY_DS]
    rb = next((r for r in fid if r and r["ds"] == "beijing"), None)
    bj = beijing_specifics(rb, OUT_DIR) if rb else None
    trn = [training(ds, runs[ds]) for ds in TRAINING_DS]
    figs = figures(fid, bj, trn)
    with open(REPORT, "w") as fh:
        fh.write(report_md(fid, bj, trn, figs, runs))
    print(f"wrote {REPORT} and {OUT_DIR}/")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["nmi", "report"])
    ap.add_argument("--exp_root", default=None, help="$TRANSFER_ROOT (holds exp_tabdiff_gnn/)")
    ap.add_argument("--run_dir", nargs="*", default=[], metavar="DS=PATH", help="per-dataset run dir override")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="base dir for the report (default analysis/tabdiff/)")
    a = ap.parse_args(argv)
    if a.out:
        set_out(a.out)
    return cmd_nmi(a) if a.step == "nmi" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Figure 5 -- signed relative difference vs TabDiff (as reported in its paper), per rule.

    python3 analysis/summary/fig_vs_tabdiff_heatmap.py [--magic-fixed {dense,topk3}] [--vlim PCT]

Cell = 100 x relative difference, oriented so that POSITIVE = OURS IS BETTER:
  lower is better (Shape, Trend, MLE = RMSE on beijing):  (TabDiff - ours) / TabDiff
  higher is better (MLE = AUC, alpha, beta, C2ST-LR):     (ours - TabDiff) / TabDiff
Ours = mean over the 20 tables (build_summary.b_values; C2ST-LR = avg_std "c2st" / 100);
TabDiff = build_summary.b_tabdiff_values (table B5, analysis/tabdiff/reported.py), checked
here against the numbers of the TabDiff paper's Tables 1-3, 7-9.
Tie (as table B6): |ours - TabDiff| <= our std over the 20 tables -> "≈", lighter shade.
Every cell's W/T/L is checked against csv/B6_detail.csv; the per-rule counts against csv/B6.csv
(exactly for --magic-fixed topk3, which is what B6 uses; with the default dense the magic
fixed column is the topk_dense run and only that column may differ).
Writes figures/vs_tabdiff_reldiff.{png,pdf} and csv/vs_tabdiff_reldiff.csv.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import route_a_figs as F  # noqa: E402

np, pd, plt, ct = F.np, F.pd, F.plt, F.ct
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

RULES = ["fixed", "jenks", "entmax15"]
METRICS = F.BS.PAPER_METRICS                      # shape, trend, mle, alpha, beta, c2st
M_LABEL = {"shape": "Shape", "trend": "Trend", "mle": "MLE", "alpha": "α-precision", "beta": "β-recall",
           "c2st": "C2ST-LR"}
LOWER_BETTER = {"shape", "trend"}                 # + MLE on regression (RMSE)
# TabDiff paper (Tables 1-3, 7-9): Shape %, Trend %, MLE (AUC; beijing RMSE), alpha, beta, C2ST-LR
TABDIFF_PAPER = {"adult": (.63, 1.49, .912, 99.02, 51.64, .9950), "beijing": (1.03, 2.59, .555, 98.06, 59.63, .9781),
                 "default": (1.24, 2.55, .763, 98.49, 51.09, .9774), "diabetes": (.89, 2.20, .689, 95.69, 41.74, .9865),
                 "magic": (.78, .76, .936, 99.47, 48.01, .9989), "shoppers": (1.28, 1.74, .921, 99.11, 49.75, .9843)}


def lower_is_better(ds: str, metric: str) -> bool:
    return metric in LOWER_BETTER or (metric == "mle" and F.RC.TASK[ds] == "regression")


def rel_diff(ours: float, tabdiff: float, lower_better: bool) -> float:
    """Signed relative difference in %, positive = ours better."""
    return 100 * ((tabdiff - ours) if lower_better else (ours - tabdiff)) / tabdiff


def outcome(ours: float, std: float, tabdiff: float, lower_better: bool) -> str:
    """B6's rule: tie if |ours - TabDiff| <= our std, else W / L by the metric's direction."""
    if abs(ours - tabdiff) <= std:
        return "T"
    return "W" if rel_diff(ours, tabdiff, lower_better) > 0 else "L"


def check_reference(tdv) -> None:
    for ds, vals in TABDIFF_PAPER.items():
        for m, want in zip(METRICS, vals):
            got = tdv[(ds, F.BS.paper_key(ds, m))]
            assert F.ok(got) and abs(got["mean"] - want) < 1e-12, (ds, m, got.get("mean"), want)


def collect(magic_fixed: str = "dense") -> pd.DataFrame:
    assert magic_fixed in ("dense", "topk3")
    bv, tdv = F.b_values(), F.tabdiff_values()
    check_reference(tdv)
    rows = []
    for rule in RULES:
        for ds in F.DATASETS:
            arm = "dense" if (rule == "fixed" and ds == "magic" and magic_fixed == "dense") else rule
            run = F.run_dir(ds, arm)
            for m in METRICS:
                k = F.BS.paper_key(ds, m)
                o, t = bv.get((ds, F.ARM_TAG[arm], k)), tdv[(ds, k)]
                if not F.ok(o):
                    rows.append({"rule": rule, "dataset": ds, "metric": m, "arm": arm, "status": o["status"] if o
                                 else "no run on disk"})
                    continue
                lb = lower_is_better(ds, m)
                rows.append({"rule": rule, "dataset": ds, "metric": m, "arm": arm, "status": "ok",
                             "direction": "lower" if lb else "higher", "ours": o["mean"], "ours_std": o["std"],
                             "n_tables": o["n"], "tabdiff": t["mean"],
                             "rel_diff_pct": rel_diff(o["mean"], t["mean"], lb),
                             "result": outcome(o["mean"], o["std"], t["mean"], lb),
                             "run": F.BS.rel(run) if run else ""})
    return pd.DataFrame(rows)


def b6_counts() -> dict:
    """rule -> metric -> (W, T, L) as printed in csv/B6.csv."""
    b6 = pd.read_csv(F.CSV / "B6.csv")
    rule_of = {f"GNN {mode}": tag for tag, mode in F.BS.B_ARMS}       # "GNN adaptive_jenks" -> jenks
    out = {}
    for _, r in b6.iterrows():
        if r.model not in rule_of or not r.metric.startswith("wtl_"):
            continue
        w, t, l = map(int, re.fullmatch(r"derived: wins (\d+), ties (\d+), losses (\d+)", r.note).groups())
        out.setdefault(rule_of[r.model], {})[r.metric[4:]] = (w, t, l)
    assert sorted(out) == sorted(RULES), sorted(out)
    return out


def check_b6(D: pd.DataFrame, magic_fixed: str) -> list[str]:
    """Cell-by-cell vs B6_detail.csv and counts vs B6.csv; returns the notes on the cells that
    use a different run than B6 (magic fixed = dense)."""
    det = pd.read_csv(F.CSV / "B6_detail.csv").set_index(["rule", "dataset", "metric"])
    ok_ = D[D.status == "ok"]
    notes = []
    for _, r in ok_.iterrows():
        key = (r.rule, r.dataset, r.metric)
        same_run = r.arm == r.rule
        assert key in det.index or not same_run, f"{key} missing from B6_detail.csv"
        if same_run:
            b = det.loc[key]
            assert abs(b.ours - r.ours) < 1e-12 and abs(b.tabdiff - r.tabdiff) < 1e-12, key
            assert b.result == r.result, (key, b.result, r.result)
        elif key in det.index and det.loc[key].result != r.result:
            notes.append(f"{r.dataset} {r.metric}: {det.loc[key].result} (B6, topk3) -> {r.result} (dense)")
    counts = b6_counts()
    for rule in RULES:
        for m in METRICS:
            s = ok_[(ok_.rule == rule) & (ok_.metric == m)]
            got = tuple(int((s.result == x).sum()) for x in "WTL")
            if rule == "fixed" and magic_fixed == "dense":
                # the same counts with the magic cell taken from B6 (topk3) must reproduce B6
                key = ("fixed", "magic", m)
                if key in det.index and (s.dataset == "magic").any():
                    adj = list(got)
                    adj["WTL".index(s[s.dataset == "magic"].result.iloc[0])] -= 1
                    adj["WTL".index(det.loc[key].result)] += 1
                    got = tuple(adj)
            assert got == counts[rule][m], (rule, m, got, counts[rule][m])
    return notes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--magic-fixed", choices=("dense", "topk3"), default="dense",
                    help="run in the fixed panel's magic column (default: dense, k = N-1, as the LaTeX tables)")
    ap.add_argument("--vlim", type=float, default=None,
                    help="colour limit in %% (symmetric, shared); default: the largest |cell| rounded up to 1")
    a = ap.parse_args(argv)
    D = collect(a.magic_fixed)
    notes = check_b6(D, a.magic_fixed)
    ok_ = D[D.status == "ok"]
    vmax_data = float(ok_.rel_diff_pct.abs().max())
    vlim = a.vlim or float(np.ceil(vmax_data))
    D["clipped"] = D.rel_diff_pct.abs() > vlim
    D.to_csv(F.CSV / "vs_tabdiff_reldiff.csv", index=False)

    cmap = LinearSegmentedColormap.from_list("div", [ct.C_DIV_NEG, ct.C_DIV_MID, ct.C_DIV_POS])
    norm = TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), sharey=True)
    for ax, rule in zip(axes, RULES):
        sub = D[D.rule == rule]
        for i, m in enumerate(METRICS):
            for j, ds in enumerate(F.DATASETS):
                r = sub[(sub.metric == m) & (sub.dataset == ds)].iloc[0]
                if r.status != "ok":
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="white", ec=ct.C_GRID))
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color=ct.C_TEXT2)
                    continue
                v = float(np.clip(r.rel_diff_pct, -vlim, vlim))
                tie = r.result == "T"
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc=cmap(norm(v)), alpha=0.35 if tie else 1,
                                           ec="white", lw=1.5))
                txt = ("≈" if tie else "") + f"{r.rel_diff_pct:+.1f}" + ("*" if r.clipped else "")
                strong = abs(v) > 0.6 * vlim and not tie
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if strong else ct.C_TEXT)
        ax.set_xlim(-.5, len(F.DATASETS) - .5)
        ax.set_ylim(len(METRICS) - .5, -.5)
        labels = [ds + ("\n(dense)" if rule == "fixed" and ds == "magic" and a.magic_fixed == "dense" else "")
                  for ds in F.DATASETS]
        ax.set_xticks(range(len(F.DATASETS)), labels, fontsize=8)
        ax.set_yticks(range(len(METRICS)), [M_LABEL[m] + (" (AUC / RMSE)" if m == "mle" else "") for m in METRICS],
                      fontsize=8)
        ax.tick_params(length=0, labelcolor=ct.C_TEXT)
        for s in ax.spines.values():
            s.set_visible(False)
        w = {x: int((sub.result == x).sum()) for x in "WTL"}
        ax.set_title(f"{F.ARM_LABEL[rule]}   W {w['W']} / T {w['T']} / L {w['L']}", loc="left", fontsize=10,
                     color=ct.C_TEXT)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("relative difference vs TabDiff (%), + = ours better", fontsize=8, color=ct.C_TEXT2)
    cb.ax.tick_params(labelsize=7, colors=ct.C_TEXT2)
    clip_txt = (f" Colour scale clipped at ±{vlim:g} % (cells beyond marked *; the largest |value| is "
                f"{vmax_data:.1f} %)." if D.clipped.any() else f" Shared colour scale ±{vlim:g} % (no clipping).")
    magic_txt = ("magic in the fixed panel = the dense run (k = N − 1 = 10), as in the LaTeX tables; table B6 uses "
                 "the top-k = 3 run" + (" (outcomes that change: " + "; ".join(notes) + ")" if notes else
                                        " (no outcome changes)") + "."
                 if a.magic_fixed == "dense" else "magic in the fixed panel = the top-k = 3 run (as table B6).")
    F.caption(fig, "Route A, training seed 0, ours = mean over 20 tables; TabDiff = its paper (Tables 1–3, 7–9). Cell = "
                   "signed relative difference in %, positive = ours better: lower-is-better metrics (Shape, Trend, "
                   "MLE on beijing = RMSE) (TabDiff − ours)/TabDiff, higher-is-better (MLE AUC, α, β, C2ST-LR) "
                   "(ours − TabDiff)/TabDiff. ≈ and light shade = tie, |ours − TabDiff| ≤ our std over the 20 tables "
                   "(rule of table B6). " + magic_txt + clip_txt + " news not shown (no evaluated run on disk).",
              y=0.0)
    F.savefig(fig, "vs_tabdiff_reldiff")
    print(ok_.groupby("rule").result.value_counts().unstack().reindex(RULES).to_string())
    print(f"max |rel diff| {vmax_data:.2f} %, colour limit ±{vlim:g} %; B6 checks passed"
          + (f"; magic fixed (dense) vs B6: {notes}" if notes else ""))
    print(f"wrote {F.BS.rel(F.FIG)}/vs_tabdiff_reldiff.{{pdf,png}}, {F.BS.rel(F.CSV)}/vs_tabdiff_reldiff.csv")


if __name__ == "__main__":
    main()

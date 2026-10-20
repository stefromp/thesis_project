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
import thesis_style as S  # noqa: E402

np, pd, plt = F.np, F.pd, F.plt
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

RULES = ["fixed", "jenks", "entmax15"]
METRICS = F.BS.PAPER_METRICS                      # shape, trend, mle, alpha, beta, c2st
M_LABEL = {"shape": "Shape", "trend": "Trend", "mle": "MLE", "alpha": r"$\alpha$-precision",
           "beta": r"$\beta$-recall", "c2st": "C2ST-LR"}
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


def cell_text(v: float) -> str:
    """Signed value, one decimal below 10 %, integer from 10 %."""
    return (f"{v:+.0f}" if abs(v) >= 10 else f"{v:+.1f}").replace("-", "−")


def caption(D: pd.DataFrame, magic_fixed: str, notes: list[str], vlim: float, vmax_data: float) -> str:
    ab = ", ".join(f"{S.DS_SHORT[d]} = {S.ds_label(d)}" for d in F.DATASETS)
    magic = (r"In panel (a) the Magic column (MA*) is the dense run ($k = N-1 = 10$), as in the result "
             r"tables; with the sparse fixed run ($k=3$) its Trend, MLE and $\alpha$-precision cells would be "
             r"losses instead of ties." if magic_fixed == "dense" and notes else
             r"In panel (a) the Magic column (MA*) is the dense run ($k = N-1 = 10$), as in the result tables."
             if magic_fixed == "dense" else r"In panel (a) the Magic column is the sparse fixed run ($k = 3$).")
    clip = (rf" The colour scale is clipped at $\pm{vlim:g}$\%." if D.clipped.any() else "")
    return (r"Relative difference between our AttnGNN denoiser inside TabDiff and TabDiff as reported in its "
            r"paper, for (a) the fixed top-$k$ rule, (b) Jenks and (c) entmax-1.5, on the six metrics the "
            r"TabDiff paper reports (MLE: AUC, or RMSE on Beijing). Each cell is the signed difference in "
            r"percent of TabDiff's value, oriented so that positive (blue) means ours is better: "
            r"$(\text{TabDiff} - \text{ours})/\text{TabDiff}$ for Shape and Trend errors and RMSE, "
            r"$(\text{ours} - \text{TabDiff})/\text{TabDiff}$ for AUC, $\alpha$-precision, $\beta$-recall and "
            r"C2ST-LR. Values are rounded to one decimal below 10\% and to integers above. Hatched, paler cells "
            r"marked $\approx$ in the corner are ties: the difference is within our standard deviation over the 20 tables "
            r"(the rule of the win/tie/loss table); the panel subtitles count wins / ties / losses. "
            rf"Shared colour scale $\pm{vlim:g}$\% (largest magnitude {vmax_data:.1f}\%).{clip} " + magic +
            rf" Datasets: {ab}. Route A, training seed 0, 20 sampled tables; News is not shown (no evaluated "
            r"run yet).")


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

    cmap = plt.get_cmap(S.DIV_CMAP)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)
    fig = S.figure(S.FULL, 2.6)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for k, (ax, rule) in enumerate(zip(axes, RULES)):
        sub = D[D.rule == rule]
        for i, m in enumerate(METRICS):
            for j, ds in enumerate(F.DATASETS):
                r = sub[(sub.metric == m) & (sub.dataset == ds)].iloc[0]
                if r.status != "ok":
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="white", ec=S.GRID, lw=0.5))
                    ax.text(j, i, "–", ha="center", va="center", fontsize=S.FS_CELL, color=S.INK2)
                    continue
                v = float(np.clip(r.rel_diff_pct, -vlim, vlim))
                tie = r.result == "T"
                fc = S.desaturate(cmap(norm(v))) if tie else cmap(norm(v))
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc=fc, ec="white", lw=0.8))
                if tie:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="none", ec="#bdbdbd", lw=0, hatch="///"))
                    ax.text(j + 0.47, i - 0.47, r"$\approx$", ha="right", va="top", fontsize=S.FS_MIN,
                            color=S.INK2)
                ax.text(j, i + 0.06, cell_text(r.rel_diff_pct), ha="center", va="center", fontsize=S.FS_CELL,
                        color=S.text_color_on(fc),
                        bbox=dict(boxstyle="square,pad=0.08", fc=fc, ec="none") if tie else None)
        ax.set_xlim(-.5, len(F.DATASETS) - .5)
        ax.set_ylim(len(METRICS) - .5, -.5)
        ax.set_aspect("auto")
        cols = [S.DS_SHORT[d] + ("*" if rule == "fixed" and d == "magic" and a.magic_fixed == "dense" else "")
                for d in F.DATASETS]
        ax.set_xticks(range(len(F.DATASETS)), cols, rotation=0)
        ax.set_yticks(range(len(METRICS)), [M_LABEL[m] for m in METRICS] if k == 0 else [])
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        w = {x: int((sub.result == x).sum()) for x in "WTL"}
        S.title2(ax, S.rule_label(rule), f"W {w['W']} / T {w['T']} / L {w['L']}")
        S.panel_label(ax, "abc"[k])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_subplot(gs[0, 3]))
    cb.set_label("Relative difference vs TabDiff (%)\n(blue = ours better)", fontsize=S.FS_TICK)
    cb.ax.tick_params(labelsize=S.FS_TICK, length=2, width=S.LW_AXIS)
    cb.outline.set_linewidth(S.LW_AXIS)
    size = S.save(fig, "vs_tabdiff_reldiff", caption(D, a.magic_fixed, notes, vlim, vmax_data))
    print(ok_.groupby("rule").result.value_counts().unstack().reindex(RULES).to_string())
    print(f"max |rel diff| {vmax_data:.2f} %, colour limit ±{vlim:g} %; B6 checks passed"
          + (f"; magic fixed (dense) vs B6: {notes}" if notes else ""))
    print(f"wrote vs_tabdiff_reldiff ({size[0]} x {size[1]} in), csv/vs_tabdiff_reldiff.csv")


if __name__ == "__main__":
    main()

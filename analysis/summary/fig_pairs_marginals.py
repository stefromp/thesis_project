#!/usr/bin/env python3
"""
Figures 6-7 -- where the Trend / Shape errors sit (Route A, seed 0), from the per-run CSVs
written by analysis/tabdiff/pair_and_marginals.py (<run>/analysis/).

    python3 analysis/summary/fig_pairs_marginals.py

pair_trend_error_<ds> (Fig 6: beijing; appendix: default, magic): lower-triangle heatmaps of the
  pair Trend error 100 x (1 - score_ab), mean over the 20 tables, columns in TabDiff's evaluation
  order (numerical block, then categorical; info.json), one sequential colour scale for the rule
  panels and a diverging one for the difference panel (entmax-1.5 - fixed on beijing and magic,
  Jenks - fixed on default). Panel subtitle = mean over pairs = the run's Trend error (checked
  against tabdiff_eval/all_results.csv).
marginals (Fig 7): beijing pm2.5 and DEWP -- real density with each rule's synthetic density
  (top) and synthetic - real per bin (bottom).
marginals_appendix: beijing and default, 4 columns each, chosen from column_shape_error.csv:
  the 2 numerical columns with the largest |err(rule) - err(fixed)| (rule = entmax-1.5 on
  beijing, Jenks on default) and the 2 other columns with the largest error under the best rule
  (= the arm with the lowest mean Shape error).
Each run's pair_marginals_meta.json must record self-checks within 1e-6 (else this stops).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import route_a_figs as F  # noqa: E402
import thesis_style as S  # noqa: E402

np, pd, plt = F.np, F.pd, F.plt
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(F.BS.REPO / "analysis" / "tabdiff"))
import pair_and_marginals as PM  # noqa: E402  (eval_columns: TabDiff's column order and typing)

PAIR_DS = ["beijing", "default", "magic"]
APPX_DS = ["beijing", "default"]
DIFF_RULE = {"beijing": "entmax15", "default": "jenks", "magic": "entmax15"}
ANNOTATE = {"beijing", "magic"}
MARG_COLS = ("pm2.5", "DEWP")                          # Fig 7 (beijing)
SHORT_COL = {"default payment next month": "target"}    # display only (named in the caption)


def columns(ds: str) -> list[dict]:
    return PM.eval_columns(json.loads(Path(F.RC.data_files(F.RC.target_name(ds))["info"]).read_text()))


def col_label(c: dict) -> str:
    return SHORT_COL.get(c["name"], c["name"]) + (" (c)" if c["type"] == "categorical" else "")


def load(ds: str) -> dict:
    """arm -> {pairs, shapes, marg, meta, run} for the arms with analysis outputs on disk."""
    out = {}
    for arm in F.ARMS:
        run = F.run_dir(ds, arm)
        if run is None or not (run / PM.OUT_DIR / "pair_marginals_meta.json").exists():
            continue
        a = run / PM.OUT_DIR
        meta = json.loads((a / "pair_marginals_meta.json").read_text())
        dev = meta["self_check_max_abs_dev"]
        assert max(dev.values()) <= PM.TOL, (run, dev)
        out[arm] = {"pairs": pd.read_csv(a / "pair_trend_error.csv"),
                    "shapes": pd.read_csv(a / "column_shape_error.csv"),
                    "marg": pd.read_csv(a / "marginals.csv", keep_default_na=False, na_values=[""]),
                    "meta": meta, "run": run,
                    "all_results": pd.read_csv(run / "tabdiff_eval" / "all_results.csv", index_col=0)}
    return out


def pair_matrix(pairs: pd.DataFrame, cols: list[dict]) -> np.ndarray:
    pos = {c["name"]: c["pos"] for c in cols}
    n = len(cols)
    M = np.full((n, n), np.nan)
    for _, r in pairs.iterrows():
        i, j = sorted((pos[r.col_a], pos[r.col_b]))
        M[j, i] = 100 * r.err_mean                       # lower triangle: row = later column
    assert np.isfinite(M[np.tril_indices(n, -1)]).all(), "missing pairs"
    return M


# ---------------------------------------------------------------------------
# Figure 6: pair Trend error
# ---------------------------------------------------------------------------
def _seq_text(v: float) -> str:
    return f"{v:.0f}" if round(abs(v), 1) >= 10 else f"{v:.1f}"


def _diff_text(v: float) -> str:
    return S.tnum(v, 1)


def draw_matrix(ax, M, cmap, norm, labels, annotate, fmt, show_x=True, show_y=True, tick_fs=S.FS_TICK):
    """Lower triangle without its empty first row / last column: rows labels[1:], columns labels[:-1]."""
    n = len(labels) - 1
    T = np.ma.masked_invalid(M[1:, :-1])
    ax.pcolormesh(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5, T, cmap=cmap, norm=norm, edgecolors="white",
                  linewidth=0.4)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(range(n), labels[:-1] if show_x else [], rotation=90, fontsize=tick_fs)
    ax.set_yticks(range(n), labels[1:] if show_y else [], fontsize=tick_fs)
    ax.tick_params(length=0, pad=1.5)
    if annotate:
        for i in range(n):
            for j in range(i + 1):
                v = T[i, j]
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=S.FS_CELL,
                        color=S.text_color_on(plt.get_cmap(cmap)(norm(v))))


def pair_caption(ds: str, arms: list[str], rule: str, appendix: bool) -> str:
    cols = columns(ds)
    short = [f"``target'' = {k}" for k, v in SHORT_COL.items() if any(c["name"] == k for c in cols)]
    return (rf"Pair-wise Trend error on {S.ds_label(ds)}: for every pair of columns, "
            r"$100\,(1-\text{score})$ of the Column Pair Trends metric that TabDiff's evaluation computes "
            r"(correlation similarity for two numerical columns, contingency similarity otherwise), averaged "
            r"over the 20 sampled tables; lower is better. Panels: "
            + ", ".join(f"({'abcde'[i]}) {S.rule_label(a)}" for i, a in enumerate(arms))
            + rf", and ({'abcde'[len(arms)]}) the difference {S.rule_label(rule)} $-$ {S.rule_label('fixed')} in "
            r"percentage points (blue: the adaptive rule is better). Each panel's subtitle gives the mean over all "
            r"pairs, which equals the run's Trend error. Columns are in TabDiff's evaluation order, numerical "
            r"first, categorical columns marked (c)"
            + ("; " + "; ".join(short) if short else "")
            + (r". Values $\geq 10$ are rounded to integers" if ds in ANNOTATE else "")
            + r". Route A, training seed 0, 20 sampled tables.")


def figure_pairs(ds: str, D: dict) -> tuple[float, float]:
    cols = columns(ds)
    labels = [col_label(c) for c in cols]
    arms = [a for a in F.ARMS if a in D]
    mats = {a: pair_matrix(D[a]["pairs"], cols) for a in arms}
    tri = np.tril_indices(len(cols), -1)
    for a in arms:                                     # panel mean = the run's Trend error
        assert abs(np.nanmean(mats[a][tri]) - D[a]["all_results"]["trend"].mean()) < 1e-9, (ds, a)
    rule = DIFF_RULE[ds]
    Dm = mats[rule] - mats["fixed"]
    seq_norm = Normalize(0, max(np.nanmax(m) for m in mats.values()))
    lim = float(np.nanmax(np.abs(Dm)))
    div_norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    annotate = ds in ANNOTATE
    big = len(cols) > 15
    tick_fs = S.FS_CELL if big else S.FS_TICK
    # layout: rule panels on a 3-column grid, the difference panel larger (it carries signed numbers)
    if len(arms) == 4:      # beijing, magic: Fixed | Jenks | entmax-1.5 / Dense | difference (2 columns wide)
        fig = S.figure(S.FULL, {"beijing": 5.7}.get(ds, 5.35))   # magic's labels are shorter
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1, 1.08])
        slots = {"fixed": gs[0, 0], "jenks": gs[0, 1], "entmax15": gs[0, 2], "dense": gs[1, 0], "diff": gs[1, 1:3]}
        bars = (gs[0, 3], gs[1, 3])
        no_x = {"fixed", "jenks", "entmax15"}
        y_on = {"fixed", "dense", "diff"}
    else:                   # default (24 columns, no numbers): Fixed | Jenks / entmax-1.5 | difference
        fig = S.figure(S.FULL, 6.5)                    # height chosen so all four squares are equal
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05])
        slots = {"fixed": gs[0, 0], "jenks": gs[0, 1], "entmax15": gs[1, 0], "diff": gs[1, 1]}
        bars = (gs[0, 2], gs[1, 2])
        no_x = {"fixed", "jenks"}
        y_on = {"fixed", "entmax15"}
    for k, key in enumerate(arms + ["diff"]):
        ax = fig.add_subplot(slots[key])
        ax.set_anchor("N")                             # panels of a row start at the same height
        show_x, show_y = key not in no_x, key in y_on
        if key == "diff":
            draw_matrix(ax, Dm, S.DIV_CMAP_R, div_norm, labels, annotate, _diff_text, show_x, show_y, tick_fs)
            S.title2(ax, f"{S.rule_label(rule)} − {S.rule_label('fixed')}",
                     f"mean = {S.tnum(np.nanmean(Dm[tri]), 3, sign=True)} pp")
        else:
            draw_matrix(ax, mats[key], S.SEQ_CMAP, seq_norm, labels, annotate, _seq_text, show_x, show_y, tick_fs)
            S.title2(ax, S.rule_label(key), f"mean = {np.nanmean(mats[key][tri]):.2f} %")
        S.panel_label(ax, "abcde"[k])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=seq_norm, cmap=S.SEQ_CMAP), cax=fig.add_subplot(bars[0]))
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=div_norm, cmap=S.DIV_CMAP_R), cax=fig.add_subplot(bars[1]))
    cb1.set_label("Pair Trend error (%)", fontsize=S.FS_TICK)
    cb2.set_label(f"{S.rule_label(rule)} − {S.rule_label('fixed')} (pp)", fontsize=S.FS_TICK)
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=S.FS_TICK, length=2, width=S.LW_AXIS)
        cb.outline.set_linewidth(S.LW_AXIS)
    name = f"pair_trend_error_{ds}"
    return S.save(fig, name, pair_caption(ds, arms, rule, ds != "beijing"), allow=set(labels))


# ---------------------------------------------------------------------------
# Figure 7: marginals as differences (beijing pm2.5, DEWP)
# ---------------------------------------------------------------------------
def _numeric(D: dict, arm: str, col: str):
    m = D[arm]["marg"]
    m = m[m.column == col]
    edges = np.r_[m.bin_left.to_numpy(float), m.bin_right.to_numpy(float)[-1]]
    return edges, m.real.to_numpy(float), m.syn.to_numpy(float)


def heavy_tailed(edges, dens) -> bool:
    """Right-skewed enough for a log x axis: half the mass in the first fifth of the window."""
    mass = dens * np.diff(edges)
    return edges[0] > 0 and mass[: len(mass) // 5].sum() >= 0.5 * mass.sum()


def figure_marginals(D: dict, ds: str = "beijing") -> tuple[float, float]:
    arms = [a for a in F.ARMS if a in D]
    fig = S.figure(S.FULL, 3.75)
    gs = fig.add_gridspec(2, len(MARG_COLS), height_ratios=[1.35, 1])
    logs = []
    sc = 1e3                                            # densities shown in units of 1e-3
    for c, col in enumerate(MARG_COLS):
        top = fig.add_subplot(gs[0, c])
        bot = fig.add_subplot(gs[1, c], sharex=top)
        S.style_axes(top)
        S.style_axes(bot)
        edges, real, _ = _numeric(D, arms[0], col)
        x = edges
        top.fill_between(x, sc * np.r_[real, real[-1]], step="post", color="#d9d9d9", lw=0, label="Real")
        bot.axhline(0, color=S.INK2, lw=S.LW_AXIS)
        for a in arms:
            e, r_, syn = _numeric(D, a, col)
            assert np.allclose(e, edges) and np.allclose(r_, real), (a, col)
            err = 100 * D[a]["shapes"].set_index("column").err_mean[col]
            kw = dict(color=S.RULE_COLOR[a], lw=0.9, ls="--" if a == "dense" else "-")
            top.step(x, sc * np.r_[syn, syn[-1]], where="post", label=f"{S.rule_label(a)} ({err:.2f} %)", **kw)
            d = sc * (syn - real)
            bot.step(x, np.r_[d, d[-1]], where="post", **kw)
        if heavy_tailed(edges, real):
            top.set_xscale("log")
            logs.append(col)
        top.set_xlim(edges[0], edges[-1])
        top.set_ylim(bottom=0)
        top.tick_params(labelbottom=False)
        top.set_ylabel(r"Density ($\times10^{-3}$)")
        bot.set_ylabel(r"Syn. $-$ real ($\times10^{-3}$)")
        bot.set_xlabel(f"{col}" + (" (log scale)" if col in logs else ""))
        lim = max(abs(np.array(bot.get_ylim())))
        bot.set_ylim(-lim, lim)
        h, l_ = top.get_legend_handles_labels()
        bot.legend(h, l_, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=S.FS_CELL + 0.5,
                   handlelength=1.6, columnspacing=1.0, labelspacing=0.3)
        S.title2(top, f"{S.ds_label(ds)}: {col}")
        S.panel_label(top, "ab"[c])
    sel = ", ".join(MARG_COLS)
    cap = (rf"Marginal distributions of two numerical columns of {S.ds_label(ds)} ({sel}) under each rule. Top: "
           r"density of the real training data (grey) and of the synthetic data, pooled over the 20 sampled "
           r"tables (lines); bottom: synthetic minus real density per bin, where the differences behind the "
           r"Shape error become visible. Bins: 50 equal-width bins between the real 0.5\% and 99.5\% quantiles; "
           r"both densities are counts divided by (all rows $\times$ bin width), so mass outside the window "
           r"lowers the curve instead of being renormalised. The percentage after each rule in the legend is "
           r"that rule's Shape error of the column ($100\,(1-\text{KS complement})$, mean over the 20 tables)."
           + (f" The x axis of {', '.join(logs)} is logarithmic (heavy right tail)." if logs else "")
           + r" Route A, training seed 0, 20 sampled tables.")
    return S.save(fig, "marginals", cap, allow=set(MARG_COLS) | {f"{S.ds_label(ds)}: {c}" for c in MARG_COLS})


# ---------------------------------------------------------------------------
# Appendix: the automatically chosen columns of beijing and default
# ---------------------------------------------------------------------------
def pick_columns(ds: str, D: dict) -> tuple[list[tuple[str, str]], str]:
    """[(column, why)] and the best rule (lowest mean Shape error)."""
    rule = DIFF_RULE[ds]
    sh = {a: D[a]["shapes"].set_index("column") for a in D}
    best = min(sh, key=lambda a: sh[a].err_mean.mean())
    num = sh["fixed"][sh["fixed"].type == "numerical"].index
    gap = (sh[rule].err_mean[num] - sh["fixed"].err_mean[num]).abs().sort_values(ascending=False)
    picked = [(c, f"|Δ err| {F.RA.ARM_LABEL[rule]} − fixed = {100 * gap[c]:.2f} pp") for c in gap.index[:2]]
    for c in sh[best].err_mean.sort_values(ascending=False).index:
        if len(picked) == 4:
            break
        if c not in [p[0] for p in picked]:
            picked.append((c, f"largest error under {F.RA.ARM_LABEL[best]}: {100 * sh[best].err_mean[c]:.2f} %"))
    return picked, best


def _pow10(v: float) -> int:
    """Exponent to factor out of an axis whose values reach v (0 if no scaling is needed)."""
    e = int(np.floor(np.log10(abs(v)))) if v else 0
    return e if (e >= 4 or e <= -3) else 0


def _scaled(label: str, e: int) -> str:
    return label + (rf" ($\times10^{{{e}}}$)" if e else "")


def figure_marginals_appendix(data: dict):
    rows = [ds for ds in APPX_DS if data[ds]]
    fig = S.figure(S.FULL, 2.0 * 2 * len(rows))
    gs = fig.add_gridspec(2 * len(rows), 2)
    sel, allow, k = [], set(), 0
    for r, ds in enumerate(rows):
        D = data[ds]
        picked, best = pick_columns(ds, D)
        arms = [a for a in F.ARMS if a in D]
        for c, (col, why) in enumerate(picked):
            ax = fig.add_subplot(gs[2 * r + c // 2, c % 2])
            S.style_axes(ax)
            ref = D[arms[0]]["marg"]
            m0 = ref[ref.column == col]
            typ = m0.type.iloc[0]
            errs = {a: 100 * D[a]["shapes"].set_index("column").err_mean[col] for a in arms}
            sel.append({"dataset": ds, "column": col, "type": typ, "why": why, "best_rule": best,
                        **{f"shape_err_{a}": errs[a] for a in arms}})
            if typ == "numerical":
                edges = np.r_[m0.bin_left.to_numpy(float), m0.bin_right.to_numpy(float)[-1]]
                ex = _pow10(np.abs(edges).max())
                ey = _pow10(m0.real.max())
                xs, ys = 10.0 ** -ex, 10.0 ** -ey
                ax.fill_between(xs * edges, ys * np.r_[m0.real.to_numpy(float), m0.real.iloc[-1]], step="post",
                                color="#d9d9d9", lw=0)
                for a in arms:
                    m = D[a]["marg"]
                    m = m[m.column == col]
                    ax.step(xs * edges, ys * np.r_[m.syn.to_numpy(float), m.syn.iloc[-1]], where="post",
                            color=S.RULE_COLOR[a], lw=0.9, ls="--" if a == "dense" else "-")
                ax.set_xlim(xs * edges[0], xs * edges[-1])
                ax.set_ylim(bottom=0)
                ax.set_ylabel(_scaled("Density", ey))
            else:
                cats = list(m0.category.astype(str))
                try:                                   # natural order for numeric codes (hour, day, ...)
                    cats = sorted(cats, key=float)
                except ValueError:
                    pass
                x = np.arange(len(cats))
                real = m0.set_index(m0.category.astype(str)).real
                ax.bar(x, [float(real[cc]) for cc in cats], 0.8, color="#d9d9d9", lw=0)
                for a in arms:
                    m = D[a]["marg"]
                    m = m[m.column == col].set_index(m[m.column == col].category.astype(str))
                    ax.plot(x, [float(m.syn.get(cc, 0.0)) for cc in cats], ls="none", marker="_", ms=4.5, mew=1.0,
                            color=S.RULE_COLOR[a], mec=S.RULE_COLOR[a])
                step = max(1, int(np.ceil(len(cats) / 12)))
                ax.set_xticks(x[::step], [cats[i] for i in range(0, len(cats), step)])
                allow |= {cats[i] for i in range(0, len(cats), step)}
                ax.set_xlim(-0.7, len(cats) - 0.3)
                ax.set_ylabel("Frequency")
            xlabel = col + " (categories)" if typ == "categorical" else _scaled(col, ex)
            ax.set_xlabel(xlabel)
            allow.add(xlabel)
            parts = [f"{S.rule_label(a).split(' (')[0]} {errs[a]:.2f}" for a in arms]
            sub = ", ".join(parts[:2]) + ",\n" + ", ".join(parts[2:]) if len(parts) > 2 else ", ".join(parts)
            title = f"{S.ds_label(ds)}: {col}"
            allow.add(title)
            S.title2(ax, title, "Shape error (%): " + sub)
            S.panel_label(ax, "abcdefgh"[k])
            k += 1
    handles = [plt.Rectangle((0, 0), 1, 1, color="#d9d9d9", label="Real")] + \
              [plt.Line2D([], [], color=S.RULE_COLOR[a], lw=1.2, ls="--" if a == "dense" else "-",
                          label=S.rule_label(a)) for a in F.ARMS if any(a in data[d] for d in rows)]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    Sdf = pd.DataFrame(sel)
    listing = "; ".join(f"{S.ds_label(d)}: " + ", ".join(Sdf[Sdf.dataset == d].column) for d in rows)
    best = ", ".join(f"{S.ds_label(d)} {S.rule_label(Sdf[Sdf.dataset == d].best_rule.iloc[0])}" for d in rows)
    cap = (r"Marginals of automatically chosen columns (" + listing.replace("_", r"\_") + r"). Per dataset: the "
           r"two numerical columns whose Shape error differs most between the adaptive rule and the fixed rule "
           r"(entmax-1.5 on Beijing, Jenks on Default), then the two other columns with the largest Shape error "
           rf"under the best rule (lowest mean Shape error: {best}). Numerical columns: real density (grey) and "
           r"each rule's synthetic density pooled over the 20 tables, 50 bins between the real 0.5\% and 99.5\% "
           r"quantiles, counts divided by (all rows $\times$ bin width). Categorical columns: real frequency "
           r"(grey bars) and each rule's mean frequency over the 20 tables (ticks). Subtitles: Shape error of the "
           r"column per rule. Route A, training seed 0, 20 sampled tables.")
    size = S.save(fig, "marginals_appendix", cap, allow=allow)
    Sdf.to_csv(F.CSV / "marginals_selected_columns.csv", index=False)
    return size, Sdf


def main():
    data = {ds: load(ds) for ds in sorted(set(PAIR_DS) | set(APPX_DS))}
    if not any(data.values()):
        sys.exit("no <run>/analysis/ outputs on disk: run hpc_transfer/jobs_tabdiff_gnn/analysis_pair_marginals.sh "
                 "and mirror the results first")
    for ds, D in data.items():
        for arm, d in D.items():
            dev = d["meta"]["self_check_max_abs_dev"]
            print(f"  {ds}/{arm}: self-checks max |diff| trend {dev['trend']:.1e}, shape {dev['shape']:.1e} "
                  f"({d['meta']['n_tables']} tables, sdmetrics {d['meta']['versions'].get('sdmetrics')})")
    for ds in PAIR_DS:
        if data[ds]:
            print(f"wrote pair_trend_error_{ds} {figure_pairs(ds, data[ds])}")
    if data["beijing"]:
        print(f"wrote marginals {figure_marginals(data['beijing'])}")
    size, Sdf = figure_marginals_appendix(data)
    print(Sdf[["dataset", "column", "type", "why"]].to_string(index=False))
    print(f"wrote marginals_appendix {size}, csv/marginals_selected_columns.csv")


if __name__ == "__main__":
    main()

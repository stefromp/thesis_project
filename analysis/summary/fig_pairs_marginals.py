#!/usr/bin/env python3
"""
Figures 6-7 -- where the Trend / Shape errors sit (Route A, seed 0), from the per-run CSVs
written by analysis/tabdiff/pair_and_marginals.py (<run>/analysis/, mirrored from the cluster).

    python3 analysis/summary/fig_pairs_marginals.py

Figure 6 (pair_trend_error): per dataset (beijing, default, magic) one lower-triangle heatmap
  per arm of the pair Trend error 100 x (1 - score_ab), mean over the 20 tables, columns in
  TabDiff's evaluation order (numerical block, then categorical; info.json), one colour scale per
  row; the last panel is the difference (entmax-1.5 - fixed on beijing and magic, Jenks - fixed on
  default). Panel title = mean over pairs = the run's Trend error (checked against
  tabdiff_eval/all_results.csv).
Figure 7 (marginals): beijing and default, 4 columns each, chosen from column_shape_error.csv:
  the 2 numerical columns with the largest |err(rule) - err(fixed)| (rule = entmax-1.5 on
  beijing, Jenks on default) and the 2 other columns (any type) with the largest error under the
  best rule (= the arm with the lowest mean Shape error). Real = filled grey; each arm a line
  (numerical) or a bar (categorical).
Each run's pair_marginals_meta.json must record self-checks within 1e-6 (else this stops).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import route_a_figs as F  # noqa: E402

np, pd, plt, ct = F.np, F.pd, F.plt, F.ct
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(F.BS.REPO / "analysis" / "tabdiff"))
import pair_and_marginals as PM  # noqa: E402  (eval_columns: TabDiff's column order and typing)

FIG6_DS = ["beijing", "default", "magic"]
FIG7_DS = ["beijing", "default"]
DIFF_RULE = {"beijing": "entmax15", "default": "jenks", "magic": "entmax15"}
ANNOTATE = {"beijing"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#6250d6"])          # one hue, light -> dark
DIV = LinearSegmentedColormap.from_list("div", [ct.C_DIV_POS, ct.C_DIV_MID, ct.C_DIV_NEG])  # + = worse


def columns(ds: str) -> list[dict]:
    return PM.eval_columns(json.loads(Path(F.RC.data_files(F.RC.target_name(ds))["info"]).read_text()))


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


def draw_matrix(ax, M, cmap, norm, names, title, annotate):
    """Lower triangle without its empty first row / last column: rows names[1:], columns names[:-1]."""
    n = len(names) - 1
    T = M[1:, :-1]
    ax.imshow(np.ma.masked_invalid(T), cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(n), names[:-1], rotation=90, fontsize=5.5 if n > 15 else 7)
    ax.set_yticks(range(n), names[1:], fontsize=5.5 if n > 15 else 7)
    ax.tick_params(length=0, labelcolor=ct.C_TEXT)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        for i in range(n):
            for j in range(i + 1):
                v = T[i, j]
                r, g, b, _ = cmap(norm(v))
                ax.text(j, i, f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}", ha="center", va="center", fontsize=4.8,
                        color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else ct.C_TEXT)
    ax.set_title(title, loc="left", fontsize=8.5, color=ct.C_TEXT)


def figure6(data: dict):
    rows = [ds for ds in FIG6_DS if data[ds]]
    # columns: 4 arm panels | colour bar | difference panel | colour bar
    fig = plt.figure(figsize=(21, 4.6 * len(rows)), layout="constrained")
    gs = fig.add_gridspec(len(rows), 7, width_ratios=[1, 1, 1, 1, 0.05, 1, 0.05])
    summary = []
    for r, ds in enumerate(rows):
        cols, D = columns(ds), data[ds]
        names = [c["name"] + ("" if c["type"] == "numerical" else " ᶜ") for c in cols]
        mats = {arm: pair_matrix(D[arm]["pairs"], cols) for arm in D}
        vmax = max(np.nanmax(m) for m in mats.values())
        norm = plt.Normalize(0.0, vmax)
        arms = [a for a in F.ARMS if a in mats]
        for c, arm in enumerate(arms):
            M = mats[arm]
            mean = np.nanmean(M[np.tril_indices(len(cols), -1)])
            want = D[arm]["all_results"]["trend"].mean()
            assert abs(mean - want) < 1e-9, (ds, arm, mean, want)       # panel mean = the run's Trend error
            summary.append((ds, arm, mean))
            draw_matrix(fig.add_subplot(gs[r, c]), M, SEQ, norm, names,
                        f"{ds} — {F.ARM_LABEL[arm]}\nmean = Trend err {mean:.3f} %", ds in ANNOTATE)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=fig.add_subplot(gs[r, 4]))
        cb.set_label("pair Trend error (%)", fontsize=7, color=ct.C_TEXT2)
        cb.ax.tick_params(labelsize=6, colors=ct.C_TEXT2)
        rule = DIFF_RULE[ds]
        if rule in mats and "fixed" in mats:
            Dm = mats[rule] - mats["fixed"]
            lim = np.nanmax(np.abs(Dm))
            dn = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
            dmean = np.nanmean(Dm[np.tril_indices(len(cols), -1)])
            draw_matrix(fig.add_subplot(gs[r, 5]), Dm, DIV, dn, names,
                        f"{ds} — {F.ARM_LABEL[rule]} − fixed\nmean = {dmean:+.3f} pp", ds in ANNOTATE)
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=dn, cmap=DIV), cax=fig.add_subplot(gs[r, 6]))
            cb.set_label(f"{F.ARM_LABEL[rule]} − fixed (pp); red = rule worse", fontsize=7, color=ct.C_TEXT2)
            cb.ax.tick_params(labelsize=6, colors=ct.C_TEXT2)
    absent = [ds for ds in FIG6_DS if not data[ds]]
    F.caption(fig, "Route A, training seed 0. Each cell = Trend error of one column pair, 100 × (1 − score) of "
                   "sdmetrics' Column Pair Trends inside TabDiff's own evaluate_density (CorrelationSimilarity for "
                   "two numerical columns, ContingencySimilarity otherwise), mean over the 20 tables; columns in "
                   "TabDiff's evaluation order (numerical block, then categorical, ᶜ). Panel mean = the run's Trend "
                   "error (equals all_results.csv). One colour scale per dataset row; difference panel: pp, red = "
                   "the adaptive rule worse than fixed. Values printed for beijing."
              + (f" Not shown (no analysis outputs on disk): {', '.join(absent)}." if absent else ""), y=0.0)
    F.savefig(fig, "pair_trend_error")
    return summary


def pick_columns(ds: str, D: dict) -> tuple[list[tuple[str, str]], str]:
    """[(column, why)] and the best rule (lowest mean Shape error)."""
    rule = DIFF_RULE[ds]
    sh = {a: D[a]["shapes"].set_index("column") for a in D}
    best = min(sh, key=lambda a: sh[a].err_mean.mean())
    num = sh["fixed"][sh["fixed"].type == "numerical"].index
    gap = (sh[rule].err_mean[num] - sh["fixed"].err_mean[num]).abs().sort_values(ascending=False)
    picked = [(c, f"|Δ err| {F.ARM_LABEL[rule]} − fixed = {100 * gap[c]:.2f} pp") for c in gap.index[:2]]
    for c in sh[best].err_mean.sort_values(ascending=False).index:
        if len(picked) == 4:
            break
        if c not in [p[0] for p in picked]:
            picked.append((c, f"largest error under {F.ARM_LABEL[best]}: {100 * sh[best].err_mean[c]:.2f} %"))
    return picked, best


def figure7(data: dict):
    rows = [ds for ds in FIG7_DS if data[ds]]
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 3.7 * len(rows)), squeeze=False)
    sel = []
    for r, ds in enumerate(rows):
        D = data[ds]
        picked, best = pick_columns(ds, D)
        arms = [a for a in F.ARMS if a in D]
        for c, (col, why) in enumerate(picked):
            ax = axes[r, c]
            F.style_ax(ax)
            ref = D[arms[0]]["marg"]
            m0 = ref[ref.column == col]
            typ = m0.type.iloc[0]
            errs = {a: 100 * D[a]["shapes"].set_index("column").err_mean[col] for a in arms}
            sel.append({"dataset": ds, "column": col, "type": typ, "why": why, "best_rule": best,
                        **{f"shape_err_{a}": errs[a] for a in arms}})
            if typ == "numerical":
                edges = np.r_[m0.bin_left.to_numpy(float), m0.bin_right.to_numpy(float)[-1]]
                ax.fill_between(edges, np.r_[m0.real.to_numpy(float), m0.real.iloc[-1]], step="post",
                                color=ct.C_GRID, label="real", lw=0)
                for a in arms:
                    m = D[a]["marg"]
                    m = m[m.column == col]
                    assert np.allclose(m.bin_left.to_numpy(float), m0.bin_left.to_numpy(float))
                    ax.step(edges, np.r_[m.syn.to_numpy(float), m.syn.iloc[-1]], where="post", color=F.C_ARM[a],
                            lw=1.4, ls="--" if a == "dense" else "-", label=F.ARM_LABEL[a])
                ax.set_ylabel("density", fontsize=8, color=ct.C_TEXT2)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
                ax.set_xlabel(col + " (0.5–99.5 % real quantiles, 50 bins)", fontsize=8, color=ct.C_TEXT2)
            else:
                cats = list(m0.category.astype(str))
                try:                                   # natural order for numeric codes (hour, day, ...)
                    cats = sorted(cats, key=float)
                except ValueError:
                    pass
                x = np.arange(len(cats))
                w = 0.8 / (len(arms) + 1)
                real = m0.set_index(m0.category.astype(str)).real
                ax.bar(x - 0.4 + w / 2, [float(real[cc]) for cc in cats], w, color=ct.C_GRID, label="real")
                for k, a in enumerate(arms, start=1):
                    m = D[a]["marg"]
                    m = m[m.column == col].set_index(m[m.column == col].category.astype(str))
                    ax.bar(x - 0.4 + w / 2 + k * w, [float(m.syn.get(cc, 0.0)) for cc in cats], w,
                           color=F.C_ARM[a], label=F.ARM_LABEL[a])
                ax.set_xticks(x, cats, rotation=90 if len(cats) > 8 else 0, fontsize=6 if len(cats) > 12 else 7)
                ax.set_ylabel("frequency", fontsize=8, color=ct.C_TEXT2)
                ax.set_xlabel(col + " (categories)", fontsize=8, color=ct.C_TEXT2)
            et = " · ".join(f"{F.ARM_LABEL[a].split(' (')[0]} {errs[a]:.2f}" for a in arms)
            ax.set_title(f"{ds}: {col} ({typ})\n{why}\nShape err %: {et}", loc="left", fontsize=8, color=ct.C_TEXT)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ct.C_GRID, label="real")] + \
              [plt.Line2D([], [], color=F.C_ARM[a], lw=2, ls="--" if a == "dense" else "-", label=F.ARM_LABEL[a])
               for a in F.ARMS if any(a in data[d] for d in rows)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.03))
    S = pd.DataFrame(sel)
    listing = "; ".join(f"{d}: " + ", ".join(S[S.dataset == d].column) for d in rows)
    F.caption(fig, "Route A, training seed 0. Columns chosen automatically from column_shape_error.csv — " + listing +
                   ": per dataset the 2 numerical columns with the largest |Shape error(rule) − Shape error(fixed)| "
                   "(rule = entmax-1.5 on beijing, Jenks on default), then the 2 other columns with the largest "
                   "Shape error under the best rule (lowest mean Shape error: " +
                   ", ".join(f"{d} {F.ARM_LABEL[S[S.dataset == d].best_rule.iloc[0]]}" for d in rows) + "). "
                   "Numerical: 50 bins between the real 0.5 % and 99.5 % quantiles, real counted once, synthetic "
                   "summed over the 20 tables, both divided by (all rows × bin width). Categorical: frequencies, "
                   "synthetic = mean over the 20 tables. Shape error = 100 × (1 − KS / TV complement), mean over "
                   "the 20 tables.", y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    F.savefig(fig, "marginals")
    S.to_csv(F.CSV / "marginals_selected_columns.csv", index=False)
    return S


def main():
    data = {ds: load(ds) for ds in sorted(set(FIG6_DS) | set(FIG7_DS))}
    if not any(data.values()):
        sys.exit("no <run>/analysis/ outputs on disk: run hpc_transfer/jobs_tabdiff_gnn/analysis_pair_marginals.sh "
                 "and mirror the results first")
    for ds, D in data.items():
        for arm, d in D.items():
            dev = d["meta"]["self_check_max_abs_dev"]
            print(f"  {ds}/{arm}: self-checks max |diff| trend {dev['trend']:.1e}, shape {dev['shape']:.1e} "
                  f"({d['meta']['n_tables']} tables, sdmetrics {d['meta']['versions'].get('sdmetrics')})")
    for ds, arm, mean in figure6(data):
        print(f"  fig 6 {ds}/{arm}: mean pair error {mean:.4f} % = Trend error")
    S = figure7(data)
    print(S[["dataset", "column", "type", "why"]].to_string(index=False))
    print(f"wrote {F.BS.rel(F.FIG)}/pair_trend_error.{{pdf,png}}, marginals.{{pdf,png}}, "
          f"{F.BS.rel(F.CSV)}/marginals_selected_columns.csv")


if __name__ == "__main__":
    main()

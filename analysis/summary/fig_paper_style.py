#!/usr/bin/env python3
"""
Two figures that mirror the TabDiff paper's Figure 3 (marginal densities) and Figure 4 (pair-wise
correlation error heatmaps) -- same datasets, same columns, same colour scale -- with our
neighbour-selection rules in place of the paper's models (Route A, seed 0, 20 tables per run).

    python3 analysis/summary/fig_paper_style.py

paper_style_marginals      Beijing DEWP and Shoppers BounceRates: KDE of the real data and of each
                           rule's synthetic data (the 20 tables pooled); Adult marital.status, Beijing
                           month, Magic class, Shoppers OperatingSystems: category counts, real vs the
                           mean over the 20 tables.
paper_style_pair_heatmaps  Adult, Beijing, Magic, Shoppers x rules: the full symmetric matrix of the
                           pair Trend error 1 - score (fraction), mean over the 20 tables, from
                           <run>/analysis/pair_trend_error.csv (analysis/tabdiff/pair_and_marginals.py:
                           TabDiff's own evaluate_density, self-checked against all_results.csv);
                           Blues, 0 to 0.10 as in the paper.
Real data = data/tabdiff_gnn/tabdiff_<ds>/real.csv, the table TabDiff's evaluation compares against
(the one pair_and_marginals.py uses). Fixed = the main fixed run (Magic: k = 3, not the dense run).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import route_a_figs as F  # noqa: E402
import thesis_style as S  # noqa: E402

np, pd, plt = F.np, F.pd, F.plt
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.insert(0, str(F.BS.REPO / "analysis" / "tabdiff"))
import pair_and_marginals as PM  # noqa: E402

PAPER_DS = ["adult", "beijing", "magic", "shoppers"]
# the TabDiff paper's Figure 3 columns: (dataset, name as printed, expected type)
KDE_COLS = [("beijing", "DEWP", "numerical"), ("shoppers", "BounceRates", "numerical")]
BAR_COLS = [("adult", "marital.status", "categorical"), ("beijing", "month", "categorical"),
            ("magic", "class", "categorical"), ("shoppers", "OperatingSystems", "categorical")]
MAX_KDE_ROWS = 200_000
VMAX = 0.10                                          # the paper's colour-bar maximum
REAL_FILL, REAL_EDGE = "#d4d4d4", "#4d4d4d"
SPLIT = "#c0392b"                                     # Num / Cat separators (red, as in the paper)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[.\-_\s]", "", s).lower()


def columns(ds: str) -> list[dict]:
    return PM.eval_columns(json.loads(Path(F.RC.data_files(F.RC.target_name(ds))["info"]).read_text()))


def resolve_column(ds: str, name: str, typ: str) -> dict:
    """The dataset's column matching `name` up to '.', '-', '_', spaces and case; must be unique and
    of type `typ` (TabDiff's typing, info.json)."""
    hits = [c for c in columns(ds) if _norm(c["name"]) == _norm(name)]
    assert len(hits) == 1, f"{ds}: column {name!r} matches {[c['name'] for c in hits]}"
    assert hits[0]["type"] == typ, f"{ds}: {hits[0]['name']} is {hits[0]['type']}, expected {typ}"
    return hits[0]


def arms_of(ds: str) -> list[str]:
    return [a for a in F.ARMS if F.run_dir(ds, a) is not None]


def real_column(ds: str, col: str) -> pd.Series:
    return pd.read_csv(F.RC.data_files(F.RC.target_name(ds))["real"], usecols=[col])[col]


def synthetic_columns(ds: str, arm: str, col: str) -> list[pd.Series]:
    paths, _ = PM.sample_paths(str(F.run_dir(ds, arm)), F.RC.N_TABLES)
    return [pd.read_csv(p, usecols=[col])[col] for p in paths]


def kde_pair(real: np.ndarray, syn: np.ndarray, grid: np.ndarray, seed: int = 0):
    """Densities of real and synthetic on `grid` with the SAME absolute bandwidth: Scott's rule on
    the real data, passed to the synthetic KDE as a factor of its own std. Synthetic rows are
    subsampled to MAX_KDE_ROWS (fixed seed) if needed. Returns (real, syn, bandwidth, n_syn used)."""
    kr = gaussian_kde(real)                                      # bw_method=None: Scott
    bw = float(np.sqrt(kr.covariance[0, 0]))
    if len(syn) > MAX_KDE_ROWS:
        syn = np.random.default_rng(seed).choice(syn, MAX_KDE_ROWS, replace=False)
    ks = gaussian_kde(syn, bw_method=bw / syn.std(ddof=1))
    assert np.isclose(np.sqrt(ks.covariance[0, 0]), bw, rtol=1e-9), "bandwidths differ"
    return kr(grid), ks(grid), bw, len(syn)


def _cat_key(v) -> str:
    """Category label as text; numeric codes without a trailing .0, surrounding blanks stripped."""
    s = str(v).strip()
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else s
    except ValueError:
        return s


def category_counts(real: pd.Series, syns: list[pd.Series]) -> tuple[list[str], np.ndarray, np.ndarray, str]:
    """(categories in natural order, real counts, synthetic counts = mean over the tables, note).
    A synthetic table of another size than the real one is scaled to the real size first."""
    r = real.map(_cat_key).value_counts()
    tabs, sizes = [], set()
    for s in syns:
        sizes.add(len(s))
        tabs.append(s.map(_cat_key).value_counts() * (len(real) / len(s)))
    cats = sorted(set(r.index).union(*[t.index for t in tabs]))
    try:
        cats = sorted(cats, key=float)
    except ValueError:
        pass
    syn = np.mean([[float(t.get(c, 0.0)) for c in cats] for t in tabs], axis=0)
    note = ("synthetic tables have the real table's size" if sizes == {len(real)} else
            f"synthetic tables of sizes {sorted(sizes)} scaled to the real size {len(real)}")
    return cats, np.array([float(r.get(c, 0.0)) for c in cats]), syn, note


def full_pair_matrix(ds: str, arm: str) -> tuple[np.ndarray, int]:
    """Symmetric matrix of 1 - score (fraction) in TabDiff's column order, NaN on the diagonal, and
    the number of numerical columns."""
    cols = columns(ds)
    pos = {c["name"]: c["pos"] for c in cols}
    n = len(cols)
    M = np.full((n, n), np.nan)
    p = pd.read_csv(F.run_dir(ds, arm) / PM.OUT_DIR / "pair_trend_error.csv")
    for _, r in p.iterrows():
        i, j = pos[r.col_a], pos[r.col_b]
        M[i, j] = M[j, i] = r.err_mean
    off = ~np.eye(n, dtype=bool)
    assert np.isfinite(M[off]).all(), (ds, arm, "missing pairs")
    return M, sum(c["type"] == "numerical" for c in cols)


def trend_mean(ds: str, arm: str) -> float:
    """The run's Trend error / 100 (mean over the 20 tables of all_results.csv)."""
    return float(pd.read_csv(F.run_dir(ds, arm) / "tabdiff_eval" / "all_results.csv", index_col=0)["trend"].mean()) / 100


# ---------------------------------------------------------------------------
# Figure: marginals (TabDiff Fig. 3)
# ---------------------------------------------------------------------------
def _thousands(v, _):
    return "0" if v == 0 else (f"{v / 1000:g}k" if abs(v) >= 1000 else f"{v:g}")


def figure_marginals() -> dict:
    kde_cols = [(ds, resolve_column(ds, c, t)["name"]) for ds, c, t in KDE_COLS]
    bar_cols = [(ds, resolve_column(ds, c, t)["name"]) for ds, c, t in BAR_COLS]
    fig = S.figure(S.FULL, 5.2)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.25])
    info, letter = {"kde": {}, "bars": {}}, iter("abcdefghijklmn")
    for r, (ds, col) in enumerate(kde_cols):
        real = real_column(ds, col).to_numpy(float)
        real = real[np.isfinite(real)]
        lo, hi = np.quantile(real, [0.005, 0.995])
        grid = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 600)
        panels, top = [], 0.0
        for a in arms_of(ds):
            syn = np.concatenate([s.to_numpy(float) for s in synthetic_columns(ds, a, col)])
            syn = syn[np.isfinite(syn)]
            dr, dsyn, bw, n_used = kde_pair(real, syn, grid)
            panels.append((a, dr, dsyn))
            top = max(top, dr.max(), dsyn.max())
            info["kde"][(ds, col, a)] = {"bandwidth": bw, "n_syn": len(syn), "n_syn_kde": n_used}
        axes = []
        for c, (a, dr, dsyn) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c], sharey=axes[0] if axes else None)
            S.style_axes(ax)
            ax.fill_between(grid, dr, color=REAL_FILL, lw=0)
            ax.plot(grid, dr, color=REAL_EDGE, lw=0.6)
            ax.fill_between(grid, dsyn, color=S.RULE_COLOR[a], alpha=0.25, lw=0)
            ax.plot(grid, dsyn, color=S.RULE_COLOR[a], lw=S.LW_DATA)
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(0, 1.45 * top)                     # headroom for the in-panel legend
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            if c:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel("Density")
            handles = [plt.Rectangle((0, 0), 1, 1, fc=REAL_FILL, ec=REAL_EDGE, lw=0.6),
                       plt.Rectangle((0, 0), 1, 1, fc=S.desaturate(S.RULE_COLOR[a], 1.0, 0.75), ec=S.RULE_COLOR[a],
                                     lw=1.0)]
            peak_right = grid[np.argmax(np.maximum(dr, dsyn))] > (grid[0] + grid[-1]) / 2
            ax.legend(handles, ["Real data", S.rule_label(a)], loc="upper left" if peak_right else "upper right",
                      fontsize=S.FS_CELL + 0.5, handlelength=1.0, handleheight=0.8, borderaxespad=0.15,
                      labelspacing=0.2)
            ax.set_title(f"{S.ds_label(ds)}: {col}", loc="left", fontsize=S.FS_SUB)
            S.panel_label(ax, next(letter))
            axes.append(ax)
    for c, (ds, col) in enumerate(bar_cols):
        ax = fig.add_subplot(gs[2, c])
        S.style_axes(ax)
        arms = arms_of(ds)
        syns = {a: synthetic_columns(ds, a, col) for a in arms}
        real = real_column(ds, col)
        series = []
        for a in arms:
            cats, rc, sc, note = category_counts(real, syns[a])
            assert abs(rc.sum() - len(real)) < 1e-9 and abs(sc.sum() - len(real)) < 1e-6, (ds, col, a)
            series.append((a, cats, rc, sc))
            info["bars"][(ds, col, a)] = {"note": note, "categories": cats}
        cats = series[0][1]
        assert all(s[1] == cats for s in series), (ds, col, "category sets differ between rules")
        k = len(series) + 1
        w = 0.84 / k
        x = np.arange(len(cats))
        ax.bar(x - 0.42 + w / 2, series[0][2], w, color=REAL_FILL, ec=REAL_EDGE, lw=0.3)
        for i, (a, _, _, sc) in enumerate(series, start=1):
            ax.bar(x - 0.42 + w / 2 + i * w, sc, w, color=S.RULE_COLOR[a], lw=0)
        long = max(len(c) for c in cats) > 3
        ax.set_xticks(x, cats, rotation=60 if long else 0, ha="right" if long else "center",
                      rotation_mode="anchor", fontsize=S.FS_CELL if (long or len(cats) > 6) else S.FS_TICK)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_thousands))
        if c == 0:
            ax.set_ylabel("Count")
        ax.set_title(f"{S.ds_label(ds)}: {col}", loc="left", fontsize=S.FS_SUB)
        S.panel_label(ax, next(letter))
        info["bars"][(ds, col)] = cats
    used = [a for a in F.ARMS if any(k[2] == a for k in info["kde"] if len(k) == 3)]
    handles = [plt.Rectangle((0, 0), 1, 1, fc=REAL_FILL, ec=REAL_EDGE, lw=0.6, label="Real data")] + \
              [plt.Rectangle((0, 0), 1, 1, fc=S.RULE_COLOR[a], label=S.rule_label(a)) for a in used]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), handlelength=1.0)
    notes = {v["note"] for k, v in info["bars"].items() if isinstance(v, dict)}
    allow = {f"{S.ds_label(ds)}: {c}" for ds, c in kde_cols + bar_cols} | \
            {c for k, v in info["bars"].items() if isinstance(v, list) for c in v}
    info["size"] = S.save(fig, "paper_style_marginals", marginals_caption(kde_cols, bar_cols, notes, info),
                          allow=allow)
    info["resolved"] = {"kde": kde_cols, "bars": bar_cols}
    return info


def marginals_caption(kde_cols, bar_cols, notes, info) -> str:
    esc = lambda s: s.replace("_", r"\_")                     # noqa: E731
    kd = ", ".join(f"{S.ds_label(d)} {esc(c)}" for d, c in kde_cols)
    bd = ", ".join(f"{S.ds_label(d)} {esc(c)}" for d, c in bar_cols)
    sub = any(v["n_syn_kde"] < v["n_syn"] for v in info["kde"].values())
    return (r"Marginal distributions of the columns shown in Figure~3 of the TabDiff paper, for each of our "
            r"neighbour-selection rules. (a)--(g): kernel density estimates of two numerical columns "
            rf"({kd}), real data in grey and each rule's synthetic data in its colour; real and synthetic "
            r"densities in a panel use the same bandwidth (Scott's rule on the real data), and the "
            r"synthetic estimate pools the 20 sampled tables"
            + (rf" (subsampled to {MAX_KDE_ROWS:,} rows with a fixed seed)".replace(",", "{,}") if sub else "")
            + r"; the axis spans the real 0.5--99.5\% quantiles extended by 10\%, and the panels of a row "
            rf"share their $y$-axis. (h)--(k): category counts of four categorical columns ({bd}), real "
            r"table (grey) and each rule's mean over the 20 tables (" + "; ".join(sorted(notes)) + r"). "
            r"Layout and columns follow the TabDiff paper so that the rules can be set beside its "
            r"Figure~3, but the paper's panels show other models on other samples: the comparison is "
            r"visual and qualitative only. Fixed top-$k$ on Magic is the $k=3$ run; the dense ablation "
            r"($k=N-1$) exists for Beijing and Magic only. Real data: the training table TabDiff's "
            r"evaluation compares against. Route A, training seed 0, 20 sampled tables.")


# ---------------------------------------------------------------------------
# Figure: pair-wise correlation error heatmaps (TabDiff Fig. 4)
# ---------------------------------------------------------------------------
def figure_heatmaps() -> dict:
    fig = S.figure(S.FULL, 6.3)
    gs = fig.add_gridspec(len(PAPER_DS), len(F.ARMS) + 1, width_ratios=[1] * len(F.ARMS) + [0.06],
                          wspace=0.02, hspace=0.02)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    norm = Normalize(0, VMAX)
    out = {"clipped": {}, "trend": {}}
    for r, ds in enumerate(PAPER_DS):
        for c, arm in enumerate(F.ARMS):
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(S.rule_label(arm), loc="center", fontsize=S.FS_LABEL)
            if F.run_dir(ds, arm) is None:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal")
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=S.FS_LABEL, color=S.INK2)
                continue
            M, n_num = full_pair_matrix(ds, arm)
            n = len(M)
            off = ~np.eye(n, dtype=bool)
            tm = trend_mean(ds, arm)
            assert abs(M[off].mean() - tm) < 1e-9, (ds, arm, M[off].mean(), tm)   # mean = Trend / 100
            out["trend"][(ds, arm)] = tm
            out["clipped"][(ds, arm)] = float((M[np.triu_indices(n, 1)] > VMAX).mean())
            ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
            for v in (n_num - 0.5,):
                if 0 < n_num < n:
                    ax.axhline(v, color=SPLIT, lw=0.6)
                    ax.axvline(v, color=SPLIT, lw=0.6)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(n - 0.5, -0.5)
            if c == 0:
                _brackets(ax, n_num, n)
                ax.annotate(S.ds_label(ds), xy=(0, 0.5), xycoords="axes fraction", xytext=(-26, 0),
                            textcoords="offset points", rotation=90, ha="right", va="center",
                            fontsize=S.FS_LABEL, fontweight="bold", annotation_clip=False)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_subplot(gs[:, -1]), extend="max")
    cb.set_label("Pair-wise correlation error (lighter = better)", fontsize=S.FS_TICK)
    cb.ax.tick_params(labelsize=S.FS_TICK, length=2, width=S.LW_AXIS)
    cb.outline.set_linewidth(S.LW_AXIS)
    out["size"] = S.save(fig, "paper_style_pair_heatmaps", heatmaps_caption(out))
    return out


def _brackets(ax, n_num: int, n: int):
    """Red brackets with 'Num' / 'Cat' to the left of the matrix (data y, axes-fraction x)."""
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for (y0, y1), lab in (((-0.5, n_num - 0.5), "Num"), ((n_num - 0.5, n - 0.5), "Cat")):
        if y1 - y0 <= 0:
            continue
        pad = 0.15
        x = -0.045
        ax.plot([x, x], [y0 + pad, y1 - pad], transform=tr, color=SPLIT, lw=0.7, clip_on=False)
        for yy in (y0 + pad, y1 - pad):
            ax.plot([x, x + 0.02], [yy, yy], transform=tr, color=SPLIT, lw=0.7, clip_on=False)
        ax.text(x - 0.02, (y0 + y1) / 2, lab, transform=tr, rotation=90, ha="right", va="center",
                fontsize=S.FS_CELL, color=SPLIT, clip_on=False)


def heatmaps_caption(out) -> str:
    def tr(ds):
        return " / ".join(f"{out['trend'][(ds, a)]:.3f}" if (ds, a) in out["trend"] else "--" for a in F.ARMS)

    def cl(ds):
        return " / ".join(f"{100 * out['clipped'][(ds, a)]:.1f}" if (ds, a) in out["clipped"] else "--"
                          for a in F.ARMS)
    trends = "; ".join(f"{S.ds_label(d)}: {tr(d)}" for d in PAPER_DS)
    clips = "; ".join(f"{S.ds_label(d)}: {cl(d)}" for d in PAPER_DS)
    return (r"Pair-wise correlation error of every pair of columns, for the datasets and with the colour "
            r"scale of Figure~4 of the TabDiff paper, for each of our neighbour-selection rules (columns) "
            r"on Adult, Beijing, Magic and Shoppers (rows). Each cell of a matrix is $1-\text{score}$ of the "
            r"Column Pair Trends metric of sdmetrics~0.18.0 as computed inside TabDiff's evaluation "
            r"(correlation similarity for two numerical columns, contingency similarity otherwise), averaged "
            r"over the 20 sampled tables; the matrix is symmetric, its diagonal is left white, and the "
            r"columns are ordered numerical first, then categorical (red lines and brackets). Colour scale "
            rf"0 to {VMAX:.2f} as in the paper (lighter is better); values above {VMAX:.2f} are drawn at the "
            rf"maximum colour; percentage of pairs above {VMAX:.2f} (Fixed top-$k$ / Jenks / entmax-1.5 / "
            rf"Dense): {clips}. The mean of each matrix equals the run's Trend error divided by 100: "
            rf"{trends}. The dense ablation ($k=N-1$) exists for Beijing and Magic only (--); Fixed top-$k$ on "
            r"Magic is the $k=3$ run. The layout follows the TabDiff paper so that the rules can be set "
            r"beside its Figure~4, but the paper's panels show other models on other samples: the comparison "
            r"is visual and qualitative only. Route A, training seed 0, 20 sampled tables.")


def main() -> dict:
    missing = [f"{ds}/{a}" for ds in PAPER_DS for a in arms_of(ds)
               if not (F.run_dir(ds, a) / PM.OUT_DIR / "pair_trend_error.csv").exists()]
    if missing:
        sys.exit("no pair-trend outputs for " + ", ".join(missing) + ": run analysis/tabdiff/pair_and_marginals.py")
    m = figure_marginals()
    for kind in ("kde", "bars"):
        print(f"  resolved {kind} columns: " + ", ".join(f"{d}:{c}" for d, c in m["resolved"][kind]))
    for (ds, col, a), v in m["kde"].items():
        print(f"  KDE {ds}/{col}/{a}: bandwidth {v['bandwidth']:.4g}, synthetic rows {v['n_syn']} "
              f"(KDE on {v['n_syn_kde']})")
    print(f"wrote paper_style_marginals {m['size']}")
    h = figure_heatmaps()
    for (ds, a), v in h["clipped"].items():
        print(f"  heatmap {ds}/{a}: mean {h['trend'][(ds, a)]:.4f} = trend/100, pairs > {VMAX}: {100 * v:.1f} %")
    print(f"wrote paper_style_pair_heatmaps {h['size']}")
    return {"marginals": m, "heatmaps": h}


if __name__ == "__main__":
    main()

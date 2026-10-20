#!/usr/bin/env python3
"""
Figure 3 -- graph density vs joint-distribution fidelity (Route A, seed 0).

    python3 analysis/summary/fig_density_vs_fidelity.py

x = mean fraction of candidate neighbours kept, k / (N - 1):
  adaptive rules  sampling phase of topk_k_stats.csv, over blocks, t-bins and nodes, weighted
                  by n: sum(k_sum) / sum(n) / (N - 1)
  fixed / dense   top_k / (N - 1) from run_meta.json (dense: top_k = N - 1, so exactly 1)
y = C2ST (XGBoost) and wNMIS from tabpc (mean +- std, ddof=1, over the 20 tables), the records
of build_summary.b_values (checked against each run's tabpc_eval/avg_std.csv).
Writes figures/density_vs_fidelity.{png,pdf} and csv/density_vs_fidelity.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import route_a_figs as F  # noqa: E402
import thesis_style as S  # noqa: E402

np, pd, plt = F.np, F.pd, F.plt
YS = [("c2st_xgb", r"C2ST (XGBoost) $\uparrow$"), ("wnmis", r"wNMIS $\uparrow$")]


def kept_fraction(ds: str, arm: str, run: Path) -> tuple[float, str]:
    """(k / (N - 1), how it was obtained)."""
    n1 = F.n_minus_1(ds)
    if arm in ("fixed", "dense"):
        k = F.run_meta(run)["gnn"]["top_k"]
        if arm == "dense":
            assert k == n1, (ds, k, n1)
        assert 0 < k <= n1, (ds, arm, k)
        return k / n1, f"run_meta top_k = {k}"
    p = run / "topk_k_stats.csv"
    t = pd.read_csv(p)
    t = t[t.phase == "sample"]
    assert len(t), p
    assert t.node.nunique() == n1 + 1, (p, t.node.nunique(), n1)       # N nodes in the log
    # logged k = off-diagonal entries with weight > 0 (tabdiff_gnn/topk_log.py): a kept edge whose
    # sigmoid underflows to exactly 0 is counted as dropped, so k = 0 can appear (share in the csv)
    assert t.k_max.max() <= n1 and t.k_min.min() >= 0, p
    return t.k_sum.sum() / t.n.sum() / n1, "topk_k_stats.csv, phase sample: sum k_sum / sum n"


def zero_k_share(arm: str, run: Path) -> float:
    """Share of sampling-phase node-forwards logged with k = 0 (topk_k_hist.csv)."""
    if arm in ("fixed", "dense"):
        return np.nan
    h = pd.read_csv(run / "topk_k_hist.csv")
    h = h[h.phase == "sample"]
    return float(h.loc[h.k == 0, "count"].sum() / h["count"].sum())


def collect() -> pd.DataFrame:
    bv = F.b_values()
    rows = []
    for ds in F.DATASETS:
        for arm in F.ARMS:
            run = F.run_dir(ds, arm)
            if run is None:
                continue
            recs = {y: bv.get((ds, F.ARM_TAG[arm], y)) for y, _ in YS}
            if not all(F.ok(r) for r in recs.values()):
                continue
            # the tabpc summary the evaluation wrote agrees with the 20 per-table values
            avg = pd.read_csv(run / "tabpc_eval" / "avg_std.csv", index_col=0)
            for y, _ in YS:
                assert abs(avg.loc[y, "avg"] - recs[y]["mean"]) < 1e-12, (run, y)
                assert abs(avg.loc[y, "std"] - recs[y]["std"]) < 1e-12, (run, y)
            x, how = kept_fraction(ds, arm, run)
            rows.append({"dataset": ds, "arm": arm, "rule": F.RA.ARM_LABEL[arm], "N-1": F.n_minus_1(ds),
                         "kept_fraction": x, "kept_fraction_source": how,
                         "logged_k0_share": zero_k_share(arm, run),
                         **{f"{y}_{s}": recs[y][s] for y, _ in YS for s in ("mean", "std")},
                         "n_tables": recs["wnmis"]["n"], "run": F.BS.rel(run)})
    return pd.DataFrame(rows)


def repel_labels(ax, items, obstacles, fontsize, max_iter=300):
    """Direct labels right of each dataset's last point, nudged vertically (deterministic) until no
    two labels overlap and no label covers a data marker; a thin leader joins a moved label to its
    point. items: [(text, x, y, colour)]; obstacles: [(x, y)] marker centres (data coords)."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72
    texts = []
    for txt, x, y, c in items:
        t = ax.annotate(txt, (x, y), xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=fontsize, color=S.INK, annotation_clip=False)
        texts.append([t, x, y, 0.0])                     # last = vertical offset (pt)
    obs = ax.transData.transform(np.asarray(obstacles)) if obstacles else np.empty((0, 2))
    m = 3.0 * px                                         # marker half-size + margin (px)

    def boxes():
        out = []
        for t, *_ in texts:
            out.append(t.get_window_extent(r))
        return out

    for _ in range(max_iter):
        moved = False
        bbs = boxes()
        for i, (t, x, y, off) in enumerate(texts):
            b = bbs[i]
            push = 0.0
            for j, b2 in enumerate(bbs):
                if j != i and b.x0 < b2.x1 and b2.x0 < b.x1 and b.y0 < b2.y1 + px and b2.y0 < b.y1 + px:
                    push += 0.5 if (b.y0 + b.y1) >= (b2.y0 + b2.y1) else -0.5
            hit = (obs[:, 0] + m > b.x0) & (obs[:, 0] - m < b.x1) & (obs[:, 1] + m > b.y0) & (obs[:, 1] - m < b.y1)
            own = ax.transData.transform((x, y))
            hit &= ~((np.abs(obs[:, 0] - own[0]) < 0.5) & (np.abs(obs[:, 1] - own[1]) < 0.5))
            if hit.any():
                push += 0.5 if obs[hit, 1].mean() < (b.y0 + b.y1) / 2 else -0.5
            if push:
                texts[i][3] = off + np.sign(push) * 1.0
                t.set_position((5, texts[i][3]))
                moved = True
        if not moved:
            break
    for t, x, y, off in texts:
        if abs(off) > 2:
            t.set_position((9, off))
            ax.annotate("", (x, y), xytext=(9, off), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=S.INK2, shrinkA=0, shrinkB=2.5),
                        annotation_clip=False)
    return texts


CAPTION = (
    r"Joint-distribution fidelity against the density of the learned graph. The horizontal axis is the "
    r"mean percentage of the $N-1$ candidate neighbours that a node keeps during sampling: for the "
    r"adaptive rules the count-weighted mean of the logged $k$ over all blocks, diffusion times and nodes, "
    r"divided by $N-1$; for the fixed rule $k/(N-1)$; for the dense ablation 100\%. (a) C2ST with an "
    r"XGBoost discriminator and (b) wNMIS (both from tabpc; higher is better), mean $\pm$ std over the 20 "
    r"tables. Colour = dataset (lines join the runs of one dataset in order of density), marker = rule. "
    r"Route A, training seed 0, 20 sampled tables. The adaptive rules' logged $k$ counts edges with a "
    r"non-zero weight, so a kept edge whose weight underflows to zero counts as dropped (at most "
    r"{zero} of node evaluations).{absent}")


def main():
    D = collect()
    D.to_csv(F.CSV / "density_vs_fidelity.csv", index=False)
    fig = S.figure(S.FULL, 3.0)
    axes = fig.subplots(1, 2)
    for i, (ax, (y, lab)) in enumerate(zip(axes, YS)):
        S.style_axes(ax, grid="both")
        items, obstacles = [], []
        for ds in F.DATASETS:
            d = D[D.dataset == ds].sort_values("kept_fraction")
            if d.empty:
                continue
            c = S.DS_COLOR[ds]
            ax.plot(d.kept_fraction, d[f"{y}_mean"], color=c, lw=0.8, alpha=0.9, zorder=1)
            for _, r in d.iterrows():
                ax.errorbar(r.kept_fraction, r[f"{y}_mean"], yerr=r[f"{y}_std"], fmt="none", ecolor=c,
                            elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=2, clip_on=False)
                ax.plot(r.kept_fraction, r[f"{y}_mean"], marker=S.RULE_MARKER[r.arm], ms=4.5, color=c,
                        mec="white", mew=0.5, ls="none", zorder=3, clip_on=False)
                obstacles.append((r.kept_fraction, r[f"{y}_mean"]))
            last = d.iloc[-1]
            items.append((S.ds_label(ds), last.kept_fraction, last[f"{y}_mean"], c))
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
        ax.set_xlabel(r"% of candidate neighbours kept")
        ax.set_ylabel(lab)
        S.panel_label(ax, "ab"[i])
        ax._items, ax._obstacles = items, obstacles
    handles = [plt.Line2D([], [], color=S.INK2, marker=S.RULE_MARKER[a], lw=0, ms=4.5, mec="white", mew=0.5,
                          label=S.rule_label(a)) for a in F.ARMS if (D.arm == a).any()]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), handletextpad=0.2)
    fig.canvas.draw()
    for ax in axes:                                  # after the layout is final: measured placement
        repel_labels(ax, ax._items, ax._obstacles, S.FS_CELL + 0.5)
    absent = [S.ds_label(d) for d in F.BS.B_DS if d not in set(D.dataset)]
    zero = D.logged_k0_share.max()
    cap = CAPTION.replace("{zero}", "$" + f"{zero:.1e}".replace("e-0", r"\times 10^{-") + "}$").replace(
        "{absent}", f" {', '.join(absent)} not shown (no evaluated run yet)." if absent else "")
    size = S.save(fig, "density_vs_fidelity", cap)
    print(D[["dataset", "arm", "kept_fraction", "c2st_xgb_mean", "wnmis_mean"]].to_string(index=False))
    print(f"wrote density_vs_fidelity ({size[0]} x {size[1]} in), csv/density_vs_fidelity.csv")


if __name__ == "__main__":
    main()

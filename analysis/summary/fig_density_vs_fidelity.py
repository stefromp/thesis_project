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

np, pd, plt, ct = F.np, F.pd, F.plt, F.ct
YS = [("c2st_xgb", "C2ST (XGBoost) ↑"), ("wnmis", "wNMIS ↑")]


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
            rows.append({"dataset": ds, "arm": arm, "rule": F.ARM_LABEL[arm], "N-1": F.n_minus_1(ds),
                         "kept_fraction": x, "kept_fraction_source": how,
                         "logged_k0_share": zero_k_share(arm, run),
                         **{f"{y}_{s}": recs[y][s] for y, _ in YS for s in ("mean", "std")},
                         "n_tables": recs["wnmis"]["n"], "run": F.BS.rel(run)})
    return pd.DataFrame(rows)


def dodge(points: list[tuple[str, float, float]], gap: float, dx: float = 0.12) -> dict:
    """Direct-label y positions: greedy upward shift of labels closer than (dx, gap)."""
    placed = {}
    for name, x, y in sorted(points, key=lambda p: p[2]):
        while any(abs(x - px) < dx and abs(y - py) < gap for px, py in placed.values()):
            y += gap / 4
        placed[name] = (x, y)
    return placed


def main():
    D = collect()
    D.to_csv(F.CSV / "density_vs_fidelity.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (y, lab) in zip(axes, YS):
        F.style_ax(ax, grid="both")
        ends = []
        for ds in F.DATASETS:
            d = D[D.dataset == ds].sort_values("kept_fraction")
            if d.empty:
                continue
            c = F.C_DS[ds]
            ax.plot(d.kept_fraction, d[f"{y}_mean"], color=c, lw=0.9, alpha=0.7, zorder=1)
            for _, r in d.iterrows():
                ax.errorbar(r.kept_fraction, r[f"{y}_mean"], yerr=r[f"{y}_std"], fmt=F.MK_ARM[r.arm], ms=7,
                            color=c, mec="white", mew=1, elinewidth=1, capsize=2, zorder=2)
            last = d.iloc[-1]
            ends.append((ds, last.kept_fraction, last[f"{y}_mean"]))
        lo, hi = D[f"{y}_mean"].min(), D[f"{y}_mean"].max()
        for ds, (lx, ly) in dodge(ends, gap=0.045 * (hi - lo)).items():
            ax.annotate(ds, (lx, ly), xytext=(7, 0), textcoords="offset points", fontsize=7.5, color=ct.C_TEXT,
                        va="center")
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("mean fraction of candidate neighbours kept, k / (N − 1)", fontsize=8, color=ct.C_TEXT2)
        ax.set_ylabel(lab, fontsize=8, color=ct.C_TEXT2)
        ax.set_title(lab.replace(" ↑", "") + " vs graph density", loc="left", fontsize=10, color=ct.C_TEXT)
    handles = [plt.Line2D([], [], color=ct.C_TEXT2, marker=F.MK_ARM[a], lw=0, ms=7, label=F.ARM_LABEL[a])
               for a in F.ARMS if (D.arm == a).any()] + \
              [plt.Line2D([], [], color=F.C_DS[d], lw=2, label=d) for d in F.DATASETS if (D.dataset == d).any()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.05))
    absent = [d for d in F.BS.B_DS if d not in set(D.dataset)]
    F.caption(fig, "Route A, training seed 0, 20 tables per run (mean ± std, ddof=1); wNMIS and C2ST (XGBoost) from "
                    "tabpc. x: adaptive rules = mean k over the sampling-phase log (all blocks, t-bins, nodes, "
                    "count-weighted) / (N − 1); fixed = top_k / (N − 1); dense = 1. Points of one dataset joined in "
                    "order of x. " + (f"Not shown (no evaluated run on disk): {', '.join(absent)}." if absent else ""))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    F.savefig(fig, "density_vs_fidelity")
    print(D[["dataset", "arm", "kept_fraction", "c2st_xgb_mean", "wnmis_mean"]].to_string(index=False))
    print(f"wrote {F.BS.rel(F.FIG)}/density_vs_fidelity.{{pdf,png}}, {F.BS.rel(F.CSV)}/density_vs_fidelity.csv")


if __name__ == "__main__":
    main()

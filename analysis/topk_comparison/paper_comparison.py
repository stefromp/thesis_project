#!/usr/bin/env python3
"""Our three top-k rules vs the TabDDPM paper (Table 5, tuned CatBoost).

    python3 analysis/topk_comparison/paper_comparison.py

Reads the coverage.csv written by compare_topk.py (OUT = this directory, and
before_catfix/ if present) to know which run each cell uses, then recomputes
the paper's statistic from the raw per-run rows: mean and std (np.std, ddof=0,
as lib/metrics.py get_mean_std) over all 5 gen x 10 clf = 50 CatBoost runs,
of macro-F1 (classification) or R2 (regression). Summary-only fixed cells
(grid JSON) carry the same statistic (mean / std over all runs).

Writes paper_comparison.csv and PAPER_COMPARISON.md (per output directory).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import compare_topk as ct

# Kotelnikov et al., "TabDDPM: Modelling Tabular Data with Diffusion Models",
# arXiv:2209.15421v2, Table 5 ("machine learning efficiency computed w.r.t.
# the state-of-the-art tuned CatBoost model"): (TabDDPM mean, std, Real mean, std).
PAPER = {
    "abalone": ("AB", "R2", 0.550, 0.010, 0.556, 0.004),
    "adult": ("AD", "F1", 0.795, 0.001, 0.815, 0.002),
    "buddy": ("BU", "F1", 0.906, 0.003, 0.906, 0.002),
    "california": ("CA", "R2", 0.836, 0.002, 0.857, 0.001),
    "cardio": ("CAR", "F1", 0.737, 0.001, 0.738, 0.001),
    "churn2": ("CH", "F1", 0.755, 0.006, 0.740, 0.009),
    "default": ("DE", "F1", 0.691, 0.004, 0.688, 0.003),
    "diabetes": ("DI", "F1", 0.740, 0.020, 0.785, 0.013),
    "fb-comments": ("FB", "R2", 0.713, 0.002, 0.837, 0.001),
    "gesture": ("GE", "F1", 0.597, 0.006, 0.636, 0.007),
    "higgs-small": ("HI", "F1", 0.722, 0.001, 0.724, 0.001),
    "house": ("HO", "R2", 0.677, 0.010, 0.662, 0.003),
    "insurance": ("IN", "R2", 0.809, 0.002, 0.814, 0.001),
    "king": ("KI", "R2", 0.833, 0.014, 0.907, 0.002),
    "miniboone": ("MI", "F1", 0.936, 0.001, 0.934, 0.000),
    "wilt": ("WI", "F1", 0.904, 0.009, 0.898, 0.006),
}
METHODS = ("fixed", "jenks", "entmax15", "paper")
LABEL = {**ct.RULE_LABEL, "paper": "TabDDPM (paper)"}


def raw_runs(run_dir: Path, metric: str) -> np.ndarray:
    files = sorted(run_dir.glob("per_seed_scores.csv")) + sorted(run_dir.glob("per_seed_scores.gen*.csv"))
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw = raw[raw.clf_seed.notna()].drop_duplicates(["gen_seed", "clf_seed", "metric"], keep="last")
    seeds = ct.load_run_dir(run_dir)["gen_seeds"]
    return raw[(raw.metric == metric) & raw.gen_seed.isin(seeds)].value.to_numpy(float)


def cell_stats(row: pd.Series, task: str) -> dict:
    metric = "r2" if task == "regression" else "macro avg/f1-score"
    if not str(row.status).startswith("complete"):
        return {"mean": np.nan, "std": np.nan, "n_runs": 0, "kind": row.status}
    if row.data_kind.startswith("summary"):
        path = ct.REPO / row.source.split(" [")[0]
        exp = row.source.split("[")[1].rstrip("]")
        rec = next(r for r in json.load(open(path)) if r["exp_name"] == exp)
        m = rec["metrics"][metric]
        return {"mean": m["mean"], "std": m["std"], "n_runs": rec["n_gen_seeds"] * rec["n_clf_seeds"],
                "kind": "grid summary"}
    v = raw_runs(ct.REPO / row.source, metric)
    return {"mean": v.mean(), "std": v.std(), "n_runs": len(v), "kind": "per run"}


def build(out_dir: Path) -> pd.DataFrame:
    cov = pd.read_csv(out_dir / "coverage.csv")
    rows = []
    for ds, g in cov.groupby("dataset"):
        if ds not in PAPER:
            continue
        ab, mname, pm, ps, rm, rs = PAPER[ds]
        task, prov = g.task_type.iloc[0], g.provenance.iloc[0]
        for _, r in g.iterrows():
            s = cell_stats(r, task)
            d = s["mean"] - pm
            sd = max(s["std"], ps) if np.isfinite(s["std"]) else np.nan
            rows.append({"dataset": ds, "abbrev": ab, "paper_metric": mname, "provenance": prov,
                         "subset": "held_out" if prov == "transfer" else "development",
                         "variant": r.variant, "rule": r.rule, **s,
                         "paper_tabddpm_mean": pm, "paper_tabddpm_std": ps, "paper_real_mean": rm,
                         "paper_real_std": rs, "delta_vs_paper": d,
                         "delta_rel_pct": 100 * d / abs(pm),
                         "vs_paper": ("—" if not np.isfinite(d) else
                                      "within 1 sd" if abs(d) <= sd else ("above" if d > 0 else "below")),
                         "delta_vs_paper_real": s["mean"] - rm})
    return pd.DataFrame(rows)


def across(df: pd.DataFrame) -> list[str]:
    L = ["| subset | n | avg rank fixed | Jenks | entmax-1.5 | TabDDPM (paper) | Friedman p |",
         "|---|---|---|---|---|---|---|"]
    W = ["| subset | rule | n | mean Δ | median Δ | above / within 1 sd / below | Wilcoxon p (Δ vs 0) |",
         "|---|---|---|---|---|---|---|"]
    for sub, _, lab in ct.SUBSETS:
        d = df if sub == "pooled" else df[df.subset == sub]
        piv = d.pivot(index="dataset", columns="rule", values="mean").dropna()
        if piv.empty:
            continue
        piv["paper"] = [PAPER[x][2] for x in piv.index]
        M = piv[list(METHODS)].to_numpy()
        avg = np.vstack([stats.rankdata(-r) for r in M]).mean(0)
        fp = stats.friedmanchisquare(*M.T).pvalue if len(M) >= 3 else np.nan
        fps = f"{fp:.3f}" if np.isfinite(fp) else "n/a"
        L.append(f"| {lab} | {len(M)} | " + " | ".join(f"{a:.2f}" for a in avg) + f" | {fps} |")
        for r in ct.RULES:
            x = d[(d.rule == r) & d.dataset.isin(piv.index)]
            dv = x.delta_vs_paper.to_numpy()
            p = stats.wilcoxon(dv).pvalue if len(dv) and np.any(dv != 0) else np.nan
            oc = x.vs_paper.value_counts()
            W.append(f"| {lab} | {ct.RULE_LABEL[r]} | {len(dv)} | {dv.mean():+.4f} | {np.median(dv):+.4f} | "
                     f"{oc.get('above', 0)} / {oc.get('within 1 sd', 0)} / {oc.get('below', 0)} | {p:.3f} |")
    return L + [""] + W


def write(df: pd.DataFrame, out_dir: Path, title: str):
    L = [f"# Comparison with the TabDDPM paper — {title}\n",
         "_Generated by `paper_comparison.py`. Paper values: Kotelnikov et al., arXiv:2209.15421v2, Table 5 "
         "(tuned CatBoost; F1 for classification, R2 for regression)._\n",
         "Ours: mean ± std over the 5 gen × 10 CatBoost seeds = 50 runs (np.std, ddof=0, as the paper's code), "
         "macro-F1 (`lib/metrics.py`) or R2, on the same tuned CatBoost configs (`tuned_models/catboost`). "
         "\"within 1 sd\" = |Δ| ≤ the larger of the two stds; the 50 runs are 5 clusters of 10, so no p-value is "
         "given per dataset. **The paper's TabDDPM is an MLP tuned per dataset on the validation split (50 Optuna "
         "trials); our rules use one GNN configuration per dataset, transferred (held-out) or selected on the test "
         "split (development).** fb-comments: no results (decision pending). higgs-small, miniboone: still running, "
         "excluded.\n"]
    L.append("## Per dataset (held-out first)\n")
    L.append("| dataset | subset | metric | fixed | Jenks | entmax-1.5 | TabDDPM (paper) | Real (paper) |")
    L.append("|---|---|---|---|---|---|---|---|")
    order = df.assign(_o=df.subset.map({"held_out": 0, "development": 1})).sort_values(["_o", "dataset"])
    for ds in order.dataset.unique():
        g = df[df.dataset == ds].set_index("rule")
        ab, mn, pm, ps, rm, rs = PAPER[ds]
        cells = []
        for r in ct.RULES:
            if r not in g.index or not np.isfinite(g.loc[r, "mean"]):
                cells.append("—")
                continue
            x = g.loc[r]
            mark = {"above": " ▲", "below": " ▼", "within 1 sd": ""}.get(x.vs_paper, "")
            cells.append(f"{x['mean']:.3f} ± {x['std']:.3f} ({x.delta_vs_paper:+.3f}){mark}"
                         + (" ‡" if x.kind == "grid summary" else ""))
        sub = g.subset.iloc[0].replace("_", "-")
        L.append(f"| {ds} ({ab}) | {sub} | {mn} | " + " | ".join(cells) + f" | {pm:.3f} ± {ps:.3f} | {rm:.3f} ± {rs:.3f} |")
    L.append("\n(Δ vs paper TabDDPM) ▲ above / ▼ below by more than 1 sd; ‡ fixed = grid summary (training seed 0).\n")
    L.append("## Across datasets\n")
    L.append("Ranks over the four methods per dataset (1 = best), then averaged; Wilcoxon signed-rank of the "
             "per-dataset Δ (ours − paper TabDDPM). n = 4 held-out: smallest attainable two-sided p = 0.125.\n")
    L.extend(across(df))
    (out_dir / "PAPER_COMPARISON.md").write_text("\n".join(L) + "\n")


def main():
    for out_dir, title in ((ct.OUT, "after the categorical fix"), (ct.OUT / "before_catfix", "before the fix")):
        if (out_dir / "coverage.csv").exists():
            df = build(out_dir)
            df.to_csv(out_dir / "paper_comparison.csv", index=False)
            write(df, out_dir, title)
            print(f"wrote {ct.rel(out_dir / 'PAPER_COMPARISON.md')}")


if __name__ == "__main__":
    main()

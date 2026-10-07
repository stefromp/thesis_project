#!/usr/bin/env python3
"""Does the categorical-input fix (cat_input="prob") work on the real runs?

    python3 analysis/dependency_fidelity/catfix_verification.py

For every dataset with categorical columns and every rule, compares the
original runs (cat_input="log", trees without suffix) with the *_catfix runs:

  catfix_tvd.csv    per run x gen seed x categorical column: TVD between the
                    synthetic and the real-train marginal; plus the real-test
                    vs real-train TVD of the same column as a floor.
  catfix_mloss.csv  per run: final MLoss (mean of the last 10 logged values of
                    loss.csv) next to the always-uniform and real-marginal
                    baselines (cat_input_evidence.loss_baselines).
  CATFIX_VERIFICATION.md  the two tables, rendered.

Read-only on transfer_runs/exp. The legacy fixed rule of an ablated dataset is
the grid summary JSON (no samples, no loss.csv) and so has no row here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "topk_comparison"))
sys.path.insert(0, str(HERE))
import compare_topk as ct  # noqa: E402
from cat_input_local_check import _setup  # noqa: E402

REPO, ROOT = ct.REPO, ct.RESULTS_ROOT
TREES = {"log": {"fixed": "ablation", "jenks": ct.RULE_DIR["jenks"], "entmax15": ct.RULE_DIR["entmax15"]},
         "prob": {"fixed": "ablation_catfix", "jenks": ct.RULE_DIR["jenks"] + "_catfix",
                  "entmax15": ct.RULE_DIR["entmax15"] + "_catfix"}}
# Columns singled out in the bug report (cardio cat_4/cat_5, churn2 cat_1/cat_2
# copied the shared noise; the user also asked for cardio cat_2).
FOCUS = {"cardio": (2, 4, 5), "churn2": (1, 2)}


def tvd(real: np.ndarray, syn: np.ndarray) -> float:
    vals = np.union1d(real, syn)
    pr = np.array([(real == v).mean() for v in vals])
    ps = np.array([(syn == v).mean() for v in vals])
    return float(0.5 * np.abs(pr - ps).sum())


def cat_datasets() -> list[str]:
    return sorted(d.name for d in ROOT.iterdir()
                  if d.is_dir() and d.name not in ct.EXCLUDE and ct.n_categorical(d.name) > 0
                  and (REPO / "data" / d.name / "X_cat_train.npy").exists()
                  # only datasets with catfix runs (fb-comments: none, its log
                  # runs unfinished -> reported as missing elsewhere)
                  and (d / TREES["prob"]["fixed"]).is_dir())


def main():
    _setup()
    import cat_input_evidence as ev

    tvd_rows, ml_rows = [], []
    for ds in cat_datasets():
        real_tr = np.load(REPO / "data" / ds / "X_cat_train.npy", allow_pickle=True).astype(str)
        real_te = np.load(REPO / "data" / ds / "X_cat_test.npy", allow_pickle=True).astype(str)
        floor = [tvd(real_tr[:, j], real_te[:, j]) for j in range(real_tr.shape[1])]
        base = None
        for cat_input, trees in TREES.items():
            for rule, tree in trees.items():
                runs = sorted(p for p in (ROOT / ds / tree).glob("gnn_*") if p.is_dir())
                if not runs:
                    continue
                run = runs[0]
                for g in sorted(run.glob("gen_seed_*/X_cat_train.npy")):
                    syn = np.load(g, allow_pickle=True).astype(str)
                    seed = int(g.parent.name.split("_")[-1])
                    for j in range(real_tr.shape[1]):
                        tvd_rows.append({"dataset": ds, "cat_input": cat_input, "rule": rule, "gen_seed": seed,
                                         "column": f"cat_{j}", "n_levels": len(set(real_tr[:, j])),
                                         "tvd": tvd(real_tr[:, j], syn[:, j]), "floor_test_vs_train": floor[j],
                                         "run": str(run.relative_to(ROOT))})
                lp = run / "loss.csv"
                if lp.exists():
                    base = base or ev.loss_baselines(ds)
                    lo = pd.read_csv(lp)
                    ml_rows.append({"dataset": ds, "cat_input": cat_input, "rule": rule,
                                    "final_mloss": float(lo.mloss.tail(10).mean()),
                                    "uniform": base["uniform"], "marginal": base["marginal"],
                                    "last_step": int(lo.step.iloc[-1]) if "step" in lo else len(lo),
                                    "run": str(run.relative_to(ROOT))})
        print(ds, "done", flush=True)
    T, M = pd.DataFrame(tvd_rows), pd.DataFrame(ml_rows)
    T.to_csv(HERE / "catfix_tvd.csv", index=False)
    M.to_csv(HERE / "catfix_mloss.csv", index=False)
    write_md(T, M)


def verdict(v: float, u: float, m: float) -> str:
    if v >= u - 0.02:
        return "≥U (learned nothing)"
    if v < m - 0.05:
        return "<M (uses dependencies)"
    return "≈M"


def write_md(T: pd.DataFrame, M: pd.DataFrame):
    L = ["# Categorical-input fix: verification on the real runs\n",
         "_Generated by `catfix_verification.py` from `transfer_runs/exp` and `data/`._\n",
         "TVD = total variation distance between a categorical column's synthetic and real-train marginal, "
         "mean ± sd over the 5 generation seeds. Floor = real test vs real train for the same column. "
         "log = original runs (`cat_input=\"log\"`); prob = `*_catfix` runs. The legacy fixed rule on ablated "
         "datasets is a grid summary without samples (—).\n"]
    L.append("## Focus columns\n")
    L.append("| dataset | column | rule | log | prob | floor |")
    L.append("|---|---|---|---|---|---|")
    agg = (T.groupby(["dataset", "column", "rule", "cat_input"]).tvd.agg(["mean", "std"]).reset_index())
    fl = T.groupby(["dataset", "column"]).floor_test_vs_train.first()

    def cell(ds, col, rule, ci):
        r = agg[(agg.dataset == ds) & (agg.column == col) & (agg.rule == rule) & (agg.cat_input == ci)]
        return f"{r['mean'].iloc[0]:.3f} ± {r['std'].iloc[0]:.3f}" if len(r) else "—"
    for ds, cols in FOCUS.items():
        for j in cols:
            col = f"cat_{j}"
            for rule in ct.RULES:
                L.append(f"| {ds} | {col} | {ct.RULE_LABEL[rule]} | {cell(ds, col, rule, 'log')} | "
                         f"{cell(ds, col, rule, 'prob')} | {fl.get((ds, col), np.nan):.3f} |")
    L.append("\n## All categorical columns (max and mean TVD over columns, per gen seed then averaged)\n")
    L.append("| dataset | n cat | rule | log: mean TVD | log: max TVD | prob: mean TVD | prob: max TVD | floor mean |")
    L.append("|---|---|---|---|---|---|---|---|")
    per = (T.groupby(["dataset", "rule", "cat_input", "gen_seed"]).tvd.agg(["mean", "max"])
           .groupby(["dataset", "rule", "cat_input"]).mean().reset_index())
    for ds in sorted(T.dataset.unique()):
        ncat = T[T.dataset == ds].column.nunique()
        fm = T[T.dataset == ds].groupby("column").floor_test_vs_train.first().mean()
        for rule in ct.RULES:
            def g(ci, k):
                r = per[(per.dataset == ds) & (per.rule == rule) & (per.cat_input == ci)]
                return f"{r[k].iloc[0]:.3f}" if len(r) else "—"
            L.append(f"| {ds} | {ncat} | {ct.RULE_LABEL[rule]} | {g('log', 'mean')} | {g('log', 'max')} | "
                     f"{g('prob', 'mean')} | {g('prob', 'max')} | {fm:.3f} |")
    L.append("\n## Final MLoss vs trivial predictors\n")
    L.append("Final MLoss = mean of the last 10 logged values of `loss.csv`. U = always-uniform x0 predictor, "
             "M = real-marginal predictor (pipeline diffusion, T = 1000, cosine). ≥U: within 0.02 of U or above; "
             "<M: more than 0.05 below M.\n")
    L.append("| dataset | U | M | rule | log | prob | prob verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for ds in sorted(M.dataset.unique()):
        d = M[M.dataset == ds]
        u, m = d.uniform.iloc[0], d.marginal.iloc[0]
        for rule in ct.RULES:
            lo = d[(d.rule == rule) & (d.cat_input == "log")].final_mloss
            pr = d[(d.rule == rule) & (d.cat_input == "prob")].final_mloss
            L.append(f"| {ds} | {u:.3f} | {m:.3f} | {ct.RULE_LABEL[rule]} | "
                     f"{f'{lo.iloc[0]:.3f}' if len(lo) else '—'} | {f'{pr.iloc[0]:.3f}' if len(pr) else '—'} | "
                     f"{verdict(pr.iloc[0], u, m) if len(pr) else '—'} |")
    (HERE / "CATFIX_VERIFICATION.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

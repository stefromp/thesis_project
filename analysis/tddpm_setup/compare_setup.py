#!/usr/bin/env python3
"""Budget-matched GNN runs vs the current runs vs TabDDPM.

    python3 analysis/tddpm_setup/compare_setup.py

Targeted check, not a full re-run: one rule (fixed), five datasets (+ higgs-
small / miniboone when present). The question is whether training the GNN
denoiser with TabDDPM's own tuned setup (lr, steps, batch size, num_timesteps,
scheduler from exp/<ds>/ddpm_cb_best/config.toml; --train_setup tabddpm)
closes its utility gap to TabDDPM.

Per dataset, four columns:
  current   the fixed-rule run the current analysis uses (catfix tree for
            categorical datasets; ablation_fixed_seed0 / original tree for the
            numerical ones), ablation setup (lr 1e-4, 20 000 steps)
  budget    the same configuration with --train_setup tabddpm
            (exp/<ds>/ablation_tddpm_setup[_catfix]/<same combo>/)
  TabDDPM   the authors' exp/<ds>/ddpm_cb_best/eval_catboost.json (synthetic)
  Real      the same file's "real" block (CatBoost on real train)

Utility: macro-F1 / R2, mean +- std over the 50 runs (np.std, ddof=0, the
authors' convention). z = delta / sqrt(sd_a^2 + sd_b^2); tie if |z| < 1.
Privacy: DCR (mean L2), DCR baseline (median + label, the recipe of the
authors' privacy.json), my DCR baseline / the authors' privacy.json DCR, MIA AUC.
Dependency fidelity: Column Pair Trends (shared real bins) and association
distance, from analysis/dependency_fidelity/dependency_fidelity.py.
Training: final loss (mean of the last 10 logged values), loss curves.

Missing budget runs are reported as pending (diverged runs as diverged, from
divergence.json); nothing is imputed. Outputs in this directory.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "analysis" / "topk_comparison"), str(REPO / "analysis" / "dependency_fidelity"),
                str(REPO / "ablation")]
import compare_topk as ct  # noqa: E402

RESULTS = Path(os.environ.get("COMPARE_SETUP_RESULTS_ROOT", REPO / "transfer_runs" / "exp"))
OUT = Path(os.environ.get("COMPARE_SETUP_OUT", HERE))
# dataset -> tree of the CURRENT fixed run (as the current analysis uses it), cat_input
DATASETS = {
    "cardio": ("ablation_catfix", "prob"), "house": ("ablation", "log"), "king": ("ablation_catfix", "prob"),
    "gesture": ("ablation_fixed_seed0", "log"), "california": ("ablation_fixed_seed0", "log"),
    "higgs-small": ("ablation", "log"), "miniboone": ("ablation", "log"),
}
OPTIONAL = {"higgs-small", "miniboone"}      # expensive; shown only when a run exists
HELD_OUT = {"cardio", "house", "king", "higgs-small", "miniboone"}
C_CUR, C_BUD, C_TAB = "#2a78d6", "#eb6834", "#52514e"


def budget_tree(cat: str) -> str:
    from ablation_runner import results_subdir
    return results_subdir("fixed", cat, "tabddpm")


def run_dir(ds: str, tree: str) -> Path | None:
    c = sorted(p for p in (RESULTS / ds / tree).glob("gnn_*") if p.is_dir())
    return c[0] if c else None


def status(p: Path | None) -> str:
    if p is None:
        return "pending (no run)"
    if (p / "divergence.json").exists():
        d = json.load(open(p / "divergence.json"))
        return f"diverged at step {d['step']} ({d['reason']})"
    n = ct.load_run_dir(p)["n_gen"]
    return "complete" if n >= ct.EXPECTED_GEN else f"pending ({n}/{ct.EXPECTED_GEN} gen seeds)"


def utility(p: Path, task: str) -> dict:
    import paper_comparison as pc
    v = pc.raw_runs(p, "r2" if task == "regression" else "macro avg/f1-score")
    return {"mean": float(v.mean()), "std": float(v.std()), "n": len(v)}


def gen_metrics(p: Path) -> pd.DataFrame:
    return ct.load_run_dir(p)["per_gen"]


def z(a: dict, b: dict) -> tuple[float, float, str]:
    d = a["mean"] - b["mean"]
    s = float(np.hypot(a["std"], b["std"]))
    zz = d / s if s > 0 else np.nan
    return d, zz, ("tie" if abs(zz) < 1 else ("better" if zz > 0 else "worse"))


def dependency(ds: str, runs: dict) -> pd.DataFrame:
    import dependency_fidelity as dfm
    rows = [{"dataset": ds, "run": "real test (floor)", **{k: v for k, v in dfm.job_floor(ds).items()
                                                          if not k.startswith("_")}}]
    for name, p in runs.items():
        for g in sorted(p.glob("gen_seed_*")):
            r = dfm.job_seed((ds, name, str(g)))
            rows.append({k: v for k, v in r.items() if not k.startswith("_")} | {"run": name})
    return pd.DataFrame(rows)


def final_loss(p: Path) -> dict:
    lo = pd.read_csv(p / "loss.csv")
    t = lo.tail(10)
    return {"final_mloss": t.mloss.mean(), "final_gloss": t.gloss.mean(), "final_loss": t.loss.mean(),
            "steps": int(lo.step.iloc[-1])}


def figure_losses(curves: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not curves:
        return
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.0), squeeze=False)
    for ax, (ds, cs) in zip(axes[0], curves.items()):
        for name, lo in cs.items():
            ax.plot(lo.step, lo.loss, lw=1.6, color=C_CUR if name == "current" else C_BUD,
                    label="current (lr 1e-4, 20k)" if name == "current" else "budget-matched")
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(ds + (" (held-out)" if ds in HELD_OUT else ""), fontsize=9, color="#0b0b0b", loc="left")
        ax.set_xlabel("step", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color="#e4e3df", lw=0.7)
        ax.tick_params(labelsize=7, colors="#52514e")
    axes[0][0].set_ylabel("train loss (mloss + gloss)", fontsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "loss_curves.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    order = sorted(DATASETS, key=lambda d: (d not in HELD_OUT, d))
    util_rows, priv_rows, loss_rows, dep_frames, curves, st_rows = [], [], [], [], {}, []
    for ds in order:
        tree, cat = DATASETS[ds]
        cur, bud = run_dir(ds, tree), run_dir(ds, budget_tree(cat))
        s_cur, s_bud = status(cur), status(bud)
        if ds in OPTIONAL and bud is None:
            st_rows.append({"dataset": ds, "current": s_cur, "budget": "not submitted (optional)"})
            continue
        st_rows.append({"dataset": ds, "current": s_cur, "budget": s_bud,
                        "current_dir": str(cur.relative_to(RESULTS)) if cur else "",
                        "budget_dir": str(bud.relative_to(RESULTS)) if bud else ""})
        task = json.load(open(REPO / "data" / ds / "info.json"))["task_type"]
        ev = json.load(open(REPO / "exp" / ds / "ddpm_cb_best" / "eval_catboost.json"))
        k = "r2" if task == "regression" else "f1"
        tab = {"mean": ev["synthetic"]["test"][f"{k}-mean"], "std": ev["synthetic"]["test"][f"{k}-std"], "n": 50}
        real = {"mean": ev["real"]["test"][f"{k}-mean"], "std": ev["real"]["test"][f"{k}-std"], "n": 10}
        u = {"current": utility(cur, task) if s_cur == "complete" else None,
             "budget": utility(bud, task) if s_bud == "complete" else None}
        row = {"dataset": ds, "subset": "held-out" if ds in HELD_OUT else "development",
               "metric": "R2" if k == "r2" else "macro-F1",
               "tabddpm": tab["mean"], "tabddpm_std": tab["std"], "real": real["mean"], "real_std": real["std"]}
        for name in ("current", "budget"):
            x = u[name]
            row[f"{name}"] = x["mean"] if x else np.nan
            row[f"{name}_std"] = x["std"] if x else np.nan
            row[f"{name}_n"] = x["n"] if x else 0
            if x:
                row[f"{name}_vs_tabddpm"], row[f"{name}_vs_tabddpm_z"], row[f"{name}_vs_tabddpm_outcome"] = z(x, tab)
        if u["current"] and u["budget"]:
            row["budget_vs_current"], row["budget_vs_current_z"], row["budget_vs_current_outcome"] = \
                z(u["budget"], u["current"])
            gap = tab["mean"] - u["current"]["mean"]
            row["gap_closed_frac"] = ((u["budget"]["mean"] - u["current"]["mean"]) / gap + 0.0) if gap != 0 else np.nan
        util_rows.append(row)

        priv = json.load(open(REPO / "exp" / ds / "ddpm_cb_best" / "privacy.json"))["privacy"]
        for name, p, s in (("current", cur, s_cur), ("budget", bud, s_bud)):
            if s != "complete":
                continue
            g = gen_metrics(p)
            pr = {"dataset": ds, "run": name}
            for m in ("dcr_tabddpm_mean", "dcr_baseline", "mia_auc"):
                if m in g:
                    pr[m] = g[m].mean()
                    pr[m + "_sd"] = g[m].std(ddof=1)
            pr["tabddpm_privacy_json_dcr"] = priv
            pr["dcr_baseline_over_tabddpm"] = pr.get("dcr_baseline", np.nan) / priv
            priv_rows.append(pr)
            loss_rows.append({"dataset": ds, "run": name, **final_loss(p)})
            curves.setdefault(ds, {})[name] = pd.read_csv(p / "loss.csv")
        runs = {n: p for n, p, s in (("current", cur, s_cur), ("budget", bud, s_bud)) if s == "complete"}
        if runs:
            dep_frames.append(dependency(ds, runs))
        print(f"{ds}: current {s_cur}; budget {s_bud}", flush=True)

    U, P, Lf = pd.DataFrame(util_rows), pd.DataFrame(priv_rows), pd.DataFrame(loss_rows)
    D = pd.concat(dep_frames, ignore_index=True) if dep_frames else pd.DataFrame()
    S = pd.DataFrame(st_rows)
    U.to_csv(OUT / "utility.csv", index=False)
    P.to_csv(OUT / "privacy.csv", index=False)
    Lf.to_csv(OUT / "training.csv", index=False)
    D.to_csv(OUT / "dependency_per_seed.csv", index=False)
    S.to_csv(OUT / "status.csv", index=False)
    figure_losses(curves)
    write_report(U, P, Lf, D, S)
    print(f"wrote {OUT}")


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def write_report(U, P, Lf, D, S):
    L = ["# Budget-matched GNN runs: does TabDDPM's training setup close the gap?\n",
         "_Generated by `compare_setup.py`. **Targeted check, not a full re-run**: fixed rule only, five datasets "
         "(held-out cardio, house, king; development gesture, california; higgs-small and miniboone only if "
         "submitted). n is small; per-dataset results are descriptive. default is excluded (local data differs "
         "from the authors')._\n",
         "budget-matched = the same GNN configuration, cat_input, seeds, EMA, sampling size and CatBoost config "
         "as the current run, trained with the authors' lr / steps / batch size / num_timesteps / scheduler "
         "(`--train_setup tabddpm`). TabDDPM and Real = the authors' `exp/<ds>/ddpm_cb_best/eval_catboost.json`. "
         "z = Δ / √(sd₁² + sd₂²) over the 50 runs; tie if |z| < 1. The 50 runs are 5 clusters of 10, so z "
         "overstates precision; it is a scale, not a test.\n"]
    L.append("## Status\n")
    L.append("| dataset | current run | budget-matched run |")
    L.append("|---|---|---|")
    for _, r in S.iterrows():
        L.append(f"| {r.dataset} | {r.current} | {r.budget} |")
    L.append("")

    L.append("## Utility (held-out first)\n")
    L.append("| dataset | subset | metric | current | budget-matched | TabDDPM | Real | budget − current (z) | "
             "current − TabDDPM (z) | budget − TabDDPM (z) | gap closed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in U.iterrows():
        def pm(m, s):
            return f"{m:.4f} ± {s:.4f}" if np.isfinite(m) else "pending"

        def dz(a, b, o):
            return f"{a:+.4f} ({b:+.1f}, {o})" if isinstance(a, float) and np.isfinite(a) else "—"
        L.append(f"| {r.dataset} | {r.subset} | {r.metric} | {pm(r.current, r.current_std)} | "
                 f"{pm(r.budget, r.budget_std)} | {r.tabddpm:.4f} ± {r.tabddpm_std:.4f} | "
                 f"{r.real:.4f} ± {r.real_std:.4f} | "
                 f"{dz(r.get('budget_vs_current'), r.get('budget_vs_current_z'), r.get('budget_vs_current_outcome'))} | "
                 f"{dz(r.get('current_vs_tabddpm'), r.get('current_vs_tabddpm_z'), r.get('current_vs_tabddpm_outcome'))} | "
                 f"{dz(r.get('budget_vs_tabddpm'), r.get('budget_vs_tabddpm_z'), r.get('budget_vs_tabddpm_outcome'))} | "
                 f"{_f(r.get('gap_closed_frac'), 2)} |")
    L.append("\ngap closed = (budget − current) / (TabDDPM − current): 1 = reaches TabDDPM, 0 = no change, "
             "negative = moves away. Undefined when current already equals TabDDPM.\n")

    L.append("## Privacy (mean over gen seeds)\n")
    if P.empty:
        L.append("No complete budget-matched run yet.\n")
    else:
        L.append("| dataset | run | DCR (mean L2) | DCR baseline (median + label) | authors' privacy.json DCR | "
                 "DCR baseline / authors' | MIA AUC |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in P.iterrows():
            L.append(f"| {r.dataset} | {r.run} | {_f(r.get('dcr_tabddpm_mean'))} | {_f(r.get('dcr_baseline'))} | "
                     f"{_f(r.tabddpm_privacy_json_dcr)} | {_f(r.dcr_baseline_over_tabddpm, 2)} | {_f(r.get('mia_auc'))} |")
        L.append("\nThe authors' privacy.json uses the same recipe as `dcr_baseline` (median nearest-neighbour L2, "
                 "label included; `scripts/eval_tabddpm_protocol.compute_dcr_baseline`) but on their own synthetic sample, so the ratio compares the two generators' samples, not identical computations.\n")

    L.append("## Dependency fidelity (mean ± sd over gen seeds; floor = real test vs real train)\n")
    if D.empty:
        L.append("No runs to score.\n")
    else:
        L.append("| dataset | run | Column Pair Trends (↑) | association distance, mean abs (↓) |")
        L.append("|---|---|---|---|")
        for (ds, run), g in D.groupby(["dataset", "run"], sort=False):
            cpt, ass = g.column_pair_trends, g.assoc_mean_abs
            sd = (lambda x: f" ± {x.std(ddof=1):.4f}" if len(x) > 1 else "")
            L.append(f"| {ds} | {run} | {cpt.mean():.4f}{sd(cpt)} | {ass.mean():.4f}{sd(ass)} |")
        L.append("")

    L.append("## Training\n")
    if Lf.empty:
        L.append("No loss curves yet.\n")
    else:
        L.append("| dataset | run | steps | final mloss | final gloss | final loss |")
        L.append("|---|---|---|---|---|---|")
        for _, r in Lf.iterrows():
            L.append(f"| {r.dataset} | {r.run} | {r.steps} | {r.final_mloss:.4f} | {r.final_gloss:.4f} | "
                     f"{r.final_loss:.4f} |")
        L.append("\nFigure: `figures/loss_curves.png`. king's budget-matched run uses num_timesteps = 100 "
                 "(authors' config) vs 1000: the loss is averaged over a different t distribution, so its level "
                 "is not comparable with the current run's; compare shapes only.\n")

    L.append("## Summary\n")
    done = U[U.budget.notna()] if "budget" in U else U.iloc[0:0]
    if done.empty:
        L.append("**No budget-matched run has finished yet** — every comparison above is pending. Rerun this "
                 "script after syncing `exp/<ds>/ablation_tddpm_setup*/`.\n")
    else:
        for sub in ("held-out", "development"):
            d = done[done.subset == sub]
            if d.empty:
                continue
            L.append(f"**{sub}** ({len(d)} dataset(s)):")
            for _, r in d.iterrows():
                L.append(f"- {r.dataset}: current {r.current:.4f} → budget {r.budget:.4f} vs TabDDPM "
                         f"{r.tabddpm:.4f}; budget − current {r.budget_vs_current:+.4f} "
                         f"(z {r.budget_vs_current_z:+.1f}, {r.budget_vs_current_outcome}); budget vs TabDDPM "
                         f"{r.budget_vs_tabddpm_outcome} (z {r.budget_vs_tabddpm_z:+.1f}); gap closed "
                         f"{_f(r.get('gap_closed_frac'), 2)}.")
            L.append("")
        for ds in ("house", "gesture"):
            r = done[done.dataset == ds]
            if len(r):
                r = r.iloc[0]
                closes = r.budget_vs_tabddpm_outcome in ("tie", "better")
                L.append(f"- **{ds}**: the gap to TabDDPM {'closes' if closes else 'does not close'} "
                         f"(current {r.current_vs_tabddpm:+.4f} → budget {r.budget_vs_tabddpm:+.4f}).")
        L.append("\nWith one training seed per setup and n ≤ 5 datasets this is a targeted check of the "
                 "training-setup explanation, not an estimate of the GNN's general performance.\n")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

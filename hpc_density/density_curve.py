#!/usr/bin/env python3
"""Density curve: fixed top-k at a grid of k vs the three adaptive rules.

For each dataset (one fixed GATv2 architecture, hpc_density/density_manifest.py):

    fixed top-k at every grid k  ->  the curve "metric vs k"
    adaptive_jenks / _sparsemax / _entmax15  ->  points at their achieved mean k

An adaptive point BELOW the fixed-k curve on Wasserstein (lower is better) means
the rule adds something beyond the density it lands on; ON the curve means it is
just choosing a density.

Reads density_runs/exp/<ds>/ablation[_<mode>]/<combo>/ only (no reused runs):
  results_full_averaged.json, per_seed_scores.csv, topk_stats.csv (adaptive).

  mean k      fixed: k (dense = N-1). Adaptive: sampling-phase k_mean from
              topk_stats.csv, n_samples-weighted; its std is over generation
              seeds (the tag column).
  metrics     mean +/- std over GENERATION seeds (population std, as
              ablation_runner): per_seed_scores.csv gives one value per
              generation seed (score: mean over its 10 classifier seeds). If a
              run has no per-seed file, the JSON is used and flagged.

Verdict per adaptive point and metric:
  curve       linear interpolation in log k between the two fixed points that
              bracket the adaptive mean k (the std is interpolated the same
              way). No extrapolation: outside the fixed k range -> "outside
              the curve".
  position    below / on / above the curve; "on" when |difference| is smaller
              than the larger of the two stds (adaptive, interpolated curve).
  effect      better / same / worse in the metric's direction: lower is better
              for wasserstein_mean and mia_auc, higher for score and
              dcr_tabddpm_mean.
  paired      vs the nearest fixed-k condition (in log k), over the generation
              seeds both runs share: mean paired difference (adaptive - fixed)
              and a sign count. No p-values anywhere.
  Primary metric: wasserstein_mean. score is secondary. DCR is context only,
  reported next to Wasserstein, never a privacy claim on its own.

Outputs (default <repo_root>/hpc_density/results/; never under hpc/,
hpc_entmax/ or hpc_wide/):
  density_points.csv     one row per (dataset, condition)
  density_verdicts.csv   one row per (dataset, adaptive rule, metric)
  figures/density_<ds>.{png,pdf}   Wasserstein and score vs mean k (log)
  figures/density_all.{png,pdf}    Wasserstein vs k/(N-1), one panel per dataset

    python3 hpc_density/density_curve.py --repo_root "$REPO_ROOT"
    python3 hpc_density/density_curve.py --repo_root "$REPO_ROOT" --datasets wilt abalone --no_figures

Stdlib + matplotlib (figures only; skipped with a warning if missing).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

NAN = float("nan")

# (metric, +1 = higher is better / -1 = lower is better, role)
METRICS = (
    ("wasserstein_mean", -1, "primary"),
    ("score", +1, "secondary"),
    ("dcr_tabddpm_mean", +1, "context (next to Wasserstein; not a privacy claim)"),
    ("mia_auc", -1, "context"),
)
DIRECTION = {m: d for m, d, _ in METRICS}
ROLE = {m: r for m, _, r in METRICS}

LABEL = {"adaptive_jenks": "Jenks", "adaptive_sparsemax": "sparsemax", "adaptive_entmax15": "entmax-1.5"}

# Validated categorical slots 1-3 (dataviz reference palette, light mode, all
# pairs: CVD dE 9.2, normal-vision dE 24.0). Aqua is below 3:1 contrast, so
# every adaptive point also carries a direct label and a distinct marker.
STYLE = {
    "fixed":              {"color": "#52514e", "marker": "o", "label": "fixed top-k"},
    "adaptive_jenks":     {"color": "#2a78d6", "marker": "s", "label": "Jenks"},
    "adaptive_sparsemax": {"color": "#eb6834", "marker": "^", "label": "sparsemax"},
    "adaptive_entmax15":  {"color": "#1baf7a", "marker": "D", "label": "entmax-1.5"},
}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _pstd(values) -> float:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else NAN)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_per_seed(path: str):
    """{metric: {gen_seed: value}} -- classifier metrics averaged over clf seeds."""
    if not os.path.exists(path):
        return None
    gen_level = defaultdict(dict)
    clf_level = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                g, v = int(row["gen_seed"]), float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if row.get("clf_seed", "") == "":
                gen_level[row["metric"]][g] = v
            else:
                clf_level[row["metric"]][g].append(v)
    out = {m: dict(d) for m, d in gen_level.items()}
    for m, by_gen in clf_level.items():
        out[m] = {g: statistics.fmean(vs) for g, vs in by_gen.items()}
    return out


def read_budget(path: str):
    """(n_samples-weighted sampling k_mean, its std over generation seeds)."""
    if not os.path.exists(path):
        return NAN, NAN
    total = count = 0.0
    by_tag = defaultdict(lambda: [0.0, 0.0])
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("phase") != "sample":
                continue
            try:
                k, n = float(row["k_mean"]), float(row["n_samples"])
            except (KeyError, TypeError, ValueError):
                continue
            total += k * n
            count += n
            by_tag[row.get("tag", "")][0] += k * n
            by_tag[row.get("tag", "")][1] += n
    if not count:
        return NAN, NAN
    per_seed = [s / n for s, n in by_tag.values() if n]
    return total / count, (_pstd(per_seed) if len(per_seed) > 1 else NAN)


def load_point(repo_root: str, runs_root: str, dm, dataset: str, condition: str) -> dict:
    r = dm.resolve(repo_root, dataset, condition)
    run_dir = os.path.join(runs_root, "exp", dataset, dm.run_subdir(r), dm.exp_dir_name(r))
    point = {"dataset": dataset, "condition": condition, "topk_mode": r["topk_mode"],
             "top_k": r["top_k"], "n_nodes": r["n_nodes"], "run_dir": run_dir,
             "status": "missing", "metrics": {}, "per_seed": False,
             # a fixed condition's k is known before it runs; an adaptive one's is measured
             "mean_k": float(r["k"]) if r["topk_mode"] == "fixed" else NAN,
             "mean_k_std": 0.0 if r["topk_mode"] == "fixed" else NAN}
    results = os.path.join(run_dir, "results_full_averaged.json")
    if not os.path.exists(results):
        return point
    with open(results) as fh:
        agg = json.load(fh)
    per_seed = read_per_seed(os.path.join(run_dir, "per_seed_scores.csv"))
    point["status"], point["per_seed"] = "done", per_seed is not None
    if r["topk_mode"] != "fixed":
        point["mean_k"], point["mean_k_std"] = read_budget(os.path.join(run_dir, "topk_stats.csv"))
    for metric, _, _ in METRICS:
        seeds = (per_seed or {}).get(metric)
        if seeds:
            values = [seeds[g] for g in sorted(seeds)]
            point["metrics"][metric] = {"mean": statistics.fmean(values), "std": _pstd(values),
                                        "n": len(values), "by_seed": seeds, "basis": "per generation seed"}
        elif isinstance(agg.get(metric), dict):
            m, s = agg[metric].get("mean", NAN), agg[metric].get("std", NAN)
            point["metrics"][metric] = {"mean": m if _finite(m) else NAN, "std": s if _finite(s) else NAN,
                                        "n": None, "by_seed": {}, "basis": "results json (no per-seed file)"}
    return point


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def verdict(fixed_points: list, point: dict, metric: str) -> dict:
    direction = DIRECTION[metric]
    out = {"metric": metric, "role": ROLE[metric], "better": "higher" if direction > 0 else "lower",
           "mean_k": point.get("mean_k", NAN), "k_lo": NAN, "k_hi": NAN,
           "curve_mean": NAN, "curve_std": NAN, "adaptive_mean": NAN, "adaptive_std": NAN,
           "difference": NAN, "position": "", "effect": "n/a", "nearest_fixed": "",
           "paired_mean_diff": NAN, "n_pairs": 0, "n_better": 0, "n_worse": 0, "n_equal": 0}
    a = point["metrics"].get(metric)
    if point["status"] != "done" or not a or not _finite(a["mean"]):
        out["position"] = "no result"
        return out
    out["adaptive_mean"], out["adaptive_std"] = a["mean"], a["std"]
    curve = sorted((p["mean_k"], p["metrics"][metric], p["condition"]) for p in fixed_points
                   if p["status"] == "done" and metric in p["metrics"]
                   and _finite(p["metrics"][metric]["mean"]))
    k = point.get("mean_k", NAN)
    if len(curve) < 2:
        out["position"] = "fewer than 2 fixed points"
        return out
    if not _finite(k) or k <= 0:
        out["position"] = "no budget logged"
        return out

    # Paired comparison vs the nearest fixed-k condition (log k), whatever the position.
    k_near, f_near, c_near = min(curve, key=lambda c: (abs(math.log(c[0]) - math.log(k)), c[0]))
    out["nearest_fixed"] = c_near
    shared = sorted(set(a["by_seed"]) & set(f_near["by_seed"]))
    if shared:
        diffs = [a["by_seed"][g] - f_near["by_seed"][g] for g in shared]
        out["paired_mean_diff"] = statistics.fmean(diffs)
        out["n_pairs"] = len(diffs)
        out["n_better"] = sum(1 for d in diffs if d * direction > 0)
        out["n_worse"] = sum(1 for d in diffs if d * direction < 0)
        out["n_equal"] = sum(1 for d in diffs if d == 0)

    if k < curve[0][0] or k > curve[-1][0]:
        out["position"] = "outside the curve"
        return out
    for (k_lo, f_lo, _), (k_hi, f_hi, _) in zip(curve, curve[1:]):
        if k_lo <= k <= k_hi:
            break
    w = 0.0 if k_hi == k_lo else (math.log(k) - math.log(k_lo)) / (math.log(k_hi) - math.log(k_lo))
    c_mean = f_lo["mean"] + w * (f_hi["mean"] - f_lo["mean"])
    c_std = (f_lo["std"] if _finite(f_lo["std"]) else 0.0) + w * (
        (f_hi["std"] if _finite(f_hi["std"]) else 0.0) - (f_lo["std"] if _finite(f_lo["std"]) else 0.0))
    diff = a["mean"] - c_mean
    tol = max(a["std"] if _finite(a["std"]) else 0.0, c_std)
    out.update({"k_lo": k_lo, "k_hi": k_hi, "curve_mean": c_mean, "curve_std": c_std, "difference": diff})
    if abs(diff) < tol or diff == 0:      # identical (e.g. both stds 0) is on the curve
        out["position"], out["effect"] = "on", "same"
    else:
        out["position"] = "below" if diff < 0 else "above"
        out["effect"] = "better" if diff * direction > 0 else "worse"
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def _plot_panel(ax, fixed, adaptive, metric, x_of, title, x_label, y_label, label_points=True):
    from matplotlib.ticker import FixedLocator, NullLocator
    _style_axes(ax)
    pts = sorted((x_of(p), p["metrics"][metric]["mean"], p["metrics"][metric]["std"])
                 for p in fixed if p["status"] == "done" and metric in p["metrics"])
    handles, labels = [], []
    if pts:
        xs, ys, es = zip(*pts)
        st = STYLE["fixed"]
        h = ax.errorbar(xs, ys, yerr=[e if _finite(e) else 0 for e in es], color=st["color"],
                        marker=st["marker"], markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.0,
                        linewidth=1.5, elinewidth=1.0, capsize=0, label=st["label"], zorder=2)
        handles.append(h)
    for p in adaptive:
        if p["status"] != "done" or metric not in p["metrics"] or not _finite(p.get("mean_k", NAN)):
            continue
        st = STYLE[p["condition"]]
        x, y = x_of(p), p["metrics"][metric]["mean"]
        xerr = p["mean_k_std"] * x / p["mean_k"] if _finite(p.get("mean_k_std", NAN)) else None
        yerr = p["metrics"][metric]["std"] if _finite(p["metrics"][metric]["std"]) else None
        h = ax.errorbar([x], [y], xerr=None if xerr is None else [xerr], yerr=None if yerr is None else [yerr],
                        color=st["color"], marker=st["marker"], markersize=7, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, linestyle="none", elinewidth=1.0, capsize=0,
                        label=st["label"], zorder=3)
        handles.append(h)
        if label_points:
            labels.append((st["label"], x, y))
    ax.set_xscale("log")
    ticks = sorted({x_of(p) for p in fixed if p["status"] == "done"})
    if ticks:
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    # Direct labels after the scale is final: a point in the right quarter of
    # the axis gets its label on the left, so it never runs past the edge.
    lo, hi = (math.log(v) for v in ax.get_xlim())
    for text, x, y in labels:
        right = hi > lo and (math.log(x) - lo) / (hi - lo) > 0.75
        ax.annotate(text, (x, y), xytext=(-6 if right else 6, 5), textcoords="offset points",
                    ha="right" if right else "left", fontsize=7.5, color=INK, zorder=4)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(x_label, fontsize=8, color=INK_2)
    ax.set_ylabel(y_label, fontsize=8, color=INK_2)
    return handles


def make_figures(points_by_ds: dict, fig_dir: str) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available -- figures skipped")
        return []
    os.makedirs(fig_dir, exist_ok=True)
    written = []
    y_lab = {"wasserstein_mean": "Wasserstein (lower is better)",
             "score": "score (higher is better)"}
    name = {"wasserstein_mean": "Wasserstein", "score": "score"}
    for ds, pts in points_by_ds.items():
        fixed = [p for p in pts if p["topk_mode"] == "fixed"]
        adaptive = [p for p in pts if p["topk_mode"] != "fixed"]
        N = pts[0]["n_nodes"]
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), facecolor=SURFACE)
        handles = []
        for ax, metric in zip(axes, ("wasserstein_mean", "score")):
            handles = _plot_panel(ax, fixed, adaptive, metric, lambda p: p["mean_k"],
                                  f"{ds} (N = {N}): {name[metric]}",
                                  f"mean kept neighbours k (log; dense = {N - 1})", y_lab[metric]) or handles
        if handles:
            fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="upper center",
                       ncol=len(handles), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        for ext in ("png", "pdf"):
            path = os.path.join(fig_dir, f"density_{ds}.{ext}")
            fig.savefig(path, dpi=200, facecolor=SURFACE)
            written.append(path)
        plt.close(fig)

    # Combined: one panel per dataset (Wasserstein scales differ), x = k/(N-1).
    n = len(points_by_ds)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.1), facecolor=SURFACE, squeeze=False)
        handles = []
        for ax, (ds, pts) in zip(axes[0], points_by_ds.items()):
            N = pts[0]["n_nodes"]
            frac = lambda p, N=N: p["mean_k"] / (N - 1)
            fixed = [p for p in pts if p["topk_mode"] == "fixed"]
            adaptive = [p for p in pts if p["topk_mode"] != "fixed"]
            h = _plot_panel(ax, fixed, adaptive, "wasserstein_mean", frac, f"{ds} (N = {N})",
                            "k / (N-1)  (log)", "Wasserstein" if ax is axes[0][0] else "",
                            label_points=False)
            handles = h if len(h) > len(handles) else handles
            ax.set_xticklabels([f"{t:.2g}" for t in sorted({frac(p) for p in fixed if p["status"] == "done"})])
        if handles:
            fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="upper center",
                       ncol=len(handles), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        for ext in ("png", "pdf"):
            path = os.path.join(fig_dir, f"density_all.{ext}")
            fig.savefig(path, dpi=200, facecolor=SURFACE)
            written.append(path)
        plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", required=True)
    p.add_argument("--runs_root", default=None, help="default: <repo_root>/density_runs")
    p.add_argument("--out_dir", default=None, help="default: <repo_root>/hpc_density/results")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--no_figures", action="store_true")
    args = p.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    runs_root = os.path.abspath(args.runs_root or os.path.join(repo_root, "density_runs"))
    out_dir = os.path.abspath(args.out_dir or os.path.join(repo_root, "hpc_density", "results"))
    for protected in ("hpc", "hpc_entmax", "hpc_wide"):
        root = os.path.join(repo_root, protected)
        if os.path.commonpath([out_dir, root]) == root:
            p.error(f"refusing to write under {protected}/")
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(repo_root, "hpc_density"))
    import density_manifest as dm
    datasets = args.datasets or list(dm.DATASETS)

    points_by_ds, verdict_rows = {}, []
    for ds in datasets:
        pts = [load_point(repo_root, runs_root, dm, ds, c) for c in dm.conditions(repo_root, ds)]
        points_by_ds[ds] = pts
        fixed = [q for q in pts if q["topk_mode"] == "fixed"]
        for q in pts:
            if q["topk_mode"] == "fixed":
                continue
            for metric, _, _ in METRICS:
                verdict_rows.append({"dataset": ds, "condition": q["condition"], **verdict(fixed, q, metric)})

    # ---- CSVs ------------------------------------------------------------------
    point_cols = ["dataset", "n_nodes", "condition", "topk_mode", "top_k", "status", "mean_k",
                  "mean_k_std", "k_frac", "per_seed"]
    for metric, _, _ in METRICS:
        point_cols += [f"{metric}_mean", f"{metric}_std", f"{metric}_n_gen_seeds", f"{metric}_basis"]
    point_cols.append("run_dir")

    def fmt(v):
        return "" if isinstance(v, float) and math.isnan(v) else v

    points_csv = os.path.join(out_dir, "density_points.csv")
    with open(points_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=point_cols)
        w.writeheader()
        for pts in points_by_ds.values():
            for q in pts:
                row = {c: q.get(c, "") for c in ("dataset", "n_nodes", "condition", "topk_mode",
                                                  "top_k", "status", "per_seed", "run_dir")}
                mk = q.get("mean_k", NAN)
                row.update({"mean_k": fmt(mk), "mean_k_std": fmt(q.get("mean_k_std", NAN)),
                            "k_frac": fmt(mk / (q["n_nodes"] - 1)) if _finite(mk) else ""})
                for metric, _, _ in METRICS:
                    m = q["metrics"].get(metric, {})
                    row.update({f"{metric}_mean": fmt(m.get("mean", NAN)), f"{metric}_std": fmt(m.get("std", NAN)),
                                f"{metric}_n_gen_seeds": m.get("n") or "", f"{metric}_basis": m.get("basis", "")})
                w.writerow(row)

    verdict_cols = ["dataset", "condition", "metric", "role", "better", "mean_k", "k_lo", "k_hi",
                    "curve_mean", "curve_std", "adaptive_mean", "adaptive_std", "difference",
                    "position", "effect", "nearest_fixed", "paired_mean_diff", "n_pairs",
                    "n_better", "n_worse", "n_equal"]
    verdicts_csv = os.path.join(out_dir, "density_verdicts.csv")
    with open(verdicts_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=verdict_cols)
        w.writeheader()
        for row in verdict_rows:
            w.writerow({c: fmt(row.get(c, "")) for c in verdict_cols})

    figures = [] if args.no_figures else make_figures(points_by_ds, os.path.join(out_dir, "figures"))

    # ---- summary ---------------------------------------------------------------
    def f(x, d=4):
        return f"{x:.{d}f}" if _finite(x) else "--"

    print("=" * 100)
    print("  Density curve: fixed top-k vs adaptive rules at the same mean k")
    print("  Primary: wasserstein_mean (lower is better). Mean +/- std over generation seeds. No p-values.")
    print("=" * 100)
    for ds, pts in points_by_ds.items():
        done = sum(q["status"] == "done" for q in pts)
        print(f"\n  {ds}  (N = {pts[0]['n_nodes']}, {done}/{len(pts)} conditions done)")
        for q in pts:
            m = q["metrics"].get("wasserstein_mean", {})
            print(f"    {q['condition']:19s} k = {f(q.get('mean_k', NAN), 2):>6s}   Wasserstein "
                  f"{f(m.get('mean', NAN))} +/- {f(m.get('std', NAN))}   {q['status']}"
                  + ("" if q["status"] != "done" or q["per_seed"] else "   (no per-seed file)"))
        for row in verdict_rows:
            if row["dataset"] != ds or row["metric"] not in ("wasserstein_mean", "score"):
                continue
            paired = (f"   paired vs {row['nearest_fixed']}: {f(row['paired_mean_diff'])} "
                      f"(better {row['n_better']} / worse {row['n_worse']} / equal {row['n_equal']} "
                      f"of {row['n_pairs']})" if row["n_pairs"] else "")
            print(f"    -> {LABEL[row['condition']]:10s} {row['metric']:17s} {row['position']:26s} "
                  f"{row['effect']:6s} diff {f(row['difference'])}{paired}")
    print()
    for path in [points_csv, verdicts_csv] + figures:
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

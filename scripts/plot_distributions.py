"""
Reproduce the TabDDPM paper's "additional visualizations" for this thesis:

  Figure 5 style : per-feature marginal distributions, real vs each synthetic
                   generator (one column per generator, one row per feature).
  Figure 6 style : |corr(real) - corr(synthetic)| heat maps, one row per
                   generator, one column per dataset.

Real data is read from  data/<dataset>/{X_num_train,X_cat_train,y_train}.npy
Synthetic data is read from the results directory, which may be laid out either
as one generator per dataset:

    results_plots/<dataset>_plots/{X_num_train,X_cat_train,y_train}.npy

or as several named generators per dataset (needed to compare GNN vs TabDDPM):

    results_plots/<dataset>_plots/<method>/{X_num_train,X_cat_train,y_train}.npy

Usage
-----
    python scripts/plot_distributions.py                     # both figures
    python scripts/plot_distributions.py --only marginals
    python scripts/plot_distributions.py --datasets abalone wilt
"""

import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Colours follow the paper: real in muted blue, synthetic in salmon.
REAL_COLOR = "#4c5aa8"
SYN_COLOR = "#f08a6c"

# A generator directory must contain this file to be recognised.
_MARKER = "X_num_train.npy"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _load_split(path):
    """Load {X_num, X_cat, y} from a directory. Missing pieces come back None."""
    out = {}
    for key, fname in (("X_num", "X_num_train.npy"),
                       ("X_cat", "X_cat_train.npy"),
                       ("y", "y_train.npy")):
        fpath = os.path.join(path, fname)
        out[key] = np.load(fpath, allow_pickle=True) if os.path.exists(fpath) else None
    return out


def discover(results_dir, default_method):
    """Map dataset name -> {method name: directory}."""
    found = {}
    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry)
        if not os.path.isdir(path):
            continue
        dataset = entry[: -len("_plots")] if entry.endswith("_plots") else entry

        methods = {}
        for sub in sorted(os.listdir(path)):
            subpath = os.path.join(path, sub)
            if os.path.isdir(subpath) and os.path.exists(os.path.join(subpath, _MARKER)):
                methods[sub] = subpath
        # Flat layout: the dataset directory is itself a single generator.
        if not methods and os.path.exists(os.path.join(path, _MARKER)):
            methods[default_method] = path

        if methods:
            found[dataset] = methods
    return found


def feature_names(real, include_y, task_type):
    """Panel labels, in the column order used everywhere below."""
    names = []
    if real["X_num"] is not None:
        names += [f"num $N^o${i}" for i in range(real["X_num"].shape[1])]
    if real["X_cat"] is not None:
        names += [f"cat $N^o${i}" for i in range(real["X_cat"].shape[1])]
    if include_y and real["y"] is not None:
        names.append("y (target)" if task_type == "regression" else "y (label)")
    return names


def columns(split, include_y, task_type):
    """
    Yield (values, is_categorical) per feature, in the same order as
    feature_names. The label of a classification task is treated as categorical.
    """
    cols = []
    if split["X_num"] is not None:
        for i in range(split["X_num"].shape[1]):
            cols.append((np.asarray(split["X_num"][:, i], dtype=float), False))
    if split["X_cat"] is not None:
        for i in range(split["X_cat"].shape[1]):
            cols.append((split["X_cat"][:, i], True))
    if include_y and split["y"] is not None:
        y = split["y"]
        if task_type == "regression":
            cols.append((np.asarray(y, dtype=float), False))
        else:
            cols.append((np.asarray(y).astype(str), True))
    return cols


# --------------------------------------------------------------------------- #
# figure 5 : marginal distributions
# --------------------------------------------------------------------------- #
def _plot_numeric(ax, real_vals, syn_vals, bins):
    """Overlaid density histograms on a shared, real-data-derived binning."""
    lo, hi = np.percentile(real_vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(real_vals)), float(np.max(real_vals))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)

    # Clip rather than drop, so tail mass stays visible in the edge bins.
    ax.hist(np.clip(real_vals, lo, hi), bins=edges, density=True,
            color=REAL_COLOR, alpha=0.75, edgecolor="none")
    ax.hist(np.clip(syn_vals, lo, hi), bins=edges, density=True,
            color=SYN_COLOR, alpha=0.70, edgecolor="none")
    ax.set_xlim(lo, hi)


def _plot_categorical(ax, real_vals, syn_vals):
    """Side-by-side normalised frequency bars over the union of categories."""
    cats = sorted(set(np.unique(real_vals).tolist()) | set(np.unique(syn_vals).tolist()),
                  key=str)
    idx = np.arange(len(cats))

    def freq(vals):
        counts = {c: 0 for c in cats}
        u, n = np.unique(vals, return_counts=True)
        for cat, cnt in zip(u.tolist(), n.tolist()):
            counts[cat] = cnt
        total = max(len(vals), 1)
        return np.array([counts[c] / total for c in cats])

    width = 0.4
    ax.bar(idx - width / 2, freq(real_vals), width, color=REAL_COLOR, alpha=0.85)
    ax.bar(idx + width / 2, freq(syn_vals), width, color=SYN_COLOR, alpha=0.85)
    ax.set_xticks(idx)
    if len(cats) <= 8:
        ax.set_xticklabels([str(c)[:8] for c in cats], fontsize=5, rotation=0)
    else:
        ax.set_xticklabels([])


def plot_marginals(dataset, real, methods, task_type, args, out_dir):
    names = feature_names(real, args.include_y, task_type)
    real_cols = columns(real, args.include_y, task_type)
    syn_cols = {m: columns(methods[m]["data"], args.include_y, task_type)
                for m in methods}
    method_names = list(methods)

    # Wrap the feature list into side-by-side blocks, as the paper does, so a
    # wide dataset does not become one unreadably tall column.
    n_groups = max(1, min(args.groups, len(names)))
    rows_per_group = -(-len(names) // n_groups)
    n_rows = rows_per_group
    n_cols = n_groups * len(method_names)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(args.panel_w * n_cols + 0.9, args.panel_h * n_rows + 0.9),
        squeeze=False,
    )

    for g in range(n_groups):
        for r in range(rows_per_group):
            f = g * rows_per_group + r
            for k, mname in enumerate(method_names):
                ax = axes[r][g * len(method_names) + k]
                if f >= len(names):
                    ax.axis("off")
                    continue

                real_vals, is_cat = real_cols[f]
                syn_vals = syn_cols[mname][f][0]
                if is_cat:
                    _plot_categorical(ax, real_vals, syn_vals)
                else:
                    _plot_numeric(ax, real_vals, syn_vals, args.bins)

                ax.set_yticks([])
                ax.tick_params(axis="x", labelsize=5, length=2, pad=1)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.grid(axis="x", lw=0.3, alpha=0.25)
                if r == 0:
                    ax.set_title(mname, fontsize=8, pad=6)
                if k == 0:
                    ax.set_ylabel(names[f], fontsize=6, rotation=0,
                                  ha="right", va="center", labelpad=8)

    handles = [Patch(facecolor=REAL_COLOR, alpha=0.8, label="Real"),
               Patch(facecolor=SYN_COLOR, alpha=0.8, label="Synthetic")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{dataset}: feature distributions, real vs synthetic",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.98))
    _save(fig, out_dir, f"marginals_{dataset}", args)


# --------------------------------------------------------------------------- #
# figure 6 : correlation-difference heat maps
# --------------------------------------------------------------------------- #
def _corr_matrix(split, include_y, task_type, cat_maps):
    """
    Pearson correlation over all features. Categorical columns are ordinal
    encoded against a shared category ordering so real and synthetic matrices
    are index-comparable.
    """
    blocks = []
    if split["X_num"] is not None:
        blocks.append(np.asarray(split["X_num"], dtype=float))
    if split["X_cat"] is not None:
        enc = np.zeros(split["X_cat"].shape, dtype=float)
        for i in range(split["X_cat"].shape[1]):
            lookup = cat_maps[i]
            enc[:, i] = [lookup.get(v, np.nan) for v in split["X_cat"][:, i]]
        blocks.append(enc)
    if include_y and split["y"] is not None:
        blocks.append(np.asarray(split["y"], dtype=float).reshape(-1, 1))

    X = np.hstack(blocks)
    with warnings.catch_warnings():
        # Constant columns give a 0/0 correlation; they surface as NaN below.
        warnings.simplefilter("ignore")
        C = np.corrcoef(X.T)
    return np.nan_to_num(C, nan=0.0)


def _category_maps(real, syn_splits):
    """Shared category -> integer code per categorical column."""
    if real["X_cat"] is None:
        return []
    maps = []
    for i in range(real["X_cat"].shape[1]):
        cats = set(np.unique(real["X_cat"][:, i]).tolist())
        for s in syn_splits:
            if s["X_cat"] is not None:
                cats |= set(np.unique(s["X_cat"][:, i]).tolist())
        maps.append({c: j for j, c in enumerate(sorted(cats, key=str))})
    return maps


def plot_corr_diff(found, args, out_dir):
    method_names = sorted({m for e in found.values() for m in e["_methods"]})
    dataset_names = list(found)

    fig, axes = plt.subplots(
        len(method_names), len(dataset_names),
        figsize=(args.heat_size * len(dataset_names) + 1.2,
                 args.heat_size * len(method_names) + 0.9),
        squeeze=False,
    )

    im = None
    for c, dataset in enumerate(dataset_names):
        entry = found[dataset]
        real = entry["_real"]
        task_type = entry["_task"]
        cat_maps = _category_maps(real, [m["data"] for m in entry["_methods"].values()])
        C_real = _corr_matrix(real, args.include_y, task_type, cat_maps)

        for r, mname in enumerate(method_names):
            ax = axes[r][c]
            method = entry["_methods"].get(mname)
            if method is None:
                ax.axis("off")
                continue
            C_syn = _corr_matrix(method["data"], args.include_y, task_type, cat_maps)
            D = np.abs(C_real - C_syn)
            entry.setdefault("_dpcm", {})[mname] = float(
                np.mean(D[np.triu_indices_from(D, k=1)])
            )

            im = ax.imshow(D, cmap="Reds", vmin=0.0, vmax=args.corr_vmax,
                           interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(dataset, fontsize=8, pad=6)
            if c == 0:
                ax.set_ylabel(mname, fontsize=8)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
        cbar.set_label(r"$|\rho_{real} - \rho_{syn}|$", fontsize=8)
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle("Absolute difference between real and synthetic correlation matrices",
                 fontsize=10, y=0.99)
    _save(fig, out_dir, "correlation_difference", args)


# --------------------------------------------------------------------------- #
def _save(fig, out_dir, stem, args):
    for ext in args.formats:
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results_plots")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="subset of datasets (default: everything discovered)")
    p.add_argument("--default-method", default="GNN-TabDDPM",
                   help="name given to a dataset directory holding arrays directly")
    p.add_argument("--only", choices=["marginals", "corr", "both"], default="both")
    p.add_argument("--include-y", action="store_true", default=True)
    p.add_argument("--no-include-y", dest="include_y", action="store_false")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--groups", type=int, default=1,
                   help="wrap features into this many side-by-side blocks")
    p.add_argument("--panel-w", type=float, default=2.0)
    p.add_argument("--panel-h", type=float, default=1.05)
    p.add_argument("--heat-size", type=float, default=1.7)
    p.add_argument("--corr-vmax", type=float, default=1.0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--serif", action="store_true", default=True,
                   help="serif fonts, to match a LaTeX thesis body")
    p.add_argument("--no-serif", dest="serif", action="store_false")
    args = p.parse_args()

    if args.serif:
        plt.rcParams.update({
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.6,
        })

    found = discover(args.results_dir, args.default_method)
    if args.datasets:
        found = {k: v for k, v in found.items() if k in args.datasets}
    if not found:
        raise SystemExit(f"no synthetic samples found under {args.results_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    resolved = {}
    for dataset, methods in found.items():
        real_dir = os.path.join(args.data_root, dataset)
        if not os.path.isdir(real_dir):
            print(f"! skipping {dataset}: no real data at {real_dir}")
            continue
        with open(os.path.join(real_dir, "info.json")) as fh:
            task_type = json.load(fh)["task_type"]

        entry = {
            "_real": _load_split(real_dir),
            "_task": task_type,
            "_methods": {m: {"dir": d, "data": _load_split(d)} for m, d in methods.items()},
        }
        resolved[dataset] = entry
        print(f"{dataset} [{task_type}]: {', '.join(entry['_methods'])}")

    if args.only in ("marginals", "both"):
        print("\nfigure 5 - marginal distributions")
        for dataset, entry in resolved.items():
            plot_marginals(dataset, entry["_real"], entry["_methods"],
                           entry["_task"], args, args.out_dir)

    if args.only in ("corr", "both"):
        print("\nfigure 6 - correlation differences")
        plot_corr_diff(resolved, args, args.out_dir)
        print("\nmean |rho_real - rho_syn| over off-diagonal pairs (lower = better)")
        for dataset, entry in resolved.items():
            for mname, val in entry.get("_dpcm", {}).items():
                print(f"  {dataset:12s} {mname:16s} {val:.4f}")


if __name__ == "__main__":
    main()

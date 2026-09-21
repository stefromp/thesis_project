"""
Numbers behind the distribution figures, for the thesis tables.

For every dataset found in the results directory it reports:

  * two-sample KS per numerical feature, against the 95% critical value
    1.358 * sqrt((n1+n2)/(n1*n2)) -- a feature below the critical value is
    indistinguishable from real at that sample size;
  * total variation distance per categorical feature and for the class label;
  * the share of synthetic values falling outside the real [min, max];
  * correlation attenuation: mean |rho| real vs synthetic, and the share of
    feature pairs whose dependence was pulled toward zero;
  * the feature pairs with the largest |rho_real - rho_syn|.

Directory layout matches scripts/plot_distributions.py.

Usage
-----
    python scripts/fidelity_stats.py
    python scripts/fidelity_stats.py --datasets abalone diabetes --top-pairs 6
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import ks_2samp


def _load(path, name):
    fpath = os.path.join(path, name)
    return np.load(fpath, allow_pickle=True) if os.path.exists(fpath) else None


def _discover(results_dir):
    out = {}
    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry)
        if not os.path.isdir(path):
            continue
        dataset = entry[: -len("_plots")] if entry.endswith("_plots") else entry
        subs = {s: os.path.join(path, s) for s in sorted(os.listdir(path))
                if os.path.exists(os.path.join(path, s, "X_num_train.npy"))}
        if subs:
            out[dataset] = subs
        elif os.path.exists(os.path.join(path, "X_num_train.npy")):
            out[dataset] = {"synthetic": path}
    return out


def _tvd(real_col, syn_col):
    cats = sorted(set(np.unique(real_col).tolist()) | set(np.unique(syn_col).tolist()),
                  key=str)

    def freq(col):
        u, n = np.unique(col, return_counts=True)
        d = dict(zip(u.tolist(), n.tolist()))
        return np.array([d.get(c, 0) for c in cats]) / len(col)

    return 0.5 * float(np.abs(freq(real_col) - freq(syn_col)).sum()), cats


def _full_matrix(X_num, y, task_type):
    """Numerical block plus the target, which is what the correlations use."""
    blocks = [np.asarray(X_num, dtype=float)]
    if y is not None and task_type == "regression":
        blocks.append(np.asarray(y, dtype=float).reshape(-1, 1))
    elif y is not None:
        blocks.append(np.asarray(y, dtype=float).reshape(-1, 1))
    return np.hstack(blocks)


def report(dataset, real_dir, syn_dir, method, top_pairs):
    task_type = json.load(open(os.path.join(real_dir, "info.json")))["task_type"]
    rn, sn = _load(real_dir, "X_num_train.npy"), _load(syn_dir, "X_num_train.npy")
    rc, sc = _load(real_dir, "X_cat_train.npy"), _load(syn_dir, "X_cat_train.npy")
    ry, sy = _load(real_dir, "y_train.npy"), _load(syn_dir, "y_train.npy")

    print(f"\n{'=' * 78}\n{dataset}  [{task_type}]  method={method}")
    if rn is None:
        print("  no numerical features")
        return
    rn, sn = rn.astype(float), sn.astype(float)
    n1, n2 = len(rn), len(sn)
    crit = 1.358 * np.sqrt((n1 + n2) / (n1 * n2))
    print(f"  n_real={n1}  n_syn={n2}  KS 95% critical value={crit:.4f}")

    # marginals
    ks = [ks_2samp(rn[:, i], sn[:, i]).statistic for i in range(rn.shape[1])]
    over = [i for i, k in enumerate(ks) if k > crit]
    print(f"  KS  mean={np.mean(ks):.4f}  max={np.max(ks):.4f} "
          f"(worst = num {int(np.argmax(ks))})")
    print(f"      per feature: " + " ".join(f"{v:.3f}" for v in ks))
    print(f"      exceeding critical value: {len(over)}/{len(ks)}"
          + (f"  -> {over}" if over else ""))
    if task_type == "regression" and ry is not None:
        print(f"  KS  y(target)={ks_2samp(np.asarray(ry, float), np.asarray(sy, float)).statistic:.4f}")

    # values outside the real support
    outside = []
    for i in range(rn.shape[1]):
        lo, hi = rn[:, i].min(), rn[:, i].max()
        outside.append(float(((sn[:, i] < lo) | (sn[:, i] > hi)).mean()))
    print(f"  outside real [min,max]: mean={np.mean(outside) * 100:.2f}%  "
          f"max={np.max(outside) * 100:.2f}%")

    # categoricals
    if rc is not None and sc is not None:
        tvds = []
        for i in range(rc.shape[1]):
            t, cats = _tvd(rc[:, i], sc[:, i])
            tvds.append(t)
            flag = "  <-- collapsed" if t > 0.10 else ""
            print(f"  cat {i}: TVD={t:.4f}  ({len(cats)} categories){flag}")
        print(f"  cat TVD mean={np.mean(tvds):.4f}")

    # label balance
    if task_type != "regression" and ry is not None:
        t, cats = _tvd(np.asarray(ry).astype(str), np.asarray(sy).astype(str))
        rp = {c: float((np.asarray(ry).astype(str) == c).mean()) for c in cats}
        sp = {c: float((np.asarray(sy).astype(str) == c).mean()) for c in cats}
        print(f"  label TVD={t:.4f}  real={ {c: round(v, 3) for c, v in rp.items()} }  "
              f"syn={ {c: round(v, 3) for c, v in sp.items()} }")

    # correlation structure
    R = _full_matrix(rn, ry, task_type)
    S = _full_matrix(sn, sy, task_type)
    Cr, Cs = np.corrcoef(R.T), np.corrcoef(S.T)
    Cr, Cs = np.nan_to_num(Cr), np.nan_to_num(Cs)
    iu = np.triu_indices_from(Cr, k=1)
    ar, as_ = np.abs(Cr[iu]), np.abs(Cs[iu])
    print(f"  corr  mean|r_real|={ar.mean():.4f}  mean|r_syn|={as_.mean():.4f}  "
          f"ratio={as_.mean() / ar.mean():.3f}  shrunk={100 * np.mean(as_ < ar):.1f}% of pairs")
    print(f"  corr  mean|r_real - r_syn| (DPCM) = {np.abs(Cr[iu] - Cs[iu]).mean():.4f}"
          "   [numerical + target only; plot_distributions.py also folds in"
          " ordinal-encoded categoricals, so its value differs]")

    n_num = rn.shape[1]
    label = lambda k: "y" if k >= n_num else f"num{k}"
    D = np.abs(Cr - Cs)
    pairs = sorted(((D[i, j], i, j) for i in range(D.shape[0])
                    for j in range(i + 1, D.shape[0])), reverse=True)[:top_pairs]
    print("  worst pairs:")
    for d, i, j in pairs:
        flip = "  SIGN FLIP" if Cr[i, j] * Cs[i, j] < 0 else ""
        print(f"    {label(i):>6s} ~ {label(j):<6s} delta={d:.3f}  "
              f"r: {Cr[i, j]:+.2f} -> {Cs[i, j]:+.2f}{flip}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results_plots")
    p.add_argument("--data-root", default="data")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--top-pairs", type=int, default=4)
    args = p.parse_args()

    found = _discover(args.results_dir)
    if args.datasets:
        found = {k: v for k, v in found.items() if k in args.datasets}

    for dataset, methods in found.items():
        real_dir = os.path.join(args.data_root, dataset)
        if not os.path.isdir(real_dir):
            print(f"! skipping {dataset}: no real data at {real_dir}")
            continue
        for method, syn_dir in methods.items():
            report(dataset, real_dir, syn_dir, method, args.top_pairs)


if __name__ == "__main__":
    main()
